# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Production-grade migration engine for LocalAI storage relocations.

`MigrationEngine` is a headless (no Tk imports) state-machine that owns the
"move-this-directory-safely" lifecycle for ComfyUI / models / Ollama-blobs /
HuggingFace cache / torch cache / pip cache. The companion `migration_ui`
module owns the CTkToplevel dialogs that subscribe to `MigrationEngine`'s
progress events and call back into `cancel()` / `acknowledge_failure()` etc.

State machine
-------------
::

    PENDING -> PRE_FLIGHT -> COPYING -> VERIFYING -> COMMITTING -> CLEANUP -> DONE
                                |          |            |             |
                                v          v            v             v
                              CANCEL -> ROLLBACK ----------------------+--> DONE
                                |
                                v
                              FAILED  (manual remediation)

Each transition is persisted to ``<app>/.localai-migration-state.json``
which doubles as a single-writer lock file. On the next launch the resume
orchestrator (``find_resumable_state`` + ``MigrationEngine.from_state_file``)
restarts at the correct phase or shows the user a Resume / Roll back / Decide
later dialog.

Key invariants
~~~~~~~~~~~~~~
* **Cancel is always safe.** Pressing Cancel stops the copy subprocess,
  removes the partial target, and leaves the source bit-identical to what
  it was before the migration started.
* **Free-space hard gate.** Pre-flight refuses to start unless the target
  drive has ``source_size * 1.1`` free bytes (or source size when source and
  target share a volume — same-volume moves are metadata-only). Today's
  4.5-GB-free disaster would have been blocked at this gate.
* **Manifest <-> blob cross-validation for Ollama.** Verify phase walks
  every manifest at target and confirms every referenced blob exists on
  disk. A single missing blob fails the whole verify and leaves source
  intact.
* **Identity-key preservation.** Cleanup scans the source for top-level
  non-bulk files matching ``IDENTITY_KEY_FILENAMES`` (e.g. ``id_ed25519``,
  ``id_ed25519.pub``) and copies them to target BEFORE deleting source.
  Never breaks the user's Ollama / SSH / HuggingFace identity.
* **Never silently delete user data.** Verify failure -> source intact,
  state marked FAILED, user shown actionable message via the UI layer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from src import logger as _log


# ── Constants ────────────────────────────────────────────────────────────────


STATE_FILE_NAME = ".localai-migration-state.json"


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right

# Identity files we never want to leave behind when cleaning up a source
# directory.  Each entry is a relative path under the source root.  We match
# basename-insensitive to also cover ``ID_ed25519`` from a Windows-cased
# SSH config.
IDENTITY_KEY_FILENAMES: frozenset[str] = frozenset({
    "id_ed25519",
    "id_ed25519.pub",
    "id_rsa",
    "id_rsa.pub",
    "id_ecdsa",
    "id_ecdsa.pub",
})

# Drive-type values returned by Windows ``GetDriveTypeW``.
_GDT_REMOVABLE = 2
_GDT_FIXED = 3
_GDT_REMOTE = 4
_GDT_CDROM = 5
_GDT_RAMDISK = 6
_GDT_UNKNOWN = 0
_GDT_NO_ROOT_DIR = 1

# Reboot-volatile drive types per the user-facing guard contract: Removable,
# RAMDisk, Unknown all warrant the "drive may be wiped on reboot" warning.
# We exclude FIXED (3) and REMOTE (4) and CDROM (5).
REBOOT_VOLATILE_DRIVE_TYPES: frozenset[int] = frozenset({
    _GDT_REMOVABLE, _GDT_RAMDISK, _GDT_UNKNOWN, _GDT_NO_ROOT_DIR,
})

# Pre-flight free-space headroom multiplier.  10% headroom keeps NTFS happy
# during a multi-GB copy and matches resource_manager.check_disk_for_download.
FREE_SPACE_HEADROOM = 1.1

# Robocopy throttle: we report progress events at most this often.  Set
# generously so we don't pin a CPU on a fast NVMe copy when the parser sees
# a flood of percentage updates.
_PROGRESS_EVENT_MIN_INTERVAL_S = 0.25


# ── Enums + dataclasses ──────────────────────────────────────────────────────


class MigrationPhase(str, Enum):
    """Migration lifecycle phase. String-valued for JSON-friendly persistence."""

    PENDING = "pending"
    PRE_FLIGHT = "pre_flight"
    COPYING = "copying"
    VERIFYING = "verifying"
    COMMITTING = "committing"
    CLEANUP = "cleanup"
    DONE = "done"
    CANCEL = "cancel"
    ROLLBACK = "rollback"
    FAILED = "failed"


# Phases at which a crash-resume dialog must ask the user what to do.  PENDING
# / PRE_FLIGHT can silently restart; COMMITTING / CLEANUP idempotently retry;
# COPYING / VERIFYING leave a half-copied target that the user must confirm.
USER_PROMPT_RESUME_PHASES: frozenset[MigrationPhase] = frozenset({
    MigrationPhase.COPYING,
    MigrationPhase.VERIFYING,
})


@dataclass
class ProgressEvent:
    """A single progress update from the copy subprocess parser."""

    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    current_file: str = ""
    percent: float = 0.0


@dataclass
class PreflightResult:
    """Outcome of a pre-flight gate evaluation."""

    ok: bool
    reason: str = ""
    source_size: int = 0
    target_free: int = 0
    target_required: int = 0
    same_volume: bool = False
    drive_mismatch: bool = False
    target_is_reboot_volatile: bool = False
    source_is_reboot_volatile: bool = False


@dataclass
class MigrationState:
    """Persisted state used to resume after a crash / restart."""

    kind: str                       # "comfyui" / "models" / "ollama" / "hf_cache" / "torch_cache" / "pip_cache"
    source: str
    target: str
    phase: str = MigrationPhase.PENDING.value
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    bytes_done: int = 0
    bytes_total: int = 0
    files_done: int = 0
    files_total: int = 0
    last_error: str = ""
    schedule_delete_path: str = ""  # e.g. <source>.deleteme awaiting next-launch delete

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MigrationState":
        # Keep only known fields so an older state file doesn't blow up.
        known = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**known)


# ── State file helpers ───────────────────────────────────────────────────────


def state_file_path(app_root: Path) -> Path:
    """Return the canonical path to the migration state lock file."""
    return Path(app_root) / STATE_FILE_NAME


def write_state(app_root: Path, state: MigrationState) -> bool:
    """Persist state atomically to ``<app_root>/.localai-migration-state.json``.

    Returns True on success, False on any OSError.
    """
    state.updated_at = time.time()
    path = state_file_path(app_root)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        # On Windows ``Path.replace`` is the closest thing to an atomic rename
        # and is good enough for our single-writer lock semantics.
        tmp.replace(path)
        return True
    except OSError as exc:
        _log.error(f"Migration state write failed for {path}: {exc}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def read_state(app_root: Path) -> Optional[MigrationState]:
    """Return persisted MigrationState if one exists and is valid, else None."""
    path = state_file_path(app_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.error(f"Migration state read failed for {path}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    try:
        return MigrationState.from_dict(data)
    except TypeError as exc:
        _log.error(f"Migration state shape unexpected at {path}: {exc}")
        return None


def clear_state(app_root: Path) -> None:
    """Delete the migration state file. Idempotent."""
    path = state_file_path(app_root)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _log.error(f"Migration state clear failed for {path}: {exc}")


def find_resumable_state(app_root: Path) -> Optional[MigrationState]:
    """Return state if a resumable migration is present, else None.

    A state file is considered "resumable" when its phase is anything other
    than DONE.  Callers MUST decide whether to show a resume dialog
    (``phase in USER_PROMPT_RESUME_PHASES``) or attempt silent re-entry.
    """
    state = read_state(app_root)
    if state is None:
        return None
    if state.phase == MigrationPhase.DONE.value:
        # Clean up DONE state files we forgot to delete.
        clear_state(app_root)
        return None
    return state


# ── Drive helpers ────────────────────────────────────────────────────────────


def get_drive_type(path: Path) -> int:
    """Return Windows ``GetDriveTypeW`` value for the volume holding *path*.

    Returns ``_GDT_FIXED`` on non-Windows / probe failure so we never
    false-positive into the reboot-volatile warning.
    """
    if sys.platform != "win32":
        return _GDT_FIXED
    try:
        import ctypes
        drive = Path(path).drive
        if not drive:
            return _GDT_FIXED
        root = drive.rstrip("\\") + "\\"
        kernel32 = ctypes.windll.kernel32
        kernel32.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetDriveTypeW.restype = ctypes.c_uint
        return int(kernel32.GetDriveTypeW(root))
    except Exception:
        return _GDT_FIXED


def is_reboot_volatile_drive(path: Path) -> bool:
    """Return True if *path* lives on a removable / RAM / unknown drive."""
    return get_drive_type(path) in REBOOT_VOLATILE_DRIVE_TYPES


def same_volume(a: Path, b: Path) -> bool:
    """Return True if *a* and *b* live on the same filesystem volume.

    On Windows we compare drive letters case-insensitively.  On
    macOS/Linux we compare ``os.stat().st_dev`` of the nearest existing
    parent of each path.
    """
    try:
        if sys.platform == "win32":
            return Path(a).drive.upper() == Path(b).drive.upper()
        # Walk up until we find an existing parent (target may not exist yet).
        def _existing(p: Path) -> Path:
            cur: Optional[Path] = p
            while cur is not None and not cur.exists():
                parent = cur.parent
                if parent == cur:
                    break
                cur = parent
            return cur or Path("/")
        return os.stat(str(_existing(Path(a)))).st_dev == os.stat(str(_existing(Path(b)))).st_dev
    except OSError:
        return False


# ── Path size / free-space helpers ───────────────────────────────────────────


def measure_tree_size(root: Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)`` for *root* and its children.

    Best-effort: unreadable entries are skipped and counted as zero bytes.
    """
    total_bytes = 0
    file_count = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                file_count += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    continue
    except OSError as exc:
        _log.warning(f"measure_tree_size({root}): {exc}")
    return total_bytes, file_count


def free_space_bytes(path: Path) -> int:
    """Return free bytes on the volume holding *path* (or nearest parent)."""
    probe: Optional[Path] = path
    while probe is not None and not probe.exists():
        parent = probe.parent
        if parent == probe:
            return 0
        probe = parent
    if probe is None:
        return 0
    try:
        return shutil.disk_usage(str(probe)).free
    except OSError:
        return 0


# ── Sentinel validation ──────────────────────────────────────────────────────


def has_comfyui_sentinel(root: Path) -> bool:
    """True if *root* contains the ComfyUI main entrypoint."""
    return (Path(root) / "main.py").is_file()


def has_ollama_sentinel(root: Path) -> bool:
    """True if *root* is a plausible Ollama-blobs directory.

    Accepts BOTH the OLLAMA_MODELS layout (``<root>/blobs``) and the default
    ``~/.ollama/models`` layout (``<root>/blobs`` too — the difference is the
    parent path, not the leaf shape).
    """
    return (Path(root) / "blobs").is_dir()


def has_models_sentinel(root: Path) -> bool:
    """True if *root* contains at least one model file or known subfolder."""
    root = Path(root)
    if not root.is_dir():
        return False
    if (root / "onnx").is_dir() or (root / "ov").is_dir():
        return True
    for ext in (".gguf", ".safetensors", ".pt", ".bin", ".onnx"):
        try:
            for _ in root.rglob(f"*{ext}"):
                return True
        except OSError:
            continue
    return False


# ── Ollama manifest <-> blob cross-validation ────────────────────────────────


_SHA256_PREFIX_RE = re.compile(r"^sha256[:\-]?([0-9a-fA-F]{64})$")


def _normalise_digest(digest: str) -> Optional[str]:
    """Convert an Ollama digest ('sha256:abc...') to the blob filename form.

    Ollama writes blobs as ``sha256-<64-hex>`` on disk but stores them as
    ``sha256:<64-hex>`` in manifest JSON.  We return the dashed form
    (``sha256-...``) which matches the on-disk filename, or None when the
    digest doesn't look right.
    """
    if not isinstance(digest, str):
        return None
    m = _SHA256_PREFIX_RE.match(digest.strip())
    if m is None:
        return None
    return "sha256-" + m.group(1).lower()


def iter_manifest_paths(target_root: Path) -> Iterable[Path]:
    """Yield every ``.../manifests/.../<tag>`` regular file under *target_root*.

    Ollama manifests don't carry a file extension on disk — they're stored
    as ``manifests/registry.ollama.ai/library/<model>/<tag>`` plain JSON
    files.  We walk the whole ``manifests`` subtree and yield regular files.
    """
    manifests_root = Path(target_root) / "manifests"
    if not manifests_root.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(manifests_root):
        for name in filenames:
            yield Path(dirpath) / name


def collect_manifest_blob_digests(manifest_path: Path) -> set[str]:
    """Parse one Ollama manifest and return the set of blob filenames it refers to.

    Filenames are returned in the on-disk dashed form (``sha256-...``).  An
    unreadable / non-JSON manifest yields an empty set; the caller treats
    that as a hard verify failure but does NOT crash the engine.
    """
    digests: set[str] = set()
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return digests
    config = data.get("config") or {}
    d = _normalise_digest(config.get("digest", ""))
    if d:
        digests.add(d)
    layers = data.get("layers") or []
    if isinstance(layers, list):
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            d = _normalise_digest(layer.get("digest", ""))
            if d:
                digests.add(d)
    return digests


def cross_validate_ollama_manifests(target_root: Path) -> tuple[bool, list[str]]:
    """Walk every manifest at *target_root* and confirm every blob exists.

    Returns ``(ok, missing_blobs)``.  ``ok`` is True only when every
    manifest's referenced blob filenames are present under
    ``<target_root>/blobs/``.  ``missing_blobs`` is a sorted list of the
    missing blob filenames (capped at 50 entries so we don't blow the
    error dialog).

    This is the core safety check that would have prevented today's
    "115 of 121 blobs copied, manifests not copied at all" disaster:
    a single missing blob is enough to fail the migration and keep
    source intact.
    """
    blobs_dir = Path(target_root) / "blobs"
    on_disk: set[str] = set()
    if blobs_dir.is_dir():
        try:
            on_disk = {p.name for p in blobs_dir.iterdir() if p.is_file()}
        except OSError:
            on_disk = set()
    required: set[str] = set()
    manifest_count = 0
    for manifest in iter_manifest_paths(target_root):
        manifest_count += 1
        required |= collect_manifest_blob_digests(manifest)
    if manifest_count == 0:
        # No manifests = nothing to validate.  Caller decides whether that's
        # acceptable (e.g. pure-blobs scratch dir) or a verify failure.  We
        # report ok=False so the engine fails closed and asks the user.
        return False, []
    missing = sorted(required - on_disk)
    return (len(missing) == 0), missing[:50]


# ── Robocopy / rsync wrappers ────────────────────────────────────────────────


def _robocopy_cmd(source: Path, target: Path, *, mt: int = 8) -> list[str]:
    """Build the robocopy command we use for migration copies.

    Flags rationale:
    - ``/E`` copy subdirectories including empty.
    - ``/COPY:DAT`` data + attributes + timestamps (NO security; we don't
      want to preserve broken managed-VM ACLs into the new location).
    - ``/R:2 /W:5`` retry twice with 5 s back-off (defaults are 1M retries
      which makes a network blip look like a hang).
    - ``/MT:N`` multi-thread copy.  ``8`` is the sweet spot for NVMe + cold
      cache; higher numbers hurt on spinning rust and 8 is the documented
      sane default.
    - ``/NP`` no per-file percentage (avoids flooding stdout).
    - ``/NDL`` no directory list.
    - ``/BYTES`` show sizes in bytes (we parse the summary).
    - ``/TEE`` mirror to stdout so we can read it without losing the log.
    """
    return [
        "robocopy", str(source), str(target),
        "/E", "/COPY:DAT", "/R:2", "/W:5",
        f"/MT:{int(mt)}",
        "/NP", "/NDL", "/BYTES", "/TEE",
    ]


def _rsync_cmd(source: Path, target: Path) -> list[str]:
    """Build the rsync command we use on macOS / Linux."""
    s = str(source).rstrip("/") + "/"
    return ["rsync", "-aE", "--info=progress2", s, str(target)]


# ── Robocopy stdout parser ──────────────────────────────────────────────────


# robocopy in /BYTES mode emits per-file lines like
#   "  100%        12345    foo/bar/baz.bin"
# and a final summary block.  We track current_file from the "Newer" /
# "*EXTRA File" / "newDir" markers and per-file completion from the percent
# tokens.  The parser is intentionally lenient: a flood of unparseable
# lines should not crash the engine.

_ROBOCOPY_FILE_LINE_RE = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)?)%\s+)?(\d+)\s+(.+?)\s*$"
)


def parse_robocopy_line(line: str) -> Optional[tuple[float, int, str]]:
    """Parse one robocopy /BYTES line.

    Returns ``(percent, bytes, filename)`` or None if the line doesn't look
    like a file-progress line.  ``bytes`` is the file size; ``percent`` is
    0..100 indicating how much of the file has been copied so far.  We use
    these to update the per-file portion of the cumulative byte total.
    """
    if not line:
        return None
    s = line.rstrip("\r\n")
    # Strip the leading column of category markers (e.g. "  New File")
    # robocopy emits.  Those are noise for the percentage parser.
    if "%" not in s and "\t" not in s:
        return None
    m = _ROBOCOPY_FILE_LINE_RE.match(s.replace("\t", " "))
    if m is None:
        return None
    pct_str, size_str, name = m.groups()
    try:
        pct = float(pct_str) if pct_str else 0.0
        size = int(size_str)
    except (TypeError, ValueError):
        return None
    if size <= 0 or len(name.strip()) == 0:
        return None
    return pct, size, name.strip()


# ── Cancellation exception ───────────────────────────────────────────────────


class MigrationCancelled(Exception):
    """Raised internally when the user (or a test) calls ``cancel()``."""


class MigrationVerifyFailed(Exception):
    """Raised when verification phase fails — source MUST stay intact."""

    def __init__(self, message: str, *, missing_blobs: Optional[list[str]] = None):
        super().__init__(message)
        self.missing_blobs = missing_blobs or []


# ── MigrationEngine ──────────────────────────────────────────────────────────


@dataclass
class MigrationPlan:
    """Inputs that fully describe one migration."""

    kind: str
    source: Path
    target: Path
    sentinel_check: Callable[[Path], bool] = field(default=lambda p: True)
    is_ollama: bool = False
    extra_ignore_volatile_warn: bool = False
    # Optional override of which top-level files at the source are identity
    # keys that must be preserved on cleanup.  Defaults to IDENTITY_KEY_FILENAMES.
    identity_keys: Optional[frozenset[str]] = None


class MigrationEngine:
    """Headless state-machine that owns a single in-flight migration.

    Construction is cheap; ``run()`` actually does the work and may block
    for a long time.  Wire ``progress_callback`` to a thread-safe UI emitter
    so the user sees per-file updates.

    UI integration:
        engine = MigrationEngine(plan, app_root=APP_ROOT,
                                  progress_callback=ui.on_progress)
        with engine:
            engine.run()
        # success — engine.phase == DONE

    Test integration:
        engine = MigrationEngine(plan, app_root=tmp,
                                  copy_runner=fake_copy_runner)
        engine.run()
    """

    def __init__(
        self,
        plan: MigrationPlan,
        *,
        app_root: Path,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        copy_runner: Optional[Callable[["MigrationEngine"], None]] = None,
        free_space_probe: Callable[[Path], int] = free_space_bytes,
        size_probe: Callable[[Path], tuple[int, int]] = measure_tree_size,
        same_volume_probe: Callable[[Path, Path], bool] = same_volume,
        config_commit: Optional[Callable[[Path], bool]] = None,
        resume_from: Optional[MigrationState] = None,
    ):
        self.plan = plan
        self.app_root = Path(app_root)
        self.progress_callback = progress_callback or (lambda evt: None)
        self.copy_runner = copy_runner  # override for tests (skips robocopy)
        self.free_space_probe = free_space_probe
        self.size_probe = size_probe
        self.same_volume_probe = same_volume_probe
        self.config_commit = config_commit
        self.state: MigrationState = MigrationState(
            kind=plan.kind,
            source=str(plan.source),
            target=str(plan.target),
        )
        if resume_from is not None:
            self.state = resume_from
        self.phase = MigrationPhase(self.state.phase)
        self._cancel_event = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()
        self._last_progress_emit = 0.0
        self._lock = threading.RLock()

    # ── Context manager so we always release the state file lock ─────────

    def __enter__(self) -> "MigrationEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Best-effort: if we crashed mid-flight the state file is the resume
        # trigger; we deliberately don't delete it on exception.
        return None

    # ── Public API ───────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Request a safe cancellation.

        Sets the cancel event AND terminates the running copy process.
        The next phase transition will raise MigrationCancelled internally
        which we catch and convert to ROLLBACK.
        """
        self._cancel_event.set()
        with self._process_lock:
            proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                # Give robocopy 2s to flush; otherwise force-kill.
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except OSError as exc:
            _log.warning(f"Cancel: terminating copy process failed: {exc}")

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def transition(self, phase: MigrationPhase, *, error: str = "") -> None:
        """Move to *phase* and persist state."""
        with self._lock:
            self.phase = phase
            self.state.phase = phase.value
            if error:
                self.state.last_error = error
            write_state(self.app_root, self.state)

    def emit_progress(self, evt: ProgressEvent, *, force: bool = False) -> None:
        """Throttled progress emission to the UI callback.

        We coalesce updates so a fast NVMe copy doesn't pin a CPU on the
        UI thread; the `force=True` path is used for terminal updates.
        """
        now = time.time()
        if not force and (now - self._last_progress_emit) < _PROGRESS_EVENT_MIN_INTERVAL_S:
            return
        self._last_progress_emit = now
        try:
            self.progress_callback(evt)
        except Exception as exc:
            _log.warning(f"progress_callback raised: {exc}")

    # ── Pre-flight ───────────────────────────────────────────────────────

    def preflight(self) -> PreflightResult:
        """Evaluate every pre-flight gate and return a structured result.

        Caller decides whether to proceed (`result.ok == True`) or show the
        user the failure dialog.  Engine.run() always calls this first and
        bails before touching the filesystem if the result is not ok.
        """
        source = Path(self.plan.source)
        target = Path(self.plan.target)
        result = PreflightResult(ok=False, source_size=0, target_free=0, target_required=0)

        if not source.exists() or not source.is_dir():
            result.reason = f"Source path does not exist or is not a directory: {source}"
            return result

        if not self.plan.sentinel_check(source):
            result.reason = (
                f"Source path is missing expected sentinel files: {source}. "
                "Migration refused to avoid moving the wrong directory."
            )
            return result

        # Target rules: must not exist OR must be empty.
        if target.exists():
            try:
                if any(target.iterdir()):
                    result.reason = (
                        f"Target path is not empty: {target}. Migration refused "
                        "to avoid overwriting existing user data."
                    )
                    return result
            except OSError as exc:
                result.reason = f"Could not enumerate target {target}: {exc}"
                return result

        # Target parent must be writable.
        target_parent = target.parent
        if not target_parent.exists():
            try:
                target_parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                result.reason = (
                    f"Could not create target parent directory {target_parent}: {exc}"
                )
                return result
        if not os.access(str(target_parent), os.W_OK):
            result.reason = f"Target parent directory is not writable: {target_parent}"
            return result

        # Measure source size + figure out free-space requirement.
        source_size, file_count = self.size_probe(source)
        result.source_size = source_size
        self.state.bytes_total = source_size
        self.state.files_total = file_count

        same_vol = self.same_volume_probe(source, target)
        result.same_volume = same_vol
        target_free = self.free_space_probe(target_parent)
        result.target_free = target_free
        # Same-volume moves are metadata-only — we still want SOMETHING free
        # (256 MB) so a hiccup doesn't strand us, but we don't need source_size.
        target_required = int(source_size * FREE_SPACE_HEADROOM) if not same_vol else 256 * 1024 * 1024
        result.target_required = target_required

        if target_free < target_required:
            need_gb = target_required / 1_073_741_824
            free_gb = target_free / 1_073_741_824
            result.reason = (
                f"Target drive ({target.drive or target_parent}) has "
                f"{free_gb:.1f} GB free but the migration needs ~{need_gb:.1f} GB "
                "(source size + 10% headroom). Free up space and retry."
            )
            return result

        # Drive-mismatch + reboot-volatile flags (informational; caller may
        # confirm via dialog but the engine doesn't refuse on these alone).
        app_drive = Path(self.app_root).drive.upper()
        source_drive = Path(source).drive.upper()
        target_drive = Path(target).drive.upper()
        if app_drive and target_drive and target_drive != app_drive and target_drive != source_drive:
            result.drive_mismatch = True

        result.target_is_reboot_volatile = is_reboot_volatile_drive(target_parent)
        result.source_is_reboot_volatile = is_reboot_volatile_drive(source)

        result.ok = True
        return result

    # ── Copy phase ───────────────────────────────────────────────────────

    def _run_copy_subprocess(self) -> None:
        """Spawn robocopy / rsync and stream stdout into the progress callback.

        Tests can swap out by passing a ``copy_runner`` to __init__ that
        does its own fake copy + emits ProgressEvents.
        """
        if self.copy_runner is not None:
            self.copy_runner(self)
            return

        source = Path(self.plan.source)
        target = Path(self.plan.target)
        target.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            cmd = _robocopy_cmd(source, target)
        else:
            cmd = _rsync_cmd(source, target)

        _log.info(f"Migration copy starting: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise RuntimeError(f"Could not launch copy subprocess: {exc}") from exc

        with self._process_lock:
            self._process = proc

        try:
            assert proc.stdout is not None
            bytes_done = 0
            files_done = 0
            current_per_file = 0.0
            current_size = 0
            current_name = ""
            for raw_line in proc.stdout:
                if self.is_cancelled():
                    raise MigrationCancelled("user cancelled")
                parsed = parse_robocopy_line(raw_line) if sys.platform == "win32" else None
                if parsed is None:
                    continue
                pct, size, name = parsed
                if name != current_name:
                    # New file started — fold the previous file's progress.
                    bytes_done += current_size
                    files_done += 1 if current_size > 0 else 0
                    current_name = name
                    current_size = size
                current_per_file = pct
                if pct >= 100:
                    bytes_done += size
                    files_done += 1
                    current_size = 0
                    current_per_file = 0.0
                self.state.bytes_done = bytes_done
                self.state.files_done = files_done
                self.emit_progress(ProgressEvent(
                    files_done=files_done,
                    files_total=self.state.files_total,
                    bytes_done=bytes_done + int(current_size * current_per_file / 100.0),
                    bytes_total=self.state.bytes_total,
                    current_file=current_name,
                    percent=(bytes_done / self.state.bytes_total * 100.0)
                    if self.state.bytes_total > 0 else 0.0,
                ))
            rc = proc.wait()
        finally:
            with self._process_lock:
                self._process = None

        if self.is_cancelled():
            raise MigrationCancelled("user cancelled")

        # Robocopy success codes: 0 nothing to do, 1 files copied OK, 2 extra
        # files at dest, 3 files copied + extra files.  4+ means at least one
        # file mismatch; 8+ means at least one file failed; 16 means usage
        # error.  We treat >= 8 as a hard failure.
        if sys.platform == "win32":
            if rc >= 8:
                raise RuntimeError(
                    f"Copy subprocess returned exit code {rc}; at least one file failed."
                )
        else:
            if rc != 0:
                raise RuntimeError(f"rsync returned exit code {rc}.")

    # ── Verify phase ─────────────────────────────────────────────────────

    def verify(self) -> None:
        """Run the verification gates; raises MigrationVerifyFailed on failure."""
        source = Path(self.plan.source)
        target = Path(self.plan.target)
        source_bytes, source_files = self.size_probe(source)
        target_bytes, target_files = self.size_probe(target)

        # File count match (allow off-by-N for identity keys we'll copy
        # separately in the cleanup phase — that's still in the future at
        # verify time, so source_files == target_files is the correct check).
        if source_files != target_files:
            raise MigrationVerifyFailed(
                f"Verify failed: source has {source_files} files but target has "
                f"{target_files}. Source left intact."
            )
        if source_bytes != target_bytes:
            raise MigrationVerifyFailed(
                f"Verify failed: source is {source_bytes} bytes but target is "
                f"{target_bytes} bytes. Source left intact."
            )

        # Sentinel must exist at target.
        if not self.plan.sentinel_check(target):
            raise MigrationVerifyFailed(
                f"Verify failed: target {target} is missing expected sentinel files. "
                "Source left intact."
            )

        # Ollama-specific cross-validation.
        if self.plan.is_ollama:
            ok, missing = cross_validate_ollama_manifests(target)
            if not ok:
                if missing:
                    detail = (
                        f"{len(missing)} blob{'s' if len(missing) != 1 else ''} "
                        f"referenced by manifests are missing at target: "
                        f"{', '.join(missing[:5])}"
                        + (" …" if len(missing) > 5 else "")
                    )
                else:
                    detail = (
                        "no manifests found at target; refusing to mark migration "
                        "complete because we cannot verify blob integrity"
                    )
                raise MigrationVerifyFailed(
                    f"Verify failed (Ollama manifest <-> blob cross-validation): "
                    f"{detail}. Source left intact.",
                    missing_blobs=missing,
                )

    # ── Commit phase ─────────────────────────────────────────────────────

    def commit(self) -> None:
        """Save config / persist the new target.

        The actual config update is delegated to the caller-supplied
        ``config_commit`` callable to keep this module decoupled from
        ``src.config``.  We just record success/failure in state.
        """
        target = Path(self.plan.target)
        if self.config_commit is None:
            return
        ok = False
        try:
            ok = bool(self.config_commit(target))
        except Exception as exc:
            _log.error(f"config_commit raised: {exc}")
        if not ok:
            raise RuntimeError("Config commit failed.")

    # ── Cleanup phase ────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Move source to ``<source>.deleteme`` and copy identity keys to target.

        Identity keys are scanned at the top level (not recursively) and
        copied BEFORE the source-rename.  Deletion of ``<source>.deleteme``
        is scheduled for next launch via the state file — we do NOT block
        on a multi-GB delete during the migration.
        """
        source = Path(self.plan.source)
        target = Path(self.plan.target)
        if not source.exists():
            return  # Nothing to do — already gone.

        # 1. Preserve identity keys.
        identity_keys = self.plan.identity_keys or IDENTITY_KEY_FILENAMES
        try:
            for entry in source.iterdir():
                if not entry.is_file():
                    continue
                if entry.name.lower() in {k.lower() for k in identity_keys}:
                    dest = target / entry.name
                    if not dest.exists():
                        try:
                            shutil.copy2(str(entry), str(dest))
                            _log.info(
                                f"Preserved identity file {entry.name} -> {dest}"
                            )
                        except OSError as exc:
                            _log.warning(
                                f"Could not preserve identity file {entry}: {exc}"
                            )
        except OSError as exc:
            _log.warning(f"Identity-key scan failed for {source}: {exc}")

        # 2. Rename source to <source>.deleteme.  Atomic on same drive.
        deleteme = source.with_name(source.name + ".deleteme")
        # If a leftover .deleteme exists, just append a suffix.
        if deleteme.exists():
            for n in range(1, 10):
                cand = source.with_name(f"{source.name}.deleteme.{n}")
                if not cand.exists():
                    deleteme = cand
                    break
        try:
            source.rename(deleteme)
            self.state.schedule_delete_path = str(deleteme)
            write_state(self.app_root, self.state)
        except OSError as exc:
            # Cross-volume rename can fail; fall back to copy + delete (the
            # copy already happened so we can just rmtree source).
            _log.warning(
                f"Could not rename source {source} -> {deleteme}: {exc}; falling back to rmtree."
            )
            try:
                shutil.rmtree(str(source), ignore_errors=True)
            except OSError:
                pass

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self) -> MigrationState:
        """Execute the full state machine.

        Raises MigrationCancelled when the user cancels; the caller is
        responsible for the rollback / dialog.  Other exceptions surface to
        the caller with state.phase == FAILED.
        """
        try:
            self.transition(MigrationPhase.PRE_FLIGHT)
            result = self.preflight()
            if not result.ok:
                raise RuntimeError(f"Pre-flight failed: {result.reason}")

            self.transition(MigrationPhase.COPYING)
            self._run_copy_subprocess()

            if self.is_cancelled():
                raise MigrationCancelled("user cancelled after copy")

            self.transition(MigrationPhase.VERIFYING)
            self.verify()

            self.transition(MigrationPhase.COMMITTING)
            self.commit()

            self.transition(MigrationPhase.CLEANUP)
            self.cleanup()

            self.transition(MigrationPhase.DONE)
            clear_state(self.app_root)
            self.emit_progress(ProgressEvent(
                files_done=self.state.files_done,
                files_total=self.state.files_total,
                bytes_done=self.state.bytes_total,
                bytes_total=self.state.bytes_total,
                current_file="",
                percent=100.0,
            ), force=True)
            return self.state
        except MigrationCancelled:
            self.transition(MigrationPhase.CANCEL)
            self._rollback_target_only()
            self.transition(MigrationPhase.ROLLBACK)
            self.transition(MigrationPhase.DONE)
            # Delete state file LAST so next launch isn't prompted again.
            clear_state(self.app_root)
            raise
        except MigrationVerifyFailed as exc:
            # Source is intact by contract — DO NOT cleanup, DO NOT delete state.
            self.transition(MigrationPhase.FAILED, error=str(exc))
            raise
        except Exception as exc:
            self.transition(MigrationPhase.FAILED, error=str(exc))
            raise

    def _rollback_target_only(self) -> None:
        """Delete the partial target so the user is left exactly as before."""
        target = Path(self.plan.target)
        try:
            if target.exists():
                shutil.rmtree(str(target), ignore_errors=True)
        except OSError as exc:
            _log.warning(f"Rollback: could not remove partial target {target}: {exc}")


# ── Scheduled-delete on next launch ──────────────────────────────────────────


def process_scheduled_deletes(app_root: Path) -> list[str]:
    """Walk the app root for ``*.deleteme`` directories and remove them.

    Returns the list of paths that were deleted.  Best-effort: any failure
    leaves the .deleteme directory in place so the next launch tries again.
    """
    deleted: list[str] = []
    try:
        for entry in Path(app_root).iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.endswith(".deleteme") and ".deleteme." not in entry.name:
                continue
            try:
                shutil.rmtree(str(entry), ignore_errors=False)
                deleted.append(str(entry))
                _log.info(f"Scheduled-delete completed: {entry}")
            except OSError as exc:
                _log.warning(f"Scheduled-delete failed for {entry}: {exc}")
    except OSError as exc:
        _log.warning(f"Scheduled-delete scan failed at {app_root}: {exc}")
    return deleted


# ── Verify & Repair scan engine ──────────────────────────────────────────────


# Severity values used by Finding rows.  "info" is shown for clean state;
# "warn" surfaces a yellow row with no fix needed; "action" pairs a Fix /
# Ignore button.  Any value outside this set is treated as "warn" by the UI.
SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_ACTION = "action"

# Well-known on-disk subdirectory names under <models_dir> that are caches /
# scaffolding, NOT user-installed model directories.  These are skipped by
# scan_uncatalogued_onnx_dirs so we never offer to delete legacy / accidental
# cache directories under <models_dir>.  Note: ``phase1`` is intentionally
# included even though v5.3.10's HF cache lives at <app>/.cache/huggingface —
# upgraders from <= v5.3.10 may still have a stale ``models/phase1`` directory
# and the dedicated ``scan_legacy_hf_cache`` finding (with an explicit
# migration handler) is the right surface for offering to move it.
RESERVED_MODELS_DIR_NAMES: frozenset[str] = frozenset({
    "phase1",      # legacy <= v5.3.10 HF cache; see scan_legacy_hf_cache
    "huggingface", # alt HF cache layout
    "hub",         # HF hub cache subfolder
    "onnx",        # the parent of per-id onnx dirs; scanned separately
    "ov",          # OpenVINO model cache parent
    ".cache",      # generic scratch
    "tmp",
    "temp",
    "__pycache__",
})

# Free-disk threshold below which scan_disk_space emits a warning row.
DISK_SPACE_WARN_THRESHOLD_GB = 10.0


@dataclass
class Finding:
    """One reconcilable issue surfaced by the Verify & Repair scan engine.

    The UI layer renders one row per Finding.  ``fix_callable`` and
    ``ignore_callable`` are optional; when both are None the row is purely
    informational (e.g. a disk-space warning).
    """

    kind: str               # short machine-readable identifier: "orphan_blobs", "legacy_onnx", ...
    severity: str           # SEVERITY_INFO / SEVERITY_WARN / SEVERITY_ACTION
    summary: str            # one-line headline shown in the row
    detail: str = ""        # additional context shown under the headline
    fix_callable: Optional[Callable[[], Any]] = None
    ignore_callable: Optional[Callable[[], Any]] = None
    fix_label: str = "Fix"
    ignore_label: str = "Ignore"


def normalize_ollama_tag(tag: str) -> str:
    """Return *tag* canonicalised so bare names compare equal to ``:latest``.

    Ollama treats ``phi4`` and ``phi4:latest`` as the same model.  Both the
    orphan-blob check (which uses manifest filenames) and the Uncatalogued
    panel (which compares catalog ``ollama_tag`` values against ``ollama
    list`` output) need that equivalence, so we factor it out here.

    Examples
    --------
    >>> normalize_ollama_tag("phi4")
    'phi4:latest'
    >>> normalize_ollama_tag("phi4:latest")
    'phi4:latest'
    >>> normalize_ollama_tag(" Phi4:Latest ")
    'phi4:latest'
    >>> normalize_ollama_tag("")
    ''
    """
    if not isinstance(tag, str):
        return ""
    s = tag.strip().lower()
    if not s:
        return ""
    if s.endswith(":"):
        s = s[:-1]
    if ":" not in s:
        s = s + ":latest"
    return s


def scan_orphan_ollama_blobs(
    *,
    candidate_roots: Optional[Iterable[Path]] = None,
) -> list[tuple[Path, int]]:
    """Return ``(root, blob_count)`` pairs for Ollama dirs with blobs but no manifests.

    Default candidates are ``$OLLAMA_MODELS`` (if set) and
    ``~/.ollama/models``; tests can pass a fixed list via *candidate_roots*.
    A root is reported when ``<root>/blobs`` contains at least one entry
    AND ``<root>/manifests`` is missing or recursively contains zero
    regular files.
    """
    if candidate_roots is None:
        cands: list[Path] = []
        env_dir = os.environ.get("OLLAMA_MODELS")
        if env_dir:
            cands.append(Path(env_dir))
        cands.append(Path.home() / ".ollama" / "models")
        candidate_roots = cands

    seen: set[str] = set()
    results: list[tuple[Path, int]] = []
    for raw in candidate_roots:
        root = Path(raw)
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        blobs = root / "blobs"
        manifests = root / "manifests"
        if not blobs.is_dir():
            continue
        try:
            blob_count = sum(1 for _ in blobs.iterdir())
        except OSError:
            blob_count = 0
        if blob_count == 0:
            continue
        manifest_count = 0
        if manifests.is_dir():
            try:
                for _dirpath, _dirnames, filenames in os.walk(manifests):
                    manifest_count += len(filenames)
            except OSError:
                manifest_count = 0
        if manifest_count > 0:
            continue
        results.append((root, blob_count))
    return results


def scan_legacy_onnx_paths(
    *,
    localappdata: Optional[Path] = None,
    catalog_entries: Optional[list[dict]] = None,
) -> list[dict]:
    """Return per-dir descriptors for catalog ONNX models living under ``%LOCALAPPDATA%\\LocalAI``.

    Each entry is ``{"id": str, "name": str, "path": Path}``.  Empty list
    when no catalog or no LOCALAPPDATA path is supplied; safe to call on
    non-Windows.
    """
    if sys.platform != "win32":
        return []
    if localappdata is None:
        raw = os.environ.get("LOCALAPPDATA", "")
        if not raw:
            return []
        localappdata = Path(raw)
    legacy_root = Path(localappdata) / "LocalAI"
    if not legacy_root.exists():
        return []
    catalog_entries = catalog_entries or []
    offenders: list[dict] = []
    for entry in catalog_entries:
        try:
            if not isinstance(entry, dict):
                continue
            repo = (entry.get("onnx_repo") or entry.get("hf_repo") or "").strip()
            if not repo:
                continue
            entry_id = str(entry.get("id") or "")
            name = str(entry.get("name") or entry_id or entry.get("ollama_tag") or "")
            # Prefer an exact <legacy_root>/<id> directory match (the v5.3.9
            # convention) before falling back to repo-owner globbing.
            id_dir = legacy_root / entry_id if entry_id else None
            if id_dir and id_dir.is_dir():
                offenders.append({"id": entry_id, "name": name, "path": id_dir})
                continue
            owner_name = repo.replace("/", "_")
            try:
                for candidate in legacy_root.rglob(f"*{owner_name}*"):
                    if candidate.is_dir():
                        offenders.append({"id": entry_id, "name": name, "path": candidate})
                        break
            except OSError:
                continue
        except Exception:
            continue
    return offenders


def scan_config_coherence(cfg: dict) -> list[dict]:
    """Return descriptors for ``cfg`` path keys whose targets are missing / wrong.

    Each returned dict has ``{"key": str, "value": str, "reason": str,
    "expected_path": Optional[Path], "expected_sentinel": Optional[str]}``.
    The caller turns each entry into a Repair / Ignore row.  Sentinels:

    * ``comfyui_dir`` -> ``main.py`` must exist inside it.
    * ``models_dir`` -> custom locations must exist as a directory; the
      canonical first-run ``<app>/models`` location is auto-created if missing.
    """
    if not isinstance(cfg, dict):
        return []
    findings: list[dict] = []

    # comfyui_dir: must exist AND contain main.py.
    comfyui_dir = (cfg.get("comfyui_dir") or "").strip()
    if comfyui_dir:
        p = Path(comfyui_dir)
        if not p.exists():
            findings.append({
                "key": "comfyui_dir",
                "value": comfyui_dir,
                "reason": "directory does not exist",
                "expected_path": p,
                "expected_sentinel": "main.py",
            })
        elif p.is_dir() and not (p / "main.py").is_file():
            findings.append({
                "key": "comfyui_dir",
                "value": comfyui_dir,
                "reason": "directory exists but is missing main.py sentinel",
                "expected_path": p,
                "expected_sentinel": "main.py",
            })

    # models_dir: must exist as a directory.  We don't require a sentinel
    # here because a fresh install has no models yet — the dir itself is
    # enough.
    models_dir = (cfg.get("models_dir") or "").strip()
    if models_dir:
        p = Path(models_dir)
        if not p.exists():
            if _same_path(p, _default_models_dir()):
                try:
                    p.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    findings.append({
                        "key": "models_dir",
                        "value": models_dir,
                        "reason": f"directory does not exist and could not be created: {exc}",
                        "expected_path": p,
                        "expected_sentinel": None,
                    })
            else:
                findings.append({
                    "key": "models_dir",
                    "value": models_dir,
                    "reason": "directory does not exist",
                    "expected_path": p,
                    "expected_sentinel": None,
                })

    return findings


def is_empty_tree(path: Path) -> bool:
    """Return True if ``path`` exists as a directory containing zero files.

    Used by the migration-apply paths to distinguish an *empty scaffold*
    (e.g. ``<app>\\ComfyUI\\models\\checkpoints`` pre-created by ``setup.bat``
    or by a half-finished install) from a real install with content.  This
    is the gating check that lets us safely ``shutil.rmtree`` the empty
    scaffold and then ``shutil.move`` the source on top — instead of
    falling into Windows' ``shutil.move``-into-existing-directory trap that
    nests the entire source one level deep.

    Missing paths return False so callers can dispatch the "destination
    absent" branch to bare ``shutil.move`` (which handles that case
    correctly).  Non-directory paths also return False.

    Empty *directories* (a leaf with no children at all) are reported as
    empty.  A nested tree of empty directories with no files anywhere is
    also reported as empty.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    try:
        for child in p.rglob("*"):
            try:
                if child.is_file():
                    return False
            except OSError:
                # Couldn't stat — treat as "not empty" so we never rmtree
                # something we can't see clearly.
                return False
    except OSError:
        return False
    return True


def safe_merge_directory(
    source: Path,
    destination: Path,
) -> dict:
    """Non-clobbering directory merge — mirrors ``robocopy /E /XO`` semantics.

    For every file in ``source``:

    * If the corresponding file in ``destination`` does not exist, copy it
      across (preserving metadata via :func:`shutil.copy2`).
    * If the destination file exists and the source's mtime is strictly
      newer, overwrite the destination.
    * Otherwise skip — destination wins, so anything the user downloaded
      since the stranded source last saw a write is preserved.

    This is the safe alternative to refusing the merge.  Empty source
    subdirectories are intentionally not propagated — HF caches re-create
    their dir layout on first access, so there is no value in carrying
    them over and no risk of "missing" data.

    Returns a dict with keys:

    * ``copied`` — int, files copied (new or overwritten).
    * ``skipped`` — int, source files skipped because dest is newer/same.
    * ``errors`` — list[str], best-effort error descriptions (one per file
      that could not be copied).  An overall scan failure is recorded with
      key ``"<scan>"``.
    """
    result: dict = {"copied": 0, "skipped": 0, "errors": []}
    source = Path(source)
    destination = Path(destination)
    if not source.exists() or not source.is_dir():
        return result
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["errors"].append(f"<dest mkdir>: {exc}")
        return result
    try:
        iterator = source.rglob("*")
    except OSError as exc:
        result["errors"].append(f"<scan>: {exc}")
        return result
    for src_file in iterator:
        try:
            if not src_file.is_file():
                continue
        except OSError:
            continue
        try:
            rel = src_file.relative_to(source)
        except ValueError:
            continue
        dst_file = destination / rel
        try:
            if dst_file.exists():
                try:
                    src_mtime = src_file.stat().st_mtime
                    dst_mtime = dst_file.stat().st_mtime
                except OSError:
                    result["skipped"] += 1
                    continue
                if src_mtime <= dst_mtime:
                    result["skipped"] += 1
                    continue
            try:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                result["errors"].append(f"{rel}: {exc}")
                continue
            shutil.copy2(str(src_file), str(dst_file))
            result["copied"] += 1
        except OSError as exc:
            result["errors"].append(f"{rel}: {exc}")
    return result


def scan_legacy_hf_cache(
    *,
    app_root: Optional[Path] = None,
    home: Optional[Path] = None,
) -> list[dict]:
    """Return per-dir descriptors for stale HuggingFace cache locations to migrate.

    Two known-bad legacy locations are detected:

    1. ``<app_root>/models/phase1`` — the pre-v5.3.10 hardcoded HF cache.  v5.3.10
       silently overrode any ambient ``HF_HOME`` to point here, so existing
       installs may still carry tens of GB at this path even after the env-var
       redirection in ``setup.bat`` / ``run.bat`` lands.  The cache is misnamed
       (it's not phase-specific, it's just HF Hub) and misplaced (caches under
       ``models/`` is confusing).
    2. ``~/.cache/huggingface`` — the HuggingFace SDK default; on cloud VMs and
       roamed user profiles this directory is often size-capped and was a major
       source of the original "no install disk space" install failure.

    The canonical post-fix location is ``<app_root>/.cache/huggingface``.

    Each returned entry: ``{"source": Path, "destination": Path, "size_gb":
    float, "label": str}``.  Empty list when neither legacy location has any
    real content.
    """
    if app_root is None:
        app_root = Path(__file__).parent.parent
    app_root = Path(app_root)
    if home is None:
        try:
            home = Path.home()
        except (RuntimeError, OSError):
            home = None

    canonical = app_root / ".cache" / "huggingface"

    legacy: list[tuple[Path, str]] = [
        (app_root / "models" / "phase1", "Bundled HF cache under models/phase1"),
    ]
    if home is not None:
        legacy.append((home / ".cache" / "huggingface", "User-profile HF cache (~/.cache/huggingface)"))

    findings: list[dict] = []
    for source, label in legacy:
        try:
            if not source.exists() or not source.is_dir():
                continue
            try:
                if source.resolve() == canonical.resolve():
                    continue
            except OSError:
                pass
            size_bytes = 0
            try:
                for f in source.rglob("*"):
                    if f.is_file():
                        try:
                            size_bytes += f.stat().st_size
                        except OSError:
                            continue
            except OSError:
                continue
            if size_bytes <= 0:
                continue
            findings.append({
                "source": source,
                "destination": canonical,
                "size_gb": size_bytes / 1_073_741_824,
                "label": label,
            })
        except Exception:
            continue
    return findings


def scan_disk_space(
    cfg: dict,
    *,
    threshold_gb: float = DISK_SPACE_WARN_THRESHOLD_GB,
    extra_paths: Optional[Iterable[Path]] = None,
) -> list[dict]:
    """Return warn-rows for any drive holding a configured path with low free space.

    Each entry: ``{"drive": str, "free_gb": float, "paths": list[str]}``.
    Drives are deduplicated; a single low-free-space drive yields a single
    row regardless of how many configured paths live on it.
    """
    if not isinstance(cfg, dict):
        return []
    candidates: list[Path] = []
    for key in ("models_dir", "comfyui_dir"):
        v = (cfg.get(key) or "").strip()
        if v:
            candidates.append(Path(v))
    env_ollama = os.environ.get("OLLAMA_MODELS", "").strip()
    if env_ollama:
        candidates.append(Path(env_ollama))
    if extra_paths:
        for p in extra_paths:
            candidates.append(Path(p))

    by_drive: dict[str, dict[str, Any]] = {}
    for p in candidates:
        try:
            drive = p.drive or str(p)
            key = drive.upper()
        except Exception:
            continue
        free = free_space_bytes(p)
        free_gb = free / 1_073_741_824
        entry = by_drive.setdefault(key, {"drive": drive, "free_gb": free_gb, "paths": []})
        if str(p) not in entry["paths"]:
            entry["paths"].append(str(p))
        # Keep the smallest free-gb observation for the drive (consistency).
        entry["free_gb"] = min(entry["free_gb"], free_gb)

    return [
        entry for entry in by_drive.values()
        if entry["free_gb"] < float(threshold_gb)
    ]


def scan_uncatalogued_ollama_tags(
    installed: Iterable[str],
    catalog_entries: Iterable[dict],
) -> list[str]:
    """Return installed Ollama tags that have no matching ``ollama_tag`` in *catalog_entries*.

    Both sides are normalised via :func:`normalize_ollama_tag` so bare
    catalog entries like ``"phi4"`` correctly match installed
    ``"phi4:latest"``.
    """
    catalog_set: set[str] = set()
    for entry in catalog_entries:
        if not isinstance(entry, dict):
            continue
        tag = entry.get("ollama_tag") or ""
        norm = normalize_ollama_tag(tag)
        if norm:
            catalog_set.add(norm)
    uncatalogued: list[str] = []
    seen: set[str] = set()
    for tag in installed:
        norm = normalize_ollama_tag(tag)
        if not norm:
            continue
        if norm in catalog_set:
            continue
        # Preserve the originally-presented spelling.
        display = (tag or "").strip()
        if display and display not in seen:
            uncatalogued.append(display)
            seen.add(display)
    return uncatalogued


def scan_uncatalogued_onnx_dirs(
    models_dir: Path,
    catalog_entries: Iterable[dict],
) -> list[Path]:
    """Return on-disk ONNX model directories not present as ``id`` in the catalog.

    Scans both ``<models_dir>\\onnx\\*`` (the canonical v5.3.9 layout) and
    direct children of ``<models_dir>`` (legacy / hand-installed entries).
    Reserved cache / scaffolding names (see :data:`RESERVED_MODELS_DIR_NAMES`)
    are NEVER reported — that's the safeguard that keeps the canonical
    HuggingFace cache off the deletion list.
    """
    root = Path(models_dir)
    if not root.is_dir():
        return []
    catalog_ids: set[str] = set()
    catalog_aliases: set[str] = set()
    for entry in catalog_entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("onnx_repo"):
            entry_id = str(entry.get("id") or "").strip()
            if entry_id:
                catalog_ids.add(entry_id)
                # Filesystem-safe alias (the v5.3.9 onnx_repo extraction
                # used owner_name conversion).
                catalog_aliases.add(entry_id.replace(":", "_"))
                catalog_aliases.add(entry_id.replace("/", "_"))
            repo = str(entry.get("onnx_repo") or "").strip()
            if repo:
                catalog_aliases.add(repo.replace("/", "_"))

    reserved_lower = {n.lower() for n in RESERVED_MODELS_DIR_NAMES}
    candidates: list[Path] = []
    onnx_root = root / "onnx"
    try:
        if onnx_root.is_dir():
            for entry in onnx_root.iterdir():
                if entry.is_dir():
                    candidates.append(entry)
    except OSError:
        pass
    try:
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.lower() in reserved_lower:
                continue
            candidates.append(entry)
    except OSError:
        pass

    uncatalogued: list[Path] = []
    seen: set[str] = set()
    for entry in candidates:
        key = str(entry).lower()
        if key in seen:
            continue
        seen.add(key)
        name = entry.name
        if name in catalog_ids:
            continue
        if name in catalog_aliases:
            continue
        # Some catalog entries embed slashes; also accept any catalog id
        # whose filesystem-safe form equals this dir's name.
        if any(name == cid.replace(":", "_").replace("/", "_") for cid in catalog_ids):
            continue
        uncatalogued.append(entry)
    return uncatalogued


__all__ = [
    "MigrationEngine",
    "MigrationPlan",
    "MigrationPhase",
    "MigrationState",
    "ProgressEvent",
    "PreflightResult",
    "MigrationCancelled",
    "MigrationVerifyFailed",
    "STATE_FILE_NAME",
    "IDENTITY_KEY_FILENAMES",
    "REBOOT_VOLATILE_DRIVE_TYPES",
    "USER_PROMPT_RESUME_PHASES",
    "RESERVED_MODELS_DIR_NAMES",
    "DISK_SPACE_WARN_THRESHOLD_GB",
    "SEVERITY_INFO",
    "SEVERITY_WARN",
    "SEVERITY_ACTION",
    "Finding",
    "state_file_path",
    "read_state",
    "write_state",
    "clear_state",
    "find_resumable_state",
    "process_scheduled_deletes",
    "get_drive_type",
    "is_reboot_volatile_drive",
    "same_volume",
    "measure_tree_size",
    "free_space_bytes",
    "has_comfyui_sentinel",
    "has_ollama_sentinel",
    "has_models_sentinel",
    "cross_validate_ollama_manifests",
    "iter_manifest_paths",
    "collect_manifest_blob_digests",
    "parse_robocopy_line",
    "normalize_ollama_tag",
    "scan_orphan_ollama_blobs",
    "scan_legacy_onnx_paths",
    "scan_legacy_hf_cache",
    "scan_config_coherence",
    "scan_disk_space",
    "scan_uncatalogued_ollama_tags",
    "scan_uncatalogued_onnx_dirs",
]

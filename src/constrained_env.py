# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Constrained-VM environment detection and Ollama disk-pressure helpers.

Some virtual desktop SKUs ship with a much tighter disk-space envelope
than a normal desktop: the user-profile container is a fixed-size volume
(often 30 GB) regardless of how much physical disk is free on the
underlying host, real-time antivirus scans run on every blob write, and
sessions are non-persistent and can be paused/recycled at any moment.

The most user-visible failure mode of these constraints is that Ollama
``pull_model`` succeeds for the FIRST model in a benchmark run but fails
on the SECOND because the user-profile container is full — Ollama emits
a daemon-level JSON error (``{"error": "...no space left..."}``) that
our retry loop correctly treats as fatal.  We surface a profile-aware
hint when that happens, and a one-time relocation workaround (the
shipped helper ``set_ollama_models_dir.bat`` flips ``OLLAMA_MODELS`` to
``<app_path>\\Ollama`` for the user).

Helpers exported here are intentionally **best-effort**: every probe
wraps its registry / WMI / env-var access in ``try/except`` so a missing
key or a permission error never crashes the caller.  Results are cached
per-process because the answer is stable for the life of a session and
the performance reviewer flagged the pre-pull disk-space check as a
warm-path cost we must not pay twice.

Privacy invariant: this module is allowed to SAY "constrained VM" in
user-visible text, but it must never leak hardware specs (GPU model,
VRAM, RAM, CPU counts).  See the SKU-privacy section of
``docs/architecture.md`` §5.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from src import logger as _log

# Free-space threshold below which we tell the user the user-profile
# container is "tight".  3 GB is conservative — most Ollama chat models
# are 0.5–8 GB and the user needs headroom for at least one more pull
# plus the generation cache.  Below this we offer the OLLAMA_MODELS
# relocation hint.
_CONSTRAINED_DISK_PRESSURE_GB = 3.0

# Substrings we treat as "the daemon is telling us disk is full".  Match
# is case-insensitive.  Keep this list narrow: matching too broadly
# turns a legitimate model-not-found error into a confusing disk-full
# hint.
_DISK_FULL_ERROR_SUBSTRINGS: tuple[str, ...] = (
    "no space left",
    "disk full",
    "errno 28",
    "enospc",
    "insufficient storage",
    "not enough space",
    "input/output error",
    "permission denied",
    "access is denied",
    "write-protected",
)

# Sentinel string the rest of the app uses to identify quota-aware
# Ollama errors.  Tests pin this exact wording so the UX reviewer's
# actionability requirement can be regression-tested.  Keep it ONE line
# (no markdown, no newlines) — it has to fit cleanly in the benchmark
# log textbox.
CONSTRAINED_OLLAMA_HINT_PREFIX = (
    "Ollama pull failed (the user-profile drive may be full — try "
    "clearing previously downloaded models from the Models tab, or "
    "relocate Ollama models off the user-profile drive by running "
    "set_ollama_models_dir.bat from your LocalAI folder and restarting "
    "Ollama)"
)

# Friendly banner the BatchRunner prints at the top of the log when
# EVERY Ollama model in the run was skipped for environment reasons.
# Also pinned by tests.
CONSTRAINED_ALL_OLLAMA_SKIPPED_BANNER = (
    "All Ollama models were skipped because the user-profile drive is "
    "too full to download them — clear unused models from the Models "
    "tab or relocate the Ollama models directory off the user-profile "
    "drive."
)

# Per-process cache so the pre-pull check stays a noop on the warm path.
_CONSTRAINED_ENV_CACHE: dict[str, object] = {}
_CONSTRAINED_ENV_CACHE_LOCK = threading.Lock()


# ── Environment detection ─────────────────────────────────────────────────────


def _looks_like_user_profile_container_present() -> bool:
    """Return True if a user-profile-container agent appears installed.

    The standard roaming profile container installation ships ``frxsvc.exe``
    (the agent service) and ``frxccd.sys`` (the filter driver).  We
    don't need to talk to either — the presence of the install directory
    or the service registry key is enough.  All probes are best-effort.
    """
    if sys.platform != "win32":
        return False
    candidates = [
        Path(r"C:\Program Files\FSLogix\Apps\frxsvc.exe"),
        Path(r"C:\Program Files\FSLogix\Apps\frxccd.sys"),
    ]
    for c in candidates:
        try:
            if c.exists():
                return True
        except Exception:
            continue
    try:
        import winreg  # type: ignore
        for service in ("frxsvc", "frxdrv", "frxccd"):
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    rf"SYSTEM\CurrentControlSet\Services\{service}",
                ):
                    return True
            except OSError:
                continue
    except Exception:
        pass
    return False


def _looks_like_managed_vm() -> bool:
    """Return True if this looks like a managed virtual desktop session.

    We check a few small signals: managed-VM-flavoured environment
    variables set by the provisioning flow, the directory-joined marker
    via ``dsregcmd``, and host-naming conventions used by virtual
    desktop fleets.  Any one of these is enough — they're all weak
    signals on their own but together they're very specific to a
    constrained managed VM.
    """
    if sys.platform != "win32":
        return False
    # Env-var hints that the provisioning agent leaves behind.
    for var in ("CLOUDPC", "CLOUDPCSESSIONID", "WVD_SESSION_ID", "AVD_SESSION_ID"):
        if os.environ.get(var):
            return True
    # Computer name pattern: many managed-VM hosts ship as <PREFIX>-<user>-<id>.
    computer = os.environ.get("COMPUTERNAME", "")
    if computer.upper().startswith("CPC-"):
        return True
    # Optional fallback: cloud IMDS metadata service.  We deliberately
    # do NOT hit the network from here — it adds latency on the warm
    # path and a hung probe would slow startup.  Tests can synthesise
    # this via the env-var path above.
    return False


def is_constrained_vm(*, force_refresh: bool = False) -> bool:
    """Return True if we're running on a quota-constrained managed VM.

    We combine the managed-VM signal with the user-profile-container
    signal: either alone is too noisy to act on (profile containers are
    sometimes installed on company laptops; managed-VM env vars can leak
    through RDP sessions to a normal VM) but together they're a strong
    signal that the profile-container-fills-up failure mode applies.

    Override with ``LOCALAI_FORCE_CONSTRAINED=1`` (force True) or
    ``LOCALAI_FORCE_CONSTRAINED=0`` (force False) so tests and operators
    can manually toggle without modifying the host.
    """
    cache_key = "is_constrained"
    if not force_refresh:
        with _CONSTRAINED_ENV_CACHE_LOCK:
            cached = _CONSTRAINED_ENV_CACHE.get(cache_key)
            if cached is not None:
                return bool(cached)
    override = os.environ.get("LOCALAI_FORCE_CONSTRAINED", "")
    if override == "1":
        result = True
    elif override == "0":
        result = False
    else:
        result = _looks_like_managed_vm() and _looks_like_user_profile_container_present()
    with _CONSTRAINED_ENV_CACHE_LOCK:
        _CONSTRAINED_ENV_CACHE[cache_key] = result
    return result


# ── Ollama models-dir helpers ────────────────────────────────────────────────


def _default_ollama_models_dir() -> Path:
    """Return the LocalAI-preferred default Ollama models directory.

    This is the **proposed** default we surface to the user in the
    Settings ``Ollama Models Directory`` row and in the first-run
    relocation prompt. It is *not* applied automatically because
    ``OLLAMA_MODELS`` is global state read by every other Ollama client
    on the box (Open-WebUI, Continue.dev, etc.); LocalAI must never
    redirect that infrastructure without explicit user consent.

    Resolves to ``<app_path>/Ollama`` on both Windows and macOS — same
    app-path-relative shape ``src.config._default_data_dir`` uses for
    ``models_dir`` and ``comfyui_dir``. Drive letter (Windows) / volume
    (macOS) is whichever drive the app itself was unpacked to; we never
    hardcode ``D:`` / ``E:`` / ``%HOMEDRIVE%`` / ``%LOCALAPPDATA%``
    anywhere. Pure function — no filesystem or env probes.

    ``constrained_env.py`` sits at ``<app_path>/src/constrained_env.py``
    so ``Path(__file__).parent.parent`` is the app path.
    """
    return Path(__file__).parent.parent / "Ollama"


def get_ollama_models_dir() -> Path:
    """Return the directory Ollama writes blob files into.

    Honors ``$OLLAMA_MODELS`` exactly the way the Ollama daemon does:
    if the env var is set AND points at an existing directory we use
    it, otherwise we fall back to the per-user default
    ``~/.ollama/models``.

    Result is **not** cached — the user can change ``OLLAMA_MODELS`` and
    restart Ollama between two batch runs and we want to pick that up
    without restarting LocalAI.
    """
    env = os.environ.get("OLLAMA_MODELS", "").strip()
    if env:
        p = Path(env)
        try:
            if p.is_dir():
                return p
        except Exception:
            pass
        # Even if the directory doesn't exist yet we still return the
        # env-var value so the caller can report it back to the user.
        return p
    return Path.home() / ".ollama" / "models"


def get_ollama_models_dir_free_gb() -> float:
    """Return free space (GB) on the volume holding the Ollama models dir.

    Returns 0.0 if anything goes wrong (volume offline, permission
    error, path doesn't exist and parent is missing).  Callers must
    treat 0.0 as "unknown, don't block".
    """
    target = get_ollama_models_dir()
    # disk_usage requires an existing path; walk up to the first parent
    # that exists.
    probe: Optional[Path] = target
    while probe is not None and not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    if probe is None or not probe.exists():
        return 0.0
    try:
        return shutil.disk_usage(str(probe)).free / 1_073_741_824
    except Exception:
        return 0.0


def constrained_disk_pressure() -> Optional[str]:
    """Return a human-readable warning if any quota-sensitive volume is tight.

    Volumes checked: ``%USERPROFILE%`` (user-profile container),
    ``%TEMP%``, the Ollama models directory.  Returns None when
    everything is healthy.

    The threshold is ``_CONSTRAINED_DISK_PRESSURE_GB`` (3 GB) —
    conservative so we offer the relocation hint before the user hits
    the wall, not after.  Result is intentionally NOT cached: the whole
    point is that the user-profile container fills up DURING a run.
    """
    if not is_constrained_vm():
        return None

    checks: list[tuple[str, Path]] = []
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        checks.append(("User-profile drive", Path(user_profile)))
    temp = os.environ.get("TEMP", "")
    if temp:
        checks.append(("Temp drive", Path(temp)))
    checks.append(("Ollama models drive", get_ollama_models_dir()))

    tight: list[str] = []
    for label, path in checks:
        # Walk up to the first existing parent so a not-yet-created
        # OLLAMA_MODELS directory doesn't hide a disk-pressure signal
        # on the parent volume.
        probe: Optional[Path] = path
        while probe is not None and not probe.exists():
            parent = probe.parent
            if parent == probe:
                probe = None
                break
            probe = parent
        if probe is None:
            continue
        try:
            free_gb = shutil.disk_usage(str(probe)).free / 1_073_741_824
        except Exception:
            continue
        if free_gb < _CONSTRAINED_DISK_PRESSURE_GB:
            tight.append(f"{label}: {free_gb:.1f} GB free")

    if not tight:
        return None
    return "Constrained-VM disk pressure — " + "; ".join(tight)


# ── Ollama error classification ──────────────────────────────────────────────


def is_disk_full_error_text(text: str) -> bool:
    """Return True if *text* (an Ollama daemon error string) signals disk-full.

    Centralised here so the test suite, the BatchRunner, and the UI
    download path all agree on what counts as a disk-pressure error.
    """
    lower = str(text or "").lower()
    return any(token in lower for token in _DISK_FULL_ERROR_SUBSTRINGS)


def quota_aware_ollama_error(original: str) -> str:
    """Wrap an Ollama daemon error with a profile-drive-aware hint if applicable.

    If we detect a disk-full pattern in *original* AND we're on a
    constrained managed VM with a profile-drive cap, prepend
    :data:`CONSTRAINED_OLLAMA_HINT_PREFIX` and suffix the original
    error.  Otherwise return *original* unchanged.

    The returned string is single-line (newlines collapsed to spaces)
    and capped at 400 chars so it renders cleanly in the BatchRunner
    log column without scrolling.

    Renamed from ``profile_aware_ollama_error`` to use neutral
    terminology — the old name is kept as a backward-compatible alias
    for in-tree tests.
    """
    text = (original or "").strip()
    if not text:
        text = "unknown Ollama error"
    if not is_disk_full_error_text(text):
        return text
    if not is_constrained_vm():
        return text
    combined = f"{CONSTRAINED_OLLAMA_HINT_PREFIX}: {text}"
    combined = " ".join(combined.split())
    if len(combined) > 400:
        combined = combined[:399] + "…"
    return combined


# Backward-compatible alias for past rename.  Keep the older name exported
# so legacy callers and any existing test suite keep working.
profile_aware_ollama_error = quota_aware_ollama_error


# ── Pre-pull disk-space check ────────────────────────────────────────────────


def precheck_ollama_pull(size_gb: float) -> Optional[str]:
    """Return a skip-reason if the Ollama models dir can't fit *size_gb*.

    Returns None when there is enough room (or we couldn't tell, in
    which case we MUST let the pull proceed — failing-closed here would
    be a regression vs. the pre-constrained-VM behaviour).  When a
    skip-reason is returned the caller should classify the failure as
    ``failure_phase="environment_skip"`` so it doesn't count against
    ``_consecutive_failures``.

    The 10 % headroom matches ``resource_manager.check_disk_for_download``
    so the two paths produce consistent answers.
    """
    needed = float(size_gb or 0) * 1.1
    if needed <= 0:
        return None
    free = get_ollama_models_dir_free_gb()
    if free <= 0:
        # Couldn't measure — don't block.  Returning None here is what
        # keeps Ron's local machine on the happy path even if the
        # disk_usage probe fails for some path-quirk reason.
        return None
    if free >= needed:
        return None
    target = get_ollama_models_dir()
    if is_constrained_vm():
        hint = (
            " Relocate Ollama models off the user-profile drive by "
            "running set_ollama_models_dir.bat from your LocalAI folder "
            "and restarting Ollama, or clear unused models from the "
            "Models tab."
        )
    else:
        hint = " Free up disk and try again."
    msg = (
        f"needs ~{size_gb:.1f} GB but the Ollama models directory "
        f"({target}) has only {free:.1f} GB free."
    ) + hint
    return msg


# ── Test / diagnostic helpers ────────────────────────────────────────────────


def _reset_cache_for_tests() -> None:
    """Drop the per-process cache.  Used by the unit tests only."""
    with _CONSTRAINED_ENV_CACHE_LOCK:
        _CONSTRAINED_ENV_CACHE.clear()


__all__ = [
    "CONSTRAINED_OLLAMA_HINT_PREFIX",
    "CONSTRAINED_ALL_OLLAMA_SKIPPED_BANNER",
    "quota_aware_ollama_error",
    "profile_aware_ollama_error",  # legacy alias
    "constrained_disk_pressure",
    "get_ollama_models_dir",
    "get_ollama_models_dir_free_gb",
    "is_disk_full_error_text",
    "is_constrained_vm",
    "precheck_ollama_pull",
]

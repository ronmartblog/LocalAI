# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Batch benchmark report: structured results and report generation.
"""

import html
import json
import os
import platform
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# v5.5.1+ short prefix used by every new run. The historical
# ``localai_benchmark`` prefix is still recognised by
# :func:`find_latest_report_json` so old reports stored in
# ``benchmark_results\archive\`` still load for "Retry Failed".
REPORT_PREFIX = "bench"
LEGACY_REPORT_PREFIX = "localai_benchmark"
LEGACY_JSON_NAME = "batch_results.json"
LEGACY_TEXT_NAME = "batch_results.txt"
LEGACY_HTML_NAME = "batch_results.html"
LATEST_QUICK_JSON_NAME = "latest_quick_benchmark.json"
LATEST_QUICK_HTML_NAME = "latest_quick_benchmark.html"
LATEST_EXTENDED_JSON_NAME = "latest_extended_benchmark.json"
LATEST_EXTENDED_HTML_NAME = "latest_extended_benchmark.html"
ARCHIVE_SUBDIR_NAME = "archive"
VALID_RUN_MODES = ("quick", "extended")

# Filename suffixes/markers we recognise when sweeping previous-run files into
# ``archive\``. Per-mode "latest_*" aliases are intentionally NOT in this list
# because they are overwritten on every save and never accumulate.
_REPORT_EXTENSIONS: tuple[str, ...] = (".json", ".html", ".txt")
_REPORT_PREFIXES: tuple[str, ...] = (REPORT_PREFIX, LEGACY_REPORT_PREFIX)
_LATEST_ALIAS_NAMES: frozenset[str] = frozenset({
    LATEST_QUICK_JSON_NAME,
    LATEST_QUICK_HTML_NAME,
    LATEST_EXTENDED_JSON_NAME,
    LATEST_EXTENDED_HTML_NAME,
})
# Diagnostic sidecars dropped next to a partial-failure run; archive_previous
# treats them as part of a run when sweeping the folder.
_DIAGNOSTIC_SIDECAR_SUFFIXES: tuple[str, ...] = (
    "_failures.txt",
    "_env.txt",
    "_run.log",
)
# Sibling directories that travel with a report stem (image artifacts, raw
# per-model logs). When archiving a previous run we move these too so the
# folder stays clean.
_REPORT_SIBLING_SUFFIXES: tuple[str, ...] = ("_images", "_logs")
_MACHINE_INFO_CACHE: dict[str, dict] = {}
_LOCALAI_VERSION_CACHE: str | None = None
UNKNOWN_LOCALAI_VERSION = "unknown"


def normalize_run_mode(value: object, *, default: str = "quick") -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_RUN_MODES else default


def _per_mode_latest_json_name(run_mode: str) -> str:
    if run_mode == "extended":
        return LATEST_EXTENDED_JSON_NAME
    return LATEST_QUICK_JSON_NAME


def _per_mode_latest_html_name(run_mode: str) -> str:
    if run_mode == "extended":
        return LATEST_EXTENDED_HTML_NAME
    return LATEST_QUICK_HTML_NAME


def _run_mode_label(run_mode: str) -> str:
    if run_mode == "extended":
        return "Extended"
    return "Quick"


def detect_localai_version() -> str:
    """Best-effort APP_VERSION discovery without importing the Tk app module."""
    global _LOCALAI_VERSION_CACHE
    if _LOCALAI_VERSION_CACHE is not None:
        return _LOCALAI_VERSION_CACHE
    try:
        app_path = Path(__file__).resolve().parent / "app.py"
        text = app_path.read_text(encoding="utf-8")
        match = re.search(
            r'^\s*APP_VERSION\s*=\s*["\']([^"\']+)["\']',
            text,
            flags=re.MULTILINE,
        )
        if match:
            version = str(match.group(1) or "").strip()
            if version:
                _LOCALAI_VERSION_CACHE = version
                return version
    except Exception:
        pass
    _LOCALAI_VERSION_CACHE = UNKNOWN_LOCALAI_VERSION
    return _LOCALAI_VERSION_CACHE


def display_method(method: str) -> str:
    """Return a user-facing method label while preserving stored method IDs."""
    if method == "phase1":
        return "utility"
    if method == "image_comfyui":
        return "image (ComfyUI)"
    return method


_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
)


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_seconds(start: object, end: object) -> int:
    start_dt = _parse_timestamp(start)
    end_dt = _parse_timestamp(end)
    if not start_dt or not end_dt:
        return 0
    delta = end_dt - start_dt
    seconds = int(delta.total_seconds())
    return max(seconds, 0)


def _format_duration_hms(seconds: object) -> str:
    """Format a non-negative second count as ``H:MM:SS``.

    Handles >24 h runs by extending the hours field rather than rolling over
    into days, so a 50-hour benchmark reads ``50:12:03`` instead of ``2 days``.
    """
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return "0:00:00"
    total = max(total, 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _model_guide_fragment(model_id: object) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(model_id or ""))
    slug = "-".join(part for part in slug.split("-") if part)
    return f"model-{slug}" if slug else ""


def _model_guide_href(model_id: object) -> str:
    fragment = _model_guide_fragment(model_id)
    return f"../docs/Model-Guide.html#{fragment}" if fragment else "../docs/Model-Guide.html"


def _model_guide_anchor(model_id: object, label_html: str) -> str:
    href = html.escape(_model_guide_href(model_id), quote=True)
    return f'<a href="{href}">{label_html}</a>'


def _render_prompt_source(prompt_source: str, model_id: object) -> str:
    source = str(prompt_source or "").strip()
    if source.lower().startswith("model-guide.html"):
        return _model_guide_anchor(model_id, "Model Guide")
    return html.escape(source)


def _default_surface_for_method(method: str) -> str:
    if method in {"phase1", "utility"}:
        return "utility"
    if method == "image_comfyui":
        return "image"
    if method.startswith("onnx_"):
        return "onnx"
    if method.startswith("ollama_"):
        return "chat"
    return "chat"


def _safe_filename_part(value: object, *, fallback: str = "unknown", max_len: int = 64) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    if not text:
        text = fallback
    return text[:max_len]


def _start_time_for_filename(start_time: str) -> str:
    """Render *start_time* as ``YYYY-MM-DD_HHMM`` for the report stem.

    v5.5.1+ drops the seconds field — minute precision is unique enough
    because the stem also carries the machine name + run mode, and previous
    runs are moved into ``archive/`` on the next save anyway.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(start_time, fmt).strftime("%Y-%m-%d_%H%M")
        except ValueError:
            pass
    return _safe_filename_part(start_time.replace(":", "-"), fallback="unknown-time", max_len=32)


_GPU_BRAND_PREFIX_RE = re.compile(
    r"^(NVIDIA|GeForce|RTX|AMD|Radeon|Intel|Arc|Apple)[\s_]+",
    flags=re.IGNORECASE,
)


def _short_gpu_label(value: object) -> str:
    """Compact a GPU display string for use inside a benchmark filename.

    Strips one leading brand word (``NVIDIA_``, ``AMD_``, ``Intel_``, ``Apple_``,
    etc.) so reports read ``A10-24Q`` instead of ``NVIDIA_A10-24Q``. Always
    routes through :func:`_safe_filename_part` so OS-illegal characters and
    whitespace are normalised.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    text = _GPU_BRAND_PREFIX_RE.sub("", text, count=1).strip("_-. ")
    return _safe_filename_part(text, fallback="", max_len=24)


def _round_gb(value: object) -> int:
    try:
        return max(0, int(round(float(value or 0))))
    except (TypeError, ValueError):
        return 0


def collect_machine_info(models_path: str | Path = ".") -> dict:
    """Collect stable run-level machine metadata for benchmark reports."""
    try:
        cache_key = str(Path(models_path).resolve())
    except Exception:
        cache_key = str(models_path)
    cached = _MACHINE_INFO_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    node = platform.node() or ""
    info = {
        "machine_name": node or "unknown-machine",
        "os": platform.platform(),
        "platform": platform.processor(),
        "python": platform.python_version(),
        "cpu": os.cpu_count() or 0,
        "ram_gb": 0,
        "gpu_name": "",
        "gpu_count": 0,
        "vram_gb": 0,
        "npu_count": 0,
        "storage_total_gb": 0.0,
        "storage_free_gb": 0.0,
    }
    try:
        from src import system_info

        ram = system_info.get_ram_info()
        if ram:
            info["ram_gb"] = _round_gb((ram.get("total_mb", 0) or 0) / 1024)
            info["ram_available_gb"] = round((ram.get("available_mb", 0) or 0) / 1024, 1)
        gpus = system_info.get_gpu_info()
        info["gpu_count"] = len(gpus)
        if gpus:
            best = max(gpus, key=lambda g: g.get("vram_total_mb", 0) or 0)
            info["gpu_name"] = best.get("name", "") or ""
            info["vram_gb"] = _round_gb((best.get("vram_total_mb", 0) or 0) / 1024)
            info["gpu_unified_memory"] = bool(best.get("unified_memory"))
        info["npu_count"] = len(system_info.get_npu_info())
        storage = system_info.get_storage_info(models_path)
        info["storage_total_gb"] = round(float(storage.get("total_gb", 0) or 0), 1)
        info["storage_free_gb"] = round(float(storage.get("free_gb", 0) or 0), 1)
        try:
            sku = system_info.build_local_sku()
            if sku:
                info["machine_model"] = sku.get("name", "")
                info["vm_size_pattern"] = sku.get("vm_size_pattern", "")
                info["gpu_fraction"] = sku.get("gpu_fraction", "")
                if sku.get("cpu"):
                    info["cpu"] = int(sku["cpu"])
                if sku.get("ram_gb"):
                    info["ram_gb"] = _round_gb(sku["ram_gb"])
                if sku.get("vram_gb") is not None:
                    info["vram_gb"] = _round_gb(sku.get("vram_gb"))
        except Exception:
            pass
    except Exception:
        pass
    _MACHINE_INFO_CACHE[cache_key] = dict(info)
    return dict(info)


def make_report_stem(
    machine_info: dict,
    start_time: str,
    *,
    run_mode: str | None = None,
) -> str:
    """Build the per-run filename stem.

    v5.5.1+ uses the compact ``bench_<mode>_<machine>_<cpu>cpu_<ram>ram_<gpu>_<vram>vram_<TS>``
    shape so the report folder stays readable (the old
    ``localai_benchmark_..._<vcpu>vcpu_<ram>gb-ram_..._<vram>gb-vram_<HMS>``
    stems were ~90 chars and crowded out the actual report when archive
    sweeps weren't yet enabled). The same machine_info dict and start_time
    feed both naming generations; legacy stems are still understood by
    :func:`find_latest_report_json` so older reports in ``archive/`` load
    cleanly for "Retry Failed".
    """
    machine = _safe_filename_part(
        machine_info.get("machine_name")
        or machine_info.get("machine_model")
        or "machine",
        fallback="machine",
        max_len=32,
    )
    cpu = _round_gb(machine_info.get("cpu") or machine_info.get("vcpu"))
    ram_gb = _round_gb(machine_info.get("ram_gb"))
    vram_gb = _round_gb(machine_info.get("vram_gb"))
    gpu_raw = machine_info.get("gpu_name") or (
        "unified" if machine_info.get("gpu_unified_memory") else "no-gpu"
    )
    gpu_part = _short_gpu_label(gpu_raw) or "no-gpu"
    timestamp = _start_time_for_filename(start_time)
    mode_part = ""
    if run_mode:
        mode_norm = normalize_run_mode(run_mode, default="")
        if mode_norm:
            mode_part = f"_{mode_norm}"
    return (
        f"{REPORT_PREFIX}{mode_part}_{machine}_"
        f"{cpu}cpu_{ram_gb}ram_{gpu_part}_{vram_gb}vram_{timestamp}"
    )


def find_latest_report_json(output_dir: Path) -> Optional[Path]:
    """Return the newest benchmark JSON report.

    Recognises both the new ``bench_*.json`` stems (v5.5.1+) and the
    historical ``localai_benchmark_*.json`` stems so that retry / "open
    latest" flows still find reports that have been archived from older
    runs. Also still resolves the long-deprecated ``batch_results.json``
    legacy alias for users whose folders predate archive sweeps.
    """
    output_dir = Path(output_dir)
    candidates: list[Path] = []
    for prefix in _REPORT_PREFIXES:
        candidates.extend(output_dir.glob(f"{prefix}_*.json"))
    # Sweep the archive subdirectory too so "Retry Failed" can target the
    # most recent run that has already been archived by a subsequent save.
    archive_dir = output_dir / ARCHIVE_SUBDIR_NAME
    if archive_dir.is_dir():
        for prefix in _REPORT_PREFIXES:
            candidates.extend(archive_dir.glob(f"{prefix}_*.json"))
    legacy = output_dir / LEGACY_JSON_NAME
    if legacy.exists():
        legacy_is_alias = False
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
            file_stem = legacy_data.get("file_stem")
            alias_target = output_dir / f"{file_stem}.json" if file_stem else None
            if alias_target and alias_target != legacy and alias_target.exists():
                legacy_is_alias = True
        except Exception:
            legacy_is_alias = False
        if not legacy_is_alias:
            candidates.append(legacy)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _looks_like_report_basename(name: str) -> bool:
    """Return True when *name* looks like a benchmark report file/dir to sweep."""
    for prefix in _REPORT_PREFIXES:
        if name.startswith(prefix + "_"):
            return True
    return False


def archive_previous_runs(
    output_dir: Path,
    *,
    keep_stem: str | None = None,
) -> list[Path]:
    """Move previous run files into ``output_dir / archive``.

    Walks *output_dir* and relocates every file whose name looks like a
    benchmark report stem (``bench_*`` or ``localai_benchmark_*``) into
    the ``archive/`` subdirectory, along with matching ``_images/`` /
    ``_logs/`` sibling folders and the optional per-run diagnostic
    sidecars (``*_failures.txt``, ``*_env.txt``, ``*_run.log``). Stale
    pre-v5.5.1 legacy aliases (``batch_results.json`` /``.html`` /``.txt``)
    are swept up too.

    *keep_stem* preserves the in-flight run's own files (and their image /
    log siblings) so incremental saves during the run don't immediately
    self-archive. Per-mode ``latest_*_benchmark.{json,html}`` aliases stay
    in place — they overwrite, not accumulate.

    Returns the list of source paths that were moved (useful for tests
    and the runner banner).
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    archive_dir = output_dir / ARCHIVE_SUBDIR_NAME
    archive_dir_created = False
    moved: list[Path] = []
    # v5.5.1+ perf: use os.scandir() instead of Path.iterdir() so the
    # follow-up is_file()/is_dir() checks hit the cached DirEntry stat
    # info instead of issuing a fresh syscall per entry. On OneDrive-
    # backed folders (the user's actual cwd) this cuts the per-entry
    # cost ~10x because every stat triggers reparse-point hydration
    # otherwise. Drops the first-save sweep from ~1s to ~100ms on a
    # folder with 50 prior runs.
    try:
        scan_iter = list(os.scandir(output_dir))
    except OSError:
        return []
    try:
        for entry in scan_iter:
            name = entry.name
            if name == ARCHIVE_SUBDIR_NAME:
                continue
            if name in _LATEST_ALIAS_NAMES:
                continue
            try:
                is_file = entry.is_file(follow_symlinks=False)
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            # Skip files belonging to the currently-active run so incremental
            # saves don't trigger spurious moves mid-run.
            if keep_stem:
                if is_file:
                    if name == f"{keep_stem}.json" or name == f"{keep_stem}.html" or name == f"{keep_stem}.txt":
                        continue
                    if any(name == f"{keep_stem}{suffix}" for suffix in _DIAGNOSTIC_SIDECAR_SUFFIXES):
                        continue
                if is_dir:
                    if any(name == f"{keep_stem}{suffix}" for suffix in _REPORT_SIBLING_SUFFIXES):
                        continue
            # Files: recognised report stem, diagnostic sidecar, or legacy alias.
            if is_file:
                suffix_lower = Path(name).suffix.lower()
                is_report = _looks_like_report_basename(name) and suffix_lower in _REPORT_EXTENSIONS
                is_diag = any(name.endswith(suffix) for suffix in _DIAGNOSTIC_SIDECAR_SUFFIXES) and _looks_like_report_basename(name)
                is_legacy = name in {LEGACY_JSON_NAME, LEGACY_TEXT_NAME, LEGACY_HTML_NAME}
                if not (is_report or is_diag or is_legacy):
                    continue
            elif is_dir:
                if not _looks_like_report_basename(name):
                    continue
                if not any(name.endswith(suffix) for suffix in _REPORT_SIBLING_SUFFIXES):
                    continue
            else:
                continue
            try:
                # Lazily create the archive dir only when we actually have
                # something to move; empty folders shouldn't pay the syscall.
                if not archive_dir_created:
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    archive_dir_created = True
                destination = archive_dir / name
                if destination.exists():
                    # Collision is rare (same minute, same machine, same mode,
                    # same archive run) but possible after a clock reset; append
                    # an incremental suffix instead of clobbering the older copy.
                    stem = destination.stem
                    ext = destination.suffix
                    idx = 2
                    while destination.exists():
                        destination = archive_dir / f"{stem}__{idx}{ext}"
                        idx += 1
                shutil.move(str(Path(entry.path)), str(destination))
                moved.append(Path(entry.path))
            except Exception:
                # Best-effort housekeeping — never crash a benchmark over an
                # unmovable file (locked HTML in an open browser, antivirus
                # scan window, etc.). The next save will retry.
                continue
    finally:
        # On Windows, os.scandir() iterators can hold file handles; closing
        # the iterator explicitly avoids leaking handles on long-running runs.
        for entry in scan_iter:
            try:
                entry.close() if hasattr(entry, "close") else None
            except Exception:
                pass
    return moved


@dataclass
class RunResult:
    """Result of a single model + method benchmark run."""
    model_id: str
    model_name: str
    method: str                          # ollama_gpu, ollama_cpu, onnx_directml, onnx_cpu, image_comfyui, phase1
    success: bool
    error: Optional[str] = None
    response_text: str = ""
    # Timing (seconds)
    total_time: float = 0.0
    ttft: float = 0.0                    # time to first token
    load_time: float = 0.0
    tokens_per_sec: float = 0.0
    token_count: int = 0
    metric_kind: str = "tokens"             # tokens | utility | image
    metric_label: str = ""
    metric_value: str = ""
    failure_phase: str = ""
    download_time: float = 0.0
    generation_time: float = 0.0
    warm_cache: bool = False
    prompt: str = ""
    options: dict = field(default_factory=dict)
    log_path: str = ""
    # Raw Ollama nanosecond stats (0 for ONNX runs)
    ollama_total_duration_ns: int = 0
    ollama_load_duration_ns: int = 0
    ollama_eval_duration_ns: int = 0
    ollama_eval_count: int = 0
    # System snapshot at time of run
    system_snapshot: dict = field(default_factory=dict)
    timestamp: str = ""
    # Run mode + sample plumbing (Quick uses a single shared prompt; Demo uses
    # one curated showcase sample; Extended iterates model-specific samples
    # and adds image-gen rows on GPU profiles). Defaults preserve back-compat
    # with reports/tests written before these fields existed.
    surface: str = ""                    # chat | onnx | utility | image
    sample_id: str = ""
    sample_title: str = ""
    sample_index: int = 0
    sample_count: int = 0
    prompt_source: str = ""              # e.g. Model-Guide.html / DEFAULT_PROMPT
    # Image-generation artifacts (relative to the report folder)
    image_path: str = ""
    thumbnail_path: str = ""
    negative_prompt: str = ""
    image_width: int = 0
    image_height: int = 0
    image_steps: int = 0
    image_cfg: float = 0.0
    image_sampler: str = ""
    image_scheduler: str = ""
    image_seed: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.surface:
            self.surface = _default_surface_for_method(self.method)


class BatchReport:
    """Accumulates RunResults and writes JSON / HTML reports."""

    def __init__(
        self,
        *,
        start_time: str | None = None,
        machine_info: dict | None = None,
        file_stem: str | None = None,
        run_mode: str | None = None,
        localai_version: str | None = None,
    ):
        self.results: list[RunResult] = []
        self.start_time: str = start_time or time.strftime("%Y-%m-%dT%H:%M:%S")
        self.end_time: str = ""
        self.machine_info: dict = machine_info or collect_machine_info()
        self.run_mode: str = normalize_run_mode(run_mode, default="quick")
        self.localai_version: str = str(localai_version or "").strip() or detect_localai_version()
        self.file_stem: str = file_stem or make_report_stem(
            self.machine_info, self.start_time, run_mode=self.run_mode,
        )
        # Sweep-previous-runs-into-archive is a one-shot per BatchReport
        # instance: only the FIRST incremental save of a run should move
        # leftover files (otherwise every incremental save would inspect
        # the folder unnecessarily). Reset to False during ``load_json``
        # so a "Retry Failed" round preserves the existing archive
        # behaviour for its caller.
        self._archived_previous: bool = False

    def stamp_end_time(self, end_time: str | None = None) -> str:
        """Pin or refresh the run end-time stamp.

        ``BatchRunner.run()`` calls this in its ``finally`` block to lock in the
        final stop time.  Incremental ``save_partial`` calls also refresh it so
        a hung/cancelled run still shows the wall-clock time of the last
        completed result instead of empty text.
        """
        self.end_time = end_time or time.strftime("%Y-%m-%dT%H:%M:%S")
        return self.end_time

    def add(self, result: RunResult) -> None:
        self.results.append(result)

    # ── Load / query / merge ───────────────────────────────────────────────────

    @classmethod
    def load_json(cls, path: Path) -> "BatchReport":
        """Reconstruct a BatchReport from a previously saved JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = cls(
            start_time=data.get("start_time"),
            machine_info=data.get("machine_info") or data.get("system_info") or None,
            file_stem=data.get("file_stem") or path.stem,
            run_mode=data.get("run_mode"),
            localai_version=data.get("localai_version"),
        )
        prior_end = data.get("end_time")
        if prior_end:
            report.end_time = str(prior_end)
        for entry in data.get("results", []):
            report.results.append(RunResult(
                model_id=entry.get("model_id", ""),
                model_name=entry.get("model_name", ""),
                method=entry.get("method", ""),
                success=entry.get("success", False),
                error=entry.get("error"),
                response_text=entry.get("response_text", ""),
                total_time=entry.get("total_time", 0.0),
                ttft=entry.get("ttft", 0.0),
                load_time=entry.get("load_time", 0.0),
                tokens_per_sec=entry.get("tokens_per_sec", 0.0),
                token_count=entry.get("token_count", 0),
                metric_kind=entry.get("metric_kind", "tokens"),
                metric_label=entry.get("metric_label", ""),
                metric_value=entry.get("metric_value", ""),
                failure_phase=entry.get("failure_phase", ""),
                download_time=entry.get("download_time", 0.0),
                generation_time=entry.get("generation_time", 0.0),
                warm_cache=entry.get("warm_cache", False),
                prompt=entry.get("prompt", ""),
                options=entry.get("options", {}),
                log_path=entry.get("log_path", ""),
                ollama_total_duration_ns=entry.get("ollama_total_duration_ns", 0),
                ollama_load_duration_ns=entry.get("ollama_load_duration_ns", 0),
                ollama_eval_duration_ns=entry.get("ollama_eval_duration_ns", 0),
                ollama_eval_count=entry.get("ollama_eval_count", 0),
                system_snapshot=entry.get("system_snapshot", {}),
                timestamp=entry.get("timestamp", ""),
                surface=entry.get("surface", ""),
                sample_id=entry.get("sample_id", ""),
                sample_title=entry.get("sample_title", ""),
                sample_index=entry.get("sample_index", 0),
                sample_count=entry.get("sample_count", 0),
                prompt_source=entry.get("prompt_source", ""),
                image_path=entry.get("image_path", ""),
                thumbnail_path=entry.get("thumbnail_path", ""),
                negative_prompt=entry.get("negative_prompt", ""),
                image_width=entry.get("image_width", 0),
                image_height=entry.get("image_height", 0),
                image_steps=entry.get("image_steps", 0),
                image_cfg=entry.get("image_cfg", 0.0),
                image_sampler=entry.get("image_sampler", ""),
                image_scheduler=entry.get("image_scheduler", ""),
                image_seed=entry.get("image_seed", 0),
            ))
        return report

    def get_failed_combos(self) -> list[tuple[str, str, int]]:
        """Return (model_id, method, sample_index) triples for failed results."""
        failed: list[tuple[str, str, int]] = []
        seen: set[tuple[str, str, int]] = set()
        for result in self.results:
            if result.success:
                continue
            key = (result.model_id, result.method, int(result.sample_index or 0))
            if key in seen:
                continue
            seen.add(key)
            failed.append(key)
        return failed

    def get_completed_combos(self) -> set[tuple[str, str, int]]:
        """Return (model_id, method, sample_index) for every **successful** result.

        v2026.06.01.2+ — used by the Resume Today's Run flow on Start Benchmark to
        compute "skip these — they already passed". Failed combos are
        deliberately NOT included so that Resume re-runs them on the next
        attempt (matches the natural mental model "resume = finish the
        run, including retrying anything that failed"). Adaptively-skipped
        combos are reported as ``success=False`` and so will be re-attempted
        on resume — the smart-skip ceiling carries forward via the
        runner's per-session state, so combos that are still over-budget
        will skip again quickly.

        Earlier (v5.5.7 – v2026.06.01.1) semantics: this returned every
        keyed result including failures, which meant Resume silently
        skipped previously-failed combos. Ron's 2026-06-01 bench report
        ("said yes to retry and it ran nothing") triggered the change.

        See also: ``get_failed_combos`` (Retry Failed button), which
        returns only the failures regardless of whether Resume already
        re-ran them.
        """
        return {
            (r.model_id, r.method, int(r.sample_index or 0))
            for r in self.results
            if r.success
        }

    def compute_time_seconds(self) -> int:
        """Total compute time = sum of per-result total_time across all results.

        v5.5.6+ — honest "time spent benchmarking" figure for the report
        header.  Unlike ``duration_seconds`` (wall-clock end_time − start_time),
        this excludes idle gaps from Resume Today's Run, paused windows, or
        long Ollama pulls that happen between rows but before a result is
        added.  Surfaced as ``compute_time_seconds`` / ``compute_time_hms``
        in :meth:`to_dict` and as the primary timing on the HTML report;
        wall-clock duration is kept as a secondary diagnostic.
        """
        total = 0.0
        for r in self.results:
            try:
                value = float(getattr(r, "total_time", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                total += value
        return int(total)

    def merge(self, retry_report: "BatchReport") -> None:
        """Replace failed results with retry outcomes, in-place."""
        retry_lookup: dict[tuple[str, str, int], RunResult] = {
            (r.model_id, r.method, int(r.sample_index or 0)): r
            for r in retry_report.results
        }
        merged: list[RunResult] = []
        seen: set[tuple[str, str, int]] = set()
        for orig in self.results:
            key = (orig.model_id, orig.method, int(orig.sample_index or 0))
            if key in seen:
                continue
            seen.add(key)
            if key in retry_lookup:
                merged.append(retry_lookup.pop(key))
            else:
                merged.append(orig)
        # Append any combos that weren't in the original (shouldn't happen)
        for leftover in retry_lookup.values():
            merged.append(leftover)
        self.results = merged

    def append_resume_results(self, new_report: "BatchReport") -> None:
        """Merge results from a resumed run, in-place.

        v2026.06.01.2+ semantics — Resume now re-runs failed combos
        (see ``get_completed_combos``), so a resumed run may emit a
        fresh result for a key that already exists in this report.
        Behaviour:

        * **Key already exists AND existing result was a failure** →
          replace with the new result (whether the retry passed or
          failed — freshest data wins).
        * **Key already exists AND existing result was a success** →
          keep the existing (defensive; the resume filter shouldn't
          have re-run a passing combo, but never overwrite a pass
          with a later result).
        * **Key did not exist** → append the new result.

        New end_time and machine_info from the resumed run are stamped onto
        the merged report so the wall-clock window reflects the latest
        activity (compute_time_seconds — the primary metric — naturally
        sums across both runs because per-result total_time is preserved).

        Earlier (v5.5.7 – v2026.06.01.1) semantics: this was a
        strict "fill in the gaps" append that NEVER touched existing
        same-keyed results — paired with the old "completed includes
        failures" rule that meant resume never re-ran failures anyway.
        Both changed together as part of the Resume-retries-failures
        fix.
        """
        existing_by_key: dict[tuple[str, str, int], int] = {
            (r.model_id, r.method, int(r.sample_index or 0)): idx
            for idx, r in enumerate(self.results)
        }
        for r in new_report.results:
            key = (r.model_id, r.method, int(r.sample_index or 0))
            existing_idx = existing_by_key.get(key)
            if existing_idx is None:
                existing_by_key[key] = len(self.results)
                self.results.append(r)
                continue
            if self.results[existing_idx].success:
                continue
            self.results[existing_idx] = r
        if new_report.end_time:
            self.end_time = new_report.end_time

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        end_time = self.end_time or time.strftime("%Y-%m-%dT%H:%M:%S")
        duration_seconds = _duration_seconds(self.start_time, end_time)
        compute_seconds = self.compute_time_seconds()
        return {
            "file_stem": self.file_stem,
            "run_mode": self.run_mode,
            "start_time": self.start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "duration_hms": _format_duration_hms(duration_seconds),
            # v5.5.6+: compute_time = sum(per-result total_time) — the
            # honest "time spent benchmarking" figure that ignores Resume
            # Today's Run idle gaps and long Ollama pulls between rows.
            # The HTML report surfaces this as the primary metric;
            # duration_seconds stays as the secondary wall-clock figure.
            "compute_time_seconds": compute_seconds,
            "compute_time_hms": _format_duration_hms(compute_seconds),
            "machine_info": self.machine_info,
            "localai_version": self.localai_version,
            "total_runs": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "image_runs": self._image_run_count(),
            "images_generated": self._image_generated_count(),
            "results": [asdict(r) for r in self.results],
        }

    def _archive_if_first_save(self, output_dir: Path) -> None:
        """Sweep prior-run files into ``archive/`` once per BatchReport.

        Idempotent: the second and later incremental saves during the same
        run are no-ops. Best-effort: failures (locked file, permission
        error) silently skip rather than crash mid-benchmark.
        """
        if self._archived_previous:
            return
        try:
            archive_previous_runs(output_dir, keep_stem=self.file_stem)
        except Exception:
            pass
        self._archived_previous = True

    def save_json(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._archive_if_first_save(output_dir)
        path = output_dir / f"{self.file_stem}.json"
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Per-mode latest alias so quick and extended runs do not
        # clobber each other's "open the most recent of my kind" path.
        # The legacy ``batch_results.json`` alias was dropped in v5.5.1
        # because it produced a stale third copy in every results folder;
        # ``archive_previous_runs`` sweeps any leftover legacy file into
        # ``archive\`` on the next save.
        per_mode_alias = _per_mode_latest_json_name(self.run_mode)
        if path.name != per_mode_alias:
            with open(output_dir / per_mode_alias, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return path

    @staticmethod
    def _metric_text(result: RunResult) -> str:
        if result.metric_kind == "utility" or result.method == "phase1":
            if result.metric_value:
                return result.metric_value
            return f"{result.total_time:.1f}s"
        return f"{result.tokens_per_sec:.1f} tok/s"

    @staticmethod
    def _ttft_text(result: RunResult) -> str:
        if result.metric_kind == "utility" or result.method == "phase1":
            return "-"
        return f"{result.ttft:.2f}s"

    @staticmethod
    def _tokens_text(result: RunResult) -> str:
        if result.metric_kind == "utility" or result.method == "phase1":
            return "-"
        return str(result.token_count)

    def save_text(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._archive_if_first_save(output_dir)
        path = output_dir / f"{self.file_stem}.txt"
        machine = self.machine_info
        machine_line = (
            f"  Machine: {machine.get('machine_name', 'unknown')}"
            + (f" ({machine.get('machine_model')})" if machine.get("machine_model") else "")
        )
        specs_line = (
            f"  Specs: {machine.get('cpu') or machine.get('vcpu', 0)} CPU cores | "
            f"{machine.get('ram_gb', 0)} GB RAM | "
            f"{machine.get('gpu_name') or 'No GPU'} | "
            f"{machine.get('vram_gb', 0)} GB VRAM"
        )

        lines = [
            "=" * 90,
            "  BATCH BENCHMARK REPORT",
            f"  Report file: {self.file_stem}",
            f"  LocalAI Studio: {self.localai_version or UNKNOWN_LOCALAI_VERSION}",
            f"  Started: {self.start_time}",
            f"  Finished: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            machine_line,
            specs_line,
            f"  OS: {machine.get('os', '')}",
            f"  Python: {machine.get('python', '')}",
            f"  Storage: {machine.get('storage_free_gb', 0)} GB free of {machine.get('storage_total_gb', 0)} GB",
            f"  Total runs: {len(self.results)}  |  "
            f"Pass: {sum(1 for r in self.results if r.success)}  |  "
            f"Fail: {sum(1 for r in self.results if not r.success)}",
            f"  Images generated: {self._image_generated_count()} / {self._image_run_count()} image runs",
            "=" * 90,
            "",
            f"{'Model':<28} {'Method':<16} {'Status':<8} "
            f"{'Metric':>16} {'TTFT':>8} {'Total':>8} {'Tokens':>7}",
            "-" * 90,
        ]

        for r in self.results:
            status = "OK" if r.success else "FAIL"
            metric = self._metric_text(r) if r.success else "-"
            ttft = self._ttft_text(r) if r.success else "-"
            total = f"{r.total_time:.1f}s" if r.success else "-"
            toks = self._tokens_text(r) if r.success else "-"
            lines.append(
                f"{r.model_name:<28} {display_method(r.method):<16} {status:<8} "
                f"{metric:>16} {ttft:>8} {total:>8} {toks:>7}"
            )
            if not r.success and r.error:
                phase = f" [{r.failure_phase}]" if r.failure_phase else ""
                lines.append(f"  ERROR{phase}: {r.error[:70]}")
            if r.log_path:
                lines.append(f"  LOG: {r.log_path}")

        lines.append("-" * 90)
        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # v5.5.1+: legacy ``batch_results.txt`` alias dropped; it accumulated
        # as the third stale copy in every benchmark folder. Leftover legacy
        # files are swept into ``archive\`` on the next save.
        return path

    def save_html(self, output_dir: Path) -> Path:
        """Write the LocalAI cool-blue HTML report alongside the JSON.

        Includes a per-mode ``latest_*_benchmark.html`` alias so users can
        always click the "latest" report without needing to know the
        timestamped stem. The pre-v5.5.1 ``batch_results.html`` legacy
        alias was dropped — it created a third stale copy in every results
        folder; older copies are swept into ``archive/`` on the next save.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self._archive_if_first_save(output_dir)
        path = output_dir / f"{self.file_stem}.html"
        html_text = self._render_html()
        path.write_text(html_text, encoding="utf-8")

        per_mode_alias = _per_mode_latest_html_name(self.run_mode)
        if path.name != per_mode_alias:
            (output_dir / per_mode_alias).write_text(html_text, encoding="utf-8")
        return path

    def print_summary(self) -> None:
        """Print a compact summary table to stdout."""
        print()
        print("=" * 90)
        print("  BATCH BENCHMARK SUMMARY")
        print("=" * 90)
        print(
            f"{'Model':<28} {'Method':<16} {'Status':<8} "
            f"{'Metric':>16} {'TTFT':>8} {'Total':>8} {'Tokens':>7}"
        )
        print("-" * 90)
        for r in self.results:
            status = "OK" if r.success else "FAIL"
            metric = self._metric_text(r) if r.success else "-"
            ttft = self._ttft_text(r) if r.success else "-"
            total = f"{r.total_time:.1f}s" if r.success else "-"
            toks = self._tokens_text(r) if r.success else "-"
            print(
                f"{r.model_name:<28} {display_method(r.method):<16} {status:<8} "
                f"{metric:>16} {ttft:>8} {total:>8} {toks:>7}"
            )
        print("-" * 90)
        ok = sum(1 for r in self.results if r.success)
        fail = sum(1 for r in self.results if not r.success)
        images_generated = self._image_generated_count()
        image_runs = self._image_run_count()
        print(f"  Total: {len(self.results)}  |  Pass: {ok}  |  Fail: {fail}")
        print(f"  LocalAI Studio: {self.localai_version or UNKNOWN_LOCALAI_VERSION}")
        print(f"  Images generated: {images_generated} / {image_runs} image runs")
        print("=" * 90)
        print()

    # ── HTML rendering ─────────────────────────────────────────────────────────

    def _render_html(self) -> str:
        machine = self.machine_info or {}
        run_mode_label = _run_mode_label(self.run_mode)
        machine_name = machine.get("machine_name") or "unknown"
        gpu = machine.get("gpu_name") or ("Unified GPU" if machine.get("gpu_unified_memory") else "No GPU")
        cpu = machine.get("cpu") or machine.get("vcpu", 0) or 0
        ram = machine.get("ram_gb", 0) or 0
        vram = machine.get("vram_gb", 0) or 0
        finished = time.strftime("%Y-%m-%d %H:%M")
        localai_version = self.localai_version or UNKNOWN_LOCALAI_VERSION
        end_time = self.end_time or time.strftime("%Y-%m-%dT%H:%M:%S")
        duration_seconds = _duration_seconds(self.start_time, end_time)
        duration_hms = _format_duration_hms(duration_seconds)
        compute_seconds = self.compute_time_seconds()
        compute_hms = _format_duration_hms(compute_seconds)
        subtitle = (
            f"{machine_name} · {cpu} CPU cores · {ram} GB RAM · {gpu}"
            + (f" · {vram} GB VRAM" if vram else "")
            + f" · LocalAI Studio {localai_version} · generated {finished}. "
            + {
                "extended": "Extended runs use per-model sample prompts and add image generation on GPU-capable profiles.",
            }.get(self.run_mode, "Quick runs use one shared prompt across selected text/chat models.")
        )

        cards_html = "\n".join(self._render_card(r) for r in self.results)
        counts = self._surface_counts()
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        models = len({r.model_id for r in self.results})
        image_count = self._image_run_count()
        images_generated = self._image_generated_count()
        median_tps = self._median_text_tokens_per_sec()
        tps_label = f"{median_tps:.1f} tok/s" if median_tps else "—"
        result_table_html = self._render_result_summary_table()

        json_filename = f"{self.file_stem}.json"

        replacements = {
            "__TITLE__": html.escape(f"LocalAI Benchmark · {run_mode_label} · {self.file_stem}"),
            "__RUN_MODE_LABEL__": html.escape(run_mode_label),
            "__EYEBROW__": html.escape(f"Benchmark report · {run_mode_label} run · {self.file_stem}"),
            "__SUBTITLE__": html.escape(subtitle),
            "__TOTAL__": str(total),
            "__PASSED__": str(passed),
            "__FAILED__": str(failed),
            "__MODELS__": str(models),
            "__IMAGES__": str(images_generated),
            "__LOCALAI_VERSION__": html.escape(localai_version),
            "__MEDIAN_TPS__": html.escape(tps_label),
            "__START_TIME__": html.escape(self.start_time or "—"),
            "__END_TIME__": html.escape(end_time or "—"),
            "__DURATION_HMS__": html.escape(duration_hms),
            "__COMPUTE_HMS__": html.escape(compute_hms),
            "__COUNT_ALL__": str(total),
            "__COUNT_TEXT__": str(counts.get("chat", 0) + counts.get("onnx", 0)),
            "__COUNT_IMAGE__": str(image_count),
            "__COUNT_FAILED__": str(failed),
            "__CARDS__": cards_html or '<p class="empty">No results recorded yet.</p>',
            "__RESULTS_TABLE__": result_table_html,
            "__JSON_FILENAME__": html.escape(json_filename),
        }
        rendered = _HTML_TEMPLATE
        for key, value in replacements.items():
            rendered = rendered.replace(key, value)
        return rendered

    @staticmethod
    def _is_image_result(result: RunResult) -> bool:
        surface = result.surface or _default_surface_for_method(result.method)
        return surface == "image" or result.method == "image_comfyui"

    def _image_run_count(self) -> int:
        return sum(1 for r in self.results if self._is_image_result(r))

    def _image_generated_count(self) -> int:
        return sum(1 for r in self.results if self._is_image_result(r) and r.success)

    def _surface_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            surface = r.surface or _default_surface_for_method(r.method)
            out[surface] = out.get(surface, 0) + 1
        return out

    def _median_text_tokens_per_sec(self) -> float:
        speeds = [
            r.tokens_per_sec for r in self.results
            if r.success and r.tokens_per_sec > 0
            and (r.surface in {"chat", "onnx"} or r.method.startswith(("ollama_", "onnx_")))
        ]
        if not speeds:
            return 0.0
        speeds = sorted(speeds)
        mid = len(speeds) // 2
        if len(speeds) % 2 == 1:
            return float(speeds[mid])
        return float((speeds[mid - 1] + speeds[mid]) / 2.0)

    def _render_result_summary_table(self) -> str:
        if not self.results:
            return '<p class="empty">No result rows recorded.</p>'
        rows: list[str] = []
        for index, r in enumerate(self.results, start=1):
            sample = ""
            if r.sample_count and r.sample_count > 1:
                sample = f"sample {r.sample_index + 1 if r.sample_index >= 0 else 1}/{r.sample_count}"
            elif r.sample_title:
                sample = r.sample_title
            method = display_method(r.method)
            context = method + (f" · {sample}" if sample else "")
            status = "OK" if r.success else "FAIL"
            tps_value = r.tokens_per_sec if r.success and r.tokens_per_sec > 0 else None
            tps_display = f"{tps_value:.1f}" if tps_value is not None else "—"
            tps_sort = f"{tps_value:.6f}" if tps_value is not None else ""
            raw_value = float(r.total_time) if r.total_time else 0.0
            raw_display = f"{raw_value:.2f}"
            raw_sort = f"{raw_value:.6f}"
            status_sort = "0" if r.success else "1"
            model_label = r.model_name or r.model_id
            rows.append(
                "<tr>"
                f'<td class="row-num" data-sort="{index}">{index}</td>'
                f'<td data-sort="{html.escape(model_label.lower(), quote=True)}">'
                f'{_model_guide_anchor(r.model_id, html.escape(model_label))}</td>'
                f'<td data-sort="{html.escape(context.lower(), quote=True)}">{html.escape(context)}</td>'
                f'<td data-sort="{status_sort}">'
                f'<span class="status-pill {status.lower()}">{status}</span></td>'
                f'<td class="metric-num" data-sort="{tps_sort}">{html.escape(tps_display)}</td>'
                f'<td class="metric-num" data-sort="{raw_sort}">{html.escape(raw_display)}</td>'
                "</tr>"
            )
        return (
            '<table class="result-summary-grid" id="result-summary-grid">'
            "<thead><tr>"
            '<th scope="col" role="button" tabindex="0" aria-sort="none" '
            'data-sort-key="index" data-sort-mode="numeric">'
            '#<span class="sort-indicator" aria-hidden="true"></span></th>'
            '<th scope="col" role="button" tabindex="0" aria-sort="none" '
            'data-sort-key="model" data-sort-mode="text">'
            'Model<span class="sort-indicator" aria-hidden="true"></span></th>'
            '<th scope="col" role="button" tabindex="0" aria-sort="none" '
            'data-sort-key="method" data-sort-mode="text">'
            'Method / sample<span class="sort-indicator" aria-hidden="true"></span></th>'
            '<th scope="col" role="button" tabindex="0" aria-sort="none" '
            'data-sort-key="status" data-sort-mode="enum">'
            'Status<span class="sort-indicator" aria-hidden="true"></span></th>'
            '<th scope="col" role="button" tabindex="0" aria-sort="none" '
            'data-sort-key="tps" data-sort-mode="numeric">'
            'Tok/s<span class="sort-indicator" aria-hidden="true"></span></th>'
            '<th scope="col" role="button" tabindex="0" aria-sort="none" '
            'data-sort-key="raw" data-sort-mode="numeric">'
            'Raw seconds<span class="sort-indicator" aria-hidden="true"></span></th>'
            "</tr></thead>"
            "<tbody>"
            + "\n".join(rows)
            + "</tbody></table>"
        )

    def _render_card(self, r: RunResult) -> str:
        surface = r.surface or _default_surface_for_method(r.method)
        status = "passed" if r.success else "failed"
        method_id = r.method or "unknown"
        method_label = display_method(method_id)
        surface_label = {
            "chat": "Chat",
            "onnx": "ONNX",
            "utility": "Utility",
            "image": "Image generation",
        }.get(surface, surface.title() if surface else "Run")
        sample_part = ""
        if r.sample_count and r.sample_count > 1:
            idx = r.sample_index + 1 if r.sample_index >= 0 else 1
            sample_part = f" · sample {idx}/{r.sample_count}"
        elif r.sample_title:
            sample_part = f" · {html.escape(r.sample_title)}"
        eyebrow = f"{surface_label} · {method_label}{sample_part}"

        title = _model_guide_anchor(r.model_id, html.escape(r.model_name or r.model_id))
        if r.sample_title and r.sample_title != r.model_name:
            title += f" — {html.escape(r.sample_title)}"

        meta_parts = []
        if r.prompt_source:
            meta_parts.append(_render_prompt_source(r.prompt_source, r.model_id))
        if r.warm_cache:
            meta_parts.append("warm cache")
        elif r.method.startswith("ollama_"):
            meta_parts.append("cold cache")
        if r.failure_phase and not r.success:
            meta_parts.append(f"failure phase: <code>{html.escape(r.failure_phase)}</code>")
        meta = " · ".join(meta_parts)

        badges = self._render_badges(r)
        body_html = self._render_card_body(r, surface)

        return (
            f'<article class="card" data-surface="{html.escape(surface)}" '
            f'data-status="{status}" data-method="{html.escape(method_id)}" '
            f'data-model="{html.escape(r.model_id)}">'
            f'<div class="card-head">'
            f'<div><span class="eyebrow">{eyebrow}</span>'
            f'<h2>{title}</h2>'
            + (f'<p class="meta">{meta}</p>' if meta else "")
            + f'</div><div class="badges">{badges}</div></div>'
            + body_html
            + "</article>"
        )

    def _render_badges(self, r: RunResult) -> str:
        parts: list[str] = []
        if r.success:
            parts.append('<span class="badge ok">Passed</span>')
        else:
            parts.append('<span class="badge fail">Failed</span>')

        if r.surface == "image" or r.method == "image_comfyui":
            if r.image_width and r.image_height:
                parts.append(f'<span class="badge">{r.image_width} × {r.image_height}</span>')
            if r.image_steps:
                parts.append(f'<span class="badge">{r.image_steps} steps</span>')
            if r.total_time:
                parts.append(f'<span class="badge">{r.total_time:.1f}s</span>')
        elif r.surface == "utility" or r.method == "phase1":
            parts.append('<span class="badge">Utility</span>')
            if r.metric_value:
                parts.append(f'<span class="badge">{html.escape(r.metric_value)}</span>')
            elif r.total_time:
                parts.append(f'<span class="badge">{r.total_time:.1f}s</span>')
        else:
            if r.total_time:
                parts.append(f'<span class="badge">{r.total_time:.1f}s total</span>')
            if r.tokens_per_sec:
                parts.append(f'<span class="badge">{r.tokens_per_sec:.1f} tok/s</span>')
            if r.ttft:
                parts.append(f'<span class="badge">TTFT {r.ttft:.2f}s</span>')
            if r.failure_phase and not r.success:
                parts.append(f'<span class="badge warn">{html.escape(r.failure_phase)}</span>')
        return "".join(parts)

    def _render_card_body(self, r: RunResult, surface: str) -> str:
        prompt = html.escape(r.prompt or "(no prompt recorded)")
        if surface == "image" or r.method == "image_comfyui":
            return self._render_image_card_body(r)
        if r.success:
            response = html.escape(r.response_text or "(no response captured)")
            response_label = "Response"
            if surface == "utility":
                response_label = "Result"
            return (
                '<div class="grid">'
                f'<section class="panel"><h3>Prompt</h3><pre>{prompt}</pre></section>'
                f'<section class="panel"><h3>{response_label}</h3><pre>{response}</pre></section>'
                "</div>"
            )
        error_text = html.escape(r.error or "Run failed without an error message.")
        if r.response_text:
            error_text += "\n\nPartial output:\n" + html.escape(r.response_text)
        return (
            '<div class="grid">'
            f'<section class="panel"><h3>Prompt</h3><pre>{prompt}</pre></section>'
            f'<section class="panel"><h3>Error</h3><pre>{error_text}</pre></section>'
            "</div>"
        )

    def _render_image_card_body(self, r: RunResult) -> str:
        prompt = html.escape(r.prompt or "(no prompt recorded)")
        if r.success and r.image_path:
            thumb = html.escape(r.thumbnail_path or r.image_path)
            full = html.escape(r.image_path)
            settings_lines = []
            if r.image_width and r.image_height:
                settings_lines.append(f"Size: {r.image_width} × {r.image_height}")
            if r.image_sampler or r.image_scheduler:
                settings_lines.append(
                    "Sampler/Scheduler: "
                    + " / ".join(p for p in [r.image_sampler, r.image_scheduler] if p)
                )
            if r.image_steps:
                settings_lines.append(f"Steps: {r.image_steps}")
            if r.image_cfg:
                settings_lines.append(f"CFG: {r.image_cfg:g}")
            if r.image_seed:
                settings_lines.append(f"Seed: {r.image_seed}")
            negative = (
                html.escape(r.negative_prompt)
                if r.negative_prompt
                else "(ignored at CFG=1.0 for Flux/Z-Image/Chroma/Turbo/Lightning families.)"
            )
            settings_html = html.escape("\n".join(settings_lines)) if settings_lines else "(settings not recorded)"
            return (
                '<div class="image-grid">'
                '<section class="panel"><h3>Generated image</h3>'
                f'<a href="{full}" data-full title="Open full-size image">'
                f'<img class="thumb" alt="Generated benchmark image" src="{thumb}"></a>'
                '<p class="thumb-hint">Click the image to open it full-size.</p></section>'
                '<section class="panel"><h3>Prompt and settings</h3>'
                f'<pre>Prompt:\n{prompt}\n\nNegative prompt:\n{negative}\n\nSettings:\n{settings_html}</pre>'
                "</section></div>"
            )
        error_text = html.escape(r.error or "Image generation failed without an error message.")
        return (
            '<div class="grid">'
            f'<section class="panel"><h3>Prompt</h3><pre>{prompt}</pre></section>'
            f'<section class="panel"><h3>Error</h3><pre>{error_text}</pre></section>'
            "</div>"
        )


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script>
(() => {
  const param = new URLSearchParams(window.location.search).get('clawpilotTheme');
  if (param === 'light' || param === 'dark') {
    document.documentElement.setAttribute('data-theme', param);
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();
</script>
<style>
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg:       #1a1a2e;
  --surface:  #16213e;
  --surface-2:#1c2a4a;
  --card:     #0f3460;
  --card-alt: #12294f;
  --accent:   #4f9cf9;
  --accent-2: #7ad7ff;
  --accent-soft: rgba(79,156,249,.14);
  --accent-fg:#0d1117;
  --good:     #56d364;
  --warn:     #ffb347;
  --bad:      #ff7b72;
  --text:     #e7edf5;
  --text-soft:#c9d1d9;
  --muted:    #9aa7bb;
  --border:   #2a3b58;
  --border-strong:#3d5278;
  --code-bg:  #0c1320;
  --shadow:   0 18px 48px rgba(0,0,0,.42);
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg:       #f1f4fb;
  --surface:  #ffffff;
  --surface-2:#eaf0fa;
  --card:     #ffffff;
  --card-alt: #f3f7fd;
  --accent:   #1864c4;
  --accent-2: #2a7bd9;
  --accent-soft: rgba(24,100,196,.10);
  --accent-fg:#ffffff;
  --good:     #1f8a3a;
  --warn:     #b76b00;
  --bad:      #b91c1c;
  --text:     #121b2e;
  --text-soft:#1f2c46;
  --muted:    #4f5e7a;
  --border:   #d2dcef;
  --border-strong:#a6b6d3;
  --code-bg:  #f4f7fd;
  --shadow:   0 18px 48px rgba(20,40,80,.12);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); }
body { font-family: "Segoe UI", Aptos, Calibri, system-ui, sans-serif; line-height: 1.55; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font-family: Consolas, "Cascadia Code", "Cascadia Mono", monospace; }
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-x: auto;
  margin: 0;
  color: var(--text-soft);
  font-size: 0.92rem;
}
.header {
  background: linear-gradient(135deg, var(--surface), var(--surface-2));
  border-bottom: 1px solid var(--border);
  padding: 28px 40px 24px;
  box-shadow: var(--shadow);
}
.header-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}
.eyebrow {
  display: inline-block;
  background: var(--accent-soft);
  color: var(--accent-2);
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-weight: 600;
  margin-bottom: 8px;
}
h1 { font-size: 1.9rem; margin: 0 0 8px; letter-spacing: -0.01em; }
.subtitle { color: var(--muted); margin: 0; max-width: 80ch; }
.toolbar {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}
.toolbar button, .toolbar a.btn {
  background: var(--card);
  color: var(--text);
  border: 1px solid var(--border-strong);
  padding: 8px 14px;
  border-radius: 8px;
  font: inherit;
  cursor: pointer;
  transition: background 120ms ease, border-color 120ms ease;
}
.toolbar button:hover, .toolbar a.btn:hover {
  background: var(--card-alt);
  border-color: var(--accent);
}
.toolbar .primary {
  background: var(--accent);
  color: var(--accent-fg);
  border-color: var(--accent);
}
.toolbar .primary:hover {
  background: var(--accent-2);
  border-color: var(--accent-2);
}
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 20px;
}
.stat {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
}
.stat .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
.stat .value { font-size: 1.5rem; font-weight: 600; margin-top: 4px; color: var(--text); }
.stat.passed .value { color: var(--good); }
.stat.failed .value { color: var(--bad); }
.run-timing {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.run-timing .stat .value {
  font-size: 1.1rem;
  font-variant-numeric: tabular-nums;
}
.run-timing .stat.duration .value { font-size: 1.4rem; color: var(--accent); }
.page-shell {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 22px;
  max-width: 1600px;
  margin: 0 auto;
  padding: 28px 40px 60px;
}
.side-rail {
  position: sticky;
  top: 16px;
  align-self: start;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 14px;
}
.side-rail .rail-title {
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.side-rail a {
  display: block;
  color: var(--text-soft);
  border-radius: 8px;
  padding: 7px 8px;
  font-size: 0.9rem;
}
.side-rail a:hover {
  background: var(--accent-soft);
  color: var(--accent-2);
  text-decoration: none;
}
main { min-width: 0; }
.tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.tab {
  background: transparent;
  border: none;
  color: var(--muted);
  font: inherit;
  padding: 10px 16px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-weight: 500;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent-2); border-bottom-color: var(--accent); }
.tab .count {
  margin-left: 6px;
  background: var(--surface-2);
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 0.75rem;
  color: var(--muted);
}
.filters {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 18px;
}
.filters input[type="search"], .filters select {
  background: var(--card);
  color: var(--text);
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  padding: 6px 10px;
  font: inherit;
}
.filters input[type="search"] { min-width: 220px; }
.filters label { color: var(--muted); font-size: 0.9rem; }
.cards { display: grid; gap: 18px; }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: var(--shadow);
}
.card.hidden { display: none; }
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  flex-wrap: wrap;
}
.card h2 { margin: 0; font-size: 1.15rem; color: var(--text); }
.card h2 a { color: inherit; }
.card h2 a:hover { color: var(--accent); }
.card .meta { margin: 6px 0 0; color: var(--muted); font-size: 0.85rem; }
.card .meta a { font-weight: 700; }
.badges { display: flex; gap: 6px; flex-wrap: wrap; }
.badge {
  display: inline-block;
  background: var(--surface-2);
  color: var(--text-soft);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.78rem;
}
.badge.ok { background: rgba(86,211,100,.16); color: var(--good); border-color: rgba(86,211,100,.3); }
.badge.fail { background: rgba(255,123,114,.16); color: var(--bad); border-color: rgba(255,123,114,.3); }
.badge.warn { background: rgba(255,179,71,.16); color: var(--warn); border-color: rgba(255,179,71,.3); }
.grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 14px;
  margin-top: 14px;
}
.image-grid {
  display: grid;
  grid-template-columns: minmax(280px, 1fr) minmax(0, 1fr);
  gap: 14px;
  margin-top: 14px;
}
.panel { background: var(--card-alt); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.panel h3 { margin: 0 0 8px; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
.thumb {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid var(--border-strong);
  cursor: zoom-in;
}
.thumb-hint { color: var(--muted); font-size: 0.8rem; margin: 6px 0 0; }
.empty { color: var(--muted); text-align: center; padding: 40px 0; }
.result-table-section {
  margin-top: 24px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 18px 20px;
}
.section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.section-head h2 { margin: 0; font-size: 1.1rem; }
.section-head p { margin: 0; color: var(--muted); font-size: 0.9rem; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
.result-summary-grid {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}
.result-summary-grid th,
.result-summary-grid td {
  padding: 7px 9px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  white-space: nowrap;
}
.result-summary-grid th {
  color: var(--muted);
  background: var(--card-alt);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.72rem;
}
.result-summary-grid th[role="button"] {
  cursor: pointer;
  user-select: none;
  position: relative;
}
.result-summary-grid th[role="button"]:hover,
.result-summary-grid th[role="button"]:focus { color: var(--text); outline: none; }
.result-summary-grid th[role="button"]:focus-visible {
  box-shadow: inset 0 0 0 2px var(--accent);
}
.result-summary-grid th .sort-indicator {
  display: inline-block;
  min-width: 12px;
  margin-left: 6px;
  font-size: 0.7rem;
  color: var(--muted);
  font-weight: 700;
}
.result-summary-grid th[aria-sort="ascending"] .sort-indicator::before { content: "▲"; color: var(--accent); }
.result-summary-grid th[aria-sort="descending"] .sort-indicator::before { content: "▼"; color: var(--accent); }
.result-summary-grid th[aria-sort="ascending"],
.result-summary-grid th[aria-sort="descending"] { color: var(--text); }
.result-summary-grid tbody tr:hover { background: var(--accent-soft); }
.result-summary-grid tbody tr:last-child td { border-bottom: 0; }
.row-num { color: var(--muted); width: 1%; }
.metric-num { text-align: right !important; font-variant-numeric: tabular-nums; }
.status-pill {
  display: inline-block;
  min-width: 42px;
  text-align: center;
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 0.72rem;
  font-weight: 700;
}
.status-pill.ok { color: var(--good); background: rgba(86,211,100,.12); }
.status-pill.fail { color: var(--bad); background: rgba(255,123,114,.14); }
.modal-bg {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,.78);
  display: none;
  align-items: center;
  justify-content: center;
  padding: 30px;
  z-index: 100;
  cursor: zoom-out;
}
.modal-bg.open { display: flex; }
.modal-bg img { max-width: 100%; max-height: 100%; border-radius: 12px; border: 1px solid var(--border-strong); }
@media (max-width: 900px) {
  .grid, .image-grid { grid-template-columns: 1fr; }
  .header { padding: 22px 22px 18px; }
  .page-shell { display: block; padding: 22px 22px 60px; }
  .side-rail { position: static; margin-bottom: 18px; }
}
</style>
</head>
<body>
<header class="header" id="overview">
  <div class="header-row">
    <div>
      <span class="eyebrow">__EYEBROW__</span>
      <h1>LocalAI benchmark report · __RUN_MODE_LABEL__</h1>
      <p class="subtitle">__SUBTITLE__</p>
    </div>
    <div class="toolbar">
      <a class="btn" href="__JSON_FILENAME__" download>Download JSON</a>
      <button type="button" class="primary" id="theme-toggle">Toggle theme</button>
    </div>
  </div>
  <div class="summary">
    <div class="stat"><div class="label">Total runs</div><div class="value">__TOTAL__</div></div>
    <div class="stat passed"><div class="label">Passed</div><div class="value">__PASSED__</div></div>
    <div class="stat failed"><div class="label">Failed</div><div class="value">__FAILED__</div></div>
    <div class="stat"><div class="label">Distinct models</div><div class="value">__MODELS__</div></div>
    <div class="stat"><div class="label">Images generated</div><div class="value">__IMAGES__</div></div>
    <div class="stat"><div class="label">LocalAI Studio</div><div class="value">__LOCALAI_VERSION__</div></div>
    <div class="stat"><div class="label">Median tok/s (text)</div><div class="value">__MEDIAN_TPS__</div></div>
  </div>
  <div class="run-timing" id="run-timing">
    <div class="stat"><div class="label">Started</div><div class="value">__START_TIME__</div></div>
    <div class="stat"><div class="label">Ended</div><div class="value">__END_TIME__</div></div>
    <div class="stat duration"><div class="label">Compute time (H:MM:SS)</div><div class="value">__COMPUTE_HMS__</div></div>
    <div class="stat"><div class="label">Wall clock (incl. gaps)</div><div class="value">__DURATION_HMS__</div></div>
  </div>
</header>
<div class="page-shell">
  <aside class="side-rail" aria-label="Report navigation">
    <div class="rail-title">Jump to</div>
    <a href="#overview">Overview</a>
    <a href="#run-timing">Run timing</a>
    <a href="#surface-tabs">Filters</a>
    <a href="#cards">Detailed cards</a>
    <a href="#result-summary-table">Summary table</a>
  </aside>
  <main>
    <nav class="tabs" id="surface-tabs">
      <button type="button" class="tab active" data-surface="all">All <span class="count">__COUNT_ALL__</span></button>
      <button type="button" class="tab" data-surface="text">Text / chat <span class="count">__COUNT_TEXT__</span></button>
      <button type="button" class="tab" data-surface="image">Image <span class="count">__COUNT_IMAGE__</span></button>
      <button type="button" class="tab" data-surface="failed">Failed <span class="count">__COUNT_FAILED__</span></button>
    </nav>
    <div class="filters">
      <input type="search" id="filter-text" placeholder="Filter by model, sample, or text...">
      <label for="filter-status">Status:</label>
      <select id="filter-status">
        <option value="any">Any</option>
        <option value="passed">Passed</option>
        <option value="failed">Failed</option>
      </select>
    </div>
    <section class="cards" id="cards">
      __CARDS__
    </section>
    <section class="result-table-section" id="result-summary-table">
      <div class="section-head">
        <h2>Summary table</h2>
        <p>Compact model performance grid: tokens/sec where available and raw wall time for every run.</p>
      </div>
      <div class="table-wrap">
        __RESULTS_TABLE__
      </div>
    </section>
  </main>
</div>
<div class="modal-bg" id="modal"><img alt="full-size benchmark image" id="modal-img"></div>
<script>
(() => {
  const tabs = document.querySelectorAll('#surface-tabs .tab');
  const cards = document.querySelectorAll('.card');
  const search = document.getElementById('filter-text');
  const statusSel = document.getElementById('filter-status');
  let activeSurface = 'all';

  function matchesSurface(card) {
    if (activeSurface === 'all') return true;
    if (activeSurface === 'failed') return card.dataset.status === 'failed';
    if (activeSurface === 'text') return ['chat','onnx'].includes(card.dataset.surface);
    return card.dataset.surface === activeSurface;
  }

  function apply() {
    const needle = (search.value || '').trim().toLowerCase();
    const wantedStatus = statusSel.value;
    cards.forEach(card => {
      const text = card.innerText.toLowerCase();
      const surfaceOk = matchesSurface(card);
      const statusOk = wantedStatus === 'any' || card.dataset.status === wantedStatus;
      const textOk = !needle || text.includes(needle);
      card.classList.toggle('hidden', !(surfaceOk && statusOk && textOk));
    });
  }

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      activeSurface = tab.dataset.surface || 'all';
      apply();
    });
  });
  search.addEventListener('input', apply);
  statusSel.addEventListener('change', apply);

  document.getElementById('theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
  });

  // ── Sortable summary table ────────────────────────────────────────────────
  const summaryTable = document.getElementById('result-summary-grid');
  if (summaryTable) {
    const headers = summaryTable.querySelectorAll('thead th[role="button"]');
    const tbody = summaryTable.querySelector('tbody');
    let activeKey = null;
    let activeDir = 'asc';

    function cellSortValue(row, columnIndex, mode) {
      const cell = row.children[columnIndex];
      if (!cell) return mode === 'numeric' ? null : '';
      const raw = cell.getAttribute('data-sort');
      if (mode === 'numeric') {
        if (raw === null || raw === '') return null;
        const n = parseFloat(raw);
        return isNaN(n) ? null : n;
      }
      return (raw ?? cell.textContent ?? '').toString();
    }

    function sortBy(th) {
      const key = th.getAttribute('data-sort-key');
      const mode = th.getAttribute('data-sort-mode') || 'text';
      const columnIndex = Array.prototype.indexOf.call(th.parentNode.children, th);
      if (activeKey === key) {
        activeDir = activeDir === 'asc' ? 'desc' : 'asc';
      } else {
        activeKey = key;
        activeDir = 'asc';
      }
      headers.forEach(h => h.setAttribute('aria-sort', 'none'));
      th.setAttribute('aria-sort', activeDir === 'asc' ? 'ascending' : 'descending');

      const rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
      const dir = activeDir === 'asc' ? 1 : -1;
      rows.sort((a, b) => {
        const va = cellSortValue(a, columnIndex, mode);
        const vb = cellSortValue(b, columnIndex, mode);
        // Empty / null values always sort to the bottom regardless of direction.
        const aEmpty = (va === null || va === '');
        const bEmpty = (vb === null || vb === '');
        if (aEmpty && bEmpty) return 0;
        if (aEmpty) return 1;
        if (bEmpty) return -1;
        if (mode === 'numeric') {
          return (va - vb) * dir;
        }
        return va.toString().localeCompare(vb.toString(), undefined, {numeric: true, sensitivity: 'base'}) * dir;
      });
      const frag = document.createDocumentFragment();
      rows.forEach(r => frag.appendChild(r));
      tbody.appendChild(frag);
    }

    headers.forEach(th => {
      th.addEventListener('click', () => sortBy(th));
      th.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          sortBy(th);
        }
      });
    });
  }

  const modal = document.getElementById('modal');
  const modalImg = document.getElementById('modal-img');
  document.querySelectorAll('a[data-full]').forEach(link => {
    link.addEventListener('click', (ev) => {
      ev.preventDefault();
      modalImg.src = link.getAttribute('href');
      modal.classList.add('open');
    });
  });
  modal.addEventListener('click', () => modal.classList.remove('open'));
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') modal.classList.remove('open');
  });
})();
</script>
</body>
</html>
"""

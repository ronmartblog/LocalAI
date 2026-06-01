# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Batch benchmark runner: orchestrates download / run / timing / cleanup
for every model + method combination in the catalog.
"""

import signal
import sys
import threading
import time
import contextlib
import inspect
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Optional

from src.catalog import MODELS, get_model_by_id, load_catalog
from src import constrained_env, logger
from src.gpu_detect import is_snapdragon_arm64
from src.ollama_client import OllamaClient, OllamaError, ollama_tag_is_local, strip_think_blocks
from src.onnx_client import (
    ONNX_AVAILABLE, DIRECTML_AVAILABLE, OPENVINO_AVAILABLE, HF_AVAILABLE,
    GENAI_AVAILABLE,
    OnnxModelSession, OnnxGenAISession, OnnxError,
    download_onnx_model, _pick_provider, has_genai_config,
)
from src.system_info import get_system_summary, get_gpu_info, get_ram_info
from src.batch_report import RunResult, BatchReport, collect_machine_info, normalize_run_mode
from src import phase1_adapters, resource_manager
from src.model_demos import get_model_demo, _negative_prompt_for_model

# Methods that apply to every model with an ollama_tag
OLLAMA_METHODS = ["ollama_gpu", "ollama_cpu"]
# Methods that apply only to models with an onnx_repo
ONNX_METHODS = ["onnx_openvino", "onnx_directml", "onnx_cpu"]
# Image-gen method (Extended mode + GPU profiles only)
IMAGE_METHOD = "image_comfyui"

# Backend memory consumers — used by smart-skip ceilings so a CUDA OOM on a
# 13B model auto-skips 30B/70B models on the same method without re-paying
# the load + crash tax. Image methods are intentionally NOT linked to text
# OOM events: a text-Ollama OOM never blocks image-gen rows.
_TEXT_METHODS: frozenset[str] = frozenset(OLLAMA_METHODS + ONNX_METHODS + ["phase1"])
_OOM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"cuda(?:malloc| out of memory|errorout)",
        r"out of memory",
        r"\boom\b",
        r"oomkilled",
        r"cudaerrormemoryallocation",
        r"hipmalloc",
        r"hipout of memory",
        r"\bdml[\w_ ]*alloc",
        r"directml.*alloc",
        r"failed to allocate",
        r"unable to allocate",
        r"memoryerror",
        r"std::bad_alloc",
        # Windows STATUS_ACCESS_VIOLATION often follows a near-OOM crash in the
        # native Ollama/llama.cpp worker; treat it as OOM-ish for the purpose
        # of ceiling tightening so we don't immediately retry the same backend
        # at a larger size.
        r"-?1073741819",
        r"\bcode\s+139\b",
        r"\bcode\s+137\b",
        r"killed by signal",
        r"signal 9",
        r"signal\s*sigkill",
        r"failed to load model",
        r"llama runner process has terminated",
        # v5.5.6+: Windows-specific resource-exhaustion patterns surfaced
        # by ComfyUI on AI PC class hardware when CPU image-gen runs
        # the host out of pageable memory or paged-pool handles.  Ron's
        # logs show these three repeating across every sample of every
        # image-gen model once the first one breaks the bank:
        #   "Not enough memory resources are available to complete this operation."
        #   "resource deadlock would occur"
        #   "Timeout after 300s"
        # The first two are Win32 STATUS_NO_MEMORY / EDEADLK shaped and
        # mean the model genuinely cannot fit; without these patterns the
        # OOM ceiling stayed at infinity and every subsequent (larger)
        # image-gen model paid the same dead-end startup cost.  Adding
        # them tightens the per-method ceiling on the first failure so
        # bigger same-class models pre-skip with an adaptive_skip row
        # rather than spending 5-15 min/sample on a guaranteed failure.
        r"not enough memory resources",
        r"resource deadlock",
        r"deadlock would occur",
        r"semaphore timeout",
        r"insufficient system resources",
    )
)


def _looks_like_oom_error(text: object) -> bool:
    """Best-effort classifier for OOM-shaped error strings.

    Used by the runner's adaptive smart-skip logic to lower the per-method
    OOM ceiling after a failure so subsequent same-method runs at larger
    sizes are skipped without retrying. Conservative: when in doubt we
    return False rather than over-skip, because the ceiling we set is
    monotonically tightening and a false positive blocks all larger same-
    method models for the rest of the run.
    """
    if not text:
        return False
    payload = str(text)
    if not payload.strip():
        return False
    for pattern in _OOM_PATTERNS:
        if pattern.search(payload):
            return True
    return False


def _looks_like_disk_full_text(text: object) -> bool:
    """Wrapper around constrained_env.is_disk_full_error_text that swallows None."""
    if not text:
        return False
    try:
        return bool(constrained_env.is_disk_full_error_text(str(text)))
    except Exception:
        return False


DEFAULT_PROMPT = "/no_think\nReturn only this exact sentence: A neural network learns patterns from examples."
QUICK_OLLAMA_NUM_PREDICT = 4096
EXTENDED_OLLAMA_NUM_PREDICT = 4096
EXTENDED_OLLAMA_TIMEOUT_PAD_S = 120
EXTENDED_OLLAMA_TIMEOUT_ROUND_S = 60

# Cold-start budget for the FIRST image-gen run in a benchmark. Cloud VMs
# (managed VM provisioning + roaming profile + Defender + vGPU partition +
# cold torch import + CUDA context init) need substantially longer than a
# laptop: 60 s was the original budget and was the root cause of "FAIL
# (ComfyUI is not running and could not be started)" reports across a
# range of GPU SKUs (all failed at iter [1/N] on
# `sdxl-lowvram`).
# Subsequent runs short-circuit on `client.is_running()` so this adds zero
# warm-path overhead.
COMFYUI_COLD_START_TIMEOUT_S = 180
# Crash-mid-run restart budget — leave at the original 120 s; torch is already
# in the OS cache and CUDA was initialised once, so a reactive restart is much
# faster than a cold start.
COMFYUI_CRASH_RESTART_TIMEOUT_S = 120
# Generic fallback message used when a legacy ensure-callback returns `bool`
# instead of `(bool, reason)` — preserves the original 5.3.6 user-visible
# text so headless callers and pre-update wirings keep behaving the same.
LEGACY_COMFYUI_START_FAILURE_REASON = (
    "ComfyUI is not running and could not be started"
)
_PARAMETER_B_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*b")
_HIDDEN_REASONING_PREFIXES = (
    "okay, the user",
    "hmm, the user",
    "the user wants me",
    "we are given a string",
    "we need answer",
    "i need to understand",
)


def _fmt_bytes(value: int) -> str:
    size = float(value or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _ceil_to(value: float, step: int) -> int:
    step = max(1, int(step))
    return int(((int(value) + step - 1) // step) * step)


def _model_parameter_billions(model: dict) -> float:
    for key in ("parameters", "name", "id", "ollama_tag"):
        match = _PARAMETER_B_RE.search(str(model.get(key, "") or ""))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
    return 0.0


def _benchmark_skip_methods(model: dict) -> set[str]:
    raw = model.get("benchmark_skip_methods")
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def _is_image_catalog_model(model: dict) -> bool:
    """True when the catalog entry is an image-generation model."""
    if str(model.get("backend") or "").lower() == "comfyui":
        return True
    if model.get("comfyui_model"):
        return True
    return str(model.get("category") or "").strip().lower() == "image generation"


def _bench_sort_key(model: dict) -> tuple:
    """Sort key used for benchmark ordering: smallest/fastest first.

    Falls back through ``size_gb`` -> ``min_vram_gb`` -> ``min_ram_gb`` so a
    catalog entry whose ``size_gb`` is 0 (e.g. a profile-only image-gen row)
    doesn't sort first incorrectly. Ties break on the catalog id so order is
    stable across runs.
    """
    def _pos(value) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.0
        return f if f > 0 else 0.0

    size = _pos(model.get("size_gb"))
    if size == 0:
        size = _pos(model.get("min_vram_gb"))
    if size == 0:
        size = _pos(model.get("min_ram_gb"))
    return (size, str(model.get("id") or model.get("name") or ""))


def _order_models_text_then_image(models: list[dict]) -> list[dict]:
    """Partition into text-first / image-last; each partition smallest-first.

    Keeps text models (chat, ONNX adapters, utility entries) ahead of every
    image-generation model so a benchmark run shows fast progress on the
    cheap rows before paying for big SDXL/Flux/Z-Image generations. Within
    each partition rows are sorted by ``_bench_sort_key``.
    """
    text_models: list[dict] = []
    image_models: list[dict] = []
    for model in models:
        if _is_image_catalog_model(model):
            image_models.append(model)
        else:
            text_models.append(model)
    text_models.sort(key=_bench_sort_key)
    image_models.sort(key=_bench_sort_key)
    return text_models + image_models


def _looks_like_unstripped_hidden_reasoning(text: str) -> bool:
    lower = str(text or "").lstrip().lower()
    if "<think" in lower or "</think>" in lower:
        return True
    return any(lower.startswith(prefix) for prefix in _HIDDEN_REASONING_PREFIXES)


# Repetition-loop guard for streaming text generations.
#
# Pathology: small instruction-tuned models (e.g. MiniCPM-V on a meta-prompt at
# temperature=0.0) can fall into a fake-roleplay or near-identical-paragraph
# loop and never emit an end token. They then burn the full benchmark token
# budget producing the same block 30+ times, and the run reports as
# ``output_truncated`` even though the budget is not the real cause. The
# extended-mode high-VRAM run on 2026-05-23 hit this once on
# minicpm-v-vision/ollama_gpu sample 2 — same paragraph emitted ~40 times
# before the 4096-token ceiling cut it off.
#
# Heuristic: once the response has at least ``_REPETITION_MIN_CHARS`` chars,
# look at the trailing ``_REPETITION_TAIL_CHARS`` and try to find a non-trivial
# block at the very end that repeats ``_REPETITION_REQUIRED_REPEATS`` times
# back-to-back. We require blocks of at least ``_REPETITION_BLOCK_MIN`` chars
# so trivial single-token repetition (``aaaa``) — which real models rarely
# produce in serious answers — does not trigger. We also reject candidates
# whose suffix is whitespace-only or has fewer than 3 unique characters.
_REPETITION_MIN_CHARS = 600
_REPETITION_TAIL_CHARS = 1800
_REPETITION_BLOCK_MIN = 50
_REPETITION_BLOCK_MAX = 600
_REPETITION_REQUIRED_REPEATS = 3
_REPETITION_STREAM_CHECK_EVERY = 32  # streamed chunks between in-stream polls


def _looks_like_repetition_loop(text: object) -> bool:
    """Detect that *text* ends with a non-trivial block repeated >=3 times.

    Used by the batch runner to short-circuit Ollama generations that fall
    into degenerate loops, so we report ``failure_phase=repetition_loop``
    instead of the misleading ``output_truncated``. Conservative by design:
    when in doubt we return False so genuine answers are never flagged.

    The block comparison is **byte-exact** — a loop where each repeat has
    any differing byte (e.g. trailing whitespace drift, monotonic prefixes
    like ``"Block 1:"`` / ``"Block 2:"`` with otherwise identical bodies)
    will NOT trip. This is intentional: relaxing to fuzzy matching has
    historically produced false positives on legitimate structured output
    (numbered lists, repeated table rows with similar formatting but
    different data). If a real loop pattern slips through, prefer adding
    a curated ``MODEL_DEMO_SAMPLE_OVERRIDES`` entry or a per-model
    ``benchmark_repeat_penalty`` override over loosening this detector.
    """
    if not text:
        return False
    payload = str(text)
    total = len(payload)
    if total < _REPETITION_MIN_CHARS:
        return False
    tail = payload[-_REPETITION_TAIL_CHARS:]
    tail_len = len(tail)
    max_block = min(_REPETITION_BLOCK_MAX, tail_len // _REPETITION_REQUIRED_REPEATS)
    if max_block < _REPETITION_BLOCK_MIN:
        return False
    # Cheap pre-check (perf): if any block in [Bmin..Bmax] repeats >=3x at
    # the end, then ``tail[-Bmin:]`` MUST also appear somewhere in the
    # window ``[tail_len - Bmax - Bmin, tail_len - Bmin]``. ``str.find`` on
    # a small buffer is microseconds; on clean prose tails (the common
    # case) this short-circuits the O(Bmax-Bmin) scan below, taking the
    # detector from ~1.5 ms to ~5–20 μs per call. Semantics-preserving.
    suffix_min = tail[-_REPETITION_BLOCK_MIN:]
    search_start = max(0, tail_len - _REPETITION_BLOCK_MAX - _REPETITION_BLOCK_MIN)
    search_end = tail_len - _REPETITION_BLOCK_MIN
    if tail.find(suffix_min, search_start, search_end) < 0:
        return False
    # Search from the largest plausible block down so we report the longest
    # matching cycle (cheaper than checking every size; the first hit wins).
    for block in range(max_block, _REPETITION_BLOCK_MIN - 1, -1):
        suffix = tail[-block:]
        if not suffix.strip() or len(set(suffix)) <= 2:
            continue
        matches = True
        for i in range(1, _REPETITION_REQUIRED_REPEATS):
            start = tail_len - block * (i + 1)
            if start < 0 or tail[start:start + block] != suffix:
                matches = False
                break
        if matches:
            return True
    return False


class BatchRunner:
    """Runs every model × method combo, collecting benchmark results."""

    def __init__(
        self,
        prompt: str = DEFAULT_PROMPT,
        timeout: int = 300,
        cleanup: bool = False,
        model_ids: Optional[list[str]] = None,
        skip_gpu: bool = False,
        skip_cpu: bool = False,
        skip_onnx: bool = False,
        max_failures: int = 10,
        output_dir: Path = Path("."),
        models_dir: Path = Path("models"),
        specific_combos: Optional[list[tuple]] = None,
        low_resources_mode: bool = False,
        comfyui_client=None,
        ensure_comfyui_ready=None,
        prepare_image_model=None,
        skip_phase1: bool = True,
        cleanup_downloaded_only: bool = False,
        capacity_ram_gb: float | None = None,
        capacity_vram_gb: float | None = None,
        capacity_has_gpu: bool | None = None,
        allow_oversize: bool = False,
        force_all: bool = False,
        report_file_stem: str | None = None,
        run_mode: str = "quick",
        skip_image: bool = False,
        image_output_subdir: str = "",
        skip_combos: Optional[set[tuple[str, str, int]]] = None,
    ):
        self.prompt = prompt
        self.timeout = timeout
        self.cleanup = cleanup
        self.model_ids = model_ids
        self.skip_gpu = skip_gpu
        self.skip_cpu = skip_cpu
        self.skip_onnx = skip_onnx
        # v5.5.1+ Force-All mode (best-effort baseline) bypasses the
        # consecutive-failure ceiling so a single dead backend can't bail
        # the whole run. Per-method OOM and disk ceilings still tighten
        # adaptively so the run stays "aggressive but not stupid".
        self.force_all = bool(force_all)
        if self.force_all:
            self.max_failures = sys.maxsize
        else:
            self.max_failures = max_failures
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.specific_combos = specific_combos
        self.low_resources_mode = low_resources_mode
        self._comfyui_client = comfyui_client
        # Optional callback used by image-gen runs to (re)start ComfyUI when it
        # is not running.  Signature: (timeout_s: int) -> bool.  When None,
        # _run_image_comfyui falls back to a bare is_running() probe.
        self._ensure_comfyui_ready = ensure_comfyui_ready
        self._prepare_image_model = prepare_image_model
        self.skip_phase1 = skip_phase1
        self.cleanup_downloaded_only = cleanup_downloaded_only
        self.capacity_ram_gb = capacity_ram_gb
        self.capacity_vram_gb = capacity_vram_gb
        self.capacity_has_gpu = capacity_has_gpu
        # Force-All implies allow_oversize for capacity shortfalls so every
        # model in the catalog gets attempted at least once.
        self.allow_oversize = bool(allow_oversize) or self.force_all
        self.run_mode = normalize_run_mode(run_mode, default="quick")
        self.skip_image = skip_image
        self.image_output_subdir = image_output_subdir
        # v5.5.6+ Resume Today's Run: pre-populated set of (model_id, method,
        # sample_index) triples that have already been run **successfully**
        # in a prior session and should be silently skipped here (no result
        # row added, no progress tick, no setup cost paid). v2026.06.01.2+
        # — failures are NOT in this set (they're re-attempted by Resume),
        # see ``BatchReport.get_completed_combos`` for the contract. When
        # None or empty, behaves exactly like a fresh run. The caller
        # (_start_benchmark_resume) is responsible for merging the new
        # partial report back into the prior report via
        # ``BatchReport.append_resume_results``.
        self.skip_combos: set[tuple[str, str, int]] = set(skip_combos or ())

        self.ollama = OllamaClient()
        self.report = BatchReport(
            machine_info=collect_machine_info(models_dir),
            file_stem=report_file_stem,
            run_mode=self.run_mode,
        )
        self._consecutive_failures = 0
        # v5.3.6+: total count of ``environment_skip`` failures across the
        # run.  Used to emit a banner at end-of-run if EVERY Ollama row was
        # skipped so Ron sees one clear "address disk pressure" message
        # rather than thinking the whole benchmark broke.  This is a count
        # of skip events (one per row), not a streak — it never resets.
        self._environment_skip_count = 0
        self._ollama_attempt_count = 0
        # v5.5.1+ adaptive smart-skip ceilings. Each maps backend method →
        # smallest model size (in GB) that has hit OOM or disk pressure.
        # Future runs on the same method at size >= ceiling auto-skip with
        # ``failure_phase="adaptive_skip"`` without spending the load/run
        # time.  Image methods are intentionally NOT linked to text OOM
        # events: a text-Ollama OOM never blocks image-gen rows.
        self._oom_ceiling_gb: dict[str, float] = {}
        self._disk_blocked_ceiling_gb: dict[str, float] = {}
        # Aggregate of every captured stdout/stderr buffer so a partial-
        # failure run can persist one combined ``<stem>_run.log`` for
        # post-mortem analysis without re-reading per-model log files.
        self._captured_log_chunks: list[str] = []
        # v5.5.1+ perf: track running byte total in O(1) instead of
        # re-summing on every append.  Without this, the bound loop
        # in _run_with_captured_logs is O(N) per result and grows
        # quadratic over a long Force-All run; the truncation guard
        # below also defends against any single ComfyUI/diffusers
        # debug dump that exceeds the cap on its own.
        self._captured_log_bytes: int = 0
        self._captured_log_cap_bytes: int = 4_000_000
        self._interrupted = False
        self._active_stop_event: Optional[threading.Event] = None
        self._purge_done = False  # one-time purge flag for low resources mode
        self._selected_batch_ids: set[str] = set()
        self._selected_batch_tags: set[str] = set()
        self._downloaded_ollama_tags: set[str] = set()
        # Per-sample overrides; set before each call into _run_one() and
        # cleared after.  Used by _run_ollama / _run_onnx / _run_image_comfyui
        # to read the current sample's prompt/options without changing their
        # signatures.
        self._current_sample: dict | None = None
        # Cold-start guard: once an image-gen run fails to bring up
        # ComfyUI (deps install failure, missing install, polling timeout,
        # subprocess crash, etc.), remember the reason so iterations 2/3 of
        # the same model — and every subsequent image-gen model in the batch —
        # fail INSTANTLY with the same reason instead of re-paying the
        # multi-minute cold-start cost.  Cleared via `_reset_image_failure_cache`
        # if the caller ever wants a fresh probe (unused today; the bench UI
        # discards the runner at end-of-batch).
        self._image_global_failure_reason: Optional[str] = None
        self._image_failure_cache_lock = threading.Lock()
        # v5.5.12: cached registry-reachability probe. The May 2026 Mac M4
        # run logged 94/121 failures as ~3-minute sequential ``ollama pull``
        # timeouts against ``registry.ollama.ai`` returning NXDOMAIN — DNS
        # was dead the whole run. Probing once at the first non-local
        # Ollama row (and reusing the result) lets us downgrade those rows
        # to ``environment_skip`` instead of paying the timeout per row.
        # ``None`` = not yet probed; the result is ``(reachable, reason)``.
        self._registry_dns_cache: Optional[tuple[bool, str]] = None
        phase1_adapters.configure_benchmark_environment()

    # ── Public ─────────────────────────────────────────────────────────────────

    def request_stop(self) -> None:
        """Request cancellation of the batch and the currently active run."""
        self._interrupted = True
        if self._active_stop_event is not None:
            self._active_stop_event.set()

    def save_partial(self) -> tuple[Optional[Path], Optional[Path]]:
        """Persist whatever results are available right now.

        Returns ``(json_path, html_path)``.  In retry mode (specific_combos is
        not None), the caller owns the merged report so we return ``(None, None)``.
        """
        if self.specific_combos is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return None, None
        if not self.report.results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return None, None
        # Refresh end-time so an in-flight (or cancelled) run still shows the
        # real wall-clock time of the last completed result, not blank text.
        self.report.stamp_end_time()
        return self.report.save_json(self.output_dir), self.report.save_html(self.output_dir)

    def _num_predict_for(self, model: dict, method: str) -> int:
        # Extended mode validates sample-prompt answers. It must not use
        # Ollama's "-2 = fill context" sentinel because large 131K-context
        # models can spend many minutes generating instead of producing a
        # bounded, reportable answer. Length stops are treated as failures
        # below so capped answers are never silently reported as complete.
        if self.run_mode == "extended":
            value = _positive_int(model.get("benchmark_num_predict"))
            if value:
                return value
            return EXTENDED_OLLAMA_NUM_PREDICT
        value = _positive_int(model.get("benchmark_num_predict"))
        if value:
            return value
        return QUICK_OLLAMA_NUM_PREDICT

    def _ollama_generation_timeout_for(self, model: dict, method: str, num_predict: int) -> int:
        configured = _positive_int(model.get("benchmark_timeout_s"))
        timeout = max(int(self.timeout), configured or 0)

        budget = _positive_int(num_predict) or EXTENDED_OLLAMA_NUM_PREDICT
        params_b = _model_parameter_billions(model)
        timeout_pad = EXTENDED_OLLAMA_TIMEOUT_PAD_S if self.run_mode == "extended" else 60
        if method == "ollama_cpu":
            if params_b >= 60:
                token_floor = 0.50
                load_floor = 300
            elif params_b >= 25:
                token_floor = 0.75
                load_floor = 240
            elif params_b >= 10:
                token_floor = 1.00
                load_floor = 180
            else:
                token_floor = 2.00
                load_floor = 90
        else:
            if params_b >= 25:
                token_floor = 4.00
                load_floor = 180
            elif params_b >= 10:
                token_floor = 6.00
                load_floor = 120
            else:
                token_floor = 10.00
                load_floor = 60
        estimated = load_floor + (budget / token_floor) + timeout_pad
        return max(timeout, _ceil_to(estimated, EXTENDED_OLLAMA_TIMEOUT_ROUND_S))

    def _log_stem(self, model: dict, method: str) -> str:
        raw = f"{model.get('id', 'model')}_{method}"
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)[:120]

    def _run_with_captured_logs(self, model: dict, method: str, fn):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = fn()
        text = buffer.getvalue()
        if not text.strip():
            return result, ""
        # v5.5.1+: aggregate every captured per-model buffer into a single
        # in-memory list so a partial-failure run can persist one combined
        # ``<stem>_run.log`` next to the report for easy post-mortem
        # without having to walk N per-model log files. Bounded to roughly
        # 4 MB (via _captured_log_cap_bytes) to protect against
        # pathological repeats; truncate any single mega-chunk first so
        # one runaway 30 MB diffusers debug dump can't pin the buffer
        # above the cap indefinitely.
        chunk_header = f"\n===== {model.get('id', 'model')} / {method} =====\n"
        chunk = chunk_header + text
        cap = self._captured_log_cap_bytes
        if len(chunk) > cap:
            half = max(cap // 2, 1)
            head = chunk[:half]
            tail = chunk[-(cap - half):]
            chunk = head + "\n... [chunk truncated to stay under run.log cap] ...\n" + tail
        self._captured_log_chunks.append(chunk)
        self._captured_log_bytes += len(chunk)
        while self._captured_log_bytes > cap and len(self._captured_log_chunks) > 1:
            dropped = self._captured_log_chunks.pop(0)
            self._captured_log_bytes -= len(dropped)
        logs_dir = self.output_dir / "logs" / self.report.file_stem
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{self._log_stem(model, method)}.log"
        if log_path.exists():
            index = 2
            while True:
                candidate = logs_dir / f"{self._log_stem(model, method)}_{index}.log"
                if not candidate.exists():
                    log_path = candidate
                    break
                index += 1
        log_path.write_text(text, encoding="utf-8", errors="replace")
        return result, str(log_path)

    def run(self) -> BatchReport:
        """Execute the full benchmark suite. Returns the report."""
        # Graceful Ctrl+C handling (only works on the main thread)
        original_sigint = None
        _is_main = threading.current_thread() is threading.main_thread()
        if _is_main:
            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_interrupt)

        models = self._select_models()
        self._selected_batch_ids = {m["id"] for m in models}
        self._selected_batch_tags = {m.get("ollama_tag", "") for m in models if m.get("ollama_tag")}
        total_combos = self._count_combos(models)
        done = 0

        print(f"\nBatch benchmark: {len(models)} models, ~{total_combos} runs ({self.run_mode} mode)")
        print(f"Prompt: {self.prompt!r}")
        if self.force_all:
            # NB: deliberately do NOT print the host's CPU/RAM/VRAM specs
            # here (regression-critical row 303 — SKU privacy). The
            # banner reports behaviour only.
            print(
                "Force All mode: best-effort baseline. Profile capacity gating bypassed, "
                "consecutive-failure ceiling lifted, and oversize override is on. "
                "Per-method OOM and disk ceilings still skip larger same-backend models, "
                "so failures in one category never block the next category."
            )
        print(f"Timeout per run: {self.timeout}s | Max consecutive failures: {self.max_failures}")
        # v5.5.12: doctor banner. Up-front heads-up so the user knows
        # download-requiring Ollama rows will be skipped instead of
        # piling up ~3-minute pull timeouts. Only fires when there is at
        # least one Ollama tag in this batch (otherwise the message is
        # noise).
        ollama_in_batch = any(m.get("ollama_tag") for m in models)
        if ollama_in_batch:
            reachable, dns_reason = self._is_registry_reachable()
            if not reachable:
                print(
                    f"[doctor] registry.ollama.ai DNS lookup failed ({dns_reason}). "
                    "Ollama models that are not already pulled locally will be "
                    "skipped as environment_skip instead of failing with a "
                    "~3-minute timeout each. Pre-pull with 'ollama pull <tag>' "
                    "on a network-up machine, or re-run once the connection is back."
                )
        print()

        try:
            for model in models:
                if self._interrupted:
                    break
                if self._consecutive_failures >= self.max_failures:
                    print(f"\n!! Stopping early: {self.max_failures} consecutive failures reached.")
                    break

                methods = self._methods_for(model)
                if not methods:
                    continue

                all_skipped = True
                for method in methods:
                    if self._interrupted:
                        break
                    selected_samples = self._iter_selected_samples_for(model, method)
                    for sample_index, sample, sample_count in selected_samples:
                        if self._interrupted:
                            break
                        if self._consecutive_failures >= self.max_failures:
                            break
                        done += 1
                        display_method = "utility" if method == "phase1" else (
                            "image" if method == IMAGE_METHOD else method
                        )
                        label = model["name"]
                        if sample_count > 1:
                            label = f"{label} [{sample_index + 1}/{sample_count}]"
                        print(
                            f"[{done}/{total_combos}] {label} / {display_method} ... ",
                            end="", flush=True,
                        )

                        self._current_sample = dict(sample)
                        self._current_sample["index"] = sample_index
                        self._current_sample["count"] = sample_count
                        try:
                            # v5.5.1+ adaptive smart-skip: a prior OOM or
                            # disk-pressure failure on the same method has
                            # already lowered the size ceiling for this
                            # method.  Synthesize a fast adaptive_skip
                            # result instead of paying the download/load
                            # tax.  This is the "aggressive but not
                            # stupid" half of Force-All: every model still
                            # gets a chance at SMALLER backends; only the
                            # larger same-backend rows are pre-filtered.
                            smart_skip = self._smart_skip_reason(model, method)
                            if smart_skip:
                                result = RunResult(
                                    model_id=model["id"],
                                    model_name=model["name"],
                                    method=method,
                                    success=False,
                                    error=smart_skip,
                                    failure_phase="adaptive_skip",
                                    system_snapshot=self._system_snapshot(),
                                )
                            else:
                                result = self._run_one(model, method)
                        finally:
                            self._current_sample = None

                        # Tag with sample plumbing so the HTML report can group
                        # multiple runs per model and link prompt sources.
                        if not result.sample_id:
                            result.sample_id = sample.get("id", "")
                        if not result.sample_title:
                            result.sample_title = sample.get("title", "")
                        result.sample_index = sample_index
                        result.sample_count = sample_count
                        if not result.prompt_source:
                            result.prompt_source = sample.get("source", "")
                        if not result.prompt:
                            result.prompt = sample.get("prompt", "") or self.prompt

                        self.report.add(result)

                        if result.success:
                            all_skipped = False
                            self._consecutive_failures = 0
                            if result.method == IMAGE_METHOD or result.surface == "image":
                                size = f"{result.image_width}x{result.image_height}" if result.image_width else "image"
                                print(f"OK  {size}  total={result.total_time:.1f}s")
                            elif result.method == "phase1" or result.metric_kind == "utility":
                                metric = result.metric_value or f"{result.total_time:.1f}s"
                                print(f"OK  utility={metric}  total={result.total_time:.1f}s")
                            else:
                                print(
                                    f"OK  {result.tokens_per_sec:.1f} tok/s  "
                                    f"ttft={result.ttft:.2f}s  total={result.total_time:.1f}s"
                                )
                        else:
                            # v5.3.6+: ``environment_skip`` is a separate
                            # failure phase for fixable environment issues
                            # (roaming profile container full, OLLAMA_MODELS
                            # not relocated, etc.) that should NOT count
                            # against ``_consecutive_failures`` — otherwise
                            # the constrained GPU cloud VM bench bails after
                            # 5 disk-full models even though smaller models
                            # would still fit.  Other failure phases (real
                            # runtime errors, generation timeouts, missing
                            # tags) keep the original strict counter so a
                            # genuinely broken environment still stops the
                            # run.
                            # v5.5.1+: ``adaptive_skip`` (synthesised above
                            # when a prior OOM/disk failure lowered the
                            # method ceiling) is also excluded from the
                            # streak counter — the actual failure already
                            # counted on the first try.
                            if result.failure_phase in {"environment_skip", "adaptive_skip"}:
                                if result.failure_phase == "environment_skip":
                                    self._environment_skip_count += 1
                            else:
                                self._consecutive_failures += 1
                            # v5.5.1+: feed real failures into the per-
                            # method ceilings so subsequent larger same-
                            # backend rows can be smart-skipped.
                            self._record_outcome_for_smart_skip(model, method, result)
                            reason = result.error or "unknown error"
                            # v5.3.6+: 60-char truncation was chopping the
                            # actual ComfyUI cold-start failure
                            # mid-word ("FAIL  (ComfyUI is not running and
                            # could not be )"), so Ron's bench screenshots
                            # showed nothing actionable.  Expanded to 200
                            # chars so the real reason ("…subprocess exited
                            # with code 3221225477 during startup — see
                            # comfyui.log") prints in full; _normalise_ensure_result
                            # already strips newlines and caps at 240 chars.
                            # v5.5.1+: ``adaptive_skip`` and ``environment_skip``
                            # were both printing as "FAIL  (...)" which made
                            # Force-All sweeps look catastrophically broken in
                            # screenshots and contradicted the reason text
                            # ("Skipped: ollama_gpu hit OOM at 13 GB earlier"
                            # under a FAIL header).  Both are excluded from
                            # the consecutive-failure streak just above, so
                            # mirror that classification in the user-visible
                            # log line — SKIP for adaptive_skip, SKIP (env)
                            # for environment_skip, FAIL for everything else.
                            if result.failure_phase == "adaptive_skip":
                                print(f"SKIP  ({reason[:200]})")
                            elif result.failure_phase == "environment_skip":
                                print(f"SKIP (env)  ({reason[:200]})")
                            else:
                                print(f"FAIL  ({reason[:200]})")

                        # Incremental save: flush the JSON after every result so a
                        # hard kill / hang / crash mid-run still leaves a usable
                        # report on disk (and lets "Retry Failed" work afterwards).
                        # Skipped in retry mode (specific_combos != None) — the
                        # caller owns the file there and will merge + save itself.
                        if self.specific_combos is None:
                            try:
                                # Stamp end_time on every incremental save so a
                                # cancelled or hung run reflects the real wall
                                # clock of the last completed result instead of
                                # blank text in the timing block.
                                self.report.stamp_end_time()
                                self.report.save_json(self.output_dir)
                                # HTML rebuilt incrementally so users can refresh
                                # the report tab while a run is in progress.
                                self.report.save_html(self.output_dir)
                            except Exception as e:
                                print(f"  (incremental save failed: {e})")

                    if self.cleanup:
                        self._cleanup_model(model, method)
                    else:
                        self._release_after_run(model, method)

                # If every method for this model was skipped/failed, larger models will too
                if all_skipped and model.get("size_gb", 0) > 0:
                    # Don't increment failure counter for resource skips
                    pass

        finally:
            if _is_main and original_sigint is not None:
                signal.signal(signal.SIGINT, original_sigint)
            # v5.3.6+: if EVERY Ollama row in the run was skipped for
            # environment reasons (typically roaming profile container
            # full on a constrained GPU cloud VM), emit a single banner so
            # the operator sees one actionable message rather than scrolling
            # through N identical "Skipped" lines and thinking the bench
            # is broken.  Guarded on ``_ollama_attempt_count > 0`` so we
            # don't print the banner when the run had no Ollama models
            # to begin with (e.g. image-only Extended run).
            if (
                self._ollama_attempt_count > 0
                and self._environment_skip_count >= self._ollama_attempt_count
            ):
                print()
                print(f"!! {constrained_env.CONSTRAINED_ALL_OLLAMA_SKIPPED_BANNER}")
            # When running a retry (specific_combos), the caller handles
            # merge + save so we must NOT overwrite the full results file.
            if self.specific_combos is None:
                # Pin the final end-time before the closing _save_report so the
                # report's timing block reflects the actual run completion (or
                # cancel) wall clock instead of an incrementally-stamped value
                # from an earlier sample.
                self.report.stamp_end_time()
                self._save_report()

        return self.report

    # ── Model selection ────────────────────────────────────────────────────────

    def _benchmark_capacity(self) -> tuple[float, bool, float, bool]:
        """Return RAM GB, GPU availability, VRAM GB, and unified-memory flag.

        Force-All override (v5.5.1+): when ``self.force_all`` is True we
        bypass the synthetic profile caps and report the host's real RAM /
        VRAM / GPU so GPU methods can be attempted on a profile whose SKU
        is nominally CPU-only. We never invent hardware: when the host
        genuinely has no GPU, ``has_gpu`` stays False and ``min_vram``
        guards still fail closed at the `_fits_benchmark_capacity` layer.
        """
        if self.force_all:
            ram = get_ram_info()
            total_ram_gb = ram.get("total_mb", 0) / 1024
            real_ram_gb = max(total_ram_gb, round(total_ram_gb))
            gpus = get_gpu_info()
            if not gpus:
                return real_ram_gb, False, 0.0, False
            gpu = max(gpus, key=lambda g: g.get("vram_total_mb", 0))
            return (
                real_ram_gb,
                True,
                gpu.get("vram_total_mb", 0) / 1024,
                bool(gpu.get("unified_memory")),
            )

        if self.capacity_ram_gb is not None:
            capacity_ram_gb = max(float(self.capacity_ram_gb), round(float(self.capacity_ram_gb)))
        else:
            ram = get_ram_info()
            total_ram_gb = ram.get("total_mb", 0) / 1024
            capacity_ram_gb = max(total_ram_gb, round(total_ram_gb))

        if self.capacity_has_gpu is not None or self.capacity_vram_gb is not None:
            capacity_vram_gb = max(float(self.capacity_vram_gb or 0), 0.0)
            has_gpu = bool(self.capacity_has_gpu) if self.capacity_has_gpu is not None else capacity_vram_gb > 0
            return capacity_ram_gb, has_gpu and capacity_vram_gb > 0, capacity_vram_gb, False

        gpus = get_gpu_info()
        if not gpus:
            return capacity_ram_gb, False, 0.0, False
        gpu = max(gpus, key=lambda g: g.get("vram_total_mb", 0))
        return (
            capacity_ram_gb,
            True,
            gpu.get("vram_total_mb", 0) / 1024,
            bool(gpu.get("unified_memory")),
        )

    def _select_models(self) -> list[dict]:
        """Return catalog models sorted smallest-to-largest, optionally filtered."""
        # Use load_catalog() (not raw MODELS) so models_catalog.json's
        # disabled_builtin_ids and JSON-only additions are honored. Falls back
        # to MODELS automatically when the JSON is missing/invalid.
        catalog_models = load_catalog()
        if self.specific_combos is not None:
            combo_ids = {combo[0] for combo in self.specific_combos if len(combo) >= 1}
            models = [m for m in catalog_models if m["id"] in combo_ids]
        elif self.model_ids:
            models = [m for m in catalog_models if m["id"] in self.model_ids]
            missing = set(self.model_ids) - {m["id"] for m in models}
            if missing:
                print(f"Warning: model IDs not found in catalog: {missing}")
        else:
            models = list(catalog_models)
        selected = _order_models_text_then_image(models)
        runnable = []
        for model in selected:
            if self.skip_phase1 and model.get("phase1_adapter"):
                # Toolbox / utility models are excluded from benchmark runs by
                # design. They are exercised through the Toolbox UI instead.
                continue
            if model.get("benchmark_skip_reason"):
                print(
                    f"Skipping {model.get('name', model.get('id'))}: "
                    f"{model.get('benchmark_skip_reason')}"
                )
                continue
            methods = self._methods_for(model)
            if methods:
                runnable.append(model)
            elif model.get("phase1_adapter") and not self.skip_phase1:
                missing = phase1_adapters.missing_dependencies_for_model(model)
                if missing:
                    print(
                        f"Skipping {model.get('name', model.get('id'))}: "
                        f"missing optional utility packages: {', '.join(missing[:5])}"
                    )
            elif self.specific_combos is None:
                ok, reason = self._fits_benchmark_capacity(model, "ollama_cpu")
                if not ok:
                    print(f"Skipping {model.get('name', model.get('id'))}: {reason}")
        return runnable

    def _methods_for(self, model: dict) -> list[str]:
        """Return the list of methods to test for this model."""
        if model.get("benchmark_skip_reason"):
            return []
        skipped_methods = _benchmark_skip_methods(model)
        if self.specific_combos is not None:
            methods = []
            for combo in self.specific_combos:
                if len(combo) < 2:
                    continue
                mid, method = combo[0], combo[1]
                if mid == model["id"] and method not in skipped_methods and method not in methods:
                    methods.append(method)
            return methods
        if self._is_image_model(model):
            if (
                self.run_mode in ("extended", "quick")
                and not self.skip_image
                and self._image_gen_supported(model)
            ):
                return [IMAGE_METHOD]
            return []
        if model.get("phase1_adapter"):
            if phase1_adapters.missing_dependencies_for_model(model):
                return []
            if self.skip_phase1 or not self._fits_benchmark_capacity(model, "phase1")[0]:
                return []
            return ["phase1"]
        methods = []
        if model.get("ollama_tag"):
            if not self.skip_gpu and self._fits_benchmark_capacity(model, "ollama_gpu")[0]:
                methods.append("ollama_gpu")
            if not self.skip_cpu and self._fits_benchmark_capacity(model, "ollama_cpu")[0]:
                methods.append("ollama_cpu")
        if model.get("onnx_repo") and not self.skip_onnx:
            if ONNX_AVAILABLE and OPENVINO_AVAILABLE and self._fits_benchmark_capacity(model, "onnx_openvino")[0]:
                methods.append("onnx_openvino")
            if ONNX_AVAILABLE and DIRECTML_AVAILABLE and self._fits_benchmark_capacity(model, "onnx_directml")[0]:
                methods.append("onnx_directml")
            if ONNX_AVAILABLE and self._fits_benchmark_capacity(model, "onnx_cpu")[0]:
                methods.append("onnx_cpu")
        return [method for method in methods if method not in skipped_methods]

    @staticmethod
    def _is_image_model(model: dict) -> bool:
        return (
            model.get("backend") == "comfyui"
            or bool(model.get("comfyui_model"))
        )

    def _image_gen_supported(self, model: dict) -> bool:
        """Image generation needs a real GPU profile and a fitting model.

        v5.5.6+: Force All / allow_oversize lets CPU-only profiles attempt
        image-gen.  This is the runtime mirror of the UI gate at
        ``_bench_method_fits_capacity`` ("image" branch).  Without this
        bypass, CPU rigs under Force All saw image-gen rows in the planning
        UI but every row was silently dropped at runtime — the v5.5.3 B1
        promise ("Force All enables image-gen on CPU rigs") was half-
        implemented before this fix.

        On CPU + Force All the smallest image-gen models (~2 GB) succeed on
        16+ CPU SKUs with adequate RAM; bigger models OOM, and the
        adaptive ``_oom_ceiling_gb`` mechanism auto-skips them once the
        first OOM/timeout/deadlock pins the wall — see
        ``_record_outcome_for_smart_skip``.

        v5.5.9: Snapdragon X (Windows ARM64) is the one
        exception to the Force-All escape hatch — torch-directml has no
        ARM64 wheel and ComfyUI's torchaudio import crashes with the
        ``torch_library_impl could not be located in _torchaudio.pyd``
        Windows popup on startup. Return False unconditionally so Force All
        skips image-gen rows on Snapdragon instead of triggering the popup.
        """
        if is_snapdragon_arm64():
            return False
        capacity_ram_gb, has_gpu, total_vram_gb, unified = self._benchmark_capacity()
        if not has_gpu:
            return self.allow_oversize
        min_vram = float(model.get("min_vram_gb") or 0)
        if min_vram <= 0:
            return True
        if total_vram_gb + 0.001 >= min_vram:
            return True
        return self.allow_oversize

    def _fits_benchmark_capacity(self, model: dict, method: str) -> tuple[bool, str]:
        """Return whether the active RAM/VRAM capacity can benchmark this method."""
        min_ram = float(model.get("min_ram_gb") or 0)
        min_vram = float(model.get("min_vram_gb") or 0)
        capacity_ram_gb, has_gpu, total_vram_gb, unified_memory = self._benchmark_capacity()

        if method == "onnx_openvino":
            if not has_gpu:
                return False, "No GPU/NPU accelerator available"
            if capacity_ram_gb < min_ram:
                if self.allow_oversize:
                    return True, "Allowed by oversize override"
                return False, f"Not enough installed RAM: need {min_ram:g} GB, have {capacity_ram_gb:.0f} GB"
            return True, "OK"

        if method in {"ollama_gpu", "onnx_directml"}:
            if not has_gpu:
                return False, "No GPU/NPU accelerator available"
            if unified_memory:
                required = max(min_ram, min_vram)
                if capacity_ram_gb < required:
                    if self.allow_oversize:
                        return True, "Allowed by oversize override"
                    return False, f"Not enough unified memory: need {required:g} GB, have {capacity_ram_gb:.0f} GB"
                return True, "OK"
            if min_vram <= 0:
                return False, "Model has no GPU memory profile"
            if total_vram_gb < min_vram:
                if self.allow_oversize:
                    return True, "Allowed by oversize override"
                return False, f"Not enough GPU VRAM: need {min_vram:g} GB, have {total_vram_gb:.0f} GB installed"
            return True, "OK"

        if capacity_ram_gb < min_ram:
            if self.allow_oversize:
                return True, "Allowed by oversize override"
            return False, f"Not enough installed RAM: need {min_ram:g} GB, have {capacity_ram_gb:.0f} GB"
        return True, "OK"

    # ── Adaptive smart-skip (v5.5.1+) ─────────────────────────────────────────
    #
    # The ceilings below tighten ONLY downward after a real failure and are
    # consulted BEFORE _run_one() spends any time downloading or loading.
    # Keeping them per-method (not per-backend or per-model) means a CUDA OOM
    # on Ollama-GPU at 13B skips Ollama-GPU at 30B/70B without blocking the
    # same model's Ollama-CPU row or any image-gen rows later in the run.

    def _record_outcome_for_smart_skip(self, model: dict, method: str, result: RunResult) -> None:
        """Tighten the per-method OOM / disk ceiling when *result* failed."""
        if result.success:
            return
        if method not in _TEXT_METHODS and method != IMAGE_METHOD:
            return
        size_gb = float(model.get("size_gb") or 0)
        if size_gb <= 0:
            return
        phase = (result.failure_phase or "").lower()
        error_text = result.error or ""
        if phase == "adaptive_skip":
            return
        # OOM-shaped: tighten the OOM ceiling for THIS method only.
        if _looks_like_oom_error(error_text) or phase in {"oom", "oom_skip"}:
            current = self._oom_ceiling_gb.get(method, float("inf"))
            self._oom_ceiling_gb[method] = min(current, size_gb)
            return
        # v5.5.6+: Image-gen runtime timeouts are treated as OOM-shaped for
        # ceiling-tightening purposes.  Rationale: a 300s timeout on an
        # image-gen sample almost always means the model is paging hard or
        # the VAE/UNet can't fit at all — the next-larger model is
        # guaranteed to be slower and just as memory-bound.  Text-method
        # timeouts are NOT treated as OOM (a 70B chat model can legitimately
        # be slow on CPU without being memory-starved; the consecutive-
        # failure counter handles that case).
        if method == IMAGE_METHOD and (
            phase == "runtime_timeout" or "timeout after" in error_text.lower()
        ):
            current = self._oom_ceiling_gb.get(method, float("inf"))
            self._oom_ceiling_gb[method] = min(current, size_gb)
            return
        # Disk-full on download: skip larger models on the SAME method to
        # avoid re-paying the failed-download tax. Smaller models on the
        # same method are still attempted because they may still fit.
        if phase == "environment_skip" and _looks_like_disk_full_text(error_text):
            current = self._disk_blocked_ceiling_gb.get(method, float("inf"))
            self._disk_blocked_ceiling_gb[method] = min(current, size_gb)

    def _smart_skip_reason(self, model: dict, method: str) -> Optional[str]:
        """Return a human reason when this model+method should be auto-skipped.

        Called from the run loop BEFORE _run_one() so the runner doesn't
        waste time on a download/load that is already known to fail. The
        result is reported with ``failure_phase="adaptive_skip"`` which is
        excluded from the consecutive-failure counter so a single dead
        backend never bails the run (matches existing
        ``environment_skip`` treatment).
        """
        size_gb = float(model.get("size_gb") or 0)
        if size_gb <= 0:
            return None
        oom_ceiling = self._oom_ceiling_gb.get(method)
        if oom_ceiling is not None and size_gb >= oom_ceiling:
            return (
                f"Skipped: {method} hit OOM at {oom_ceiling:g} GB earlier in this run; "
                f"larger same-backend models are skipped (Force-All / adaptive smart-skip)."
            )
        disk_ceiling = self._disk_blocked_ceiling_gb.get(method)
        if disk_ceiling is not None and size_gb >= disk_ceiling:
            return (
                f"Skipped: disk pressure blocked {method} at {disk_ceiling:g} GB earlier in this run; "
                f"larger same-backend pulls are skipped (adaptive smart-skip)."
            )
        return None

    def _count_combos(self, models: list[dict]) -> int:
        total = 0
        for model in models:
            for method in self._methods_for(model):
                total += len(self._iter_selected_samples_for(model, method))
        return total

    # ── Sample plumbing ────────────────────────────────────────────────────────

    def _iter_selected_samples_for(self, model: dict, method: str) -> list[tuple[int, dict, int]]:
        samples = self._iter_samples_for(model, method)
        indexed = list(enumerate(samples))
        if self.specific_combos is not None:
            requested_indexes: set[int] = set()
            for combo in self.specific_combos:
                if len(combo) < 3 or combo[0] != model.get("id") or combo[1] != method:
                    continue
                try:
                    requested_indexes.add(int(combo[2]))
                except (TypeError, ValueError):
                    continue
            if requested_indexes:
                indexed = [
                    (sample_index, sample)
                    for sample_index, sample in indexed
                    if sample_index in requested_indexes
                ]
        # v5.5.6+ Resume Today's Run: drop combos that already PASSED in a
        # prior session (failures are re-attempted — see
        # ``BatchReport.get_completed_combos``). Filtering here means both
        # _count_combos (total progress shown in the bench log) and the
        # run loop see only the not-yet-run / re-attempt combos for an
        # honest "N/M" progress display.
        if self.skip_combos:
            mid = str(model.get("id") or "")
            indexed = [
                (sample_index, sample)
                for sample_index, sample in indexed
                if (mid, method, int(sample_index)) not in self.skip_combos
            ]
        sample_count = len(samples)
        return [(sample_index, sample, sample_count) for sample_index, sample in indexed]

    def _iter_samples_for(self, model: dict, method: str) -> list[dict]:
        """Return the sample prompts to run for a model+method combo.

        Quick mode always returns one shared prompt. Extended mode returns up
        to three model-specific sample prompts from
        ``model_demos.get_model_demo`` for chat/onnx/image methods. Utility
        (phase1) keeps a single run because each adapter has its own internal
        fixture.
        """
        if self.run_mode != "extended":
            return [{
                "id": "quick",
                "title": "",
                "prompt": self.prompt,
                "source": "Quick prompt (shared across selected models)",
            }]
        if method == "phase1":
            return [{
                "id": "utility",
                "title": "",
                "prompt": self.prompt,
                "source": "Utility adapter fixture",
            }]
        try:
            demo = get_model_demo(model) or {}
        except Exception:
            demo = {}
        samples = list(demo.get("samples") or [])
        cleaned: list[dict] = []
        for idx, raw in enumerate(samples):
            text = str(raw or "").strip()
            if not text:
                continue
            title_words = text.split()[:6]
            title = " ".join(title_words)
            if len(text) > len(title):
                title = f"{title}…"
            cleaned.append({
                "id": f"{model.get('id', 'model')}-sample-{idx + 1}",
                "title": title,
                "prompt": text,
                "source": "Model-Guide.html",
            })
        if not cleaned:
            cleaned = [{
                "id": f"{model.get('id', 'model')}-quick",
                "title": "",
                "prompt": self.prompt,
                "source": "Fallback quick prompt (no sample prompt registered)",
            }]
        return cleaned

    def _sample_prompt(self) -> str:
        if self._current_sample:
            text = str(self._current_sample.get("prompt") or "").strip()
            if text:
                return text
        return self.prompt

    # ── Single run ─────────────────────────────────────────────────────────────

    def _run_one(self, model: dict, method: str) -> RunResult:
        """Execute a single benchmark run inside a timeout wrapper."""
        stop_event = threading.Event()
        self._active_stop_event = stop_event

        if method.startswith("ollama_"):
            try:
                return self._run_inner(model, method, stop_event)
            except Exception as e:
                stop_event.set()
                return RunResult(
                    model_id=model["id"],
                    model_name=model["name"],
                    method=method,
                    success=False,
                    error=str(e),
                    failure_phase="runtime_error",
                    system_snapshot=self._system_snapshot(),
                )
            finally:
                if self._active_stop_event is stop_event:
                    self._active_stop_event = None

        def _inner():
            return self._run_inner(model, method, stop_event)

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_inner)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeout:
                stop_event.set()
                return RunResult(
                    model_id=model["id"],
                    model_name=model["name"],
                    method=method,
                    success=False,
                    error=f"Timeout after {self.timeout}s",
                    failure_phase="runtime_timeout",
                    system_snapshot=self._system_snapshot(),
                )
            except Exception as e:
                stop_event.set()
                return RunResult(
                    model_id=model["id"],
                    model_name=model["name"],
                    method=method,
                    success=False,
                    error=str(e),
                    failure_phase="runtime_error",
                    system_snapshot=self._system_snapshot(),
                )
        finally:
            if self._active_stop_event is stop_event:
                self._active_stop_event = None
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_inner(self, model: dict, method: str, stop_event: threading.Event) -> RunResult:
        """Core logic for a single model + method benchmark."""
        # Pre-flight resource check
        if method in {"ollama_gpu", "ollama_cpu", "onnx_openvino", "onnx_directml", "onnx_cpu", "phase1"}:
            ok, reason = self._fits_benchmark_capacity(model, method)
            if not ok:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False, error=f"Skipped: {reason}",
                    system_snapshot=self._system_snapshot(),
                )

        try:
            if method.startswith("ollama_"):
                return self._run_ollama(model, method, stop_event)
            elif method == "phase1":
                return self._run_phase1(model, method, stop_event)
            elif method.startswith("onnx_"):
                return self._run_onnx(model, method, stop_event)
            elif method == IMAGE_METHOD:
                return self._run_image_comfyui(model, method, stop_event)
            else:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False, error=f"Unknown method: {method}",
                )
        except Exception as e:
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False, error=str(e),
                failure_phase="runtime_error",
                system_snapshot=self._system_snapshot(),
            )

    # ── Ollama runs ────────────────────────────────────────────────────────────

    def _wait_for_ollama_running(self, stop_event: threading.Event, timeout_s: float = 60.0) -> bool:
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        announced = False
        while not stop_event.is_set():
            if self.ollama.is_running():
                return True
            if time.perf_counter() >= deadline:
                break
            if not announced:
                print("\n  waiting for Ollama to come back ... ", end="", flush=True)
                announced = True
            time.sleep(2.0)
        return False

    def _run_phase1(self, model: dict, method: str, stop_event: threading.Event) -> RunResult:
        if stop_event.is_set():
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False, error="Stopped",
                system_snapshot=self._system_snapshot(),
            )
        wall_start = time.perf_counter()
        result, log_path = self._run_with_captured_logs(
            model,
            method,
            lambda: phase1_adapters.run_transformers_adapter(model, self.output_dir),
        )
        wall_total = time.perf_counter() - wall_start
        if result.get("status") != "ok":
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False,
                error=result.get("error") or result.get("output_text") or "Utility adapter failed",
                response_text=result.get("output_text", ""),
                total_time=wall_total,
                generation_time=wall_total,
                failure_phase="runtime_error",
                log_path=log_path,
                system_snapshot=self._system_snapshot(),
            )
        return RunResult(
            model_id=model["id"],
            model_name=model["name"],
            method=method,
            success=True,
            response_text=result.get("output_text", ""),
            total_time=wall_total,
            generation_time=wall_total,
            metric_kind="utility",
            metric_label=result.get("metric_label", "Elapsed"),
            metric_value=result.get("metric_value") or f"{wall_total:.1f}s",
            prompt=result.get("test_prompt", ""),
            log_path=log_path,
            system_snapshot=self._system_snapshot(),
        )

    @staticmethod
    def _probe_registry_dns(
        host: str = "registry.ollama.ai", timeout_seconds: float = 3.0,
    ) -> tuple[bool, str]:
        """Cheap socket-level DNS probe for the Ollama model registry.

        Returns ``(True, "")`` if name resolution succeeds, else
        ``(False, reason)``. Pure DNS — does not open a TCP connection.

        v5.5.12: Introduced after the May 2026 5h 07m Mac run where 94/121
        failures were sequential ``ollama pull`` timeouts against a host
        whose DNS resolver was dead the entire time. Probing once at the
        start lets ``_run_ollama`` fast-skip download-requiring rows with
        ``environment_skip`` instead of paying ~3 min per row.
        """
        import socket
        prev = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout_seconds)
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            return True, ""
        except (socket.gaierror, socket.timeout, OSError) as exc:
            reason = str(exc).strip() or exc.__class__.__name__
            return False, reason
        finally:
            socket.setdefaulttimeout(prev)

    def _is_registry_reachable(self) -> tuple[bool, str]:
        """Cached wrapper around ``_probe_registry_dns``. Probes once per
        ``BatchRunner`` lifetime and reuses the result for every
        download-requiring Ollama row in the same run."""
        if self._registry_dns_cache is None:
            self._registry_dns_cache = self._probe_registry_dns()
        return self._registry_dns_cache

    @staticmethod
    def _is_network_unreachable_error_text(text: str) -> bool:
        """Recognise a transient network-down error so the runner can
        downgrade a mid-run ``ollama pull`` failure from
        ``download_failed`` (which counts against the consecutive-failure
        ceiling and trips an early bail) to ``environment_skip`` (which
        does not, and lets the rest of the catalog still run on any
        pre-pulled models).

        Matches the actual Mac-failure substrings from the 2026-05-23 run:
        ``dial tcp: lookup ... no such host`` (NXDOMAIN), ``max retries
        exceeded`` (R2 blob download died mid-transfer), and the obvious
        ``connection refused`` / ``network is unreachable`` / ``temporary
        failure in name resolution`` variants the Ollama daemon emits on
        Linux/macOS when the host has no Internet at all.
        """
        if not text:
            return False
        t = text.lower()
        needles = (
            "no such host",
            "temporary failure in name resolution",
            "name resolution failure",
            "network is unreachable",
            "no route to host",
            "connection refused",
            "max retries exceeded",
            "dns lookup",
            "could not resolve host",
            "nodename nor servname provided",
        )
        return any(n in t for n in needles)

    def _run_ollama(self, model: dict, method: str, stop_event: threading.Event) -> RunResult:
        tag = model["ollama_tag"]
        # v5.3.6+: count every Ollama row we attempt so the end-of-run
        # banner ("All Ollama models skipped — address disk pressure")
        # can compare attempts against environment_skip count.
        self._ollama_attempt_count += 1

        # Ollama can briefly disappear while a previous large model unloads or
        # the daemon restarts under memory pressure. Wait before blaming a row.
        if not self._wait_for_ollama_running(stop_event):
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False, error="Ollama is not running",
                failure_phase="runtime_error",
            )

        self._unload_running_ollama_models()
        was_local = self._is_ollama_tag_local(tag)

        # v5.5.12: Network pre-flight short-circuit. The May 2026 Mac M4
        # run logged 94/121 failures as ~3-minute sequential ``ollama
        # pull`` timeouts against ``registry.ollama.ai`` returning
        # NXDOMAIN. If the model isn't already local AND DNS for the
        # registry can't even be resolved, skip with ``environment_skip``
        # — preserving the right to still run any pre-pulled models in
        # the same batch and avoiding ~5h of timeouts. Pre-pulled rows
        # (``was_local``) bypass this gate entirely because ``ollama
        # pull`` is a no-op for an already-cached tag.
        if not was_local:
            reachable, dns_reason = self._is_registry_reachable()
            if not reachable:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error=(
                        f"Skipped: model not pre-pulled and registry."
                        f"ollama.ai DNS unreachable ({dns_reason}). "
                        f"Run 'ollama pull {tag}' on a network-up "
                        "machine first, or re-run once the connection "
                        "is back."
                    ),
                    failure_phase="environment_skip",
                    warm_cache=False,
                    system_snapshot=self._system_snapshot(),
                )

        # Constrained-cloud-VM (GPU) pre-pull disk-space
        # gate.  If the Ollama models directory cannot fit this model we
        # skip with ``failure_phase="environment_skip"`` so the run
        # continues with smaller models — disk pressure is a fixable
        # environment issue, not a model-bug, and counting it against
        # ``_consecutive_failures`` would bail the whole benchmark after
        # 5 models even when smaller ones would still fit.  Only runs
        # when the model isn't already local (we don't re-check disk for
        # a noop pull) and only when we have a real size estimate.
        if not was_local and float(model.get("size_gb", 0) or 0) > 0:
            skip_reason = constrained_env.precheck_ollama_pull(model.get("size_gb", 0))
            if skip_reason:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error=f"Skipped: {skip_reason}",
                    failure_phase="environment_skip",
                    warm_cache=was_local,
                    system_snapshot=self._system_snapshot(),
                )

        # Low Resources Mode: check disk before download, RAM before run.
        # v5.5.4: Force-All bypasses the live-RAM gate because available RAM
        # is noisy on cloud VMs (Ollama may pin prior models, Defender may be
        # scanning) and produces false skips like the 8 CPU run where 25/66
        # cases were skipped with "Insufficient RAM: need X GB, only Y GB
        # available" even though the host had 32 GB total. Under Force-All
        # we let the real Ollama load OOM (which the runner classifies as
        # ``ram_oom`` and uses to tighten the same-method ceiling) instead
        # of pre-rejecting on stale psutil readings.  When the gate DOES
        # fire (low_resources_mode + NOT force_all), the synthesized
        # RunResult is labelled ``environment_skip`` so the smart-skip
        # ceiling treats it as a real environment limit and prunes larger
        # same-method models -- matching the behaviour of the inline gate
        # at line ~1380.
        if self.low_resources_mode and not self.force_all:
            # Check RAM
            ok, reason = resource_manager.check_ram_for_model(model)
            if not ok:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False, error=f"Skipped: {reason}",
                    failure_phase="environment_skip",
                    system_snapshot=self._system_snapshot(),
                )
            # Check disk and free space if needed
            ok, reason = resource_manager.check_disk_for_download(
                model.get("size_gb", 0), self.models_dir,
            )
            if not ok:
                freed, self._purge_done = resource_manager.free_space_for_download(
                    model.get("size_gb", 0), self.ollama,
                    self.models_dir, self._selected_batch_tags,
                    purge_done=self._purge_done,
                    comfyui_client=self._comfyui_client,
                )
                if not freed:
                    return RunResult(
                        model_id=model["id"], model_name=model["name"],
                        method=method, success=False,
                        error=f"Skipped: insufficient disk space",
                        system_snapshot=self._system_snapshot(),
                    )

        # Always pull to guarantee this exact tag is local.
        # (Ollama pull is a fast no-op if already downloaded.)
        print("pulling... ", end="", flush=True)
        logger.info(
            f"Benchmark pull started: {tag} ({method})",
            category=logger.CATEGORY_MODEL_PULL,
        )
        download_start = time.perf_counter()
        pull_progress = {"bucket": None, "status": "", "time": 0.0}

        def _pull_progress_cb(status: str, completed: int, total: int) -> None:
            now = time.perf_counter()
            pct_text = ""
            bucket = pull_progress["bucket"]
            if total and total > 0:
                pct = max(0.0, min(100.0, completed / total * 100.0))
                bucket = int(pct // 5) * 5
                if completed >= total:
                    bucket = 100
                pct_text = f"{pct:.0f}% ({_fmt_bytes(completed)} / {_fmt_bytes(total)})"
            status_changed = bool(status) and status != pull_progress["status"]
            should_emit = status_changed or bucket != pull_progress["bucket"] or now - pull_progress["time"] >= 15.0
            if not should_emit:
                return
            pull_progress.update({"bucket": bucket, "status": status, "time": now})
            msg = f"{tag}: {status or 'pulling'} {pct_text}".strip()
            print(f"\n  {msg}", end="", flush=True)
            logger.info(
                f"Benchmark pull progress: {msg}",
                category=logger.CATEGORY_MODEL_PULL,
            )

        try:
            self.ollama.pull_model(tag, progress_cb=_pull_progress_cb, stop_event=stop_event)
        except OllamaError as exc:
            # Constrained-aware error wrapping.  If the Ollama daemon
            # reported a disk-pressure error AND we're on a cloud VM,
            # ``profile_aware_ollama_error`` prepends the OLLAMA_MODELS
            # relocation hint so the operator sees something actionable in the
            # benchmark log instead of a bare "no space left on device".
            # Disk-pressure errors are reclassified as
            # ``environment_skip`` so they don't count against
            # ``_consecutive_failures``.
            raw_error = str(exc)
            mapped_error = constrained_env.profile_aware_ollama_error(raw_error)
            # v5.5.12: A pull failure that's clearly a network outage
            # (NXDOMAIN, connection refused, no route to host, max
            # retries exceeded, ...) is reclassified as
            # ``environment_skip`` so it doesn't count against the
            # consecutive-failure ceiling and bail the entire run.  The
            # pre-flight ``_is_registry_reachable`` gate catches the
            # most common case up front, but DNS can flap mid-run (or
            # the R2 blob CDN can stall after the manifest succeeds —
            # the May 2026 Mac run had 3/94 such mid-transfer kills) so
            # we also fall back to error-text matching here.
            phase = (
                "environment_skip"
                if (
                    constrained_env.is_disk_full_error_text(raw_error)
                    or self._is_network_unreachable_error_text(raw_error)
                )
                else "download_failed"
            )
            logger.error(
                f"Benchmark pull failed: {tag}: {mapped_error}",
                category=logger.CATEGORY_MODEL_PULL,
            )
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error=mapped_error,
                failure_phase=phase,
                download_time=time.perf_counter() - download_start,
                warm_cache=was_local,
                system_snapshot=self._system_snapshot(),
            )
        download_time = time.perf_counter() - download_start
        print(f"\n  pull complete in {download_time:.1f}s. ", end="", flush=True)
        logger.info(
            f"Benchmark pull complete: {tag} in {download_time:.1f}s (warm_cache={was_local})",
            category=logger.CATEGORY_MODEL_PULL,
        )
        if not was_local:
            self._downloaded_ollama_tags.add(tag)
        if stop_event.is_set():
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error="Stopped",
                failure_phase="stopped",
                download_time=download_time,
                warm_cache=was_local,
                system_snapshot=self._system_snapshot(),
            )

        num_gpu = -1 if method == "ollama_gpu" else 0
        prompt_text = self._sample_prompt()
        messages = [{"role": "user", "content": prompt_text}]
        options = {"temperature": 0.0, "num_gpu": num_gpu, "num_predict": self._num_predict_for(model, method)}
        # Repetition-loop guards (Ollama options). Defaults are activated in
        # extended mode so the broad benchmark matrix is resilient to the
        # MiniCPM-V-style fake-roleplay loops that burn the entire token
        # budget. Per-model overrides via the catalog keys
        # ``benchmark_repeat_penalty`` and ``benchmark_repeat_last_n`` win
        # when set to a positive value. NOTE: a non-positive value
        # (``0`` / ``-1`` / ``false``) is treated as "absent" and falls
        # back to the extended-mode defaults — there is currently no real
        # opt-out for extended mode. To bypass the guard for a model, set
        # ``benchmark_repeat_penalty`` to a positive value close to ``1.0``
        # (e.g. ``1.01``) rather than ``0``.
        repeat_penalty = _positive_float(model.get("benchmark_repeat_penalty"))
        repeat_last_n = _positive_int(model.get("benchmark_repeat_last_n"))
        if self.run_mode == "extended":
            if repeat_penalty is None:
                repeat_penalty = 1.15
            if repeat_last_n is None:
                repeat_last_n = 256
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if repeat_last_n is not None:
            options["repeat_last_n"] = repeat_last_n
        generation_timeout = self._ollama_generation_timeout_for(model, method, options["num_predict"])
        options["timeout_s"] = generation_timeout

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(
                self._run_ollama_generation,
                model,
                method,
                tag,
                messages,
                options,
                stop_event,
                download_time,
                was_local,
            )
            try:
                return future.result(timeout=generation_timeout)
            except FuturesTimeout:
                stop_event.set()
                return RunResult(
                    model_id=model["id"],
                    model_name=model["name"],
                    method=method,
                    success=False,
                    error=f"Generation timeout after {generation_timeout}s",
                    failure_phase="generation_timeout",
                    download_time=download_time,
                    warm_cache=was_local,
                    prompt=self._sample_prompt(),
                    options=options,
                    system_snapshot=self._system_snapshot(),
                )
            except Exception as exc:
                stop_event.set()
                return RunResult(
                    model_id=model["id"],
                    model_name=model["name"],
                    method=method,
                    success=False,
                    error=str(exc),
                    failure_phase="runtime_error",
                    download_time=download_time,
                    warm_cache=was_local,
                    prompt=self._sample_prompt(),
                    options=options,
                    system_snapshot=self._system_snapshot(),
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_ollama_generation(
        self,
        model: dict,
        method: str,
        tag: str,
        messages: list[dict],
        options: dict,
        stop_event: threading.Event,
        download_time: float,
        was_local: bool,
    ) -> RunResult:
        num_gpu = options["num_gpu"]
        num_predict = options["num_predict"]
        read_timeout = max(600, int(options.get("timeout_s", self.timeout)))
        repeat_penalty = options.get("repeat_penalty")
        repeat_last_n = options.get("repeat_last_n")

        wall_start = time.perf_counter()
        first_token_wall = None
        tokens = []
        last_progress = wall_start
        repetition_detected = False
        tokens_since_repetition_check = 0
        logger.info(
            f"Benchmark generation started: {tag} ({method})",
            category=logger.CATEGORY_BENCHMARK,
        )

        stats = {}
        for token, chunk_stats in self.ollama.chat_stream_with_stats(
            tag,
            messages,
            num_gpu=num_gpu,
            temperature=options["temperature"],
            stop_event=stop_event,
            num_predict=num_predict,
            read_timeout=read_timeout,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
        ):
            if first_token_wall is None and token:
                first_token_wall = time.perf_counter()
                print(f"\n  first token in {first_token_wall - wall_start:.2f}s", end="", flush=True)
                logger.info(
                    f"Benchmark first token: {tag} ({method}) after {first_token_wall - wall_start:.2f}s",
                    category=logger.CATEGORY_BENCHMARK,
                )
            if token:
                tokens.append(token)
                now = time.perf_counter()
                if now - last_progress >= 15.0:
                    chars = len("".join(tokens))
                    print(f"\n  generating... {chars} chars in {now - wall_start:.0f}s", end="", flush=True)
                    logger.info(
                        f"Benchmark generation progress: {tag} ({method}), {chars} chars in {now - wall_start:.0f}s",
                        category=logger.CATEGORY_BENCHMARK,
                    )
                    last_progress = now
                tokens_since_repetition_check += 1
                if tokens_since_repetition_check >= _REPETITION_STREAM_CHECK_EVERY:
                    tokens_since_repetition_check = 0
                    if _looks_like_repetition_loop("".join(tokens)):
                        repetition_detected = True
                        print(
                            "\n  repetition loop detected — stopping early",
                            end="", flush=True,
                        )
                        logger.warning(
                            f"Benchmark repetition loop detected: {tag} ({method}) "
                            f"after {len(tokens)} tokens",
                            category=logger.CATEGORY_BENCHMARK,
                        )
                        break
            if chunk_stats is not None:
                stats = chunk_stats

        wall_total = time.perf_counter() - wall_start
        response_text = strip_think_blocks("".join(tokens)).strip()
        if repetition_detected:
            # The streaming detector broke out BEFORE the final ``done=true``
            # chunk arrived, so the Ollama-native eval_count/eval_duration
            # stats are missing. Fall back to wall-clock derivations so the
            # HTML matrix row still carries meaningful TTFT + tokens-per-sec
            # data instead of suspicious zeros that look like a measurement
            # bug. ``ollama_*_ns`` and ``load_time`` stay 0 — they have no
            # honest wall-clock approximation, and report renderers handle
            # 0 gracefully.
            eval_dur_s_loop = stats.get("eval_duration", 0) / 1e9
            eval_count_loop = stats.get("eval_count", 0) or len(tokens)
            if eval_dur_s_loop > 0:
                tps_loop = eval_count_loop / eval_dur_s_loop
            elif wall_total > 0 and eval_count_loop > 0:
                tps_loop = eval_count_loop / wall_total
            else:
                tps_loop = 0.0
            load_s_loop = stats.get("load_duration", 0) / 1e9
            ttft_loop = (first_token_wall - wall_start) if first_token_wall else wall_total
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error=(
                    f"Model entered a repetition loop after {len(tokens)} tokens "
                    f"({len(response_text)} chars); stopped early. The benchmark "
                    f"token budget was {num_predict}. This often indicates a "
                    "meta-prompt that primes self-roleplay or a missing "
                    "repeat_penalty for this model."
                ),
                failure_phase="repetition_loop",
                response_text=response_text,
                total_time=wall_total,
                ttft=ttft_loop,
                load_time=load_s_loop,
                download_time=download_time,
                generation_time=wall_total,
                warm_cache=was_local,
                prompt=self._sample_prompt(),
                options=dict(options),
                tokens_per_sec=tps_loop,
                token_count=eval_count_loop,
                ollama_total_duration_ns=stats.get("total_duration", 0),
                ollama_load_duration_ns=stats.get("load_duration", 0),
                ollama_eval_duration_ns=stats.get("eval_duration", 0),
                ollama_eval_count=stats.get("eval_count", 0),
                system_snapshot=self._system_snapshot(),
            )
        if stop_event.is_set():
            logger.warning(
                f"Benchmark generation stopped: {tag} ({method}) after {wall_total:.1f}s",
                category=logger.CATEGORY_BENCHMARK,
            )
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error="Stopped",
                failure_phase="stopped",
                response_text=response_text,
                total_time=wall_total,
                download_time=download_time,
                generation_time=wall_total,
                warm_cache=was_local,
                prompt=self._sample_prompt(),
                options=options,
                system_snapshot=self._system_snapshot(),
            )

        # Derive timing from Ollama's native nanosecond stats
        eval_dur_s = stats.get("eval_duration", 0) / 1e9
        eval_count = stats.get("eval_count", 0)
        tps = eval_count / eval_dur_s if eval_dur_s > 0 else 0.0
        load_s = stats.get("load_duration", 0) / 1e9
        prompt_s = stats.get("prompt_eval_duration", 0) / 1e9
        done_reason = str(stats.get("done_reason") or "").strip()
        result_options = dict(options)
        if done_reason:
            result_options["done_reason"] = done_reason
        ttft = (first_token_wall - wall_start) if first_token_wall else wall_total
        timing_msg = (
            f"timing load={load_s:.2f}s ttft={ttft:.2f}s "
            f"prompt={prompt_s:.2f}s eval={eval_dur_s:.2f}s "
            f"tokens={eval_count or len(tokens)} tps={tps:.2f}"
            + (f" done={done_reason}" if done_reason else "")
        )
        print(f"\n  {timing_msg}", end="", flush=True)
        logger.info(
            f"Benchmark complete: {tag} ({method}) {timing_msg}",
            category=logger.CATEGORY_BENCHMARK,
        )

        if not response_text.strip():
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error="Model returned an empty response.",
                failure_phase="empty_response",
                response_text=response_text,
                total_time=wall_total,
                ttft=ttft,
                load_time=load_s,
                download_time=download_time,
                generation_time=wall_total,
                warm_cache=was_local,
                prompt=self._sample_prompt(),
                options=result_options,
                tokens_per_sec=tps,
                token_count=eval_count or len(tokens),
                ollama_total_duration_ns=stats.get("total_duration", 0),
                ollama_load_duration_ns=stats.get("load_duration", 0),
                ollama_eval_duration_ns=stats.get("eval_duration", 0),
                ollama_eval_count=eval_count,
                system_snapshot=self._system_snapshot(),
            )

        if _looks_like_unstripped_hidden_reasoning(response_text):
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error="Model returned hidden reasoning/meta-analysis instead of a benchmark answer.",
                failure_phase="hidden_reasoning_response",
                response_text=response_text,
                total_time=wall_total,
                ttft=ttft,
                load_time=load_s,
                download_time=download_time,
                generation_time=wall_total,
                warm_cache=was_local,
                prompt=self._sample_prompt(),
                options=result_options,
                tokens_per_sec=tps,
                token_count=eval_count or len(tokens),
                ollama_total_duration_ns=stats.get("total_duration", 0),
                ollama_load_duration_ns=stats.get("load_duration", 0),
                ollama_eval_duration_ns=stats.get("eval_duration", 0),
                ollama_eval_count=eval_count,
                system_snapshot=self._system_snapshot(),
            )

        if done_reason == "length":
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error=(
                    f"Response hit the benchmark token budget "
                    f"({num_predict} tokens) before Ollama emitted a stop reason; "
                    "increase benchmark_num_predict for this model or shorten the sample prompt."
                ),
                failure_phase="output_truncated",
                response_text=response_text,
                total_time=wall_total,
                ttft=ttft,
                load_time=load_s,
                download_time=download_time,
                generation_time=wall_total,
                warm_cache=was_local,
                prompt=self._sample_prompt(),
                options=result_options,
                tokens_per_sec=tps,
                token_count=eval_count or len(tokens),
                ollama_total_duration_ns=stats.get("total_duration", 0),
                ollama_load_duration_ns=stats.get("load_duration", 0),
                ollama_eval_duration_ns=stats.get("eval_duration", 0),
                ollama_eval_count=eval_count,
                system_snapshot=self._system_snapshot(),
            )

        return RunResult(
            model_id=model["id"],
            model_name=model["name"],
            method=method,
            success=True,
            response_text=response_text,
            total_time=wall_total,
            ttft=ttft,
            load_time=load_s,
            download_time=download_time,
            generation_time=wall_total,
            warm_cache=was_local,
            prompt=self._sample_prompt(),
            options=result_options,
            tokens_per_sec=tps,
            token_count=eval_count or len(tokens),
            ollama_total_duration_ns=stats.get("total_duration", 0),
            ollama_load_duration_ns=stats.get("load_duration", 0),
            ollama_eval_duration_ns=stats.get("eval_duration", 0),
            ollama_eval_count=eval_count,
            system_snapshot=self._system_snapshot(),
        )

    # ── ONNX runs ──────────────────────────────────────────────────────────────

    def _run_onnx(self, model: dict, method: str, stop_event: threading.Event) -> RunResult:
        repo = model.get("onnx_repo")
        subfolder = model.get("onnx_subfolder", "")
        if not repo:
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False, error="No onnx_repo defined",
            )

        if not ONNX_AVAILABLE:
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False, error="onnxruntime not installed",
            )

        if method == "onnx_openvino" and OPENVINO_AVAILABLE:
            provider, provider_options = "OpenVINOExecutionProvider", {"device_type": "NPU"}
            subfolder = model.get("onnx_openvino_subfolder") or subfolder
        elif method == "onnx_directml" and DIRECTML_AVAILABLE:
            provider, provider_options = "DmlExecutionProvider", {}
        else:
            provider, provider_options = "CPUExecutionProvider", {}
            # Phi-4-style bundles ship a dedicated CPU subfolder that the
            # genai runtime knows how to load; prefer it over the GPU
            # subfolder when running CPU benchmarks.
            cpu_sub = model.get("onnx_cpu_subfolder") or model.get("onnx_openvino_subfolder")
            if cpu_sub:
                subfolder = cpu_sub

        # Download if needed (check that the specific subfolder exists)
        model_dir = self.models_dir / model["id"].replace(":", "_")
        subfolder_path = model_dir / subfolder if subfolder else model_dir
        if not subfolder_path.exists():
            print("downloading... ", end="", flush=True)
            if not HF_AVAILABLE:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error='huggingface_hub not installed — run: pip install "huggingface-hub>=0.34.0,<1.0"',
                )
            download_onnx_model(repo, subfolder, model_dir, stop_event=stop_event)

        # Dispatch: Phi-4 ONNX bundles (and other genai-packaged
        # models) ship a genai_config.json in the model subfolder. Those must
        # use onnxruntime-genai — optimum.onnxruntime's KV-cache handling does
        # not understand their Grouped-Query-Attention layout and crashes with
        # ``past_key_values.N.value: Got: <attn_heads> Expected: <kv_heads>``.
        use_genai = has_genai_config(model_dir, subfolder)

        if use_genai:
            if not GENAI_AVAILABLE:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error=(
                        "Model requires onnxruntime-genai (genai_config.json "
                        "present), but the package is not installed. "
                        "Run: pip install onnxruntime-genai-directml"
                    ),
                )
            if method == "onnx_openvino":
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error=(
                        "Skipped: onnx_openvino is not supported for "
                        "genai-packaged models with onnxruntime-genai-directml. "
                        "Use onnx_directml or onnx_cpu, or install an "
                        "OpenVINO-enabled genai variant."
                    ),
                )
            session = OnnxGenAISession(model_dir, subfolder=subfolder)
        else:
            session = OnnxModelSession(
                model_dir, provider, provider_options, subfolder=subfolder,
            )

        tokens = []
        stats = {}
        prompt_text = self._sample_prompt()
        max_new_tokens = _positive_int(model.get("benchmark_num_predict")) or 512
        for token, chunk_stats in session.generate_stream_timed(
            prompt_text, max_new_tokens=max_new_tokens, temperature=0.0, stop_event=stop_event,
        ):
            if token:
                tokens.append(token)
            if chunk_stats is not None:
                stats = chunk_stats

        response_text = strip_think_blocks("".join(tokens)).strip()
        token_count = stats.get("token_count", 0)
        if not response_text:
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error="Model returned an empty response.",
                failure_phase="empty_response",
                response_text=response_text,
                total_time=stats.get("total_time", 0.0),
                ttft=stats.get("ttft", 0.0),
                tokens_per_sec=stats.get("tokens_per_sec", 0.0),
                token_count=token_count,
                prompt=prompt_text,
                system_snapshot=self._system_snapshot(),
            )
        if token_count < max_new_tokens and (len(response_text.split()) < 5 or len(response_text) < 24):
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error="Model returned a response too short to be a useful benchmark answer.",
                failure_phase="low_quality_response",
                response_text=response_text,
                total_time=stats.get("total_time", 0.0),
                ttft=stats.get("ttft", 0.0),
                tokens_per_sec=stats.get("tokens_per_sec", 0.0),
                token_count=token_count,
                prompt=prompt_text,
                system_snapshot=self._system_snapshot(),
            )
        if token_count >= max_new_tokens:
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error=(
                    f"Response hit the benchmark token budget "
                    f"({max_new_tokens} tokens); increase the ONNX benchmark "
                    "budget or shorten the sample prompt."
                ),
                failure_phase="output_truncated",
                response_text=response_text,
                total_time=stats.get("total_time", 0.0),
                ttft=stats.get("ttft", 0.0),
                tokens_per_sec=stats.get("tokens_per_sec", 0.0),
                token_count=token_count,
                prompt=prompt_text,
                system_snapshot=self._system_snapshot(),
            )
        if _looks_like_unstripped_hidden_reasoning(response_text):
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error="Model returned hidden reasoning/meta-analysis instead of a benchmark answer.",
                failure_phase="hidden_reasoning_response",
                response_text=response_text,
                total_time=stats.get("total_time", 0.0),
                ttft=stats.get("ttft", 0.0),
                tokens_per_sec=stats.get("tokens_per_sec", 0.0),
                token_count=token_count,
                prompt=prompt_text,
                system_snapshot=self._system_snapshot(),
            )

        return RunResult(
            model_id=model["id"],
            model_name=model["name"],
            method=method,
            success=True,
            response_text=response_text,
            total_time=stats.get("total_time", 0.0),
            ttft=stats.get("ttft", 0.0),
            tokens_per_sec=stats.get("tokens_per_sec", 0.0),
            token_count=token_count,
            prompt=prompt_text,
            system_snapshot=self._system_snapshot(),
        )

    # ── Image generation runs ──────────────────────────────────────────────────

    def _record_image_failure_reason(self, reason: str) -> None:
        """Cache an image-gen startup failure reason so subsequent iterations
        of this batch fail fast instead of re-paying the cold-start cost.

        Thread-safe; idempotent (first failure wins so the user sees the
        original root cause rather than a downstream symptom).
        """
        if not reason:
            return
        with self._image_failure_cache_lock:
            if self._image_global_failure_reason is None:
                self._image_global_failure_reason = reason

    def _cached_image_failure_reason(self) -> Optional[str]:
        with self._image_failure_cache_lock:
            return self._image_global_failure_reason

    def _reset_image_failure_cache(self) -> None:
        """Clear the cached startup-failure reason.  Used by tests."""
        with self._image_failure_cache_lock:
            self._image_global_failure_reason = None

    @staticmethod
    def _normalise_ensure_result(raw, default_failure_reason: str) -> tuple[bool, str]:
        """Normalise the ensure-callback return value to a `(bool, reason)` pair.

        Callbacks may legitimately return:
          * ``True`` / ``False`` (legacy contract — pre-cold-start-fix)
          * ``(True, "")`` (new contract: success)
          * ``(False, "specific failure reason")`` (new contract: failure)

        Anything else (3-tuple, list, non-bool truthy object, None) is treated
        defensively: if the first element looks like a bool we use it; otherwise
        we collapse to ``(bool(raw), default_failure_reason)`` so a misbehaving
        callback can't silently report success-with-failure-reason.

        Reason strings are sanitised: newlines/CR/tabs → single space, trimmed
        to 240 chars so the bench-log truncation never chops mid-word AND a
        runaway pip traceback can't take over the GUI textbox.
        """
        ok: bool
        reason: str
        if (
            isinstance(raw, tuple)
            and len(raw) == 2
            and isinstance(raw[0], bool)
        ):
            ok = raw[0]
            reason = str(raw[1] or "")
        elif isinstance(raw, bool):
            ok = raw
            reason = "" if ok else default_failure_reason
        else:
            ok = bool(raw)
            reason = "" if ok else default_failure_reason
        if reason:
            reason = " ".join(reason.split())
            if len(reason) > 240:
                reason = reason[:237] + "…"
        return ok, reason

    def _ensure_comfyui_running_for_run(
        self,
        timeout: int = COMFYUI_COLD_START_TIMEOUT_S,
        model: dict | None = None,
        trust_existing_running: bool = False,
    ) -> tuple[bool, str]:
        """Return ``(True, "")`` when the ComfyUI HTTP API is responsive,
        otherwise ``(False, reason)`` where ``reason`` is the most specific
        message available so the bench log surfaces *why* the start failed
        instead of the generic "ComfyUI is not running and could not be
        started" placeholder.

        Strategy:
        1. Quick probe: if the client says it's running and the current model
           has no special launch flags, return True without starting anything.
           Models such as SDXL Low VRAM still call the host callback so the app
           can verify/restart with required flags before generation.
        2. If a host-provided start callback is wired (``_ensure_comfyui_ready``),
           call it with the requested timeout.  The host is responsible for
           launching the subprocess and waiting for the API to come up.  The
           callback may return ``bool`` (legacy) OR ``(bool, reason)``; both
           are accepted.
        3. If no callback is wired (e.g. headless ``run_batch.py``), return
           a clean failure with that as the reason so the caller fails this
           image-gen run cleanly.
        """
        client = self._comfyui_client
        if client is None:
            return False, "No ComfyUI client provided to BatchRunner"
        try:
            if client.is_running():
                if (
                    not trust_existing_running
                    and self._ensure_comfyui_ready is not None
                    and model
                    and model.get("comfyui_launch_flags")
                ):
                    try:
                        raw = self._call_ensure_comfyui_ready(timeout, model)
                    except Exception as exc:
                        return False, f"ComfyUI start callback raised: {exc}"
                    return self._normalise_ensure_result(
                        raw, default_failure_reason=LEGACY_COMFYUI_START_FAILURE_REASON
                    )
                return True, ""
        except Exception:
            pass
        if self._ensure_comfyui_ready is None:
            return (
                False,
                "ComfyUI is not running and no start callback is wired "
                "(headless run_batch.py mode — start ComfyUI manually first)",
            )
        try:
            raw = self._call_ensure_comfyui_ready(timeout, model)
        except Exception as exc:
            return False, f"ComfyUI start callback raised: {exc}"
        return self._normalise_ensure_result(
            raw, default_failure_reason=LEGACY_COMFYUI_START_FAILURE_REASON
        )

    def _call_ensure_comfyui_ready(self, timeout: int, model: dict | None):
        """Call the app readiness callback with model context when supported."""
        callback = self._ensure_comfyui_ready
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(timeout)
        accepts_model_keyword = False
        positional_slots = 0
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                return callback(timeout, model)
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_model_keyword = True
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional_slots += 1
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.name == "model":
                accepts_model_keyword = True
        if positional_slots >= 2:
            return callback(timeout, model)
        if accepts_model_keyword:
            return callback(timeout, model=model)
        return callback(timeout)

    def _call_prepare_image_model(self, model: dict, stop_event: threading.Event):
        """Call the optional image-model preparation callback when wired."""
        callback = self._prepare_image_model
        if callback is None:
            return True, ""
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(model)
        accepts_stop_keyword = False
        positional_slots = 0
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                return callback(model, stop_event)
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_stop_keyword = True
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional_slots += 1
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.name == "stop_event":
                accepts_stop_keyword = True
        if positional_slots >= 2:
            return callback(model, stop_event)
        if accepts_stop_keyword:
            return callback(model, stop_event=stop_event)
        return callback(model)

    @staticmethod
    def _looks_like_missing_image_model_error(message: str) -> bool:
        """Return True when ComfyUI says the checkpoint/UNet name is unavailable."""
        text = str(message or "").lower()
        if "not in list" not in text:
            return False
        markers = (
            "ckpt_name",
            "checkpointloadersimple",
            "unet_name",
            "unetloader",
            "unetloadergguf",
            "diffusion model",
            "checkpoint",
        )
        return any(marker in text for marker in markers)

    def _run_image_comfyui(
        self, model: dict, method: str, stop_event: threading.Event,
    ) -> RunResult:
        if self._comfyui_client is None:
            reason = "No ComfyUI client provided to BatchRunner"
            self._record_image_failure_reason(reason)
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False,
                error=reason,
                failure_phase="runtime_error",
                surface="image",
            )
        comfyui_model = (
            model.get("comfyui_model")
            or model.get("model_filename")
            or ""
        )
        if not comfyui_model:
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False,
                error="No comfyui_model filename defined for this catalog entry",
                surface="image",
            )

        prompt_text = self._sample_prompt()
        recommended = model.get("recommended_settings") or {}
        width = int(recommended.get("width") or 1024)
        height = int(recommended.get("height") or 1024)
        steps = int(recommended.get("steps") or 20)
        cfg = float(recommended.get("cfg") or 7.0)
        sampler = recommended.get("sampler") or "euler"
        scheduler = recommended.get("scheduler") or "normal"
        try:
            negative_default, _ = _negative_prompt_for_model(model.get("id", ""))
        except Exception:
            negative_default = ""
        negative = "" if recommended.get("cfg_locked") else (negative_default or "")
        seed = 1234567

        # FAST-FAIL: a previous image-gen run in this batch already failed to
        # bring up ComfyUI.  Don't re-pay the multi-minute cold-start cost on
        # iterations 2/3 of SDXL or on the next image model — surface the
        # cached root cause immediately so Ron isn't waiting 15+ minutes for
        # the same dep-install / not-installed / cold-timeout error to repeat.
        # If ComfyUI is somehow already running now (e.g. the user fixed it
        # mid-batch), trust the live probe and bypass the cache.
        cached_reason = self._cached_image_failure_reason()
        trust_existing_running = False
        if cached_reason:
            still_down = True
            try:
                if self._comfyui_client.is_running():
                    still_down = False
                    trust_existing_running = True
            except Exception:
                still_down = True
            if still_down:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error=f"{cached_reason} (cached from earlier image-gen run in this batch)",
                    failure_phase="runtime_error",
                    surface="image",
                    prompt=prompt_text,
                )

        try:
            raw_prepare = self._call_prepare_image_model(model, stop_event)
        except Exception as exc:
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error=f"Image model preparation failed: {exc}",
                failure_phase="environment_skip",
                surface="image",
                prompt=prompt_text,
                negative_prompt=negative,
                image_width=width, image_height=height,
                image_steps=steps, image_cfg=cfg,
                image_sampler=sampler, image_scheduler=scheduler,
                image_seed=seed,
                system_snapshot=self._system_snapshot(),
            )
        prepare_ok, prepare_reason = self._normalise_ensure_result(
            raw_prepare,
            default_failure_reason="Image model preparation failed",
        )
        if not prepare_ok:
            return RunResult(
                model_id=model["id"],
                model_name=model["name"],
                method=method,
                success=False,
                error=prepare_reason or "Image model preparation failed",
                failure_phase="environment_skip",
                surface="image",
                prompt=prompt_text,
                negative_prompt=negative,
                image_width=width, image_height=height,
                image_steps=steps, image_cfg=cfg,
                image_sampler=sampler, image_scheduler=scheduler,
                image_seed=seed,
                system_snapshot=self._system_snapshot(),
            )

        ok, fail_reason = self._ensure_comfyui_running_for_run(
            timeout=COMFYUI_COLD_START_TIMEOUT_S,
            model=model,
            trust_existing_running=trust_existing_running,
        )
        if not ok:
            reason = fail_reason or LEGACY_COMFYUI_START_FAILURE_REASON
            self._record_image_failure_reason(reason)
            return RunResult(
                model_id=model["id"], model_name=model["name"],
                method=method, success=False,
                error=reason,
                failure_phase="runtime_error",
                surface="image",
                prompt=prompt_text,
            )

        wall_start = time.perf_counter()
        try:
            png_bytes = self._comfyui_client.generate_image(
                model_filename=comfyui_model,
                positive_prompt=prompt_text,
                negative_prompt=negative,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg,
                seed=seed,
                sampler_name=sampler,
                scheduler=scheduler,
                stop_event=stop_event,
            )
        except Exception as exc:
            error_text = str(exc)
            if self._looks_like_missing_image_model_error(error_text):
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error=f"Skipped: {error_text}",
                    failure_phase="environment_skip",
                    total_time=time.perf_counter() - wall_start,
                    surface="image",
                    prompt=prompt_text,
                    negative_prompt=negative,
                    image_width=width, image_height=height,
                    image_steps=steps, image_cfg=cfg,
                    image_sampler=sampler, image_scheduler=scheduler,
                    image_seed=seed,
                    system_snapshot=self._system_snapshot(),
                )
            # If ComfyUI looks like it crashed mid-generation, try ONCE to
            # bring it back up and re-run this generation.  We only restart
            # reactively (after a failure), never proactively between runs.
            try:
                still_running = self._comfyui_client.is_running()
            except Exception:
                still_running = False
            if (
                not still_running
                and self._ensure_comfyui_ready is not None
                and not (stop_event is not None and stop_event.is_set())
            ):
                _log_msg = (
                    f"ComfyUI appears to have crashed during {model.get('id')} "
                    f"({exc}); attempting one restart + retry"
                )
                print(_log_msg)
                restart_ok, restart_reason = self._ensure_comfyui_running_for_run(
                    timeout=COMFYUI_CRASH_RESTART_TIMEOUT_S, model=model
                )
                if restart_ok:
                    try:
                        png_bytes = self._comfyui_client.generate_image(
                            model_filename=comfyui_model,
                            positive_prompt=prompt_text,
                            negative_prompt=negative,
                            width=width,
                            height=height,
                            steps=steps,
                            cfg_scale=cfg,
                            seed=seed,
                            sampler_name=sampler,
                            scheduler=scheduler,
                            stop_event=stop_event,
                        )
                    except Exception as exc2:
                        return RunResult(
                            model_id=model["id"], model_name=model["name"],
                            method=method, success=False,
                            error=f"ComfyUI crashed; retry failed: {exc2}",
                            failure_phase="runtime_error",
                            total_time=time.perf_counter() - wall_start,
                            surface="image",
                            prompt=prompt_text,
                            negative_prompt=negative,
                            image_width=width, image_height=height,
                            image_steps=steps, image_cfg=cfg,
                            image_sampler=sampler, image_scheduler=scheduler,
                            image_seed=seed,
                        )
                else:
                    crash_reason = (
                        f"ComfyUI crashed and could not be restarted: {restart_reason}"
                        if restart_reason
                        else "ComfyUI crashed and could not be restarted"
                    )
                    self._record_image_failure_reason(crash_reason)
                    return RunResult(
                        model_id=model["id"], model_name=model["name"],
                        method=method, success=False,
                        error=crash_reason,
                        failure_phase="runtime_error",
                        total_time=time.perf_counter() - wall_start,
                        surface="image",
                        prompt=prompt_text,
                        negative_prompt=negative,
                        image_width=width, image_height=height,
                        image_steps=steps, image_cfg=cfg,
                        image_sampler=sampler, image_scheduler=scheduler,
                        image_seed=seed,
                    )
            else:
                return RunResult(
                    model_id=model["id"], model_name=model["name"],
                    method=method, success=False,
                    error=str(exc),
                    failure_phase="runtime_error",
                    total_time=time.perf_counter() - wall_start,
                    surface="image",
                    prompt=prompt_text,
                    negative_prompt=negative,
                    image_width=width, image_height=height,
                    image_steps=steps, image_cfg=cfg,
                    image_sampler=sampler, image_scheduler=scheduler,
                    image_seed=seed,
                )
        wall_total = time.perf_counter() - wall_start

        image_rel, thumb_rel = self._save_image_artifact(
            model, png_bytes, sample_index=(self._current_sample or {}).get("index", 0),
        )

        return RunResult(
            model_id=model["id"], model_name=model["name"],
            method=method, success=True,
            total_time=wall_total,
            generation_time=wall_total,
            metric_kind="image",
            metric_label="Image",
            metric_value=f"{width}×{height}",
            surface="image",
            prompt=prompt_text,
            negative_prompt=negative,
            image_path=image_rel,
            thumbnail_path=thumb_rel,
            image_width=width, image_height=height,
            image_steps=steps, image_cfg=cfg,
            image_sampler=sampler, image_scheduler=scheduler,
            image_seed=seed,
            system_snapshot=self._system_snapshot(),
        )

    def _save_image_artifact(
        self, model: dict, png_bytes: bytes, *, sample_index: int,
    ) -> tuple[str, str]:
        """Save full PNG + 320px thumbnail beside the report.

        Returns ``(full_relative_path, thumbnail_relative_path)`` both rooted
        at the report directory so they render correctly when the HTML report
        is opened in place.
        """
        subdir = self.image_output_subdir or f"{self.report.file_stem}_images"
        full_dir = self.output_dir / subdir / "full"
        thumb_dir = self.output_dir / subdir / "thumb"
        full_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        safe_id = "".join(
            c if c.isalnum() or c in "._-" else "_" for c in str(model.get("id", "model"))
        )
        filename = f"{safe_id}_sample{sample_index + 1}.png"
        full_path = full_dir / filename
        full_path.write_bytes(png_bytes)

        thumb_path = thumb_dir / filename
        try:
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(png_bytes))
            img.thumbnail((320, 320))
            img.save(thumb_path, format="PNG", optimize=True)
        except Exception:
            # If Pillow is missing or thumbnailing fails, reuse the full image.
            thumb_path.write_bytes(png_bytes)

        # Use forward slashes for HTML compatibility on Windows.
        full_rel = f"{subdir}/full/{filename}"
        thumb_rel = f"{subdir}/thumb/{filename}"
        return full_rel, thumb_rel

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def _cleanup_model(self, model: dict, method: str) -> None:
        """Delete model after testing if --cleanup was specified."""
        try:
            if method.startswith("ollama_"):
                tag = model.get("ollama_tag", "")
                if tag and self.ollama.is_running():
                    if self.cleanup_downloaded_only and tag not in self._downloaded_ollama_tags:
                        self.ollama.unload_model(tag)
                        return
                    self.ollama.delete_model(tag)
                    self._downloaded_ollama_tags.discard(tag)
            elif method.startswith("onnx_"):
                if self.cleanup_downloaded_only:
                    return
                import shutil
                model_dir = self.models_dir / model["id"].replace(":", "_")
                if model_dir.exists():
                    shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass  # Cleanup is best-effort

    def _is_ollama_tag_local(self, tag: str) -> bool:
        try:
            local_names = self.ollama.local_model_names()
        except Exception:
            return False
        return ollama_tag_is_local(tag, local_names)

    def _release_after_run(self, model: dict, method: str) -> None:
        """Best-effort memory cleanup between benchmark runs."""
        try:
            if method.startswith("ollama_"):
                tag = model.get("ollama_tag", "")
                if tag and self.ollama.is_running():
                    self.ollama.unload_model(tag)
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    def _unload_running_ollama_models(self) -> None:
        """Best-effort unload of any currently resident Ollama model before a run."""
        try:
            for entry in self.ollama.running_models():
                name = entry.get("name") or entry.get("model")
                if name:
                    self.ollama.unload_model(name)
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _system_snapshot(self) -> dict:
        try:
            return get_system_summary(self.models_dir)
        except Exception:
            return {}

    def _handle_interrupt(self, signum, frame):
        print("\n\n!! Ctrl+C received — stopping current run and saving partial report...")
        self.request_stop()

    def _save_report(self) -> None:
        if not self.report.results:
            print("No results to save.")
            return
        json_path = self.report.save_json(self.output_dir)
        html_path = self.report.save_html(self.output_dir)
        # v5.5.1+: when a run isn't 100% pass, dump richer diagnostics next
        # to the JSON/HTML so post-mortem doesn't need to walk per-model
        # log directories.  Best-effort: failures (locked file, permission
        # error) print a warning but never abort the save.
        try:
            failure_paths = self._write_failure_diagnostics(self.output_dir)
        except Exception as exc:
            failure_paths = []
            print(f"  (failure diagnostics not written: {exc})")
        self.report.print_summary()
        print(f"Reports saved to:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        for diag_path in failure_paths:
            print(f"  DIAG: {diag_path}")

    def _write_failure_diagnostics(self, output_dir: Path) -> list[Path]:
        """Persist ``<stem>_failures.txt``, ``_env.txt``, ``_run.log`` on fail.

        Called from :meth:`_save_report` whenever the run has at least one
        failed result. Returns the list of files actually written so the
        caller can surface them in the post-run banner.
        """
        failed_results = [r for r in self.report.results if not r.success]
        if not failed_results:
            return []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.report.file_stem
        written: list[Path] = []

        failures_path = output_dir / f"{stem}_failures.txt"
        try:
            lines: list[str] = []
            lines.append(f"# Failure diagnostics for {stem}")
            lines.append(f"# Total failures: {len(failed_results)} / {len(self.report.results)}")
            if self.force_all:
                lines.append("# Force-All mode: streak-stop bypassed; ceilings tightened on OOM/disk.")
            if self._oom_ceiling_gb:
                ceil_text = ", ".join(
                    f"{m}={size:g}GB" for m, size in sorted(self._oom_ceiling_gb.items())
                )
                lines.append(f"# OOM ceilings reached: {ceil_text}")
            if self._disk_blocked_ceiling_gb:
                disk_text = ", ".join(
                    f"{m}={size:g}GB" for m, size in sorted(self._disk_blocked_ceiling_gb.items())
                )
                lines.append(f"# Disk-blocked ceilings reached: {disk_text}")
            lines.append("")
            for idx, result in enumerate(failed_results, start=1):
                display = (result.model_name or "").strip()
                header_label = result.model_id
                if display and display != result.model_id:
                    header_label = f"{result.model_id} ({display})"
                lines.append(f"───── [{idx}] {header_label} / {result.method}"
                             f" (sample {int(result.sample_index or 0) + 1}) ─────")
                lines.append(f"failure_phase : {result.failure_phase or 'unknown'}")
                lines.append(f"error         : {result.error or '(no error text)'}")
                if result.log_path:
                    lines.append(f"log_path      : {result.log_path}")
                if result.prompt:
                    lines.append("prompt        : |")
                    for prompt_line in str(result.prompt).splitlines() or [""]:
                        lines.append(f"  {prompt_line}")
                if result.response_text:
                    lines.append("response_text : |")
                    for resp_line in str(result.response_text).splitlines():
                        lines.append(f"  {resp_line}")
                lines.append("")
            failures_path.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="replace")
            written.append(failures_path)
        except Exception as exc:
            print(f"  (failures.txt not written: {exc})")

        env_path = output_dir / f"{stem}_env.txt"
        try:
            env_blob = self._collect_env_diagnostics()
            env_path.write_text(env_blob, encoding="utf-8", errors="replace")
            written.append(env_path)
        except Exception as exc:
            print(f"  (env.txt not written: {exc})")

        run_log_path = output_dir / f"{stem}_run.log"
        try:
            if self._captured_log_chunks:
                run_log_path.write_text(
                    "".join(self._captured_log_chunks),
                    encoding="utf-8",
                    errors="replace",
                )
                written.append(run_log_path)
        except Exception as exc:
            print(f"  (run.log not written: {exc})")
        return written

    def _collect_env_diagnostics(self) -> str:
        """Snapshot the host environment for the ``<stem>_env.txt`` sidecar."""
        import json as _json
        import shutil as _shutil
        import subprocess as _subprocess

        sections: list[str] = []
        try:
            sections.append("===== machine_info =====")
            sections.append(_json.dumps(self.report.machine_info or {}, indent=2, default=str))
        except Exception as exc:
            sections.append(f"(machine_info unavailable: {exc})")
        try:
            sections.append("===== python =====")
            sections.append(f"sys.version: {sys.version}")
            sections.append(f"sys.executable: {sys.executable}")
            sections.append(f"sys.platform: {sys.platform}")
        except Exception as exc:
            sections.append(f"(python info unavailable: {exc})")
        try:
            sections.append("===== ollama models dir =====")
            ollama_dir = os.environ.get("OLLAMA_MODELS") or "(default)"
            sections.append(f"OLLAMA_MODELS: {ollama_dir}")
        except Exception as exc:
            sections.append(f"(ollama dir unavailable: {exc})")
        try:
            sections.append("===== disk free =====")
            roots: set[str] = set()
            for candidate in [str(self.output_dir), str(self.models_dir), os.environ.get("OLLAMA_MODELS", "")]:
                if not candidate:
                    continue
                try:
                    drive = Path(candidate).resolve().drive or Path(candidate).resolve().anchor
                    roots.add(str(Path(candidate).resolve().anchor or drive))
                except Exception:
                    continue
            for root in sorted(roots):
                try:
                    total_b, used_b, free_b = _shutil.disk_usage(root)
                    sections.append(
                        f"{root}: total={total_b/1e9:.1f}GB used={used_b/1e9:.1f}GB free={free_b/1e9:.1f}GB"
                    )
                except Exception as exc:
                    sections.append(f"{root}: disk_usage failed: {exc}")
        except Exception as exc:
            sections.append(f"(disk free unavailable: {exc})")
        try:
            sections.append("===== pip freeze =====")
            proc = _subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
                capture_output=True, text=True, timeout=20,
            )
            if proc.returncode == 0:
                sections.append(proc.stdout.strip() or "(pip freeze produced no output)")
            else:
                sections.append(f"(pip freeze exit={proc.returncode}: {proc.stderr.strip()[:300]})")
        except Exception as exc:
            sections.append(f"(pip freeze unavailable: {exc})")
        return "\n\n".join(sections) + "\n"

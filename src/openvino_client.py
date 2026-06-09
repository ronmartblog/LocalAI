# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
OpenVINO GenAI inference client for Intel CPU, GPU, and NPU.

Requires:
  pip install openvino openvino-genai

This module gracefully degrades if the packages are not installed.
"""

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Generator, Optional

OV_GENAI_AVAILABLE = False
OV_DEVICES: list[str] = []
_OV_CORE = None

try:
    import openvino as _ov
    import openvino_genai as _ov_genai
    OV_GENAI_AVAILABLE = True
except Exception:
    pass


class OVError(Exception):
    pass


def available_ov_devices() -> list[str]:
    """Return OpenVINO devices, constructing Core only when a caller asks."""
    global _OV_CORE, OV_DEVICES
    if not OV_GENAI_AVAILABLE:
        return []
    if _OV_CORE is None:
        _OV_CORE = _ov.Core()
        OV_DEVICES = list(_OV_CORE.available_devices)
    return list(OV_DEVICES)


def pick_ov_device(prefer: str = "GPU") -> str:
    """Return the best available OpenVINO device.

    An explicit *prefer* is always honored when that device is actually
    present (so ``pick_ov_device("NPU")`` returns ``"NPU"`` on a Panther
    Lake / Lunar Lake / Meteor Lake box). When *prefer* is unavailable the
    fallback order is GPU > NPU > CPU: GPU is fast and reliable, NPU can
    still be selected explicitly, and CPU is the universal floor.
    """
    devices = available_ov_devices()
    for dev in (prefer, "GPU", "NPU", "CPU"):
        if dev in devices:
            return dev
    return "CPU"


def ov_device_full_name(device: str) -> str:
    """Return the OpenVINO ``FULL_DEVICE_NAME`` for *device* (e.g. the NPU's
    ``Intel(R) AI Boost``), or the bare device id when the property can't be
    read. Best-effort: never raises."""
    if not OV_GENAI_AVAILABLE:
        return device
    try:
        global _OV_CORE
        if _OV_CORE is None:
            available_ov_devices()  # constructs and caches the Core
        if _OV_CORE is None:
            return device
        return str(_OV_CORE.get_property(device, "FULL_DEVICE_NAME"))
    except Exception:
        return device


class OVModelSession:
    """
    Wraps openvino_genai.LLMPipeline with the same streaming interface as
    OnnxModelSession so the rest of the app can treat them identically.

    Device fallback: tries the requested device; if the NPU compiler rejects
    the model it automatically falls back to GPU then CPU.
    """

    def __init__(self, model_dir: str | Path, device: str = "NPU",
                 cache_dir: "str | Path | None" = None):
        if not OV_GENAI_AVAILABLE:
            raise OVError(
                "openvino-genai is not installed. "
                "Run: pip install openvino openvino-genai"
            )
        import openvino_genai as ov_genai

        model_dir = str(model_dir)
        fallback_order = _fallback_chain(device)
        errors: list[str] = []
        for dev in fallback_order:
            # CACHE_DIR makes the expensive first NPU/GPU compile happen once
            # per model upgrade instead of once per launch (cold ~30-60 s →
            # warm ~2-3 s). Without it every restart looks like a hang.
            props = _ov_pipeline_properties(dev, cache_dir)
            try:
                t0 = time.perf_counter()
                if props:
                    self._pipe = ov_genai.LLMPipeline(model_dir, dev, props)
                else:
                    self._pipe = ov_genai.LLMPipeline(model_dir, dev)
                elapsed = time.perf_counter() - t0
                self._device = dev
                cache_state = "cache hit" if elapsed < 8.0 else "cache miss"
                _log_compile(dev, elapsed, cache_state)
                return
            except Exception as e:
                errors.append(f"{dev}: {type(e).__name__}: {e}")
                continue
        tried = ", ".join(fallback_order)
        summary = " | ".join(errors) or "no devices available"
        raise OVError(f"Failed to load model on {tried}. Details — {summary}")

    # ── Streaming helpers ─────────────────────────────────────────────────────

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Yield decoded tokens one at a time."""
        import openvino_genai as ov_genai

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.temperature = temperature
            cfg.do_sample = True

        tok_q: queue.Queue = queue.Queue()
        _DONE = object()

        def _streamer(subword: str) -> ov_genai.StreamingStatus:
            tok_q.put(subword)
            if stop_event and stop_event.is_set():
                return ov_genai.StreamingStatus.STOP
            return ov_genai.StreamingStatus.RUNNING

        def _run():
            try:
                self._pipe.generate(prompt, generation_config=cfg, streamer=_streamer)
            finally:
                tok_q.put(_DONE)

        threading.Thread(target=_run, daemon=True).start()

        # v5: bounded queue.get — OpenVINO pipeline can wedge on certain quants.
        # 600 s overall ceiling; stop_event aborts sooner.
        deadline = time.time() + 600.0
        while True:
            if stop_event and stop_event.is_set():
                return
            try:
                item = tok_q.get(timeout=min(2.0, max(0.1, deadline - time.time())))
            except queue.Empty:
                if time.time() >= deadline:
                    raise OVError(
                        "OpenVINO generate() exceeded 600 s with no output. "
                        "The model may be stuck; restart LocalAI to recover."
                    )
                continue
            if item is _DONE:
                break
            if item:
                yield item

    def generate_stream_timed(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[tuple[str, dict | None], None, None]:
        """
        Yield (token, stats) tuples. stats is None for intermediate tokens
        and a dict on the last token containing:
          total_time, ttft, token_count, tokens_per_sec.
        """
        import openvino_genai as ov_genai

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.temperature = temperature
            cfg.do_sample = True

        tok_q: queue.Queue = queue.Queue()
        _DONE = object()

        def _streamer(subword: str) -> ov_genai.StreamingStatus:
            tok_q.put(subword)
            if stop_event and stop_event.is_set():
                return ov_genai.StreamingStatus.STOP
            return ov_genai.StreamingStatus.RUNNING

        def _run():
            try:
                self._pipe.generate(prompt, generation_config=cfg, streamer=_streamer)
            finally:
                tok_q.put(_DONE)

        t0 = time.perf_counter()
        first_token_time: float | None = None
        token_count = 0
        tokens_buffer: list[str] = []

        threading.Thread(target=_run, daemon=True).start()

        # v5: bounded loop — see generate_stream(); 600 s ceiling.
        deadline = time.time() + 600.0
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                item = tok_q.get(timeout=min(2.0, max(0.1, deadline - time.time())))
            except queue.Empty:
                if time.time() >= deadline:
                    raise OVError(
                        "OpenVINO generate() exceeded 600 s with no output. "
                        "The model may be stuck; restart LocalAI to recover."
                    )
                continue
            if item is _DONE:
                break
            if item:
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens_buffer.append(item)

        total_time = time.perf_counter() - t0
        ttft = (first_token_time - t0) if first_token_time is not None else total_time
        tps = token_count / total_time if total_time > 0 else 0.0

        stats = {
            "total_time": total_time,
            "ttft": ttft,
            "token_count": token_count,
            "tokens_per_sec": tps,
        }

        for i, tok in enumerate(tokens_buffer):
            yield tok, (stats if i == len(tokens_buffer) - 1 else None)


# ── Model download ────────────────────────────────────────────────────────────

def download_ov_model(
    repo_id: str,
    local_dir: str | Path,
    progress_cb: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Path:
    """Download a pre-built OpenVINO model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise OVError(
            'huggingface_hub is not installed. Run: pip install "huggingface-hub>=0.34.0,<1.0"'
        )

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(f"Downloading {repo_id} …")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
        if progress_cb:
            progress_cb(f"Download complete: {local_dir}")
        return local_dir
    except Exception as e:
        raise OVError(f"HuggingFace download failed: {e}") from e


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ov_pipeline_properties(device: str, cache_dir: "str | Path | None") -> dict:
    """Build the OpenVINO LLMPipeline property dict for *device*.

    Always sets ``CACHE_DIR`` when a cache directory is provided. For the NPU
    it additionally feature-detects safe acceleration hints by querying
    ``SUPPORTED_PROPERTIES`` and only sets hints the installed runtime
    actually exposes — this avoids version-skew breakage across OpenVINO
    releases.
    """
    props: dict = {}
    if cache_dir is not None:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        props["CACHE_DIR"] = str(cache_dir)

    if device == "NPU" and OV_GENAI_AVAILABLE:
        supported: list[str] = []
        try:
            global _OV_CORE
            if _OV_CORE is None:
                available_ov_devices()
            if _OV_CORE is not None:
                supported = list(_OV_CORE.get_property("NPU", "SUPPORTED_PROPERTIES"))
        except Exception:
            supported = []
        # Only set hints the runtime advertises.
        for key, value in (("NPU_USE_NPUW", "YES"),):
            if key in supported:
                props[key] = value
    return props


def _log_compile(device: str, elapsed: float, cache_state: str) -> None:
    """Log a single OV compile line; tolerant of a missing logger import."""
    try:
        from src import logger as _logger
        _logger.info(
            f"OV compile on {device}: {elapsed:.1f}s ({cache_state})",
            category=_logger.CATEGORY_SYSTEM,
        )
    except Exception:
        pass


def _fallback_chain(preferred: str) -> list[str]:
    """Return devices to try in order, starting from preferred."""
    order = ["NPU", "GPU", "CPU"]
    if preferred in order:
        idx = order.index(preferred)
        return order[idx:] + order[:idx]
    return [preferred] + [d for d in order if d != preferred]


def _cli_probe(model_dir: str, device: str, cache_dir: str, max_tokens: int) -> dict:
    """Compile *model_dir* on *device* and run a short generate, returning a
    JSON-serialisable result dict. Used by the Settings 'Test NPU' button via a
    subprocess so the (GIL-holding, occasionally-hanging) NPU compile can never
    freeze the desktop UI."""
    t0 = time.perf_counter()
    session = OVModelSession(model_dir, device, cache_dir=cache_dir or None)
    compile_s = time.perf_counter() - t0
    ntok = 0
    tps = 0.0
    for _tok, stats in session.generate_stream_timed(
        "Hello, how are you?", max_new_tokens=max_tokens
    ):
        if stats:
            ntok = stats["token_count"]
            tps = stats["tokens_per_sec"]
    return {
        "ok": True,
        "device": getattr(session, "_device", device),
        "compile_s": compile_s,
        "ntok": ntok,
        "tps": tps,
    }


if __name__ == "__main__":
    # Subprocess entry point: python -m src.openvino_client <model_dir> <device> <cache_dir> [max_tokens]
    import json as _json
    import sys as _sys

    try:
        _model_dir = _sys.argv[1]
        _device = _sys.argv[2] if len(_sys.argv) > 2 else "NPU"
        _cache_dir = _sys.argv[3] if len(_sys.argv) > 3 else ""
        _max_tokens = int(_sys.argv[4]) if len(_sys.argv) > 4 else 16
        _result = _cli_probe(_model_dir, _device, _cache_dir, _max_tokens)
        print(_json.dumps(_result))
    except Exception as _exc:  # noqa: BLE001 — surface any failure as a JSON result
        print(_json.dumps({"ok": False, "error": f"{type(_exc).__name__}: {_exc}"}))

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
    """Return the best available OpenVINO device. Priority: GPU > NPU > CPU.

    NPU is intentionally deprioritised because the VPUX compiler may crash the
    process (calling exit()) on models it cannot compile.  GPU is fast and
    reliable; NPU can still be selected explicitly by passing prefer='NPU'.
    """
    devices = available_ov_devices()
    for dev in (prefer, "GPU", "NPU", "CPU"):
        if dev in devices:
            return dev
    return "CPU"


class OVModelSession:
    """
    Wraps openvino_genai.LLMPipeline with the same streaming interface as
    OnnxModelSession so the rest of the app can treat them identically.

    Device fallback: tries the requested device; if the NPU compiler rejects
    the model it automatically falls back to GPU then CPU.
    """

    def __init__(self, model_dir: str | Path, device: str = "NPU"):
        if not OV_GENAI_AVAILABLE:
            raise OVError(
                "openvino-genai is not installed. "
                "Run: pip install openvino openvino-genai"
            )
        import openvino_genai as ov_genai

        model_dir = str(model_dir)
        fallback_order = _fallback_chain(device)
        last_err = None
        for dev in fallback_order:
            try:
                self._pipe = ov_genai.LLMPipeline(model_dir, dev)
                self._device = dev
                return
            except Exception as e:
                last_err = e
                continue
        raise OVError(f"Failed to load model on any device: {last_err}") from last_err

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

def _fallback_chain(preferred: str) -> list[str]:
    """Return devices to try in order, starting from preferred."""
    order = ["NPU", "GPU", "CPU"]
    if preferred in order:
        idx = order.index(preferred)
        return order[idx:] + order[:idx]
    return [preferred] + [d for d in order if d != preferred]

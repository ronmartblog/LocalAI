# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
ONNX Runtime inference client for DirectML (NPU / iGPU) backends.
Used for models like Phi-3 Mini that ship as ONNX, and
for Phi-4 ONNX bundles (which require the dedicated
onnxruntime-genai runtime instead of optimum.onnxruntime — the
Phi-4 ONNX graphs use a GQA layout that optimum's KV-cache handling does
not understand, producing
``past_key_values.N.value: Got: <attn_heads> Expected: <kv_heads>`` errors).

Requires:
  pip install onnxruntime-directml optimum[onnxruntime] transformers
  (or onnxruntime for CPU-only)

  # Optional, required for Phi-4 ONNX bundles:
  pip install onnxruntime-genai-directml

This module gracefully degrades if the packages are not installed.
"""

import threading
import time
from pathlib import Path
from typing import Callable, Generator, Optional

# Availability flags set at import time
import sys as _sys

ONNX_AVAILABLE = False
DIRECTML_AVAILABLE = False
COREML_AVAILABLE = False
OPENVINO_AVAILABLE = False
HF_AVAILABLE = False
GENAI_AVAILABLE = False
GENAI_DML_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    _providers = ort.get_available_providers()
    DIRECTML_AVAILABLE = "DmlExecutionProvider" in _providers
    COREML_AVAILABLE = "CoreMLExecutionProvider" in _providers
    OPENVINO_AVAILABLE = "OpenVINOExecutionProvider" in _providers
except Exception:
    # v2026.06.01.4 defensive fix: a broken onnxruntime install
    # (e.g. onnxruntime-directml without a Python 3.13 wheel) can satisfy
    # ``import onnxruntime`` but be missing ``get_available_providers`` —
    # an AttributeError there used to crash the entire app at module load.
    # Treat ANY exception (not just ImportError) as "ONNX unavailable" so
    # the app still starts on the Ollama backend. Matches the defensive
    # ``except Exception`` pattern used in src/openvino_client.py.
    ONNX_AVAILABLE = False
    DIRECTML_AVAILABLE = False
    COREML_AVAILABLE = False
    OPENVINO_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime_genai as _og  # noqa: F401 — feature probe
    GENAI_AVAILABLE = True
    try:
        GENAI_DML_AVAILABLE = bool(_og.is_dml_available())
    except Exception:
        GENAI_DML_AVAILABLE = False
except Exception:
    # Same defensive pattern as the onnxruntime block above — a partial
    # onnxruntime-genai install can satisfy import but raise non-ImportError
    # exceptions on first symbol access. Treat all such failures as
    # "GenAI unavailable" so app startup is never blocked.
    GENAI_AVAILABLE = False
    GENAI_DML_AVAILABLE = False


class OnnxError(Exception):
    pass


def has_genai_config(model_dir: str | Path, subfolder: str = "") -> bool:
    """Return True if *model_dir*/*subfolder* contains a genai_config.json.

    Phi-4 ONNX bundles (and several Phi-3.x bundles) ship a
    `genai_config.json` file that the onnxruntime-genai runtime uses to
    drive token generation. Its presence is the signal that we should
    bypass optimum.onnxruntime and use ``OnnxGenAISession`` instead.
    """
    base = Path(model_dir) / subfolder if subfolder else Path(model_dir)
    return (base / "genai_config.json").exists()


def _format_genai_chat_prompt(prompt: str) -> str:
    """Wrap raw prompts in the Phi/ONNX GenAI chat markers expected by genai bundles."""
    text = str(prompt or "")
    if "<|im_start|>" in text or "<|im_sep|>" in text:
        return text
    return f"<|im_start|>user<|im_sep|>{text}<|im_end|><|im_start|>assistant<|im_sep|>"


def _pick_provider(prefer_npu: bool = True) -> tuple[str, dict]:
    """Return (provider_name, provider_options) for the best available accelerator.

    Windows priority: OpenVINO NPU > DirectML > CPU.
    macOS priority:   CoreML > CPU.
    """
    if _sys.platform == "darwin":
        if COREML_AVAILABLE:
            return "CoreMLExecutionProvider", {}
        return "CPUExecutionProvider", {}
    if prefer_npu:
        if OPENVINO_AVAILABLE:
            return "OpenVINOExecutionProvider", {"device_type": "NPU"}
        if DIRECTML_AVAILABLE:
            return "DmlExecutionProvider", {}
    return "CPUExecutionProvider", {}


class OnnxModelSession:
    """Wraps an Optimum-loaded ONNX causal-LM model."""

    def __init__(
        self,
        model_dir: str | Path,
        provider: str,
        provider_options: dict | None = None,
        subfolder: str = "",
    ):
        """Load an ONNX causal-LM model.

        Args:
            model_dir:        Root directory containing tokenizer files and the
                              onnx subfolder (as downloaded by download_onnx_model).
            provider:         ONNX Runtime execution provider string.
            provider_options: Provider-specific options (e.g. {"device_type": "NPU"}).
            subfolder:        Sub-path within model_dir where the ONNX files live
                              (e.g. "directml/directml-int4-awq-block-128").
        """
        if not ONNX_AVAILABLE:
            raise OnnxError("onnxruntime is not installed. Run: pip install onnxruntime-openvino")
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise OnnxError(
                "optimum[onnxruntime] and transformers are required. "
                "Run: pip install optimum[onnxruntime] transformers"
            )

        self._provider = provider
        # Resolve the full path to the directory containing the ONNX files.
        # Tokenizer files live alongside the model in the same subfolder.
        model_path = Path(model_dir) / subfolder if subfolder else Path(model_dir)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            kwargs: dict = {"provider": provider}
            if provider_options:
                kwargs["provider_options"] = provider_options
            self.model = ORTModelForCausalLM.from_pretrained(str(model_path), **kwargs)
        except Exception as e:
            raise OnnxError(f"Failed to load ONNX model: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Yield decoded tokens one at a time."""
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            raise OnnxError("transformers is required for streaming")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }

        # If model.generate() raises inside this daemon thread, the streamer's
        # internal queue never receives a stop sentinel and the consumer
        # `for token in streamer:` below blocks forever. Always call
        # streamer.end() in finally, and capture the exception so we can
        # re-raise it on the consumer thread instead of leaking a hang.
        error_box: dict = {}

        def _runner():
            try:
                self.model.generate(**gen_kwargs)
            except BaseException as e:  # noqa: BLE001 — propagate everything
                error_box["exc"] = e
            finally:
                try:
                    streamer.end()
                except Exception:
                    pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

        for token in streamer:
            if stop_event and stop_event.is_set():
                break
            yield token

        # Bounded join — generate() should already have ended by the time the
        # streamer iterator is exhausted, but keep the watchdog as a backstop.
        t.join(timeout=600)
        if t.is_alive():
            raise OnnxError(
                "ONNX generate() did not return within 600 s. "
                "The model may be stuck; restart LocalAI to recover."
            )
        if "exc" in error_box:
            raise OnnxError(
                f"ONNX generate() failed: {error_box['exc']}"
            ) from error_box["exc"]

    def generate_stream_timed(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[tuple[str, dict | None], None, None]:
        """
        Yield (token, stats) tuples with manual perf_counter timing.
        *stats* is None for intermediate tokens and a dict on the last
        token containing: total_time, ttft, token_count, tokens_per_sec.
        """
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            raise OnnxError("transformers is required for streaming")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }

        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0

        # See generate_stream() for the streamer-hang rationale — same
        # contract: ensure streamer.end() always runs and surface any
        # in-thread exception on the consumer side.
        error_box: dict = {}

        def _runner():
            try:
                self.model.generate(**gen_kwargs)
            except BaseException as e:  # noqa: BLE001
                error_box["exc"] = e
            finally:
                try:
                    streamer.end()
                except Exception:
                    pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

        tokens_buffer = []
        for token in streamer:
            if stop_event and stop_event.is_set():
                break
            if token:
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens_buffer.append(token)

        # Bounded join — generate() should already have ended by the time the
        # streamer iterator is exhausted, but keep the watchdog as a backstop.
        t.join(timeout=600)
        if t.is_alive():
            raise OnnxError(
                "ONNX generate() did not return within 600 s. "
                "The model may be stuck; restart LocalAI to recover."
            )
        if "exc" in error_box:
            raise OnnxError(
                f"ONNX generate() failed: {error_box['exc']}"
            ) from error_box["exc"]
        total_time = time.perf_counter() - t0

        ttft = (first_token_time - t0) if first_token_time is not None else total_time
        tps = token_count / total_time if total_time > 0 else 0.0

        stats = {
            "total_time": total_time,
            "ttft": ttft,
            "token_count": token_count,
            "tokens_per_sec": tps,
        }

        # Yield all buffered tokens; attach stats to the last one
        for i, tok in enumerate(tokens_buffer):
            if i == len(tokens_buffer) - 1:
                yield tok, stats
            else:
                yield tok, None


# ── onnxruntime-genai session (Phi-4 ONNX bundles) ────────────────

class OnnxGenAISession:
    """Wraps an onnxruntime-genai-loaded ONNX causal-LM model.

    Used for Phi-4 ONNX bundles (and any other ONNX model that
    ships a ``genai_config.json``). The genai runtime understands the
    Grouped-Query-Attention KV cache layout the upstream publisher bakes
    into those graphs, which optimum.onnxruntime does not.

    The execution provider is determined by which ``onnxruntime-genai-*``
    package variant is installed (``-directml`` for DML, ``-cuda`` for
    CUDA, plain ``onnxruntime-genai`` for CPU). The genai_config.json
    in the model subfolder pins the EP per-subfolder, so callers should
    pick the right model-published subfolder for their target EP
    (e.g. ``gpu/gpu-int4-rtn-block-32`` for DML,
    ``cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4`` for CPU).
    """

    def __init__(self, model_dir: str | Path, subfolder: str = ""):
        if not GENAI_AVAILABLE:
            raise OnnxError(
                "onnxruntime-genai is not installed. "
                "Run: pip install onnxruntime-genai-directml"
            )
        import onnxruntime_genai as og

        full = Path(model_dir) / subfolder if subfolder else Path(model_dir)
        if not (full / "genai_config.json").exists():
            raise OnnxError(
                f"genai_config.json not found in {full}; "
                "this model is not packaged for onnxruntime-genai."
            )
        try:
            self.model = og.Model(str(full))
            self.tokenizer = og.Tokenizer(self.model)
        except Exception as e:
            raise OnnxError(f"Failed to load ONNX genai model: {e}") from e

    def generate_stream_timed(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[tuple[str, dict | None], None, None]:
        """Same yield contract as OnnxModelSession.generate_stream_timed."""
        import onnxruntime_genai as og

        try:
            input_tokens = self.tokenizer.encode(_format_genai_chat_prompt(prompt))
        except Exception as e:
            raise OnnxError(f"genai tokenizer.encode failed: {e}") from e

        params = og.GeneratorParams(self.model)
        search_opts: dict = {
            "max_length": int(len(input_tokens) + max_new_tokens),
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            search_opts["temperature"] = float(temperature)
        try:
            params.set_search_options(**search_opts)
        except Exception as e:
            raise OnnxError(f"genai set_search_options failed: {e}") from e

        try:
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
        except Exception as e:
            raise OnnxError(f"genai Generator init failed: {e}") from e

        stream = self.tokenizer.create_stream()

        t0 = time.perf_counter()
        first_token_time: Optional[float] = None
        token_count = 0

        try:
            while not generator.is_done():
                if stop_event and stop_event.is_set():
                    break
                generator.generate_next_token()
                new_token = int(generator.get_next_tokens()[0])
                text = stream.decode(new_token)
                if text:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    yield text, None
        except Exception as e:
            raise OnnxError(f"genai generation failed: {e}") from e

        total_time = time.perf_counter() - t0
        ttft = (first_token_time - t0) if first_token_time is not None else total_time
        tps = token_count / total_time if total_time > 0 else 0.0
        stats = {
            "total_time": total_time,
            "ttft": ttft,
            "token_count": token_count,
            "tokens_per_sec": tps,
        }
        yield "", stats

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        for token, _stats in self.generate_stream_timed(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_event=stop_event,
        ):
            if token:
                yield token


# ── Downloading ONNX models from HuggingFace ─────────────────────────────────

def download_onnx_model(
    repo_id: str,
    subfolder: str,
    local_dir: str | Path,
    progress_cb: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Path:
    """
    Download an ONNX model from HuggingFace Hub into *local_dir*.
    Returns the local directory path.
    """
    if not HF_AVAILABLE:
        raise OnnxError(
            'huggingface_hub is not installed. Run: pip install "huggingface-hub>=0.34.0,<1.0"'
        )

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(f"Downloading {repo_id}/{subfolder} …")

    try:
        result = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*", "tokenizer*", "special_tokens_map.json"],
            local_dir=str(local_dir),
        )
        # The model files land inside local_dir/subfolder
        model_path = local_dir / subfolder
        if not model_path.exists():
            model_path = Path(result)
        if progress_cb:
            progress_cb(f"Download complete: {model_path}")
        return model_path
    except Exception as e:
        raise OnnxError(f"HuggingFace download failed: {e}") from e

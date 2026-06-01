from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import warnings
import wave
import importlib.util
from pathlib import Path

from src.ollama_client import OllamaClient, OllamaError


_APP_ROOT = Path(__file__).parent.parent
_DEFAULT_HF_CACHE_DIR = _APP_ROOT / ".cache" / "huggingface"

_LEGACY_HF_CACHE_DIRS: tuple[Path, ...] = (
    _APP_ROOT / "models" / "phase1",
    Path.home() / ".cache" / "huggingface",
)


def _resolve_hf_cache_dir() -> Path:
    """Pick the HF cache directory: ambient HF_HOME wins, otherwise default.

    Respecting the ambient value lets ``setup.bat`` / ``run.bat`` keep control
    of the canonical cache location and prevents this module from silently
    overriding the env-var redirection that v5.3.10 ships in its batch files.
    """
    ambient = (os.environ.get("HF_HOME") or "").strip()
    if ambient:
        try:
            return Path(ambient)
        except (TypeError, ValueError):
            pass
    return _DEFAULT_HF_CACHE_DIR


def _configure_hf_cache_env() -> Path:
    cache_dir = _resolve_hf_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    return cache_dir


@contextlib.contextmanager
def _quiet_known_hf_loader_warnings():
    """Silence known third-party loader chatter while preserving hard failures."""
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        hf_logging = None
    previous_verbosity = hf_logging.get_verbosity() if hf_logging is not None else None
    if hf_logging is not None:
        hf_logging.set_verbosity_error()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"`huggingface_hub` cache-system uses symlinks.*")
            warnings.filterwarnings("ignore", message=r"The `max_size` parameter is deprecated.*")
            warnings.filterwarnings("ignore", message=r"for .*: copying from a non-meta parameter.*")
            yield
    finally:
        if hf_logging is not None and previous_verbosity is not None:
            hf_logging.set_verbosity(previous_verbosity)


_configure_hf_cache_env()


def configure_local_hf_environment() -> Path:
    """Apply quiet, local Hugging Face cache settings before model loads."""
    return _configure_hf_cache_env()


def configure_benchmark_environment() -> Path:
    """Apply quiet, local Hugging Face cache settings before benchmark loads."""
    return configure_local_hf_environment()


_COMMON_HF_DEPS = ["torch", "transformers", "huggingface_hub", "hf_xet"]
_DEPS_BY_MODEL_ID = {
    "all-minilm": ["sentence_transformers", *_COMMON_HF_DEPS],
    "whisper-large-v3-turbo": ["soundfile", "torch", "transformers", "scipy", "accelerate", "safetensors", "huggingface_hub", "hf_xet"],
    "whisper-v3-turbo-gpu": ["soundfile", "torch", "transformers", "scipy", "accelerate", "safetensors", "huggingface_hub", "hf_xet"],
    "florence-2-base": ["PIL", "torch", "transformers", "accelerate", "safetensors", "einops", "timm", "huggingface_hub", "hf_xet"],
    "trocr-base-printed": ["PIL", *_COMMON_HF_DEPS],
    "trocr-large-printed": ["PIL", *_COMMON_HF_DEPS],
    "table-transformer": ["PIL", *_COMMON_HF_DEPS],
    "speecht5-tts": ["soundfile", "torch", "transformers", "sentencepiece", "huggingface_hub", "hf_xet"],
    "piper-tts": ["piper", "onnxruntime", "soundfile", "huggingface_hub", "hf_xet"],
    "got-ocr2": ["PIL", "torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_xet"],
    "phi4-mini-onnx": [],
    "phi-4-multimodal": [],
    "sd21": ["torch", "diffusers", "huggingface_hub", "hf_xet"],
    "sd3.5-medium": ["torch", "diffusers", "huggingface_hub", "hf_xet"],
    "sd3.5-large": ["torch", "diffusers", "huggingface_hub", "hf_xet"],
}


def required_dependencies_for_model(m: dict) -> list[str]:
    model_id = m.get("id", "")
    if model_id in _DEPS_BY_MODEL_ID:
        return list(_DEPS_BY_MODEL_ID[model_id])
    runtime = (m.get("runtime") or "").lower()
    if "sentence-transformers" in runtime:
        return list(_DEPS_BY_MODEL_ID["all-minilm"])
    return list(_COMMON_HF_DEPS)


def missing_dependencies_for_model(m: dict) -> list[str]:
    return [
        dep for dep in required_dependencies_for_model(m)
        if importlib.util.find_spec(dep) is None
    ]


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)[:120]


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype():
    import torch

    return torch.float16 if torch.cuda.is_available() else torch.float32


def _from_pretrained_with_dtype(model_cls, repo: str, **kwargs):
    dtype = _torch_dtype()
    try:
        return model_cls.from_pretrained(repo, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return model_cls.from_pretrained(repo, torch_dtype=dtype, **kwargs)


def _require_hf_revision(model: dict) -> str:
    revision = str(model.get("hf_revision") or model.get("remote_code_revision") or "").strip()
    if not revision:
        raise RuntimeError(
            f"{model.get('id', model.get('hf_repo', 'This model'))} uses trust_remote_code=True "
            "and must pin hf_revision in the catalog."
        )
    return revision


def _load_fixture_font(size: int):
    from PIL import ImageFont

    candidates = [
        "arial.ttf",
        str(Path("C:/Windows/Fonts/arial.ttf")),
        "DejaVuSans.ttf",
        str(Path("C:/Windows/Fonts/segoeui.ttf")),
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _ensure_dirs(root: Path):
    for name in ["images", "audio", "harness/assets"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def _draw_text_image(path: Path, text: str, *, table: bool = False):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(img)
    if table:
        font = _load_fixture_font(24)
        rows = ["Model | Runtime | Result", "Florence | vision | ok", "TrOCR | OCR | ok", "LocalAI | private | yes"]
        x0, y0, w, h = 70, 70, 760, 220
        for i in range(5):
            y = y0 + i * (h // 4)
            draw.line((x0, y, x0 + w, y), fill="black", width=2)
        for i in range(4):
            x = x0 + i * (w // 3)
            draw.line((x, y0, x, y0 + h), fill="black", width=2)
        draw.line((x0 + w, y0, x0 + w, y0 + h), fill="black", width=2)
        for i, row in enumerate(rows):
            for j, cell in enumerate(row.split("|")):
                draw.text((x0 + 12 + j * (w // 3), y0 + 14 + i * (h // 4)), cell.strip(), fill="black", font=font)
        draw.text((70, 330), "Synthetic table benchmark image for LocalAI Studio", fill="black", font=font)
    else:
        font = _load_fixture_font(82)
        draw.rectangle((50, 45, 1150, 375), outline="black", width=3)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            ((1200 - text_w) / 2, (420 - text_h) / 2 - 10),
            text,
            fill="black",
            font=font,
        )
    img.save(path)
    return path


def _ensure_vision_image(root: Path, model_id: str, *, table: bool = False) -> Path:
    path = root / "harness" / "assets" / f"{_safe_name(model_id)}_input.png"
    if not path.exists() or str(model_id).startswith("trocr-"):
        path.parent.mkdir(parents=True, exist_ok=True)
        _draw_text_image(path, "LOCAL AI STUDIO 2026", table=table)
    return path


def _ensure_speech_wav(root: Path) -> Path:
    wav_path = root / "audio" / "phase1_reference_speech.wav"
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return wav_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    ps = (
        "$p='{}';"
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.SetOutputToWaveFile($p);"
        "$s.Speak('Local AI Studio runs models privately on your computer.');"
        "$s.Dispose()"
    ).format(str(wav_path).replace("'", "''"))
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=60, capture_output=True)
    except Exception:
        with wave.open(str(wav_path), "w") as w:
            sample_rate = 16000
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for i in range(sample_rate * 2):
                sample = int(12000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
                w.writeframesraw(sample.to_bytes(2, "little", signed=True))
    return wav_path


def _ollama_tag_is_local(tag: str) -> bool:
    try:
        return OllamaClient().is_model_local(tag)
    except OllamaError:
        return False


def _remove_ollama_tag(tag: str) -> str:
    try:
        OllamaClient().delete_model(tag)
    except OllamaError as exc:
        return f"delete_failed: {exc}"
    return "deleted"


def _adapter_error(started: float, error: str, cleanup_status: str = "") -> dict:
    return {
        "status": "error",
        "error": error,
        "run_time_s": round(time.perf_counter() - started, 2),
        "cleanup_status": cleanup_status,
        "output_text": "",
    }


def _common_result(started: float, output_text: str, **extra):
    result = {
        "status": "ok",
        "download_time_s": 0,
        "run_time_s": round(time.perf_counter() - started, 2),
        "cleanup_status": "hf_cache_retained",
        "output_text": output_text[:4000],
    }
    result.update(extra)
    return result


def run_sentence_transformer(m: dict, root: Path) -> dict:
    started = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(m["hf_repo"], device=_device())
    sentences = [
        "Local AI Studio can run models privately on your computer.",
        "A benchmark gallery helps choose models for offline use.",
        "The weather forecast is unrelated to model evaluation.",
    ]
    vectors = model.encode(sentences, normalize_embeddings=True)
    score = float(vectors[0] @ vectors[1])
    return _common_result(
        started,
        f"Embedding benchmark completed.\nmodel={m['hf_repo']}\nembedding_dim={len(vectors[0])}\nsemantic_similarity_localai_vs_gallery={score:.4f}",
        test_prompt="Embed three LocalAI benchmark sentences and compare their similarity.",
        metric_label="Embeddings",
        metric_value=f"{len(sentences)} vectors",
    )


def run_whisper(m: dict, root: Path) -> dict:
    started = time.perf_counter()
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    audio = _ensure_speech_wav(root)
    samples, sample_rate = sf.read(str(audio), dtype="float32")
    if sample_rate != 16000:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sample_rate), 16000)
        samples = resample_poly(samples, 16000 // gcd, int(sample_rate) // gcd).astype("float32")
        sample_rate = 16000
    processor = AutoProcessor.from_pretrained(m["hf_repo"])
    model = _from_pretrained_with_dtype(WhisperForConditionalGeneration, m["hf_repo"]).to(_device())
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    encoded = processor(
        samples,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = encoded.input_features.to(_device(), _torch_dtype())
    attention_mask = encoded.get("attention_mask")
    generate_kwargs = {}
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask.to(_device())
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            **generate_kwargs,
            language="english",
            task="transcribe",
            max_new_tokens=96,
        )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return _common_result(
        started,
        f"Transcription result:\n{text}",
        audio=str(audio),
        test_prompt="Transcribe a local synthetic speech WAV about LocalAI Studio.",
        metric_label="Audio",
        metric_value=f"{len(samples) / sample_rate:.1f}s transcribed",
    )


def run_florence(m: dict, root: Path) -> dict:
    started = time.perf_counter()
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    image_path = _ensure_vision_image(root, m["id"])
    image = Image.open(image_path).convert("RGB")
    revision = _require_hf_revision(m)
    processor = AutoProcessor.from_pretrained(
        m["hf_repo"],
        trust_remote_code=True,
        use_fast=True,
        revision=revision,
    )
    model = _from_pretrained_with_dtype(
        AutoModelForCausalLM,
        m["hf_repo"],
        trust_remote_code=True,
        attn_implementation="eager",
        revision=revision,
    )
    if not hasattr(model, "_supports_sdpa"):
        model._supports_sdpa = False
    model = model.to(_device())
    prompt = "<CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(_device(), _torch_dtype())
    with _quiet_known_hf_loader_warnings():
        generated_ids = model.generate(**inputs, max_new_tokens=96, num_beams=1, use_cache=False)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
    return _common_result(
        started,
        f"Florence vision result for synthetic LocalAI image:\n{parsed}",
        image=str(image_path),
        test_prompt="Caption a synthetic LocalAI Studio benchmark image.",
        metric_label="Image",
        metric_value="1 image",
    )


def run_trocr(m: dict, root: Path) -> dict:
    started = time.perf_counter()
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    image_path = _ensure_vision_image(root, m["id"])
    image = Image.open(image_path).convert("RGB")
    processor = TrOCRProcessor.from_pretrained(m["hf_repo"])
    model = VisionEncoderDecoderModel.from_pretrained(m["hf_repo"]).to(_device())
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(_device())
    generated_ids = model.generate(pixel_values, max_new_tokens=64)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    words = set(re.findall(r"[A-Z0-9]+", text.upper()))
    if not {"LOCAL", "AI", "STUDIO"}.issubset(words):
        return _adapter_error(
            started,
            f"OCR output did not match the synthetic fixture text 'LOCAL AI STUDIO 2026': {text!r}",
        )
    return _common_result(
        started,
        f"OCR result:\n{text}",
        image=str(image_path),
        test_prompt="Read the text from a synthetic LocalAI Studio benchmark image.",
        metric_label="Image",
        metric_value="1 image",
    )


def run_table_transformer(m: dict, root: Path) -> dict:
    started = time.perf_counter()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    image_path = _ensure_vision_image(root, m["id"], table=True)
    image = Image.open(image_path).convert("RGB")
    with _quiet_known_hf_loader_warnings():
        processor = AutoImageProcessor.from_pretrained(m["hf_repo"], use_fast=False)
        model = TableTransformerForObjectDetection.from_pretrained(
            m["hf_repo"],
            low_cpu_mem_usage=False,
        ).to(_device())
    inputs = processor(images=image, return_tensors="pt").to(_device())
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=_device())
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
    labels = []
    id2label = model.config.id2label
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        labels.append({
            "label": id2label[int(label)],
            "score": round(float(score), 3),
            "box": [round(float(x), 1) for x in box],
        })
    from src.workflows import _dedupe_table_detections, _format_table_detections
    labels = _dedupe_table_detections(labels)
    return _common_result(
        started,
        _format_table_detections(labels, image.size),
        image=str(image_path),
        test_prompt="Detect table regions in a synthetic LocalAI benchmark table image.",
        metric_label="Detections",
        metric_value=f"{len(labels)} region(s)",
    )


def run_speecht5(m: dict, root: Path) -> dict:
    started = time.perf_counter()
    import soundfile as sf
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    processor = SpeechT5Processor.from_pretrained(m["hf_repo"])
    model = SpeechT5ForTextToSpeech.from_pretrained(m["hf_repo"]).to(_device())
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(_device())
    text = "Local AI Studio can run speech models privately without an API key."
    inputs = processor(text=text, return_tensors="pt").to(_device())
    speaker_embeddings = torch.zeros((1, 512), device=_device())
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    out = root / "audio" / f"{_safe_name(m['id'])}.wav"
    sf.write(str(out), speech.cpu().numpy(), samplerate=16000)
    return _common_result(
        started,
        f"Generated local TTS WAV: {out}\nPrompt: {text}",
        audio=str(out),
        test_prompt=text,
        metric_label="Audio",
        metric_value="1 wav",
    )


def run_phi_text(m: dict, root: Path, prompt: str | None = None) -> dict:
    started = time.perf_counter()
    if m["id"] not in {"phi4-mini-onnx", "phi-4-multimodal"}:
        raise RuntimeError(f"No Phi utility benchmark adapter implemented for {m.get('id')}")
    tag = "phi4-mini" if m["id"] == "phi4-mini-onnx" else "phi4"
    was_local = _ollama_tag_is_local(tag)
    should_cleanup = False
    cleanup_status = "retained_existing" if was_local else "not_downloaded"
    prompt = prompt or "Give three concise reasons a private local AI app should support multimodal and ONNX-accelerated models."
    payload = json.dumps({
        "model": tag,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 96},
    }).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        subprocess.run(
            ["ollama", "pull", tag],
            check=True,
            timeout=3600,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        should_cleanup = not was_local
        with urllib.request.urlopen(req, timeout=600) as r:
            output = json.loads(r.read().decode("utf-8")).get("response", "")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        return _adapter_error(started, f"Ollama command failed for {tag}: {detail}", cleanup_status)
    except subprocess.TimeoutExpired as exc:
        return _adapter_error(started, f"Ollama command timed out for {tag} after {exc.timeout}s", cleanup_status)
    except (urllib.error.URLError, TimeoutError) as exc:
        if should_cleanup:
            cleanup_status = _remove_ollama_tag(tag)
        return _adapter_error(started, f"Ollama API request failed for {tag}: {exc}", cleanup_status)
    except json.JSONDecodeError as exc:
        if should_cleanup:
            cleanup_status = _remove_ollama_tag(tag)
        return _adapter_error(started, f"Ollama returned invalid JSON for {tag}: {exc}", cleanup_status)
    if should_cleanup:
        cleanup_status = _remove_ollama_tag(tag)
    return _common_result(
        started,
        output,
        cleanup_status=cleanup_status,
        test_prompt=prompt,
        metric_label="Text",
        metric_value="1 response",
    )


def run_sd3_diffusers(m: dict, root: Path) -> dict:
    started = time.perf_counter()
    import torch
    from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline

    repo = m["hf_repo"]
    if m["id"] == "sd21":
        repo = "sd-research/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=_torch_dtype())
    else:
        repo = {
            "sd3.5-medium": "adamo1139/stable-diffusion-3.5-medium-ungated",
            "sd3.5-large": "adamo1139/stable-diffusion-3.5-large-ungated",
        }.get(m["id"], repo)
        pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=_torch_dtype())
    pipe = pipe.to(_device())
    prompt = "A clean desktop app screenshot concept for Local AI Studio model benchmarking, professional lighting"
    image = pipe(prompt, num_inference_steps=12, guidance_scale=4.0, height=768, width=768).images[0]
    out = root / "images" / f"{_safe_name(m['id'])}.png"
    image.save(out)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _common_result(
        started,
        f"Generated image with diffusers from {repo}",
        image=str(out),
        test_prompt=prompt,
        metric_label="Image",
        metric_value="1 image",
    )


def run_transformers_adapter(m: dict, root: Path) -> dict:
    _ensure_dirs(root)
    runtime = (m.get("runtime") or "").lower()
    task = (m.get("task") or "").lower()
    model_id = m.get("id", "")
    if model_id == "all-minilm" or "sentence-transformers" in runtime:
        return run_sentence_transformer(m, root)
    if "whisper" in model_id or "speech recognition" in task:
        return run_whisper(m, root)
    if model_id.startswith("florence"):
        return run_florence(m, root)
    if model_id.startswith("trocr"):
        return run_trocr(m, root)
    if model_id == "table-transformer":
        return run_table_transformer(m, root)
    if model_id == "speecht5-tts":
        return run_speecht5(m, root)
    if model_id in {"phi4-mini-onnx", "phi-4-multimodal"}:
        return run_phi_text(m, root)
    if model_id == "sd21" or model_id.startswith("sd3.5"):
        return run_sd3_diffusers(m, root)
    raise RuntimeError(f"No utility benchmark adapter implemented for {model_id} ({runtime})")

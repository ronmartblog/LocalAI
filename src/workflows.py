# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Interactive Toolbox workflows for non-chat catalog adapters."""

from __future__ import annotations

import math
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path


from src.phase1_adapters import _configure_hf_cache_env, _quiet_known_hf_loader_warnings


_configure_hf_cache_env()


def configure_toolbox_environment() -> Path:
    """Apply quiet, local Hugging Face cache settings before workflow loads."""
    return _configure_hf_cache_env()


def _device() -> str:
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


def _emit(progress_cb, message: str) -> None:
    if progress_cb:
        progress_cb(message)


def _format_florence_result(parsed, task: str) -> str:
    if isinstance(parsed, dict):
        value = parsed.get(task)
        if isinstance(value, dict):
            text = "\n".join(f"{key}: {val}" for key, val in value.items()).strip()
            return text or "No readable text was returned."
        if isinstance(value, list):
            text = "\n".join(str(item) for item in value).strip()
            return text or "No readable text was returned."
        if value:
            return str(value).strip()
        return "No readable text was returned."
    text = str(parsed).strip()
    return text or "No readable text was returned."


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _dedupe_table_detections(detections: list[dict], iou_threshold: float = 0.55) -> list[dict]:
    kept: list[dict] = []
    for item in sorted(detections, key=lambda x: x["score"], reverse=True):
        if all(_box_iou(item["box"], other["box"]) < iou_threshold for other in kept):
            kept.append(item)
    return kept


def _format_table_detections(detections: list[dict], image_size: tuple[int, int]) -> str:
    width, height = image_size
    if not detections:
        return (
            "No confident table regions were found.\n\n"
            "Try a tighter crop around the table, a higher-resolution screenshot, or a page with visible grid/table boundaries."
        )
    label = "region" if len(detections) == 1 else "regions"
    lines = [
        f"Found {len(detections)} likely table {label}.",
        "This detector finds table regions only; it does not extract cell text yet.",
        "",
    ]
    for idx, item in enumerate(detections, start=1):
        x1, y1, x2, y2 = item["box"]
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        lines.append(
            f"{idx}. {item['label']} — {item['score'] * 100:.0f}% confidence, "
            f"x={x1:.0f}, y={y1:.0f}, w={box_w:.0f}, h={box_h:.0f} "
            f"({box_w / max(width, 1) * 100:.0f}% of image width, {box_h / max(height, 1) * 100:.0f}% of image height)"
        )
        if item.get("crop"):
            lines.append(f"   Crop saved: {item['crop']}")
    lines.extend([
        "",
        "For text in the table, run Read image text on a saved crop.",
    ])
    return "\n".join(lines)


def _save_table_crops(image, detections: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(detections, start=1):
        x1, y1, x2, y2 = item["box"]
        crop_box = (
            int(max(0, round(x1))),
            int(max(0, round(y1))),
            int(min(image.width, round(x2))),
            int(min(image.height, round(y2))),
        )
        crop = image.crop(crop_box)
        crop_path = output_dir / f"table_region_{int(time.time())}_{idx}.png"
        crop.save(crop_path)
        item["crop"] = str(crop_path)


@dataclass
class WorkflowResult:
    output_text: str
    output_path: Path | None = None
    metadata: dict = field(default_factory=dict)


WHISPER_LANGUAGES: dict[str, str | None] = {
    "Auto-detect": None,
    "English": "english",
    "Spanish": "spanish",
    "French": "french",
    "German": "german",
    "Italian": "italian",
    "Portuguese": "portuguese",
    "Dutch": "dutch",
    "Polish": "polish",
    "Russian": "russian",
    "Ukrainian": "ukrainian",
    "Czech": "czech",
    "Swedish": "swedish",
    "Norwegian": "norwegian",
    "Danish": "danish",
    "Finnish": "finnish",
    "Turkish": "turkish",
    "Arabic": "arabic",
    "Hindi": "hindi",
    "Bengali": "bengali",
    "Urdu": "urdu",
    "Mandarin Chinese": "chinese",
    "Cantonese": "cantonese",
    "Japanese": "japanese",
    "Korean": "korean",
    "Vietnamese": "vietnamese",
    "Thai": "thai",
    "Indonesian": "indonesian",
    "Malay": "malay",
    "Tagalog": "tagalog",
    "Hebrew": "hebrew",
    "Greek": "greek",
    "Romanian": "romanian",
    "Hungarian": "hungarian",
    "Bulgarian": "bulgarian",
    "Croatian": "croatian",
    "Serbian": "serbian",
    "Slovak": "slovak",
    "Slovenian": "slovenian",
    "Catalan": "catalan",
    "Welsh": "welsh",
    "Persian": "persian",
    "Swahili": "swahili",
    "Afrikaans": "afrikaans",
    "Tamil": "tamil",
    "Telugu": "telugu",
    "Marathi": "marathi",
    "Gujarati": "gujarati",
    "Kannada": "kannada",
    "Malayalam": "malayalam",
    "Estonian": "estonian",
    "Latvian": "latvian",
    "Lithuanian": "lithuanian",
}


PIPER_VOICES: dict[str, dict[str, str]] = {
    "English (US, female, Amy)": {
        "key": "en_US-amy-medium",
        "onnx": "en/en_US/amy/medium/en_US-amy-medium.onnx",
        "json": "en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    },
    "English (US, male, Ryan)": {
        "key": "en_US-ryan-medium",
        "onnx": "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "json": "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    },
    "English (GB, female, Alba)": {
        "key": "en_GB-alba-medium",
        "onnx": "en/en_GB/alba/medium/en_GB-alba-medium.onnx",
        "json": "en/en_GB/alba/medium/en_GB-alba-medium.onnx.json",
    },
    "Spanish (Spain, male, Sharvard)": {
        "key": "es_ES-sharvard-medium",
        "onnx": "es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "json": "es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json",
    },
    "Spanish (Mexico, female, Claude)": {
        "key": "es_MX-claude-high",
        "onnx": "es/es_MX/claude/high/es_MX-claude-high.onnx",
        "json": "es/es_MX/claude/high/es_MX-claude-high.onnx.json",
    },
    "French (France, female, Siwis)": {
        "key": "fr_FR-siwis-medium",
        "onnx": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
        "json": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
    },
    "German (Thorsten)": {
        "key": "de_DE-thorsten-medium",
        "onnx": "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
        "json": "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
    },
    "Italian (Riccardo)": {
        "key": "it_IT-riccardo-x_low",
        "onnx": "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx",
        "json": "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx.json",
    },
    "Portuguese (Brazil, male, Faber)": {
        "key": "pt_BR-faber-medium",
        "onnx": "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx",
        "json": "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json",
    },
    "Dutch (NL, Mls 5809)": {
        "key": "nl_NL-mls_5809-low",
        "onnx": "nl/nl_NL/mls_5809/low/nl_NL-mls_5809-low.onnx",
        "json": "nl/nl_NL/mls_5809/low/nl_NL-mls_5809-low.onnx.json",
    },
    "Polish (Gosia)": {
        "key": "pl_PL-gosia-medium",
        "onnx": "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx",
        "json": "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json",
    },
    "Russian (Irina)": {
        "key": "ru_RU-irina-medium",
        "onnx": "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
        "json": "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json",
    },
    "Ukrainian (Lada)": {
        "key": "uk_UA-lada-x_low",
        "onnx": "uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx",
        "json": "uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx.json",
    },
    "Chinese (Mandarin, Huayan)": {
        "key": "zh_CN-huayan-medium",
        "onnx": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
        "json": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
    },
    "Vietnamese (vais1000)": {
        "key": "vi_VN-vais1000-medium",
        "onnx": "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
        "json": "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
    },
    "Arabic (Kareem)": {
        "key": "ar_JO-kareem-medium",
        "onnx": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
        "json": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json",
    },
    "Turkish (Dfki)": {
        "key": "tr_TR-dfki-medium",
        "onnx": "tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx",
        "json": "tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json",
    },
    "Catalan (Upc Ona X-Low)": {
        "key": "ca_ES-upc_ona-x_low",
        "onnx": "ca/ca_ES/upc_ona/x_low/ca_ES-upc_ona-x_low.onnx",
        "json": "ca/ca_ES/upc_ona/x_low/ca_ES-upc_ona-x_low.onnx.json",
    },
}


def transcribe(
    audio_path: Path,
    model_entry: dict,
    *,
    progress_cb=None,
    language: str | None = None,
) -> WorkflowResult:
    """Run Whisper transcription.

    ``language`` accepts the Whisper-side language string (e.g. ``"english"``,
    ``"spanish"``) or ``None`` to let Whisper auto-detect from the audio. The
    Toolbox UI surfaces this via :data:`WHISPER_LANGUAGES`. The benchmark
    adapter (:func:`src.phase1_adapters.run_whisper`) continues to hard-code
    ``language="english"`` for deterministic baselining and is not affected.
    """
    started = time.perf_counter()
    _emit(progress_cb, "Loading Whisper...")
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    samples, sample_rate = sf.read(str(audio_path), dtype="float32")
    if sample_rate != 16000:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sample_rate), 16000)
        samples = resample_poly(samples, 16000 // gcd, int(sample_rate) // gcd).astype("float32")
        sample_rate = 16000
    processor = AutoProcessor.from_pretrained(model_entry["hf_repo"])
    model = _from_pretrained_with_dtype(
        WhisperForConditionalGeneration,
        model_entry["hf_repo"],
    ).to(_device())
    # Required for both language="english" (deterministic) and the
    # multi-language / auto-detect path: when forced_decoder_ids is set,
    # passing language= to generate() raises. See HF transformers
    # whisper docs.
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    _emit(progress_cb, "Transcribing audio...")
    encoded = processor(
        samples,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = encoded.input_features.to(_device(), _torch_dtype())
    attention_mask = encoded.get("attention_mask")
    generate_kwargs = {"task": "transcribe", "max_new_tokens": 256}
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask.to(_device())
    # Omit language entirely on auto-detect so Whisper uses its built-in
    # language ID head. Pass it explicitly otherwise.
    if language:
        generate_kwargs["language"] = language
    with torch.no_grad():
        predicted_ids = model.generate(inputs, **generate_kwargs)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return WorkflowResult(text, metadata={"elapsed_s": round(time.perf_counter() - started, 2)})


def read_image(image_path: Path, model_entry: dict, *, mode: str = "ocr", progress_cb=None) -> WorkflowResult:
    started = time.perf_counter()
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    repo = model_entry["hf_repo"]
    _emit(progress_cb, "Loading image reader...")
    if "trocr" in model_entry.get("id", "").lower():
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(repo)
        model = VisionEncoderDecoderModel.from_pretrained(repo).to(_device())
        _emit(progress_cb, "Reading text...")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(_device())
        generated_ids = model.generate(pixel_values, max_new_tokens=96)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        revision = _require_hf_revision(model_entry)
        processor = AutoProcessor.from_pretrained(
            repo,
            trust_remote_code=True,
            use_fast=True,
            revision=revision,
        )
        model = _from_pretrained_with_dtype(
            AutoModelForCausalLM,
            repo,
            trust_remote_code=True,
            attn_implementation="eager",
            revision=revision,
        )
        if not hasattr(model, "_supports_sdpa"):
            model._supports_sdpa = False
        model = model.to(_device())
        task = "<OCR>" if mode == "ocr" else "<CAPTION>"
        _emit(progress_cb, "Analyzing image...")
        inputs = processor(text=task, images=image, return_tensors="pt").to(_device(), _torch_dtype())
        with _quiet_known_hf_loader_warnings():
            generated_ids = model.generate(**inputs, max_new_tokens=256, num_beams=1, use_cache=False)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
        text = _format_florence_result(parsed, task)
    return WorkflowResult(text, metadata={"elapsed_s": round(time.perf_counter() - started, 2)})


def detect_table(
    image_path: Path,
    model_entry: dict,
    *,
    output_dir: Path | None = None,
    progress_cb=None,
) -> WorkflowResult:
    started = time.perf_counter()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    image = Image.open(image_path).convert("RGB")
    _emit(progress_cb, "Loading table detector...")
    with _quiet_known_hf_loader_warnings():
        processor = AutoImageProcessor.from_pretrained(model_entry["hf_repo"], use_fast=False)
        model = TableTransformerForObjectDetection.from_pretrained(
            model_entry["hf_repo"],
            low_cpu_mem_usage=False,
        ).to(_device())
    _emit(progress_cb, "Detecting table structure...")
    inputs = processor(images=image, return_tensors="pt").to(_device())
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=_device())
    results = processor.post_process_object_detection(outputs, threshold=0.45, target_sizes=target_sizes)[0]
    labels = [
        {
            "label": model.config.id2label[int(label)],
            "score": round(float(score), 3),
            "box": [round(float(v), 1) for v in box.tolist()],
        }
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
    ]
    labels = _dedupe_table_detections(labels)
    if output_dir and labels:
        _save_table_crops(image, labels, output_dir)
    text = _format_table_detections(labels, image.size)
    return WorkflowResult(text, metadata={"elapsed_s": round(time.perf_counter() - started, 2), "objects": labels})


def synthesize(
    text: str,
    model_entry: dict,
    *,
    output_dir: Path | None = None,
    progress_cb=None,
    language: str | None = None,
) -> WorkflowResult:
    """Generate speech from text using a Piper ONNX voice.

    ``language`` is the human-readable voice key (e.g.
    ``"English (US, female, Amy)"``) from :data:`PIPER_VOICES`. If omitted or
    unrecognized, the first voice in ``PIPER_VOICES`` is used. Voice ONNX +
    config files are pulled lazily from ``rhasspy/piper-voices`` into the
    shared HF cache; each voice is ~30-150 MB.
    """
    started = time.perf_counter()
    from huggingface_hub import hf_hub_download
    from piper.voice import PiperVoice

    output_dir = output_dir or (Path.cwd() / "toolbox_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if language and language in PIPER_VOICES:
        voice = PIPER_VOICES[language]
    else:
        voice = next(iter(PIPER_VOICES.values()))
    voice_key = voice["key"]
    repo = model_entry.get("hf_repo", "rhasspy/piper-voices")

    _emit(progress_cb, f"Downloading voice {voice_key}...")
    onnx_path = hf_hub_download(repo_id=repo, filename=voice["onnx"])
    config_path = hf_hub_download(repo_id=repo, filename=voice["json"])

    _emit(progress_cb, "Loading Piper voice...")
    piper_voice = PiperVoice.load(onnx_path, config_path=config_path)
    # Fallback rate from the voice config; the per-chunk sample_rate is
    # authoritative once the first chunk arrives.
    fallback_rate = getattr(getattr(piper_voice, "config", None), "sample_rate", 22050)

    out = output_dir / f"piper_{voice_key}_{int(time.time())}.wav"
    _emit(progress_cb, "Synthesizing speech...")
    # piper-tts >=1.2 returns Iterable[AudioChunk]; we must iterate, pull
    # `audio_int16_bytes` per chunk, and write the frames ourselves. Older
    # piper-tts <1.0 accepted a wave.Wave_write target on synthesize() and
    # wrote frames directly — that signature no longer exists, and calling
    # the new API without iterating produces an empty (0-second) WAV with
    # only the header.
    sample_rate = fallback_rate
    sample_width = 2
    sample_channels = 1
    total_frames = 0
    with wave.open(str(out), "wb") as wav_file:
        header_set = False
        for chunk in piper_voice.synthesize(text):
            if not header_set:
                # Trust the first chunk's metadata over the config fallback —
                # some voices (e.g. low-quality 16 kHz models) override it.
                sample_rate = getattr(chunk, "sample_rate", sample_rate) or sample_rate
                sample_width = getattr(chunk, "sample_width", sample_width) or sample_width
                sample_channels = getattr(chunk, "sample_channels", sample_channels) or sample_channels
                wav_file.setnchannels(sample_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                header_set = True
            audio_bytes = chunk.audio_int16_bytes
            if not audio_bytes:
                continue
            wav_file.writeframes(audio_bytes)
            total_frames += len(audio_bytes) // (sample_width * sample_channels)
        if not header_set:
            # Synthesizer yielded zero chunks (empty/whitespace input). Write
            # a valid empty WAV header so downstream callers don't see a
            # zero-byte file, but surface a clear error to the caller.
            wav_file.setnchannels(sample_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)

    if total_frames == 0:
        raise RuntimeError(
            "Piper produced no audio for the given text. "
            "Try non-empty input with at least one pronounceable sentence."
        )

    duration_s = round(total_frames / float(sample_rate), 2) if sample_rate else 0.0
    return WorkflowResult(
        f"Saved synthesized speech to:\n{out}\n(duration: {duration_s}s @ {sample_rate} Hz)",
        out,
        {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "voice": voice_key,
            "duration_s": duration_s,
            "sample_rate": sample_rate,
        },
    )


def embed_and_rank(query: str, corpus: list[str], model_entry: dict, *, progress_cb=None) -> WorkflowResult:
    started = time.perf_counter()
    _emit(progress_cb, "Loading embedding model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_entry["hf_repo"], device=_device())
    _emit(progress_cb, "Ranking snippets...")
    vectors = model.encode([query] + corpus, normalize_embeddings=True)
    q = vectors[0]
    scored = sorted(
        ((float(q @ vec), text) for vec, text in zip(vectors[1:], corpus)),
        reverse=True,
        key=lambda item: item[0],
    )
    lines = ["Ranked snippets:"] + [f"{i+1}. {score:.4f} — {text}" for i, (score, text) in enumerate(scored[:5])]
    return WorkflowResult("\n".join(lines), metadata={"elapsed_s": round(time.perf_counter() - started, 2)})


def describe(image_path: Path, model_entry: dict, *, question: str | None = None, progress_cb=None) -> WorkflowResult:
    if "florence" in model_entry.get("id", "").lower():
        return read_image(image_path, model_entry, mode="caption", progress_cb=progress_cb)
    return read_image(image_path, model_entry, mode="caption", progress_cb=progress_cb)


def write_test_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        sample_rate = 16000
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(sample_rate):
            sample = int(12000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            w.writeframesraw(sample.to_bytes(2, "little", signed=True))
    return path


def _table_text_to_tsv(text: str) -> str:
    """Convert a GOT-OCR / minicpm-v plain or LaTeX table response into TSV.

    Strategy: if the text contains a LaTeX ``tabular`` / ``array`` block, parse
    that. Otherwise fall back to splitting on the most common row separator
    (newline) and per-row on tabs or two-or-more spaces. Always preserves at
    least one row so the caller can show something useful even if parsing
    fails.

    LaTeX gotcha: GOT-OCR commonly puts ``\\hline`` (and friends) INLINE at the
    start of every data row, e.g. ``\\hline Row 1 & a1 & b1 \\\\``. A naive
    "drop lines beginning with a backslash" filter eats every data row and
    leaves only the header. We strip the directives token-by-token and decide
    whether the residual line has any cell content before keeping it.
    """
    if not text:
        return ""
    import re as _re

    body = text
    for marker in (r"\begin{tabular}", r"\begin{array}", r"\begin{table}"):
        if marker in body:
            start = body.index(marker)
            end_marker = marker.replace(r"\begin", r"\end")
            end = body.find(end_marker, start)
            inner = body[start:end] if end != -1 else body[start:]
            # Strip the begin{...}{column spec} header line if present.
            if "}" in inner.split("\n", 1)[0]:
                inner = inner.split("}", 2)[-1]
            body = inner
            break

    # Normalize the row terminator so a row split on a newline is the same
    # row whether the LaTeX source put `\\` mid-line or at end-of-line.
    body = body.replace(r"\\", "\n")
    # Strip standalone LaTeX directives that appear without arguments — these
    # are NOT row content; we want to keep whatever follows them on the line.
    # Order matters: \begin{...}/\end{...} BEFORE bare \hline etc. so the
    # body-stripper inside the loop never sees a stray brace.
    _DIRECTIVE_PATTERNS = (
        _re.compile(r"\\begin\{[^}]*\}"),
        _re.compile(r"\\end\{[^}]*\}"),
        _re.compile(r"\\(?:hline|toprule|midrule|bottomrule|hrule|cline\{[^}]*\}|cmidrule\{[^}]*\}|cmidrule|noalign\{[^}]*\}|rule|vspace\{[^}]*\}|hspace\{[^}]*\})"),
    )

    rows: list[list[str]] = []
    for raw in body.splitlines():
        line = raw
        for pat in _DIRECTIVE_PATTERNS:
            line = pat.sub(" ", line)
        # Strip any residual escapes/braces commonly seen in GOT-OCR output.
        line = line.replace("\\&", "&").strip()
        if not line:
            continue
        if "&" in line:
            cells = [c.strip() for c in line.split("&")]
        elif "\t" in line:
            cells = [c.strip() for c in line.split("\t")]
        else:
            parts = _re.split(r"\s{2,}", line)
            cells = [c.strip() for c in parts]
        # Drop rows that became all-empty after directive stripping (a bare
        # `\hline` line in the source, for example).
        if not any(c for c in cells):
            continue
        rows.append(cells)
    return "\n".join("\t".join(row) for row in rows)


def _got_dtype_and_device():
    """Pick a safe dtype + device for GOT-OCR.

    GOT-OCR's HF config advertises bfloat16, but
    ``AutoModelForImageTextToText.from_pretrained`` silently falls back to
    float32 unless ``dtype`` is set explicitly. On CUDA we use bfloat16 (safe
    on Ampere+ and matches the model card); on CPU we use float32 because
    bfloat16 matmul kernels are very slow on most CPUs.
    """
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16, "cuda"
    return torch.float32, "cpu"


def extract_table_ollama(
    image_path: Path,
    model_entry: dict,
    *,
    progress_cb=None,
    output_dir: Path | None = None,
) -> WorkflowResult:
    """Extract table content from an image using an Ollama-hosted multimodal model.

    The model tag comes from ``model_entry["ollama_tag"]`` (e.g.
    ``minicpm-v:latest``). The model is expected to be pulled already - the
    Toolbox UI gates this workflow on Ollama running + tag present.
    """
    import base64

    from src.ollama_client import OllamaClient

    started = time.perf_counter()
    output_dir = output_dir or (Path.cwd() / "toolbox_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = model_entry.get("ollama_tag") or "minicpm-v:latest"
    _emit(progress_cb, f"Asking {tag} to read the table...")
    client = OllamaClient()
    prompt = (
        "Extract the table from this image. Return the result as a tab-separated "
        "values (TSV) block: one row per line, columns separated by a single tab. "
        "Preserve the original column order and skip purely decorative rows. Do not "
        "wrap the response in code fences or add commentary."
    )
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    messages = [
        {"role": "user", "content": prompt, "images": [image_b64]},
    ]
    pieces: list[str] = []
    for chunk in client.chat_stream(
        tag=tag,
        messages=messages,
        num_gpu=-1,
        temperature=0.1,
        num_predict=2048,
    ):
        pieces.append(chunk)
    raw = "".join(pieces).strip()
    tsv = _table_text_to_tsv(raw)
    out = output_dir / f"table_extract_ollama_{int(time.time())}.txt"
    out.write_text(tsv or raw, encoding="utf-8")
    summary = tsv or raw or "(model returned no content)"
    return WorkflowResult(
        f"Saved table to:\n{out}\n\n{summary}",
        out,
        {"elapsed_s": round(time.perf_counter() - started, 2), "backend": "ollama", "tag": tag},
    )


def _run_got(image_path: Path, model_entry: dict, *, progress_cb=None) -> str:
    """Load GOT-OCR 2.0 and run a single image -> LaTeX/text pass.

    Forces an explicit dtype to avoid the silent float32 load that the model's
    Hub config does not protect against.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype, device = _got_dtype_and_device()
    revision = _require_hf_revision(model_entry)
    repo = model_entry["hf_repo"]
    _emit(progress_cb, "Loading GOT-OCR 2.0...")
    processor = AutoProcessor.from_pretrained(repo, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        repo,
        torch_dtype=dtype,
        revision=revision,
    ).to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    _emit(progress_cb, "Running GOT-OCR 2.0 (format mode)...")
    inputs = processor(images=image, return_tensors="pt", format=True).to(device, dtype)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings=["<|im_end|>"],
            max_new_tokens=2048,
        )
    output = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    return output


def extract_table_got(
    image_path: Path,
    model_entry: dict,
    *,
    output_dir: Path | None = None,
    progress_cb=None,
) -> WorkflowResult:
    """Extract table content from an image using GOT-OCR 2.0.

    Saves both the raw LaTeX/text the model produced and a best-effort TSV
    conversion alongside it.
    """
    started = time.perf_counter()
    output_dir = output_dir or (Path.cwd() / "toolbox_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _run_got(image_path, model_entry, progress_cb=progress_cb)
    timestamp = int(time.time())
    latex_path = output_dir / f"table_extract_got_{timestamp}.txt"
    latex_path.write_text(raw, encoding="utf-8")
    tsv = _table_text_to_tsv(raw)
    tsv_path = output_dir / f"table_extract_got_{timestamp}_tsv.txt"
    tsv_path.write_text(tsv or raw, encoding="utf-8")
    summary = tsv or raw or "(model returned no content)"
    return WorkflowResult(
        f"Saved raw output to:\n{latex_path}\nSaved TSV to:\n{tsv_path}\n\n{summary}",
        tsv_path,
        {"elapsed_s": round(time.perf_counter() - started, 2), "backend": "got-ocr2"},
    )

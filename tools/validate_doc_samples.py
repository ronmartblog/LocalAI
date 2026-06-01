#!/usr/bin/env python
"""
Validate documented LocalAI Studio sample prompts through the app backend paths.

The runner enumerates active catalog models, asks ``model_demos.get_model_demo``
for each model's curated demo prompts + sample renders, then exercises those
prompts end-to-end through the same backends the app uses (Ollama for chat,
ComfyUI for image gen, phase-1 adapters for Toolbox). Outputs + an HTML report
land on D: by default.

Before the v5.3.4 doc consolidation this module also reverse-engineered the
generated HTML guides (ChatPromptIdeas / ImageGenPrompts / ModelDemoPrompts)
with regex. That round-trip was redundant — ``get_model_demo`` is the single
source-of-truth for documented prompts — and tightly coupled the validator to
specific HTML markup. The new Model Guide is consumed as a single document
name; the actual data comes straight from the catalog + ``get_model_demo``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import catalog, content_filter, phase1_adapters, workflows  # noqa: E402
from src.app import App  # noqa: E402
from src.gpu_detect import detect_gpu_cached  # noqa: E402
from src.model_demos import _negative_prompt_for_model, get_model_demo  # noqa: E402
from src.ollama_client import OllamaClient, strip_think_blocks, think_option_for_model  # noqa: E402
from src.system_info import BENCHMARK_SKU_PROFILES, get_gpu_info, get_ram_info  # noqa: E402


IMAGE_FEATURE_GUIDE = ROOT / "docs" / "image-gen-guide.html"
# Single consolidated doc replacing the four legacy prompt pages
# (ImageGenPrompts.html, ChatPromptIdeas.html, ModelDemoPrompts.html,
# model-value-props.html) — those were superseded in v5.3.4 and their
# redirect shims were deleted in post-v5.3.4 docs cleanup.
MODEL_GUIDE = ROOT / "docs" / "Model-Guide.html"

SAFE_REFERENCE_DIRS = [
    Path(p)
    for p in (os.environ.get("LOCALAI_DOC_SAMPLE_REFERENCE_DIRS") or "").split(os.pathsep)
    if p
] or [
    Path.home() / "Pictures" / "LocalAI-reference",
    Path.home() / "Pictures" / "LocalAI-reference" / "LoRA",
]


@dataclass
class DocSample:
    id: str
    source_doc: str
    surface: str
    model_id: str
    model_name: str
    title: str
    prompt: str
    entry: dict[str, Any] = field(default_factory=dict)
    negative: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    docs_settings: dict[str, str] = field(default_factory=dict)
    input_path: str = ""
    reference_image: str = ""
    notes: str = ""


@dataclass
class DocResult:
    id: str
    source_doc: str
    surface: str
    model_id: str
    model_name: str
    title: str
    prompt: str
    status: str
    started_at: str
    ended_at: str
    duration_s: float
    output_text: str = ""
    output_path: str = ""
    original_output_path: str = ""
    replacement_prompt: str = ""
    error: str = ""
    issue_fixed: str = ""
    token_count: int = 0
    prompt_tokens: int = 0
    tokens_per_sec: float = 0.0
    ttft_s: float = 0.0
    load_time_s: float = 0.0
    generation_time_s: float = 0.0
    metric_label: str = ""
    metric_value: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    docs_settings: dict[str, str] = field(default_factory=dict)
    input_path: str = ""
    reference_image: str = ""
    min_ram_gb: float = 0.0
    min_vram_gb: float = 0.0
    minimum_sku: str = ""
    image_quality: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _slug(value: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("_")
    return (text or "sample")[:max_len]


def _clean_text(value: str) -> str:
    text = re.sub(r"<button\b.*?</button>", "", value, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _default_output_dir() -> Path:
    root = Path("D:/")
    if not root.exists():
        root = ROOT
    return root / f"LocalAI_Doc_Sample_Validation_{_timestamp()}"


def _find_reference_image(paths: Iterable[Path]) -> Path | None:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for base in paths:
        if base.is_file() and base.suffix.lower() in exts:
            return base
        if not base.exists() or not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_file() and child.suffix.lower() in exts:
                return child
        for root, _dirs, files in os.walk(base):
            for name in files:
                path = Path(root) / name
                if path.suffix.lower() in exts:
                    return path
    return None


def _create_reference_fixture(out_dir: Path) -> Path:
    path = out_dir / "inputs" / "synthetic_reference.png"
    if path.exists():
        return path
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (900, 640), "#f5f5f5")
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 60, 840, 580), fill="#ffffff", outline="#111827", width=4)
    draw.rectangle((110, 120, 790, 220), fill="#dbeafe", outline="#2563eb", width=3)
    draw.text((140, 150), "LocalAI Studio validation reference", fill="#111827")
    draw.ellipse((160, 270, 360, 470), fill="#f59e0b", outline="#92400e", width=4)
    draw.rectangle((430, 280, 750, 440), fill="#111827", outline="#4b5563", width=4)
    draw.text((470, 340), "Private local AI", fill="#ffffff")
    draw.text((140, 520), "Prompt, OCR, tables, speech, chat, image generation", fill="#111827")
    img.save(path)
    return path


# Removed in v5.3.4: _parse_image_prompt_doc, _parse_chat_prompt_doc, and
# _parse_chat_model_meta. They reverse-engineered the now-archived HTML guides
# with regex; the new Model Guide is generated from `get_model_demo()`, which
# the main inventory loop below already iterates directly. The chat-guide
# loop also reduplicated samples already covered by `get_model_demo()`, so
# dropping it removes both dead code and a coverage gap (the old loop emitted
# only one prompt per model, while the catalog loop emits the full set).


def _settings_for_entry(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    recommended = dict(entry.get("recommended_settings") or {})
    width = int(args.width or recommended.get("width") or 512)
    height = int(args.height or recommended.get("height") or 512)
    if args.max_dimension and max(width, height) > args.max_dimension:
        if width >= height:
            height = max(64, int(round(height * args.max_dimension / width)))
            width = args.max_dimension
        else:
            width = max(64, int(round(width * args.max_dimension / height)))
            height = args.max_dimension
        width = max(64, int(round(width / 64) * 64))
        height = max(64, int(round(height / 64) * 64))
    return {
        "width": width,
        "height": height,
        "steps": int(args.steps or recommended.get("steps") or 20),
        "cfg": float(args.cfg if args.cfg is not None else recommended.get("cfg", 7.0)),
        "sampler": str(args.sampler or recommended.get("sampler") or "euler"),
        "scheduler": str(args.scheduler or recommended.get("scheduler") or "normal"),
        "seed": int(args.seed),
    }


def _minimum_sku(entry: dict[str, Any]) -> str:
    ram = float(entry.get("min_ram_gb") or 0)
    vram = float(entry.get("min_vram_gb") or 0)
    for profile in BENCHMARK_SKU_PROFILES:
        if float(profile.get("ram_gb") or 0) >= ram and float(profile.get("vram_gb") or 0) >= vram:
            return str(profile["name"])
    return "Above max profile" if vram > 24 or ram > 440 else "Current/local profile"


def _supports_vision_chat(entry: dict[str, Any]) -> bool:
    tags = " ".join(entry.get("tags") or []).lower()
    category = str(entry.get("category") or "").lower()
    model_id = str(entry.get("id") or "").lower()
    return bool(entry.get("supports_vision")) or "vision" in category or "vision" in tags or "vl" in model_id


def _local_names() -> set[str]:
    return OllamaClient().local_model_names()


def _ollama_tag_is_local(tag: str, local_names: set[str]) -> bool:
    if not tag:
        return False
    if tag in local_names or f"{tag}:latest" in local_names:
        return True
    base = tag.split(":", 1)[0]
    return base in local_names or f"{base}:latest" in local_names


def collect_samples(models: list[dict[str, Any]], args: argparse.Namespace, reference_image: Path) -> list[DocSample]:
    by_id = {m.get("id"): m for m in models}
    by_name = {_slug(str(m.get("name", "")).lower()): m for m in models}
    by_ollama_tag = {str(m.get("ollama_tag")): m for m in models if m.get("ollama_tag")}
    active_image = [
        m for m in models if m.get("backend") == "comfyui" or m.get("category") == "Image Generation"
    ]
    active_image_ids = {str(m["id"]) for m in active_image if m.get("id")}
    # Acknowledge the lookups + active-image set so future filters can use
    # them; they were also referenced by the now-removed chat/image-guide
    # parser loops.
    del by_name, by_ollama_tag, active_image_ids
    samples: list[DocSample] = []

    def add(sample: DocSample) -> None:
        if args.source and sample.source_doc not in args.source:
            return
        if args.surface and sample.surface not in args.surface:
            return
        if args.model and sample.model_id.lower() not in {m.lower() for m in args.model}:
            return
        samples.append(sample)

    for entry in models:
        demo = get_model_demo(entry)
        for index, prompt in enumerate(demo["samples"], start=1):
            surface = "chat"
            if entry.get("backend") == "comfyui" or entry.get("category") == "Image Generation":
                surface = "image"
            elif entry.get("phase1_adapter") or entry.get("category") in {"Speech", "Embeddings", "Document AI"}:
                surface = "toolbox"
            settings = _settings_for_entry(entry, args) if surface == "image" else {}
            # Pull the shared negative from the same source the Model Guide
            # renders. Non-image-gen and CFG-locked families return "".
            negative = ""
            if surface == "image":
                neg_text, is_image_gen = _negative_prompt_for_model(str(entry.get("id") or ""))
                if is_image_gen:
                    negative = neg_text or ""
            add(DocSample(
                id=f"modelguide-{_slug(entry['id'])}-{index}",
                source_doc=MODEL_GUIDE.name,
                surface=surface,
                model_id=str(entry["id"]),
                model_name=str(entry.get("name") or entry["id"]),
                title=f"Model demo sample {index}",
                prompt=prompt,
                entry=entry,
                negative=negative,
                settings=settings,
                reference_image=str(reference_image) if _supports_vision_chat(entry) and surface == "chat" else "",
            ))

    for entry in active_image:
        if not entry.get("supports_img2img"):
            continue
        demo_prompt = get_model_demo(entry)["samples"][0]
        add(DocSample(
            id=f"reference-{_slug(entry['id'])}",
            source_doc=IMAGE_FEATURE_GUIDE.name,
            surface="reference_image",
            model_id=str(entry["id"]),
            model_name=str(entry.get("name") or entry["id"]),
            title="Reference image generation",
            prompt=demo_prompt,
            entry=entry,
            settings=_settings_for_entry(entry, args),
            reference_image=str(reference_image),
            notes="Validates the Image Gen reference-image path for models with supports_img2img=true.",
        ))

    for entry in models:
        if not _supports_vision_chat(entry):
            continue
        if str(entry.get("id")) not in {"gemma3:4b-vision", "gemma3:12b-vision", "minicpm-v-vision"}:
            continue
        add(DocSample(
            id=f"img2prompt-{_slug(entry['id'])}",
            source_doc=IMAGE_FEATURE_GUIDE.name,
            surface="image_to_prompt",
            model_id=str(entry["id"]),
            model_name=str(entry.get("name") or entry["id"]),
            title="Analyze to Prompt",
            prompt="Analyze this image and generate the prompt.",
            entry=entry,
            reference_image=str(reference_image),
            notes="Uses the same system/user prompt and image payload as App._analyze_reference_image.",
        ))

    seen: set[str] = set()
    unique: list[DocSample] = []
    for sample in samples:
        if sample.id in seen:
            continue
        seen.add(sample.id)
        unique.append(sample)
    return unique


def _result_from_error(sample: DocSample, started_at: str, started: float, error: str, status: str = "failed") -> DocResult:
    ended_at = _now_iso()
    entry = sample.entry or {}
    return DocResult(
        id=sample.id,
        source_doc=sample.source_doc,
        surface=sample.surface,
        model_id=sample.model_id,
        model_name=sample.model_name,
        title=sample.title,
        prompt=sample.prompt,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=round(time.perf_counter() - started, 2),
        error=error,
        settings=sample.settings,
        docs_settings=sample.docs_settings,
        input_path=sample.input_path,
        reference_image=sample.reference_image,
        min_ram_gb=float(entry.get("min_ram_gb") or 0),
        min_vram_gb=float(entry.get("min_vram_gb") or 0),
        minimum_sku=_minimum_sku(entry),
    )


def _image_quality(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    with Image.open(path) as img:
        width, height = img.size
        probe = img.convert("RGB")
        probe.thumbnail((192, 192))
        stat = ImageStat.Stat(probe)
        avg_stddev = sum(stat.stddev) / max(len(stat.stddev), 1)
        extrema = probe.getextrema()
        flat_channels = sum(1 for lo, hi in extrema if abs(hi - lo) < 3)
        pixel_iter = probe.get_flattened_data() if hasattr(probe, "get_flattened_data") else probe.getdata()
        pixels = list(pixel_iter)
        unique_sample = len(set(pixels[:: max(1, len(pixels) // 4096)]))
    file_size = path.stat().st_size
    bad = file_size < 10_000 or width < 64 or height < 64 or avg_stddev < 4.0 or flat_channels >= 3 or unique_sample < 24
    return {
        "width": width,
        "height": height,
        "file_size": file_size,
        "avg_stddev": round(avg_stddev, 2),
        "unique_sample": unique_sample,
        "bad": bad,
        "reason": "blank/low-detail heuristic" if bad else "readable image with adequate variation",
    }


def _save_image(out_dir: Path, sample: DocSample, image_bytes: bytes) -> Path:
    path = out_dir / "images" / f"{_slug(sample.id)}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    return path


def _wait_for_comfyui(app: App, timeout_s: int) -> bool:
    started_by_tool = False
    if not app.comfyui.is_running():
        if not app._start_comfyui_process():
            raise RuntimeError("LocalAI could not start ComfyUI. Check comfyui.log.")
        started_by_tool = True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            app.update()
        except Exception:
            pass
        if app.comfyui.is_running():
            return started_by_tool
        proc = getattr(app, "comfyui_process", None)
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during startup with code {proc.returncode}. Check comfyui.log.")
        time.sleep(2)
    raise TimeoutError(f"ComfyUI did not become ready within {timeout_s}s.")


def _restart_comfyui_for_model_support(app: App, timeout_s: int) -> None:
    proc = getattr(app, "comfyui_process", None)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
    app.comfyui_process = None
    try:
        app._close_comfyui_log_handle()
    except Exception:
        pass
    try:
        app._kill_orphan_comfyui_processes()
        time.sleep(1)
    except Exception:
        pass
    app.comfyui.reconnect()
    if not app._start_comfyui_process():
        raise RuntimeError("LocalAI could not restart ComfyUI after installing model support.")
    setattr(app, "_doc_validator_started_comfyui", True)
    _wait_for_comfyui(app, timeout_s)


def _stop_owned_comfyui(app: App | None, started_by_tool: bool) -> None:
    if app is None or not (started_by_tool or getattr(app, "_doc_validator_started_comfyui", False)):
        return
    proc = getattr(app, "comfyui_process", None)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
    try:
        app._close_comfyui_log_handle()
    except Exception:
        pass


def _loaded_model_for_entry(entry: dict[str, Any], loaded: list[str]) -> str:
    expected = str(entry.get("comfyui_model") or "")
    for filename in loaded:
        if filename == expected or Path(filename).name == expected or filename.endswith("/" + expected):
            return filename
    return ""


def _run_image_sample(app: App, loaded: list[str], sample: DocSample, out_dir: Path) -> DocResult:
    started = time.perf_counter()
    started_at = _now_iso()
    entry = sample.entry
    filename = _loaded_model_for_entry(entry, loaded)
    if not filename:
        return _result_from_error(sample, started_at, started, "ComfyUI model file is not loaded on this system.", "not_run")
    blocked = content_filter.check_prompt(sample.prompt)
    if blocked:
        return _result_from_error(sample, started_at, started, f"Prompt blocked by content filter term: {blocked}")
    progress: list[str] = []

    def progress_cb(message: str) -> None:
        progress.append(message)
        if len(progress) > 10:
            del progress[:-10]
        print(f"[image] {sample.model_name} / {sample.title}: {message}", flush=True)

    try:
        if not app._ensure_image_model_runtime_support(filename, prompt=False):
            raise RuntimeError(f"Could not prepare ComfyUI support files for {filename}.")
        if app._image_model_runtime_needs_restart(filename) or not app.comfyui.is_running():
            progress_cb("Restarting ComfyUI to load model support ...")
            _restart_comfyui_for_model_support(app, 120)
        reference = Path(sample.reference_image) if sample.reference_image else None
        denoise = float((entry.get("img2img_workflows") or {}).get("denoise_default") or 0.75)
        image_bytes = app.comfyui.generate_image(
            model_filename=filename,
            positive_prompt=sample.prompt,
            negative_prompt=sample.negative or app._default_negative_prompt_for_entry(entry),
            width=int(sample.settings["width"]),
            height=int(sample.settings["height"]),
            steps=int(sample.settings["steps"]),
            cfg_scale=float(sample.settings["cfg"]),
            seed=int(sample.settings["seed"]),
            sampler_name=str(sample.settings["sampler"]),
            scheduler=str(sample.settings["scheduler"]),
            reference_image_path=str(reference) if reference else None,
            denoise=denoise,
            progress_cb=progress_cb,
            stop_event=threading.Event(),
        )
        path = _save_image(out_dir, sample, image_bytes)
        quality = _image_quality(path)
        replacement_prompt = ""
        original_path = ""
        status = "passed"
        if quality.get("bad"):
            original_path = str(path)
            replacement_prompt = get_model_demo(entry)["samples"][0]
            replacement_blocked = content_filter.check_prompt(replacement_prompt)
            if replacement_blocked:
                ended_at = _now_iso()
                return DocResult(
                    id=sample.id,
                    source_doc=sample.source_doc,
                    surface=sample.surface,
                    model_id=sample.model_id,
                    model_name=sample.model_name,
                    title=sample.title,
                    prompt=sample.prompt,
                    status="failed",
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_s=round(time.perf_counter() - started, 2),
                    output_path=str(path),
                    original_output_path=original_path,
                    replacement_prompt=replacement_prompt,
                    error=f"Replacement prompt blocked by content filter term: {replacement_blocked}",
                    settings=sample.settings,
                    docs_settings=sample.docs_settings,
                    input_path=sample.input_path,
                    reference_image=sample.reference_image,
                    min_ram_gb=float(entry.get("min_ram_gb") or 0),
                    min_vram_gb=float(entry.get("min_vram_gb") or 0),
                    minimum_sku=_minimum_sku(entry),
                    image_quality=quality,
                    metric_label="Image",
                    metric_value="0 replacement PNGs",
                )
            discarded = out_dir / "discarded" / path.name
            discarded.parent.mkdir(parents=True, exist_ok=True)
            path.replace(discarded)
            image_bytes = app.comfyui.generate_image(
                model_filename=filename,
                positive_prompt=replacement_prompt,
                negative_prompt=sample.negative or app._default_negative_prompt_for_entry(entry),
                width=int(sample.settings["width"]),
                height=int(sample.settings["height"]),
                steps=int(sample.settings["steps"]),
                cfg_scale=float(sample.settings["cfg"]),
                seed=int(sample.settings["seed"]) + 997,
                sampler_name=str(sample.settings["sampler"]),
                scheduler=str(sample.settings["scheduler"]),
                reference_image_path=str(reference) if reference else None,
                denoise=denoise,
                progress_cb=progress_cb,
                stop_event=threading.Event(),
            )
            path = _save_image(out_dir, sample, image_bytes)
            quality = _image_quality(path)
            status = "passed_replaced" if not quality.get("bad") else "failed"
        ended_at = _now_iso()
        return DocResult(
            id=sample.id,
            source_doc=sample.source_doc,
            surface=sample.surface,
            model_id=sample.model_id,
            model_name=sample.model_name,
            title=sample.title,
            prompt=sample.prompt,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=round(time.perf_counter() - started, 2),
            output_path=str(path),
            original_output_path=original_path,
            replacement_prompt=replacement_prompt,
            error="" if status != "failed" else f"Generated image failed quality check: {quality}",
            issue_fixed="Original docs prompt generated a low-detail/blank-looking image; replacement output used the fallback prompt shown here." if replacement_prompt else "",
            settings=sample.settings,
            docs_settings=sample.docs_settings,
            input_path=sample.input_path,
            reference_image=sample.reference_image,
            min_ram_gb=float(entry.get("min_ram_gb") or 0),
            min_vram_gb=float(entry.get("min_vram_gb") or 0),
            minimum_sku=_minimum_sku(entry),
            image_quality=quality,
            metric_label="Image",
            metric_value="1 PNG",
        )
    except Exception as exc:
        return _result_from_error(sample, started_at, started, str(exc))


def _chat_payload(tag: str, messages: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": tag,
        "messages": messages,
        "stream": True,
        "keep_alive": 0 if args.unload_chat else "5m",
        "options": {
            "temperature": args.chat_temperature,
            "num_predict": args.chat_max_tokens,
            "num_gpu": -1,
        },
    }
    think = think_option_for_model(tag)
    if think is not None:
        payload["think"] = think
    return payload


def _run_chat_like_sample(sample: DocSample, out_dir: Path, args: argparse.Namespace, *, analyze_prompt: bool = False) -> DocResult:
    started = time.perf_counter()
    started_at = _now_iso()
    entry = sample.entry
    tag = str(entry.get("ollama_tag") or "")
    local = _local_names()
    if not _ollama_tag_is_local(tag, local):
        return _result_from_error(sample, started_at, started, f"Ollama model is not installed locally: {tag}", "not_run")
    try:
        import requests

        messages: list[dict[str, Any]]
        if analyze_prompt:
            messages = [
                {"role": "system", "content": App._ANALYZE_SYSTEM_PROMPT},
                {"role": "user", "content": "Analyze this image and generate the prompt."},
            ]
        else:
            messages = [{"role": "user", "content": sample.prompt}]
        if sample.reference_image:
            raw = Path(sample.reference_image).read_bytes()
            messages[-1]["images"] = [base64.b64encode(raw).decode("utf-8")]
        payload = _chat_payload(tag, messages, args)
        if analyze_prompt:
            min_vram = float(entry.get("min_vram_gb") or App.VISION_MIN_VRAM_GB)
            vram = _actual_vram_gb()
            payload["options"]["temperature"] = 0.3
            payload["options"]["num_predict"] = min(args.chat_max_tokens, 240)
            payload["options"]["num_gpu"] = 0 if vram < min_vram else -1
        response_parts: list[str] = []
        stats: dict[str, Any] = {}
        first_token_at = 0.0
        token_count = 0
        with requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            stream=True,
            timeout=(15, args.chat_timeout),
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                token = data.get("message", {}).get("content", "")
                if token:
                    if token_count == 0:
                        first_token_at = time.perf_counter()
                    token_count += 1
                    response_parts.append(token)
                if data.get("done"):
                    stats = data
                    break
        text = strip_think_blocks("".join(response_parts)).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        out_path = out_dir / "chat" / f"{_slug(sample.id)}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        total_ns = int(stats.get("total_duration") or 0)
        eval_ns = int(stats.get("eval_duration") or 0)
        load_ns = int(stats.get("load_duration") or 0)
        eval_count = int(stats.get("eval_count") or token_count or 0)
        prompt_count = int(stats.get("prompt_eval_count") or 0)
        ended_at = _now_iso()
        duration = time.perf_counter() - started
        return DocResult(
            id=sample.id,
            source_doc=sample.source_doc,
            surface=sample.surface,
            model_id=sample.model_id,
            model_name=sample.model_name,
            title=sample.title,
            prompt=sample.prompt,
            status="passed",
            started_at=started_at,
            ended_at=ended_at,
            duration_s=round(duration, 2),
            output_text=text,
            output_path=str(out_path),
            token_count=eval_count,
            prompt_tokens=prompt_count,
            tokens_per_sec=round(eval_count / (eval_ns / 1_000_000_000), 2) if eval_ns else 0.0,
            ttft_s=round(first_token_at - started, 2) if first_token_at else 0.0,
            load_time_s=round(load_ns / 1_000_000_000, 2),
            generation_time_s=round(eval_ns / 1_000_000_000, 2),
            metric_label="Tokens",
            metric_value=f"{eval_count} generated",
            settings={
                "temperature": payload["options"]["temperature"],
                "num_predict": payload["options"]["num_predict"],
                "num_gpu": payload["options"]["num_gpu"],
                "keep_alive": payload["keep_alive"],
            },
            input_path=sample.input_path,
            reference_image=sample.reference_image,
            min_ram_gb=float(entry.get("min_ram_gb") or 0),
            min_vram_gb=float(entry.get("min_vram_gb") or 0),
            minimum_sku=_minimum_sku(entry),
        )
    except Exception as exc:
        return _result_from_error(sample, started_at, started, str(exc))


def _actual_vram_gb() -> float:
    gpus = get_gpu_info()
    if not gpus:
        return 0.0
    return float(gpus[0].get("vram_total_mb") or 0) / 1024.0


def _toolbox_fixture_paths(out_dir: Path) -> dict[str, Path]:
    root = out_dir / "toolbox"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "audio": phase1_adapters._ensure_speech_wav(root),
        "image": phase1_adapters._ensure_vision_image(root, "doc_sample"),
        "table": phase1_adapters._ensure_vision_image(root, "doc_sample_table", table=True),
    }


def _run_toolbox_sample(sample: DocSample, out_dir: Path) -> DocResult:
    started = time.perf_counter()
    started_at = _now_iso()
    entry = sample.entry
    fixtures = _toolbox_fixture_paths(out_dir)
    progress: list[str] = []

    def progress_cb(message: str) -> None:
        progress.append(message)
        print(f"[toolbox] {sample.model_name} / {sample.title}: {message}", flush=True)

    try:
        output_root = out_dir / "toolbox_outputs"
        if sample.model_id in {"whisper-large-v3-turbo", "whisper-v3-turbo-gpu"}:
            input_path = fixtures["audio"]
            result = workflows.transcribe(input_path, entry, progress_cb=progress_cb)
        elif sample.model_id == "speecht5-tts":
            input_path = Path("")
            text = sample.prompt[:220]
            result = workflows.synthesize(text, entry, output_dir=output_root, progress_cb=progress_cb)
        elif sample.model_id == "all-minilm":
            input_path = Path("")
            corpus = [
                "LocalAI Studio runs models privately on your computer.",
                "The benchmark gallery helps pick reliable offline models.",
                "The weather forecast is unrelated to local AI validation.",
                "Image generation uses ComfyUI and local checkpoints.",
            ]
            result = workflows.embed_and_rank(sample.prompt, corpus, entry, progress_cb=progress_cb)
        elif sample.model_id == "table-transformer":
            input_path = fixtures["table"]
            result = workflows.detect_table(input_path, entry, output_dir=output_root, progress_cb=progress_cb)
        elif sample.model_id.startswith("trocr"):
            input_path = fixtures["image"]
            result = workflows.read_image(input_path, entry, progress_cb=progress_cb)
        elif sample.model_id == "florence-2-base":
            input_path = fixtures["image"]
            mode = "caption" if "caption" in sample.prompt.lower() or "describe" in sample.prompt.lower() else "ocr"
            result = workflows.read_image(input_path, entry, mode=mode, progress_cb=progress_cb)
        elif sample.model_id == "phi-4-multimodal":
            input_path = fixtures["image"]
            adapter = phase1_adapters.run_phi_text(entry, output_root, prompt=sample.prompt)
            if adapter.get("status") == "error":
                raise RuntimeError(str(adapter.get("error") or "Phi utility adapter failed."))
            output_text = str(adapter.get("output_text") or "")
            if not output_text.strip():
                raise RuntimeError("Toolbox workflow returned empty output.")
            ended_at = _now_iso()
            return DocResult(
                id=sample.id,
                source_doc=sample.source_doc,
                surface=sample.surface,
                model_id=sample.model_id,
                model_name=sample.model_name,
                title=sample.title,
                prompt=sample.prompt,
                status="passed",
                started_at=started_at,
                ended_at=ended_at,
                duration_s=round(time.perf_counter() - started, 2),
                output_text=output_text,
                output_path=str(adapter.get("image") or adapter.get("audio") or ""),
                metric_label=str(adapter.get("metric_label") or "Utility"),
                metric_value=str(adapter.get("metric_value") or ""),
                settings={"workflow_progress": progress[-10:], "adapter": "phi_ollama_text"},
                input_path=str(input_path),
                min_ram_gb=float(entry.get("min_ram_gb") or 0),
                min_vram_gb=float(entry.get("min_vram_gb") or 0),
                minimum_sku=_minimum_sku(entry),
            )
        else:
            return _result_from_error(sample, started_at, started, f"No Toolbox route for {sample.model_id}", "not_run")
        ended_at = _now_iso()
        output_text = result.output_text or ""
        if not output_text.strip():
            raise RuntimeError("Toolbox workflow returned empty output.")
        return DocResult(
            id=sample.id,
            source_doc=sample.source_doc,
            surface=sample.surface,
            model_id=sample.model_id,
            model_name=sample.model_name,
            title=sample.title,
            prompt=sample.prompt,
            status="passed",
            started_at=started_at,
            ended_at=ended_at,
            duration_s=round(time.perf_counter() - started, 2),
            output_text=output_text,
            output_path=str(result.output_path or ""),
            metric_label="Utility",
            metric_value=result.metadata.get("elapsed_s", "") if isinstance(result.metadata, dict) else "",
            settings={"workflow_progress": progress[-10:]},
            input_path=str(input_path) if input_path else "",
            min_ram_gb=float(entry.get("min_ram_gb") or 0),
            min_vram_gb=float(entry.get("min_vram_gb") or 0),
            minimum_sku=_minimum_sku(entry),
        )
    except Exception as exc:
        return _result_from_error(sample, started_at, started, str(exc))


def _write_json(out_dir: Path, samples: list[DocSample], results: list[DocResult], meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(
        json.dumps([asdict(s) for s in samples], indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if results:
        with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "status", "surface", "source_doc", "model_id", "model_name", "title",
                "duration_s", "token_count", "tokens_per_sec", "metric_label", "metric_value",
                "output_path", "error",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in results:
                payload = asdict(row)
                writer.writerow({field: payload.get(field, "") for field in fields})


def _dedupe_results(results: list[DocResult]) -> list[DocResult]:
    latest: dict[str, DocResult] = {}
    for result in results:
        latest[result.id] = result
    return list(latest.values())


def _replace_result(results: list[DocResult], result: DocResult) -> None:
    for index, existing in enumerate(results):
        if existing.id == result.id:
            results[index] = result
            return
    results.append(result)


def _rel_link(path: str, out_dir: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.relative_to(out_dir).as_posix()
    except ValueError:
        return p.as_uri() if p.exists() else html.escape(path)


def _status_class(status: str) -> str:
    if status == "passed":
        return "ok"
    if status == "passed_replaced":
        return "warn"
    if status == "not_run":
        return "muted"
    return "fail"


def _theme_css() -> str:
    return """
:root {
  color-scheme: light;
  --cp-bg: #f7f4ef;
  --cp-bg-elevated: #fcfbf8;
  --cp-surface: #ffffff;
  --cp-surface-soft: #f5f5f5;
  --cp-border: #dedede;
  --cp-border-strong: #919191;
  --cp-text: #242424;
  --cp-text-muted: #5c5c5c;
  --cp-text-soft: #6f6f6f;
  --cp-accent: #b11f4b;
  --cp-accent-hover: #9a1a41;
  --cp-accent-soft: rgba(177, 31, 75, 0.08);
  --cp-accent-fg: #ffffff;
  --cp-success: #16a34a;
  --cp-danger: #dc2626;
  --cp-warning: #f59e0b;
  --cp-link: #0078d4;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.12);
  --cp-overlay: rgba(255, 255, 255, 0.8);
  --cp-panel: rgba(255, 255, 255, 0.86);
  --cp-panel-strong: rgba(255, 255, 255, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.55);
  --cp-highlight: rgba(177, 31, 75, 0.12);
}
html[data-theme="dark"] {
  color-scheme: dark;
  --cp-bg: #3d3b3a;
  --cp-bg-elevated: #343231;
  --cp-surface: #292929;
  --cp-surface-soft: #2e2e2e;
  --cp-border: #474747;
  --cp-border-strong: #5f5f5f;
  --cp-text: #dedede;
  --cp-text-muted: #919191;
  --cp-text-soft: #b0b0b0;
  --cp-accent: #fd8ea1;
  --cp-accent-hover: #fb7b91;
  --cp-accent-soft: rgba(253, 142, 161, 0.14);
  --cp-accent-fg: #1a1a1a;
  --cp-success: #4ade80;
  --cp-danger: #f87171;
  --cp-warning: #fbbf24;
  --cp-link: #4da6ff;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
  --cp-overlay: rgba(41, 41, 41, 0.88);
  --cp-panel: rgba(41, 41, 41, 0.72);
  --cp-panel-strong: rgba(41, 41, 41, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.04);
  --cp-highlight: rgba(253, 142, 161, 0.12);
}
"""


def write_html_report(out_dir: Path, samples: list[DocSample], results: list[DocResult], meta: dict[str, Any]) -> Path:
    by_id = {r.id: r for r in results}
    rows = [by_id.get(sample.id) for sample in samples if by_id.get(sample.id)]
    passed = sum(1 for r in rows if r.status == "passed")
    failed = sum(1 for r in rows if r.status == "failed")
    not_run = sum(1 for r in rows if r.status == "not_run")
    replaced = sum(1 for r in rows if r.status == "passed_replaced")
    duration = sum(r.duration_s for r in rows)
    surface_counts: dict[str, int] = {}
    for sample in samples:
        surface_counts[sample.surface] = surface_counts.get(sample.surface, 0) + 1

    cards = []
    for r in rows:
        output = ""
        link = _rel_link(r.output_path, out_dir)
        if link and r.output_path.lower().endswith(".png"):
            output = f'<a href="{html.escape(link)}" target="_blank"><img class="thumb" src="{html.escape(link)}" alt="{html.escape(r.title)}"></a>'
        elif link:
            output = f'<a href="{html.escape(link)}" target="_blank">Open output</a>'
        if r.output_text:
            output += f"<pre>{html.escape(r.output_text)}</pre>"
        if r.error:
            output += f'<pre class="error">{html.escape(r.error)}</pre>'
        settings = json.dumps(r.settings, indent=2, ensure_ascii=False)
        docs_settings = json.dumps(r.docs_settings, indent=2, ensure_ascii=False) if r.docs_settings else ""
        quality = json.dumps(r.image_quality, indent=2, ensure_ascii=False) if r.image_quality else ""
        cards.append(f"""
<article class="case {html.escape(_status_class(r.status))}" data-surface="{html.escape(r.surface)}" data-status="{html.escape(r.status)}">
  <div class="case-head">
    <div>
      <div class="eyebrow">{html.escape(r.source_doc)} / {html.escape(r.surface)}</div>
      <h3>{html.escape(r.model_name)} - {html.escape(r.title)}</h3>
    </div>
    <span class="status">{html.escape(r.status)}</span>
  </div>
  <div class="metrics">
    <span>{r.duration_s:.1f}s</span>
    <span>{html.escape(r.metric_label or "Tokens")}: {html.escape(str(r.metric_value or r.token_count))}</span>
    <span>{r.tokens_per_sec:.2f} tok/s</span>
    <span>TTFT {r.ttft_s:.2f}s</span>
    <span>Min SKU: {html.escape(r.minimum_sku)}</span>
    <span>RAM {r.min_ram_gb:g} GB / VRAM {r.min_vram_gb:g} GB</span>
  </div>
  <details open>
    <summary>Prompt and settings</summary>
    <pre>{html.escape(r.prompt)}</pre>
    <div class="settings-grid">
      <div><strong>UI/backend settings</strong><pre>{html.escape(settings)}</pre></div>
      <div><strong>Doc card settings</strong><pre>{html.escape(docs_settings or "{}")}</pre></div>
      <div><strong>Image quality</strong><pre>{html.escape(quality or "{}")}</pre></div>
    </div>
  </details>
  <div class="output">{output}</div>
  {f'<p class="fix">Fix/replacement: {html.escape(r.issue_fixed)} {html.escape(r.replacement_prompt)}</p>' if r.issue_fixed or r.replacement_prompt else ''}
</article>
""")

    surface_html = "".join(f"<span>{html.escape(k)}: {v}</span>" for k, v in sorted(surface_counts.items()))
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalAI doc sample validation</title>
<script>
  (() => {{
    const param = new URLSearchParams(window.location.search).get("clawpilotTheme");
    const theme =
      param || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
  }})();
</script>
<style>
{_theme_css()}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--cp-bg); color: var(--cp-text); font-family: "Segoe UI", Aptos, Calibri, -apple-system, BlinkMacSystemFont, sans-serif; }}
a {{ color: var(--cp-link); }}
header {{ padding: 28px 32px; background: var(--cp-panel-strong); border-bottom: 1px solid var(--cp-border); position: sticky; top: 0; z-index: 5; }}
h1 {{ margin: 0 0 8px; }}
.sub {{ color: var(--cp-text-muted); margin: 0; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; padding: 20px 32px; }}
.stat {{ background: var(--cp-surface); border: 1px solid var(--cp-border); border-radius: 16px; padding: 16px; box-shadow: 0 0 2px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.14); }}
.stat strong {{ display: block; font-size: 24px; }}
.stat span {{ color: var(--cp-text-muted); }}
.filters {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 0 32px 16px; }}
.filters button {{ border: 1px solid var(--cp-border); background: var(--cp-surface); color: var(--cp-text); border-radius: 0.625rem; padding: 8px 12px; cursor: pointer; }}
.filters button:hover {{ border-color: var(--cp-accent); }}
.surface-counts {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 0 32px 12px; color: var(--cp-text-muted); }}
.surface-counts span, .metrics span {{ border: 1px solid var(--cp-border); background: var(--cp-surface-soft); border-radius: 999px; padding: 4px 9px; }}
main {{ display: grid; gap: 14px; padding: 0 32px 36px; }}
.case {{ background: var(--cp-surface); border: 1px solid var(--cp-border); border-radius: 16px; padding: 16px; box-shadow: 0 0 2px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.14); }}
.case.ok {{ border-left: 6px solid var(--cp-success); }}
.case.warn {{ border-left: 6px solid var(--cp-warning); }}
.case.fail {{ border-left: 6px solid var(--cp-danger); }}
.case.muted {{ border-left: 6px solid var(--cp-border-strong); }}
.case-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; }}
.eyebrow {{ color: var(--cp-text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
h3 {{ margin: 4px 0 0; }}
.status {{ color: var(--cp-accent); background: var(--cp-accent-soft); border: 1px solid var(--cp-border); border-radius: 999px; padding: 5px 10px; }}
.metrics {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; font-size: 13px; }}
details {{ border-top: 1px solid var(--cp-border); padding-top: 10px; }}
summary {{ cursor: pointer; color: var(--cp-accent); font-weight: 600; }}
pre {{ white-space: pre-wrap; overflow: auto; background: var(--cp-surface-soft); color: var(--cp-text); border: 1px solid var(--cp-border); border-radius: 0.625rem; padding: 10px; font-family: Consolas, "Courier New", Courier, monospace; font-size: 12px; }}
.settings-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }}
.output {{ margin-top: 12px; }}
.thumb {{ max-width: 360px; max-height: 260px; border: 1px solid var(--cp-border); border-radius: 0.625rem; background: var(--cp-surface-soft); }}
.error {{ border-color: var(--cp-danger); }}
.fix {{ color: var(--cp-warning); }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>LocalAI Studio doc sample validation</h1>
  <p class="sub">Started {html.escape(str(meta.get("started_at", "")))}; ended {html.escape(str(meta.get("ended_at", "")))}. Output folder: <code>{html.escape(str(out_dir))}</code></p>
</header>
<section class="summary">
  <div class="stat"><strong>{len(samples)}</strong><span>Inventoried samples</span></div>
  <div class="stat"><strong>{len(rows)}</strong><span>Executed samples</span></div>
  <div class="stat"><strong>{passed}</strong><span>Passed</span></div>
  <div class="stat"><strong>{failed}</strong><span>Failed</span></div>
  <div class="stat"><strong>{not_run}</strong><span>Not run</span></div>
  <div class="stat"><strong>{replaced}</strong><span>Images replaced</span></div>
  <div class="stat"><strong>{duration/60:.1f}</strong><span>Total runtime minutes</span></div>
</section>
<section class="surface-counts">{surface_html}</section>
<section class="summary">
  <div class="stat"><strong>{html.escape(str(meta.get("gpu", "")))}</strong><span>GPU</span></div>
  <div class="stat"><strong>{html.escape(str(meta.get("ram_gb", "")))}</strong><span>RAM GB</span></div>
  <div class="stat"><strong>{html.escape(str(meta.get("vram_gb", "")))}</strong><span>VRAM GB</span></div>
  <div class="stat"><strong>{html.escape(str(meta.get("bugs_fixed") or "None during this run"))}</strong><span>Bugs found/fixed</span></div>
</section>
<section class="filters">
  <button type="button" data-filter="all">All</button>
  <button type="button" data-filter="failed">Failed</button>
  <button type="button" data-filter="image">Image</button>
  <button type="button" data-filter="reference_image">Reference image</button>
  <button type="button" data-filter="image_to_prompt">Image to prompt</button>
  <button type="button" data-filter="chat">Chat</button>
  <button type="button" data-filter="toolbox">Toolbox</button>
</section>
<main>{''.join(cards)}</main>
<script>
document.querySelectorAll('[data-filter]').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const filter = btn.dataset.filter;
    document.querySelectorAll('.case').forEach(card => {{
      const show = filter === 'all' || (filter === 'failed' ? card.dataset.status === 'failed' : card.dataset.surface === filter);
      card.classList.toggle('hidden', !show);
    }});
  }});
}});
</script>
</body>
</html>
"""
    path = out_dir / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--source", action="append", help="Limit to source doc filename. Repeatable.")
    parser.add_argument("--surface", action="append", help="Limit to surface: chat, image, reference_image, image_to_prompt, toolbox.")
    parser.add_argument("--model", action="append", help="Limit to catalog model id. Repeatable.")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--sampler")
    parser.add_argument("--scheduler")
    parser.add_argument("--max-dimension", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--chat-timeout", type=int, default=900)
    parser.add_argument("--chat-max-tokens", type=int, default=1024)
    parser.add_argument("--chat-temperature", type=float, default=0.7)
    parser.add_argument("--unload-chat", action="store_true", default=False)
    parser.add_argument("--bugs-fixed", default="", help="Short bug-fix note to include in the HTML report summary.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _run_sample(
    sample: DocSample,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    app: App | None,
    loaded: list[str],
) -> DocResult:
    print(f"[{sample.surface}] {sample.model_name} / {sample.title}", flush=True)
    if sample.surface in {"image", "reference_image"}:
        if app is None:
            return _result_from_error(sample, _now_iso(), time.perf_counter(), "Image app/ComfyUI was not initialized.")
        return _run_image_sample(app, loaded, sample, out_dir)
    if sample.surface == "image_to_prompt":
        return _run_chat_like_sample(sample, out_dir, args, analyze_prompt=True)
    if sample.surface == "chat":
        return _run_chat_like_sample(sample, out_dir, args)
    if sample.surface == "toolbox":
        return _run_toolbox_sample(sample, out_dir)
    return _result_from_error(sample, _now_iso(), time.perf_counter(), f"Unknown surface: {sample.surface}", "not_run")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_image = args.reference_image or _find_reference_image(SAFE_REFERENCE_DIRS) or _create_reference_fixture(out_dir)
    models = catalog.load_catalog(ROOT / "models_catalog.json")
    samples = collect_samples(models, args, reference_image)
    gpu = detect_gpu_cached(auto_fix=False)
    ram = get_ram_info()
    gpus = get_gpu_info()
    vram_gb = round(float((gpus[0].get("vram_total_mb") if gpus else 0) or 0) / 1024, 1)
    meta = {
        "started_at": _now_iso(),
        "reference_image": str(reference_image),
        "sample_count": len(samples),
        "gpu": str(gpu),
        "ram_gb": round(float(ram.get("total_mb") or 0) / 1024, 1),
        "vram_gb": vram_gb,
        "bugs_fixed": args.bugs_fixed,
    }

    results: list[DocResult] = []
    if args.resume and (out_dir / "results.json").exists():
        data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        results = _dedupe_results([DocResult(**row) for row in data])
    completed = {r.id for r in results if r.status == "passed"}

    _write_json(out_dir, samples, results, meta)
    if args.inventory_only:
        meta["ended_at"] = _now_iso()
        _write_json(out_dir, samples, results, meta)
        report = write_html_report(out_dir, samples, results, meta)
        print(f"Inventory: {len(samples)} samples")
        print(f"Report: {report}")
        return 0

    needs_image = any(sample.surface in {"image", "reference_image"} and sample.id not in completed for sample in samples)
    app: App | None = None
    started_comfyui = False
    loaded: list[str] = []
    try:
        if needs_image:
            app = App()
            app.withdraw()
            app.update()
            app._switch_page("image_gen")
            app.update()
            started_comfyui = _wait_for_comfyui(app, args.startup_timeout)
            app.comfyui.clear_queue()
            loaded = sorted(app.comfyui.get_model_list(), key=str.lower)

        for sample in samples:
            if sample.id in completed:
                continue
            result = _run_sample(sample, out_dir, args, app=app, loaded=loaded)
            _replace_result(results, result)
            _write_json(out_dir, samples, results, meta)
            write_html_report(out_dir, samples, results, meta)
            if result.status == "failed" and args.fail_fast:
                break
            if app is not None and sample.surface in {"image", "reference_image"}:
                try:
                    app.comfyui.free_vram()
                except Exception:
                    pass
        meta["ended_at"] = _now_iso()
        _write_json(out_dir, samples, results, meta)
        report = write_html_report(out_dir, samples, results, meta)
        failed = sum(1 for row in results if row.status == "failed")
        not_run = sum(1 for row in results if row.status == "not_run")
        print(f"Report: {report}", flush=True)
        replaced = sum(1 for row in results if row.status == "passed_replaced")
        print(f"Passed={sum(r.status == 'passed' for r in results)} Replaced={replaced} Failed={failed} NotRun={not_run}", flush=True)
        return 1 if failed or replaced else 0
    finally:
        _stop_owned_comfyui(app, started_comfyui)
        if app is not None:
            try:
                app.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

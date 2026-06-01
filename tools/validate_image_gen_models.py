#!/usr/bin/env python
"""
Run end-to-end LocalAI image-generation validation for locally installed models.

The runner starts ComfyUI through the App startup path, uses the same
ComfyUIClient.generate_image() helper as the Image Gen page, and writes every
generated PNG plus a JSON/CSV/Markdown report to a timestamped review folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import catalog, content_filter  # noqa: E402
from src.app import App  # noqa: E402
from src.comfyui_client import ComfyUIError  # noqa: E402


SAFE_TXT2IMG_PROMPT = (
    "A professional product photo of a polished ceramic coffee mug on a marble "
    "desk, morning window light, realistic, high detail, clean commercial style"
)
SAFE_IMG2IMG_PROMPT = (
    "Transform the reference into a polished professional studio illustration, "
    "preserving the main subject composition, clean lighting, high detail"
)
SAFE_BLOCKED_FIXTURE_PROMPT = (
    "An enterprise content safety test poster in a modern office, clearly showing "
    "the letters NSFW on a red warning label, no people, no violence, clean graphic design"
)


@dataclass
class ValidationCase:
    model_id: str
    model_name: str
    model_filename: str
    case: str
    status: str
    duration_s: float = 0.0
    output_path: str = ""
    width: int = 0
    height: int = 0
    steps: int = 0
    cfg: float = 0.0
    sampler: str = ""
    scheduler: str = ""
    seed: int = 0
    prompt_kind: str = "safe"
    blocked_term: str = ""
    reference_image: str = ""
    error: str = ""
    progress_tail: list[str] = field(default_factory=list)


def _slug(value: str, max_len: int = 90) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return (slug or "unknown")[:max_len]


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _default_output_dir() -> Path:
    return ROOT / "image_validation_results" / _timestamp()


def _find_reference_image(paths: list[Path]) -> Path | None:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for base in paths:
        if base.is_file() and base.suffix.lower() in exts:
            return base
        if not base.exists() or not base.is_dir():
            continue
        for path in base.iterdir():
            if path.is_file() and path.suffix.lower() in exts:
                return path
        for root, _dirs, files in os.walk(base):
            for name in files:
                path = Path(root) / name
                if path.suffix.lower() in exts:
                    return path
    return None


def _create_synthetic_reference_image(out_dir: Path) -> Path:
    path = out_dir / "synthetic_reference.png"
    if path.exists():
        return path
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (768, 768), "#edf2f7")
    draw = ImageDraw.Draw(img)
    draw.rectangle((120, 170, 648, 598), fill="#ffffff", outline="#2563eb", width=8)
    draw.ellipse((240, 230, 528, 518), fill="#f59e0b", outline="#92400e", width=6)
    draw.rectangle((306, 510, 462, 620), fill="#2563eb")
    draw.text((190, 90), "LocalAI reference fixture", fill="#111827")
    img.save(path)
    return path


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


def _catalog_entry_by_filename(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_file: dict[str, dict[str, Any]] = {}
    for entry in models:
        filename = entry.get("comfyui_model")
        if entry.get("category") == "Image Generation" and filename:
            by_file.setdefault(str(filename), entry)
    return by_file


def _entry_for_loaded_model(
    filename: str,
    by_file: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if filename in by_file:
        return by_file[filename]
    for known, entry in by_file.items():
        if filename.endswith(f"/{known}"):
            return entry
    return None


def _settings(entry: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    recommended = dict((entry or {}).get("recommended_settings") or {})
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


def _write_case_rows(out_dir: Path, rows: list[ValidationCase]) -> None:
    payload = [row.__dict__ for row in rows]
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload[0].keys()) if payload else [])
        if payload:
            writer.writeheader()
            writer.writerows(payload)

    passed = sum(1 for row in rows if row.status == "passed")
    failed = sum(1 for row in rows if row.status == "failed")
    blocked = sum(1 for row in rows if row.status == "blocked")
    skipped = sum(1 for row in rows if row.status == "skipped")
    lines = [
        "# LocalAI image-generation validation",
        "",
        f"- Generated: {_timestamp()}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Blocked: {blocked}",
        f"- Skipped: {skipped}",
        "",
        "| Status | Case | Model | Seconds | Output / Error |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        detail = row.output_path or row.error.replace("\n", " ")[:240]
        lines.append(
            f"| {row.status} | {row.case} | {row.model_name} | {row.duration_s:.1f} | {detail} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_png(out_dir: Path, model_filename: str, case_name: str, image_bytes: bytes) -> Path:
    path = out_dir / f"{_slug(model_filename)}__{_slug(case_name)}.png"
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


def _stop_owned_comfyui(app: App, started_by_tool: bool) -> None:
    if not started_by_tool:
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


def _generate_case(
    app: App,
    out_dir: Path,
    entry: dict[str, Any] | None,
    filename: str,
    case_name: str,
    prompt: str,
    negative: str,
    settings: dict[str, Any],
    reference_image: Path | None = None,
    denoise: float = 0.75,
    prompt_kind: str = "safe",
) -> ValidationCase:
    model_id = str((entry or {}).get("id") or filename)
    model_name = str((entry or {}).get("name") or filename)
    progress: list[str] = []

    def _progress(message: str) -> None:
        progress.append(message)
        if len(progress) > 8:
            del progress[:-8]
        print(f"[{model_name} / {case_name}] {message}", flush=True)

    start = time.time()
    try:
        image_bytes = app.comfyui.generate_image(
            model_filename=filename,
            positive_prompt=prompt,
            negative_prompt=negative,
            width=settings["width"],
            height=settings["height"],
            steps=settings["steps"],
            cfg_scale=settings["cfg"],
            seed=settings["seed"],
            sampler_name=settings["sampler"],
            scheduler=settings["scheduler"],
            reference_image_path=str(reference_image) if reference_image else None,
            denoise=denoise,
            progress_cb=_progress,
            stop_event=threading.Event(),
        )
        elapsed = time.time() - start
        png = _save_png(out_dir, filename, case_name, image_bytes)
        actual_width, actual_height = _image_dimensions(png)
        if actual_width <= 0 or actual_height <= 0:
            raise RuntimeError(f"Generated file is not a readable image: {png}")
        return ValidationCase(
            model_id=model_id,
            model_name=model_name,
            model_filename=filename,
            case=case_name,
            status="passed",
            duration_s=round(elapsed, 2),
            output_path=str(png),
            width=actual_width,
            height=actual_height,
            steps=settings["steps"],
            cfg=settings["cfg"],
            sampler=settings["sampler"],
            scheduler=settings["scheduler"],
            seed=settings["seed"],
            prompt_kind=prompt_kind,
            reference_image=str(reference_image or ""),
            progress_tail=progress[-8:],
        )
    except Exception as exc:
        return ValidationCase(
            model_id=model_id,
            model_name=model_name,
            model_filename=filename,
            case=case_name,
            status="failed",
            duration_s=round(time.time() - start, 2),
            width=settings["width"],
            height=settings["height"],
            steps=settings["steps"],
            cfg=settings["cfg"],
            sampler=settings["sampler"],
            scheduler=settings["scheduler"],
            seed=settings["seed"],
            prompt_kind=prompt_kind,
            reference_image=str(reference_image or ""),
            error=str(exc),
            progress_tail=progress[-8:],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        action="append",
        default=[
            Path(p)
            for p in (os.environ.get("LOCALAI_IMAGE_GEN_REFERENCE_DIRS") or "").split(os.pathsep)
            if p
        ] or [
            Path.home() / "Pictures" / "LocalAI-reference",
            Path.home() / "Pictures" / "LocalAI-reference" / "LoRA",
        ],
    )
    parser.add_argument("--model", action="append", help="Limit to a ComfyUI filename or catalog id. Repeatable.")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--sampler")
    parser.add_argument("--scheduler")
    parser.add_argument("--max-dimension", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--safety-fixture-mode", choices=("once", "all", "skip"), default="once")
    parser.add_argument("--skip-safety-fixture", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_image = args.reference_image or _find_reference_image(args.reference_dir)
    if reference_image is None:
        reference_image = _create_synthetic_reference_image(out_dir)
    print(f"Reference image: {reference_image}", flush=True)
    models = catalog.load_catalog()
    by_file = _catalog_entry_by_filename(models)
    rows: list[ValidationCase] = []
    started_comfyui = False
    direct_safety_fixture_ran = False

    app = App()
    app.withdraw()
    try:
        app.update()
        app._switch_page("image_gen")
        app.update()
        started_comfyui = _wait_for_comfyui(app, args.startup_timeout)
        app.comfyui.clear_queue()

        loaded = sorted(app.comfyui.get_model_list(), key=str.lower)
        if args.model:
            wanted = {m.lower() for m in args.model}
            loaded = [
                filename
                for filename in loaded
                if filename.lower() in wanted
                or Path(filename).name.lower() in wanted
                or str((_entry_for_loaded_model(filename, by_file) or {}).get("id", "")).lower() in wanted
            ]
        if not loaded:
            raise RuntimeError("No loaded ComfyUI image models were found.")

        (out_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "loaded_models": loaded,
                    "reference_image": str(reference_image or ""),
                    "safe_txt2img_prompt": SAFE_TXT2IMG_PROMPT,
                    "safe_img2img_prompt": SAFE_IMG2IMG_PROMPT,
                    "safe_blocked_fixture_prompt": SAFE_BLOCKED_FIXTURE_PROMPT,
                    "explicit_nsfw_generation": False,
                    "note": (
                        "The safety fixture intentionally uses a blocked token in a non-explicit poster prompt. "
                        "It validates UI blocking and direct ComfyUI bypass behavior without generating explicit content."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for index, filename in enumerate(loaded, start=1):
            entry = _entry_for_loaded_model(filename, by_file)
            model_name = str((entry or {}).get("name") or filename)
            settings = _settings(entry, args)
            negative = app._default_negative_prompt_for_entry(entry) if entry else ""
            print(f"\n[{index}/{len(loaded)}] Validating {model_name} ({filename})", flush=True)

            for case in [
                ("txt2img", SAFE_TXT2IMG_PROMPT, None, "safe"),
            ]:
                rows.append(
                    _generate_case(
                        app,
                        out_dir,
                        entry,
                        filename,
                        case[0],
                        case[1],
                        negative,
                        settings,
                        reference_image=case[2],
                        prompt_kind=case[3],
                    )
                )
                _write_case_rows(out_dir, rows)
                if rows[-1].status == "failed" and args.fail_fast:
                    return 2

            if not args.skip_reference and entry and entry.get("supports_img2img"):
                if reference_image:
                    img2img = entry.get("img2img_workflows") or {}
                    denoise = float(img2img.get("denoise_default") or 0.75)
                    rows.append(
                        _generate_case(
                            app,
                            out_dir,
                            entry,
                            filename,
                            "img2img_reference",
                            SAFE_IMG2IMG_PROMPT,
                            negative,
                            settings,
                            reference_image=reference_image,
                            denoise=denoise,
                            prompt_kind="safe_reference",
                        )
                    )
                else:
                    rows.append(
                        ValidationCase(
                            model_id=str(entry.get("id") or filename),
                            model_name=model_name,
                            model_filename=filename,
                            case="img2img_reference",
                            status="skipped",
                            error="No reference image found.",
                        )
                    )
                _write_case_rows(out_dir, rows)
                if rows[-1].status == "failed" and args.fail_fast:
                    return 2

            safety_mode = "skip" if args.skip_safety_fixture else args.safety_fixture_mode
            if safety_mode != "skip":
                blocked_term = content_filter.check_prompt(SAFE_BLOCKED_FIXTURE_PROMPT) or ""
                rows.append(
                    ValidationCase(
                        model_id=str((entry or {}).get("id") or filename),
                        model_name=model_name,
                        model_filename=filename,
                        case="regular_startup_content_filter",
                        status="blocked" if blocked_term else "failed",
                        prompt_kind="blocked_fixture_non_explicit",
                        blocked_term=blocked_term,
                        error="" if blocked_term else "Blocked fixture prompt was not blocked by content_filter.",
                    )
                )
                _write_case_rows(out_dir, rows)
                if not blocked_term and args.fail_fast:
                    return 2
                should_run_direct_fixture = safety_mode == "all" or (
                    safety_mode == "once" and not direct_safety_fixture_ran
                )
                if should_run_direct_fixture:
                    rows.append(
                        _generate_case(
                            app,
                            out_dir,
                            entry,
                            filename,
                            "direct_comfyui_blocked_fixture",
                            SAFE_BLOCKED_FIXTURE_PROMPT,
                            negative,
                            settings,
                            reference_image=None,
                            prompt_kind="blocked_fixture_non_explicit_direct",
                        )
                    )
                    direct_safety_fixture_ran = True
                    rows[-1].blocked_term = blocked_term
                    _write_case_rows(out_dir, rows)
                    if rows[-1].status == "failed" and args.fail_fast:
                        return 2

            try:
                app.comfyui.free_vram()
            except Exception:
                pass

        failures = [row for row in rows if row.status == "failed"]
        print(f"\nValidation output: {out_dir}", flush=True)
        print(f"Passed={sum(r.status == 'passed' for r in rows)} Blocked={sum(r.status == 'blocked' for r in rows)} "
              f"Skipped={sum(r.status == 'skipped' for r in rows)} Failed={len(failures)}", flush=True)
        return 1 if failures else 0
    finally:
        _stop_owned_comfyui(app, started_comfyui)
        try:
            app.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

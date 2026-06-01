#!/usr/bin/env python3
# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
LocalAI Studio — Batch Benchmark Mode
Headless CLI that downloads, runs, and benchmarks every catalog model
across every supported inference method (GPU, CPU, NPU/DirectML).
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.catalog import load_catalog
from src.batch_runner import BatchRunner, DEFAULT_PROMPT, IMAGE_METHOD
from src.batch_report import BatchReport, find_latest_report_json
from src.comfyui_benchmark_host import HeadlessComfyUIBenchmarkHost


def list_models() -> None:
    """Print every catalog model and exit."""
    print(f"\n{'ID':<24} {'Name':<28} {'Size':>6}  {'ONNX':<6}  Category")
    print("-" * 90)
    models = load_catalog()
    for m in sorted(models, key=lambda x: x.get("size_gb", 0)):
        onnx = "yes" if m.get("onnx_repo") else "-"
        print(
            f"{m['id']:<24} {m['name']:<28} {m['size_gb']:>5.1f}G  {onnx:<6}  {m['category']}"
        )
    print(f"\nTotal: {len(models)} models")


def _needs_comfyui_host(run_mode: str, skip_image: bool, combos=None) -> bool:
    if skip_image:
        return False
    if combos is not None:
        return any(combo[1] == IMAGE_METHOD for combo in combos)
    return run_mode == "extended"


def _comfyui_runner_kwargs(run_mode: str, skip_image: bool, combos=None) -> dict:
    if not _needs_comfyui_host(run_mode, skip_image, combos):
        return {}
    host = HeadlessComfyUIBenchmarkHost()
    print("Image benchmarks: headless ComfyUI startup enabled.")
    return {
        "comfyui_client": host.client,
        "ensure_comfyui_ready": host.ensure_comfyui_ready,
        "prepare_image_model": host.prepare_image_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LocalAI Studio — Batch Benchmark Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_batch.py --list-models\n"
            "  python run_batch.py --models qwen2.5:0.5b --skip-onnx --timeout 120\n"
            "  python run_batch.py --cleanup --output results/\n"
        ),
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Test prompt (default: %(default)r)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-run timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete models after testing to save disk space",
    )
    parser.add_argument(
        "--cleanup-downloaded-only", action="store_true",
        help="With --cleanup, delete only models downloaded by this run",
    )
    parser.add_argument(
        "--low-resources", action="store_true",
        help="Enable Low Resources Mode disk/RAM checks during the run",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="ID",
        help="Only test specific model IDs (space-separated)",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip Ollama GPU tests",
    )
    parser.add_argument(
        "--skip-cpu", action="store_true",
        help="Skip Ollama CPU tests",
    )
    parser.add_argument(
        "--skip-onnx", action="store_true",
        help="Skip all ONNX tests (DirectML + CPU)",
    )
    parser.add_argument(
        "--skip-utility", "--skip-phase1", dest="skip_utility", action="store_true",
        default=True,
        help=(
            "Skip utility demo adapters (OCR/speech/embeddings/document AI). "
            "Default is on; pass --include-utility to opt in."
        ),
    )
    parser.add_argument(
        "--include-utility", dest="include_utility", action="store_true",
        help="Include utility / toolbox adapters in benchmark runs (opt-in).",
    )
    parser.add_argument(
        "--max-failures", type=int, default=10,
        help="Stop after N consecutive failures (default: 10)",
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Report output directory (default: current directory)",
    )
    parser.add_argument(
        "--capacity-ram-gb", type=float,
        help="Override benchmark capacity RAM in GB, e.g. a target SKU's installed RAM",
    )
    parser.add_argument(
        "--capacity-vram-gb", type=float,
        help="Override benchmark capacity VRAM in GB for a target GPU SKU",
    )
    parser.add_argument(
        "--capacity-no-gpu", action="store_true",
        help="Treat the benchmark capacity profile as CPU-only even on a GPU machine",
    )
    parser.add_argument(
        "--allow-oversize", "--capacity-allow-oversize", dest="allow_oversize", action="store_true",
        help="Allow selected models that exceed the RAM/VRAM capacity profile to run anyway",
    )
    parser.add_argument(
        "--force-all", dest="force_all", action="store_true",
        help=(
            "Force-All mode (best-effort baseline): bypass the profile capacity gate so GPU "
            "methods are attempted even on CPU-only profiles when real hardware is present, "
            "lift the consecutive-failure ceiling so a single dead backend can't bail the run, "
            "and enable adaptive smart-skip so larger same-backend models are skipped after an "
            "OOM or disk-pressure failure. Implies --allow-oversize."
        ),
    )
    parser.add_argument(
        "--run-mode", choices=["quick", "extended"], default="quick",
        help=(
            "quick (default): run one shared prompt per selected text/utility model. "
            "extended: iterate per-model sample prompts and add image generation on "
            "GPU-capable profiles."
        ),
    )
    parser.add_argument(
        "--skip-image", action="store_true",
        help="Skip image-generation benchmarks in extended mode",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all catalog models and exit",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry only the failed tests from the previous run, then merge results",
    )
    parser.add_argument(
        "--retry-file", type=str,
        help="Retry failures from a specific benchmark JSON report instead of the newest report",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    output_dir = Path(args.output)
    capacity_has_gpu = (
        False if args.capacity_no_gpu
        else (args.capacity_vram_gb > 0 if args.capacity_vram_gb is not None else None)
    )

    if args.retry_failed:
        json_path = Path(args.retry_file) if args.retry_file else find_latest_report_json(output_dir)
        if json_path is None or not json_path.exists():
            print(f"Error: No previous benchmark results found in {output_dir}")
            sys.exit(1)
        output_dir = json_path.parent

        prev_report = BatchReport.load_json(json_path)
        failed = prev_report.get_failed_combos()
        if not failed:
            print("All tests in the previous run passed. Nothing to retry.")
            sys.exit(0)

        print(f"Retrying {len(failed)} failed test(s):")
        for combo in failed:
            mid, method = combo[0], combo[1]
            sample = f" sample {int(combo[2]) + 1}" if len(combo) >= 3 else ""
            print(f"  {mid} / {method}{sample}")
        print()

        run_mode = prev_report.run_mode or args.run_mode
        image_kwargs = _comfyui_runner_kwargs(run_mode, args.skip_image, failed)
        runner = BatchRunner(
            prompt=args.prompt,
            timeout=args.timeout,
            cleanup=args.cleanup,
            cleanup_downloaded_only=args.cleanup_downloaded_only,
            low_resources_mode=args.low_resources,
            max_failures=args.max_failures,
            output_dir=output_dir,
            specific_combos=failed,
            capacity_ram_gb=args.capacity_ram_gb,
            capacity_vram_gb=args.capacity_vram_gb,
            capacity_has_gpu=capacity_has_gpu,
            allow_oversize=args.allow_oversize,
            force_all=args.force_all,
            report_file_stem=prev_report.file_stem,
            run_mode=run_mode,
            skip_image=args.skip_image,
            **image_kwargs,
        )
        try:
            runner.run()
        except KeyboardInterrupt:
            print("\n!! Interrupted — merging partial retry results...")
        finally:
            # Always merge whatever was collected (even partial) with the
            # original report so previous successes are never lost.
            prev_report.merge(runner.report)
            json_out = prev_report.save_json(output_dir)
            html_out = prev_report.save_html(output_dir)

            print("\n--- MERGED RESULTS ---")
            prev_report.print_summary()
            print(f"Merged reports saved to:")
            print(f"  JSON: {json_out}")
            print(f"  HTML: {html_out}")
    else:
        image_kwargs = _comfyui_runner_kwargs(args.run_mode, args.skip_image)
        runner = BatchRunner(
            prompt=args.prompt,
            timeout=args.timeout,
            cleanup=args.cleanup,
            cleanup_downloaded_only=args.cleanup_downloaded_only,
            model_ids=args.models,
            skip_gpu=args.skip_gpu,
            skip_cpu=args.skip_cpu,
            skip_onnx=args.skip_onnx,
            skip_phase1=(args.skip_utility and not args.include_utility),
            max_failures=args.max_failures,
            output_dir=output_dir,
            low_resources_mode=args.low_resources,
            capacity_ram_gb=args.capacity_ram_gb,
            capacity_vram_gb=args.capacity_vram_gb,
            capacity_has_gpu=capacity_has_gpu,
            allow_oversize=args.allow_oversize,
            force_all=args.force_all,
            run_mode=args.run_mode,
            skip_image=args.skip_image,
            **image_kwargs,
        )
        runner.run()


if __name__ == "__main__":
    main()

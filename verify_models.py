"""
LocalAI Studio - Model Integrity Verifier & Dependency Repair
Checks every downloaded model against its expected file size.
Truncated or corrupted files are deleted so the user can re-download them.
Also checks and repairs runtime dependencies (gguf package, custom nodes).
"""
import json
import subprocess
import sys
from pathlib import Path

# ANSI colour codes (work in Windows Terminal / pwsh; fall back gracefully)
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RESET  = '\033[0m'

# ── Model list ────────────────────────────────────────────────────────────────
# Exact byte counts from HuggingFace Content-Length headers (March 2025).
# Tolerance: file must be within ±5 % of expected size to be considered valid.
# Anything smaller is a truncated/partial download; anything far larger would
# indicate an unexpected file was placed at that path.
MODELS = [
    # (filename,  subdir,  expected_bytes,  display_name)
    # ── Main checkpoints ──────────────────────────────────────────────────────
    ('v1-5-pruned-emaonly.safetensors',
     'checkpoints', 4_265_146_304, 'Stable Diffusion 1.5'),
    ('sd_xl_base_1.0.safetensors',
     'checkpoints', 6_938_078_334, 'Stable Diffusion XL'),
    ('Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',
     'checkpoints', 7_105_348_188, 'Juggernaut XL v9'),
    ('Realistic_Vision_V6.0_NV_B1_fp16.safetensors',
     'checkpoints', 2_132_625_894, 'Realistic Vision v6'),
    ('flux1-dev.safetensors',
     'checkpoints', 23_800_000_000, 'FLUX.1-dev FP16'),
    # ── GGUF / UNet models (diffusion_models/) ────────────────────────────────
    ('flux1-schnell-Q4_K_S.gguf',
     'diffusion_models', 6_783_943_712, 'FLUX.1-schnell Q4'),
    ('flux1-dev-Q4_K_S.gguf',
     'diffusion_models', 6_805_988_640, 'FLUX.1-dev Q4'),
    ('z_image_bf16.safetensors',
     'diffusion_models', 12_309_866_400, 'Z-Image'),
    ('z_image_turbo_bf16.safetensors',
     'diffusion_models', 12_309_866_400, 'Z-Image Turbo'),
    # ── Support files ─────────────────────────────────────────────────────────
    ('qwen_3_4b_fp8_mixed.safetensors',
     'text_encoders', 5_631_994_051, 'Qwen text encoder (Z-Image support)'),
    ('ae.safetensors',
     'vae', 335_304_388, 'AE VAE (Flux / Z-Image support)'),
    ('t5xxl_fp8_e4m3fn.safetensors',
     'clip', 4_893_934_904, 'T5-XXL encoder (Flux support)'),
    ('clip_l.safetensors',
     'clip', 246_144_152, 'CLIP-L encoder (Flux support)'),
]

TOLERANCE = 0.05  # ±5 %

def find_comfyui(script_dir: Path) -> Path | None:
    """Return the ComfyUI install path or None."""
    cfg_file = script_dir / 'config.json'
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
            d = cfg.get('comfyui_dir', '')
            if d and (Path(d) / 'main.py').exists():
                return Path(d)
        except Exception:
            pass
    bat_file = script_dir / 'comfyui_path.bat'
    if bat_file.exists():
        for line in bat_file.read_text(encoding='utf-8').splitlines():
            if 'LOCALAI_COMFYUI=' in line.upper():
                p = line.split('=', 1)[1].strip()
                if p and (Path(p) / 'main.py').exists():
                    return Path(p)
    return None


def check_models(comfyui: Path) -> tuple[int, int, int, int]:
    """Check all model files. Returns (checked, ok, deleted, skipped)."""
    print(f'{CYAN}Model file integrity{RESET}')
    print('------------------------------------------------------------')

    checked = ok = deleted = skipped = 0

    for filename, subdir, expected_bytes, display_name in MODELS:
        path = comfyui / 'models' / subdir / filename
        if not path.exists():
            skipped += 1
            continue

        checked += 1
        actual = path.stat().st_size
        min_ok = int(expected_bytes * (1 - TOLERANCE))
        max_ok = int(expected_bytes * (1 + TOLERANCE))
        actual_gb = round(actual        / 1_073_741_824, 2)
        expect_gb = round(expected_bytes / 1_073_741_824, 2)

        if min_ok <= actual <= max_ok:
            print(f'{GREEN}  [OK]      {display_name}  ({actual_gb} GB){RESET}')
            ok += 1
        else:
            status = 'TRUNCATED' if actual < min_ok else 'WRONG SIZE'
            print(
                f'{RED}  [DELETED] {display_name} -- {status} '
                f'(got {actual_gb} GB, expected ~{expect_gb} GB){RESET}'
            )
            try:
                path.unlink()
                deleted += 1
            except Exception as e:
                print(f'{YELLOW}            Could not delete: {e}{RESET}')

    print()
    if checked == 0:
        print(f'{YELLOW}  No model files found -- nothing to verify.{RESET}')
    else:
        print(
            f'  Checked: {checked}  |  OK: {ok}  |  '
            f'Deleted: {deleted}  |  Not yet downloaded: {skipped}'
        )
    if deleted > 0:
        print()
        print(f'{YELLOW}  Deleted files can be re-downloaded from the Models tab.{RESET}')

    return checked, ok, deleted, skipped


def _gguf_node_present(comfyui: Path) -> bool:
    return (comfyui / 'custom_nodes' / 'ComfyUI-GGUF').is_dir()


def _gguf_models_present(comfyui: Path) -> bool:
    dm = comfyui / 'models' / 'diffusion_models'
    return any(dm.glob('*.gguf')) if dm.exists() else False


def repair_dependencies(comfyui: Path) -> tuple[int, int]:
    """Check and repair runtime dependencies. Returns (fixed, failed)."""
    print(f'{CYAN}Runtime dependency check & repair{RESET}')
    print('------------------------------------------------------------')

    fixed = failed = 0

    # ── 1. gguf Python package ────────────────────────────────────────────────
    if _gguf_node_present(comfyui) or _gguf_models_present(comfyui):
        import importlib.util
        if importlib.util.find_spec('gguf') is not None:
            print(f'{GREEN}  [OK]   gguf Python package{RESET}')
        else:
            print(f'{YELLOW}  [FIX]  gguf Python package -- not installed, installing...{RESET}')
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', 'gguf>=0.13.0'],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    print(f'{GREEN}         gguf installed successfully.{RESET}')
                    print(f'{YELLOW}         Restart LocalAI to reload the ComfyUI-GGUF custom node.{RESET}')
                    fixed += 1
                else:
                    err = (result.stderr or result.stdout).strip().splitlines()[-1]
                    print(f'{RED}         FAILED: {err}{RESET}')
                    failed += 1
            except Exception as e:
                print(f'{RED}         FAILED: {e}{RESET}')
                failed += 1
    else:
        print(f'  [--]   gguf Python package -- GGUF node/models not present, skipping')

    print()
    if fixed == 0 and failed == 0:
        print('  All dependencies OK.')
    else:
        summary = []
        if fixed:  summary.append(f'{fixed} fixed')
        if failed: summary.append(f'{RED}{failed} failed{RESET}')
        print(f'  Repair summary: {", ".join(summary)}')

    return fixed, failed


def main():
    script_dir = Path(__file__).parent.resolve()
    comfyui = find_comfyui(script_dir)
    if not comfyui:
        print(f'{RED}[ERROR] ComfyUI not found. Run setup.bat first.{RESET}')
        sys.exit(1)

    print(f'  ComfyUI: {comfyui}')
    print()

    _checked, _ok, deleted, _skipped = check_models(comfyui)

    print()
    fixed, failed = repair_dependencies(comfyui)
    print()

    if deleted > 0 or fixed > 0:
        print(f'{YELLOW}  Action required: restart LocalAI after this check completes.{RESET}')
        print()

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

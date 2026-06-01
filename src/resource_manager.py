# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Low Resources Mode — disk space and RAM management.

When enabled, checks free disk/RAM before downloads and model runs.
In batch mode, can delete previously downloaded models to free space,
following a strict priority order with user consent.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src import logger as _log
from src.ollama_client import ollama_tag_is_local

# 20% headroom on top of model size for safe downloads
_HEADROOM = 1.2
# Minimum free disk to maintain (GB)
_MIN_FREE_DISK_GB = 2.0


def _disk_usage_probe_path(path: str | Path) -> Path:
    """Return an existing path on the same volume as ``path`` for disk_usage."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def get_free_disk_gb(path: str | Path = ".") -> float:
    """Return free disk space in GB for the volume containing *path*."""
    try:
        usage = shutil.disk_usage(str(_disk_usage_probe_path(path)))
        return usage.free / 1_073_741_824
    except Exception as exc:
        _log.warning(f"Low Resources: could not inspect free disk for {path}: {exc}")
        return 0.0


def get_available_ram_gb() -> float:
    """Return available (not total) RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1_073_741_824
    except ImportError:
        return 0.0


def get_total_ram_gb() -> float:
    """Return total RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1_073_741_824
    except ImportError:
        return 0.0


def is_unified_memory() -> bool:
    """Return True if this machine uses unified memory (Apple Silicon)."""
    if sys.platform != "darwin":
        return False
    try:
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        return r.returncode == 0 and "Apple" in r.stdout
    except Exception:
        return False


# ── Pre-download check ────────────────────────────────────────────────────────

def check_disk_for_download(
    model_size_gb: float,
    models_path: str | Path = ".",
) -> tuple[bool, str]:
    """Check if there's enough disk space to download a model.

    Returns (ok, reason). If ok is False, reason explains why.
    """
    free = get_free_disk_gb(models_path)
    needed = model_size_gb * _HEADROOM

    if free >= needed:
        _log.debug(f"Low Resources: {free:.1f}GB free, {needed:.1f}GB needed — OK")
        return True, "OK"

    reason = (
        f"Insufficient disk space: need {needed:.1f} GB "
        f"(model {model_size_gb:.1f} GB + headroom), "
        f"only {free:.1f} GB free."
    )
    _log.warning(f"Low Resources: {reason}")
    return False, reason


# ── Pre-run RAM/VRAM check ────────────────────────────────────────────────────

def check_ram_for_model(model: dict) -> tuple[bool, str]:
    """Check if there's enough RAM to run a model.

    Returns (ok, reason). Accounts for unified memory on Apple Silicon.
    """
    min_ram = model.get("min_ram_gb", 0)
    min_vram = model.get("min_vram_gb", 0)
    avail = get_available_ram_gb()

    # On unified memory, VRAM comes from the same pool as RAM
    if is_unified_memory():
        effective_need = max(min_ram, min_vram)
        if effective_need > 0 and avail < effective_need:
            reason = (
                f"Insufficient memory: need {effective_need:.1f} GB, "
                f"only {avail:.1f} GB available (unified memory — shared CPU/GPU)."
            )
            _log.warning(f"Low Resources: skipping {model.get('name', '?')} — {reason}")
            return False, reason
        return True, "OK"

    # Discrete GPU systems — check RAM for CPU runs
    if min_ram > 0 and avail < min_ram:
        reason = (
            f"Insufficient RAM: need {min_ram:.1f} GB, "
            f"only {avail:.1f} GB available."
        )
        _log.warning(f"Low Resources: skipping {model.get('name', '?')} — {reason}")
        return False, reason

    return True, "OK"


# ── Batch pre-assessment ──────────────────────────────────────────────────────

def assess_batch_space(
    models: list[dict],
    ollama_client,
    models_path: str | Path = ".",
    cleanup_after_each_model: bool = False,
) -> dict:
    """Assess whether a batch run has enough disk space.

    Returns dict with:
        ok: bool — True if enough space without deletions
        possible: bool — True if enough space after deletions
        free_gb: float — current free space
        needed_gb: float — total download size for models not yet local
        required_gb: float — disk space needed at one time
        deletable_gb: float — total size of models that could be deleted
        shortfall_gb: float — how much more space is needed (0 if ok)
    """
    free_gb = get_free_disk_gb(models_path)

    # Calculate how much needs to be downloaded
    protected_ollama_tags = {m.get("ollama_tag", "") for m in models if m.get("ollama_tag")}
    missing_sizes_gb = []
    local_names = set()
    try:
        local_names = ollama_client.local_model_names()
    except Exception:
        pass

    for m in models:
        tag = m.get("ollama_tag", "")
        is_local = _is_ollama_tag_local(tag, local_names)
        if tag and not is_local:
            missing_sizes_gb.append(float(m.get("size_gb", 0) or 0))

    needed_gb = sum(missing_sizes_gb)
    required_gb = max(missing_sizes_gb, default=0.0) if cleanup_after_each_model else needed_gb
    needed_with_headroom = required_gb * _HEADROOM

    if free_gb >= needed_with_headroom:
        return {
            "ok": True,
            "possible": True,
            "free_gb": free_gb,
            "needed_gb": needed_gb,
            "required_gb": required_gb,
            "required_with_headroom_gb": needed_with_headroom,
            "cleanup_after_each_model": cleanup_after_each_model,
            "deletable_gb": 0,
            "shortfall_gb": 0,
        }

    # Calculate how much we could free by deleting non-batch models
    deletable_gb = _estimate_deletable_gb(ollama_client, protected_ollama_tags, models_path)

    shortfall = needed_with_headroom - free_gb
    possible = (free_gb + deletable_gb) >= needed_with_headroom

    return {
        "ok": False,
        "possible": possible,
        "free_gb": free_gb,
        "needed_gb": needed_gb,
        "required_gb": required_gb,
        "required_with_headroom_gb": needed_with_headroom,
        "cleanup_after_each_model": cleanup_after_each_model,
        "deletable_gb": deletable_gb,
        "shortfall_gb": max(0, shortfall),
    }


def _estimate_deletable_gb(
    ollama_client,
    protected_ollama_tags: set,
    models_path: str | Path,
) -> float:
    """Estimate how much disk space could be freed by deleting models."""
    total = 0.0

    # Ollama models not in batch
    try:
        for m in ollama_client.list_local_models():
            name = m.get("name", "")
            # Skip models in the batch
            if _is_protected_ollama_name(name, protected_ollama_tags):
                continue
            size_bytes = m.get("size", 0)
            total += size_bytes / 1_073_741_824
    except Exception:
        pass

    # ComfyUI checkpoints
    comfyui_dirs = [
        Path(models_path) / "ComfyUI" / "models" / "checkpoints",
        Path(models_path) / "ComfyUI" / "models" / "diffusion_models",
    ]
    for d in comfyui_dirs:
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    total += f.stat().st_size / 1_073_741_824

    return total


# ── Model deletion for space ──────────────────────────────────────────────────

def free_space_for_download(
    needed_gb: float,
    ollama_client,
    models_path: str | Path,
    protected_ollama_tags: set,
    purge_done: bool = False,
    comfyui_client=None,
) -> tuple[bool, bool]:
    """Delete models in priority order until needed_gb is free.

    Returns (success, purge_was_done).
    """
    if get_free_disk_gb(models_path) >= needed_gb * _HEADROOM:
        return True, purge_done

    # Priority 1: Purge caches / empty trash
    if not purge_done:
        _purge_system_caches()
        purge_done = True
        if get_free_disk_gb(models_path) >= needed_gb * _HEADROOM:
            return True, purge_done

    # Priority 2: Vision models (largest first)
    _delete_ollama_by_category(
        ollama_client, protected_ollama_tags, models_path,
        needed_gb, filter_fn=_is_vision_model,
    )
    if get_free_disk_gb(models_path) >= needed_gb * _HEADROOM:
        return True, purge_done

    # Priority 3: Image generation models (largest first)
    _delete_comfyui_models(models_path, needed_gb, comfyui_client)
    if get_free_disk_gb(models_path) >= needed_gb * _HEADROOM:
        return True, purge_done

    # Priority 4: Other Ollama models not in batch (largest first)
    _delete_ollama_by_category(
        ollama_client, protected_ollama_tags, models_path,
        needed_gb, filter_fn=None,
    )
    if get_free_disk_gb(models_path) >= needed_gb * _HEADROOM:
        return True, purge_done

    _log.error(
        f"Low Resources: insufficient disk space. "
        f"Need {needed_gb * _HEADROOM:.1f}GB, "
        f"only {get_free_disk_gb(models_path):.1f}GB free after cleanup."
    )
    return False, purge_done


def _is_vision_model(model_info: dict) -> bool:
    """Check if an Ollama model is a vision model."""
    name = model_info.get("name", "").lower()
    details = model_info.get("details", {})
    families = details.get("families", []) if isinstance(details, dict) else []
    families_lower = [f.lower() for f in families]
    return (
        "vision" in name
        or "llava" in name
        or "clip" in families_lower
    )


def _delete_ollama_by_category(
    ollama_client,
    protected_ollama_tags: set,
    models_path: str | Path,
    needed_gb: float,
    filter_fn=None,
):
    """Delete Ollama models matching filter_fn, largest first, until enough space."""
    try:
        local_models = ollama_client.list_local_models()
    except Exception:
        return

    # Sort largest first
    candidates = sorted(local_models, key=lambda m: m.get("size", 0), reverse=True)

    for m in candidates:
        if get_free_disk_gb(models_path) >= needed_gb * _HEADROOM:
            return

        name = m.get("name", "")

        # Never delete batch models
        if _is_protected_ollama_name(name, protected_ollama_tags):
            continue

        # Apply category filter if provided
        if filter_fn and not filter_fn(m):
            continue

        size_gb = m.get("size", 0) / 1_073_741_824
        try:
            # Unload first if loaded
            ollama_client.unload_model(name)
            ollama_client.delete_model(name)
            _log.info(f"Low Resources: deleted {name} ({size_gb:.1f}GB) to free disk space")
        except Exception as e:
            _log.warning(f"Low Resources: could not delete {name}: {e}")


def _is_ollama_tag_local(tag: str, local_names: set[str]) -> bool:
    # Low-resource cleanup deliberately over-protects base-name matches so a
    # downloaded sibling tag is not deleted when the requested tag is ambiguous.
    return ollama_tag_is_local(tag, local_names)


def _is_protected_ollama_name(name: str, protected_ollama_tags: set[str]) -> bool:
    if not name:
        return False
    for tag in protected_ollama_tags:
        if _is_ollama_tag_local(tag, {name}):
            return True
    return False


def _delete_comfyui_models(
    models_path: str | Path,
    needed_gb: float,
    comfyui_client=None,
):
    """Delete ComfyUI checkpoint files, largest first, until enough space."""
    # Free VRAM first if ComfyUI is running
    if comfyui_client:
        try:
            comfyui_client.free_vram()
        except Exception:
            pass

    candidates = []
    for subdir in ("checkpoints", "diffusion_models"):
        d = Path(models_path) / "ComfyUI" / "models" / subdir
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and f.suffix in (".safetensors", ".gguf", ".ckpt", ".pt"):
                    candidates.append(f)

    # Largest first
    candidates.sort(key=lambda f: f.stat().st_size, reverse=True)

    for f in candidates:
        if get_free_disk_gb(models_path) >= needed_gb * _HEADROOM:
            return

        size_gb = f.stat().st_size / 1_073_741_824
        try:
            f.unlink()
            _log.info(f"Low Resources: deleted {f.name} ({size_gb:.1f}GB) to free disk space")
        except Exception as e:
            _log.warning(f"Low Resources: could not delete {f.name}: {e}")


# ── System cache purge ────────────────────────────────────────────────────────

def _purge_system_caches():
    """Best-effort purge of system caches and trash."""
    if sys.platform == "darwin":
        # macOS: purge disk cache (no data loss, no password needed)
        try:
            subprocess.run(["purge"], capture_output=True, timeout=10)
            _log.info("Low Resources: ran macOS purge")
        except Exception:
            pass
        # macOS: empty Trash
        try:
            subprocess.run(
                ["osascript", "-e", 'tell app "Finder" to empty trash'],
                capture_output=True, timeout=15,
            )
            _log.info("Low Resources: emptied macOS Trash")
        except Exception:
            pass
    else:
        # Windows: clear only app-owned temp files. Do not purge all of %TEMP%;
        # shared/redirected temp folders can contain unrelated user or system work.
        temp = os.environ.get("TEMP", "")
        if temp and Path(temp).is_dir():
            try:
                deleted = 0
                for item in Path(temp).iterdir():
                    if not item.name.lower().startswith(("localai", "comfyui")):
                        continue
                    try:
                        if item.is_file():
                            item.unlink()
                            deleted += 1
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                            deleted += 1
                    except Exception:
                        pass
                _log.info(f"Low Resources: cleared {deleted} LocalAI/ComfyUI temp item(s)")
            except Exception:
                pass
        if os.environ.get("LOCALAI_ALLOW_SYSTEM_CLEANUP") == "1":
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Clear-RecycleBin -Force -ErrorAction SilentlyContinue"],
                    capture_output=True, timeout=15,
                )
                _log.info("Low Resources: emptied Windows Recycle Bin")
            except Exception:
                pass


# ── Startup detection ─────────────────────────────────────────────────────────

def should_suggest_low_resources(models_path: str | Path = ".") -> tuple[bool, str]:
    """Check if this machine has low resources and should suggest enabling the mode.

    Returns (suggest, reason).
    """
    reasons = []

    free_disk = get_free_disk_gb(models_path)
    if free_disk < 20:
        reasons.append(f"Low disk space: {free_disk:.0f} GB free")

    avail_ram = get_available_ram_gb()
    total_ram = get_total_ram_gb()
    if 0 < total_ram < 12:
        reasons.append(f"Limited RAM: {total_ram:.0f} GB total")
    elif total_ram > 0 and 0 < avail_ram < 4:
        reasons.append(f"Low available RAM: {avail_ram:.1f} GB free of {total_ram:.0f} GB")

    if reasons:
        return True, ". ".join(reasons) + "."
    return False, ""

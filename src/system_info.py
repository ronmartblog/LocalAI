# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Hardware detection: RAM, GPU, NPU, storage.
Works on Windows 11 and macOS without requiring admin rights.
"""

import subprocess
import json
import shutil
import os
import platform
import re
import sys
from pathlib import Path
from typing import Optional

from src import logger as _log
from src.persistence import atomic_write_json

_IS_MAC = sys.platform == "darwin"


# Hardware SKU catalog. Populated at module load from
# ``skus.json`` (see ``_init_benchmark_sku_profiles`` at the bottom of
# this file). The app must NEVER hardcode SKU display names — adding,
# renaming, or swapping a SKU is a JSON-only change.
#
# When the JSON file is missing or unreadable, this list stays empty and
# ``get_benchmark_sku_profiles()`` falls back to a single synthetic
# "This Device" entry derived from the local hardware. UI surfaces that
# need to enumerate SKUs (model_guide chip strip, benchmark profile
# selector, doc-sample validator) call ``get_benchmark_sku_profiles()``
# rather than touching this list directly.
BENCHMARK_SKU_PROFILES: list[dict] = []


def benchmark_sku_profile_name(cpu: int | float = 0, ram_gb: int | float = 0, vram_gb: int | float = 0) -> str:
    """Return the canonical Benchmark profile name for a known SKU capacity."""
    try:
        cpu_int = int(round(float(cpu or 0)))
        ram_int = int(round(float(ram_gb or 0)))
        vram = float(vram_gb or 0)
    except (TypeError, ValueError):
        return ""
    for profile in BENCHMARK_SKU_PROFILES:
        profile_vram = float(profile.get("vram_gb", 0) or 0)
        if cpu_int == int(profile.get("cpu", 0) or 0) and ram_int == int(profile.get("ram_gb", 0) or 0):
            if abs(profile_vram - vram) <= 1.0:
                return str(profile["name"])
    return ""


def get_recommended_models_for_sku(sku_name: str) -> list[str]:
    """Return the list of model ids surfaced as 'Recommended for: <sku_name>'.

    Returns an empty list when *sku_name* is unknown or when the SKU has no
    ``recommended_models`` entry. The canonical source is ``skus.json`` —
    when that file is missing the function returns an empty list (the public
    code path).
    """
    target = str(sku_name or "").strip().lower()
    if not target:
        return []
    for sku in BENCHMARK_SKU_PROFILES:
        if str(sku.get("name", "")).strip().lower() == target:
            recs = sku.get("recommended_models") or []
            return [str(item) for item in recs]
    return []


def get_recommended_skus_for_model(model_id: str, model: dict | None = None) -> list[str]:
    """Return SKU display names that recommend *model_id* — in capability order.

    Combines two sources:

    * **Primary** — per-SKU ``recommended_models`` arrays in ``skus.json``
      (the canonical, gitignored source).
    * **Fallback / supplement** — the legacy per-model ``recommended_for``
      list, when *model* is provided. Honored for HuggingFace imports and
      user-curated catalog entries that pin a model to a SKU directly.

    De-duplicated case-insensitively while preserving the order in which a
    name first appears (SKU ascending order first, then the model's own
    override list).
    """
    mid = str(model_id or "").strip()
    out: list[str] = []
    seen: set[str] = set()
    if mid:
        for sku in BENCHMARK_SKU_PROFILES:
            recs = sku.get("recommended_models") or []
            if mid in recs:
                name = str(sku.get("name") or "").strip()
                key = name.lower()
                if name and key not in seen:
                    out.append(name)
                    seen.add(key)
    if isinstance(model, dict):
        for sku_name in (model.get("recommended_for") or []):
            name = str(sku_name or "").strip()
            key = name.lower()
            if name and key not in seen:
                out.append(name)
                seen.add(key)
    return out


def _vm_size_matches(pattern: object, vm_size: str) -> bool:
    """Return whether an optional SKU vm_size_pattern matches a cloud VM size string."""
    if isinstance(pattern, (list, tuple, set)):
        return any(_vm_size_matches(item, vm_size) for item in pattern)
    pattern_text = str(pattern or "").strip()
    vm_text = str(vm_size or "").strip()
    if not pattern_text or not vm_text:
        return False
    if pattern_text.casefold() == vm_text.casefold():
        return True

    regex_parts = []
    for char in pattern_text:
        if char == "#":
            regex_parts.append(r"\d")
        elif char == "?":
            regex_parts.append(".")
        elif char == "*":
            regex_parts.append(".*")
        else:
            regex_parts.append(re.escape(char))
    return re.fullmatch("".join(regex_parts), vm_text, flags=re.IGNORECASE) is not None


def _run(cmd: list[str], timeout: int = 8) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ── RAM ───────────────────────────────────────────────────────────────────────

def get_ram_info() -> dict:
    """Return total and available RAM in MB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_mb": vm.total // 1_048_576,
            "available_mb": vm.available // 1_048_576,
            "used_mb": vm.used // 1_048_576,
            "percent": vm.percent,
        }
    except ImportError:
        pass

    if _IS_MAC:
        # Fallback: sysctl
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                total_bytes = int(out)
                total_mb = total_bytes // 1_048_576
                # macOS doesn't expose "available" easily without psutil
                return {
                    "total_mb": total_mb,
                    "available_mb": total_mb,  # best-effort
                    "used_mb": 0,
                    "percent": 0.0,
                }
            except ValueError:
                pass
    else:
        # Fallback: wmic (Windows)
        out = _run(["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/VALUE"])
        info = {}
        for line in out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                info[k.strip()] = v.strip()
        total_kb = int(info.get("TotalVisibleMemorySize", 0))
        free_kb = int(info.get("FreePhysicalMemory", 0))
        total_mb = total_kb // 1024
        avail_mb = free_kb // 1024
        return {
            "total_mb": total_mb,
            "available_mb": avail_mb,
            "used_mb": total_mb - avail_mb,
            "percent": round((total_mb - avail_mb) / max(total_mb, 1) * 100, 1),
        }

    return {"total_mb": 0, "available_mb": 0, "used_mb": 0, "percent": 0.0}


def get_cpu_percent() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.5)
    except ImportError:
        return 0.0


# ── GPU ───────────────────────────────────────────────────────────────────────

def _get_gpu_info_mac() -> list[dict]:
    """Detect GPUs on macOS via system_profiler."""
    gpus = []
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=10)
    if not out:
        return gpus
    try:
        data = json.loads(out)
        displays = data.get("SPDisplaysDataType", [])
        for disp in displays:
            name = disp.get("sppci_model", "Unknown GPU")
            vendor = "Apple" if "apple" in name.lower() else "AMD" if "amd" in name.lower() else "Unknown"
            # Apple Silicon uses unified memory — report system RAM as shared GPU memory
            vram_str = disp.get("sppci_vram", "") or disp.get("spdisplays_vram", "")
            vram_mb = 0
            if vram_str:
                # Parse strings like "8192 MB" or "16 GB"
                parts = vram_str.strip().split()
                if len(parts) >= 2:
                    try:
                        val = int(parts[0])
                        if "GB" in parts[1].upper():
                            vram_mb = val * 1024
                        else:
                            vram_mb = val
                    except ValueError:
                        pass
            # For Apple Silicon (unified memory), use total system RAM
            if vram_mb == 0 and vendor == "Apple":
                ram = get_ram_info()
                vram_mb = ram.get("total_mb", 0)
            gpus.append({
                "name": name,
                "vendor": vendor,
                "vram_total_mb": vram_mb,
                "vram_free_mb": vram_mb,  # no live usage tracking on macOS
                "vram_used_mb": 0,
                "type": "GPU",
                "unified_memory": vendor == "Apple",
            })
    except (json.JSONDecodeError, KeyError):
        pass
    return gpus


def _get_gpu_info_windows() -> list[dict]:
    """Detect GPUs on Windows via nvidia-smi, rocm-smi, or WMI."""
    gpus = []

    # NVIDIA via nvidia-smi
    nvidia_out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ])
    for line in nvidia_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append({
                    "name": parts[0],
                    "vendor": "NVIDIA",
                    "vram_total_mb": int(parts[1]),
                    "vram_free_mb": int(parts[2]),
                    "vram_used_mb": int(parts[3]),
                    "type": "GPU",
                })
            except ValueError:
                pass

    # AMD via rocm-smi (if ROCM installed)
    if not gpus:
        rocm_out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
        if rocm_out:
            try:
                data = json.loads(rocm_out)
                for card_id, card in data.items():
                    total = int(card.get("VRAM Total Memory (B)", 0)) // 1_048_576
                    used = int(card.get("VRAM Total Used Memory (B)", 0)) // 1_048_576
                    gpus.append({
                        "name": card.get("GPU ID", f"AMD GPU {card_id}"),
                        "vendor": "AMD",
                        "vram_total_mb": total,
                        "vram_free_mb": total - used,
                        "vram_used_mb": used,
                        "type": "GPU",
                    })
            except (json.JSONDecodeError, KeyError):
                pass

    # Intel Arc / integrated via WMI
    # v5.5.11 (Ron, 2026-05-26): when no NVIDIA / ROCm GPU is present the
    # WMI fallback below is almost always an integrated GPU (Intel Iris/Arc
    # Graphics, AMD Radeon Graphics, Snapdragon Adreno). Windows reports a
    # small dedicated VRAM carve-out (typically 1-2 GB) via AdapterRAM, but
    # the actual runtime memory ceiling for these iGPUs is system RAM
    # through DXGI's "Shared GPU Memory" pool — same architectural pattern
    # as Apple Silicon unified memory. We mark these as `unified_memory: True`
    # and report vram_total/vram_free in terms of system RAM so
    # can_run_model() uses the unified-memory branch (lines ~766-778) instead
    # of failing with a misleading "Not enough GPU VRAM: only 1.7 GB free"
    # on machines that actually have 13+ GB of shared memory available.
    # The original WMI AdapterRAM is preserved in `dedicated_vram_mb` for
    # display purposes.
    #
    # Carve-outs (NIT-1 from v5.5.11 SQT setup-release review):
    #   1. Discrete Intel Arc cards (A380/A580/A750/A770, B570/B580/B770)
    #      are NOT detected by nvidia-smi or rocm-smi and would fall into
    #      this WMI block, where they'd be incorrectly flagged as unified.
    #      Detect them by combining "Arc" in the name with a separate
    #      A###/B### model-number match (the WMI name typically renders as
    #      "Intel(R) Arc(TM) A770 Graphics" so the (TM) sits BETWEEN "Arc"
    #      and the model number — a naive "Arc\s*A###" regex misses it).
    #      Integrated Intel Arc Graphics in Lunar Lake / Arrow Lake / Meteor
    #      Lake reports as "Intel(R) Arc(TM) Graphics" with no model number,
    #      so the A###/B### gate cleanly distinguishes integrated from
    #      discrete.
    #   2. AdapterRAM is a uint32 field so cards with >= 4 GB VRAM cap at
    #      ~4 GB-1; the size threshold alone is unreliable, but combined
    #      with the discrete-name match it's robust enough to keep the
    #      AI PC iGPU case working without misclassifying real discrete
    #      Intel hardware.
    _ARC_DISCRETE_MODEL = re.compile(r"\b[AB]\d{3}\b", re.IGNORECASE)
    if not gpus:
        ps_cmd = (
            "Get-WmiObject Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json"
        )
        ps_out = _run(["powershell", "-NoProfile", "-Command", ps_cmd], timeout=10)
        if ps_out:
            try:
                data = json.loads(ps_out)
                if isinstance(data, dict):
                    data = [data]
                ram_info = get_ram_info()
                total_ram_mb = int(ram_info.get("total_mb") or 0)
                avail_ram_mb = int(ram_info.get("available_mb") or 0)
                for card in data:
                    name = card.get("Name", "Unknown GPU")
                    dedicated_mb = int(card.get("AdapterRAM") or 0) // 1_048_576
                    # Skip Microsoft Basic Display or unknown
                    if "microsoft" in name.lower() or dedicated_mb == 0:
                        continue
                    name_lower = name.lower()
                    if "intel" in name_lower:
                        vendor = "Intel"
                    elif "amd" in name_lower or "radeon" in name_lower:
                        vendor = "AMD"
                    elif "qualcomm" in name_lower or "snapdragon" in name_lower or "adreno" in name_lower:
                        vendor = "Qualcomm"
                    else:
                        vendor = "Unknown"
                    # Discrete Intel Arc carve-out — see comment above.
                    # Combine "arc" in name with a separate A###/B### model
                    # number match (the (TM) trademark sits between them in
                    # the actual WMI string "Intel(R) Arc(TM) A770 Graphics",
                    # so a single contiguous regex misses it).
                    is_intel_discrete_arc = (
                        "arc" in name_lower
                        and bool(_ARC_DISCRETE_MODEL.search(name))
                    )
                    is_integrated = not is_intel_discrete_arc
                    if is_integrated:
                        gpus.append({
                            "name": name,
                            "vendor": vendor,
                            # v5.5.11: report system RAM as the iGPU memory
                            # pool; see comment above. dedicated_vram_mb
                            # keeps the WMI AdapterRAM value so the system
                            # page can show "X GB dedicated carve-out"
                            # alongside the shared ceiling.
                            "vram_total_mb": total_ram_mb or dedicated_mb,
                            "vram_free_mb": avail_ram_mb or dedicated_mb,
                            "vram_used_mb": max(0, (total_ram_mb or dedicated_mb) - (avail_ram_mb or dedicated_mb)),
                            "dedicated_vram_mb": dedicated_mb,
                            "type": "GPU",
                            "unified_memory": True,
                        })
                    else:
                        # Discrete Intel Arc — real GDDR6 VRAM. Note that
                        # AdapterRAM is uint32-capped near 4 GB, so the
                        # reported value will undercount on 8/16 GB cards;
                        # but it's a real dedicated pool, not shared with
                        # system RAM. Leave unified_memory unset so
                        # can_run_model uses the dedicated VRAM gate.
                        gpus.append({
                            "name": name,
                            "vendor": vendor,
                            "vram_total_mb": dedicated_mb,
                            "vram_free_mb": dedicated_mb,
                            "vram_used_mb": 0,
                            "type": "GPU",
                        })
            except (json.JSONDecodeError, ValueError):
                pass

    return gpus


def get_gpu_info() -> list[dict]:
    """Detect discrete GPUs. Returns list of dicts."""
    if _IS_MAC:
        return _get_gpu_info_mac()
    return _get_gpu_info_windows()


# ── NPU ───────────────────────────────────────────────────────────────────────

def _get_npu_info_mac() -> list[dict]:
    """Detect Apple Neural Engine on macOS."""
    # All Apple Silicon Macs (M1+) have a Neural Engine.
    # Detect by checking for Apple Silicon chip.
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out and "Apple" in out:
        return [{
            "name": "Apple Neural Engine",
            "status": "OK",
            "type": "NPU",
            "class": "NeuralEngine",
            "note": "Accessed via CoreML",
        }]

    # Also try system_profiler for chip info
    hw_out = _run(["system_profiler", "SPHardwareDataType", "-json"], timeout=10)
    if hw_out:
        try:
            data = json.loads(hw_out)
            hw = data.get("SPHardwareDataType", [{}])[0]
            chip = hw.get("chip_type", "")
            if chip and "Apple" in chip:
                return [{
                    "name": f"Apple Neural Engine ({chip})",
                    "status": "OK",
                    "type": "NPU",
                    "class": "NeuralEngine",
                    "note": "Accessed via CoreML",
                }]
        except (json.JSONDecodeError, IndexError):
            pass

    return []


def _get_npu_info_windows() -> list[dict]:
    """Detect NPU / AI accelerator devices on Windows 11."""
    npus = []

    # Exclude USB hubs, input devices, generic ports
    EXCLUDE_KEYWORDS = ("USB", "HID", "Input Device", "Hub", "Root", "Controller")
    # v5.5.4: also exclude by Windows device CLASS. Virtualized hosts surface
    # "Microsoft Hyper-V Input" with class HIDClass that slips through the
    # name-based filter ("HID" is not a substring of "microsoft hyper-v
    # input") and was being counted as an NPU in benchmark env sidecars on
    # CPU-only virtualized hosts. A true NPU never reports these classes.
    EXCLUDE_CLASSES = {
        "HIDClass", "USB", "Mouse", "Keyboard", "Net", "AudioEndpoint",
        "Bluetooth", "Image", "Camera", "Monitor", "Printer", "MEDIA",
        "DiskDrive", "Volume", "System", "Processor",
    }

    ps_cmd = (
        "Get-PnpDevice -Status OK | "
        "Where-Object { "
        "  $_.Class -eq 'ComputeAccelerator' -or "
        "  $_.FriendlyName -like '*NPU*' -or "
        "  $_.FriendlyName -like '*Neural*' -or "
        "  $_.FriendlyName -like '*Hexagon*' -or "
        "  $_.FriendlyName -like '*AI Boost*' -or "
        "  $_.FriendlyName -like '*VPU*' "
        "} | Select-Object FriendlyName,Status,Class | ConvertTo-Json"
    )
    ps_out = _run(["powershell", "-NoProfile", "-Command", ps_cmd], timeout=15)
    if ps_out:
        try:
            data = json.loads(ps_out)
            if isinstance(data, dict):
                data = [data]
            for dev in data:
                name = dev.get("FriendlyName", "Unknown NPU")
                dev_class = dev.get("Class", "")
                if any(ex.lower() in name.lower() for ex in EXCLUDE_KEYWORDS):
                    continue
                if dev_class in EXCLUDE_CLASSES:
                    continue
                npus.append({
                    "name": name,
                    "status": dev.get("Status", "Unknown"),
                    "type": "NPU",
                    "class": dev_class,
                })
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: check for Intel NPU via device description
    if not npus:
        ps_cmd2 = (
            "Get-PnpDevice | Where-Object { "
            "  ($_.FriendlyName -like '*Intel*NPU*' -or "
            "   $_.FriendlyName -like '*Qualcomm*NPU*' -or "
            "   $_.FriendlyName -like '*AMD*NPU*') "
            "} | Select-Object FriendlyName,Status,Class | ConvertTo-Json"
        )
        ps_out2 = _run(["powershell", "-NoProfile", "-Command", ps_cmd2], timeout=15)
        if ps_out2:
            try:
                data2 = json.loads(ps_out2)
                if isinstance(data2, dict):
                    data2 = [data2]
                for dev in data2:
                    name = dev.get("FriendlyName", "Unknown NPU")
                    dev_class = dev.get("Class", "")
                    if any(ex.lower() in name.lower() for ex in EXCLUDE_KEYWORDS):
                        continue
                    if dev_class in EXCLUDE_CLASSES:
                        continue
                    npus.append({
                        "name": name,
                        "status": dev.get("Status", "Unknown"),
                        "type": "NPU",
                        "class": dev_class,
                    })
            except (json.JSONDecodeError, TypeError):
                pass

    return npus


def get_npu_info() -> list[dict]:
    """Detect NPU / AI accelerator devices."""
    if _IS_MAC:
        return _get_npu_info_mac()
    return _get_npu_info_windows()


def has_directml_support() -> bool:
    """Check whether DirectML (DirectX 12) is available for ONNX acceleration."""
    if _IS_MAC:
        return False
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def has_coreml_support() -> bool:
    """Check whether CoreML is available for ONNX acceleration (macOS)."""
    if not _IS_MAC:
        return False
    try:
        import onnxruntime as ort
        return "CoreMLExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


# ── Storage ──────────────────────────────────────────────────────────────────

def get_storage_info(path: str | Path = ".") -> dict:
    """Return free/total disk space for the drive containing *path*."""
    try:
        usage = shutil.disk_usage(str(path))
        return {
            "total_gb": usage.total / 1_073_741_824,
            "used_gb": usage.used / 1_073_741_824,
            "free_gb": usage.free / 1_073_741_824,
            "free_percent": usage.free / usage.total * 100,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "free_percent": 0}


# ── Optional SKU profiles ────────────────────────────────────────────────────
# Source of truth: skus.json (v2 schema with bench_defaults.baselines +
# per-SKU bench_quick_models/bench_extended_models {inherit, add, remove}).
# Loaded at import on every platform via _init_benchmark_sku_profiles() →
# BENCHMARK_SKU_PROFILES. When the file is missing or invalid,
# get_benchmark_sku_profiles() falls back to a single synthetic "This Device"
# entry built from local hardware (NOT a hardcoded list of SKU names).
# Per v5.5.12: the App must NEVER hardcode SKU display names — read every
# SKU + its default-tick model sets from this module.

# Path to the optional private SKU config file
OPTIONAL_SKUS_FILE = Path(__file__).parent.parent / "skus.json"

_OPTIONAL_SKUS_README = (
    "LocalAI Studio optional SKU profile definitions. "
    "Per-SKU fields: name (unique display name used in filters), vm_size_pattern "
    "(optional cloud VM size string or pattern: # = one digit, ? = one character, "
    "* = any characters; e.g. Standard_D16s_v# matches Standard_D16s_v5/v6 — "
    "accepts a list of patterns too), cpu (CPU/vCPU count), ram_gb, vram_gb "
    "(0 for CPU-only), gpu_fraction, the per-SKU Benchmark default-tick sets "
    "bench_quick_models / bench_extended_models, and recommended_models (model "
    "ids surfaced as 'Recommended for: <SKU>' badges on the Models page and "
    "used to auto-tick the Recommended preset in Benchmark and Image Gen). "
    "Each bench field accepts EITHER a flat list of model ids OR a resolver "
    "object {inherit: <baseline name OR list>, add: [...], remove: [...]} that "
    "resolves to (union of named baselines) ∪ add − remove. Top-level "
    "bench_defaults.baselines is a name→list-of-model-ids mapping shared across "
    "SKUs (e.g. quick_chat_ultra_small, quick_image_smallest, extended_chat_high, "
    "gpu_image_base). SKUs must be "
    "listed in ascending capability order. Optional feature labels may be "
    "supplied in a top-level 'feature' object. This file is the SINGLE source "
    "of truth for SKU display names AND SKU→model recommendations — the app "
    "must never hardcode them. Legacy field names azure_size and vcpu are "
    "still accepted by the loader for back-compat."
)

# Required fields for each SKU entry
_REQUIRED_SKU_FIELDS = {"name", "vram_gb"}

OPTIONAL_SKUS: list[dict] = []


def build_local_sku() -> dict:
    """
    Build a synthetic SKU dict from this machine's actual hardware.

    Called when optional SKU detection finds no match in the loaded SKU list so
    that the current machine still appears as a selectable filter entry.
    """
    if _IS_MAC:
        return _build_local_sku_mac()
    return _build_local_sku_windows()


def _build_local_sku_mac() -> dict:
    """Build a local SKU from macOS hardware info.

    The display name is always "This Device" — we intentionally do NOT
    expose the machine model / vendor in the SKU picker because that
    would leak host vendor branding into a UI that's supposed to be
    vendor-neutral. Hardware capacity (cpu / ram / vram) is still
    detected so capability-gated logic continues to work.
    """
    vcpu = 0
    ram_gb = 0

    hw_out = _run(["system_profiler", "SPHardwareDataType", "-json"], timeout=10)
    if hw_out:
        try:
            data = json.loads(hw_out)
            hw = data.get("SPHardwareDataType", [{}])[0]
            # number_processors can be "proc 10:4:6" (total:perf:eff) or plain int
            cpu_raw = hw.get("number_processors", "0") or "0"
            cpu_str = str(cpu_raw).strip()
            if cpu_str.startswith("proc "):
                # "proc 10:4:6" → take the first number (total cores)
                cpu_str = cpu_str.split()[1].split(":")[0]
            try:
                vcpu = int(cpu_str)
            except ValueError:
                vcpu = 0
            ram_str = hw.get("physical_memory", "")
            if ram_str:
                parts = ram_str.split()
                if parts:
                    try:
                        ram_gb = int(parts[0])
                        if len(parts) > 1 and "TB" in parts[1].upper():
                            ram_gb *= 1024
                    except ValueError:
                        pass
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    # Apple Silicon uses unified memory — GPU shares system RAM
    gpus = get_gpu_info()
    vram_gb = 0
    gpu_fraction = "None"
    if gpus:
        best = max(gpus, key=lambda g: g.get("vram_total_mb", 0))
        vram_mb = best.get("vram_total_mb", 0)
        if vram_mb > 0:
            vram_gb = max(1, round(vram_mb / 1024))
            is_unified = best.get("unified_memory", False)
            gpu_fraction = "Unified Memory" if is_unified else "Dedicated"

    return {
        "name":             "This Device",
        "vm_size_pattern":  "",
        "cpu":              vcpu,
        "ram_gb":           ram_gb,
        "vram_gb":          vram_gb,
        "gpu_fraction":     gpu_fraction,
    }


def _build_local_sku_windows() -> dict:
    """Build a local SKU from Windows hardware info.

    The display name is always "This Device" — we intentionally do NOT
    expose ``Win32_ComputerSystem.Manufacturer + Model`` (which on
    Hyper-V VMs concatenates the hypervisor vendor and the model
    string) in the SKU picker because that would leak host vendor
    branding into a UI that's supposed to be vendor-neutral. Hardware
    capacity (cpu / ram / vram) is still detected so capability-gated
    logic continues to work.
    """
    raw = _run([
        "powershell", "-NoProfile", "-Command",
        "$cs=Get-CimInstance Win32_ComputerSystem;"
        "$cs.NumberOfLogicalProcessors.ToString()+'|'"
        "+[int][Math]::Round($cs.TotalPhysicalMemory/1GB).ToString()",
    ])
    parts = raw.split("|")
    vcpu   = int(parts[0].strip()) if len(parts) > 0 and parts[0].strip().isdigit() else 0
    ram_gb = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0

    gpus = get_gpu_info()
    vram_gb      = 0
    gpu_fraction = "None"
    if gpus:
        best = max(gpus, key=lambda g: g.get("vram_total_mb", 0))
        vram_mb = best.get("vram_total_mb", 0)
        if vram_mb > 0:
            vram_gb = max(1, round(vram_mb / 1024))
            gpu_fraction = "Dedicated"

    return {
        "name":             "This Device",
        "vm_size_pattern":  "",
        "cpu":              vcpu,
        "ram_gb":           ram_gb,
        "vram_gb":          vram_gb,
        "gpu_fraction":     gpu_fraction,
    }


def detect_optional_sku() -> dict | None:
    """
    Detect whether this machine matches an optional SKU profile.
    Returns None on macOS or if no match is found on Windows.
    """
    if _IS_MAC:
        return None

    import urllib.request

    # --- Primary: cloud IMDS (works on any cloud VM, no GPU required) ---
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            headers={"Metadata": "true"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        vm_size = data.get("compute", {}).get("vmSize", "")
        if vm_size:
            for sku in OPTIONAL_SKUS:
                if _vm_size_matches(sku.get("vm_size_pattern", ""), vm_size):
                    return sku

            _log.info(f"IMDS vmSize '{vm_size}' not in SKU table — building dynamic SKU.")

            # We intentionally do NOT name the dynamic SKU after the
            # cloud offer / vm_size / tier — those strings can leak
            # vendor branding into a picker that's meant to be neutral.
            # Fall back to "This Device" when no canonical SKU matches;
            # the vm_size_pattern + capacity fields are still recorded
            # so internal matching logic continues to work.
            raw = _run([
                "powershell", "-NoProfile", "-Command",
                "$cs=Get-CimInstance Win32_ComputerSystem;"
                "$cs.NumberOfLogicalProcessors.ToString()+'|'"
                "+[int][Math]::Round($cs.TotalPhysicalMemory/1GB).ToString()",
            ])
            parts = raw.split("|")
            try:
                vcpu   = int(parts[0].strip()) if len(parts) > 0 else 0
                ram_gb = int(parts[1].strip()) if len(parts) > 1 else 0
            except ValueError:
                vcpu = ram_gb = 0

            canonical = benchmark_sku_profile_name(vcpu, ram_gb, 0)
            name = canonical or "This Device"
            return {
                "name":             name,
                "vm_size_pattern":  vm_size,
                "cpu":              vcpu,
                "ram_gb":           ram_gb,
                "vram_gb":          0,
                "gpu_fraction":     "None",
            }
    except Exception:
        pass

    # --- Fallback: NVIDIA A10 GPU detection ---
    gpus = get_gpu_info()
    a10 = next((g for g in gpus if "A10" in g.get("name", "")), None)
    if not a10:
        return None
    vram_gb = a10["vram_total_mb"] / 1024
    for sku in OPTIONAL_SKUS:
        if abs(vram_gb - sku.get("vram_gb", -1)) <= 1.0:
            return sku
    return None


# ── Optional SKU file I/O ─────────────────────────────────────────────────────

def optional_skus_enabled(path: Path | None = None) -> bool:
    """Return True when the optional private SKU file exists."""
    return (path or OPTIONAL_SKUS_FILE).exists()


def resolve_bench_models(spec: object, baselines: dict | None) -> set[str]:
    """Resolve a ``bench_quick_models`` / ``bench_extended_models`` spec.

    ``spec`` is one of:
      * ``None`` or missing — returns the empty set.
      * a flat ``list`` of model ids — returns ``set(spec)``.
      * a dict ``{"inherit": <baseline name OR list of names>, "add": [...],
        "remove": [...]}`` — resolves to ``(union of named baselines)
        ∪ add − remove``.

    ``baselines`` is the ``bench_defaults.baselines`` mapping. An unknown
    inherit key logs a warning and contributes nothing; missing/None
    baselines treat all inherits as unknown.
    """
    if spec is None:
        return set()
    if isinstance(spec, list):
        return {str(x) for x in spec}
    if not isinstance(spec, dict):
        _log.warning(
            f"bench_*_models: expected list or object, got {type(spec).__name__}; ignoring."
        )
        return set()

    result: set[str] = set()
    inherit = spec.get("inherit")
    if isinstance(inherit, str):
        keys: list[str] = [inherit]
    elif isinstance(inherit, list):
        keys = [str(k) for k in inherit]
    elif inherit in (None, ""):
        keys = []
    else:
        _log.warning(
            f"bench_*_models.inherit: expected str or list, got {type(inherit).__name__}; ignoring."
        )
        keys = []

    table = baselines if isinstance(baselines, dict) else {}
    for key in keys:
        baseline = table.get(key)
        if baseline is None:
            _log.warning(f"bench_*_models: unknown baseline {key!r}; ignoring.")
            continue
        if not isinstance(baseline, (list, tuple, set)):
            _log.warning(
                f"bench_*_models: baseline {key!r} must be a list, got {type(baseline).__name__}; ignoring."
            )
            continue
        result |= {str(x) for x in baseline}

    add_list = spec.get("add") or []
    if isinstance(add_list, (list, tuple, set)):
        for item in add_list:
            result.add(str(item))

    remove_list = spec.get("remove") or []
    if isinstance(remove_list, (list, tuple, set)):
        for item in remove_list:
            result.discard(str(item))

    return result


def load_optional_sku_config(path: Path | None = None) -> dict:
    """Load the optional SKU config. Missing or invalid files disable it."""
    target = path or OPTIONAL_SKUS_FILE
    if not target.exists():
        return {"feature": {}, "skus": [], "bench_defaults": {}}
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        _log.error(f"Optional SKUs: could not read {target} ({exc}) — feature disabled.")
        return {"feature": {}, "skus": [], "bench_defaults": {}}

    if not isinstance(data, dict) or "skus" not in data:
        _log.error(f"Optional SKUs: {target} missing 'skus' key — feature disabled.")
        return {"feature": {}, "skus": [], "bench_defaults": {}}

    raw = data["skus"]
    if not isinstance(raw, list):
        _log.error(f"Optional SKUs: 'skus' is not a list in {target} — feature disabled.")
        return {"feature": {}, "skus": [], "bench_defaults": {}}

    bench_defaults_raw = data.get("bench_defaults", {})
    if not isinstance(bench_defaults_raw, dict):
        _log.warning(
            f"Optional SKUs: bench_defaults in {target} is not an object — ignoring."
        )
        bench_defaults_raw = {}
    baselines = bench_defaults_raw.get("baselines", {})
    if not isinstance(baselines, dict):
        _log.warning(
            f"Optional SKUs: bench_defaults.baselines in {target} is not an object — ignoring."
        )
        baselines = {}

    valid = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        missing = _REQUIRED_SKU_FIELDS - entry.keys()
        if missing:
            name = entry.get("name", "?")
            fields = ", ".join(sorted(missing))
            _log.warning(f"Optional SKUs: skipping {name!r} — missing fields: {fields}")
            continue
        # Back-compat: migrate legacy field names (v2 schema → current). The
        # canonical keys are `vm_size_pattern` and `cpu`; older private
        # skus.json files may still use `azure_size` and `vcpu`. Rename in
        # place so downstream readers see only the canonical keys.
        if "vm_size_pattern" not in entry and "azure_size" in entry:
            entry["vm_size_pattern"] = entry.pop("azure_size")
        if "cpu" not in entry and "vcpu" in entry:
            entry["cpu"] = entry.pop("vcpu")
        try:
            entry["vram_gb"] = float(entry["vram_gb"])
        except (ValueError, TypeError):
            name = entry.get("name", "?")
            _log.warning(f"Optional SKUs: skipping {name!r} — vram_gb is not numeric.")
            continue
        # Resolve per-SKU bench model sets in place so downstream consumers
        # just read ``entry["bench_quick_models"]`` / ``["bench_extended_models"]``
        # as ``set[str]`` of model ids. Missing fields resolve to an empty set
        # (UI shows fits-but-unticked for that SKU).
        entry["bench_quick_models"] = resolve_bench_models(
            entry.get("bench_quick_models"), baselines
        )
        entry["bench_extended_models"] = resolve_bench_models(
            entry.get("bench_extended_models"), baselines
        )
        # Normalize per-SKU recommended_models into a stable list[str] (empty
        # when absent). Reading consumers ask
        # ``get_recommended_models_for_sku`` / ``get_recommended_skus_for_model``
        # rather than touching the field directly.
        rec_raw = entry.get("recommended_models") or []
        if isinstance(rec_raw, (list, tuple, set)):
            entry["recommended_models"] = [str(item) for item in rec_raw if str(item).strip()]
        else:
            entry["recommended_models"] = []
        valid.append(entry)

    if not valid:
        _log.warning(f"Optional SKUs: no valid SKUs in {target} — feature disabled.")
        return {"feature": {}, "skus": [], "bench_defaults": bench_defaults_raw}

    feature = data.get("feature", {})
    if not isinstance(feature, dict):
        feature = {}
    _log.info(f"Optional SKUs: loaded {len(valid)} SKUs from {target}.")
    return {"feature": feature, "skus": valid, "bench_defaults": bench_defaults_raw}


def load_optional_skus(path: Path | None = None) -> list[dict]:
    """Load optional SKU definitions from file. Missing/invalid files return []."""
    return load_optional_sku_config(path).get("skus", [])


def _scrub_sku_for_save(entry: dict) -> dict:
    """Return a JSON-serializable copy of *entry*.

    The loader resolves per-SKU ``bench_quick_models`` / ``bench_extended_models``
    fields in place from the raw ``{inherit, add, remove}`` spec into a
    ``set[str]`` of resolved model ids. ``json.dumps`` cannot serialize ``set``,
    so before writing back to disk we coerce any set to a sorted list. The
    round-trip is lossy in *shape* (the resolver object is replaced with the
    flat-list shorthand) but identical in *semantics* — the next load resolves
    to the same set of model ids.
    """
    out: dict = {}
    for key, value in entry.items():
        if isinstance(value, set):
            out[key] = sorted(value)
        elif isinstance(value, frozenset):
            out[key] = sorted(value)
        else:
            out[key] = value
    return out


def save_optional_skus(skus: list[dict], path: Path | None = None) -> bool:
    """Write *skus* to *path* as JSON (default: OPTIONAL_SKUS_FILE)."""
    target = path or OPTIONAL_SKUS_FILE
    payload = {
        "_readme": _OPTIONAL_SKUS_README,
        "version": 3,
        "skus": [_scrub_sku_for_save(s) for s in skus],
    }
    try:
        atomic_write_json(target, payload, indent=2, ensure_ascii=False)
        _log.info(f"Optional SKUs: saved {len(skus)} SKUs to {target}.")
        return True
    except OSError as exc:
        _log.error(f"Optional SKUs: could not write {target}: {exc}")
        return False


def ensure_optional_skus_file(path: Path | None = None) -> bool:
    """Create an empty optional SKU config file if missing."""
    target = path or OPTIONAL_SKUS_FILE
    if target.exists():
        return False
    save_optional_skus(list(OPTIONAL_SKUS), target)
    _log.info(f"Optional SKUs: created empty SKU file at {target}.")
    return True


# ── Summary ───────────────────────────────────────────────────────────────────

def get_system_summary(models_path: str | Path = ".") -> dict:
    return {
        "ram": get_ram_info(),
        "gpus": get_gpu_info(),
        "npus": get_npu_info(),
        "storage": get_storage_info(models_path),
        "cpu_percent": get_cpu_percent(),
        "platform": platform.processor(),
    }


def can_run_model(model: dict, gpu_index: int | None = None) -> tuple[bool, str]:
    """
    Check whether the local hardware can run *model*.
    Returns (ok, reason_string).

    On Apple Silicon (unified memory), both RAM and VRAM checks use the
    same pool since CPU and GPU share memory.
    """
    ram = get_ram_info()
    gpus = get_gpu_info()
    avail_ram_gb = ram["available_mb"] / 1024

    min_ram = model.get("min_ram_gb", 0)
    min_vram = model.get("min_vram_gb", 0)

    # Apple Silicon unified memory: VRAM check uses system RAM
    if gpus and gpus[0].get("unified_memory"):
        total_ram_gb = ram["total_mb"] / 1024
        if min_vram > 0 and total_ram_gb < min_vram:
            return False, (
                f"Not enough unified memory: need {min_vram:.1f} GB VRAM, "
                f"only {total_ram_gb:.1f} GB total (shared CPU/GPU)."
            )
        if avail_ram_gb < min_ram:
            return False, (
                f"Not enough RAM: need {min_ram} GB, "
                f"only {avail_ram_gb:.1f} GB available."
            )
        return True, "OK"

    if gpu_index is not None and gpus:
        gpu = gpus[gpu_index] if gpu_index < len(gpus) else gpus[0]
        free_vram_gb = gpu["vram_free_mb"] / 1024
        if free_vram_gb < min_vram:
            return False, (
                f"Not enough GPU VRAM: need {min_vram:.1f} GB, "
                f"only {free_vram_gb:.1f} GB free on {gpu['name']}."
            )
    else:
        # CPU / NPU run — need RAM
        if avail_ram_gb < min_ram:
            return False, (
                f"Not enough RAM: need {min_ram} GB, "
                f"only {avail_ram_gb:.1f} GB available."
            )

    return True, "OK"


# ── BENCHMARK_SKU_PROFILES initialization ─────────────────────────────────────
#
# Populated at module-load time from ``skus.json`` so the app code can
# import a stable list reference (``from src.system_info import
# BENCHMARK_SKU_PROFILES``) without ever knowing the SKU display names. When
# the user reloads SKUs at runtime, ``app.py``'s
# ``_apply_optional_skus_to_modules`` rebinds the list contents in place so
# every prior importer keeps seeing the live data.

# Built-in fallback profiles surfaced only when ``skus.json`` is missing
# / unreadable / has no valid entries. Intentionally EMPTY — the public
# repo ships zero hardcoded SKU display names. When no ``skus.json``
# exists the benchmark picker contains exactly one entry: the synthetic
# "This Device" SKU built from local hardware. Users who want a richer
# benchmark catalog drop in their own ``skus.json`` (the schema is
# documented at the top of this module).
_FALLBACK_PUBLIC_SKU_PROFILES: list[dict] = []


def get_benchmark_sku_profiles() -> list[dict]:
    """Return a copy of the currently loaded benchmark SKU profile list.

    When no SKUs are loaded (``skus.json`` missing, unreadable, or
    contained no valid entries) this falls back to a single synthetic
    "This Device" entry built from the local hardware. The public repo
    ships no hardcoded SKU display names — users who want a richer
    benchmark catalog drop in their own ``skus.json``. The app must
    never rely on a specific hardcoded SKU display name being present
    in the populated path.
    """
    if BENCHMARK_SKU_PROFILES:
        return [dict(s) for s in BENCHMARK_SKU_PROFILES]
    fallback: list[dict] = []
    try:
        local = build_local_sku()
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(f"BENCHMARK_SKU_PROFILES: could not build local SKU fallback: {exc}")
        local = None
    if isinstance(local, dict) and local.get("name"):
        fallback.append(dict(local))
    fallback.extend(dict(p) for p in _FALLBACK_PUBLIC_SKU_PROFILES)
    return fallback


def _init_benchmark_sku_profiles() -> None:
    """Load ``skus.json`` once at import and populate the module-level list."""
    try:
        cfg = load_optional_sku_config()
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(f"BENCHMARK_SKU_PROFILES: initial load failed ({exc}) — empty list.")
        return
    skus = cfg.get("skus") or []
    BENCHMARK_SKU_PROFILES.clear()
    BENCHMARK_SKU_PROFILES.extend(skus)


_init_benchmark_sku_profiles()


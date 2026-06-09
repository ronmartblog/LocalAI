# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
GPU detection utility for LocalAI Studio.

Detects available GPU acceleration methods for ComfyUI and provides
appropriate command-line flags.

Windows: NVIDIA CUDA > DirectML > CPU
macOS:   Apple Metal (MPS) > CPU
"""

import os
import platform
import re
import shlex
import subprocess
import sys
from typing import Optional, Literal

from src import logger

GPUType = Literal["cuda", "mps", "directml", "cpu"]


def _subprocess_flags() -> dict:
    """Return platform-appropriate subprocess kwargs."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


# vGPU device-name patterns. Order matters only for documentation; each
# regex is tested independently. The canonical lessons doc is referenced
# from the vGPU section of docs/architecture.md, §1.
#
# - r"-\d+[QC]\b"   matches NVIDIA vGPU profile suffixes for both Q (compute
#                   + display) and C (compute-only) profiles. Also catches
#                   MIG-Capable Graphics GPU DC-N-NNQ profiles via the
#                   trailing -NNQ token (e.g. "DC-1-24Q" ends in "-24Q").
# - r"\bGRID\b"     matches the legacy GRID vGPU device-name prefix.
#                   Real-world examples: "GRID K2", "GRID K520",
#                   "GRID M60-8Q" (the latter is also covered by the first
#                   regex, the GRID one is the fallback for older K-series
#                   that have no profile suffix in the name).
_VGPU_NAME_PATTERNS = (
    re.compile(r"-\d+[QC]\b"),
    re.compile(r"\bGRID\b"),
)


def _name_matches_vgpu_profile(name: str) -> bool:
    """Return True when a GPU device-name string carries a vGPU marker."""
    if not name:
        return False
    return any(p.search(name) for p in _VGPU_NAME_PATTERNS)


def _split_extra_flags(raw: str) -> list[str]:
    """Parse the ``LOCALAI_COMFYUI_EXTRA_FLAGS`` env var into argv tokens.

    Accepts two formats so power users do not have to remember which one
    LocalAI prefers:

    - Semicolon-separated   ("``--disable-cuda-malloc;--reserve-vram 1.5``")
    - Standard shell tokens ("``--disable-cuda-malloc --reserve-vram 1.5``")

    Always returns a flat ``list[str]`` of cleaned argv tokens. Empty
    tokens are dropped. On Windows the tokenizer runs in non-POSIX mode
    so backslash-bearing paths (``--output-directory C:\\models\\out``)
    survive verbatim; on macOS/Linux it runs in POSIX mode so flag
    values containing spaces can be quoted with the standard POSIX
    rules.
    """
    if not raw:
        return []
    posix = sys.platform != "win32"
    if ";" in raw:
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        flat: list[str] = []
        for part in parts:
            try:
                tokens = shlex.split(part, posix=posix)
            except ValueError:
                tokens = [part]
            for tok in tokens:
                tok = tok.strip()
                if tok:
                    flat.append(tok)
        return flat
    try:
        return [t.strip() for t in shlex.split(raw, posix=posix) if t.strip()]
    except ValueError:
        return [t.strip() for t in raw.split() if t.strip()]


def is_snapdragon_arm64() -> bool:
    """True only on Windows ARM64 (Snapdragon X family).

    Image generation via ComfyUI requires ``torch-directml``, whose wheels on
    PyPI are x64-Windows only as of this release. Calling code uses this helper
    to short-circuit DirectML installs (in ``setup.bat`` /
    ``fix_directml_pytorch.bat``) and to render an "unsupported" panel on the
    Image Generation page instead of letting ComfyUI fail with a
    ``torch_library_impl could not be located in _torchaudio.pyd`` popup.

    Apple Silicon (``darwin`` + ``arm64``) is **not** included — those machines
    use the MPS path, which is fully supported.

    Detection is intentionally cheap: ``platform.machine()`` reads cached
    Windows API values; no subprocess calls.
    """
    if sys.platform != "win32":
        return False
    try:
        return platform.machine().upper() == "ARM64"
    except Exception:
        return False


class GPUInfo:
    """Information about available GPU acceleration."""

    def __init__(self, gpu_type: GPUType, device_name: str = "",
                 npu_name: str = "", npu_devices: "Optional[list]" = None):
        self.gpu_type = gpu_type
        self.device_name = device_name
        # NPU is supplementary, not a replacement for ``gpu_type`` (ComfyUI
        # image-gen still routes through DirectML/CUDA). These fields let the
        # Chat dropdown default, Settings panel, and benchmark capacity checks
        # route OpenVINO chat workloads onto an Intel NPU when one is present.
        self.npu_name = npu_name
        self.npu_devices = list(npu_devices) if npu_devices else []

    @property
    def vram_gb(self) -> float:
        """Best-effort total VRAM in GiB for the active CUDA device.

        Returns 0.0 when not on CUDA, when the probe fails, or when no GPU is
        present. Used by :py:meth:`get_comfyui_flags` to decide whether the
        small-vGPU memory tweaks (``--reserve-vram`` + ``--cache-none``) should
        be auto-applied. Tries PyTorch first (cheapest), then ``nvidia-smi``.
        """
        if self.gpu_type != "cuda" or sys.platform == "darwin":
            return 0.0
        try:
            import torch

            if torch.cuda.is_available():
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                return float(total_bytes) / (1024 ** 3)
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
                **_subprocess_flags(),
            )
            if result.returncode == 0 and result.stdout.strip():
                first = result.stdout.strip().splitlines()[0].strip()
                return float(first) / 1024.0
        except Exception:
            pass
        return 0.0

    @property
    def is_virtual_gpu(self) -> bool:
        """Detect NVIDIA virtual GPU or MIG (Multi-Instance GPU).

        These environments have restricted CUDA support compared to bare-metal:
        - vGPU Q-profile (compute + display): A10-24Q, T4-16Q, L40S-48Q
        - vGPU C-profile (compute only):      A10-24C, T4-16C
        - vGPU DC-profile (MIG-Capable Graphics GPU): DC-1-24Q, DC-2-12Q
        - Older GRID vGPU (legacy device-name prefix): "GRID K2", "GRID M60-8Q"
        - MIG: Multi-Instance GPU partitions (multiple NVIDIA generations)

        All of these need ``--disable-cuda-malloc``, ``--disable-dynamic-vram``,
        and ``--disable-async-offload`` for ComfyUI (see the vGPU lessons
        doc referenced from docs/architecture.md — the async offload pipeline
        depends on the same VBAR reservation that fails on vGPU, so missing
        the third flag still surfaces a ``MemoryError: VBAR allocation
        failed`` in a different code path when a model is patched).

        Always False on macOS (no vGPU scenario).
        """
        if sys.platform == "darwin":
            return False

        name = self.device_name

        if _name_matches_vgpu_profile(name):
            return True

        # Check nvidia-smi for MIG mode or vGPU profile
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=mig.mode.current",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
                **_subprocess_flags(),
            )
            if result.returncode == 0 and "Enabled" in result.stdout:
                return True
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
                **_subprocess_flags(),
            )
            if result.returncode == 0:
                smi_name = result.stdout.strip()
                if _name_matches_vgpu_profile(smi_name):
                    return True
        except Exception:
            pass

        return False

    @property
    def is_vgpu(self) -> bool:
        """Legacy alias — now checks both vGPU and MIG."""
        return self.is_virtual_gpu

    def get_comfyui_flags(self) -> list[str]:
        """Return appropriate ComfyUI command-line flags as a list.

        Order:

        1. Backend-selection flag (``--directml`` / ``--cpu`` / nothing for
           CUDA + MPS auto-detect).
        2. The mandatory **three-flag vGPU/MIG triple** when running on a
           virtual GPU partition or MIG slice - see the vGPU lessons
           doc referenced from docs/architecture.md. Missing the third
           (``--disable-async-offload``) reproduces the
           ``MemoryError: VBAR allocation failed`` crash on first model
           patch even when dynamic VRAM is already off.
        3. **Small-vGPU memory tweaks** when the partition has <=12 GB
           VRAM (typical: A10-12Q, T4-16Q sliced into halves). Adds
           ``--reserve-vram 1.5`` (guaranteed VAE working set) and
           ``--cache-none`` (drop UNet/VAE between prompts) to prevent
           the back-to-back-prompt VAE-decode hang documented in the
           v5.5.17 ComfyUI hang runbook. Trade-off: ~3-5s reload between
           prompts.
        4. Any user-injected flags from the
           ``LOCALAI_COMFYUI_EXTRA_FLAGS`` env var (semicolon-separated,
           ``;`` as the delimiter). This is the documented escape hatch
           for early-generation bare-metal driver regressions on recent
           NVIDIA cards (``--disable-cuda-malloc`` workaround - lessons
           section 9) and for further tight-memory vGPU tweaks (``--lowvram``,
           larger ``--reserve-vram``). Duplicates of flags already in
           the list are dropped silently, so user overrides win for
           single-value knobs only when a different value is supplied.
        """
        flags = []
        if self.gpu_type == "directml":
            flags.append("--directml")
        elif self.gpu_type == "cpu":
            flags.append("--cpu")
        # MPS (macOS Metal) - ComfyUI auto-detects, no flag needed
        # CUDA - ComfyUI auto-detects, no flag needed

        # Virtual GPU / MIG workarounds (Windows only). All three flags
        # are required; see the vGPU lessons doc referenced from
        # docs/architecture.md.
        if self.is_virtual_gpu:
            flags.append("--disable-dynamic-vram")
            flags.append("--disable-cuda-malloc")
            flags.append("--disable-async-offload")
            logger.info(
                f"Virtual GPU/MIG detected ({self.device_name}), "
                "disabling DynamicVRAM, cudaMallocAsync, and AsyncOffload"
            )

            # Small-vGPU back-to-back VAE-decode hang prevention. The
            # vGPU triple stops the FIRST prompt from crashing; without
            # these two extra flags the SECOND back-to-back SDXL prompt
            # silently deadlocks during VAE decode on partitions in the
            # 8-12 GB range (validated on A10-12Q, 2026-05-30). The
            # threshold is generous on purpose - any partition <=12 GB
            # benefits and bigger partitions get only a tiny reload-cache
            # cost. Skip dedupe-aware injection so a user-supplied
            # different value via LOCALAI_COMFYUI_EXTRA_FLAGS can still
            # override (the env-var pass below applies AFTER and dedupes
            # only on exact-string match).
            vram_gb = self.vram_gb
            if 0 < vram_gb <= 12:
                small_vgpu_flags = ["--reserve-vram", "1.5", "--cache-none"]
                for flag in small_vgpu_flags:
                    if flag not in flags:
                        flags.append(flag)
                logger.info(
                    f"Small vGPU detected ({vram_gb:.1f} GB <= 12), "
                    "adding --reserve-vram 1.5 and --cache-none to prevent "
                    "back-to-back VAE-decode hang"
                )

        # User-supplied extra flags via env var. Tokenizer accepts both ';'
        # and the platform-conventional pip-style separators; the env var
        # name is documented in the vGPU lessons referenced from docs/architecture.md.
        extra = (os.environ.get("LOCALAI_COMFYUI_EXTRA_FLAGS") or "").strip()
        if extra:
            for raw in _split_extra_flags(extra):
                if raw and raw not in flags:
                    flags.append(raw)
            logger.info(
                f"LOCALAI_COMFYUI_EXTRA_FLAGS applied: {extra}"
            )
        return flags

    def get_comfyui_flag(self) -> str:
        """Return the appropriate ComfyUI command-line flag (legacy)."""
        flags = self.get_comfyui_flags()
        return flags[0] if flags else ""

    def __str__(self) -> str:
        if self.device_name:
            return f"{self.gpu_type.upper()} ({self.device_name})"
        return self.gpu_type.upper()


def _nvidia_gpu_present() -> Optional[str]:
    """Check for an NVIDIA GPU using nvidia-smi.

    Returns the GPU name string or None.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            **_subprocess_flags(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def _torch_cuda_state() -> tuple[str, str]:
    """Return a coarse PyTorch CUDA status and human-readable details."""
    try:
        import torch
    except ImportError:
        return "missing", "PyTorch is not installed"
    except Exception as e:
        return "error", f"PyTorch import failed: {e}"

    version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        if torch.cuda.is_available():
            return "available", f"PyTorch {version}, CUDA {cuda_version}"
    except Exception as e:
        return "cuda_unavailable", f"PyTorch {version} CUDA check failed: {e}"

    if cuda_version or "+cu" in version:
        return "cuda_unavailable", f"PyTorch {version} has CUDA {cuda_version} but CUDA is unavailable"
    return "cpu_wheel", f"PyTorch {version} is CPU-only"


def _fix_pytorch_cuda(python_exe: Optional[str] = None) -> bool:
    """Uninstall CPU-only PyTorch and install CUDA 12.8 version.

    Returns True on success.  Windows only.
    """
    py = python_exe or sys.executable
    logger.info("Auto-fixing PyTorch: installing CUDA 12.8 build …")

    try:
        subprocess.run(
            [py, "-m", "pip", "uninstall", "-y",
             "torch", "torchvision", "torchaudio", "torch-directml"],
            capture_output=True, timeout=120,
            **_subprocess_flags(),
        )
        result = subprocess.run(
            [py, "-m", "pip", "install",
             "--no-input", "--disable-pip-version-check",
             "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu128"],
            capture_output=True, text=True, timeout=600,
            **_subprocess_flags(),
        )
        if result.returncode == 0:
            verify = subprocess.run(
                [py, "-c",
                 "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after install'; "
                 "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"],
                capture_output=True, text=True, timeout=60,
                **_subprocess_flags(),
            )
            if verify.returncode == 0:
                logger.info(f"PyTorch CUDA verified: {verify.stdout.strip()}")
                return True
            logger.error(f"PyTorch CUDA install did not verify: {verify.stderr[:500]}")
            return False
        else:
            logger.error(f"pip install failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        logger.error(f"Failed to install CUDA PyTorch: {e}")
        return False


def _apple_gpu_name() -> Optional[str]:
    """Detect Apple GPU chip name. macOS only.
    Uses sysctl first (instant) and falls back to system_profiler (slow).
    """
    # Fast path: sysctl is instant
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and "Apple" in result.stdout:
            return result.stdout.strip()
    except Exception:
        pass
    # Slow fallback: system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                return displays[0].get("sppci_model", "Apple GPU")
    except Exception:
        pass
    return None


def _detect_npu_info() -> "tuple":
    """Best-effort Intel NPU enumeration via OpenVINO. Returns ``(name, devices)``,
    or ``("", [])`` when no NPU / OpenVINO is present. Never raises."""
    if sys.platform != "win32":
        return "", []
    try:
        from src import openvino_client
    except Exception:
        return "", []
    if not getattr(openvino_client, "OV_GENAI_AVAILABLE", False):
        return "", []
    try:
        devices = openvino_client.available_ov_devices()
    except Exception:
        return "", []
    if "NPU" not in devices:
        return "", []
    try:
        name = openvino_client.ov_device_full_name("NPU")
    except Exception:
        name = "NPU"
    return (name or "NPU"), list(devices)


def _augment_npu(info: "GPUInfo") -> None:
    """Populate ``info.npu_name`` / ``info.npu_devices`` and log once when an
    NPU is found. Supplementary to ``gpu_type`` — does not change it."""
    name, devices = _detect_npu_info()
    info.npu_name = name
    info.npu_devices = devices
    if name:
        logger.info(f"Detected NPU: {name}")


def detect_gpu(auto_fix: bool = False) -> GPUInfo:
    """Detect the best GPU backend and augment it with NPU presence.

    Thin public wrapper over :func:`_detect_gpu_impl`. The base call decides
    ``gpu_type`` (CUDA / DirectML / MPS / CPU); this wrapper additionally
    enumerates an Intel NPU via OpenVINO (Windows only) and records it on the
    returned :class:`GPUInfo` so the rest of the app can route OpenVINO chat
    workloads onto the NPU without re-probing.
    """
    info = _detect_gpu_impl(auto_fix=auto_fix)
    _augment_npu(info)
    return info


def _detect_gpu_impl(auto_fix: bool = False) -> GPUInfo:
    """
    Detect the best available GPU acceleration method.

    Windows priority: NVIDIA CUDA > DirectML > CPU
    macOS priority:   Apple Metal (MPS) > CPU

    Startup detection is non-mutating by default. If *auto_fix* is explicitly
    True and an NVIDIA GPU is present with a CPU-only PyTorch wheel,
    reinstall PyTorch with CUDA support (Windows).

    Returns:
        GPUInfo object with detection results
    """
    # ── macOS: check for Apple Metal (MPS) ────────────────────────────────
    if sys.platform == "darwin":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = _apple_gpu_name() or "Apple Metal"
                logger.info(f"Detected Apple Metal (MPS): {device_name}")
                return GPUInfo("mps", device_name)
        except Exception as e:
            logger.debug(f"MPS check failed: {e}")

        # Fallback: MPS hardware present but PyTorch not installed or old
        gpu_name = _apple_gpu_name()
        if gpu_name:
            logger.info(f"Apple GPU detected ({gpu_name}) but MPS unavailable "
                        "(PyTorch may not be installed). Using CPU.")
        logger.info("No GPU acceleration detected on macOS, using CPU")
        return GPUInfo("cpu", "CPU")

    # ── Windows: CUDA > DirectML > CPU ────────────────────────────────────
    # Try CUDA first (NVIDIA GPUs)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Detected CUDA GPU: {device_name}")
            return GPUInfo("cuda", device_name)
    except Exception as e:
        logger.debug(f"CUDA check failed: {e}")

    nvidia_name = _nvidia_gpu_present()
    if nvidia_name:
        torch_state, torch_detail = _torch_cuda_state()
        if torch_state in ("cpu_wheel", "missing"):
            if auto_fix:
                logger.warning(
                    f"NVIDIA GPU detected ({nvidia_name}) but CUDA PyTorch is unavailable "
                    f"({torch_detail}). Installing CUDA PyTorch because auto_fix=True."
                )
                if _fix_pytorch_cuda():
                    logger.info(f"CUDA PyTorch installed for {nvidia_name}. "
                                "GPU will be used on next ComfyUI start.")
                    return GPUInfo("cuda", nvidia_name)
                logger.error("Auto-fix failed. Run fix_nvidia_pytorch.bat manually.")
            else:
                logger.warning(
                    f"NVIDIA GPU detected ({nvidia_name}) but CUDA PyTorch is unavailable "
                    f"({torch_detail}). Startup will not install packages; run setup.bat "
                    "or fix_nvidia_pytorch.bat to enable CUDA acceleration."
                )
        elif torch_state == "cuda_unavailable":
            logger.warning(
                f"NVIDIA GPU detected ({nvidia_name}) and CUDA PyTorch is installed, "
                f"but CUDA is unavailable ({torch_detail}). Check the NVIDIA driver or "
                "run fix_nvidia_pytorch.bat."
            )
        elif torch_state == "error":
            logger.warning(
                f"NVIDIA GPU detected ({nvidia_name}) but PyTorch could not be checked "
                f"({torch_detail}). Falling back to other acceleration backends."
            )

    # Try DirectML (Windows integrated GPU/NPU)
    if is_snapdragon_arm64():
        # torch-directml has no Windows-ARM64 wheel on PyPI, so DirectML
        # cannot run on Snapdragon X. Returning CPU here keeps the rest of
        # the app (chat / vision / ONNX / embeddings) working; the image
        # generation page renders an "unsupported on Snapdragon ARM64"
        # panel separately so users see a friendly message instead of the
        # torch_library_impl symbol popup when ComfyUI starts.
        logger.info(
            "Snapdragon ARM64 detected — torch-directml is x64-only on Windows. "
            "Image generation will be unavailable. Chat / vision / ONNX paths "
            "still work on CPU."
        )
        return GPUInfo("cpu", "CPU (Snapdragon ARM64)")

    try:
        import torch_directml
        try:
            device = torch_directml.device()
            device_name = "DirectML Device"
            logger.info(f"Detected DirectML GPU: {device_name}")
            return GPUInfo("directml", device_name)
        except Exception:
            logger.info("DirectML package found")
            return GPUInfo("directml", "DirectML")
    except ImportError:
        logger.debug("DirectML not available (torch-directml not installed)")
    except Exception as e:
        logger.debug(f"DirectML check failed: {e}")

    # Fallback to CPU
    logger.info("No GPU acceleration detected, using CPU")
    return GPUInfo("cpu", "CPU")


def detect_gpu_cached(
    cache_path: "Optional[object]" = None,
    ttl_hours: float = 24.0,
    cpu_ttl_hours: float = 1.0,
    auto_fix: bool = False,
    force_refresh: bool = False,
) -> GPUInfo:
    """
    Cached wrapper around :func:`detect_gpu`.

    The first call performs full detection (subprocess + PyTorch probes — typically
    1-3 s on Windows). Subsequent calls within *ttl_hours* hours read from a JSON
    cache file and return quickly; cached CUDA entries still validate the current
    PyTorch CUDA state before reuse.

    Cache file (default: ``~/.localai/gpu_cache.json``) stores ``gpu_type``,
    ``device_name`` and an ISO ``cached_at`` timestamp.

    The cache is invalidated automatically when:
      * It is older than ``ttl_hours``
      * The cached ``gpu_type`` is ``"cpu"`` (CPU results are not reused so
        driver or package fixes are picked up on the next launch)
      * The cached ``gpu_type`` is ``"cuda"`` but the current PyTorch
        installation no longer reports CUDA available
      * ``force_refresh=True`` is passed

    On any cache read / write error we fall back to the live detector. This is
    intentionally never a hard failure — the cache is purely a speed optimisation.

    Args:
        cache_path: Path-like (str or pathlib.Path) to the cache file. If
            None, defaults to ``~/.localai/gpu_cache.json``.
        ttl_hours: Maximum age of a GPU cache entry in hours. Default 24.
        cpu_ttl_hours: Deprecated compatibility argument; CPU cache entries are
            never reused.
        auto_fix: Passed through to :func:`detect_gpu` on a cache miss.
        force_refresh: If True, ignore the cache entirely.

    Returns:
        :class:`GPUInfo` describing the active acceleration backend.
    """
    import json
    import time as _time
    from pathlib import Path

    if cache_path is None:
        try:
            cache_path = Path.home() / ".localai" / "gpu_cache.json"
        except Exception:
            cache_path = None
    else:
        cache_path = Path(cache_path)

    if cache_path and not force_refresh:
        try:
            if cache_path.is_file():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_at = float(data.get("cached_at_epoch", 0))
                age_h = (_time.time() - cached_at) / 3600.0
                gtype = data.get("gpu_type")
                dname = data.get("device_name", "")
                cached_gpu_valid = gtype in ("cuda", "mps", "directml") and age_h < ttl_hours
                if cached_gpu_valid and gtype == "cuda":
                    torch_state, torch_detail = _torch_cuda_state()
                    if torch_state != "available":
                        cached_gpu_valid = False
                        logger.warning(
                            f"GPU cache has CUDA ({dname}) but CUDA PyTorch is not available "
                            f"({torch_detail}) — re-detecting"
                        )
                if cached_gpu_valid and gtype == "directml" and sys.platform == "win32":
                    if _nvidia_gpu_present():
                        cached_gpu_valid = False
                        logger.debug(
                            "GPU cache has DirectML but NVIDIA hardware is present — re-detecting"
                        )
                if cached_gpu_valid:
                    logger.info(
                        f"GPU cache hit ({gtype}: {dname}, "
                        f"age={age_h:.1f}h)"
                    )
                    info = GPUInfo(gtype, dname)
                    _augment_npu(info)
                    return info
                logger.debug(
                    f"GPU cache stale or unusable (type={gtype}, age={age_h:.1f}h) — re-detecting"
                )
        except Exception as e:
            logger.debug(f"GPU cache read failed: {e}")

    info = detect_gpu(auto_fix=auto_fix)

    if cache_path is not None and info.gpu_type != "cpu":
        try:
            from src.persistence import atomic_write_json
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                cache_path,
                {
                    "gpu_type": info.gpu_type,
                    "device_name": info.device_name,
                    "cached_at_epoch": _time.time(),
                },
                indent=2,
                ensure_ascii=False,
            )
        except Exception as e:
            logger.debug(f"GPU cache write failed: {e}")
    elif cache_path is not None:
        try:
            cache_path.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"GPU CPU-cache cleanup failed: {e}")

    return info


def get_pytorch_device_info() -> str:
    """
    Get a human-readable description of the PyTorch installation.

    Returns:
        String describing PyTorch version and capabilities
    """
    try:
        import torch
        version = torch.__version__

        capabilities = []
        if torch.cuda.is_available():
            capabilities.append(f"CUDA {torch.version.cuda}")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            capabilities.append("Metal (MPS)")

        try:
            import torch_directml
            capabilities.append("DirectML")
        except ImportError:
            pass

        if capabilities:
            return f"PyTorch {version} ({', '.join(capabilities)})"
        else:
            return f"PyTorch {version} (CPU only)"
    except ImportError:
        return "PyTorch not installed"
    except Exception as e:
        return f"PyTorch check failed: {e}"

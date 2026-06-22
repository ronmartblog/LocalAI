import json
import os
import platform as _platform
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

from src import gpu_detect


class FakeCuda:
    def __init__(self, available=False, name="NVIDIA A10-8Q"):
        self._available = available
        self._name = name

    def is_available(self):
        return self._available

    def get_device_name(self, _index):
        return self._name


class GpuDetectTests(unittest.TestCase):
    def _with_fake_torch(self, torch_module):
        previous = sys.modules.get("torch", None)
        had_previous = "torch" in sys.modules
        sys.modules["torch"] = torch_module

        def restore():
            if had_previous:
                sys.modules["torch"] = previous
            else:
                sys.modules.pop("torch", None)

        self.addCleanup(restore)

    def _block_torch_directml(self):
        previous = sys.modules.get("torch_directml", None)
        had_previous = "torch_directml" in sys.modules
        sys.modules["torch_directml"] = None

        def restore():
            if had_previous:
                sys.modules["torch_directml"] = previous
            else:
                sys.modules.pop("torch_directml", None)

        self.addCleanup(restore)

    def _force_windows_nvidia(self):
        original_platform = gpu_detect.sys.platform
        original_nvidia = gpu_detect._nvidia_gpu_present
        gpu_detect.sys.platform = "win32"
        gpu_detect._nvidia_gpu_present = lambda: "NVIDIA A10-8Q"

        def restore():
            gpu_detect.sys.platform = original_platform
            gpu_detect._nvidia_gpu_present = original_nvidia

        self.addCleanup(restore)

    def test_detect_gpu_does_not_install_cuda_by_default(self):
        self._force_windows_nvidia()
        self._block_torch_directml()
        # CUDA is probed out-of-process now; mock the OOP probes to report a
        # CPU-only wheel (no in-process torch import on the detection path).
        original_name = gpu_detect._cuda_device_name_oop
        original_state = gpu_detect._torch_cuda_available_oop
        gpu_detect._cuda_device_name_oop = lambda *a, **kw: (False, "")
        gpu_detect._torch_cuda_available_oop = lambda *a, **kw: ("cpu_wheel", "PyTorch 2.9.0+cpu is CPU-only")
        self.addCleanup(lambda: setattr(gpu_detect, "_cuda_device_name_oop", original_name))
        self.addCleanup(lambda: setattr(gpu_detect, "_torch_cuda_available_oop", original_state))
        calls = []
        original_fix = gpu_detect._fix_pytorch_cuda
        gpu_detect._fix_pytorch_cuda = lambda: calls.append("fix") or True
        self.addCleanup(lambda: setattr(gpu_detect, "_fix_pytorch_cuda", original_fix))

        info = gpu_detect.detect_gpu()

        self.assertEqual(info.gpu_type, "cpu")
        self.assertEqual(calls, [])

    def test_detect_gpu_does_not_reinstall_when_cuda_wheel_cannot_initialize(self):
        self._force_windows_nvidia()
        self._block_torch_directml()
        # CUDA build present but the runtime can't initialize — probed
        # out-of-process. Must NOT trigger an auto-reinstall even with auto_fix.
        original_name = gpu_detect._cuda_device_name_oop
        original_state = gpu_detect._torch_cuda_available_oop
        gpu_detect._cuda_device_name_oop = lambda *a, **kw: (False, "")
        gpu_detect._torch_cuda_available_oop = lambda *a, **kw: ("cuda_unavailable", "PyTorch 2.9.0+cu128 CUDA unavailable")
        self.addCleanup(lambda: setattr(gpu_detect, "_cuda_device_name_oop", original_name))
        self.addCleanup(lambda: setattr(gpu_detect, "_torch_cuda_available_oop", original_state))
        calls = []
        original_fix = gpu_detect._fix_pytorch_cuda
        gpu_detect._fix_pytorch_cuda = lambda: calls.append("fix") or True
        self.addCleanup(lambda: setattr(gpu_detect, "_fix_pytorch_cuda", original_fix))

        info = gpu_detect.detect_gpu(auto_fix=True)

        self.assertEqual(info.gpu_type, "cpu")
        self.assertEqual(calls, [])

    def test_cpu_gpu_cache_entries_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "gpu_cache.json"
            cache_path.write_text(json.dumps({
                "gpu_type": "cpu",
                "device_name": "CPU",
                "cached_at_epoch": time.time(),
            }), encoding="utf-8")

            calls = []
            original_detect = gpu_detect.detect_gpu
            gpu_detect.detect_gpu = lambda auto_fix=False: calls.append(auto_fix) or gpu_detect.GPUInfo("cpu", "CPU")
            self.addCleanup(lambda: setattr(gpu_detect, "detect_gpu", original_detect))

            info = gpu_detect.detect_gpu_cached(cache_path=cache_path)
            cache_exists_after_detect = cache_path.exists()

        self.assertEqual(info.gpu_type, "cpu")
        self.assertEqual(calls, [False])
        self.assertFalse(cache_exists_after_detect)

    def test_directml_gpu_cache_is_not_reused_when_nvidia_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "gpu_cache.json"
            cache_path.write_text(json.dumps({
                "gpu_type": "directml",
                "device_name": "DirectML",
                "cached_at_epoch": time.time(),
            }), encoding="utf-8")

            original_platform = gpu_detect.sys.platform
            original_nvidia = gpu_detect._nvidia_gpu_present
            original_detect = gpu_detect.detect_gpu
            calls = []
            gpu_detect.sys.platform = "win32"
            gpu_detect._nvidia_gpu_present = lambda: "NVIDIA A10-8Q"
            gpu_detect.detect_gpu = lambda auto_fix=False: calls.append(auto_fix) or gpu_detect.GPUInfo("cuda", "NVIDIA A10-8Q")
            self.addCleanup(lambda: setattr(gpu_detect.sys, "platform", original_platform))
            self.addCleanup(lambda: setattr(gpu_detect, "_nvidia_gpu_present", original_nvidia))
            self.addCleanup(lambda: setattr(gpu_detect, "detect_gpu", original_detect))

            info = gpu_detect.detect_gpu_cached(cache_path=cache_path)

        self.assertEqual(info.gpu_type, "cuda")
        self.assertEqual(info.device_name, "NVIDIA A10-8Q")
        self.assertEqual(calls, [False])

    def test_cuda_gpu_cache_is_not_reused_when_torch_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "gpu_cache.json"
            cache_path.write_text(json.dumps({
                "gpu_type": "cuda",
                "device_name": "NVIDIA A10-8Q",
                "cached_at_epoch": time.time(),
            }), encoding="utf-8")

            # Cache validation now runs out-of-process (so a cold `import torch`
            # cannot freeze the UI). Patch the OOP probe to report a CPU-only
            # wheel — the cuda cache entry must then NOT be reused.
            original_oop = gpu_detect._torch_cuda_available_oop
            gpu_detect._torch_cuda_available_oop = lambda *a, **kw: ("cpu_wheel", "PyTorch 2.9.0+cpu is CPU-only")
            self.addCleanup(lambda: setattr(gpu_detect, "_torch_cuda_available_oop", original_oop))
            self._block_torch_directml()
            original_detect = gpu_detect.detect_gpu
            calls = []
            gpu_detect.detect_gpu = lambda auto_fix=False: calls.append(auto_fix) or gpu_detect.GPUInfo("cpu", "CPU")
            self.addCleanup(lambda: setattr(gpu_detect, "detect_gpu", original_detect))

            info = gpu_detect.detect_gpu_cached(cache_path=cache_path)

        self.assertEqual(info.gpu_type, "cpu")
        self.assertEqual(calls, [False])

    def test_cuda_gpu_cache_is_reused_when_oop_probe_confirms_cuda(self):
        """A fresh CUDA cache entry is reused (no re-detect) when the
        out-of-process probe confirms CUDA is available."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "gpu_cache.json"
            cache_path.write_text(json.dumps({
                "gpu_type": "cuda",
                "device_name": "NVIDIA A10-8Q",
                "cached_at_epoch": time.time(),
            }), encoding="utf-8")

            original_oop = gpu_detect._torch_cuda_available_oop
            gpu_detect._torch_cuda_available_oop = lambda *a, **kw: ("available", "PyTorch 2.9.0+cu128 (CUDA available)")
            self.addCleanup(lambda: setattr(gpu_detect, "_torch_cuda_available_oop", original_oop))
            original_detect = gpu_detect.detect_gpu
            calls = []
            gpu_detect.detect_gpu = lambda auto_fix=False: calls.append(auto_fix) or gpu_detect.GPUInfo("cpu", "CPU")
            self.addCleanup(lambda: setattr(gpu_detect, "detect_gpu", original_detect))

            info = gpu_detect.detect_gpu_cached(cache_path=cache_path)

        self.assertEqual(info.gpu_type, "cuda")
        self.assertEqual(info.device_name, "NVIDIA A10-8Q")
        self.assertEqual(calls, [])  # cache hit — detect_gpu must NOT be called

    def test_oop_torch_probe_uses_subprocess_not_in_process_import(self):
        """The OOP CUDA validator must shell out (own GIL) rather than
        `import torch` in-process — that import is what froze the UI."""
        captured = {}

        def fake_run(cmd, *a, **kw):
            captured["cmd"] = cmd
            return types.SimpleNamespace(stdout="available|2.9.0+cu128|12.8", stderr="", returncode=0)

        original_run = gpu_detect.subprocess.run
        gpu_detect.subprocess.run = fake_run
        self.addCleanup(lambda: setattr(gpu_detect.subprocess, "run", original_run))

        state, _ = gpu_detect._torch_cuda_available_oop()

        self.assertEqual(state, "available")
        self.assertEqual(captured["cmd"][0], sys.executable)
        self.assertIn("-c", captured["cmd"])
        self.assertTrue(any("import torch" in part for part in captured["cmd"]))



class SnapdragonArm64Tests(unittest.TestCase):
    """v5.5.9 (Ron, 2026-05-26): torch-directml has no Windows-ARM64 wheel
    on PyPI, so ComfyUI image generation cannot run on Snapdragon X. The
    ``is_snapdragon_arm64()`` helper drives short-circuits in setup.bat,
    fix_directml_pytorch.bat, and the Image Generation page so users see a
    friendly explanation rather than the "Entry Point Not Found:
    torch_library_impl ... _torchaudio.pyd" popup."""

    def _patch(self, platform_str: str, machine_str: str):
        original_platform = gpu_detect.sys.platform
        original_machine = gpu_detect.platform.machine
        gpu_detect.sys.platform = platform_str
        gpu_detect.platform.machine = lambda: machine_str
        self.addCleanup(lambda: setattr(gpu_detect.sys, "platform", original_platform))
        self.addCleanup(lambda: setattr(gpu_detect.platform, "machine", original_machine))

    def test_true_on_windows_arm64(self):
        self._patch("win32", "ARM64")
        self.assertTrue(gpu_detect.is_snapdragon_arm64())

    def test_true_on_windows_arm64_lowercase(self):
        # Some Python builds report 'arm64' rather than 'ARM64'. Detection is
        # case-insensitive.
        self._patch("win32", "arm64")
        self.assertTrue(gpu_detect.is_snapdragon_arm64())

    def test_false_on_windows_amd64(self):
        self._patch("win32", "AMD64")
        self.assertFalse(gpu_detect.is_snapdragon_arm64())

    def test_false_on_apple_silicon(self):
        # Apple Silicon also reports arm64 but is fully supported via MPS;
        # this helper must not fire on darwin.
        self._patch("darwin", "arm64")
        self.assertFalse(gpu_detect.is_snapdragon_arm64())

    def test_false_on_linux_arm64(self):
        # Linux ARM64 is not a LocalAI Studio target today, but the helper is
        # specifically about *Windows* ARM64 / torch-directml — keep the
        # invariant tight to avoid false-positives in CI.
        self._patch("linux", "aarch64")
        self.assertFalse(gpu_detect.is_snapdragon_arm64())

    def test_detect_gpu_returns_cpu_snapdragon_label_on_arm64(self):
        """On Windows ARM64, detect_gpu must NOT attempt to import
        torch_directml (it has no ARM64 wheel and the import side-effects
        can be slow/noisy). The CPU GPUInfo must carry a 'Snapdragon ARM64'
        label so the UI can detect the SKU even without holding the bool."""
        # Patch platform + nvidia detection.
        self._patch("win32", "ARM64")
        original_nvidia = gpu_detect._nvidia_gpu_present
        gpu_detect._nvidia_gpu_present = lambda: None
        self.addCleanup(lambda: setattr(gpu_detect, "_nvidia_gpu_present", original_nvidia))

        # The CUDA branch in detect_gpu() probes torch CUDA OUT-OF-PROCESS
        # first — on a dev box with real CUDA this would return "cuda" before
        # the Snapdragon check ever fires. Mock the OOP probe to report no CUDA
        # so we exercise the Snapdragon short-circuit specifically.
        original_name = gpu_detect._cuda_device_name_oop
        gpu_detect._cuda_device_name_oop = lambda *a, **kw: (False, "")
        self.addCleanup(lambda: setattr(gpu_detect, "_cuda_device_name_oop", original_name))

        # Sentinel: torch_directml should NEVER be imported on Snapdragon —
        # leave it unmocked. If detect_gpu reached the import line, an
        # ImportError-on-real-systems would still be caught and we'd return
        # CPU but WITHOUT the "Snapdragon ARM64" device label.

        info = gpu_detect.detect_gpu()
        self.assertEqual(info.gpu_type, "cpu")
        self.assertIn("Snapdragon", info.device_name)


class VGpuDetectionPatternTests(unittest.TestCase):
    """Field-tested vGPU/MIG device-name patterns from the vGPU lessons
    doc (§1). The helper `_name_matches_vgpu_profile` must:

      * recognise classic Q-profile suffixes (A10-24Q, T4-16Q),
      * recognise compute-only C-profile suffixes (A10-24C, T4-16C),
      * recognise the MIG-Capable Graphics GPU DC-N-NNQ form (DC-1-24Q),
      * recognise the legacy GRID device-name prefix (GRID K2, GRID K520),
      * **not** false-positive on bare-metal cards whose names happen to
        contain digits or hyphens but no vGPU marker (Generic Graphics
        GPU, GeForce RTX, A100-SXM4-40GB, M1 Max).

    These match the canonical table in the lessons doc; any regression
    here re-opens the VBAR-allocation crash class of bugs.
    """

    def test_a10_q_profile(self):
        self.assertTrue(gpu_detect._name_matches_vgpu_profile("NVIDIA A10-24Q"))

    def test_a10_smaller_q_profile(self):
        self.assertTrue(gpu_detect._name_matches_vgpu_profile("NVIDIA A10-12Q"))

    def test_t4_c_profile_compute_only(self):
        self.assertTrue(gpu_detect._name_matches_vgpu_profile("NVIDIA T4-16C"))

    def test_a10_c_profile_compute_only(self):
        self.assertTrue(gpu_detect._name_matches_vgpu_profile("NVIDIA A10-24C"))

    def test_mig_capable_graphics_gpu_dc_profile(self):
        # Real-world name reported by nvidia-smi on a MIG-Capable Graphics GPU host.
        self.assertTrue(gpu_detect._name_matches_vgpu_profile(
            "NVIDIA Generic Graphics GPU DC-1-24Q"
        ))

    def test_grid_k2_legacy_vgpu(self):
        self.assertTrue(gpu_detect._name_matches_vgpu_profile("GRID K2"))

    def test_grid_m60_with_q_suffix(self):
        # Covered by both regexes — should still return True.
        self.assertTrue(gpu_detect._name_matches_vgpu_profile("GRID M60-8Q"))

    def test_bare_metal_mig_capable_graphics_gpu_not_vgpu(self):
        self.assertFalse(gpu_detect._name_matches_vgpu_profile(
            "NVIDIA Generic Graphics GPU"
        ))

    def test_bare_metal_geforce_not_vgpu(self):
        self.assertFalse(gpu_detect._name_matches_vgpu_profile(
            "NVIDIA GeForce Consumer Card"
        ))

    def test_bare_metal_a100_sxm_not_vgpu(self):
        # A100 SXM4 names carry digits + hyphens but no vGPU profile token.
        self.assertFalse(gpu_detect._name_matches_vgpu_profile(
            "NVIDIA A100-SXM4-40GB"
        ))

    def test_apple_silicon_not_vgpu(self):
        self.assertFalse(gpu_detect._name_matches_vgpu_profile("Apple M1 Max"))

    def test_empty_name_not_vgpu(self):
        self.assertFalse(gpu_detect._name_matches_vgpu_profile(""))


class ComfyUiFlagTests(unittest.TestCase):
    """Pins the mandatory ComfyUI launch-flag triple on vGPU and the
    LOCALAI_COMFYUI_EXTRA_FLAGS env-var escape hatch.

    The triple (`--disable-dynamic-vram`, `--disable-cuda-malloc`,
    `--disable-async-offload`) is from the lessons doc §2; missing any
    one of them surfaces a `MemoryError: VBAR allocation failed` on
    first model patch / KSampler load. The env var is the documented
    escape hatch for early-generation NVIDIA driver regressions
    (lessons §9) and tight-memory vGPU tweaks.
    """

    def setUp(self):
        # Make sure the env var doesn't bleed between tests or from the
        # surrounding shell.
        self._saved = os.environ.pop("LOCALAI_COMFYUI_EXTRA_FLAGS", None)
        self.addCleanup(self._restore_env)
        # Stub subprocess.run so the nvidia-smi MIG/name fallback in
        # is_virtual_gpu cannot leak the host SKU into the test (CI/dev
        # boxes may themselves be on A10-24Q vGPU partitions, which
        # would otherwise pin every test to "vGPU" regardless of the
        # in-process device name).
        self._original_run = gpu_detect.subprocess.run
        gpu_detect.subprocess.run = lambda *a, **kw: types.SimpleNamespace(
            returncode=1, stdout="", stderr=""
        )
        self.addCleanup(
            lambda: setattr(gpu_detect.subprocess, "run", self._original_run)
        )
        # Force the "Windows" branch in is_virtual_gpu — vGPU/MIG only
        # exists on Windows hosts in LocalAI's supported matrix.
        self._original_platform = gpu_detect.sys.platform
        gpu_detect.sys.platform = "win32"
        self.addCleanup(
            lambda: setattr(gpu_detect.sys, "platform", self._original_platform)
        )

    def _restore_env(self):
        if self._saved is None:
            os.environ.pop("LOCALAI_COMFYUI_EXTRA_FLAGS", None)
        else:
            os.environ["LOCALAI_COMFYUI_EXTRA_FLAGS"] = self._saved

    def _force_vgpu(self, info: "gpu_detect.GPUInfo"):
        """Sanity-check that the in-process name is already a vGPU marker."""
        self.assertTrue(info.is_virtual_gpu)

    def test_vgpu_triple_present_and_in_order(self):
        info = gpu_detect.GPUInfo("cuda", "NVIDIA A10-24Q")
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        # All three mandatory flags must be present.
        self.assertIn("--disable-dynamic-vram", flags)
        self.assertIn("--disable-cuda-malloc", flags)
        self.assertIn("--disable-async-offload", flags)
        # The regression-critical contract pins the emission order; assert it.
        triple = [
            f for f in flags
            if f in {
                "--disable-dynamic-vram",
                "--disable-cuda-malloc",
                "--disable-async-offload",
            }
        ]
        self.assertEqual(
            triple,
            [
                "--disable-dynamic-vram",
                "--disable-cuda-malloc",
                "--disable-async-offload",
            ],
        )

    def test_mig_capable_graphics_gpu_vgpu_triple(self):
        info = gpu_detect.GPUInfo(
            "cuda", "NVIDIA Generic Graphics GPU DC-1-24Q"
        )
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        for required in (
            "--disable-dynamic-vram",
            "--disable-cuda-malloc",
            "--disable-async-offload",
        ):
            self.assertIn(required, flags)

    def test_bare_metal_cuda_emits_no_vgpu_triple(self):
        # Bare-metal MIG-Capable Graphics GPU must NOT pay the vGPU flag tax by default
        # (it slightly slows model load + async overlap on bare metal).
        info = gpu_detect.GPUInfo("cuda", "NVIDIA Generic Graphics GPU")
        self.assertFalse(info.is_virtual_gpu)
        flags = info.get_comfyui_flags()
        self.assertNotIn("--disable-dynamic-vram", flags)
        self.assertNotIn("--disable-cuda-malloc", flags)
        self.assertNotIn("--disable-async-offload", flags)

    def test_env_var_extra_flags_semicolon_separated(self):
        os.environ["LOCALAI_COMFYUI_EXTRA_FLAGS"] = (
            "--disable-cuda-malloc;--reserve-vram 1.5"
        )
        info = gpu_detect.GPUInfo("cuda", "NVIDIA GeForce Consumer Card")
        # Bare-metal — but env var still applies (the escape hatch for
        # early-generation driver-regression workarounds, lessons §9).
        flags = info.get_comfyui_flags()
        self.assertIn("--disable-cuda-malloc", flags)
        self.assertIn("--reserve-vram", flags)
        self.assertIn("1.5", flags)

    def test_env_var_extra_flags_space_separated(self):
        os.environ["LOCALAI_COMFYUI_EXTRA_FLAGS"] = "--lowvram --cache-none"
        info = gpu_detect.GPUInfo("cuda", "NVIDIA GeForce Consumer Card")
        flags = info.get_comfyui_flags()
        self.assertIn("--lowvram", flags)
        self.assertIn("--cache-none", flags)

    def test_env_var_extra_flags_deduped_against_vgpu_triple(self):
        # If the user also asks for --disable-cuda-malloc on a vGPU, the
        # triple already contains it — it must appear at most once.
        os.environ["LOCALAI_COMFYUI_EXTRA_FLAGS"] = (
            "--disable-cuda-malloc --reserve-vram 1.5"
        )
        info = gpu_detect.GPUInfo("cuda", "NVIDIA A10-24Q")
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        self.assertEqual(flags.count("--disable-cuda-malloc"), 1)
        self.assertIn("--reserve-vram", flags)

    def test_env_var_extra_flags_empty_or_whitespace_is_noop(self):
        os.environ["LOCALAI_COMFYUI_EXTRA_FLAGS"] = "   "
        info = gpu_detect.GPUInfo("cuda", "NVIDIA GeForce Consumer Card")
        flags = info.get_comfyui_flags()
        # Bare-metal + no env var content = nothing emitted at all.
        self.assertEqual(flags, [])

    # ── Small-vGPU memory tweaks (back-to-back VAE-decode hang fix) ─────
    # Validated on A10-12Q against the SDXL workflow shipped with
    # localai_image-gen.yaml on 2026-05-30. Without --reserve-vram 1.5 +
    # --cache-none, the SECOND back-to-back prompt deadlocks during VAE
    # decode even with the mandatory vGPU triple already applied.

    def _patch_vram(self, vram_gb: float):
        """Stub GPUInfo.vram_gb so tests don't depend on the host GPU."""
        original = gpu_detect.GPUInfo.vram_gb
        gpu_detect.GPUInfo.vram_gb = property(lambda self: vram_gb)
        self.addCleanup(lambda: setattr(gpu_detect.GPUInfo, "vram_gb", original))

    def test_small_vgpu_adds_reserve_vram_and_cache_none(self):
        self._patch_vram(12.0)
        info = gpu_detect.GPUInfo("cuda", "NVIDIA A10-12Q")
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        self.assertIn("--reserve-vram", flags)
        # The two tokens are split: ["--reserve-vram", "1.5"].
        idx = flags.index("--reserve-vram")
        self.assertEqual(flags[idx + 1], "1.5")
        self.assertIn("--cache-none", flags)

    def test_small_vgpu_threshold_is_12gb_inclusive(self):
        self._patch_vram(8.0)
        info = gpu_detect.GPUInfo("cuda", "NVIDIA A10-8Q")
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        self.assertIn("--reserve-vram", flags)
        self.assertIn("--cache-none", flags)

    def test_large_vgpu_does_not_get_small_vgpu_tweaks(self):
        # 24 GB vGPU partitions don't need the back-to-back-hang
        # mitigations and shouldn't pay the reload-cache cost.
        self._patch_vram(24.0)
        info = gpu_detect.GPUInfo("cuda", "NVIDIA A10-24Q")
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        self.assertNotIn("--reserve-vram", flags)
        self.assertNotIn("--cache-none", flags)

    def test_unknown_vram_does_not_add_small_vgpu_tweaks(self):
        # vram_gb=0 means the probe failed (no torch.cuda + nvidia-smi
        # unavailable). Skip the auto-bake rather than guess.
        self._patch_vram(0.0)
        info = gpu_detect.GPUInfo("cuda", "NVIDIA A10-12Q")
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        self.assertNotIn("--reserve-vram", flags)
        self.assertNotIn("--cache-none", flags)

    def test_small_vgpu_does_not_double_apply_with_env_var(self):
        # If the user supplied the same flags via env var, they appear once.
        os.environ["LOCALAI_COMFYUI_EXTRA_FLAGS"] = "--reserve-vram 1.5 --cache-none"
        self._patch_vram(12.0)
        info = gpu_detect.GPUInfo("cuda", "NVIDIA A10-12Q")
        self._force_vgpu(info)
        flags = info.get_comfyui_flags()
        self.assertEqual(flags.count("--reserve-vram"), 1)
        self.assertEqual(flags.count("--cache-none"), 1)
        self.assertEqual(flags.count("1.5"), 1)

    def test_split_extra_flags_handles_quoted_value(self):
        # On macOS/Linux shlex runs in POSIX mode so the user can quote
        # a flag value containing spaces and the quotes are stripped from
        # the resulting tokens.
        original_platform = gpu_detect.sys.platform
        try:
            gpu_detect.sys.platform = "linux"
            tokens = gpu_detect._split_extra_flags(
                '--cache-ram "8 4"'
            )
        finally:
            gpu_detect.sys.platform = original_platform
        self.assertEqual(tokens, ["--cache-ram", "8 4"])

    def test_split_extra_flags_preserves_windows_path_backslashes(self):
        # On Windows shlex must run in non-POSIX mode so backslash-bearing
        # paths survive. Naked `C:\Temp\foo` in POSIX mode would silently
        # collapse to `C:Tempfoo` because `\T` is an escape sequence.
        original_platform = gpu_detect.sys.platform
        try:
            gpu_detect.sys.platform = "win32"
            tokens = gpu_detect._split_extra_flags(
                "--output-directory C:\\models\\out"
            )
        finally:
            gpu_detect.sys.platform = original_platform
        self.assertEqual(
            tokens, ["--output-directory", "C:\\models\\out"]
        )

    @unittest.skipUnless(
        sys.platform == "win32",
        "MIG/vGPU nvidia-smi classification is validated on the Windows "
        "target platform (LocalAI ships Windows-only).",
    )
    def test_mig_mode_enabled_classifies_bare_metal_name_as_vgpu(self):
        # A100/H100 MIG slices report a bare-metal device name through the
        # Python torch APIs but expose `mig.mode.current = Enabled` via
        # nvidia-smi. The fallback branch in is_virtual_gpu MUST classify
        # them as vGPU so they receive the three-flag triple; otherwise
        # MIG users silently re-open the VBAR-allocation crash class.
        original_run = gpu_detect.subprocess.run

        def fake_run(args, **_kw):
            # First nvidia-smi call queries mig.mode.current.
            if "mig.mode.current" in " ".join(args):
                return types.SimpleNamespace(
                    returncode=0, stdout="Enabled\n", stderr=""
                )
            return types.SimpleNamespace(
                returncode=1, stdout="", stderr=""
            )

        gpu_detect.subprocess.run = fake_run
        try:
            info = gpu_detect.GPUInfo("cuda", "NVIDIA A100-SXM4-40GB")
            self.assertTrue(info.is_virtual_gpu)
            flags = info.get_comfyui_flags()
            self.assertIn("--disable-dynamic-vram", flags)
            self.assertIn("--disable-cuda-malloc", flags)
            self.assertIn("--disable-async-offload", flags)
        finally:
            gpu_detect.subprocess.run = original_run


if __name__ == "__main__":
    unittest.main()
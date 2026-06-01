import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.comfyui_benchmark_host import HeadlessComfyUIBenchmarkHost


class HeadlessComfyUIBenchmarkHostTests(unittest.TestCase):
    def _host(self):
        host = object.__new__(HeadlessComfyUIBenchmarkHost)
        host.cfg = {}
        host._comfyui_current_launch_flags = []
        host.comfyui_process = None
        host._comfyui_log_handle = None
        host._comfyui_last_start_failure_reason = ""
        host._comfyui_last_start_failure_lock = threading.Lock()
        host._comfyui_dependency_ok_by_python = {}
        host._comfyui_dependency_lock = threading.Lock()
        host._chroma_node_needs_restart = False
        return host

    def test_model_download_target_places_gguf_in_diffusion_models(self):
        host = self._host()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = host._comfyui_model_download_target(
                {"comfyui_model": "flux1-schnell-Q4_K_S.gguf"},
                root,
            )
        self.assertEqual(target.parts[-2:], ("diffusion_models", "flux1-schnell-Q4_K_S.gguf"))

    def test_model_download_target_honors_diffusion_model_destination(self):
        host = self._host()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = host._comfyui_model_download_target(
                {
                    "comfyui_model": "z_image_turbo_bf16.safetensors",
                    "comfyui_model_dir": "diffusion_models",
                },
                root,
            )
        self.assertEqual(target.parts[-2:], ("diffusion_models", "z_image_turbo_bf16.safetensors"))

    def test_launch_flag_restart_detects_missing_required_flags(self):
        host = self._host()
        host.comfyui = Mock()
        host.comfyui.is_running.return_value = True
        host._active_external_comfyui_flags = lambda: {"--disable-cuda-malloc"}

        needs_restart = host._image_model_launch_flags_need_restart(
            {"comfyui_launch_flags": ["--lowvram"]}
        )

        self.assertTrue(needs_restart)

    def test_launch_flag_restart_accepts_running_process_with_required_flags(self):
        host = self._host()
        host.comfyui = Mock()
        host.comfyui.is_running.return_value = True
        host._active_external_comfyui_flags = lambda: {"--disable-cuda-malloc", "--lowvram"}

        needs_restart = host._image_model_launch_flags_need_restart(
            {"comfyui_launch_flags": ["--lowvram"]}
        )

        self.assertFalse(needs_restart)


if __name__ == "__main__":
    unittest.main()

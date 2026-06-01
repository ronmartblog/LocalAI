# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Headless ComfyUI runtime support for ``run_batch.py`` image benchmarks."""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Optional

import requests

from src import config, logger, system_info
from src.comfyui_client import COMFYUI_DEFAULT_HOST, ComfyUIClient
from src.gpu_detect import GPUInfo, detect_gpu_cached


COMFYUI_CORE_PYTHON_DEPS = {
    "sqlalchemy": "SQLAlchemy",
    "alembic": "alembic",
    "torchsde": "torchsde",
    "av": "av",
    "comfy_kitchen": "comfy-kitchen",
    "comfy_aimdo": "comfy-aimdo",
    "simpleeval": "simpleeval",
    "gguf": "gguf>=0.13.0",
    "yaml": "PyYAML>=5.1",
    "tqdm": "tqdm>=4.27",
    "google.protobuf": "protobuf",
    "pydantic_settings": "pydantic-settings~=2.0",
    "spandrel": "spandrel",
}


def _subprocess_flags() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


class HeadlessComfyUIBenchmarkHost:
    """Owns the ComfyUI client/startup callbacks used by headless benchmarks."""

    _CHROMA_NODE_CODE = '''\
import torch

class ChromaLatentToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "LocalAI"

    def convert(self, samples):
        t = samples["samples"].float()
        t = t.clamp(-1.0, 1.0)
        t = (t + 1.0) / 2.0
        t = t.permute(0, 2, 3, 1)
        return (t,)


NODE_CLASS_MAPPINGS = {"ChromaLatentToImage": ChromaLatentToImage}
NODE_DISPLAY_NAME_MAPPINGS = {"ChromaLatentToImage": "Chroma: Latent To Image"}
'''

    def __init__(self, host: str = COMFYUI_DEFAULT_HOST):
        self.cfg = config.load()
        self.comfyui = ComfyUIClient(host)
        try:
            self.gpu_info = detect_gpu_cached(auto_fix=False)
        except Exception as exc:
            logger.warning(f"ComfyUI benchmark GPU detection failed: {exc}")
            self.gpu_info = GPUInfo("cpu", "CPU")
        self.comfyui_process: subprocess.Popen | None = None
        self._comfyui_current_launch_flags: list[str] = []
        self._comfyui_log_handle = None
        self._comfyui_last_start_failure_reason = ""
        self._comfyui_last_start_failure_lock = threading.Lock()
        self._comfyui_dependency_ok_by_python: dict[str, bool] = {}
        self._comfyui_dependency_lock = threading.Lock()
        self._chroma_node_needs_restart = False

    @property
    def client(self) -> ComfyUIClient:
        return self.comfyui

    def prepare_image_model(
        self, model: dict, stop_event: Optional[threading.Event] = None
    ) -> tuple[bool, str]:
        """Ensure the selected image checkpoint and support files are present."""
        filename = model.get("comfyui_model", "")
        url = model.get("comfyui_model_url", "")
        if not filename:
            return False, "No comfyui_model filename defined for this catalog entry"

        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            return False, "ComfyUI not installed at expected paths"

        missing_support = self._image_model_runtime_support_missing_items(filename)
        if not self._ensure_image_model_runtime_support(filename):
            return False, f"Could not prepare image-model support files for {filename}"

        output_file = self._comfyui_model_download_target(model, comfyui_path)
        downloaded = False
        if not output_file.exists():
            if not url:
                return False, f"Image model '{filename}' is not installed and has no automatic download URL"
            if stop_event is not None and stop_event.is_set():
                return False, "Stopped before image model download"

            free_gb = system_info.get_storage_info(output_file.parent)["free_gb"]
            required_gb = max(5.0, float(model.get("size_gb") or 0.0) + 1.0)
            if free_gb < required_gb:
                return (
                    False,
                    f"Not enough disk space to download {filename}: "
                    f"{free_gb:.1f} GB free, need about {required_gb:.1f} GB",
                )

            try:
                logger.info(
                    f"Benchmark downloading ComfyUI model: {filename}",
                    category=logger.CATEGORY_IMAGE_GEN,
                )
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                self._download_stream_to_path(response, output_file, stop_event=stop_event)
                downloaded = True
                logger.info(
                    f"Benchmark ComfyUI model ready: {filename}",
                    category=logger.CATEGORY_IMAGE_GEN,
                )
            except Exception as exc:
                return False, f"Could not download image model {filename}: {exc}"

        if downloaded or missing_support:
            try:
                if self.comfyui.is_running():
                    self._stop_comfyui_for_restart(
                        reason="benchmark image model/support preparation",
                        kill_orphans=True,
                    )
            except Exception as exc:
                logger.debug(f"Benchmark ComfyUI restart-after-prepare probe failed: {exc}")
        return True, ""

    def ensure_comfyui_ready(
        self, timeout: int = 180, model: dict | None = None
    ) -> tuple[bool, str]:
        """Start/restart ComfyUI for a benchmark run and wait for the API."""
        try:
            try:
                is_running = self.comfyui.is_running()
            except Exception:
                is_running = False
            if is_running:
                if not self._image_model_launch_flags_need_restart(model):
                    return True, ""
                logger.info(
                    "ComfyUI: restarting for benchmark launch flags "
                    f"{self._comfyui_model_launch_flags(model)}",
                    category=logger.CATEGORY_COMFYUI,
                )
                self._stop_comfyui_for_restart(
                    reason="benchmark launch flag change", kill_orphans=True
                )

            self._reset_comfyui_start_failure_reason()
            if not self._start_comfyui_process(model):
                return (
                    False,
                    self._get_comfyui_start_failure_reason()
                    or "ComfyUI subprocess failed to start (no specific reason recorded)",
                )

            deadline = max(5, int(timeout))
            poll_start = time.perf_counter()
            restart_attempted = False
            log_file = Path(__file__).parent.parent / "comfyui.log"
            while time.perf_counter() - poll_start < deadline:
                proc = self.comfyui_process
                if proc is not None:
                    try:
                        exit_code = proc.poll()
                    except (OSError, AttributeError, ValueError) as poll_exc:
                        logger.warning(f"ComfyUI proc.poll() raised: {poll_exc}")
                        exit_code = None
                    if exit_code is not None:
                        tail = self._tail_comfyui_log(log_file, lines=40)
                        signal_line = self._comfyui_startup_signal_line(tail)
                        reason = (
                            f"ComfyUI subprocess (PID {proc.pid}) exited with "
                            f"code {exit_code} during startup - startup signal: "
                            f"{signal_line}"
                        )
                        logger.error(
                            f"{reason}\n--- last 40 lines of {log_file} ---\n{tail}",
                            category=logger.CATEGORY_COMFYUI,
                        )
                        if not restart_attempted:
                            restart_attempted = True
                            self._stop_comfyui_for_restart(
                                reason="benchmark startup early exit",
                                kill_orphans=True,
                            )
                            self._reset_comfyui_start_failure_reason()
                            if not self._start_comfyui_process(model):
                                start_reason = (
                                    self._get_comfyui_start_failure_reason()
                                    or "ComfyUI subprocess failed to restart"
                                )
                                return False, f"{reason}; restart attempt failed: {start_reason}"
                            poll_start = time.perf_counter()
                            continue
                        return False, reason
                try:
                    if self.comfyui.is_running():
                        return True, ""
                except Exception:
                    pass
                time.sleep(2.0)

            proc = self.comfyui_process
            proc_alive = bool(proc and proc.poll() is None)
            exit_code = proc.poll() if proc else None
            tail = self._tail_comfyui_log(log_file, lines=40)
            elapsed_s = int(time.perf_counter() - poll_start)
            reason = (
                "ComfyUI subprocess started but didn't respond on "
                f"/system_stats within {elapsed_s}s - process alive: "
                f"{proc_alive}, exit code: {exit_code}; see comfyui.log"
            )
            logger.error(
                f"{reason}\n--- last 40 lines of {log_file} ---\n{tail}",
                category=logger.CATEGORY_COMFYUI,
            )
            return False, reason
        except Exception as exc:
            reason = f"headless ComfyUI benchmark startup failed: {exc}"
            logger.warning(reason)
            return False, reason

    def _comfyui_installed_path(self) -> Optional[Path]:
        cfg_dir = self.cfg.get("comfyui_dir", "")
        if cfg_dir:
            path = Path(cfg_dir)
            if (path / "main.py").exists():
                self._sync_comfyui_path_bat(path)
                return path

        if sys.platform == "win32":
            bat = Path(__file__).parent.parent / "comfyui_path.bat"
            if bat.exists():
                try:
                    for line in bat.read_text(encoding="utf-8").splitlines():
                        if line.strip().upper().startswith("SET LOCALAI_COMFYUI="):
                            path = Path(line.split("=", 1)[1].strip())
                            if (path / "main.py").exists():
                                self.cfg["comfyui_dir"] = str(path)
                                config.save(self.cfg)
                                self._sync_comfyui_path_bat(path)
                                return path
                except OSError as exc:
                    logger.debug(f"Could not read comfyui_path.bat: {exc}")

        path = Path(__file__).parent.parent / "ComfyUI"
        if (path / "main.py").exists():
            self.cfg["comfyui_dir"] = str(path)
            config.save(self.cfg)
            self._sync_comfyui_path_bat(path)
            return path
        return None

    def _sync_comfyui_path_bat(self, path: Path) -> None:
        if sys.platform != "win32":
            return
        bat = Path(__file__).parent.parent / "comfyui_path.bat"
        text = f"set LOCALAI_COMFYUI={path}\n"
        try:
            if not bat.exists() or bat.read_text(encoding="utf-8") != text:
                bat.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.debug(f"Could not sync comfyui_path.bat: {exc}")

    def _comfyui_python_exe(self) -> str:
        python_exe = sys.executable
        if sys.platform == "win32":
            python_path_file = Path(__file__).parent.parent / "python_path.bat"
            if python_path_file.exists():
                try:
                    for line in python_path_file.read_text(encoding="utf-8").splitlines():
                        if line.startswith("set LOCALAI_PYTHON="):
                            candidate = line.split("=", 1)[1].strip()
                            if candidate:
                                python_exe = candidate
                            break
                except OSError as exc:
                    logger.debug(f"Could not read python_path.bat: {exc}")
        return python_exe

    def _set_comfyui_start_failure_reason(self, reason: str) -> None:
        with self._comfyui_last_start_failure_lock:
            if reason and not self._comfyui_last_start_failure_reason:
                self._comfyui_last_start_failure_reason = reason

    def _reset_comfyui_start_failure_reason(self) -> None:
        with self._comfyui_last_start_failure_lock:
            self._comfyui_last_start_failure_reason = ""

    def _get_comfyui_start_failure_reason(self) -> str:
        with self._comfyui_last_start_failure_lock:
            return self._comfyui_last_start_failure_reason

    def _missing_comfyui_python_modules(self, python_exe: str, modules: list[str]) -> list[str]:
        check_code = (
            "import importlib.metadata, importlib.util, sys; "
            f"modules={modules!r}; "
            "missing=[]\n"
            "for m in modules:\n"
            "    if m.startswith('dist:'):\n"
            "        try:\n"
            "            importlib.metadata.version(m[5:])\n"
            "        except importlib.metadata.PackageNotFoundError:\n"
            "            missing.append(m)\n"
            "    elif importlib.util.find_spec(m) is None:\n"
            "        missing.append(m)\n"
            "print('\\n'.join(missing)); "
            "sys.exit(1 if missing else 0)"
        )
        result = subprocess.run(
            [python_exe, "-c", check_code],
            capture_output=True,
            text=True,
            timeout=30,
            **_subprocess_flags(),
        )
        if result.returncode == 0:
            return []
        if result.returncode == 1:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"dependency probe failed with exit code {result.returncode}")

    def _ensure_comfyui_core_dependencies(self, python_exe: str) -> bool:
        cache_key = str(python_exe)
        if self._comfyui_dependency_ok_by_python.get(cache_key):
            return True
        modules = list(COMFYUI_CORE_PYTHON_DEPS)
        try:
            missing = self._missing_comfyui_python_modules(python_exe, modules)
        except Exception as exc:
            reason = f"ComfyUI dependency probe failed: {exc}"
            self._set_comfyui_start_failure_reason(reason)
            logger.error(reason, category=logger.CATEGORY_COMFYUI)
            return False
        if not missing:
            self._comfyui_dependency_ok_by_python[cache_key] = True
            return True

        with self._comfyui_dependency_lock:
            try:
                missing = self._missing_comfyui_python_modules(python_exe, modules)
            except Exception as exc:
                reason = f"ComfyUI dependency probe failed after lock: {exc}"
                self._set_comfyui_start_failure_reason(reason)
                logger.error(reason, category=logger.CATEGORY_COMFYUI)
                return False
            if not missing:
                self._comfyui_dependency_ok_by_python[cache_key] = True
                return True

            packages = sorted({COMFYUI_CORE_PYTHON_DEPS[module] for module in missing})
            logger.warning(
                "ComfyUI Python dependencies missing "
                f"({', '.join(missing)}); installing {', '.join(packages)}",
                category=logger.CATEGORY_COMFYUI,
            )
            result = subprocess.run(
                [
                    python_exe,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--no-input",
                    "--disable-pip-version-check",
                    *packages,
                ],
                capture_output=True,
                text=True,
                timeout=300,
                **_subprocess_flags(),
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                last = next((line.strip() for line in reversed(detail.splitlines()) if line.strip()), "")
                self._set_comfyui_start_failure_reason(
                    f"ComfyUI dependency install failed: {last or 'pip exited with code ' + str(result.returncode)}"
                )
                logger.error(
                    f"Failed to install ComfyUI Python dependencies: {detail[-1000:]}",
                    category=logger.CATEGORY_COMFYUI,
                )
                return False

            try:
                still_missing = self._missing_comfyui_python_modules(python_exe, modules)
            except Exception as exc:
                reason = f"ComfyUI dependency re-probe after install failed: {exc}"
                self._set_comfyui_start_failure_reason(reason)
                logger.error(reason, category=logger.CATEGORY_COMFYUI)
                return False
            if still_missing:
                reason = (
                    "ComfyUI dependencies still missing after install attempt: "
                    + ", ".join(still_missing)
                )
                self._set_comfyui_start_failure_reason(reason)
                logger.error(reason, category=logger.CATEGORY_COMFYUI)
                return False
            self._comfyui_dependency_ok_by_python[cache_key] = True
            return True

    def _ensure_gguf_support(self, comfyui_path: Path) -> bool:
        custom_nodes_dir = comfyui_path / "custom_nodes"
        gguf_node_dir = custom_nodes_dir / "ComfyUI-GGUF"
        if not gguf_node_dir.exists():
            logger.info("Installing ComfyUI-GGUF custom node...", category=logger.CATEGORY_IMAGE_GEN)
            try:
                import shutil

                custom_nodes_dir.mkdir(parents=True, exist_ok=True)
                response = requests.get(
                    "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip",
                    timeout=60,
                )
                response.raise_for_status()
                tmp_dir = custom_nodes_dir / ".ComfyUI-GGUF.download"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    zf.extractall(tmp_dir)
                extracted = tmp_dir / "ComfyUI-GGUF-main"
                if extracted.exists():
                    shutil.move(str(extracted), str(gguf_node_dir))
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as exc:
                logger.error(f"Failed to install GGUF custom node: {exc}", category=logger.CATEGORY_IMAGE_GEN)
                return False
        return self._ensure_flux_clip_vae(comfyui_path)

    def _ensure_flux_clip_vae(self, comfyui_path: Path) -> bool:
        missing_files = {}
        clip_dir = comfyui_path / "models" / "clip"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for name, url in {
            "t5xxl_fp8_e4m3fn.safetensors": (
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
                "t5xxl_fp8_e4m3fn.safetensors"
            ),
            "clip_l.safetensors": (
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
                "clip_l.safetensors"
            ),
        }.items():
            path = clip_dir / name
            if not path.exists():
                missing_files[path] = url

        vae_dir = comfyui_path / "models" / "vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        vae_path = vae_dir / "ae.safetensors"
        if not vae_path.exists():
            missing_files[vae_path] = "https://huggingface.co/sirorable/flux-ae-vae/resolve/main/ae.safetensors"

        return self._download_support_files("Flux", missing_files)

    def _ensure_z_image_support(self, comfyui_path: Path) -> bool:
        missing_files = {}
        te_dir = comfyui_path / "models" / "text_encoders"
        te_dir.mkdir(parents=True, exist_ok=True)
        qwen_path = te_dir / "qwen_3_4b_fp8_mixed.safetensors"
        if not qwen_path.exists():
            missing_files[qwen_path] = (
                "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
                "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors"
            )

        vae_dir = comfyui_path / "models" / "vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        vae_path = vae_dir / "ae.safetensors"
        if not vae_path.exists():
            missing_files[vae_path] = "https://huggingface.co/sirorable/flux-ae-vae/resolve/main/ae.safetensors"

        return self._download_support_files("Z-Image Turbo", missing_files)

    def _download_support_files(self, label: str, missing_files: dict[Path, str]) -> bool:
        if not missing_files:
            return True
        for dest, url in missing_files.items():
            logger.info(f"Downloading {label} support file: {dest.name}", category=logger.CATEGORY_IMAGE_GEN)
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                self._download_stream_to_path(response, dest)
            except Exception as exc:
                logger.error(f"Failed to download {dest.name}: {exc}", category=logger.CATEGORY_IMAGE_GEN)
                return False
        return True

    def _ensure_chroma_support(self, comfyui_path: Path) -> bool:
        node_dir = comfyui_path / "custom_nodes" / "ComfyUI-LocalAI-Chroma"
        init_file = node_dir / "__init__.py"
        if init_file.exists():
            return True
        try:
            node_dir.mkdir(parents=True, exist_ok=True)
            init_file.write_text(self._CHROMA_NODE_CODE, encoding="utf-8")
            self._chroma_node_needs_restart = True
            logger.info(
                "ChromaLatentToImage custom node written - ComfyUI restart required",
                category=logger.CATEGORY_IMAGE_GEN,
            )
            return True
        except OSError as exc:
            logger.error(f"Failed to write Chroma custom node: {exc}", category=logger.CATEGORY_IMAGE_GEN)
            return False

    def _ensure_image_model_runtime_support(self, model_filename: str) -> bool:
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            return False
        lower = (model_filename or "").lower()
        is_gguf = lower.endswith(".gguf")
        is_chroma = "chroma" in lower
        is_z_image = "z_image" in lower
        is_flux = "flux" in lower or is_chroma or is_gguf
        if is_gguf:
            return self._ensure_gguf_support(comfyui_path)
        if is_z_image:
            return self._ensure_z_image_support(comfyui_path)
        if is_chroma:
            return self._ensure_chroma_support(comfyui_path) and self._ensure_flux_clip_vae(comfyui_path)
        if is_flux:
            return self._ensure_flux_clip_vae(comfyui_path)
        return True

    def _image_model_runtime_support_missing_items(self, model_filename: str) -> list[str]:
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            return ["ComfyUI install"]
        lower = (model_filename or "").lower()
        is_gguf = lower.endswith(".gguf")
        is_chroma = "chroma" in lower
        is_z_image = "z_image" in lower
        is_flux = "flux" in lower or is_chroma or is_gguf
        missing: list[str] = []
        if is_gguf:
            if not (comfyui_path / "custom_nodes" / "ComfyUI-GGUF").exists():
                missing.append("ComfyUI-GGUF custom node")
            if importlib.util.find_spec("gguf") is None:
                missing.append("gguf Python package")
        if is_chroma and not (comfyui_path / "custom_nodes" / "ComfyUI-LocalAI-Chroma" / "__init__.py").exists():
            missing.append("ChromaLatentToImage custom node")
        if is_z_image:
            required = [
                comfyui_path / "models" / "text_encoders" / "qwen_3_4b_fp8_mixed.safetensors",
                comfyui_path / "models" / "vae" / "ae.safetensors",
            ]
            missing.extend(path.name for path in required if not path.exists())
        elif is_flux:
            required = [
                comfyui_path / "models" / "clip" / "t5xxl_fp8_e4m3fn.safetensors",
                comfyui_path / "models" / "clip" / "clip_l.safetensors",
                comfyui_path / "models" / "vae" / "ae.safetensors",
            ]
            missing.extend(path.name for path in required if not path.exists())
        return missing

    def _comfyui_model_download_target(self, model: dict, comfyui_path: Path) -> Path:
        filename = model.get("comfyui_model", "")
        is_gguf = filename.lower().endswith(".gguf")
        catalog_dest = str(model.get("comfyui_model_dest") or model.get("comfyui_model_dir") or "").lower()
        if is_gguf or catalog_dest in {"diffusion_models", "unet", "unets"}:
            model_dir = comfyui_path / "models" / "diffusion_models"
        else:
            model_dir = comfyui_path / "models" / "checkpoints"
        model_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_dir / filename
        if is_gguf:
            legacy_file = comfyui_path / "models" / "checkpoints" / filename
            if legacy_file.exists() and not output_file.exists():
                import shutil

                shutil.move(str(legacy_file), str(output_file))
        return output_file

    def _download_stream_to_path(
        self,
        response,
        dest: Path,
        *,
        stop_event: Optional[threading.Event] = None,
        chunk_size: int = 1_048_576,
    ) -> None:
        partial = dest.with_name(dest.name + ".part")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if partial.exists():
                partial.unlink()
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if stop_event and stop_event.is_set():
                        raise RuntimeError("Download cancelled")
                    if chunk:
                        handle.write(chunk)
            os.replace(partial, dest)
        except Exception:
            try:
                if partial.exists():
                    partial.unlink()
            except OSError:
                pass
            raise

    def _comfyui_model_launch_flags(self, model: dict | None = None) -> list[str]:
        raw = (model or {}).get("comfyui_launch_flags") or []
        if isinstance(raw, str):
            raw = [raw]
        flags: list[str] = []
        for flag in raw:
            text = str(flag or "").strip()
            if text and text not in flags:
                flags.append(text)
        return flags

    def _comfyui_effective_launch_flags(self, model: dict | None = None) -> list[str]:
        flags = list(self.gpu_info.get_comfyui_flags())
        for flag in self._comfyui_model_launch_flags(model):
            if flag not in flags:
                flags.append(flag)
        return flags

    def _active_external_comfyui_flags(self) -> set[str]:
        try:
            import psutil
        except ImportError:
            return set()
        flags: set[str] = set()
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if self.comfyui_process is not None and proc.info["pid"] == self.comfyui_process.pid:
                    continue
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if "ComfyUI" in cmd_str and "main.py" in cmd_str:
                    flags.update(arg for arg in cmdline if str(arg).startswith("--"))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return flags

    def _image_model_launch_flags_need_restart(self, model: dict | None = None) -> bool:
        required = set(self._comfyui_model_launch_flags(model))
        try:
            if not self.comfyui.is_running():
                return False
        except Exception:
            return False
        proc = self.comfyui_process
        owns_live_process = bool(proc and proc.poll() is None)
        active = (
            set(self._comfyui_current_launch_flags)
            if owns_live_process
            else self._active_external_comfyui_flags()
        )
        return bool(required and not required.issubset(active))

    def _start_comfyui_process(self, model: dict | None = None) -> bool:
        if self.comfyui_process and self.comfyui_process.poll() is None:
            if self._image_model_launch_flags_need_restart(model):
                self._stop_comfyui_for_restart(reason="launch flag change", kill_orphans=True)
            else:
                return True

        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            reason = "ComfyUI not installed at expected paths (config.json comfyui_dir, comfyui_path.bat, or ./ComfyUI)"
            self._set_comfyui_start_failure_reason(reason)
            logger.warning(reason, category=logger.CATEGORY_COMFYUI)
            return False

        python_exe = self._comfyui_python_exe()
        logger.info(f"ComfyUI install path: {comfyui_path}", category=logger.CATEGORY_COMFYUI)
        logger.info(f"ComfyUI python: {python_exe}", category=logger.CATEGORY_COMFYUI)
        if not self._ensure_comfyui_core_dependencies(python_exe):
            if not self._get_comfyui_start_failure_reason():
                self._set_comfyui_start_failure_reason("ComfyUI core dependency check/install failed")
            return False

        self._chroma_node_needs_restart = False
        self._ensure_chroma_support(comfyui_path)

        flags = self._comfyui_effective_launch_flags(model)
        cmd = [python_exe, str(comfyui_path / "main.py"), "--listen", "127.0.0.1", *flags]
        log_file = Path(__file__).parent.parent / "comfyui.log"
        try:
            self._close_comfyui_log_handle()
            self._comfyui_log_handle = open(log_file, "w", encoding="utf-8", errors="replace")
            # v5.5.6+: CPU thread saturation — see app.py _start_comfyui_process
            # for the rationale.  On 16/32-core CPU systems without a discrete GPU,
            # ComfyUI in --cpu mode otherwise only uses half the logical cores.
            env_for_popen = None
            if "--cpu" in flags:
                try:
                    logical = os.cpu_count() or 1
                except Exception:
                    logical = 1
                logical = max(1, int(logical))
                env_for_popen = {**os.environ}
                env_for_popen["OMP_NUM_THREADS"] = str(logical)
                env_for_popen["MKL_NUM_THREADS"] = str(logical)
                env_for_popen["NUMEXPR_MAX_THREADS"] = str(logical)
                env_for_popen["OPENBLAS_NUM_THREADS"] = str(logical)
                logger.info(
                    f"ComfyUI CPU thread saturation: OMP_NUM_THREADS={logical}",
                    category=logger.CATEGORY_COMFYUI,
                )
            self.comfyui_process = subprocess.Popen(
                cmd,
                stdout=self._comfyui_log_handle,
                stderr=subprocess.STDOUT,
                env=env_for_popen,
                **_subprocess_flags(),
            )
            self._comfyui_current_launch_flags = list(flags)
            logger.info(
                f"ComfyUI subprocess started (PID {self.comfyui_process.pid}); waiting for /system_stats...",
                category=logger.CATEGORY_COMFYUI,
            )
            logger.info(f"ComfyUI launch cmd: {cmd}", category=logger.CATEGORY_COMFYUI)
            self._close_comfyui_log_handle()
            return True
        except Exception as exc:
            reason = f"ComfyUI subprocess.Popen failed: {exc}"
            self._set_comfyui_start_failure_reason(reason)
            logger.error(reason, category=logger.CATEGORY_COMFYUI)
            self._comfyui_current_launch_flags = []
            self._close_comfyui_log_handle()
            return False

    def _stop_comfyui_for_restart(self, *, reason: str, kill_orphans: bool = False) -> None:
        proc = self.comfyui_process
        if proc and proc.poll() is None:
            logger.info(
                f"ComfyUI: terminating owned process PID {proc.pid} for {reason}",
                category=logger.CATEGORY_COMFYUI,
            )
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self.comfyui_process = None
        self._comfyui_current_launch_flags = []
        self._close_comfyui_log_handle()
        if kill_orphans:
            killed = self._kill_orphan_comfyui_processes()
            if killed:
                time.sleep(1)
        try:
            self.comfyui.reconnect()
        except Exception:
            pass

    def _kill_orphan_comfyui_processes(self) -> list[int]:
        try:
            import psutil
        except ImportError as exc:
            logger.warning(f"ComfyUI: orphan scan unavailable (psutil missing): {exc}")
            return []
        own_pid = (
            self.comfyui_process.pid
            if (self.comfyui_process and self.comfyui_process.poll() is None)
            else None
        )
        killed: list[int] = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if "ComfyUI" not in cmd_str or "main.py" not in cmd_str:
                    continue
                pid = int(proc.info["pid"])
                if pid == own_pid:
                    continue
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                logger.info(f"ComfyUI: killed orphan process PID {pid}", category=logger.CATEGORY_COMFYUI)
                killed.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as exc:
                logger.debug(f"ComfyUI: orphan process disappeared/denied: {exc}")
        return killed

    def _close_comfyui_log_handle(self) -> None:
        handle = self._comfyui_log_handle
        if handle:
            try:
                handle.close()
            except OSError:
                pass
        self._comfyui_log_handle = None

    @staticmethod
    def _tail_comfyui_log(log_file: Path, lines: int = 40) -> str:
        try:
            if not log_file.exists():
                return ""
            tail = deque(maxlen=max(1, int(lines)))
            with open(log_file, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    tail.append(line.rstrip())
            return "\n".join(tail)
        except OSError:
            return ""

    @staticmethod
    def _comfyui_startup_signal_line(log_tail: str) -> str:
        if not log_tail:
            return "no log output"
        signals = (
            "traceback",
            "error",
            "exception",
            "failed",
            "no module named",
            "dll load failed",
            "out of memory",
            "cuda",
            "torch",
        )
        for line in reversed(log_tail.splitlines()):
            text = line.strip()
            if text and any(signal in text.lower() for signal in signals):
                return text[:220]
        for line in reversed(log_tail.splitlines()):
            text = line.strip()
            if text:
                return text[:220]
        return "no non-empty log output"

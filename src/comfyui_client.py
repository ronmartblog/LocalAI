# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
ComfyUI client for LocalAI Studio.

Wraps the ComfyUI REST API for text-to-image generation.
ComfyUI must be running separately — see https://github.com/comfyanonymous/ComfyUI
"""

import base64
import json
import os
import random
import socket
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests

from src import logger

COMFYUI_DEFAULT_HOST = "http://127.0.0.1:8188"


# ── reference-image helpers ─────────────────────────────────────────────────
#
# SDXL bucket families that produce well-conditioned generations. The native
# resolution is 1024 (so total pixel count stays close to 1024²) — this is
# the set the SDXL refiner was trained on and is what most popular fine-tunes
# expect. SD1.5 buckets are the same family halved (native 512).

_SDXL_ASPECT_BUCKETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1152,  896), (896, 1152),
    (1216,  832), (832, 1216),
    (1344,  768), (768, 1344),
    (1536,  640), (640, 1536),
)


def snap_to_aspect_bucket(
    ref_width: int,
    ref_height: int,
    *,
    native: int = 1024,
) -> tuple[int, int]:
    """Snap (ref_width, ref_height) to the closest SDXL/SD1.5 bucket by aspect.

    Returns a (width, height) tuple from the standard bucket family scaled by
    ``native / 1024``. Always returns dimensions >= 256 so SD1.5 callers still
    get reasonable sizes when scaling down (256² is the lower practical bound).

    Used by the reference-image (img2img) generator to match the reference's
    aspect — avoids the "ImageScale crop center cuts off the head" failure
    mode when the reference is portrait but the user's chosen size is square.
    """
    try:
        rw = max(1, int(ref_width))
        rh = max(1, int(ref_height))
    except (TypeError, ValueError):
        return (native, native)
    ref_aspect = rw / rh
    best = min(
        _SDXL_ASPECT_BUCKETS,
        key=lambda wh: abs((wh[0] / wh[1]) - ref_aspect),
    )
    scale = native / 1024.0
    w = max(256, int(round(best[0] * scale / 8.0)) * 8)
    h = max(256, int(round(best[1] * scale / 8.0)) * 8)
    return (w, h)


class ComfyUIError(Exception):
    pass


class ComfyUIClient:
    def __init__(self, host: str = COMFYUI_DEFAULT_HOST):
        self.host = host.rstrip("/")
        self._session = requests.Session()
        # Unique ID per LocalAI session — prevents ComfyUI from reusing any
        # client-keyed state (e.g. cached node outputs) across app restarts.
        self._client_id = f"localai_{uuid.uuid4().hex[:12]}"

    def reconnect(self):
        """Drop all pooled connections so the next request starts fresh."""
        self._session.close()
        self._session = requests.Session()

    def is_running(self) -> bool:
        try:
            r = self._session.get(f"{self.host}/system_stats", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_system_stats(self) -> dict:
        try:
            r = self._session.get(f"{self.host}/system_stats", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise ComfyUIError(f"Could not get system stats: {e}") from e

    def get_model_list(self) -> list[str]:
        """Return list of checkpoint model filenames available in ComfyUI.

        Queries both the standard CheckpointLoaderSimple and the GGUF
        UnetLoaderGGUF node (if the ComfyUI-GGUF custom node is installed).
        """
        models = set()

        # Standard checkpoint loader (safetensors, ckpt, pt)
        try:
            r = self._session.get(
                f"{self.host}/object_info/CheckpointLoaderSimple", timeout=5
            )
            r.raise_for_status()
            data = r.json()
            ckpt_info = (
                data.get("CheckpointLoaderSimple", {})
                .get("input", {})
                .get("required", {})
                .get("ckpt_name", [[]])
            )
            if ckpt_info and ckpt_info[0]:
                models.update(ckpt_info[0])
        except Exception:
            pass

        # GGUF loader (if ComfyUI-GGUF custom node is installed)
        try:
            r = self._session.get(
                f"{self.host}/object_info/UnetLoaderGGUF", timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                gguf_info = (
                    data.get("UnetLoaderGGUF", {})
                    .get("input", {})
                    .get("required", {})
                    .get("unet_name", [[]])
                )
                if gguf_info and gguf_info[0]:
                    models.update(gguf_info[0])
        except Exception:
            pass

        # UNETLoader (native ComfyUI — used for z_image_turbo and other UNet-only models)
        try:
            r = self._session.get(
                f"{self.host}/object_info/UNETLoader", timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                unet_info = (
                    data.get("UNETLoader", {})
                    .get("input", {})
                    .get("required", {})
                    .get("unet_name", [[]])
                )
                if unet_info and unet_info[0]:
                    models.update(unet_info[0])
        except Exception:
            pass

        return sorted(models)

    def is_model_available(self, model_filename: str,
                           comfyui_path: Optional[str] = None) -> bool:
        """Return True if *model_filename* is available.

        Checks ComfyUI API first, then falls back to filesystem check.
        """
        # Check via API
        try:
            models = self.get_model_list()
            if any(
                m == model_filename or m.endswith(f"/{model_filename}")
                for m in models
            ):
                return True
        except ComfyUIError:
            pass

        # Fallback: check filesystem directly
        if comfyui_path:
            from pathlib import Path
            # Check both checkpoints/ and diffusion_models/ (GGUF models live there)
            for subdir in ("checkpoints", "diffusion_models"):
                candidate = Path(comfyui_path) / "models" / subdir / model_filename
                if candidate.exists():
                    return True

        return False

    def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to ComfyUI. Returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self._client_id}
        try:
            r = self._session.post(f"{self.host}/prompt", json=payload, timeout=10)
            if r.status_code >= 400:
                detail = ""
                try:
                    data = r.json()
                except Exception:
                    data = {}
                if isinstance(data, dict):
                    node_errors = data.get("node_errors", {})
                    if node_errors:
                        detail = "; ".join(
                            f"{nid}: {err}" for nid, err in node_errors.items()
                        )
                    elif "error" in data:
                        detail = str(data.get("error") or "").strip()
                    elif "message" in data:
                        detail = str(data.get("message") or "").strip()
                if not detail:
                    raw_text = (r.text or "").strip()
                    if raw_text:
                        detail = raw_text[:500]
                status = f"{r.status_code} {r.reason}".strip()
                if detail:
                    raise ComfyUIError(f"Could not queue prompt ({status}): {detail}")
                raise ComfyUIError(f"Could not queue prompt ({status})")
            data = r.json()
            if "error" in data:
                raise ComfyUIError(f"Workflow error: {data['error']}")
            node_errors = data.get("node_errors", {})
            if node_errors:
                details = "; ".join(
                    f"{nid}: {e}" for nid, e in node_errors.items()
                )
                raise ComfyUIError(f"Workflow validation failed: {details}")
            return data["prompt_id"]
        except ComfyUIError:
            raise
        except Exception as e:
            raise ComfyUIError(f"Could not queue prompt: {e}") from e

    def get_history(self, prompt_id: str) -> Optional[dict]:
        """Return history entry for prompt_id once complete, or None if still running.
        Raises ComfyUIError immediately if ComfyUI reports an execution error."""
        try:
            r = self._session.get(f"{self.host}/history/{prompt_id}", timeout=5)
            r.raise_for_status()
            data = r.json()
            entry = data.get(prompt_id)
            if not entry:
                return None
            status = entry.get("status", {})
            status_str = status.get("status_str", "")
            if status_str == "error":
                # Extract the human-readable exception_message (not the full traceback dict)
                messages = status.get("messages", [])
                err_msg = ""
                node_type = ""
                for msg in messages:
                    if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
                        detail = msg[1]
                        if isinstance(detail, dict):
                            err_msg  = detail.get("exception_message", "") or str(detail)
                            node_type = detail.get("node_type", "")
                        else:
                            err_msg = str(detail)
                        break
                # Friendly hint for known mis-configuration
                hint = ""
                if "channels" in err_msg and "expected" in err_msg:
                    hint = f" (node: {node_type}) — wrong VAE or latent format for this model"
                raise ComfyUIError(f"ComfyUI execution error: {err_msg or 'unknown error'}{hint}")
            if status.get("completed") or status_str == "success":
                return entry
            return None
        except ComfyUIError:
            raise
        except Exception as e:
            raise ComfyUIError(f"Could not get history: {e}") from e

    def get_image_bytes(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        """Download image bytes from ComfyUI's /view endpoint."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        try:
            r = self._session.get(f"{self.host}/view", params=params, timeout=30)
            r.raise_for_status()
            return r.content
        except Exception as e:
            raise ComfyUIError(f"Could not fetch image: {e}") from e

    def upload_input_image(self, image_path: str) -> str:
        """Upload a local image to ComfyUI's input folder and return its ComfyUI name."""
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise ComfyUIError(f"Reference image not found: {image_path}")
        comfy_name = f"localai_ref_{uuid.uuid4().hex[:10]}{path.suffix.lower() or '.png'}"
        try:
            with path.open("rb") as fh:
                files = {"image": (comfy_name, fh, "application/octet-stream")}
                data = {"type": "input", "overwrite": "true"}
                r = self._session.post(
                    f"{self.host}/upload/image",
                    files=files,
                    data=data,
                    timeout=60,
                )
            r.raise_for_status()
            payload = r.json()
            return payload.get("name") or comfy_name
        except Exception as e:
            raise ComfyUIError(f"Could not upload reference image: {e}") from e

    def interrupt(self):
        """Interrupt the currently running generation."""
        try:
            self._session.post(f"{self.host}/interrupt", timeout=5)
        except Exception:
            pass

    def clear_queue(self):
        """Clear ComfyUI's pending execution queue.

        Removes any prompts queued by previous LocalAI sessions that haven't
        run yet — prevents stale generations from a prior session contaminating
        a new one (e.g. a queued coffee generation running after a puppy prompt).
        """
        try:
            self._session.post(f"{self.host}/queue", json={"clear": True}, timeout=5)
        except Exception:
            pass

    def free_vram(self) -> bool:
        """Tell ComfyUI to unload all models and free VRAM. Returns True on success."""
        try:
            r = self._session.post(
                f"{self.host}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    def has_gguf_node(self) -> bool:
        """Return True if the ComfyUI-GGUF custom node is loaded."""
        try:
            r = self._session.get(
                f"{self.host}/object_info/UnetLoaderGGUF", timeout=5
            )
            return r.status_code == 200
        except Exception:
            return False

    def has_chroma_node(self) -> bool:
        """Return True if the ChromaLatentToImage custom node is loaded."""
        try:
            r = self._session.get(
                f"{self.host}/object_info/ChromaLatentToImage", timeout=5
            )
            return r.status_code == 200
        except Exception:
            return False

    # ── WebSocket progress monitoring ─────────────────────────────────────────

    def _ws_connect(self) -> socket.socket:
        """Open a raw WebSocket connection to ComfyUI's /ws endpoint."""
        from urllib.parse import urlparse
        parsed = urlparse(self.host)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8188

        key = base64.b64encode(os.urandom(16)).decode()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))

        upgrade = (
            f"GET /ws?clientId={self._client_id} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        sock.sendall(upgrade.encode())

        # Read HTTP 101 Switching Protocols
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket upgrade got empty response")
            buf += chunk
        if b" 101 " not in buf.split(b"\r\n")[0]:
            raise ConnectionError(f"WebSocket upgrade failed: {buf[:120]}")

        sock.settimeout(2)
        return sock

    def _ws_read_frame(self, sock: socket.socket):
        """Read one WebSocket frame.  Returns parsed JSON dict, empty dict (skip), or None (closed)."""
        def recv_exact(n: int) -> bytes:
            buf = b""
            while len(buf) < n:
                chunk = sock.recv(n - len(buf))
                if not chunk:
                    return b""
                buf += chunk
            return buf

        try:
            header = recv_exact(2)
            if len(header) < 2:
                return None
            opcode = header[0] & 0x0F
            payload_len = header[1] & 0x7F
            is_masked = bool(header[1] & 0x80)

            if payload_len == 126:
                ext = recv_exact(2)
                payload_len = int.from_bytes(ext, "big")
            elif payload_len == 127:
                ext = recv_exact(8)
                payload_len = int.from_bytes(ext, "big")

            mask = recv_exact(4) if is_masked else None

            # Skip large binary frames (preview images can be MBs — don't buffer them)
            if opcode == 0x2 or payload_len > 65536:
                # Drain the socket without buffering
                drained = 0
                while drained < payload_len:
                    chunk = sock.recv(min(4096, payload_len - drained))
                    if not chunk:
                        return None
                    drained += len(chunk)
                return {}

            payload = recv_exact(payload_len)
            if is_masked and mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x8:   # close frame
                return None
            if opcode == 0x9:   # ping — ignore
                return {}
            if opcode != 0x1:   # not text
                return {}

            return json.loads(payload.decode("utf-8"))

        except socket.timeout:
            return {}   # normal — no data yet
        except Exception:
            return None  # connection error

    # Maps ComfyUI node class_type → human-readable phase label
    _NODE_PHASE: dict = {
        "CheckpointLoaderSimple":  "Loading model …",
        "UNETLoader":              "Loading model …",
        "DualCLIPLoader":          "Loading text encoders …",
        "CLIPLoader":              "Loading text encoder …",
        "CLIPTextEncode":          "Encoding prompt …",
        "EmptySD3LatentImage":     "Preparing latent …",
        "EmptyLatentImage":        "Preparing latent …",
        "ModelSamplingFlux":       "Configuring sampler …",
        "ModelSamplingAuraFlow":   "Configuring sampler …",
        "FluxGuidance":            "Configuring guidance …",
        "VAELoader":               "Loading VAE …",
        "VAEDecode":               "Decoding image (VAE) …",
        "VAEDecodeTiled":          "Decoding image (VAE tiled) …",
        "SaveImage":               "Saving image …",
        "PreviewImage":            "Saving image …",
    }

    def _ws_progress_listener(self, prompt_id: str, state: dict,
                               stop_event: Optional[threading.Event],
                               workflow: Optional[dict] = None) -> None:
        """Background thread: listen on ComfyUI WebSocket and update state with step progress.

        state keys updated:
            value  — current step (int)
            max    — total steps (int)
            phase  — human-readable current phase (str, e.g. "Loading model …")
            done   — True when execution complete or connection closed
            error  — set to a short string when the listener aborts due to a stuck pipeline

        v5: adaptive stuck-detector. ComfyUI normally emits a steady stream of
        ``progress`` / ``executing`` events during inference. If no event is
        received for ``WS_STUCK_WARN_S`` we log a warning; at ``WS_STUCK_ABORT_S``
        we set ``state["error"]`` and exit so the caller can fail gracefully
        instead of hanging the UI forever.
        """
        WS_STUCK_WARN_S = 60.0
        # Flux GGUF/Chroma first runs can spend several minutes compiling or
        # dequantizing before ComfyUI emits another websocket event.
        WS_STUCK_ABORT_S = 900.0

        # Build node_id → phase label map from the workflow
        node_labels: dict = {}
        if workflow:
            for nid, node in workflow.items():
                cls = node.get("class_type", "")
                label = self._NODE_PHASE.get(cls)
                if label:
                    node_labels[str(nid)] = label

        sock = None
        last_msg_at = time.time()
        warned = False
        try:
            sock = self._ws_connect()
            while not (stop_event and stop_event.is_set()):
                msg = self._ws_read_frame(sock)
                if msg is None:
                    break   # connection closed
                if not msg:
                    # Timeout — check stuck detector then continue
                    silent_s = time.time() - last_msg_at
                    if silent_s > WS_STUCK_ABORT_S:
                        state["error"] = (
                            f"ComfyUI sent no progress events for "
                            f"{silent_s:.0f} s — aborting."
                        )
                        break
                    if silent_s > WS_STUCK_WARN_S and not warned:
                        warned = True
                        # Stash a soft warning the caller can surface
                        state["phase"] = (
                            f"Still working (no events for {silent_s:.0f} s) …"
                        )
                    continue

                last_msg_at = time.time()
                warned = False

                msg_type = msg.get("type")
                data = msg.get("data") or {}

                if msg_type == "progress" and isinstance(data, dict):
                    if data.get("prompt_id") == prompt_id:
                        state["value"] = int(data.get("value") or 0)
                        state["max"]   = int(data.get("max")   or 0)

                elif msg_type == "executing" and isinstance(data, dict):
                    if data.get("prompt_id") != prompt_id:
                        continue
                    node_id = data.get("node")
                    if node_id is None:
                        # End of execution
                        state["done"] = True
                        break
                    label = node_labels.get(str(node_id))
                    if label:
                        state["phase"] = label

        except Exception:
            pass
        finally:
            state["done"] = True
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def generate_image(
        self,
        model_filename: str,
        positive_prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        reference_image_path: Optional[str] = None,
        denoise: float = 0.75,
        progress_cb: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> bytes:
        """
        High-level helper: build a txt2img/img2img workflow, queue it, poll until done,
        and return raw PNG bytes.
        """
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        is_gguf = model_filename.lower().endswith('.gguf')
        # Flux-architecture models: native Flux + fine-tunes (Juggernaut Flux,
        # CyberRealistic Flux, Realism Flux) + Chroma-family pixel-space models.
        _lower = model_filename.lower()
        is_flux = 'flux' in _lower and 'chroma' not in _lower
        is_z_image = 'z_image' in _lower
        is_chroma = 'chroma' in _lower
        is_sdxl_fast = (
            'lightning' in _lower
            or 'sdxl_turbo' in _lower
            or 'sd_xl_turbo' in _lower
            or 'sd-turbo' in _lower
        )

        # v5: validation — refuse to silently feed an unsupported architecture
        # into the SD1.5/SDXL fallback builder. These families have specific
        # workflow requirements (custom samplers, encoders, VAEs) and the SDXL
        # fallback produces garbage or hard errors when fed one of them.
        _unsupported = {
            'sana':       'Sana models (Nvidia) need a dedicated workflow with the SanaSampler node.',
            'pixart':     'PixArt models need a dedicated workflow with PixArtCheckpointLoader + T5TextEncode.',
            'hunyuan':    'HunyuanDiT models need a dedicated workflow with HunyuanDiTCheckpointLoader.',
            'kolors':     'Kolors models need a dedicated workflow with the ChatGLM3 text encoder.',
            'sd3':        'SD3 / SD3.5 models need a dedicated workflow with TripleCLIPLoader.',
            'auraflow':   'AuraFlow models need a dedicated workflow with ModelSamplingAuraFlow.',
            'hidream':    'HiDream models need a dedicated workflow (not implemented yet).',
            'qwen-image': 'Qwen-Image models need a dedicated workflow (not implemented yet).',
            'wan-':       'Wan video models are not supported in the txt2img path.',
            'cogvideo':   'CogVideoX is a video model and is not supported in the txt2img path.',
        }
        if not (is_flux or is_z_image or is_chroma):
            for key, msg in _unsupported.items():
                if key in _lower:
                    raise ComfyUIError(
                        f"Model '{model_filename}' looks like an unsupported "
                        f"family ({key}). {msg}"
                    )

        reference_image_name = None
        if reference_image_path:
            denoise = max(0.05, min(1.0, float(denoise)))
            # Lightning/Turbo fast checkpoints fall apart with high denoise on
            # an img2img path — they only get 4–8 sampling steps, so anything
            # above ~0.85 leaves the image as noisy soup. Silently clamp and
            # tell the user via the progress callback so behavior is debuggable.
            if is_sdxl_fast and denoise > 0.85:
                if progress_cb:
                    progress_cb(
                        f"Fast checkpoint — clamping denoise {denoise:.2f} → 0.85 "
                        "(Lightning/Turbo gets noisy above this on img2img)"
                    )
                denoise = 0.85
            if progress_cb:
                progress_cb("Uploading reference image …")
            if is_flux or is_z_image or is_chroma:
                raise ComfyUIError(
                    "Reference image generation is currently supported only for "
                    "standard SD/SDXL checkpoint models. Flux, Z-Image, and Chroma "
                    "need dedicated image-to-image workflows."
                )
            reference_image_name = self.upload_input_image(reference_image_path)

        if reference_image_name:
            workflow = _build_img2img_checkpoint_workflow(
                model_filename, positive_prompt, negative_prompt,
                width, height, steps, 1.0 if is_sdxl_fast else cfg_scale, seed, reference_image_name, denoise,
                sampler_name=sampler_name, scheduler=scheduler,
            )
        elif is_gguf and is_flux:
            if not self.has_gguf_node():
                raise ComfyUIError(
                    "GGUF model selected but ComfyUI-GGUF custom node is not installed. "
                    "Download the model from the Models page to auto-install it."
                )
            workflow = _build_gguf_flux_workflow(
                model_filename, positive_prompt,
                width, height, steps, seed,
                sampler_name=sampler_name, scheduler=scheduler,
            )
        elif is_z_image:
            # Z-Image family (z_image_bf16, z_image_turbo_bf16):
            # UNETLoader + Qwen text encoder + Flux VAE
            workflow = _build_z_image_workflow(
                model_filename, positive_prompt,
                width, height, steps, seed,
            )
        elif is_chroma:
            # Chroma x0 outputs pixel-space [B,3,H,W] from KSampler.
            # Use dedicated workflow with ChromaLatentToImage instead of VAEDecode.
            if not self.has_chroma_node():
                raise ComfyUIError(
                    "This Chroma-family model requires the ChromaLatentToImage custom node. "
                    "Restart LocalAI — the node is written automatically at startup."
                )
            workflow = _build_chroma_workflow(
                model_filename, positive_prompt,
                width, height, steps, seed,
                cfg_scale=cfg_scale, negative_prompt=negative_prompt,
                sampler_name=sampler_name, scheduler=scheduler,
            )
        elif is_flux:
            # Flux safetensors (e.g. flux1-dev.safetensors from Comfy-Org)
            workflow = _build_flux_safetensors_workflow(
                model_filename, positive_prompt,
                width, height, steps, seed,
                sampler_name=sampler_name, scheduler=scheduler,
            )
        elif is_sdxl_fast:
            workflow = _build_sdxl_fast_checkpoint_workflow(
                model_filename, positive_prompt, negative_prompt,
                width, height, steps, seed,
                sampler_name=sampler_name, scheduler=scheduler,
            )
        else:
            workflow = _build_txt2img_workflow(
                model_filename, positive_prompt, negative_prompt,
                width, height, steps, cfg_scale, seed,
                sampler_name=sampler_name, scheduler=scheduler,
            )

        if progress_cb:
            progress_cb("Sending workflow to ComfyUI …")

        prompt_id = self.queue_prompt(workflow)

        if progress_cb:
            progress_cb("Queued — loading model …")

        # Start WebSocket listener for real-time step progress (no extra dependencies needed)
        _ws_state: dict = {"value": 0, "max": 0, "phase": "", "done": False}
        _ws_thread = threading.Thread(
            target=self._ws_progress_listener,
            args=(prompt_id, _ws_state, stop_event, workflow),
            daemon=True,
        )
        _ws_thread.start()

        start = time.time()
        poll_errors = 0
        generation_timeout_s = 1200 if (is_gguf or is_chroma or is_flux or is_z_image) else 600
        deadline = start + generation_timeout_s
        next_history_poll = 0.0
        history_poll_count = 0
        while time.time() < deadline:
            if stop_event and stop_event.is_set():
                self.interrupt()
                raise ComfyUIError("Generation cancelled.")

            if _ws_state.get("error"):
                raise ComfyUIError(str(_ws_state["error"]))

            now = time.time()
            result = None
            if now >= next_history_poll or _ws_state.get("done"):
                try:
                    result = self.get_history(prompt_id)
                    poll_errors = 0
                    history_poll_count += 1
                    next_history_poll = now + (0.5 if _ws_state.get("done") else 2.0)
                except ComfyUIError as e:
                    err_str = str(e)
                    if "execution error" in err_str.lower():
                        raise
                    poll_errors += 1
                    if poll_errors >= 5:
                        raise ComfyUIError(
                            f"Lost connection to ComfyUI after {poll_errors} retries: {e}"
                        )
                    self.reconnect()
                    time.sleep(2)
                    continue

                if result:
                    outputs = result.get("outputs", {})
                    for _node_id, node_output in outputs.items():
                        images = node_output.get("images", [])
                        if images:
                            img_info = images[0]
                            if progress_cb:
                                progress_cb("Retrieving image …")
                            logger.debug(f"ComfyUI history polls for {prompt_id}: {history_poll_count}")
                            return self.get_image_bytes(
                                img_info["filename"],
                                img_info.get("subfolder", ""),
                                img_info.get("type", "output"),
                            )
                    raise ComfyUIError("Generation finished but no output image found.")

            elapsed = int(time.time() - start)
            if progress_cb:
                step  = _ws_state["value"]
                total = _ws_state["max"]
                phase = _ws_state["phase"]
                if step > 0 and total > 0:
                    pct = int(100 * step / total)
                    progress_cb(f"Sampling step {step}/{total} ({pct}%) — {elapsed}s")
                elif phase:
                    progress_cb(f"{phase} {elapsed}s")
                elif elapsed < 3:
                    progress_cb("Queued — loading model …")
                else:
                    progress_cb(f"Loading model … {elapsed}s")
            time.sleep(1)

        raise ComfyUIError(
            f"Timed out waiting for image generation ({generation_timeout_s // 60} min limit)."
        )


# ── Workflow builder ───────────────────────────────────────────────────────────

def _build_txt2img_workflow(
    model_filename: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    sampler_name: str = "euler",
    scheduler: str = "normal",
) -> dict:
    """Build a standard ComfyUI txt2img workflow dict."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model_filename},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "LocalAI", "images": ["6", 0]},
        },
    }


def _build_img2img_checkpoint_workflow(
    model_filename: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    reference_image_name: str,
    denoise: float = 0.75,
    sampler_name: str = "euler",
    scheduler: str = "normal",
) -> dict:
    """Build a standard checkpoint img2img workflow using VAEEncode + KSampler."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model_filename},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
        },
        "5": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["4", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
            },
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["1", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "LocalAI_img2img", "images": ["8", 0]},
        },
    }


def _build_sdxl_fast_checkpoint_workflow(
    model_filename: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    sampler_name: str = "euler",
    scheduler: str = "sgm_uniform",
) -> dict:
    """Build SDXL Turbo/Lightning checkpoint workflow with backend CFG lock."""
    return _build_txt2img_workflow(
        model_filename,
        positive_prompt,
        negative_prompt,
        width,
        height,
        steps,
        1.0,
        seed,
        sampler_name=sampler_name,
        scheduler=scheduler,
    )


def _build_gguf_flux_workflow(
    model_filename: str,
    positive_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    sampler_name: str = "euler",
    scheduler: str = "simple",
) -> dict:
    """Build a ComfyUI workflow for GGUF Flux models.

    Uses UnetLoaderGGUF + DualCLIPLoader + the Flux-appropriate
    KSampler settings (cfg=1, euler/simple scheduler).  Flux models
    don't use a negative prompt — the guidance is embedded in the model.
    """
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": model_filename},
        },
        "2": {
            "class_type": "DualCLIPLoaderGGUF",
            "inputs": {
                "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["7", 0]},
        },
        "7": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "LocalAI", "images": ["6", 0]},
        },
    }


def _build_flux_safetensors_workflow(
    model_filename: str,
    positive_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    sampler_name: str = "euler",
    scheduler: str = "simple",
) -> dict:
    """Build a ComfyUI workflow for Flux safetensors checkpoints (e.g. Comfy-Org flux1-dev).

    The Comfy-Org flux1-dev.safetensors is UNET-only (no CLIP, no VAE).
    Uses CheckpointLoaderSimple for the model, DualCLIPLoader for external
    CLIP encoders, and VAELoader for the Flux VAE — same support files as
    the GGUF workflow.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model_filename},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["6", 0]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "LocalAI", "images": ["7", 0]},
        },
    }


def _build_flux_unet_workflow(
    model_filename: str,
    positive_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    sampler_name: str = "euler",
    scheduler: str = "simple",
) -> dict:
    """Build a ComfyUI workflow for UNet-only Flux safetensors.

    These models live in diffusion_models/ and use UNETLoader + DualCLIPLoader
    (not CheckpointLoaderSimple).  Same support files as other Flux models:
    t5xxl_fp8_e4m3fn.safetensors, clip_l.safetensors, ae.safetensors.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_filename, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["6", 0]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "LocalAI", "images": ["7", 0]},
        },
    }


def _build_chroma_workflow(
    model_filename: str,
    positive_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    cfg_scale: float = 1.0,
    negative_prompt: str = "",
    sampler_name: str = "euler",
    scheduler: str = "simple",
) -> dict:
    """Build a ComfyUI workflow for Chroma x0 pixel-space models.

    Chroma x0 is a pixel-space prediction model: KSampler outputs
    [B, 3, H, W] normalised RGB directly instead of 16-channel Flux latents.
    The standard Flux VAE (ae.safetensors) cannot decode this output.

    This workflow replaces VAELoader + VAEDecode with the ChromaLatentToImage
    custom node (written to custom_nodes/ComfyUI-LocalAI-Chroma/ by LocalAI
    at startup).  It maps the [-1, 1] pixel-space output to [0, 1] images.

    Text conditioning: Chroma ignores CLIP inputs but KSampler still requires
    positive/negative conditioning — we pass empty strings via DualCLIPLoader.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_filename, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {
            "class_type": "ChromaLatentToImage",
            "inputs": {"samples": ["6", 0]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "LocalAI", "images": ["7", 0]},
        },
    }


def _build_z_image_workflow(
    model_filename: str,
    positive_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> dict:
    """Build a ComfyUI workflow for Z-Image and Z-Image Turbo.

    Both models use UNETLoader + CLIPLoader(lumina2 type) + ModelSamplingAuraFlow
    + KSampler with res_multistep sampler.  No negative prompt.
    Text encoder: qwen_3_4b_fp8_mixed.safetensors in models/text_encoders/
    VAE: ae.safetensors (Flux 1 VAE) in models/vae/
    Z-Image Turbo: 8 steps.  Z-Image (base): 20 steps.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_filename, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_4b_fp8_mixed.safetensors",
                "type": "lumina2",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 1.73},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["6", 0],
                "positive": ["4", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "LocalAI", "images": ["8", 0]},
        },
    }

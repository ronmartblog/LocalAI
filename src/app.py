# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
LocalAI Studio — main application window.
Built with customtkinter for a modern look.
"""

import base64
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import queue
import re
import sys
import threading
import time
import subprocess
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from tkinter import TclError, filedialog, messagebox

# Internal modules
from src import a11y, catalog, config, logger, model_demos, phase1_adapters, system_info
from src import hf_compat

# Install keyboard-accessibility patches on CTk widgets *before* any widget
# is constructed. This is idempotent and additive — see ``src/a11y.py``.
a11y.install()
from src.hf_model_resolver import (
    InvalidHFUrl,
    ParsedTarget,
    parse_url as parse_hf_url,
    slug_to_model_id,
)
from src.ollama_client import OllamaClient, OllamaError, ollama_tag_is_local
from src import constrained_env
from src.onnx_client import (
    ONNX_AVAILABLE, DIRECTML_AVAILABLE, COREML_AVAILABLE, OPENVINO_AVAILABLE, HF_AVAILABLE,
    GENAI_AVAILABLE, OnnxModelSession, OnnxGenAISession, OnnxError,
    download_onnx_model, has_genai_config,
)
from src.openvino_client import (
    OV_GENAI_AVAILABLE,
    OVModelSession, OVError, available_ov_devices, download_ov_model, pick_ov_device,
)
from src.comfyui_client import ComfyUIClient, ComfyUIError
from src import content_filter
from src.batch_runner import BatchRunner, DEFAULT_PROMPT
from src.gpu_detect import GPUInfo, detect_gpu, detect_gpu_cached, get_pytorch_device_info, is_snapdragon_arm64
from src.batch_report import BatchReport, find_latest_report_json
from src import resource_manager
from src import migration as _migration


def _palette_tuple(role: str) -> tuple[str, str]:
    palettes = config.DEFAULT_THEME_PALETTES
    return (palettes["light"][role], palettes["dark"][role])


TEXT_PRIMARY = _palette_tuple("text_primary")
TEXT_SECONDARY = _palette_tuple("text_secondary")
TEXT_MUTED = _palette_tuple("text_muted")
TEXT_DISABLED = _palette_tuple("text_disabled")
LINK_TEXT = _palette_tuple("link_text")
INFO_TEXT = _palette_tuple("info_text")
WARN_TEXT = _palette_tuple("warn_text")
SUCCESS_TEXT = _palette_tuple("success_text")
ERROR_TEXT = _palette_tuple("error_text")
BORDER_STRONG = _palette_tuple("border_strong")
SURFACE_CARD = _palette_tuple("surface_card")
SURFACE_INNER = _palette_tuple("surface_inner")
INPUT_SURFACE = _palette_tuple("input_surface")
BUTTON_SECONDARY = SURFACE_CARD
BUTTON_SECONDARY_HOVER = SURFACE_INNER


def _make_bordered_card(parent, *, fg_color, border_color=BORDER_STRONG,
                        border_width: int = 1, corner_radius: int = 10,
                        **outer_kwargs):
    """Create a reliably-bordered rounded card using the nested-frame pattern.

    Why: CustomTkinter's native ``border_width`` on ``CTkFrame`` draws the
    border as part of its rounded-rect Canvas path. At many widget sizes the
    straight edge segments fail to render (missing top, bottom, or side
    borders) while only the rounded corners survive. This has regressed
    repeatedly through v5.5.0 / v5.5.1 / v5.5.4 / v5.5.19 fix attempts.

    The fix: nest two ``CTkFrame``s. The OUTER frame is a solid filled rounded
    rect in ``border_color`` (no border math). The INNER frame is a solid
    filled rounded rect in ``fg_color``, inset by ``border_width`` pixels via
    pack padding. The outer color shows through the inset gap as a visible,
    artifact-free border on all four sides AND corners simultaneously.

    Returns the OUTER frame. The inner content frame is exposed as
    ``returned_card.content_frame``. Callers should grid/pack the returned
    outer; add children to ``returned_card.content_frame``.
    """
    outer = ctk.CTkFrame(
        parent,
        fg_color=border_color,
        corner_radius=corner_radius,
        border_width=0,
        **outer_kwargs,
    )
    inner = ctk.CTkFrame(
        outer,
        fg_color=fg_color,
        corner_radius=max(0, corner_radius - 1),
        border_width=0,
    )
    inner.pack(fill="both", expand=True, padx=border_width, pady=border_width)
    outer.content_frame = inner
    return outer


def _make_chip(parent, text, *, fg_color, text_color,
               border_color=BORDER_STRONG, font=None,
               corner_radius: int = 8, label_padx: int = 4,
               label_pady: int = 1):
    """Build a perf-badge chip with bulletproof border rendering.

    Uses the same nested-frame trick as ``_make_bordered_card`` so the chip's
    top/bottom border line never goes missing on CustomTkinter's Canvas at
    awkward widget sizes / DPI scales.

    Returns ``(outer_frame, label)``. Caller packs the outer; the label is
    already packed inside the inner content frame.
    """
    outer = ctk.CTkFrame(
        parent,
        fg_color=border_color,
        corner_radius=corner_radius,
        border_width=0,
    )
    inner = ctk.CTkFrame(
        outer,
        fg_color=fg_color,
        corner_radius=max(0, corner_radius - 1),
        border_width=0,
    )
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    label = ctk.CTkLabel(
        inner,
        text=text,
        font=font,
        fg_color="transparent",
        text_color=text_color,
    )
    label.pack(padx=label_padx, pady=label_pady)
    outer.content_frame = inner
    outer.label = label
    return outer, label


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
    "kornia": "kornia>=0.7.1",
    "OpenGL": "PyOpenGL",
    "glfw": "glfw",
    "dist:comfyui-frontend-package": "comfyui-frontend-package==1.39.19",
    "dist:comfyui-workflow-templates": "comfyui-workflow-templates==0.9.11",
    "dist:comfyui-embedded-docs": "comfyui-embedded-docs==0.4.3",
}


def _ollama_tag_is_local(tag: str, local_names: set[str]) -> bool:
    return ollama_tag_is_local(tag, local_names)


CHAT_TEXT_FONT_FAMILY = "Segoe UI"
CHAT_TEXT_FIXED_FONT_FAMILY = "Cascadia Mono"
CHAT_FONT_CHOICES = [
    CHAT_TEXT_FONT_FAMILY,
    "Aptos",
    "Calibri",
    CHAT_TEXT_FIXED_FONT_FAMILY,
    "Cascadia Code",
    "Consolas",
]
CHAT_TEXT_FONT_SIZE = 14


def _apply_theme_palette_globals(cfg: dict) -> None:
    """Apply explicit app palette overrides; System mode keeps built-in tuples."""
    global TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_DISABLED, LINK_TEXT
    global INFO_TEXT, WARN_TEXT, SUCCESS_TEXT, ERROR_TEXT, BORDER_STRONG
    global SURFACE_CARD, SURFACE_INNER, INPUT_SURFACE
    global BUTTON_SECONDARY, BUTTON_SECONDARY_HOVER

    mode = config.normalize_theme_mode(cfg.get("theme_mode"))
    palettes = config.normalize_theme_palettes(cfg.get("theme_palettes"))

    def pick(role: str) -> tuple[str, str]:
        if mode in ("light", "dark"):
            value = palettes[mode][role]
            return (value, value)
        return _palette_tuple(role)

    TEXT_PRIMARY = pick("text_primary")
    TEXT_SECONDARY = pick("text_secondary")
    TEXT_MUTED = pick("text_muted")
    TEXT_DISABLED = pick("text_disabled")
    LINK_TEXT = pick("link_text")
    INFO_TEXT = pick("info_text")
    WARN_TEXT = pick("warn_text")
    SUCCESS_TEXT = pick("success_text")
    ERROR_TEXT = pick("error_text")
    BORDER_STRONG = pick("border_strong")
    SURFACE_CARD = pick("surface_card")
    SURFACE_INNER = pick("surface_inner")
    INPUT_SURFACE = pick("input_surface")
    BUTTON_SECONDARY = SURFACE_CARD
    BUTTON_SECONDARY_HOVER = SURFACE_INNER

    app_class = globals().get("App")
    if app_class is not None:
        app_class._IG_CARD_FG = SURFACE_CARD
        app_class._IG_INNER_FG = SURFACE_INNER
        app_class._IG_BORDER = BORDER_STRONG
        app_class._IG_DISABLED_TEXT = TEXT_DISABLED
        app_class._IG_MUTED_FG = TEXT_MUTED
        app_class._IG_SUBHEAD_FG = TEXT_SECONDARY

# Optional: Pillow for image display
try:
    from PIL import Image as _PIL_Image, ImageTk as _PIL_ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

APP_TITLE = "LocalAI Studio"
APP_VERSION = "2026.06.02.0"

CHAT_RESPONSE_TOKEN_MAX = 131072
CHAT_RESPONSE_TOKEN_ONNX_FALLBACK = 4096
_TIMING_FILE      = Path(__file__).parent.parent / "timing_data.json"
_GEN_TIMEOUT_S    = 600   # 10-minute generation hard limit
_TIMING_MAX_SAMPLES = 8   # per-model history depth

# NOTE: utility/Toolbox models (phase1_adapter=True) are intentionally
# excluded from benchmark runs entirely. They are exercised through the
# Toolbox tab. Do not re-add any of these ids here:
#   all-minilm, florence-2-base, phi-4-multimodal, speecht5-tts,
#   table-transformer, trocr-base-printed, trocr-large-printed,
#   whisper-large-v3-turbo, whisper-v3-turbo-gpu.
#
# v5.5.12 SKU decoupling (Ron, 2026-05-28): per-SKU Quick/Extended
# default-tick model sets used to live in two hardcoded dicts in this
# file (``_BENCH_QUICK_DEFAULT_MODEL_IDS_BY_PROFILE`` /
# ``_BENCH_EXTENDED_DEFAULT_MODEL_IDS_BY_PROFILE``) keyed by SKU
# display name. That coupled app behavior to the SKU catalog —
# adding/renaming a SKU required source edits. The dicts are GONE.
# The same Quick/Extended sets now live in ``skus.json`` under
# ``bench_defaults`` + per-SKU ``bench_quick_models`` /
# ``bench_extended_models`` and are resolved by
# ``system_info.resolve_bench_models()`` at load time. The app reads
# the resolved sets via ``App._bench_default_models_for(profile, run_mode)``.
# DO NOT reintroduce SKU display names as hardcoded constants in any
# file under ``src/`` — see docs/architecture.md §5 (SKU profiles).

# v2026.06.01.8 Quick-mode fallback baseline (Ron, 2026-06-01):
# Used ONLY when ``_bench_default_selected_for_model`` is evaluated against
# the synthetic ``"This Device"`` profile — that profile is the sole entry
# in the SKU dropdown when ``skus.json`` is not present (or, with
# ``skus.json`` loaded, the catch-all entry the user can manually pick).
# Without these constants the Quick fallback path returned True for every
# model that passed the fit gate, which on a workstation-class GPU
# silently auto-ticked 40+ rows and turned Quick into the slow "everything"
# run. The two sets mirror the named baselines in ``skus.json``
# (``quick_chat_ultra_small`` / ``quick_image_smallest``) so behavior is
# identical whether or not ``skus.json`` ships with the install — when a
# SKU profile IS loaded and selected, the per-SKU ``bench_quick_models``
# resolved from ``skus.json`` takes precedence and these constants are
# never consulted. ``quick_chat_ultra_small`` is derived from the catalog
# at call time (every model with ``category == "Ultra Small"`` in
# ``src/catalog.py``) so adding or removing an Ultra Small chat model
# flows through automatically — matching the
# ``tests/test_skus.py::QuickChatUltraSmallBaselineContractTests``
# contract. The image-gen fallback is pinned to ``realistic-vision-v6``
# (4 GB VRAM, only smallest image-gen with no benchmark_skip_reason) to
# match ``test_quick_image_smallest_baseline_is_single_smallest_image_model``.
_BENCH_QUICK_FALLBACK_IMAGE_GEN_ID = "realistic-vision-v6"


def _bench_quick_fallback_model_ids(*, has_gpu: bool) -> set[str]:
    """Return the Quick-mode default-tick model ids for the synthetic
    ``"This Device"`` profile (no ``skus.json`` loaded, or the user
    manually selected ``"This Device"`` from a populated SKU dropdown).
    Always includes the catalog's ``Ultra Small`` chat models; adds the
    single ``quick_image_smallest`` image-gen model when ``has_gpu`` is
    True. Image-gen rows are gated by the same ``has_gpu`` rule that
    ``_bench_default_selected_for_model`` enforces earlier in the
    pipeline.
    """
    ids = {
        str(m.get("id"))
        for m in catalog.MODELS
        if m.get("category") == "Ultra Small"
    }
    if has_gpu:
        ids.add(_BENCH_QUICK_FALLBACK_IMAGE_GEN_ID)
    return ids


# v5.5.14 img2img prompt defaults (Ron, 2026-05-29): when the user checks
# "Use reference image for generation" on the Image Gen page we auto-load
# a sample positive + negative prompt so the user always has a known-good
# starting point. These constants MUST stay byte-identical to the
# "default selection" of the prompt builder in
# ``docs/image-gen-guide.html`` section 5.3 — the builder is what the user
# is sent to when they want a different man/woman/age/lighting/clothing
# combo. If you change one, change the other in the same commit.
# Test guard: ``tests/test_app_static_contracts.py:: \
# Img2ImgDefaultPromptContractTests`` pins both strings AND asserts they
# appear verbatim in the shipped doc.
IMG2IMG_DEFAULT_POSITIVE = (
    "professional photograph of a 30-year-old man wearing casual clothes, "
    "natural daylight, sharp focus, photorealistic, detailed skin texture, 8k"
)
IMG2IMG_DEFAULT_NEGATIVE = (
    "deformed, distorted, disfigured, extra fingers, bad anatomy, "
    "malformed limbs, mutated hands, blurry, low quality, jpeg artifacts, "
    "oversaturated, cartoon, anime, painting, sketch, 3d render, "
    "watermark, signature, text"
)


def _benchmark_skip_methods(model: dict) -> set[str]:
    raw = model.get("benchmark_skip_methods")
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


# ── Helpers ───────────────────────────────────────────────────────────────────

class HelpTooltip:
    def __init__(self, widget, text: str, title: str = "Help", delay_ms: int = 350):
        self.widget = widget
        self.text = text
        self.title = title
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip = None

        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<Button-1>", self._show_dialog, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip or not self.widget.winfo_exists():
            return
        self._tip = ctk.CTkToplevel(self.widget)
        self._tip.overrideredirect(True)
        self._tip.attributes("-topmost", True)
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._tip.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(self._tip, corner_radius=8, border_width=1)
        frame.pack()
        ctk.CTkLabel(
            frame,
            text=self.text,
            font=ctk.CTkFont(size=11),
            justify="left",
            wraplength=320,
            padx=10,
            pady=8,
        ).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def _show_dialog(self, _event=None):
        self._hide()
        messagebox.showinfo(self.title, self.text, parent=self.widget.winfo_toplevel())


def _fmt_duration(seconds: int) -> str:
    """Format seconds to a compact human-readable string: '42s', '1m 30s', '5m'."""
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if s else f"{m}m"


def _fmt_bytes(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    return f"{n / 1024:.1f} KB"


def _fmt_gb(gb: float) -> str:
    return f"{gb:.1f} GB"


class _ThreadWriter:
    """
    Wraps stdout to tee output to both the real stdout and a callback.

    The callback is invoked directly from the writing thread; pair it with a
    thread-safe sink (e.g. :meth:`App._enqueue_bench_log`) — *not* with a UI
    primitive — to avoid blocking the writer.
    """
    def __init__(self, original, callback):
        self._orig = original
        self._cb = callback
    def write(self, s):
        if s:
            try:
                self._orig.write(s)
            except Exception:
                pass
            try:
                self._cb(s)
            except Exception:
                pass
    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass


# ── Model Card widget ─────────────────────────────────────────────────────────

class ModelCard(ctk.CTkFrame):
    """Deprecated: superseded by ModelListRow; kept temporarily for compatibility."""

    def __init__(
        self,
        parent,
        model: dict,
        app: "App",
        local_names: set | None = None,
        comfyui_model_names: set | None = None,
        **kwargs,
    ):
        super().__init__(parent, corner_radius=8, **kwargs)
        self.model = model
        self.app = app

        self.grid_columnconfigure(0, weight=1)

        # Header row
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text=model["name"],
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        # Vendor badge
        ctk.CTkLabel(
            header, text=model["vendor"],
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            anchor="e",
        ).grid(row=0, column=1, sticky="e")

        # Description
        ctk.CTkLabel(
            self, text=model["description"],
            font=ctk.CTkFont(size=11),
            wraplength=380,
            justify="left",
            anchor="w",
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=2)

        # v5.3: Perf badges (image-gen only) — speed/quality/bucket/fit chips
        self._badge_frame = None
        if model.get("backend") == "comfyui" and model.get("perf_profile"):
            self._badge_frame = ctk.CTkFrame(self, fg_color="transparent")
            self._badge_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 2))
            self._render_perf_badges()

        # Specs row
        if model.get("backend") == "comfyui":
            ctx_part = f"  Default: {model.get('default_width', 512)}×{model.get('default_height', 512)}"
        else:
            ctx_part = f"  Context: {model.get('context_length', 0):,}"
        specs = (
            f"  {model.get('parameters', '—')}  |  "
            f"  {model['size_gb']} GB disk  |  "
            f"  Min RAM: {model['min_ram_gb']} GB  |  "
            f"  Min VRAM: {model['min_vram_gb']} GB  |  "
            f"{ctx_part}"
        )
        ctk.CTkLabel(
            self, text=specs,
            font=ctk.CTkFont(size=11),
            anchor="w",
            text_color=TEXT_MUTED,
        ).grid(row=3, column=0, sticky="ew", padx=10, pady=2)

        next_row = 4
        if getattr(self.app, "_optional_skus_enabled", False):
            from src import catalog as _catalog
            min_vram = model.get("min_vram_gb", 0)
            min_ram  = model.get("min_ram_gb", 0)
            is_image_gen = model.get("backend") == "comfyui"
            compat_skus = []
            for s in _catalog.OPTIONAL_SKU_ORDER:
                sku_vram = _catalog.OPTIONAL_SKU_VRAM.get(s, 0)
                if sku_vram == 0:
                    if is_image_gen:
                        continue
                    if _catalog.OPTIONAL_SKU_RAM.get(s, 0) >= min_ram:
                        compat_skus.append(s)
                else:
                    if sku_vram >= min_vram:
                        compat_skus.append(s)
            if compat_skus:
                min_sku = _catalog.get_optional_min_sku(model)
                sku_list_str = "  ".join(
                    f"[{s}]" if s == min_sku else s for s in compat_skus
                )
                prefix = self.app._sku_text("compatibility_prefix", "Compatible:")
                sku_text = f"  {prefix} {sku_list_str}"
                sku_color = INFO_TEXT
            else:
                sku_text = "  " + self.app._sku_text(
                    "requires_more_text",
                    "Requires more VRAM than any known SKU",
                )
                sku_color = TEXT_MUTED
            ctk.CTkLabel(
                self, text=sku_text,
                font=ctk.CTkFont(size=10),
                anchor="w",
                text_color=sku_color,
            ).grid(row=next_row, column=0, sticky="ew", padx=10, pady=(0, 2))
            next_row += 1

        # Tags
        tag_str = "  " + "  ·  ".join(f"#{t}" for t in model["tags"])
        ctk.CTkLabel(
            self, text=tag_str,
            font=ctk.CTkFont(size=10),
            anchor="w",
            text_color=LINK_TEXT,
        ).grid(row=next_row, column=0, sticky="ew", padx=10, pady=(0, 4))
        next_row += 1

        # Buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=next_row, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            btn_row, text="Checking…", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        )
        self.status_label.pack(side="left", padx=(0, 8))

        self._is_image_model = model.get("backend") == "comfyui"
        self._is_utility_demo_model = (
            not self._is_image_model
            and not catalog.is_chat_selectable_model(model)
            and bool(model.get("hf_repo") or model.get("onnx_repo"))
        )

        if self._is_image_model:
            self.btn_download = ctk.CTkButton(
                btn_row, text="Install", width=100,
                command=self._on_download,
                **self.app._outline_button_style(),
            )
            self.btn_download.pack(side="left", padx=2)

            self.btn_load = ctk.CTkButton(
                btn_row, text="Generate Images", width=130,
                command=self._on_load,
                **self.app._solid_button_style(self.app._IG_HERO, self.app._IG_HERO_HOVER),
            )
            self.btn_load.pack(side="left", padx=2)

            self.btn_delete = ctk.CTkButton(
                btn_row, text="Delete Local Model", width=140,
                command=self._on_delete,
                **self.app._solid_button_style(self.app._IG_DANGER, "#8a2424"),
            )
            self.btn_delete.pack(side="left", padx=2)
        else:
            self.btn_download = ctk.CTkButton(
                btn_row, text="Install", width=100,
                command=self._on_download,
                **self.app._outline_button_style(),
            )
            self.btn_download.pack(side="left", padx=2)

            self.btn_load = ctk.CTkButton(
                btn_row, text="Run Demo" if self._is_utility_demo_model else "Load & Chat", width=110,
                command=self._on_load,
                **self.app._solid_button_style(self.app._IG_SUCCESS, self.app._IG_SUCCESS_HOVER),
            )
            self.btn_load.pack(side="left", padx=2)

            self.btn_delete = ctk.CTkButton(
                btn_row, text="Delete Local Model", width=140,
                command=self._on_delete,
                **self.app._solid_button_style(self.app._IG_DANGER, "#8a2424"),
            )
            self.btn_delete.pack(side="left", padx=2)

        self.btn_ideas = ctk.CTkButton(
            btn_row, text="Prompt ideas", width=105,
            command=self._on_prompt_ideas,
            **self.app._outline_button_style(),
        )
        self.btn_ideas.pack(side="left", padx=2)

        demo = model_demos.get_model_demo(self.model)
        if demo.get("primary"):
            ctk.CTkLabel(
                self,
                text="  Best demo: " + demo["primary"],
                font=ctk.CTkFont(size=10),
                text_color=TEXT_MUTED,
                anchor="w",
                wraplength=920,
            ).grid(row=next_row + 1, column=0, sticky="ew", padx=10, pady=(0, 2))
            next_row += 1

        self.next_step_label = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=10), text_color=TEXT_MUTED,
            anchor="w",
        )
        self.next_step_label.grid(row=next_row + 1, column=0, sticky="ew", padx=10, pady=(0, 8))

        self.refresh_status(local_names=local_names, comfyui_model_names=comfyui_model_names)

    # ── Status ────────────────────────────────────────────────────────────────

    def _render_perf_badges(self) -> None:
        """v5.3: Lay out perf badges inside ``self._badge_frame``. Idempotent —
        clears any prior badge widgets first. Reads the active device's VRAM
        from the App so the fit pill reflects current SKU selection.
        """
        frame = self._badge_frame
        if frame is None:
            return
        for w in frame.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        try:
            vram = int(self.app._active_device_vram_gb())
        except Exception:
            vram = 0
        specs = self.app._build_perf_badge_specs(self.model, vram)
        for text, fg, txt, _kind in specs:
            # v5.5.4 a11y (P2-C) / v5.5.12 Mac crash fix: the hairline border
            # lifts every badge to >=3:1 non-text contrast against the
            # surrounding card (quality #2a2d3a 1.03:1, speed-slow #5a3a6b
            # 1.52:1, fit-exceeds #7a2a2a 1.48:1, fit-tight #8f6b1f 2.89:1,
            # speed-fast #1f6f8f 2.51:1 were under WCAG 1.4.11 in dark mode
            # without it). v5.5.19: switched to ``_make_chip`` so the border
            # renders reliably on all four sides at any DPI / widget size
            # (CTk native ``border_width=1`` dropped top/bottom edges).
            chip, _ = _make_chip(
                frame,
                " " + text + " ",
                fg_color=fg,
                text_color=txt,
                border_color=BORDER_STRONG,
                font=ctk.CTkFont(size=10, weight="bold"),
                corner_radius=8,
                label_padx=4,
                label_pady=0,
            )
            chip.pack(side="left", padx=(0, 4), pady=2)

    def refresh_perf_badges(self) -> None:
        """Called by the App when the active SKU changes so each card can
        re-render the fit pill against the new device."""
        if self._badge_frame is not None:
            self._render_perf_badges()

    def refresh_status(self, local_names: set | None = None, comfyui_model_names: set | None = None):
        if self._is_image_model:
            self._refresh_status_comfyui(comfyui_model_names=comfyui_model_names)
            return

        tag = self.model.get("ollama_tag", "")
        if not tag:
            if self._is_utility_demo_model:
                demo = model_demos.get_model_demo(self.model)
                self.status_label.configure(text="Demo ready", text_color=SUCCESS_TEXT)
                self.next_step_label.configure(text=f"Next: run the {demo['feature']} sample demo.")
                self.btn_download.configure(state="disabled")
                self.btn_load.configure(state="normal")
                if self.btn_delete:
                    self.btn_delete.configure(state="disabled")
                return
            self.status_label.configure(text="No Ollama tag", text_color=TEXT_MUTED)
            self.next_step_label.configure(text="This catalog entry cannot be installed through Ollama.")
            self.btn_download.configure(state="disabled")
            self.btn_load.configure(state="disabled")
            if self.btn_delete:
                self.btn_delete.configure(state="disabled")
            return

        if local_names is not None:
            is_local = _ollama_tag_is_local(tag, local_names)
        else:
            is_local = self.app.ollama.is_model_local(tag) if self.app.ollama_ok else False
        if is_local:
            self.status_label.configure(text="Installed", text_color=SUCCESS_TEXT)
            self.next_step_label.configure(text="Next: load it to start chatting.")
            self.btn_download.configure(state="disabled")
            self.btn_load.configure(state="normal")
            if self.btn_delete:
                self.btn_delete.configure(state="normal")
        else:
            self.status_label.configure(text="Not installed", text_color=TEXT_MUTED)
            self.next_step_label.configure(text="Next: install this model locally.")
            self.btn_download.configure(state="normal")
            self.btn_load.configure(state="disabled")
            if self.btn_delete:
                self.btn_delete.configure(state="disabled")

    def _refresh_status_comfyui(self, comfyui_model_names: set | None = None):
        model_filename = self.model.get("comfyui_model", "")
        comfyui_path = self.app._comfyui_installed_path()

        # Check filesystem regardless of ComfyUI status — model may already be downloaded
        file_exists = False
        if comfyui_path and model_filename:
            for subdir in ("checkpoints", "diffusion_models"):
                if (comfyui_path / "models" / subdir / model_filename).exists():
                    file_exists = True
                    break

        if not self.app.comfyui_ok:
            if file_exists:
                self.status_label.configure(text="Installed (ComfyUI offline)", text_color=WARN_TEXT)
                self.next_step_label.configure(text="Next: start ComfyUI to generate with this model.")
                self.btn_download.configure(state="disabled")
                self.btn_delete.configure(state="normal")
            else:
                self.status_label.configure(text="ComfyUI offline", text_color=TEXT_MUTED)
                self.next_step_label.configure(text="Next: start or install ComfyUI, then install the model.")
                self.btn_download.configure(state="normal")
                self.btn_delete.configure(state="disabled")
            self.btn_load.configure(state="disabled")
            return

        is_ready = file_exists
        if not is_ready and model_filename:
            if comfyui_model_names is not None:
                is_ready = model_filename in comfyui_model_names
            else:
                is_ready = self.app.comfyui.is_model_available(
                    model_filename,
                    comfyui_path=str(comfyui_path) if comfyui_path else None,
                )
        if is_ready:
            self.status_label.configure(text="Installed", text_color=SUCCESS_TEXT)
            self.next_step_label.configure(text="Next: open Image Gen and enter a prompt.")
            self.btn_download.configure(state="disabled")
            self.btn_load.configure(state="normal")
            self.btn_delete.configure(state="normal")
        else:
            self.status_label.configure(text="Not installed", text_color=WARN_TEXT)
            self.next_step_label.configure(text="Next: install this image model locally.")
            self.btn_download.configure(state="normal")
            self.btn_load.configure(state="disabled")
            self.btn_delete.configure(state="disabled")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_download(self):
        if self._is_image_model:
            self.app.download_comfyui_model(self.model)
        else:
            self.app.start_download(self.model)

    def _on_load(self):
        if self._is_image_model:
            self.app.open_image_gen_for_model(self.model)
        elif self._is_utility_demo_model:
            self.app.open_toolbox_for_model(self.model)
        else:
            self.app.load_model_for_chat(self.model)

    def _on_prompt_ideas(self):
        self.app.open_prompt_ideas_for_model(self.model)

    def _on_delete(self):
        ok = messagebox.askyesno(
            "Delete model",
            f"Delete '{self.model['name']}' from local storage?\n"
            "It will need to be re-downloaded to use again.",
            parent=self.app,
        )
        if not ok:
            return
        if self._is_image_model:
            self.app.delete_comfyui_model(self.model)
        else:
            self.app.delete_model(self.model)


class ModelListRow(ctk.CTkFrame):
    """Compact Models-page row. Actions live in the detail pane, not per row."""

    def __init__(
        self,
        parent,
        model: dict,
        app: "App",
        local_names: set | None = None,
        comfyui_model_names: set | None = None,
        **kwargs,
    ):
        super().__init__(
            parent,
            corner_radius=6,
            fg_color=INPUT_SURFACE,
            border_width=1,
            border_color=BORDER_STRONG,
            height=42,
            **kwargs,
        )
        self.model = model
        self.app = app
        self._selected = False
        self._has_focus = False
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, minsize=96)
        self.grid_columnconfigure(2, minsize=86)
        self.grid_columnconfigure(3, minsize=72)
        self.grid_columnconfigure(4, minsize=92)

        self.name_label = ctk.CTkLabel(
            self,
            text=model.get("name", model.get("id", "Model")),
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
            text_color=TEXT_PRIMARY,
        )
        self.name_label.grid(row=0, column=0, sticky="ew", padx=(10, 6), pady=9)
        self.type_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11), anchor="w", text_color=TEXT_SECONDARY)
        self.type_label.grid(row=0, column=1, sticky="ew", padx=4, pady=9)
        self.fit_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11), anchor="w")
        self.fit_label.grid(row=0, column=2, sticky="ew", padx=4, pady=9)
        self.size_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11), anchor="e", text_color=TEXT_MUTED)
        self.size_label.grid(row=0, column=3, sticky="ew", padx=4, pady=9)
        self.status_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11), anchor="e")
        self.status_label.grid(row=0, column=4, sticky="ew", padx=(4, 10), pady=9)

        for widget in (self, self.name_label, self.type_label, self.fit_label, self.size_label, self.status_label):
            widget.bind("<Button-1>", self._select)
            widget.bind("<Double-Button-1>", self._activate)
            widget.bind("<Return>", self._activate)
            widget.bind("<space>", self._activate)
        try:
            self._canvas.configure(takefocus=1)
            self._canvas.bind("<FocusIn>", self._focus_in)
            self._canvas.bind("<FocusOut>", self._focus_out)
            self._canvas.bind("<Return>", self._activate)
            self._canvas.bind("<space>", self._activate)
        except Exception:
            pass

        self.refresh_status(local_names=local_names, comfyui_model_names=comfyui_model_names)

    def _select(self, _event=None) -> None:
        self.app._select_model_row(self.model.get("id", ""))

    def _activate(self, _event=None) -> None:
        self.app._select_model_row(self.model.get("id", ""))
        self.app._run_selected_model_primary_action()

    def _focus_in(self, _event=None) -> None:
        self._has_focus = True
        self._apply_row_style()
        # Focus-follows-selection: arrow-key navigation should update the
        # detail pane on the right just like clicking does. This matches
        # the Windows Explorer / Outlook / Settings pattern where moving
        # the focus highlight ALSO previews the focused item. Enter still
        # "activates" (runs the primary action via _activate).
        try:
            self.app._select_model_row(self.model.get("id", ""))
        except Exception:
            pass

    def _focus_out(self, _event=None) -> None:
        self._has_focus = False
        self._apply_row_style()

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self._apply_row_style()

    def _apply_row_style(self) -> None:
        self.configure(
            fg_color=("#dbeafe", "#1f3658") if self._selected else INPUT_SURFACE,
            border_color=self.app._IG_ACCENT if (self._selected or self._has_focus) else BORDER_STRONG,
            border_width=2 if (self._selected or self._has_focus) else 1,
        )

    def refresh_perf_badges(self) -> None:
        fit_text, fit_color = self.app._model_fit_display(self.model)
        self.fit_label.configure(text=fit_text, text_color=fit_color)

    def refresh_status(self, local_names: set | None = None, comfyui_model_names: set | None = None):
        fit_text, fit_color = self.app._model_fit_display(self.model)
        status_text, status_color, _state = self.app._model_install_status(
            self.model,
            local_names=local_names,
            comfyui_model_names=comfyui_model_names,
        )
        # User-added entries get a "(yours)" suffix so the catalog stays
        # legible when both built-ins and user additions are interleaved.
        base_name = self.model.get("name", self.model.get("id", "Model"))
        display_name = f"{base_name}  (yours)" if self.model.get("user_added") else base_name
        self.name_label.configure(text=display_name)
        self.type_label.configure(text=self.app._model_type_label(self.model))
        self.fit_label.configure(text=fit_text, text_color=fit_color)
        self.size_label.configure(text=f"{self.model.get('size_gb', 0):g} GB")
        self.status_label.configure(text=status_text, text_color=status_color)


# ── Main Application ──────────────────────────────────────────────────────────

class App(ctk.CTk):
    # Thread-safe after() for Python 3.13+: tkinter no longer allows
    # .after() from non-main threads.  We queue the callback and let a
    # main-thread poller dispatch it.
    _ts_queue: queue.Queue = queue.Queue()
    _main_thread_id: int = threading.get_ident()

    def __init__(self):
        super().__init__()
        self._poll_threadsafe()

        self._startup_t0 = time.perf_counter()
        logger.set_log_file(Path(__file__).parent.parent / "localai.log")
        logger.add_listener(self._on_log_entry)
        self._pending_log_entries: list[dict] = []
        self._log_flush_scheduled = False

        _t = time.perf_counter()
        self.cfg = config.load()
        _apply_theme_palette_globals(self.cfg)
        self._log_startup_step("config load", _t)
        ctk.set_appearance_mode(config.normalize_theme_mode(self.cfg.get("theme_mode")).title())
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_TITLE}  v{APP_VERSION}")
        self.minsize(900, 600)

        # Center the window on the primary screen
        w, h = 1280, 800
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

        # State
        self.ollama = OllamaClient(self.cfg["ollama_host"])
        self.ollama_ok = False
        self.comfyui = ComfyUIClient(self.cfg.get("comfyui_host", "http://127.0.0.1:8188"))
        self.comfyui_ok = False
        self.comfyui_process: Optional[subprocess.Popen] = None
        self._comfyui_current_launch_flags: list[str] = []
        self._comfyui_dependency_lock = threading.Lock()
        self._comfyui_dependency_ok_by_python: dict[str, bool] = {}
        self._comfyui_initialized: bool = False
        # v5.3.6+ cold-start fix: every False return from
        # _start_comfyui_process / _ensure_comfyui_core_dependencies stamps
        # this with the specific reason (not-installed / pip-failed / popen-
        # raised / probe-raised), and _bench_ensure_comfyui_ready surfaces it
        # in the (bool, reason) tuple it returns to BatchRunner so the bench
        # log shows the actual root cause instead of "ComfyUI is not running
        # and could not be started".  The lock guards against the
        # (theoretical) case where the Image Gen UI thread and the benchmark
        # worker thread both touch the start path concurrently — in practice
        # the benchmark UI disables the Image Gen page, but defensive locking
        # keeps the reason string from being corrupted by interleaved writes.
        self._comfyui_last_start_failure_reason: str = ""
        self._comfyui_last_start_failure_lock = threading.Lock()
        # Startup must never install or mutate packages; setup.bat/fix_nvidia_pytorch.bat
        # own CUDA PyTorch remediation. Run the expensive PyTorch/CUDA probe after the
        # first window paints so cached CUDA validation cannot block app launch.
        self.gpu_info = GPUInfo("cpu", "Detecting GPU")
        self._gpu_detection_pending: bool = True
        # Auto-enable CPU mode on hardware with no GPU. Users with a GPU can
        # still toggle this on via the Image Gen page checkbox.
        self._comfyui_force_cpu: bool = False
        # v5.5.12 (Ron, 2026-05-27): Cached at GPU detection completion.
        # True iff the active GPU is a Windows integrated GPU (Intel Arc/UHD,
        # AMD Radeon Graphics, Snapdragon Adreno) — i.e. unified memory AND
        # subject to DirectX TDR (~2s per kernel). Apple Silicon iGPUs are
        # also unified_memory but use Metal (no TDR), so they are NOT marked
        # here. Used by ``_populate_image_model_menu`` to gate heavy SDXL /
        # Flux models off the dropdown and by ``_clamp_image_params_for_igpu``
        # to cap default steps / resolution before first generation.
        self._windows_unified_igpu: bool = False
        self.active_model: Optional[dict] = None
        self.active_session: Optional[OnnxModelSession | OnnxGenAISession | OVModelSession] = None
        self._active_model_display: str = "None loaded"
        self.chat_history: list[dict] = []
        self._stop_event = threading.Event()
        self._closing: bool = False
        self._chat_thinking: bool = False
        self._backend_var = ctk.StringVar(value="GPU/CPU (Ollama)")
        self._low_res_var = ctk.BooleanVar(value=self.cfg.get("low_resources_mode", False))
        self._max_tokens_var = ctk.StringVar(value=self._response_token_budget_display())
        self._model_cards: list[ModelListRow] = []
        # v5: keyed pool for show/hide filtering (no destroy/recreate)
        self._model_cards_by_id: dict[str, ModelListRow] = {}
        self._selected_model_id: str | None = None
        self._model_detail_widgets: dict[str, object] = {}
        self._model_type_filter_var = ctk.StringVar(value="All types")
        self._model_size_filter_var = ctk.StringVar(value="All sizes")
        self._type_btns: dict[str, "ctk.CTkButton"] = {}
        self._size_btns: dict[str, "ctk.CTkButton"] = {}
        self._model_row_fit_cache: dict[str, str] = {}
        self._cards_empty_label = None
        self._models_page_just_built: bool = False
        self._model_population_token: int = 0
        self._model_status_refresh_generation: int = 0
        self._model_status_refresh_thread: threading.Thread | None = None
        self._model_primary_action_generation: int = 0
        self._models_list_width: int = 680
        # v5.3: section headers for image-gen sectioned layout (id -> header widget)
        self._section_headers: dict[str, "ctk.CTkFrame"] = {}
        # v5.3: per-session collapsed-state of section headers (default: doesn't-fit collapsed)
        self._section_collapsed: dict[str, bool] = {"img_doesnt_fit": True}
        # v5.3: when True, also include models that exceed the active device VRAM
        self._image_gen_show_oversize_var = ctk.BooleanVar(value=False)
        # v5.3: cached header banner (re-shown only when image-gen is in the filter)
        self._img_summary_banner = None
        self._img_summary_banner_lbl = None
        self._img_summary_banner_switch = None
        # v2026.06.01.10: in-app "incomplete setup" warning banner. Created
        # by _build_setup_warning_banner() inside _build_ui(); hidden by
        # default. _refresh_setup_banner() shows/hides it when async
        # startup checks (Ollama probe, GPU detection, ComfyUI install
        # probe) complete with a result that suggests setup.bat did not
        # finish successfully (e.g. setup window auto-closed on the user
        # before they noticed an error).
        self._setup_warning_banner = None
        self._setup_warning_label = None
        self._setup_warning_button = None
        # v2026.06.01.10: cached signal for the banner check. Set by the
        # GPU detection callback when an NVIDIA GPU is present but torch
        # is CPU-only / missing — the exact "setup.bat skipped CUDA"
        # signal we need without re-probing GPU each banner refresh.
        self._pytorch_cuda_missing_on_nvidia: bool = False
        # v5: token-batching buffer for chat streaming
        self._token_buf: list[tuple[int, str]] = []
        self._token_flush_scheduled: bool = False
        self._chat_generation_id: int = 0
        # v5: log-batching buffer for benchmark
        self._bench_log_buf: list[str] = []
        self._bench_log_flush_scheduled: bool = False
        self._chat_response_placeholder_active: bool = False
        self._toolbox_install_thread: threading.Thread | None = None
        self._toolbox_install_workflow_id: str | None = None
        self._toolbox_workflow_thread: threading.Thread | None = None
        self._toolbox_active_workflow_id: str | None = None
        self._toolbox_workflow_token: int = 0
        self._toolbox_last_progress: dict[str, str] = {}
        # v5: TTL cache for ollama.local_model_names() (5s)
        self._local_names_cache: tuple[float, set] | None = None
        self._home_disk_free_cache: tuple[float, float] | None = None
        self._toolbox_model_by_id_cache: tuple[int, dict[str, dict]] | None = None
        self._runnable_toolbox_titles_cache: tuple[float, tuple, list[str]] | None = None
        self._chat_font_family: str = self._configured_chat_font_family()
        self._chat_first_token_started_at: float | None = None
        self._chat_first_token_timer_id: Optional[str] = None
        # v5: cached state so lazily-built tabs can sync on first visit.
        self._last_comfyui_status: tuple[str, str] | None = None
        self._image_gen_enabled: bool = True
        self._vision_comfyui_deferred_restart: bool = False
        self._pending_generation_after_comfyui_restart: bool = False
        # v5: page-builder registry for lazy tab construction
        self._page_builders: dict = {}
        sku_cfg = system_info.load_optional_sku_config()
        self._optional_sku_feature: dict = sku_cfg.get("feature", {})
        initial_skus = sku_cfg.get("skus", [])
        self._optional_skus_enabled: bool = bool(initial_skus)

        # v5: state vars that callbacks can touch BEFORE the Models page is
        # lazily built. _optional_filter_var/_selected_cats/_cat_btns/_model_search_var
        # used to be created inside _build_models_page; with lazy loading the
        # device detection callback fires before that happens, so we initialize
        # them eagerly here and the build method now skips re-creation when
        # they already exist.
        self._optional_filter_var = ctk.StringVar(value=self._sku_all_filter_value())
        self._selected_cats: set[str] = set()
        self._user_set_cat_filter: bool = False
        self._cat_btns: dict = {}
        self._model_search_var = ctk.StringVar(value="")
        self._download_thread: Optional[threading.Thread] = None
        self._img_thread: Optional[threading.Thread] = None
        self._img_stop_event = threading.Event()
        self._analyze_thread: Optional[threading.Thread] = None
        self._analyze_stop_event: threading.Event = threading.Event()
        self._analyze_gen_id: int = 0
        self._optional_sku: Optional[dict] = None  # detected optional SKU or local device
        self._sku_is_auto: bool = False         # True when SKU was built from local HW, not from the file
        self._device_display_text = "Detecting …"
        self._device_display_color = TEXT_MUTED
        self._device_sidebar_text = self._sku_text("detecting_text", "This Device: detecting ...")
        self._device_sidebar_color = TEXT_MUTED

        # Start with current SKU/catalog data so the first window can render.
        # User-editable JSON files are loaded just after first paint.
        self._optional_skus: list[dict] = list(initial_skus)
        self._apply_optional_skus_to_modules()
        self._catalog_models: list[dict] = list(catalog.MODELS)

        logger.info(f"{APP_TITLE} {APP_VERSION} starting …")
        _t = time.perf_counter()
        self._build_ui()
        self._log_startup_step("first UI build", _t)

        # Delay startup checks until mainloop begins to avoid threading issues
        self.after(50,  self._load_startup_data_async)
        self.after(75,  self._start_gpu_detection_async)
        self.after(150, self._startup_kill_orphan_comfyui)
        self.after(100, self._check_ollama_async)
        self.after(180, self._resume_pending_migration_if_any_async)
        self.after(190, self._heal_orphan_ollama_blobs_async)
        self.after(195, self._heal_legacy_onnx_paths_async)
        self.after(200, self._check_storage_relocation_on_startup_async)
        self.after(220, self._process_scheduled_deletes_after_startup_async)
        self.after(300, self._detect_optional_device_async)
        self.after(400, self._start_resource_monitor)
        self.after(500, self._check_low_resources_startup)
        self.after(800, self._minimize_console)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _log_startup_step(self, label: str, start: float) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_ms = (time.perf_counter() - getattr(self, "_startup_t0", start)) * 1000
        logger.info(f"Startup: {label} took {elapsed_ms:.0f} ms ({total_ms:.0f} ms total)")

    def _start_gpu_detection_async(self) -> None:
        def _worker() -> None:
            started = time.perf_counter()
            try:
                gpu_info = detect_gpu_cached(auto_fix=False)
            except Exception as exc:
                logger.warning(f"GPU detection failed; using CPU fallback: {exc}")
                gpu_info = GPUInfo("cpu", "CPU")
            self.after(0, lambda info=gpu_info, started=started: self._apply_gpu_detection_result(info, started))

        threading.Thread(target=_worker, name="GpuDetection", daemon=True).start()

    def _apply_gpu_detection_result(self, gpu_info: GPUInfo, started: float) -> None:
        self.gpu_info = gpu_info
        self._gpu_detection_pending = False
        self._comfyui_force_cpu = gpu_info.gpu_type == "cpu"
        # v2026.06.01.10: cache the "NVIDIA card present but torch is CPU-only"
        # signal here (cheap probe — gpu_detect already loaded everything it
        # needs during detect_gpu_cached). The setup-warning banner reads
        # this flag without re-running GPU detection on every refresh.
        try:
            from src import gpu_detect as _gd
            nvidia_name = _gd._nvidia_gpu_present()
            if nvidia_name:
                torch_state, _ = _gd._torch_cuda_state()
                self._pytorch_cuda_missing_on_nvidia = (
                    torch_state in ("cpu_wheel", "missing")
                )
            else:
                self._pytorch_cuda_missing_on_nvidia = False
        except Exception:
            self._pytorch_cuda_missing_on_nvidia = False
        # v5.5.12: Detect Windows-integrated GPU (subject to DXGI TDR).
        # Cached here once because system_info.get_gpu_info() touches WMI and
        # _populate_image_model_menu can fire repeatedly on ComfyUI restarts.
        if sys.platform == "win32":
            try:
                igpu_gpus = system_info.get_gpu_info()
                if igpu_gpus:
                    g0 = igpu_gpus[0]
                    self._windows_unified_igpu = (
                        bool(g0.get("unified_memory"))
                        and g0.get("vendor") != "Apple"
                    )
            except Exception:
                self._windows_unified_igpu = False
        try:
            if hasattr(self, "_img_cpu_mode_var"):
                self._img_cpu_mode_var.set(self._comfyui_force_cpu)
            if hasattr(self, "_img_cpu_mode_cb"):
                self._img_cpu_mode_cb.configure(
                    state="disabled" if gpu_info.gpu_type == "cpu" else "normal",
                    text=(
                        "Force CPU mode  ·  no GPU detected — SD 1.5 recommended"
                        if gpu_info.gpu_type == "cpu"
                        else "Force CPU mode  ·  slower, but no VRAM pressure"
                    ),
                )
        except Exception:
            pass
        self._home_disk_free_cache = None
        self._runnable_toolbox_titles_cache = None
        self._log_startup_step("GPU detection", started)
        if hasattr(self, "_home_chip_gpu"):
            self._refresh_home_page()
        self._update_category_for_device()
        self._refresh_bench_profile_values(preserve_selection=True)
        if hasattr(self, "_img_ready_lbl"):
            self._refresh_image_readiness()
        # v2026.06.01.10: GPU result may newly satisfy / break the "PyTorch
        # is CPU-only on an NVIDIA box" condition — refresh the in-app
        # incomplete-setup banner so the warning appears (or clears).
        self._refresh_setup_warning_banner()

    def _load_startup_data_async(self):
        """Load user-editable SKU/catalog files after the first window is visible."""
        def _worker():
            try:
                t0 = time.perf_counter()
                sku_cfg = system_info.load_optional_sku_config()
                logger.info(f"Startup: optional SKU load took {(time.perf_counter() - t0) * 1000:.0f} ms")

                t0 = time.perf_counter()
                catalog.ensure_catalog_file()
                models = catalog.load_catalog()
                logger.info(f"Startup: catalog load took {(time.perf_counter() - t0) * 1000:.0f} ms")
            except Exception as exc:
                logger.error(f"Startup data load failed: {exc}")
                return
            self.after(0, lambda: self._apply_startup_data(sku_cfg, models))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_startup_data(self, sku_cfg: dict, models: list[dict]):
        self._optional_sku_feature = sku_cfg.get("feature", {})
        self._optional_skus = sku_cfg.get("skus", [])
        self._optional_skus_enabled = bool(self._optional_skus)
        self._apply_optional_skus_to_modules()
        if self._optional_sku:
            self._inject_local_sku(self._optional_sku)
        self._refresh_sku_filter_values()

        self._catalog_models = models
        self._toolbox_model_by_id_cache = None
        self._runnable_toolbox_titles_cache = None
        self._refresh_chat_model_selector()
        if self._optional_sku:
            self._update_category_for_device()
        else:
            self._schedule_model_card_population()
        self._validate_image_recommended_settings()
        self._validate_image_perf_profile()
        self._refresh_bench_profile_values(preserve_selection=True)
        logger.info("Startup: user catalog and SKU files applied")

    def _minimize_console(self):
        """Minimize the console window that launched the app (Windows only)."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
        except Exception:
            pass  # no console — silently skip

    # ── Thread-safe after() ───────────────────────────────────────────────────

    def _poll_threadsafe(self):
        """Drain the thread-safe queue on the main thread."""
        if getattr(self, "_closing", False):
            return
        while not self._ts_queue.empty():
            try:
                ms, fn, args = self._ts_queue.get_nowait()
                super().after(ms, fn, *args)
            except queue.Empty:
                break
        if not getattr(self, "_closing", False):
            super().after(50, self._poll_threadsafe)

    def after(self, ms, func=None, *args):
        """Override: if called from a background thread, enqueue instead."""
        if func is None:
            # after(ms) without callback — just a sleep, always safe
            return super().after(ms)
        if getattr(self, "_closing", False):
            return None
        if threading.get_ident() != self._main_thread_id:
            self._ts_queue.put((ms, func, args))
            return None
        return super().after(ms, func, *args)

    def _widget_is_alive(self, widget) -> bool:
        """True when a Tk/CTk widget object still has a live Tk command."""
        state = object.__getattribute__(self, "__dict__")
        if bool(state.get("_closing", False)) or widget is None:
            return False
        try:
            return bool(widget.winfo_exists())
        except (AttributeError, TclError):
            return False

    # ── Optional SKU helpers ──────────────────────────────────────────────────

    def _sku_text(self, key: str, default: str) -> str:
        value = (getattr(self, "_optional_sku_feature", {}) or {}).get(key)
        return str(value) if value else default

    def _sku_all_filter_value(self) -> str:
        if getattr(self, "_optional_skus_enabled", False):
            return self._sku_text("all_filter_value", "All Profiles")
        return "All"

    def _sku_filter_values(self) -> list[str]:
        if getattr(self, "_optional_skus_enabled", False):
            return [self._sku_all_filter_value()] + catalog.OPTIONAL_SKU_ORDER
        return ["All", "This Device"]

    def _refresh_sku_filter_values(self) -> None:
        new_values = self._sku_filter_values()
        if hasattr(self, "_optional_sku_segbtn") and self._optional_sku_segbtn is not None:
            self._optional_sku_segbtn.configure(values=new_values)
        if self._optional_filter_var.get() not in new_values:
            self._optional_filter_var.set(self._sku_all_filter_value())

    def _inject_local_sku(self, sku: dict) -> None:
        """
        Add *sku* to the in-memory SKU list at the correct capability-ordered
        position if its name is not already present, then sync catalog module
        dicts and the segmented button values. Never writes to the optional SKU file.
        """
        if not getattr(self, "_optional_skus_enabled", False):
            self._apply_optional_skus_to_modules()
            self._refresh_sku_filter_values()
            return
        if any(s["name"] == sku["name"] for s in self._optional_skus):
            return
        # Determine where this device fits by counting runnable models
        device_count = catalog.sku_model_count(
            sku.get("vram_gb", 0), sku.get("ram_gb", 0),
            self._catalog_models,
        )
        insert_idx = 0
        for i, s in enumerate(self._optional_skus):
            s_count = catalog.sku_model_count(
                catalog.OPTIONAL_SKU_VRAM.get(s["name"], s.get("vram_gb", 0)),
                catalog.OPTIONAL_SKU_RAM.get(s["name"], s.get("ram_gb", 0)),
                self._catalog_models,
            )
            if s_count <= device_count:
                insert_idx = i + 1
        self._optional_skus.insert(insert_idx, sku)
        self._apply_optional_skus_to_modules()
        self._refresh_sku_filter_values()

    def _apply_optional_skus_to_modules(self):
        """Push self._optional_skus into catalog and system_info module variables."""
        system_info.OPTIONAL_SKUS = self._optional_skus
        # Keep ``system_info.BENCHMARK_SKU_PROFILES`` (loaded at import from
        # ``skus.json``) in sync with whatever the user just reloaded so
        # all importers — including ``_bench_profile_specs`` and
        # ``benchmark_sku_profile_name`` — see the live list. We use slice
        # assignment ``[:] =`` rather than ``clear()`` + ``extend()`` so the
        # list contents transition atomically (no observable empty-list
        # window for a concurrent reader). MUST NOT rebind the attribute,
        # because ``tools/validate_doc_samples.py`` does
        # ``from src.system_info import BENCHMARK_SKU_PROFILES``.
        system_info.BENCHMARK_SKU_PROFILES[:] = list(self._optional_skus)
        catalog.OPTIONAL_SKU_ORDER = [s["name"] for s in self._optional_skus]
        catalog.OPTIONAL_SKU_VRAM  = {s["name"]: s["vram_gb"] for s in self._optional_skus}
        catalog.OPTIONAL_SKU_RAM   = {s["name"]: s.get("ram_gb", 0) for s in self._optional_skus}

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self, width=210, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(12, weight=1)

        ctk.CTkLabel(
            sidebar, text="LocalAI", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=16, pady=(20, 4))
        ctk.CTkLabel(
            sidebar, text="Studio", font=ctk.CTkFont(size=13), text_color=TEXT_MUTED
        ).grid(row=1, column=0, padx=16, pady=(0, 20))

        self._nav_btns = {}
        for i, (page, label) in enumerate([
            ("home",   "  🏠  Home"),
            ("chat",   "  💬  Chat"),
            ("toolbox", "  🧰  Toolbox"),
            ("image_gen", "  🖼  Image Gen"),
            ("models", "  📦  Models"),
            ("benchmark", "  📊  Benchmark"),
            ("system", "  ⚙  System"),
            ("logs",   "  📜  Logs"),
            ("settings", "  🛠  Settings"),
            ("docs",     "  📖  Help / Docs"),
        ]):
            btn = ctk.CTkButton(
                sidebar, text=label, anchor="w",
                fg_color=INPUT_SURFACE, text_color=TEXT_PRIMARY,
                hover_color=BUTTON_SECONDARY_HOVER,
                border_width=1, border_color=BORDER_STRONG,
                corner_radius=8,
                command=lambda p=page: self._switch_page(p),
            )
            btn.grid(row=i + 2, column=0, padx=8, pady=2, sticky="ew")
            self._nav_btns[page] = btn

        # Accessibility: bind Up/Down arrow keys on each nav button so the
        # navigation rail can be operated without repeated Tab presses.
        # Home/End jump to the first/last entry. Tab/Shift-Tab still work
        # naturally (handled by ``a11y.install()``).
        self._wire_nav_rail_arrow_keys()

        # v5.1: Image Gen nav tab is ALWAYS enabled. The page itself surfaces
        # readiness (ComfyUI status pill and readiness checklist)
        # so users can open it any time to read prompts, prep settings, and pick a
        # vision model — even before ComfyUI is up. Generate/Analyze start or
        # restart ComfyUI only when those actions actually need it.

        # Device indicator in sidebar footer
        self._device_sidebar_label = ctk.CTkLabel(
            sidebar, text=self._device_sidebar_text, font=ctk.CTkFont(size=10),
            text_color=self._device_sidebar_color, wraplength=184, anchor="w",
            justify="left",
        )
        self._device_sidebar_label.grid(row=13, column=0, padx=8, pady=(4, 0), sticky="sew")

        # Ollama status in sidebar footer
        self._ollama_status_label = ctk.CTkLabel(
            sidebar, text="Ollama: checking …", font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED, wraplength=184, anchor="w", justify="left",
        )
        self._ollama_status_label.grid(row=14, column=0, padx=8, pady=(0, 2), sticky="ew")

        # ComfyUI status in sidebar footer
        self._comfyui_status_label = ctk.CTkLabel(
            sidebar, text="ComfyUI: checking …", font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED, wraplength=184, anchor="w", justify="left",
        )
        self._comfyui_status_label.grid(row=15, column=0, padx=8, pady=(0, 4), sticky="ew")

        # Content area
        self._content = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self._content.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        # v2026.06.01.10: row 0 is reserved for the optional incomplete-setup
        # banner (hidden by default; shows when setup.bat failed silently and
        # the user has no idea their install is broken). Pages live in row 1
        # so banner appears ABOVE every page without per-page coordination.
        self._content.grid_rowconfigure(0, weight=0)
        self._content.grid_rowconfigure(1, weight=1)
        self._content.grid_columnconfigure(0, weight=1)

        self._build_setup_warning_banner()

        # Pages — v5: lazy-loaded. Only Models is built eagerly so the app
        # opens to a populated landing screen. The rest build on first switch.
        self._pages: dict = {}
        self._page_builders = {
            "home":      self._build_home_page,
            "models":    self._build_models_page,
            "chat":      self._build_chat_page,
            "toolbox":   self._build_toolbox_page,
            "image_gen": self._build_image_page,
            "benchmark": self._build_benchmark_page,
            "system":    self._build_system_page,
            "logs":      self._build_logs_page,
            "settings":  self._build_settings_page,
        }
        self._build_home_page()

        # Status bar
        status_bar = ctk.CTkFrame(self, height=28, corner_radius=0)
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        status_bar.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(
            status_bar, text="Ready.", font=ctk.CTkFont(size=11), anchor="w"
        )
        self._status_label.grid(row=0, column=0, padx=10, sticky="w")

        self._progress_bar = ctk.CTkProgressBar(status_bar, width=200, height=12)
        self._progress_bar.set(0)
        self._progress_bar.grid(row=0, column=1, padx=10, sticky="e")
        self._progress_bar.grid_remove()  # hidden until needed

        self._switch_page("home")
        # v2026.06.01.10: restore the incomplete-setup banner after a theme
        # rebuild wipes & rebuilds every widget. First-run case is a no-op
        # because async startup checks haven't completed yet — the banner
        # will appear on its own when those callbacks fire.
        try:
            self._refresh_setup_warning_banner()
        except Exception:
            pass

    def _rebuild_ui_for_theme_change(self, page: str) -> None:
        """Recreate widgets so explicit Light/Dark palette changes apply immediately."""
        for child in list(self.winfo_children()):
            child.destroy()

        self._nav_btns = {}
        self._pages = {}
        self._model_cards = []
        self._model_cards_by_id = {}
        self._selected_model_id = None
        self._model_detail_widgets = {}
        self._type_btns = {}
        self._size_btns = {}
        self._model_row_fit_cache = {}
        self._section_headers = {}
        self._cards_empty_label = None
        self._img_summary_banner = None
        self._img_summary_banner_lbl = None
        self._img_summary_banner_switch = None
        self._setup_warning_banner = None
        self._setup_warning_label = None
        self._setup_warning_button = None
        self._chat_model_menu = None
        self._chat_load_btn = None
        self._log_box = None
        self._log_level_var = None
        self._log_search_var = None

        self._build_ui()
        if page in self._page_builders:
            self._switch_page(page)

    # ── Pages ─────────────────────────────────────────────────────────────────

    # v5: Home dashboard
    def _build_home_page(self):
        """Landing page with quick actions, status chips, and a recent-activity
        rollup. This is the first thing users see when LocalAI Studio opens."""
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["home"] = page
        page.grid_rowconfigure(3, weight=1)
        page.grid_columnconfigure(0, weight=1)

        # Header
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 8))
        ctk.CTkLabel(
            hdr, text="Welcome to LocalAI Studio",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(side="left")
        ctk.CTkLabel(
            hdr, text=f"  v{APP_VERSION}",
            font=ctk.CTkFont(size=13), text_color=TEXT_MUTED,
        ).pack(side="left", padx=(8, 0), pady=(8, 0))

        # Backend status chips row
        chips_row = ctk.CTkFrame(page, fg_color="transparent")
        chips_row.grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 12))
        ctk.CTkLabel(
            chips_row, text="Backends",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 8))

        self._home_chip_ollama = ctk.CTkLabel(
            chips_row, text="  Ollama: …  ", corner_radius=10,
            font=ctk.CTkFont(size=11), fg_color=("gray80", "gray25"),
            padx=10, pady=2,
        )
        self._home_chip_ollama.pack(side="left", padx=4)
        self._home_chip_comfyui = ctk.CTkLabel(
            chips_row, text="  ComfyUI: …  ", corner_radius=10,
            font=ctk.CTkFont(size=11), fg_color=("gray80", "gray25"),
            padx=10, pady=2,
        )
        self._home_chip_comfyui.pack(side="left", padx=4)
        self._home_chip_gpu = ctk.CTkLabel(
            chips_row, text=f"  GPU: {self.gpu_info.gpu_type.upper()}  ",
            corner_radius=10,
            font=ctk.CTkFont(size=11), fg_color=("gray80", "gray25"),
            padx=10, pady=2,
        )
        self._home_chip_gpu.pack(side="left", padx=4)

        # Quick actions row
        qa_card = ctk.CTkFrame(page, corner_radius=12)
        qa_card.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 16))
        qa_card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(
            qa_card, text="Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(16, 8))

        def _qa_btn(parent, emoji, title, subtitle, command, color):
            f = ctk.CTkFrame(parent, corner_radius=10, fg_color=color)
            ctk.CTkLabel(
                f, text=emoji, font=ctk.CTkFont(size=28),
            ).pack(pady=(16, 4))
            ctk.CTkLabel(
                f, text=title, font=ctk.CTkFont(size=14, weight="bold"),
            ).pack()
            ctk.CTkLabel(
                f, text=subtitle, font=ctk.CTkFont(size=10),
                text_color=TEXT_SECONDARY, wraplength=200,
            ).pack(pady=(2, 8), padx=8)
            ctk.CTkButton(
                f, text="Open", command=command, width=120,
                **self._outline_button_style(),
            ).pack(pady=(0, 16))
            return f

        _qa_btn(
            qa_card, "💬", "Start a Chat",
            "Open a conversation with any installed text model.",
            lambda: self._switch_page("chat"),
            ("#cfe9ff", "#1a4068"),
        ).grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 16))
        _qa_btn(
            qa_card, "🖼", "Generate an Image",
            "Render with ComfyUI using SD / SDXL / Flux / Chroma.",
            lambda: self._switch_page("image_gen"),
            ("#e8d5ff", "#3d2860"),
        ).grid(row=1, column=1, sticky="nsew", padx=12, pady=(0, 16))
        _qa_btn(
            qa_card, "📦", "Browse Models",
            "Download, manage, and load models from the catalog.",
            lambda: self._switch_page("models"),
            ("#d0f0d8", "#1f4a2a"),
        ).grid(row=1, column=2, sticky="nsew", padx=12, pady=(0, 16))
        _qa_btn(
            qa_card, "🧰", "Open Toolbox",
            "Transcribe, read, speak, search, and describe locally.",
            lambda: self._switch_page("toolbox"),
            ("#fff0d6", "#4a3a1a"),
        ).grid(row=1, column=3, sticky="nsew", padx=12, pady=(0, 16))

        # System snapshot + suggestions
        bottom = ctk.CTkFrame(page, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="nsew", padx=24, pady=(0, 20))
        bottom.grid_columnconfigure((0, 1), weight=1)
        bottom.grid_rowconfigure(0, weight=1)

        # System snapshot card
        sys_card = ctk.CTkFrame(bottom, corner_radius=12)
        sys_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(
            sys_card, text="System",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 4))

        self._home_sys_text = ctk.CTkLabel(
            sys_card, text="Loading …",
            font=ctk.CTkFont(size=11), justify="left", anchor="w",
            text_color=TEXT_SECONDARY,
        )
        self._home_sys_text.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 16))

        # Suggestions card.
        sugg_card = ctk.CTkFrame(bottom, corner_radius=12)
        sugg_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ctk.CTkLabel(
            sugg_card, text="Next steps",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 4))
        next_steps = (
            "• Browse Models to download or load a text model\n"
            "• Open Chat after a model is ready\n"
            "• Use Image Gen for ComfyUI workflows\n"
            "• Try Toolbox for local transcription, speech, search, and document helpers\n"
            "• Open Help / Docs for usage details"
        )
        ctk.CTkLabel(
            sugg_card, text=next_steps,
            font=ctk.CTkFont(size=11), justify="left", anchor="w",
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 16))

    def _cached_home_disk_free_gb(self) -> float:
        now = time.monotonic()
        cached = getattr(self, "_home_disk_free_cache", None)
        if cached and now - cached[0] < 5:
            return cached[1]
        try:
            import shutil as _sh
            disk_free = _sh.disk_usage(".").free / (1024 ** 3)
        except Exception:
            disk_free = 0.0
        self._home_disk_free_cache = (now, disk_free)
        return disk_free

    def _refresh_home_page(self):
        """Update Home dashboard chips and system snapshot. Called on every
        visit to the Home tab. Safe to call before backend checks complete."""
        try:
            # Backend chips
            if getattr(self, '_home_chip_ollama', None):
                if self.ollama_ok:
                    self._home_chip_ollama.configure(
                        text="  ✓ Ollama running  ", fg_color=("#cdebd1", "#1a4a25"),
                    )
                else:
                    self._home_chip_ollama.configure(
                        text="  ⚠ Ollama offline  ", fg_color=("#f5d6d6", "#4a1a1a"),
                    )
            if getattr(self, '_home_chip_comfyui', None):
                if self.comfyui_ok:
                    self._home_chip_comfyui.configure(
                        text="  ✓ ComfyUI running  ", fg_color=("#cdebd1", "#1a4a25"),
                    )
                else:
                    self._home_chip_comfyui.configure(
                        text="  ⚠ ComfyUI offline  ", fg_color=("#f5d6d6", "#4a1a1a"),
                    )
            if getattr(self, '_home_chip_gpu', None):
                gpu_text = self.gpu_info.device_name or self.gpu_info.gpu_type.upper()
                if len(gpu_text) > 32:
                    gpu_text = gpu_text[:30] + "…"
                color = (
                    ("#cdebd1", "#1a4a25") if self.gpu_info.gpu_type == "cuda" else
                    ("#fff0d6", "#4a3a1a") if self.gpu_info.gpu_type in ("mps", "directml") else
                    ("#f5d6d6", "#4a1a1a")
                )
                self._home_chip_gpu.configure(text=f"  {gpu_text}  ", fg_color=color)
        except Exception:
            pass

        # System snapshot — cheap one-shot read
        try:
            if getattr(self, '_home_sys_text', None):
                try:
                    import psutil as _ps
                    cpu = _ps.cpu_percent(interval=None)
                    ram = _ps.virtual_memory()
                    ram_pct = ram.percent
                    ram_total = ram.total / (1024 ** 3)
                except Exception:
                    cpu = ram_pct = 0.0
                    ram_total = 0.0
                disk_free = self._cached_home_disk_free_gb()
                toolbox = ", ".join(self._runnable_toolbox_titles()[:4]) or "none"
                self._home_sys_text.configure(text=(
                    f"GPU:       {self.gpu_info.device_name or self.gpu_info.gpu_type}\n"
                    f"CPU:       {cpu:.0f}% used\n"
                    f"RAM:       {ram_pct:.0f}% used  ({ram_total:.1f} GB total)\n"
                    f"Disk free: {disk_free:.1f} GB\n"
                    f"Catalog:   {len(self._catalog_models)} models\n"
                    f"Toolbox:   {toolbox}"
                ))
        except Exception:
            pass

    # ── v5.3: Image Gen sectioning / fit helpers ──────────────────────────────

    # Order of section buckets when Image Generation is active. Each entry
    # maps to a (display_title, icon) pair. The "top_picks" section is
    # computed dynamically (whatever's flagged ``recommendation == "top_pick"``
    # AND fits the device) so it always shows first. The "doesnt_fit" section
    # is appended last when override is on.
    _IMG_SECTIONS = (
        ("img_top_picks",    "⭐  Top picks for your GPU",       "⭐"),
        ("img_speed",        "⚡  Fast — quick iteration",        "⚡"),
        ("img_quality",      "🏆  Highest quality (flagship)",   "🏆"),
        ("img_photo",        "📸  Photorealistic",               "📸"),
        ("img_art",          "🎨  Stylized & artistic",          "🎨"),
        ("img_general",      "🧰  Versatile / foundation",        "🧰"),
        ("img_doesnt_fit",   "🚫  Doesn't fit your GPU (override)", "🚫"),
    )

    # Bucket → section_id (the "category_bucket" value lives in perf_profile)
    _IMG_BUCKET_TO_SECTION = {
        "speed":   "img_speed",
        "quality": "img_quality",
        "photo":   "img_photo",
        "art":     "img_art",
        "general": "img_general",
    }

    # Stable sort order within a section: stronger recommendation first, then
    # quality desc, then speed asc, then name.
    _REC_RANK    = {"top_pick": 0, "recommended": 1, "alternative": 2, "legacy": 3}
    _QUAL_RANK   = {"sota": 0, "excellent": 1, "great": 2, "good": 3}
    _SPEED_RANK  = {"fast": 0, "balanced": 1, "slow": 2}

    def _active_device_vram_gb(self) -> int:
        """Return the device VRAM in GB used to compute fit, honoring the
        SKU filter when set, else the auto-detected local SKU. CPU SKUs and
        unknowns return 0 (only ``cpu_viable`` image-gen models are surfaced
        in that case).
        """
        try:
            filt = self._optional_filter_var.get() if hasattr(self, "_optional_filter_var") else self._sku_all_filter_value()
        except Exception:
            filt = self._sku_all_filter_value()
        if getattr(self, "_optional_skus_enabled", False) and filt and filt != self._sku_all_filter_value():
            from src import catalog as _cat
            return int(_cat.OPTIONAL_SKU_VRAM.get(filt, 0) or 0)
        try:
            return int((self._optional_sku or {}).get("vram_gb", 0) or 0)
        except Exception:
            return 0

    def _fit_tier_for_model(self, model: dict, vram_gb: int) -> str:
        """Classify model fit vs the active device VRAM.

        Returns one of: ``"fits_well"`` (≥ 1.5× headroom), ``"tight"``
        (fits but ≤ 1.5× headroom), ``"exceeds"`` (won't fit).
        """
        min_vram = int(model.get("min_vram_gb", 0) or 0)
        if vram_gb <= 0:
            # CPU mode: only cpu_viable models are even considered runnable.
            from src import catalog as _cat
            return "fits_well" if _cat.is_cpu_viable_image_model(model) else "exceeds"
        if min_vram > vram_gb:
            return "exceeds"
        if min_vram * 1.5 > vram_gb:
            return "tight"
        return "fits_well"

    def _model_sort_key(self, m: dict) -> tuple:
        """Stable sort key inside a section."""
        p = m.get("perf_profile") or {}
        return (
            self._REC_RANK.get(p.get("recommendation"), 9),
            self._QUAL_RANK.get(p.get("quality_tier"), 9),
            self._SPEED_RANK.get(p.get("speed_tier"), 9),
            str(m.get("name", "")).lower(),
        )

    def _build_image_gen_sections(
        self,
        image_models: list[dict],
        vram_gb: int,
        fit_by_id: dict[str, str] | None = None,
        show_oversize: bool | None = None,
        all_catalog: bool = False,
    ) -> list[tuple[str, list[dict]]]:
        """Group image-gen models into ordered (section_id, models) buckets.

        Algorithm:
        * Models that fit the device and have recommendation == "top_pick"
          go into ``img_top_picks``.
        * Remaining fitting models go into the bucket implied by their
          ``perf_profile.category_bucket``.
        * Models that don't fit go into ``img_doesnt_fit`` (only included
          when ``self._image_gen_show_oversize_var`` is True).
        * Empty sections are dropped.
        * Within each section, models are sorted by ``_model_sort_key``.
        """
        if show_oversize is None:
            show_oversize = bool(self._image_gen_show_oversize_var.get())
        sections: dict[str, list[dict]] = {sid: [] for sid, _, _ in self._IMG_SECTIONS}

        for m in image_models:
            if all_catalog:
                fit = "browse"
            elif fit_by_id is not None:
                fit = fit_by_id.get(m.get("id", ""), self._fit_tier_for_model(m, vram_gb))
            else:
                fit = self._fit_tier_for_model(m, vram_gb)
            if fit == "exceeds":
                if show_oversize:
                    sections["img_doesnt_fit"].append(m)
                continue
            p = m.get("perf_profile") or {}
            if p.get("recommendation") == "top_pick":
                sections["img_top_picks"].append(m)
                continue
            bucket = p.get("category_bucket") or "general"
            sid = self._IMG_BUCKET_TO_SECTION.get(bucket, "img_general")
            sections[sid].append(m)

        # Cap top picks at 8 — anything beyond is recommended quality; demote
        # excess to its natural bucket. Sorted by quality first.
        top = sections["img_top_picks"]
        if len(top) > 8:
            top.sort(key=self._model_sort_key)
            overflow = top[8:]
            sections["img_top_picks"] = top[:8]
            for m in overflow:
                p = m.get("perf_profile") or {}
                bucket = p.get("category_bucket") or "general"
                sid = self._IMG_BUCKET_TO_SECTION.get(bucket, "img_general")
                sections[sid].append(m)

        # Build ordered output, sorting within each section and dropping empties.
        out: list[tuple[str, list[dict]]] = []
        for sid, _title, _icon in self._IMG_SECTIONS:
            grp = sections.get(sid) or []
            if not grp:
                continue
            grp.sort(key=self._model_sort_key)
            out.append((sid, grp))
        return out

    def _build_perf_badge_specs(
        self, model: dict, vram_gb: int
    ) -> list[tuple[str, str, str, str]]:
        """Return a list of (text, fg_color, text_color, kind) tuples ready
        to be rendered as small badges on the model card. ``kind`` is one
        of ``rec``, ``speed``, ``quality``, ``bucket``, ``fit``, ``cpu``
        so the detail-pane renderer can route them onto two rows (rec/speed
        on top; quality/bucket/fit/cpu on the second row) — the rating
        dot badge ("●●●● Sota") is wide and never fits inline with the
        speed badge on the right detail card at 1280×800. Order matches
        prior layout for ResultsCard which keeps a single row.

        Only image-gen models with a ``perf_profile`` get badges; for
        others an empty list is returned so the caller can skip badge
        rendering.
        """
        p = model.get("perf_profile") or {}
        if not p:
            return []
        badges: list[tuple[str, str, str, str]] = []
        rec = p.get("recommendation")
        if rec == "top_pick":
            badges.append(("★ Top pick", "#caa31a", "#1a1300", "rec"))
        elif rec == "recommended":
            badges.append(("✓ Recommended", "#3a7a3a", "#ffffff", "rec"))
        elif rec == "legacy":
            badges.append(("Legacy", "#444",       "#bbb", "rec"))

        spd = p.get("speed_tier")
        if spd == "fast":
            badges.append((f"⚡ Fast · {p.get('speed_label', '~')}",     "#1f6f8f", "#e7f6ff", "speed"))
        elif spd == "balanced":
            badges.append((f"🕒 Balanced · {p.get('speed_label', '~')}", "#3a4d6b", "#e7eeff", "speed"))
        elif spd == "slow":
            badges.append((f"🐢 Heavy · {p.get('speed_label', '~')}",    "#5a3a6b", "#f1e7ff", "speed"))

        qual = p.get("quality_tier")
        if qual:
            # v5.5.4 a11y (P1-A): the previous "●●●● Sota" dot-rating string
            # was announced by NVDA/Narrator as "black circle black circle
            # black circle black circle Sota" because Unicode black-circle
            # bullets have no semantic role. The text fraction "(4/4)"
            # announces cleanly as "4 of 4" and visually still reads as a
            # quality score. Keep the badge color the same so the badge
            # still pops on the card.
            frac_map = {"good": "1/4", "great": "2/4", "excellent": "3/4", "sota": "4/4"}
            badges.append((f"({frac_map.get(qual, '?/4')}) {qual.title()}", "#2a2d3a", "#cfd8dc", "quality"))

        # v5.5.4 (SQT P3-1): emit ``fit`` BEFORE ``bucket`` so the detail
        # pane's row 2 reads quality → fit → bucket → cpu. Hardware
        # feasibility (fit) is more actionable than the category bucket
        # and should sit closer to the quality badge that anchors the row.
        # v5.5.4 (SQT P3-3): on CPU-only systems (vram_gb == 0) image
        # models can never actually run, so the "🔴 Exceeds VRAM" fit
        # badge adds noise to a row that already has a CPU OK pill telling
        # the user the truth. Suppress the fit badge in that exact case;
        # other (non-image, CPU-only) models still get the fit signal so
        # text models on small boxes keep the same UX.
        fit = self._fit_tier_for_model(model, vram_gb)
        is_image_ui = self._is_image_model_ui(model)
        suppress_fit = (vram_gb <= 0 and is_image_ui)
        if not suppress_fit:
            if fit == "fits_well":
                badges.append(("💚 Fits well", "#1f5a1f", "#dbf3db", "fit"))
            elif fit == "tight":
                badges.append(("🟡 Tight fit", "#8f6b1f", "#ffffff", "fit"))
            else:  # exceeds
                badges.append(("🔴 Exceeds VRAM", "#7a2a2a", "#ffd5d5", "fit"))

        bucket = p.get("category_bucket")
        bucket_pill = {
            "speed":   ("⚡ Speed",   "#2a4d6b"),
            "quality": ("🏆 Quality", "#5a3a6b"),
            "photo":   ("📸 Photo",   "#3a5a3a"),
            "art":     ("🎨 Art",     "#6b3a6b"),
            "general": ("🧰 General", "#4a4a4a"),
        }.get(bucket or "")
        if bucket_pill:
            badges.append((bucket_pill[0], bucket_pill[1], "#e8e8e8", "bucket"))

        if catalog.is_cpu_viable_image_model(model):
            label = model.get("expected_cpu_time_label") or "CPU OK"
            badges.append((f"CPU OK · {label}", "#5c4a1f", "#fff2c6", "cpu"))

        return badges

    def _ensure_section_header(self, section_id: str, title: str,
                               count: int) -> "ctk.CTkFrame":
        """Lazy-create or update a collapsible section header. The header
        widget is grid()'d by the caller into _cards_scroll. Click on the
        title row toggles ``self._section_collapsed[section_id]``.

        DESIGN (post-v5.3.6): borderless / transparent header with an
        accent chevron + bold title + muted count + a thin BORDER_STRONG
        bottom rule. The header reads as a section divider that *leads*
        the model rows below rather than a competing widget. This also
        sidesteps a paint bug described below.

        PAINT-BUG NOTE: the pre-v5.3.7 version used a solid two-tuple
        ``fg_color`` on a rounded ``CTkFrame`` inside ``self._cards_scroll``
        (a ``CTkScrollableFrame``). On the first Home → Models navigation
        the bar would render as a black rectangle until the next paint
        cycle, because CTkFrame's internal canvas resolves
        ``bg_color="transparent"`` against its parent at draw time and
        the scrollable frame's inner canvas had not yet propagated its
        appearance mode to the brand-new child widget. Switching to a
        transparent header with explicit theme-token children means
        there is no ``fg_color`` tuple to mispaint — the section divider
        is now built from labels + a 1-px ``BORDER_STRONG`` rule frame,
        so even on the very first paint there is no solid fill that
        could resolve to black. Do not reintroduce a solid ``fg_color``
        fill on this widget without re-validating the Home → Models
        transition in dark mode.
        """
        hdr = self._section_headers.get(section_id)
        if hdr is None:
            # Transparent shell — no fg_color tuple to mispaint on first draw.
            hdr = ctk.CTkFrame(self._cards_scroll, fg_color="transparent",
                               corner_radius=0)
            hdr.grid_columnconfigure(1, weight=1)
            chev = ctk.CTkLabel(hdr, text="▼", font=ctk.CTkFont(size=12, weight="bold"),
                                text_color=LINK_TEXT, width=22)
            chev.grid(row=0, column=0, sticky="w", padx=(10, 4), pady=(8, 4))
            title_lbl = ctk.CTkLabel(hdr, text=title,
                                     font=ctk.CTkFont(size=13, weight="bold"),
                                     anchor="w",
                                     text_color=TEXT_PRIMARY)
            title_lbl.grid(row=0, column=1, sticky="ew", pady=(8, 4))
            count_lbl = ctk.CTkLabel(hdr, text=f"{count} model{'s' if count!=1 else ''}",
                                     font=ctk.CTkFont(size=11),
                                     text_color=TEXT_MUTED)
            count_lbl.grid(row=0, column=2, sticky="e", padx=(0, 12), pady=(8, 4))
            # Thin bottom rule provides the visual separation a solid bar
            # used to provide, without the paint-bug risk.
            rule = ctk.CTkFrame(hdr, height=1, fg_color=BORDER_STRONG,
                                corner_radius=0)
            rule.grid(row=1, column=0, columnspan=3, sticky="ew",
                      padx=10, pady=(0, 2))
            hdr._chev_lbl  = chev
            hdr._title_lbl = title_lbl
            hdr._count_lbl = count_lbl
            hdr._rule_frame = rule

            def _toggle(_e=None, sid=section_id):
                self._section_collapsed[sid] = not self._section_collapsed.get(sid, False)
                self._schedule_model_card_population()
            for w in (hdr, chev, title_lbl, count_lbl, rule):
                w.bind("<Button-1>", _toggle)

            self._section_headers[section_id] = hdr
        else:
            hdr._title_lbl.configure(text=title)
            hdr._count_lbl.configure(text=f"{count} model{'s' if count!=1 else ''}")
        # Update chevron based on collapsed state
        collapsed = bool(self._section_collapsed.get(section_id, False))
        hdr._chev_lbl.configure(text="▶" if collapsed else "▼")
        return hdr

    def _ensure_img_summary_banner(self, parent) -> None:
        """Lazy-create the image-gen GPU summary banner inside the filter
        panel. Idempotent — safe to call from _populate_model_cards.
        """
        if self._img_summary_banner is not None:
            return
        b = ctk.CTkFrame(parent, fg_color=("#e7f0ff", "#1c2230"), corner_radius=6)
        b.grid_columnconfigure(0, weight=1)
        lbl = ctk.CTkLabel(b, text="", font=ctk.CTkFont(size=11),
                           anchor="w", justify="left",
                           text_color=TEXT_PRIMARY)
        lbl.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 2))
        sw = ctk.CTkSwitch(b, text="Show models that do not fit this selection",
                           variable=self._image_gen_show_oversize_var,
                           command=self._schedule_model_card_population,
                           switch_width=36, switch_height=18,
                           font=ctk.CTkFont(size=11))
        sw.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))
        self._img_summary_banner = b
        self._img_summary_banner_lbl = lbl
        self._img_summary_banner_switch = sw

    def _update_img_summary_banner(self, image_models_all: list[dict],
                                   image_models_fits: list[dict],
                                   vram_gb: int) -> None:
        """Refresh the banner text. Hides the banner when no image-gen
        models are eligible to show at all.
        """
        lbl = self._img_summary_banner_lbl
        if lbl is None:
            return
        total = len(image_models_all)
        fits  = len(image_models_fits)
        if total == 0:
            return
        gpu_name = (self.gpu_info.device_name
                    or self.gpu_info.gpu_type or "device").strip()
        if vram_gb > 0:
            head = f"Your GPU: {gpu_name} · {vram_gb} GB VRAM"
        else:
            head = f"This device: {gpu_name} · CPU mode"
        lbl.configure(text=f"{head}   ·   {fits} of {total} image-gen models fit your hardware.")

    # ── Models page ───────────────────────────────────────────────────────────

    def _build_models_page(self):
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["models"] = page
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)

        # Header row
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 4))
        hdr.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            hdr, text="Available Models",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            hdr, text="LocalAI Studio  —  Ron Martinsen  —  March 2026",
            font=ctk.CTkFont(size=10), text_color=TEXT_MUTED, anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 2))

        ctk.CTkButton(
            hdr, text="Model Guide", width=100,
            **self._outline_button_style(),
            command=self._open_model_guide,
        ).grid(row=0, column=2, padx=(8, 0))

        ctk.CTkButton(
            hdr, text="Image Gen Guide", width=120,
            **self._outline_button_style(),
            command=self._open_image_gen_guide,
        ).grid(row=0, column=3, padx=(8, 0))

        # v5.4: Refresh demoted to outline; the new + Add from Hugging Face
        # button is the primary action because it's the only header control
        # that *adds* a model rather than re-reading what's already there.
        self._refresh_btn = ctk.CTkButton(
            hdr, text="Refresh", width=80,
            **self._outline_button_style(),
            command=self._refresh_model_cards,
        )
        self._refresh_btn.grid(row=0, column=4, padx=(8, 0))

        ctk.CTkButton(
            hdr, text="+ Add from Hugging Face", width=180,
            command=self._open_add_from_hf_dialog,
            **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
        ).grid(row=0, column=5, padx=(8, 0))

        # Device/SKU banner + filters
        filter_panel = ctk.CTkFrame(page, corner_radius=8)
        filter_panel.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        filter_panel.grid_columnconfigure(0, weight=1)
        # v5.3: keep a reference so _populate_model_cards can lazy-create
        # the image-gen GPU summary banner inside this panel.
        self._filter_panel_ref = filter_panel

        # Device detection banner
        banner_row = ctk.CTkFrame(filter_panel, fg_color="transparent")
        banner_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        banner_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            banner_row, text=self._sku_text("banner_label", "This Device:"),
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        self._device_banner_label = ctk.CTkLabel(
            banner_row, text=self._device_display_text,
            font=ctk.CTkFont(size=12),
            text_color=self._device_display_color,
        )
        self._device_banner_label.grid(row=0, column=1, sticky="w")

        # Device/SKU filter row
        sku_row = ctk.CTkFrame(filter_panel, fg_color="transparent")
        sku_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 4))

        ctk.CTkLabel(
            sku_row, text=self._sku_text("filter_label", "Showing models for:"),
            font=ctk.CTkFont(size=11), text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 8))

        self._optional_filter_var.set(self._optional_filter_var.get())  # noop, ensures init
        sku_values = self._sku_filter_values()
        self._optional_sku_segbtn = ctk.CTkSegmentedButton(
            sku_row, values=sku_values, variable=self._optional_filter_var,
            command=self._apply_filter,
        )
        self._optional_sku_segbtn.pack(side="left")
        self._model_compat_summary_label = ctk.CTkLabel(
            sku_row,
            text="Calculating supported models...",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        self._model_compat_summary_label.pack(side="left", padx=(14, 0), fill="x", expand=True)

        # Type filter row. The model list is now master-detail: compact rows on
        # the left, one selected-model detail/action pane on the right.
        type_row = ctk.CTkFrame(filter_panel, fg_color="transparent")
        type_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 4))

        ctk.CTkLabel(
            type_row, text="Type:",
            font=ctk.CTkFont(size=11), text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 8))

        self._selected_cats.clear()
        self._user_set_cat_filter = False
        self._cat_btns.clear()
        self._type_btns.clear()
        _type_widths = {
            "All types": 78, "Chat": 58, "Vision": 64, "Image Gen": 88,
            "Toolbox": 76, "Embeddings": 96, "Speech": 68, "Document AI": 100,
        }
        for label in ["All types", "Chat", "Vision", "Image Gen", "Toolbox", "Embeddings", "Speech", "Document AI"]:
            btn = ctk.CTkButton(
                type_row, text=label, height=28,
                width=_type_widths.get(label, 90),
                font=ctk.CTkFont(size=11),
                fg_color=BUTTON_SECONDARY if label == self._model_type_filter_var.get() else INPUT_SURFACE,
                text_color=TEXT_PRIMARY,
                hover_color=BUTTON_SECONDARY_HOVER,
                border_width=1,
                border_color=BORDER_STRONG,
                corner_radius=6,
                command=lambda value=label: self._set_model_type_filter(value),
            )
            btn.pack(side="left", padx=1)
            self._type_btns[label] = btn
        self._cat_btns = self._type_btns

        # Wire Left/Right/Home/End across the type filter row so a user can
        # walk the filters after one Tab into the row, matching the WAI-ARIA
        # toolbar pattern. Tab still escapes to the next row.
        try:
            a11y.wire_arrow_navigation(
                list(self._type_btns.values()),
                orientation="horizontal",
            )
        except Exception:
            pass

        size_row = ctk.CTkFrame(filter_panel, fg_color="transparent")
        size_row.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 8))

        ctk.CTkLabel(
            size_row, text="Size:",
            font=ctk.CTkFont(size=11), text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 8))
        self._size_btns.clear()
        _size_widths = {
            "All sizes": 78, "Ultra Small": 90, "Small": 62, "Medium": 72,
            "Large": 62, "Extra Large": 90,
        }
        for label in ["All sizes", "Ultra Small", "Small", "Medium", "Large", "Extra Large"]:
            btn = ctk.CTkButton(
                size_row, text=label, height=28,
                width=_size_widths.get(label, 78),
                font=ctk.CTkFont(size=11),
                fg_color=BUTTON_SECONDARY if label == self._model_size_filter_var.get() else INPUT_SURFACE,
                text_color=TEXT_PRIMARY,
                hover_color=BUTTON_SECONDARY_HOVER,
                border_width=1,
                border_color=BORDER_STRONG,
                corner_radius=6,
                command=lambda value=label: self._set_model_size_filter(value),
            )
            btn.pack(side="left", padx=1)
            self._size_btns[label] = btn

        # Wire Left/Right/Home/End across the size filter row (WAI-ARIA
        # toolbar pattern, matches the type row above).
        try:
            a11y.wire_arrow_navigation(
                list(self._size_btns.values()),
                orientation="horizontal",
            )
        except Exception:
            pass

        # v5: Model search box — substring filter on name / id / tags / vendor.
        # Empty string disables the filter. The _populate_model_cards function
        # already honours self._model_search_var via getattr(), so just wiring
        # the widget here is enough.
        search_row = ctk.CTkFrame(filter_panel, fg_color="transparent")
        search_row.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkLabel(
            search_row, text="Search:",
            font=ctk.CTkFont(size=11), text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 8))
        search_entry = ctk.CTkEntry(
            search_row, textvariable=self._model_search_var,
            placeholder_text="Search supported models by name, id, vendor, or tag ...",
            placeholder_text_color=TEXT_MUTED,
            width=420,
        )
        search_entry.pack(side="left", padx=(0, 6))

        # Debounced refresh: only refire after the user stops typing for 200 ms.
        self._search_after_id: Optional[str] = None
        def _on_search_change(*_):
            if self._search_after_id is not None:
                try: self.after_cancel(self._search_after_id)
                except Exception: pass
            self._search_after_id = self.after(200, self._schedule_model_card_population)
        self._model_search_var.trace_add("write", _on_search_change)

        ctk.CTkButton(
            search_row, text="✕", width=28, height=28,
            font=ctk.CTkFont(size=12),
            **self._outline_button_style(),
            command=lambda: self._model_search_var.set(""),
        ).pack(side="left")

        results = ctk.CTkFrame(page, fg_color="transparent")
        results.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 12))
        results.grid_rowconfigure(1, weight=1)
        results.grid_columnconfigure(0, weight=0, minsize=self._models_list_width)
        results.grid_columnconfigure(1, weight=0, minsize=10)
        results.grid_columnconfigure(2, weight=1, minsize=360)
        self._models_results_frame = results

        self._model_results_summary_label = ctk.CTkLabel(
            results,
            text="Loading model list ...",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        self._model_results_summary_label.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        list_panel = _make_bordered_card(
            results,
            fg_color=SURFACE_CARD,
            border_color=BORDER_STRONG,
            border_width=1,
            corner_radius=10,
        )
        list_panel.configure(width=self._models_list_width)
        list_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        list_panel.grid_propagate(False)
        list_panel_inner = list_panel.content_frame
        list_panel_inner.grid_rowconfigure(2, weight=1)
        list_panel_inner.grid_columnconfigure(0, weight=1)
        self._models_list_panel = list_panel
        ctk.CTkLabel(
            list_panel_inner,
            text="Model list",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))
        header_row = ctk.CTkFrame(list_panel_inner, fg_color="transparent")
        header_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        header_row.grid_columnconfigure(0, weight=1)
        header_row.grid_columnconfigure(1, minsize=96)
        header_row.grid_columnconfigure(2, minsize=86)
        header_row.grid_columnconfigure(3, minsize=72)
        header_row.grid_columnconfigure(4, minsize=92)
        for col, (label, anchor, padx) in enumerate((
            ("Name", "w", (10, 6)),
            ("Type", "w", 4),
            ("Fit", "w", 4),
            ("Size", "e", 4),
            ("Status", "e", (4, 10)),
        )):
            ctk.CTkLabel(
                header_row,
                text=label,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=TEXT_MUTED,
                anchor=anchor,
            ).grid(row=0, column=col, sticky="ew", padx=padx)

        self._cards_scroll = ctk.CTkScrollableFrame(list_panel_inner, fg_color=SURFACE_INNER)
        self._cards_scroll.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._cards_scroll.grid_columnconfigure(0, weight=1)

        splitter = ctk.CTkFrame(results, width=8, fg_color=BORDER_STRONG, cursor="sb_h_double_arrow")
        splitter.grid(row=1, column=1, sticky="ns", padx=(0, 0), pady=4)
        splitter.bind("<ButtonPress-1>", self._start_models_list_resize)
        splitter.bind("<B1-Motion>", self._drag_models_list_resize)
        ctk.CTkLabel(splitter, text="⋮", text_color=TEXT_MUTED, width=8).place(relx=0.5, rely=0.5, anchor="center")

        # v5.5.19 UX fix: native ``border_width`` on CTkFrame draws the border
        # as part of the rounded-rect Canvas path. At many widget sizes the
        # straight edge segments fail to render reliably (top/bottom/side
        # borders disappear while only the corners survive). This regressed
        # repeatedly through v5.5.0 / v5.5.1 / v5.5.4 fix attempts. The
        # canonical fix is ``_make_bordered_card``: nest two solid filled
        # frames so the visible border is just the outer color showing
        # through a pack-padding gap. No rounded-rect border math, no paint
        # races, all four sides + corners render reliably in both themes.
        detail_panel = _make_bordered_card(
            results,
            fg_color=SURFACE_CARD,
            border_color=BORDER_STRONG,
            border_width=2,
            corner_radius=10,
        )
        detail_panel.grid(row=1, column=2, sticky="nsew", padx=(8, 0), pady=(0, 16))
        detail_panel_inner = detail_panel.content_frame
        detail_panel_inner.grid_columnconfigure(0, weight=1)
        detail_panel_inner.grid_rowconfigure(0, weight=1)
        self._build_model_detail_pane(detail_panel_inner)

        self._cards_empty_label = ctk.CTkLabel(
            self._cards_scroll,
            text="Loading model list ...",
            font=ctk.CTkFont(size=13), text_color=TEXT_MUTED,
        )
        self._cards_empty_label.grid(row=0, column=0, pady=40)
        self._models_page_just_built = True
        self._schedule_model_card_population()

    def _start_models_list_resize(self, event) -> None:
        self._models_list_resize_start_x = event.x_root
        self._models_list_resize_start_width = int(getattr(self, "_models_list_width", 680))

    def _drag_models_list_resize(self, event) -> None:
        start_x = int(getattr(self, "_models_list_resize_start_x", event.x_root))
        start_width = int(getattr(self, "_models_list_resize_start_width", getattr(self, "_models_list_width", 680)))
        width = start_width + int(event.x_root - start_x)
        total_width = int(getattr(self, "_models_results_frame", self).winfo_width() or self.winfo_width() or 1100)
        max_width = max(520, total_width - 430)
        width = max(500, min(width, max_width))
        self._models_list_width = width
        results = getattr(self, "_models_results_frame", None)
        if results is not None:
            results.grid_columnconfigure(0, minsize=width)
        panel = getattr(self, "_models_list_panel", None)
        if panel is not None:
            panel.configure(width=width)

    def _build_model_detail_pane(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        # Row layout (v5.5.0 UX fix):
        #   0  title           PINNED (always visible)
        #   1  meta            PINNED
        #   2  status          PINNED
        #   3  actions         PINNED (Install / Open / Prompt ideas / Learn More / Delete)
        #   4  scroll_frame    EXPANDS — wraps every description-class widget
        #                       so long image-gen descriptions (e.g. SDXL
        #                       Lightning) no longer clip off the bottom.
        # Previously rows 4–9 lived directly on ``parent`` (non-scrollable),
        # which pushed the recommendation list off-screen at 1280×800. Per
        # a11y review: the action row stays pinned so primary affordances
        # are reachable without scrolling.
        parent.grid_rowconfigure(4, weight=1)
        self._model_detail_widgets = {}

        title = ctk.CTkLabel(
            parent,
            text="Select a model",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
            justify="left",
            wraplength=240,
        )
        title.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 2))
        meta = ctk.CTkLabel(parent, text="", text_color=TEXT_MUTED, anchor="w", justify="left", wraplength=240)
        meta.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        status = ctk.CTkLabel(
            parent,
            text="Pick a row in the model list to see requirements, recommended settings, and actions.",
            text_color=TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=240,
        )
        status.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))

        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))
        actions.grid_columnconfigure((0, 1), weight=1)

        install = ctk.CTkButton(actions, text="Install", command=self._install_selected_model, **self._outline_button_style())
        install.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=3)
        primary = ctk.CTkButton(
            actions,
            text="Open",
            command=self._run_selected_model_primary_action,
            **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
        )
        primary.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=3)
        ideas = ctk.CTkButton(actions, text="Prompt ideas", command=self._open_selected_model_ideas, **self._outline_button_style())
        ideas.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=3)
        learn = ctk.CTkButton(actions, text="Learn More", command=self._open_selected_model_learn_more, **self._outline_button_style())
        learn.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=3)
        delete = ctk.CTkButton(
            actions,
            text="Delete Local Model",
            command=self._delete_selected_model,
            **self._solid_button_style(self._IG_DANGER, "#8a2424"),
        )
        delete.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 3))

        # ── Scrollable region for description/specs/settings/demo/recs ─────
        # Per macOS + perf review: pass ``fg_color=SURFACE_CARD`` explicitly
        # so the inner Canvas viewport doesn't briefly flash black on first
        # paint in dark theme (the macOS Aqua Tk + CTk appearance-mode
        # propagation gap). Without this it inherits from parent and may
        # show through the Canvas underlying widget for one frame.
        scroll_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color=SURFACE_CARD,
            corner_radius=0,
            border_width=0,
        )
        scroll_frame.grid(row=4, column=0, sticky="nsew", padx=(2, 2), pady=(0, 10))
        scroll_frame.grid_columnconfigure(0, weight=1)
        self._model_detail_scroll = scroll_frame

        # v5.5.18 follow-up UX fix (Ron, 2026-05-30): scroll_frame inner
        # widgets now use padx=(14, 22) — extra 8 px on the right reserves
        # a visible gutter between the content and the scrollbar so the
        # rightmost badge ("Photo") and long description text no longer
        # collide with / render under the vertical scrollbar. The matching
        # gutter constant in _update_wraplength was bumped from 28 to 40
        # (14 left + 22 right + ~4 buffer) so wraplength stays in sync.
        # Initial wraplength was also tightened from 340 to 240 so the
        # first paint never over-wraps before _update_wraplength runs.
        badges = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        badges.grid(row=0, column=0, sticky="ew", padx=(14, 22), pady=(0, 8))

        desc = ctk.CTkLabel(scroll_frame, text="", text_color=TEXT_SECONDARY, anchor="w", justify="left", wraplength=240)
        desc.grid(row=1, column=0, sticky="ew", padx=(14, 22), pady=(0, 8))

        specs = ctk.CTkLabel(scroll_frame, text="", text_color=TEXT_MUTED, anchor="w", justify="left", wraplength=240)
        specs.grid(row=2, column=0, sticky="ew", padx=(14, 22), pady=(0, 8))

        settings = ctk.CTkLabel(scroll_frame, text="", text_color=TEXT_SECONDARY, anchor="w", justify="left", wraplength=240)
        settings.grid(row=3, column=0, sticky="ew", padx=(14, 22), pady=(0, 8))

        demo = ctk.CTkLabel(scroll_frame, text="", text_color=TEXT_MUTED, anchor="w", justify="left", wraplength=240)
        demo.grid(row=4, column=0, sticky="ew", padx=(14, 22), pady=(0, 8))

        recs = ctk.CTkLabel(scroll_frame, text="", text_color=LINK_TEXT, anchor="w", justify="left", wraplength=240)
        recs.grid(row=5, column=0, sticky="ew", padx=(14, 22), pady=(0, 10))

        # User-added affordances. Hidden by default — _update_model_detail
        # only grids them when the selected model has ``user_added=true``.
        # Banner sits above the source URL; Remove sits below.
        review_banner = ctk.CTkLabel(
            scroll_frame, text="", anchor="w", justify="left",
            wraplength=240, text_color=WARN_TEXT,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        source_label = ctk.CTkLabel(
            scroll_frame, text="", anchor="w", justify="left",
            wraplength=240, text_color=LINK_TEXT, cursor="hand2",
            font=ctk.CTkFont(size=11, underline=True),
        )
        remove_user_btn = ctk.CTkButton(
            scroll_frame, text="Remove from catalog",
            command=self._remove_selected_user_model,
            **self._outline_button_style(text_color=ERROR_TEXT),
        )
        # Click-to-open binding for the source URL row; the actual URL is
        # set in _update_model_detail so the lambda always reads the
        # current selection rather than a stale closure.
        source_label.bind(
            "<Button-1>",
            lambda _e: self._open_selected_model_source_url(),
        )

        # v5.5.3 perf: coalesce duplicate ``<Configure>`` events into a
        # single ``_apply`` call. The binding fires on BOTH ``parent`` and
        # ``scroll_frame`` and Tk propagates Configure events to children,
        # so a single resize tick can schedule two ``_apply`` callbacks at
        # 60 fps. Tracking + cancelling the prior pending callback halves
        # the per-frame work without changing the visible behaviour.
        _wl_pending: list = [None]

        def _update_wraplength(event=None):
            # v5.5.1 UX fix: at first paint, the inner ``_scrollable_frame``
            # may report its *content* width (= widest child, which can be
            # much larger than the visible canvas), which makes wraplength
            # too big and the text clips on the right edge. Take the
            # minimum of (parent grid column, scroll canvas viewport,
            # inner content frame) and subtract the scrollbar+padding
            # gutter so wraplength tracks the *visible* drawable width.
            #
            # v5.5.18 follow-up UX fix (Ron, 2026-05-30): the prior
            # ``> 50`` threshold treated any widget whose winfo_width()
            # came back < 51 (e.g. partially realized at 20-40 px during
            # a splitter drag, or a stale 1 from the very first paint) as
            # "unknown" and fell back to the 320 default — which then
            # exceeded the real viewport on narrow panes and the text +
            # rightmost badge clipped under the scrollbar. Lowered to
            # ``> 1`` so any realized width counts; added an idletasks
            # refresh + a one-shot retry when nothing comes back so the
            # 320 default branch is only hit on a truly unmounted pane.
            def _apply(retried: bool = False):
                _wl_pending[0] = None
                try:
                    scroll_frame.update_idletasks()
                except Exception:
                    pass
                inner = getattr(scroll_frame, "_scrollable_frame", scroll_frame)
                # v5.5.3 (SQT P2): ``_parent_canvas`` is undocumented CTk
                # internal. Add a ``canvas`` fallback in case CTk renames
                # the attr; falling back to ``scroll_frame`` would include
                # the scrollbar gutter in the width and under-wrap text.
                canvas = (
                    getattr(scroll_frame, "_parent_canvas", None)
                    or getattr(scroll_frame, "canvas", None)
                    or scroll_frame
                )
                candidates: list[int] = []
                for widget in (parent, canvas, inner):
                    try:
                        w = int(widget.winfo_width() or 0)
                    except Exception:
                        w = 0
                    if w > 1:
                        candidates.append(w)
                if candidates:
                    # v5.5.3 a11y (P2-B): the gutter is a pixel literal;
                    # under Windows 125 %/150 % display scaling CTk reports
                    # DPI-aware widget widths but a hardcoded gutter
                    # under-reserves space and text clips. Scale the
                    # gutter by the same factor CTk uses for widget sizing.
                    #
                    # v5.5.18 follow-up UX fix (Ron, 2026-05-30): bumped
                    # the base gutter from 28 to 40 px to match the new
                    # ``padx=(14, 22)`` on scroll_frame inner widgets
                    # (14 left + 22 right + ~4 buffer for the scrollbar
                    # column / sub-pixel rounding). The old 28 px only
                    # covered the symmetric 14+14 padding and left zero
                    # room for the scrollbar gutter, so on narrow panes
                    # text wrapped right at the scrollbar's left edge and
                    # the rightmost glyphs were clipped/hidden under it.
                    #
                    # v5.5.18 follow-up #2 (Ron, 2026-05-30): the prior
                    # version of this code was DOUBLE-SCALING wraplength.
                    # ``winfo_width()`` returns DEVICE pixels (already
                    # DPI-scaled), but CTkLabel internally multiplies the
                    # ``wraplength=`` kwarg by the widget-scaling factor
                    # again (ctk_label.py:96). On a 125 % display that
                    # meant wraplength=343 became Tk wraplength=429 — but
                    # the widget cell was only 348 device px wide because
                    # ``padx=(14, 22)`` (logical) takes 45 device px. Text
                    # therefore overflowed the widget bbox by ~80 device px
                    # and clipped on the right. The fix is to convert the
                    # device-pixel canvas width back into LOGICAL pixels
                    # (divide by the scale), then subtract a LOGICAL
                    # gutter (40 ≈ 14 + 22 + ~4 buffer). CTk re-scales the
                    # result so the final Tk wraplength matches the
                    # widget's device-pixel allocation.
                    try:
                        scale = float(ctk.ScalingTracker.get_widget_scaling(scroll_frame))
                    except Exception:
                        scale = 1.0
                    if scale <= 0:
                        scale = 1.0
                    gutter = 40  # LOGICAL px: 14 left padx + 22 right padx + ~4 buffer
                    canvas_logical = min(candidates) / scale
                    width = max(180, int(canvas_logical) - gutter)
                else:
                    # Nothing was realized — schedule one retry after a
                    # paint cycle so the 320 fallback (which can be wider
                    # than the actual viewport) is only used as a true
                    # last resort.
                    if not retried:
                        try:
                            _wl_pending[0] = self.after(32, lambda: _apply(retried=True))
                            return
                        except Exception:
                            pass
                    width = 320
                # Pinned-row widgets (title/meta/status) live directly on
                # ``parent`` and don't have the scrollbar gutter, but
                # using the same visible width keeps line breaks aligned
                # with the scrollable body below.
                for widget in (title, meta, status):
                    try:
                        widget.configure(wraplength=width)
                    except Exception:
                        pass
                for widget in (desc, specs, settings, demo, recs, review_banner, source_label):
                    try:
                        widget.configure(wraplength=width)
                    except Exception:
                        pass
            prior = _wl_pending[0]
            if prior is not None:
                try:
                    self.after_cancel(prior)
                except Exception:
                    pass
                _wl_pending[0] = None
            try:
                _wl_pending[0] = self.after(0, _apply)
            except Exception:
                _apply()
        parent.bind("<Configure>", _update_wraplength)
        scroll_frame.bind("<Configure>", _update_wraplength)

        # a11y review: bind <FocusIn> on focusable children so keyboard
        # users get auto-scroll-to-view when Tab navigates into the
        # scrollable region. Without this, Tab can land on a widget below
        # the visible viewport and the user perceives focus as vanished.
        #
        # NVDA caveat: browse-mode's virtual buffer may not enumerate
        # CTkScrollableFrame deep contents — keyboard Tab (forms mode)
        # works because of this binding, but SR users in browse mode may
        # need NVDA object navigation (Numpad) or JAWS virtual-cursor
        # tricks to discover the inner widgets. This is a CTk/Tk
        # limitation; the focus-driven scroll here is the best fix the
        # framework allows.
        def _scroll_widget_into_view(widget) -> None:
            try:
                if not (widget and widget.winfo_exists()):
                    return
                scroll_frame.update_idletasks()
                inner = getattr(scroll_frame, "_scrollable_frame", scroll_frame)
                total_h = max(1, int(inner.winfo_reqheight() or 1))
                widget_top = int(widget.winfo_rooty() - inner.winfo_rooty())
                scroll_frame._parent_canvas.yview_moveto(max(0.0, widget_top / total_h))
            except Exception:
                pass
        for focusable in (review_banner, source_label, remove_user_btn):
            try:
                focusable.bind(
                    "<FocusIn>",
                    lambda _e, w=focusable: _scroll_widget_into_view(w),
                )
            except Exception:
                pass

        self._model_detail_widgets = {
            "title": title,
            "meta": meta,
            "status": status,
            "badges": badges,
            "desc": desc,
            "specs": specs,
            "settings": settings,
            "demo": demo,
            "recs": recs,
            "install": install,
            "primary": primary,
            "ideas": ideas,
            "learn": learn,
            "delete": delete,
            "review_banner": review_banner,
            "source_label": source_label,
            "remove_user_btn": remove_user_btn,
        }
        self._update_model_detail()

    def _schedule_model_card_population(self, delay_ms: int = 1) -> None:
        """Let the Models page paint before rendering the full catalog."""
        if not hasattr(self, "_cards_scroll"):
            return
        self._model_population_token += 1
        token = self._model_population_token
        self.after(delay_ms, lambda t=token: self._populate_model_cards(t))

    def _cached_local_names_snapshot(self) -> set:
        cache = getattr(self, "_local_names_cache", None)
        if not cache:
            return set()
        cached_at, names = cache
        if time.time() - cached_at < 5.0:
            return set(names)
        return set()

    def _cached_comfyui_model_names_snapshot(self) -> set:
        cache = getattr(self, "_comfyui_model_names_cache", None)
        if not cache:
            return set()
        cached_at, names = cache
        if time.time() - cached_at < 5.0:
            return set(names)
        return set()

    def _current_model_capacity(self) -> dict:
        filt = self._optional_filter_var.get()
        all_value = self._sku_all_filter_value()
        if filt == all_value:
            return {"label": all_value, "ram_gb": 0, "vram_gb": 0, "all_catalog": True}
        if self._optional_skus_enabled:
            sku = next((s for s in self._optional_skus if s.get("name") == filt), None) or {}
            return {
                "label": filt,
                "ram_gb": float(catalog.OPTIONAL_SKU_RAM.get(filt, sku.get("ram_gb", 0)) or 0),
                "vram_gb": float(catalog.OPTIONAL_SKU_VRAM.get(filt, sku.get("vram_gb", 0)) or 0),
                "all_catalog": False,
            }
        dev = self._optional_sku or {}
        return {
            "label": filt or "This Device",
            "ram_gb": float(dev.get("ram_gb", 0) or system_info.get_ram_info().get("total_gb", 0) or 0),
            "vram_gb": float(dev.get("vram_gb", 0) or self._active_device_vram_gb() or 0),
            "all_catalog": False,
        }

    @staticmethod
    def _is_image_model(model: dict) -> bool:
        return model.get("category") == "Image Generation" or model.get("backend") == "comfyui" or bool(model.get("comfyui_model"))

    @staticmethod
    def _is_size_category(category: str) -> bool:
        return category in {"Ultra Small", "Small", "Medium", "Large", "Extra Large"}

    def _is_utility_demo_model(self, model: dict) -> bool:
        return (
            not self._is_image_model(model)
            and not catalog.is_chat_selectable_model(model)
            and bool(model.get("hf_repo") or model.get("onnx_repo") or model.get("phase1_adapter"))
        )

    def _model_type_label(self, model: dict) -> str:
        category = str(model.get("category") or "Other")
        if self._is_image_model(model):
            return "Image Gen"
        if category == "Vision" or "vision" in (model.get("tags") or []):
            return "Vision"
        if category in {"Speech", "Embeddings", "Document AI"}:
            return category
        if self._is_utility_demo_model(model):
            return "Toolbox"
        if catalog.is_chat_selectable_model(model):
            return "Chat"
        return category

    def _model_matches_type_filter(self, model: dict) -> bool:
        selected = self._model_type_filter_var.get() if hasattr(self, "_model_type_filter_var") else "All types"
        if selected in ("", "All types"):
            return True
        label = self._model_type_label(model)
        if selected == "Toolbox":
            return self._is_utility_demo_model(model) or label in {"Speech", "Embeddings", "Document AI"}
        if selected == "Image Gen":
            return self._is_image_model(model)
        return label == selected

    def _model_matches_size_filter(self, model: dict) -> bool:
        selected = self._model_size_filter_var.get() if hasattr(self, "_model_size_filter_var") else "All sizes"
        if selected in ("", "All sizes"):
            return True
        return model.get("category") == selected

    def _model_fit_tier_for_capacity(self, model: dict, capacity: dict | None = None) -> str:
        capacity = capacity or self._current_model_capacity()
        if capacity.get("all_catalog"):
            return "browse"
        ram_gb = float(capacity.get("ram_gb") or 0)
        vram_gb = float(capacity.get("vram_gb") or 0)
        min_ram = float(model.get("min_ram_gb") or 0)
        min_vram = float(model.get("min_vram_gb") or 0)
        if self._is_image_model(model):
            if vram_gb <= 0:
                if catalog.is_cpu_viable_image_model(model) and (not min_ram or ram_gb >= min_ram):
                    return "tight"
                return "exceeds"
            if min_vram > vram_gb:
                return "exceeds"
            return "tight" if min_vram and min_vram * 1.5 > vram_gb else "fits_well"
        if min_ram and ram_gb and min_ram > ram_gb:
            return "exceeds"
        if min_vram and vram_gb and min_vram > vram_gb:
            return "exceeds"
        if min_vram and vram_gb and min_vram * 1.5 > vram_gb:
            return "tight"
        if min_ram and ram_gb and min_ram * 1.15 > ram_gb:
            return "tight"
        return "fits_well"

    def _model_fit_display(self, model: dict) -> tuple[str, object]:
        fit = self._model_row_fit_cache.get(model.get("id", "")) or self._model_fit_tier_for_capacity(model)
        if fit == "browse":
            return "Browse", TEXT_MUTED
        if fit == "fits_well":
            return "Fits", SUCCESS_TEXT
        if fit == "tight":
            return "Tight", WARN_TEXT
        return "Exceeds", ERROR_TEXT

    def _model_install_status(
        self,
        model: dict,
        local_names: set | None = None,
        comfyui_model_names: set | None = None,
    ) -> tuple[str, object, str]:
        if self._is_image_model(model):
            model_filename = model.get("comfyui_model", "")
            file_exists = False
            if model_filename and comfyui_model_names is not None:
                file_exists = model_filename in comfyui_model_names
            elif model_filename:
                comfyui_path = self._comfyui_installed_path()
                if comfyui_path:
                    for subdir in ("checkpoints", "diffusion_models"):
                        if (comfyui_path / "models" / subdir / model_filename).exists():
                            file_exists = True
                            break
            if file_exists:
                if self.comfyui_ok:
                    return "Installed", SUCCESS_TEXT, "installed"
                return "Installed", WARN_TEXT, "installed_offline"
            return ("Missing", WARN_TEXT, "missing") if self.comfyui_ok else ("ComfyUI off", TEXT_MUTED, "offline_missing")

        if self._is_utility_demo_model(model):
            return "Ready", SUCCESS_TEXT, "ready"

        tag = model.get("ollama_tag", "")
        if not tag:
            return "No tag", TEXT_MUTED, "unavailable"
        if local_names is not None:
            is_local = _ollama_tag_is_local(tag, local_names)
        else:
            cache = getattr(self, "_local_names_cache", None)
            if cache:
                cached_at, cached_names = cache
                if time.time() - cached_at < 5.0:
                    is_local = _ollama_tag_is_local(tag, set(cached_names))
                elif self.ollama_ok:
                    return "Checking", TEXT_MUTED, "checking"
                else:
                    return "Ollama off", TEXT_MUTED, "offline"
            elif self.ollama_ok:
                return "Checking", TEXT_MUTED, "checking"
            else:
                return "Ollama off", TEXT_MUTED, "offline"
        if is_local:
            return "Installed", SUCCESS_TEXT, "installed"
        return "Missing", TEXT_MUTED, "missing"

    def _model_row_sort_key(self, model: dict) -> tuple:
        fit = self._model_row_fit_cache.get(model.get("id", "")) or "fits_well"
        fit_rank = {"fits_well": 0, "tight": 1, "browse": 2, "exceeds": 3}.get(fit, 2)
        type_rank = {
            "Chat": 0, "Vision": 1, "Image Gen": 2, "Toolbox": 3,
            "Embeddings": 4, "Speech": 5, "Document AI": 6,
        }.get(self._model_type_label(model), 9)
        return (fit_rank, type_rank, str(model.get("name", "")).lower())

    def _refresh_model_filter_btn_states(self) -> None:
        selected_type = self._model_type_filter_var.get()
        for label, btn in getattr(self, "_type_btns", {}).items():
            btn.configure(fg_color=BUTTON_SECONDARY if label == selected_type else INPUT_SURFACE)
        selected_size = self._model_size_filter_var.get()
        for label, btn in getattr(self, "_size_btns", {}).items():
            btn.configure(fg_color=BUTTON_SECONDARY if label == selected_size else INPUT_SURFACE)

    def _set_model_type_filter(self, value: str) -> None:
        self._user_set_cat_filter = True
        self._model_type_filter_var.set(value)
        self._refresh_model_filter_btn_states()
        self._schedule_model_card_population()

    def _set_model_size_filter(self, value: str) -> None:
        self._user_set_cat_filter = True
        self._model_size_filter_var.set(value)
        self._refresh_model_filter_btn_states()
        self._schedule_model_card_population()

    def _selected_model(self) -> dict | None:
        mid = getattr(self, "_selected_model_id", None)
        if not mid:
            return None
        return next((m for m in self._catalog_models if m.get("id") == mid), None)

    def _select_model_row(self, model_id: str) -> None:
        if not model_id:
            return
        self._selected_model_id = model_id
        self._sync_model_row_selection()
        self._update_model_detail(
            local_names=self.__dict__.get("_model_detail_local_names", None),
            comfyui_model_names=self.__dict__.get("_model_detail_comfyui_model_names", None),
        )

    def _sync_model_row_selection(self) -> None:
        selected = getattr(self, "_selected_model_id", None)
        for mid, row in getattr(self, "_model_cards_by_id", {}).items():
            try:
                row.set_selected(mid == selected)
            except Exception:
                pass

    def _update_model_detail(
        self,
        local_names: set | None = None,
        comfyui_model_names: set | None = None,
    ) -> None:
        widgets = getattr(self, "_model_detail_widgets", {})
        if not widgets:
            return
        model = self._selected_model()
        buttons = ["install", "primary", "ideas", "learn", "delete"]
        if not model:
            widgets["title"].configure(text="Select a model")
            widgets["meta"].configure(text="")
            widgets["status"].configure(
                text="Pick a row in the model list to see requirements, recommended settings, and actions.",
                text_color=TEXT_SECONDARY,
            )
            widgets["desc"].configure(text="")
            widgets["specs"].configure(text="")
            widgets["settings"].configure(text="")
            widgets["demo"].configure(text="")
            widgets["recs"].configure(text="")
            for child in widgets["badges"].winfo_children():
                child.destroy()
            for key in buttons:
                widgets[key].configure(state="disabled")
            for opt_key in ("review_banner", "source_label", "remove_user_btn"):
                opt_widget = widgets.get(opt_key)
                if opt_widget is not None:
                    try:
                        opt_widget.grid_remove()
                    except Exception:
                        pass
            return

        fit_text, fit_color = self._model_fit_display(model)
        status_text, status_color, status_state = self._model_install_status(
            model,
            local_names=local_names,
            comfyui_model_names=comfyui_model_names,
        )
        model_type = self._model_type_label(model)
        widgets["title"].configure(text=model.get("name", model.get("id", "Model")))
        widgets["meta"].configure(
            text=f"{model.get('vendor', 'Unknown vendor')}  |  {model_type}  |  {model.get('parameters', 'params unknown')}"
        )
        status_detail = status_text
        if status_state == "installed":
            status_detail = "Installed - no install needed"
        elif self._is_image_model(model) and status_state == "installed_offline":
            status_detail = "Installed - no install needed; ComfyUI starts when you Generate"
        elif status_state == "checking":
            status_detail = "Checking install status..."
        elif status_state in {"missing", "offline_missing"}:
            status_detail = "Missing - install available" if model.get("ollama_tag") or model.get("comfyui_model") else status_text
        widgets["status"].configure(text=f"{fit_text} this selection  |  {status_detail}", text_color=fit_color if fit_text != "Browse" else status_color)
        widgets["desc"].configure(text=model.get("description", "No description available."))

        specs_lines = [
            f"{model.get('parameters', 'Params unknown')}",
            f"{model.get('size_gb', 0):g} GB disk",
            f"Min RAM: {model.get('min_ram_gb', 0)} GB",
            f"Min VRAM: {model.get('min_vram_gb', 0)} GB",
        ]
        if model.get("context_length"):
            specs_lines.append(f"Context: {int(model['context_length']):,}")
        if model.get("ollama_tag"):
            specs_lines.append(f"Ollama tag: {model['ollama_tag']}")
        if model.get("comfyui_model"):
            specs_lines.append(f"ComfyUI file: {model['comfyui_model']}")
        widgets["specs"].configure(text="  |  ".join(specs_lines))

        for child in widgets["badges"].winfo_children():
            child.destroy()
        if self._is_image_model(model):
            capacity = self._current_model_capacity()
            badge_vram = self._active_device_vram_gb() if capacity.get("all_catalog") else int(float(capacity.get("vram_gb") or 0))
            # v5.5.1 UX fix: split badges across two rows so the rating
            # dot badge ("●●●● Sota") never has to fit inline with the
            # speed pill on the right detail card at 1280×800. Row 1 =
            # recommendation + speed (the two most actionable signals);
            # row 2 = rating dots + bucket + fit pill + CPU OK. Both
            # rows are transparent CTkFrames packed inside the existing
            # ``widgets["badges"]`` container, so callers (and the
            # detach/destroy bookkeeping above) keep working.
            #
            # v5.5.19 follow-up (Ron, 2026-05-30): the CPU OK badge
            # ("CPU OK · ~90s on small CPU") is ~22 chars wide — longer
            # than any other chip — and when it shared row 2 with
            # quality + fit + bucket the four chips together overflowed
            # the ~240px wraplength target on narrow detail panes
            # (visible on SG161222 / realistic-vision-v6 model cards:
            # the trailing chip clipped to "·~90s o"). Promote ``cpu``
            # to its own row 3 so it always gets the full panel width
            # and never clips. Quality + fit + bucket (the shorter
            # rating/metadata chips) stay on row 2 where they fit. Row 3
            # is only packed when a cpu chip actually lands on it so
            # models without a CPU OK badge don't get an empty 2 px gap
            # below row 2.
            row1 = ctk.CTkFrame(widgets["badges"], fg_color="transparent")
            row1.pack(side="top", fill="x", anchor="w")
            row2 = ctk.CTkFrame(widgets["badges"], fg_color="transparent")
            row2.pack(side="top", fill="x", anchor="w", pady=(2, 0))
            row3 = ctk.CTkFrame(widgets["badges"], fg_color="transparent")
            row3_packed = False
            for text, fg, txt, kind in self._build_perf_badge_specs(model, int(badge_vram)):
                if kind in {"rec", "speed"}:
                    target = row1
                elif kind == "cpu":
                    target = row3
                    if not row3_packed:
                        row3.pack(side="top", fill="x", anchor="w", pady=(2, 0))
                        row3_packed = True
                else:
                    target = row2
                # v5.5.4 a11y (P2-C) / v5.5.12 Mac crash fix: hairline border
                # so the badge edge is distinguishable from the surrounding
                # card surface even when fg_color contrast is borderline
                # (e.g. ``quality`` #2a2d3a on SURFACE_CARD in dark mode
                # measured 1.03:1 before the border). v5.5.19: switched to
                # ``_make_chip`` so the border renders reliably on all four
                # sides at any DPI / widget size (CTk native ``border_width=1``
                # dropped top/bottom edges in Ron's bug report 2026-05-30).
                chip, _ = _make_chip(
                    target,
                    " " + text + " ",
                    fg_color=fg,
                    text_color=txt,
                    border_color=BORDER_STRONG,
                    font=ctk.CTkFont(size=10, weight="bold"),
                    corner_radius=8,
                    label_padx=4,
                    label_pady=0,
                )
                chip.pack(side="left", padx=(0, 4), pady=2)

        settings_text = ""
        if self._is_image_model(model):
            rec = model.get("recommended_settings") or {}
            if rec:
                settings_text = (
                    "Recommended image settings:\n"
                    f"{rec.get('width', model.get('default_width', 512))}x{rec.get('height', model.get('default_height', 512))}"
                    f"  |  {rec.get('steps', '?')} steps"
                    f"  |  CFG {rec.get('cfg', '?')}"
                    f"  |  {rec.get('sampler', '?')} / {rec.get('scheduler', '?')}"
                )
        elif self._is_utility_demo_model(model):
            demo = model_demos.get_model_demo(model)
            settings_text = f"Toolbox workflow: {demo.get('feature', 'local utility demo')}"
        widgets["settings"].configure(text=settings_text)

        demo = model_demos.get_model_demo(model)
        demo_text = f"Best demo: {demo['primary']}" if demo.get("primary") else ""
        widgets["demo"].configure(text=demo_text)

        recs = system_info.get_recommended_skus_for_model(model.get("id"), model)
        tags = model.get("tags") or []
        rec_text = ""
        if recs:
            rec_text += "Recommended for: " + " | ".join(map(str, recs[:3]))
        if tags:
            rec_text += ("\n" if rec_text else "") + "Tags: " + " ".join(f"#{t}" for t in tags[:8])
        widgets["recs"].configure(text=rec_text)

        is_image = self._is_image_model(model)
        is_utility = self._is_utility_demo_model(model)
        downloadable = bool(model.get("ollama_tag") or model.get("comfyui_model"))
        installed = status_state in {"installed", "installed_offline"}
        primary_enabled = (
            is_utility
            or (is_image and installed)
            or (not is_image and not is_utility and bool(model.get("ollama_tag")))
        )
        if is_image:
            primary_text = "Open Image Gen"
        elif is_utility:
            primary_text = "Open Toolbox"
        else:
            primary_text = "Load & Chat"
        install_enabled = downloadable and status_state in {"missing", "offline_missing"}
        if (not is_image) and status_state == "offline":
            install_enabled = False
        # v5.4: For user-added entries, the install button is ALSO disabled
        # when the backend has no one-click downloader (e.g. raw transformers).
        # Surfacing this up-front beats showing a modal after the user clicks.
        if install_enabled and model.get("user_added") and not self._user_added_model_can_install(model):
            install_enabled = False
        widgets["install"].configure(state="normal" if install_enabled else "disabled")
        widgets["primary"].configure(text=primary_text, state="normal" if primary_enabled else "disabled")
        widgets["ideas"].configure(state="normal")
        widgets["learn"].configure(state="normal" if self._model_learn_more_url(model) else "disabled")
        widgets["delete"].configure(state="normal" if installed and downloadable else "disabled")

        # User-added affordances: surface only when relevant, hide otherwise.
        review_banner = widgets.get("review_banner")
        source_label = widgets.get("source_label")
        remove_user_btn = widgets.get("remove_user_btn")
        is_user_added = bool(model.get("user_added"))
        requires_review = bool(model.get("requires_review"))
        source_url = str(model.get("source_url") or "")

        if review_banner is not None:
            if is_user_added and requires_review:
                review_banner.configure(
                    text="⚠ Marked for review — LocalAI couldn't fully verify this model would run. "
                         "Treat as informational until you confirm it works on your hardware."
                )
                review_banner.grid(row=6, column=0, sticky="ew", padx=(14, 22), pady=(0, 6))
            else:
                review_banner.grid_remove()
        if source_label is not None:
            if is_user_added and source_url:
                source_label.configure(text=f"Source: {source_url}")
                source_label.grid(row=7, column=0, sticky="ew", padx=(14, 22), pady=(0, 6))
            else:
                source_label.grid_remove()
        if remove_user_btn is not None:
            if is_user_added:
                remove_user_btn.grid(row=8, column=0, sticky="ew", padx=(14, 22), pady=(0, 10))
            else:
                remove_user_btn.grid_remove()

    def _open_selected_model_source_url(self) -> None:
        model = self._selected_model()
        if not model:
            return
        url = str(model.get("source_url") or "").strip()
        if not url:
            return
        try:
            webbrowser.open_new_tab(url)
        except Exception as exc:
            logger.warning(f"Could not open source URL: {exc}")

    def _remove_selected_user_model(self) -> None:
        model = self._selected_model()
        if not model:
            return
        self._remove_user_added_model_from_catalog(model)

    def _install_selected_model(self) -> None:
        model = self._selected_model()
        if not model:
            return
        # v5.4: Install gate for user-added entries whose verdict was
        # "warn" or "unsupported". The button is already disabled when the
        # gate would block (see _update_model_detail), so this branch is
        # defensive — defer to start_download() / download_comfyui_model()
        # for the canonical user-facing message rather than duplicating it.
        if model.get("user_added") and not self._user_added_model_can_install(model):
            logger.info(
                "Install blocked for user-added model '%s' (backend=%s)",
                model.get("id"), model.get("backend"),
            )
            # Fall through — start_download() / download_comfyui_model()
            # will surface the canonical "can't download automatically"
            # modal so the user sees one message, not two.
        if self._is_image_model(model):
            self.download_comfyui_model(model)
        else:
            self.start_download(model)

    def _user_added_model_can_install(self, model: dict) -> bool:
        """Return True when LocalAI has a one-click downloader that can handle
        this user-added entry. Mirrors the actual routing of
        ``_install_user_added_model`` → ``start_download`` /
        ``download_comfyui_model``: today only the Ollama text path and the
        ComfyUI image path are wired up end-to-end. ONNX, ONNX-GenAI,
        OpenVINO, and Phase 1 toolbox entries are added to the catalog for
        reference but cannot be one-click installed yet — the centralized
        gate must surface the "Can't download automatically" infobox for
        those so we never reach ``ollama.pull_model(None)`` / a no-op
        ComfyUI download.  When a real installer for those backends lands,
        widen this predicate AND extend ``start_download`` /
        ``download_comfyui_model`` together (gate + dispatcher must agree).
        """
        backend = str(model.get("backend") or "").strip().lower()
        if backend == "ollama":
            return bool(model.get("ollama_tag"))
        if backend == "comfyui":
            return bool(model.get("comfyui_model"))
        return False

    # ── Add-from-Hugging-Face dialog ─────────────────────────────────────────

    _HF_INSPECT_CACHE_LIMIT = 32  # FIFO; matches perf-reviewer guidance

    def _open_add_from_hf_dialog(self) -> None:
        """Open the modal **+ Add from Hugging Face** dialog.

        See docs/architecture.md §4 (the catalog) for how to add new models.
        The dialog does three things on a daemon thread (never blocking
        the Tk loop):

        1. Parse the pasted URL via :func:`hf_model_resolver.parse_url`.
        2. Inspect the repo via :func:`hf_compat.inspect`.
        3. Render a preview card and offer **Add & Download now** /
           **Add to catalog (download later)** / **Cancel**.

        Stale callbacks are guarded by an incrementing
        ``_hf_inspect_gen_id`` so a slow first paste can't overwrite the
        preview of a faster second paste.
        """
        existing = getattr(self, "_hf_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    return
            except Exception:
                pass

        self._hf_inspect_gen_id = int(getattr(self, "_hf_inspect_gen_id", 0))
        if not hasattr(self, "_hf_inspect_cache"):
            self._hf_inspect_cache = {}

        win = ctk.CTkToplevel(self)
        self._hf_dialog = win
        win.title("Add a model from Hugging Face")
        win.geometry("760x640")
        win.minsize(640, 520)
        try:
            win.transient(self)
        except Exception:
            pass
        try:
            win.grab_set()
        except Exception:
            pass
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_hf_dialog())

        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(3, weight=1)

        # Header
        header = ctk.CTkFrame(win, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 4))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header, text="Add a model from Hugging Face",
            font=ctk.CTkFont(size=18, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, anchor="w", text_color=TEXT_MUTED,
            text="Paste a Hugging Face URL or `org/name` slug, then press "
                 "Check compatibility. LocalAI inspects the repo and tells "
                 "you whether it can run before adding it to your catalog.",
            wraplength=700, justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # URL row
        url_row = ctk.CTkFrame(win, fg_color="transparent")
        url_row.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 4))
        url_row.grid_columnconfigure(0, weight=1)

        url_var = ctk.StringVar()
        url_entry = ctk.CTkEntry(
            url_row, textvariable=url_var, height=34,
            placeholder_text="https://huggingface.co/org/name  ·  org/name  ·  ollama:tag",
        )
        url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        url_entry.focus_set()

        check_btn = ctk.CTkButton(
            url_row, text="Check compatibility", width=180,
            command=lambda: self._run_hf_inspect_async(url_var.get()),
            **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
        )
        check_btn.grid(row=0, column=1)
        url_entry.bind("<Return>", lambda _e: self._run_hf_inspect_async(url_var.get()))

        # Secondary link row
        link_row = ctk.CTkFrame(win, fg_color="transparent")
        link_row.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 4))
        link_row.grid_columnconfigure(2, weight=1)
        ctk.CTkButton(
            link_row, text="Browse Hugging Face", width=160,
            **self._outline_button_style(),
            command=lambda: webbrowser.open_new_tab(
                "https://huggingface.co/models?library=transformers,diffusers,gguf,onnx&sort=trending"
            ),
        ).grid(row=0, column=0, padx=(0, 6))
        ctk.CTkButton(
            link_row, text="Browse Ollama Library", width=160,
            **self._outline_button_style(),
            command=lambda: webbrowser.open_new_tab("https://ollama.com/library"),
        ).grid(row=0, column=1, padx=(0, 6))
        status_label = ctk.CTkLabel(
            link_row, text="", text_color=TEXT_MUTED, anchor="w",
        )
        status_label.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        # Preview area (scrollable so long reason lists don't blow the window)
        preview = ctk.CTkScrollableFrame(win, label_text="")
        preview.grid(row=3, column=0, sticky="nsew", padx=14, pady=(4, 8))
        preview.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preview, anchor="w", justify="left", text_color=TEXT_MUTED,
            wraplength=680,
            text="No model checked yet. Paste a URL above and press "
                 "Check compatibility. Inspection runs in the background — "
                 "you can keep using LocalAI while it works.",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        # Footer (commit buttons; disabled until a successful inspect)
        footer = ctk.CTkFrame(win, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 14))
        footer.grid_columnconfigure(0, weight=1)

        cancel_btn = ctk.CTkButton(
            footer, text="Cancel", width=110,
            **self._outline_button_style(),
            command=lambda: self._close_hf_dialog(),
        )
        cancel_btn.grid(row=0, column=1, padx=(6, 0))

        add_later_btn = ctk.CTkButton(
            footer, text="Add to catalog (download later)", width=240,
            **self._outline_button_style(),
            state="disabled",
            command=lambda: self._commit_user_added_model(download_now=False),
        )
        add_later_btn.grid(row=0, column=2, padx=(6, 0))

        add_now_btn = ctk.CTkButton(
            footer, text="Add & Download now", width=190,
            **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
            state="disabled",
            command=lambda: self._commit_user_added_model(download_now=True),
        )
        add_now_btn.grid(row=0, column=3, padx=(6, 0))

        # Stash everything the async callback needs so we don't keep
        # re-finding widgets in self.
        self._hf_dialog_state = {
            "win": win,
            "url_var": url_var,
            "url_entry": url_entry,
            "check_btn": check_btn,
            "status_label": status_label,
            "preview": preview,
            "add_now_btn": add_now_btn,
            "add_later_btn": add_later_btn,
            "last_result": None,
        }

    def _close_hf_dialog(self) -> None:
        win = getattr(self, "_hf_dialog", None)
        if win is None:
            return
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        self._hf_dialog = None
        self._hf_dialog_state = None

    def _run_hf_inspect_async(self, raw_url: str) -> None:
        state = getattr(self, "_hf_dialog_state", None)
        if not state:
            return
        url = (raw_url or "").strip()
        if not url:
            state["status_label"].configure(text="Paste a URL above first.", text_color=WARN_TEXT)
            return

        try:
            parsed = parse_hf_url(url)
        except InvalidHFUrl as exc:
            state["status_label"].configure(
                text=str(exc) or "That URL didn't look like a Hugging Face or Ollama target.",
                text_color=WARN_TEXT,
            )
            return

        self._hf_inspect_gen_id = int(getattr(self, "_hf_inspect_gen_id", 0)) + 1
        my_gen = self._hf_inspect_gen_id

        # Disable Check + footer buttons during the round trip.
        state["check_btn"].configure(state="disabled", text="Checking…")
        state["add_now_btn"].configure(state="disabled")
        state["add_later_btn"].configure(state="disabled")
        state["status_label"].configure(
            text=f"Inspecting {parsed.repo_id or parsed.ollama_tag or url} …",
            text_color=TEXT_MUTED,
        )

        # Session cache: skip the HfApi round-trip when the user re-pastes
        # the exact same URL within one dialog session.
        cache = self._hf_inspect_cache
        cache_key = (parsed.route, parsed.repo_id, parsed.revision, parsed.file_path, parsed.ollama_tag)
        cached = cache.get(cache_key)
        if cached is not None:
            self.after(0, lambda r=cached, p=parsed, g=my_gen: self._render_hf_preview(p, r, g))
            return

        def _worker():
            try:
                # Mute HF Hub progress bars and telemetry inside the inspect
                # thread — they're noise here and can blow stderr.
                prev = {}
                for k in ("HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_DISABLE_TELEMETRY"):
                    prev[k] = os.environ.get(k)
                    os.environ[k] = "1"
                try:
                    result = hf_compat.inspect(parsed)
                finally:
                    for k, v in prev.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
            except Exception as exc:  # pragma: no cover — safety net
                logger.exception(f"hf inspect failed: {exc}")
                result = hf_compat.CompatResult(
                    verdict="unsupported",
                    reasons=[f"Couldn't inspect that repo: {exc}"],
                )
            self.after(0, lambda r=result, p=parsed, g=my_gen: self._on_hf_inspect_done(p, r, g))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_hf_inspect_done(self, parsed: ParsedTarget,
                            result: "hf_compat.CompatResult", gen_id: int) -> None:
        # Stale-callback barrier: another paste superseded us.
        if gen_id != int(getattr(self, "_hf_inspect_gen_id", 0)):
            return
        cache = self._hf_inspect_cache
        key = (parsed.route, parsed.repo_id, parsed.revision, parsed.file_path, parsed.ollama_tag)
        cache[key] = result
        # FIFO cap: drop the oldest entries when we exceed the limit.
        while len(cache) > self._HF_INSPECT_CACHE_LIMIT:
            try:
                cache.pop(next(iter(cache)))
            except (StopIteration, KeyError):
                break
        self._render_hf_preview(parsed, result, gen_id)

    def _render_hf_preview(self, parsed: ParsedTarget,
                           result: "hf_compat.CompatResult", gen_id: int) -> None:
        state = getattr(self, "_hf_dialog_state", None)
        if not state:
            return
        if gen_id != int(getattr(self, "_hf_inspect_gen_id", 0)):
            return
        state["last_result"] = (parsed, result)

        preview = state["preview"]
        for child in preview.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass

        verdict = result.verdict
        verdict_text = {
            "supported":   "● Supported — ready to add",
            "warn":        "▲ Supported with warnings",
            "needs_access":"🔒 Gated — needs your acceptance on Hugging Face",
            "unsupported": "✖ Not supported by LocalAI today",
        }.get(verdict, "? Unknown verdict")
        verdict_color = {
            "supported":   SUCCESS_TEXT,
            "warn":        WARN_TEXT,
            "needs_access": WARN_TEXT,
            "unsupported": ERROR_TEXT,
        }.get(verdict, TEXT_PRIMARY)

        ctk.CTkLabel(
            preview, text=verdict_text, anchor="w", text_color=verdict_color,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        entry = result.proposed_entry or {}
        name = entry.get("name") or parsed.repo_id or parsed.ollama_tag or "Unknown"
        vendor = entry.get("vendor") or "?"
        backend = result.backend or entry.get("backend") or "unknown"
        category = entry.get("category") or "—"
        size_gb = float(entry.get("size_gb") or 0.0)

        summary_lines = [
            f"Name:      {name}",
            f"Vendor:    {vendor}",
            f"Backend:   {backend}",
            f"Category:  {category}",
            f"Download:  {size_gb:g} GB",
            f"Min VRAM:  {entry.get('min_vram_gb', 0)} GB · Min RAM: {entry.get('min_ram_gb', 0)} GB",
        ]
        if result.family:
            summary_lines.append(f"Family:    {result.family}")
        if result.pipeline_tag:
            summary_lines.append(f"Pipeline:  {result.pipeline_tag}")
        if result.resolved_sha:
            summary_lines.append(f"Pinned to: {result.resolved_sha[:10]}… (40-char SHA)")
        ctk.CTkLabel(
            preview, text="\n".join(summary_lines), anchor="w", justify="left",
            font=ctk.CTkFont(size=12, family="Consolas"),
            text_color=TEXT_PRIMARY,
        ).grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))

        if result.reasons:
            ctk.CTkLabel(
                preview, text="Why:", anchor="w",
                font=ctk.CTkFont(size=12, weight="bold"),
            ).grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 0))
            ctk.CTkLabel(
                preview, anchor="w", justify="left", wraplength=680,
                text="\n".join(f"• {r}" for r in result.reasons),
                text_color=TEXT_SECONDARY,
            ).grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))

        if result.warnings:
            ctk.CTkLabel(
                preview, text="Heads-up:", anchor="w",
                font=ctk.CTkFont(size=12, weight="bold"), text_color=WARN_TEXT,
            ).grid(row=4, column=0, sticky="ew", padx=8, pady=(4, 0))
            ctk.CTkLabel(
                preview, anchor="w", justify="left", wraplength=680,
                text="\n".join(f"• {w}" for w in result.warnings),
                text_color=WARN_TEXT,
            ).grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 6))

        if verdict == "needs_access" and parsed.repo_id:
            gated_row = ctk.CTkFrame(preview, fg_color="transparent")
            gated_row.grid(row=6, column=0, sticky="ew", padx=8, pady=(6, 8))
            gated_row.grid_columnconfigure(0, weight=0)
            gated_row.grid_columnconfigure(1, weight=0)
            gated_row.grid_columnconfigure(2, weight=1)
            ctk.CTkButton(
                gated_row, text="Open model page",
                **self._outline_button_style(),
                command=lambda r=parsed.repo_id: webbrowser.open_new_tab(
                    f"https://huggingface.co/{r}"
                ),
            ).grid(row=0, column=0, sticky="w", padx=(0, 8))
            ctk.CTkButton(
                gated_row, text="Manage HF access tokens",
                **self._outline_button_style(),
                command=lambda: webbrowser.open_new_tab(
                    "https://huggingface.co/settings/tokens"
                ),
            ).grid(row=0, column=1, sticky="w")

        state["check_btn"].configure(state="normal", text="Check compatibility")
        state["status_label"].configure(text="", text_color=TEXT_MUTED)

        can_install = verdict in {"supported", "warn"} and result.proposed_entry
        can_add_only = verdict in {"supported", "warn"} and result.proposed_entry
        state["add_now_btn"].configure(state="normal" if can_install else "disabled")
        state["add_later_btn"].configure(state="normal" if can_add_only else "disabled")

    def _commit_user_added_model(self, *, download_now: bool) -> None:
        state = getattr(self, "_hf_dialog_state", None)
        if not state or not state.get("last_result"):
            return
        parsed, result = state["last_result"]
        if result.verdict in {"unsupported", "needs_access"} or not result.proposed_entry:
            messagebox.showinfo(
                "Nothing to add",
                "There's no supported model to add yet. Check the messages "
                "above and try a different URL.",
            )
            return

        # Disk preflight (perf-reviewer #6): before we kick off a downloader
        # thread, make sure the user actually has room. We only block when
        # we're about to download right now.
        if download_now and result.size_bytes_total:
            need_gb = float(result.size_bytes_total) / (1024 ** 3)
            free_gb = self._cached_home_disk_free_gb() if hasattr(self, "_cached_home_disk_free_gb") else None
            if free_gb is not None and free_gb < need_gb + 1.0:
                if not messagebox.askyesno(
                    "Not much disk space",
                    f"This model needs about {need_gb:.1f} GB and you only "
                    f"have {free_gb:.1f} GB free. Add and download anyway?",
                ):
                    return

        source_url = (state["url_var"].get() or "").strip()
        requires_review = (result.verdict == "warn")
        ok, final_id, _models = catalog.append_user_model(
            result.proposed_entry,
            source_url=source_url,
            requires_review=requires_review,
        )
        if not ok:
            messagebox.showerror("Could not add model", final_id)
            return

        self._reload_catalog()

        # Surface the new row: switch type filter to the new entry's
        # category bucket (so it's visible), then select it.
        new_model = next(
            (m for m in self._catalog_models if m.get("id") == final_id),
            None,
        )
        if new_model is not None:
            try:
                self._set_model_type_filter(self._model_type_label(new_model))
            except Exception:
                pass
            try:
                self._select_model_row(final_id)
            except Exception:
                pass

        self.set_status(
            f"Added {new_model.get('name') if new_model else final_id} to your catalog."
        )

        if download_now:
            self.after(150, lambda m=new_model: self._install_user_added_model(m))

        self._close_hf_dialog()

    def _install_user_added_model(self, model: dict | None) -> None:
        if not model:
            return
        if not self._user_added_model_can_install(model):
            messagebox.showinfo(
                "Can't download automatically",
                "LocalAI added the model to your catalog, but it can't "
                "kick off the download — there's no one-click installer "
                "for this backend yet. Use the source link in the detail "
                "pane to grab it manually.",
            )
            return
        try:
            if self._is_image_model(model):
                self.download_comfyui_model(model)
            else:
                self.start_download(model)
        except Exception as exc:
            logger.exception(f"User-added install failed: {exc}")
            messagebox.showerror("Download failed", str(exc))

    def _remove_user_added_model_from_catalog(self, model: dict) -> None:
        if not model or not model.get("user_added"):
            return
        target_id = str(model.get("id") or "")
        if not target_id:
            return
        if not messagebox.askyesno(
            "Remove from catalog?",
            f"Remove '{model.get('name', target_id)}' from your catalog?\n\n"
            "This only removes the catalog entry — any files already "
            "downloaded on disk are left alone so you can re-add the "
            "model later without re-downloading.",
        ):
            return
        models = list(self._catalog_models)
        remaining = [m for m in models if m.get("id") != target_id]
        if len(remaining) == len(models):
            return
        if not catalog.save_catalog(remaining):
            messagebox.showerror("Could not save catalog", "Catalog write failed; entry was not removed.")
            return
        self._reload_catalog()
        self.set_status(f"Removed {model.get('name', target_id)} from catalog.")

    def _run_selected_model_primary_action(self) -> None:
        model = self._selected_model()
        if not model:
            return
        is_image = self._is_image_model(model)
        is_utility = self._is_utility_demo_model(model)
        if is_image:
            _status, _color, state = self._model_install_status(model)
            if state in {"missing", "offline_missing"} and model.get("comfyui_model"):
                self._install_selected_model()
                return
            if state in {"installed", "installed_offline"}:
                self.open_image_gen_for_model(model)
        elif is_utility:
            self.open_toolbox_for_model(model)
        else:
            self._run_chat_model_primary_action(model)

    def _run_chat_model_primary_action(self, model: dict) -> None:
        tag = model.get("ollama_tag", "")
        if not tag:
            return
        local_names = self._cached_local_names_snapshot()
        if local_names:
            self._route_chat_model_primary_action(model, _ollama_tag_is_local(tag, local_names))
            return
        if not self.ollama_ok:
            self._start_chat_download_after_primary_action(model)
            return

        self._model_primary_action_generation = self.__dict__.get("_model_primary_action_generation", 0) + 1
        generation = self._model_primary_action_generation
        widgets = self.__dict__.get("_model_detail_widgets", {})
        primary = widgets.get("primary") if widgets else None
        if primary:
            primary.configure(text="Checking...", state="disabled")
        self.set_status(f"Checking whether '{model.get('name', 'model')}' is installed ...")

        def _worker():
            error = ""
            try:
                is_local = self.ollama.is_model_local(tag)
            except Exception as exc:
                is_local = False
                error = str(exc)
            self.after(
                0,
                lambda m=model, g=generation, local=is_local, err=error: self._finish_chat_model_primary_action(
                    m, g, local, err
                ),
            )

        threading.Thread(target=_worker, name="ModelPrimaryActionCheck", daemon=True).start()

    def _finish_chat_model_primary_action(
        self,
        model: dict,
        generation: int,
        is_local: bool,
        error: str = "",
    ) -> None:
        if generation != self.__dict__.get("_model_primary_action_generation", 0):
            return
        if self._selected_model_id != model.get("id"):
            return
        if error:
            logger.warning(f"Could not check Ollama model status before primary action: {error}")
        self._route_chat_model_primary_action(model, is_local)

    def _route_chat_model_primary_action(self, model: dict, is_local: bool) -> None:
        if is_local:
            self.load_model_for_chat(model)
        else:
            self._start_chat_download_after_primary_action(model)

    def _start_chat_download_after_primary_action(self, model: dict) -> bool:
        previous_pending = self.__dict__.get("_pending_chat_load_after_download", None)
        pending = {
            "model_id": model.get("id"),
            "backend": self._backend_var.get(),
        }
        self._pending_chat_load_after_download = pending
        accepted = self.start_download(model)
        if not accepted and self.__dict__.get("_pending_chat_load_after_download", None) == pending:
            self._pending_chat_load_after_download = previous_pending
            self._update_model_detail()
        return accepted

    def _open_selected_model_ideas(self) -> None:
        model = self._selected_model()
        if model:
            self.open_prompt_ideas_for_model(model)

    def _model_learn_more_url(self, model: dict | None) -> str:
        if not model:
            return ""
        explicit_url = str(model.get("learn_more_url") or "").strip()
        if explicit_url.startswith(("https://", "http://")):
            return explicit_url
        repo = str(model.get("hf_repo") or model.get("onnx_repo") or "").strip()
        if repo and "/" in repo:
            return f"https://huggingface.co/{repo.strip('/')}"
        url = str(model.get("comfyui_model_url") or "").strip()
        if "huggingface.co/" in url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2:
                return f"https://huggingface.co/{parts[0]}/{parts[1]}"
        tag = str(model.get("ollama_tag") or "").strip()
        if tag:
            base = tag.split(":", 1)[0].strip()
            if base:
                from urllib.parse import quote
                return f"https://ollama.com/library/{quote(base, safe='')}"
        return ""

    def _open_selected_model_learn_more(self) -> None:
        model = self._selected_model()
        url = self._model_learn_more_url(model)
        if not url:
            self.set_status("No Learn More page is available for this model.")
            return
        webbrowser.open(url)
        self.set_status(f"Opening Learn More for {model.get('name', 'model')} ...")

    def _delete_selected_model(self) -> None:
        model = self._selected_model()
        if not model:
            return
        ok = messagebox.askyesno(
            "Delete model",
            f"Delete '{model.get('name', 'model')}' from local storage?\n"
            "It will need to be re-downloaded to use again.",
            parent=self,
        )
        if not ok:
            return
        if self._is_image_model(model):
            self.delete_comfyui_model(model)
        else:
            self.delete_model(model)

    def _populate_model_cards(self, token: int | None = None):
        """
        Master-detail Models page: render one compact row per visible model,
        with duplicated descriptions/actions moved to the selected-model detail pane.
        Rows are still pooled and grid_remove()/grid()'d across filters.

        Noops cleanly if the Models page hasn't been lazily built yet — the
        Device detection callback in particular can fire before we ever
        visit Models.
        """
        if not hasattr(self, "_cards_scroll"):
            return
        if token is None:
            self._model_population_token += 1
            token = self._model_population_token
        elif token != self._model_population_token:
            return

        capacity = self._current_model_capacity()
        show_unsupported = bool(self._image_gen_show_oversize_var.get())

        # v5: apply model-search filter (substring on name / id / tags)
        search = (getattr(self, '_model_search_var', None) and
                  self._model_search_var.get().strip().lower())

        def _matches_search(m: dict) -> bool:
            if not search:
                return True
            hay = " ".join([
                str(m.get("name", "")),
                str(m.get("id", "")),
                " ".join(m.get("tags", []) or []),
                str(m.get("vendor", "")),
            ]).lower()
            return search in hay

        fit_counts = {"fits_well": 0, "tight": 0, "exceeds": 0, "browse": 0}
        visible_models: list[dict] = []
        banner_image_models: list[dict] = []
        self._model_row_fit_cache = {}
        for model in self._catalog_models:
            fit = self._model_fit_tier_for_capacity(model, capacity)
            fit_counts[fit] = fit_counts.get(fit, 0) + 1
            self._model_row_fit_cache[model.get("id", "")] = fit
            if not self._model_matches_type_filter(model):
                continue
            if not self._model_matches_size_filter(model):
                continue
            if not _matches_search(model):
                continue
            if self._is_image_model(model):
                banner_image_models.append(model)
            if not capacity.get("all_catalog") and fit == "exceeds" and not show_unsupported:
                continue
            visible_models.append(model)

        supported = fit_counts.get("fits_well", 0)
        tight = fit_counts.get("tight", 0)
        unsupported = fit_counts.get("exceeds", 0)
        label = str(capacity.get("label") or "this selection")
        if getattr(self, "_model_compat_summary_label", None) is not None:
            if capacity.get("all_catalog"):
                self._model_compat_summary_label.configure(
                    text=f"Browsing all {len(self._catalog_models)} catalog models.",
                    text_color=TEXT_SECONDARY,
                )
            else:
                self._model_compat_summary_label.configure(
                    text=f"{supported + tight} supported · {tight} tight fit{'s' if tight != 1 else ''} · {unsupported} unsupported",
                    text_color=TEXT_SECONDARY,
                )
        if getattr(self, "_model_results_summary_label", None) is not None:
            self._model_results_summary_label.configure(
                text=(
                    f"Showing {len(visible_models)} model{'s' if len(visible_models) != 1 else ''} for {label} "
                    f"· Type: {self._model_type_filter_var.get()} · Size: {self._model_size_filter_var.get()}"
                )
            )

        image_models = [m for m in visible_models if self._is_image_model(m)]
        non_image_models = [m for m in visible_models if not self._is_image_model(m)]

        if banner_image_models and not capacity.get("all_catalog"):
            self._ensure_img_summary_banner(self._filter_panel_ref)
            vram_gb = int(float(capacity.get("vram_gb") or 0))
            fits = [m for m in banner_image_models
                    if self._model_row_fit_cache.get(m.get("id", "")) != "exceeds"]
            self._update_img_summary_banner(banner_image_models, fits, vram_gb)
            try:
                self._img_summary_banner.grid(
                    row=5, column=0, sticky="ew", padx=12, pady=(0, 10),
                )
            except Exception:
                pass
        elif self._img_summary_banner is not None:
            try: self._img_summary_banner.grid_remove()
            except Exception: pass

        # Use current snapshots while the page paints; backend probes run off
        # the UI thread and refresh card status when they complete.
        local_names = self._cached_local_names_snapshot()
        comfyui_model_names = self._cached_comfyui_model_names_snapshot()

        render_ops: list[tuple] = []
        if non_image_models:
            title = "Supported models" if self._model_type_filter_var.get() == "All types" else self._model_type_filter_var.get()
            render_ops.append(("section", "list_non_image", title, len(non_image_models)))
            for m in sorted(non_image_models, key=self._model_row_sort_key):
                render_ops.append(("card", m))

        if image_models:
            vram_gb = int(float(capacity.get("vram_gb") or 0))
            sections = self._build_image_gen_sections(
                image_models,
                vram_gb,
                fit_by_id=self._model_row_fit_cache,
                show_oversize=show_unsupported or bool(capacity.get("all_catalog")),
                all_catalog=bool(capacity.get("all_catalog")),
            )
            for sid, group in sections:
                title = next((t for _sid, t, _i in self._IMG_SECTIONS if _sid == sid), sid)
                render_ops.append(("section", sid, title, len(group)))
                collapsed = bool(self._section_collapsed.get(sid, False))
                if not collapsed:
                    for m in group:
                        render_ops.append(("card", m))

        visible_card_ids = [m["id"] for op in render_ops
                            if op[0] == "card" for m in [op[1]]]
        visible_card_id_set = set(visible_card_ids)
        active_section_ids = {op[1] for op in render_ops if op[0] == "section"}

        # Hide cards / section headers that are no longer in the visible set
        for mid, card in list(self._model_cards_by_id.items()):
            if mid not in visible_card_id_set:
                try: card.grid_remove()
                except Exception: pass
        for sid, hdr in list(self._section_headers.items()):
            if sid not in active_section_ids:
                try: hdr.grid_remove()
                except Exception: pass

        # Render the ordered list in small batches so first navigation stays responsive.
        self._model_cards = []
        if not render_ops:
            if self._cards_empty_label is None:
                self._cards_empty_label = ctk.CTkLabel(
                    self._cards_scroll,
                    text="No models match the current filters.",
                    font=ctk.CTkFont(size=13), text_color=TEXT_MUTED,
                )
            self._cards_empty_label.configure(text="No models match the current filters.")
            self._cards_empty_label.grid(row=0, column=0, pady=40)
            self._selected_model_id = None
            self._sync_model_row_selection()
            self._update_model_detail(local_names=local_names, comfyui_model_names=comfyui_model_names)
            self._models_page_just_built = False
            return
        elif self._cards_empty_label is not None:
            self._cards_empty_label.grid_remove()
        self._continue_model_card_population(
            token, render_ops, local_names, comfyui_model_names,
            visible_card_ids=visible_card_ids,
            index=0, row=0, started_at=time.perf_counter(),
        )

    def _continue_model_card_population(
        self,
        token: int,
        render_ops: list[tuple],
        local_names: set,
        comfyui_model_names: set,
        visible_card_ids: list[str],
        index: int,
        row: int,
        started_at: float,
    ) -> None:
        if token != self._model_population_token or not hasattr(self, "_cards_scroll"):
            return
        end = min(index + 12, len(render_ops))
        while index < end:
            op = render_ops[index]
            if op[0] == "section":
                _, sid, title, count = op
                hdr = self._ensure_section_header(sid, title, count)
                hdr.grid(row=row, column=0, sticky="ew", padx=4, pady=(10 if row else 4, 4))
                row += 1
                index += 1
                continue

            model = op[1]
            mid = model["id"]
            card = self._model_cards_by_id.get(mid)
            if card is None:
                card = ModelListRow(
                    self._cards_scroll,
                    model,
                    self,
                    local_names=local_names,
                    comfyui_model_names=comfyui_model_names,
                )
                self._model_cards_by_id[mid] = card
            else:
                try:
                    card.refresh_status(local_names=local_names, comfyui_model_names=comfyui_model_names)
                    card.refresh_perf_badges()
                except Exception:
                    pass
            card.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
            self._model_cards.append(card)
            row += 1
            index += 1

        if index < len(render_ops):
            self.after(
                1,
                lambda: self._continue_model_card_population(
                    token, render_ops, local_names, comfyui_model_names,
                    visible_card_ids=visible_card_ids,
                    index=index, row=row, started_at=started_at,
                ),
            )
            return

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms > 250:
            logger.info(f"Models page row render completed in {elapsed_ms:.0f} ms.")
        if visible_card_ids:
            if self._selected_model_id not in visible_card_ids:
                self._selected_model_id = visible_card_ids[0]
        else:
            self._selected_model_id = None
        self._sync_model_row_selection()
        self._model_detail_local_names = set(local_names) if local_names is not None else None
        self._model_detail_comfyui_model_names = set(comfyui_model_names) if comfyui_model_names is not None else None
        self._update_model_detail(local_names=local_names, comfyui_model_names=comfyui_model_names)
        self._models_page_just_built = False
        self._schedule_model_status_refresh(force_refresh=False)

        # Wire Up/Down/Home/End across the now-visible model rows so users
        # can walk the list with arrow keys after one Tab into it. Re-wired
        # after every repopulation because the row set changes. The rows'
        # own bindings (Return / Space / FocusIn) are already set in
        # ModelListRow.__init__; this just adds sibling navigation.
        try:
            a11y.wire_arrow_navigation(
                list(self._model_cards),
                orientation="vertical",
            )
        except Exception:
            pass

    def _toggle_category(self, cat: str):
        # Back-compat for older callbacks/tests: category buttons were split
        # into explicit Type and Size filters for the master-detail Models UI.
        if cat in {"Ultra Small", "Small", "Medium", "Large", "Extra Large"}:
            self._set_model_size_filter("All sizes" if self._model_size_filter_var.get() == cat else cat)
            return
        mapping = {
            "All": "All types",
            "Image Generation": "Image Gen",
        }
        value = mapping.get(cat, cat)
        self._set_model_type_filter("All types" if self._model_type_filter_var.get() == value else value)

    def _refresh_cat_btn_states(self):
        self._refresh_model_filter_btn_states()

    def _apply_filter(self, _=None):
        filt_sku = self._optional_filter_var.get()
        if (
            hasattr(self, "_bench_profile_var")
            and filt_sku in self._bench_profile_values()
            and self._bench_profile_var.get() != filt_sku
            and not (getattr(self, "_bench_thread", None) and self._bench_thread.is_alive())
        ):
            self._bench_profile_var.set(filt_sku)
            self._on_bench_profile_changed(filt_sku)
        if filt_sku != self._sku_all_filter_value():
            if not self._optional_skus_enabled and filt_sku == "This Device":
                dev = self._optional_sku or {}
                self._update_category_for_device(
                    vram_gb=dev.get("vram_gb", 0),
                    ram_gb=dev.get("ram_gb", 0),
                )
                return
            sku_spec = next((s for s in self._optional_skus if s["name"] == filt_sku), None)
            if sku_spec:
                self._update_category_for_device(
                    vram_gb=sku_spec.get("vram_gb", 0),
                    ram_gb=sku_spec.get("ram_gb", 0),
                )
                return
        self._update_category_for_device()

    def _open_model_guide(self):
        """Open the consolidated Model Guide in the default browser.

        Replaces the old ``model-value-props.html`` opener. The new Model Guide
        absorbs that page's value-prop tables along with the three former
        prompt docs (ChatPromptIdeas / ImageGenPrompts / ModelDemoPrompts).
        Those four legacy HTML files were retired in v5.3.4 and the redirect
        shims that briefly stood in their place were deleted in post-v5.3.4
        docs cleanup — every entry point now opens Model-Guide.html directly.
        """
        import webbrowser
        guide_path = Path(__file__).parent.parent / "docs" / "Model-Guide.html"
        if guide_path.exists():
            webbrowser.open(guide_path.as_uri())
        else:
            webbrowser.open(str(guide_path))

    def _open_image_gen_guide(self):
        """Open the image generation feature guide in the default browser."""
        guide_path = Path(__file__).parent.parent / "docs" / "image-gen-guide.html"
        if guide_path.exists():
            webbrowser.open(guide_path.as_uri())
        else:
            webbrowser.open(str(guide_path))

    def _prompt_doc_system_params(self, force_cpu: bool = False) -> dict[str, str]:
        """Return common URL params used by prompt-guide docs."""
        params: dict[str, str] = {}
        state = object.__getattribute__(self, "__dict__")
        gpu_info = state.get("gpu_info")
        sku = state.get("_optional_sku") or {}
        if "tk" in state or "_optional_filter_var" in state:
            try:
                vram_gb = self._active_device_vram_gb()
            except Exception:
                vram_gb = int(sku.get("vram_gb", 0) or 0)
        else:
            vram_gb = int(sku.get("vram_gb", 0) or 0)
        use_cpu = (
            bool(force_cpu)
            or getattr(gpu_info, "gpu_type", "") == "cpu"
            or vram_gb == 0
        )
        if use_cpu:
            params["hardware"] = "cpu"
        elif vram_gb > 0:
            params["vram"] = str(vram_gb)

        gpu_name = sku.get("name") or getattr(gpu_info, "device_name", "") or getattr(gpu_info, "name", "") or ""
        if gpu_name:
            params["gpu"] = gpu_name
        ram_gb = sku.get("ram_gb") or 0
        if ram_gb:
            params["ram"] = str(ram_gb)
        if getattr(gpu_info, "unified_memory", False) or sku.get("unified_memory"):
            params["unified"] = "1"
        return params

    def _docs_http_base_url(self) -> str | None:
        """Return a localhost docs base URL, starting a tiny server on demand.

        Why this exists: on some Windows browser-launch paths, `file:///...` URLs
        can lose query/hash components. Prompt-ideas deep links rely on those
        components, so we prefer localhost URLs when possible.
        """
        state = object.__getattribute__(self, "__dict__")
        # Keep object.__new__(App) contract tests side-effect free.
        if "tk" not in state and "_pages" not in state:
            return None

        docs_dir = Path(__file__).parent.parent / "docs"
        if not docs_dir.exists():
            return None

        server = state.get("_docs_http_server")
        thread = state.get("_docs_http_thread")
        port = int(state.get("_docs_http_port", 0) or 0)
        if server is not None and thread is not None and thread.is_alive() and port > 0:
            return f"http://127.0.0.1:{port}"

        try:
            from functools import partial
            from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

            class _DocsHandler(SimpleHTTPRequestHandler):
                def log_message(self, _fmt: str, *_args):
                    return

            handler = partial(_DocsHandler, directory=str(docs_dir))
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(
                target=httpd.serve_forever,
                name="DocsHttpServer",
                daemon=True,
            )
            thread.start()
            self._docs_http_server = httpd
            self._docs_http_thread = thread
            self._docs_http_port = int(getattr(httpd, "server_port", 0) or 0)
            if self._docs_http_port > 0:
                return f"http://127.0.0.1:{self._docs_http_port}"
        except Exception as exc:
            logger.warning(f"Docs HTTP helper unavailable: {exc}")
        return None

    def _stop_docs_http_server(self) -> None:
        server = getattr(self, "_docs_http_server", None)
        if server is None:
            return
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        self._docs_http_server = None
        self._docs_http_thread = None
        self._docs_http_port = 0

    def _open_prompt_doc_via_help(self, target_html: str, params: dict[str, str] | None = None, fragment: str = ""):
        """Open a prompt doc with URL params so model context reaches the guide."""
        from urllib.parse import quote, urlencode

        docs_dir = Path(__file__).parent.parent / "docs"
        docs_index = docs_dir / "index.html"
        target_path = docs_dir / target_html
        clean_params = {
            str(key): str(value)
            for key, value in (params or {}).items()
            if value is not None and str(value) != ""
        }
        encoded_fragment = quote(fragment, safe="-_.~") if fragment else ""
        fragment_suffix = f"#{encoded_fragment}" if encoded_fragment else ""
        query = urlencode(clean_params)

        if target_path.exists():
            base_url = self._docs_http_base_url()
            if base_url:
                target_url = f"{base_url}/{quote(target_html, safe='-_.~/')}"
            else:
                target_url = target_path.as_uri()
            webbrowser.open(target_url + (f"?{query}" if query else "") + fragment_suffix)
            return

        if docs_index.exists():
            base_url = self._docs_http_base_url()
            fallback_query = urlencode(clean_params)
            if base_url:
                index_url = f"{base_url}/index.html"
            else:
                index_url = docs_index.as_uri()
            webbrowser.open(index_url + (f"?{fallback_query}" if fallback_query else "") + fragment_suffix)
        else:
            webbrowser.open(str(target_path))

    def _open_image_prompts(self):
        """Open image prompt ideas directly,
        passing the current system's capability as URL query params so the
        doc can filter, highlight, and label the models that apply to *this*
        machine. The doc accepts:
            ?vram=24          numeric GB
            ?hardware=cpu     forces CPU-friendly subset
            ?gpu=NVIDIA+A10   display label for the system banner
            ?ram=64           GB of system RAM (for CPU mode and Apple)
            ?unified=1        Apple Silicon shared-pool flag
            ?modelId=sdxl-base  scrolls directly to that model's prompt card
        """
        params = self._prompt_doc_system_params(
            force_cpu=bool(getattr(self, "_comfyui_force_cpu", False))
        )
        fragment = ""

        selected_entry = self._selected_image_model_catalog_entry()
        if selected_entry and selected_entry.get("id"):
            model_id = str(selected_entry["id"])
            params["modelId"] = model_id
            params["imageModel"] = model_id
            fragment = model_demos.doc_fragment(model_id)

        self._open_prompt_doc_via_help("Model-Guide.html", params, fragment)

    def _open_chat_prompt_ideas(self):
        """Open the consolidated Model Guide with the chat model in focus."""
        params = self._prompt_doc_system_params()
        try:
            selected = self._selected_chat_model()
        except Exception:
            selected = None
        active = selected or self.active_model or {}
        fragment = ""
        if active:
            model_id = str(active.get("id") or active.get("ollama_tag") or active.get("name") or "")
            if model_id:
                params["model"] = model_id
                params["modelId"] = model_id
                params["chatModel"] = model_id
                fragment = self._chat_prompt_fragment(model_id)
            tag = str(active.get("ollama_tag") or "")
            if tag:
                params["ollama"] = tag
                params["ollamaTag"] = tag

        self._open_prompt_doc_via_help("Model-Guide.html", params, fragment)

    def open_prompt_ideas_for_model(self, model: dict | None):
        """Open the unified Model Guide at a model-specific card."""
        fragment = ""
        if model:
            model_id = str(model.get("id") or model.get("ollama_tag") or model.get("name") or "")
            if model_id:
                fragment = model_demos.doc_fragment(model_id)

        self._open_prompt_doc_via_help("Model-Guide.html", self._prompt_doc_system_params(), fragment)

    @staticmethod
    def _chat_prompt_fragment(model_id: str) -> str:
        """Return the Model-Guide.html fragment id for a catalog model id."""
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(model_id))
        slug = "-".join(part for part in slug.split("-") if part)
        return f"model-{slug}" if slug else ""

    def _selected_image_model_catalog_entry(self) -> dict | None:
        """Return the catalog entry for the currently selected Image Gen model."""
        selected_display = ""
        if hasattr(self, "_img_model_var"):
            selected_display = (self._img_model_var.get() or "").strip()
        if not selected_display or selected_display.startswith("("):
            return None

        selected_filename = getattr(self, "_img_friendly_to_filename", {}).get(selected_display)
        if selected_filename:
            entry = self._find_catalog_entry_for_model(selected_filename)
            if entry:
                return entry

        return next(
            (
                m for m in getattr(self, "_catalog_models", [])
                if m.get("category") == "Image Generation"
                and (m.get("name") == selected_display or m.get("id") == selected_display)
            ),
            None,
        )

    def _refresh_model_cards(self):
        # Explicit refresh fetches backend status off the UI thread.
        self._schedule_model_status_refresh(force_refresh=True)

    def _invalidate_model_status_refresh(self) -> None:
        self._model_status_refresh_generation = self.__dict__.get("_model_status_refresh_generation", 0) + 1

    def _latest_model_status_snapshots(self) -> tuple[set | None, set | None]:
        local_names = getattr(self, "_model_detail_local_names", None)
        if local_names is None:
            cache = getattr(self, "_local_names_cache", None)
            if cache:
                _cached_at, cached_names = cache
                local_names = set(cached_names)
        else:
            local_names = set(local_names)

        comfyui_model_names = getattr(self, "_model_detail_comfyui_model_names", None)
        if comfyui_model_names is None:
            cache = getattr(self, "_comfyui_model_names_cache", None)
            if cache:
                _cached_at, cached_names = cache
                comfyui_model_names = set(cached_names)
        else:
            comfyui_model_names = set(comfyui_model_names)

        return local_names, comfyui_model_names

    def _refresh_visible_model_status_from_snapshots(
        self,
        local_names: set | None = None,
        comfyui_model_names: set | None = None,
    ) -> None:
        local_snapshot = set(local_names) if local_names is not None else None
        comfyui_snapshot = set(comfyui_model_names) if comfyui_model_names is not None else None
        if local_snapshot is not None:
            self._model_detail_local_names = local_snapshot
        if comfyui_snapshot is not None:
            self._model_detail_comfyui_model_names = comfyui_snapshot

        for card in list(getattr(self, "_model_cards", [])):
            try:
                if getattr(card, "_is_image_model", False):
                    if comfyui_snapshot is None:
                        continue
                    card.refresh_status(comfyui_model_names=comfyui_snapshot)
                else:
                    if local_snapshot is None and not getattr(card, "_is_utility_demo_model", False):
                        continue
                    card.refresh_status(local_names=local_snapshot)
                card.refresh_perf_badges()
            except Exception:
                pass
        self._update_model_detail(local_names=local_snapshot, comfyui_model_names=comfyui_snapshot)

    def _schedule_model_status_refresh(self, force_refresh: bool = False) -> None:
        if getattr(self, "_model_status_refresh_thread", None) and self._model_status_refresh_thread.is_alive():
            return
        self._model_status_refresh_generation += 1
        generation = self._model_status_refresh_generation

        def _worker() -> None:
            started_at = time.perf_counter()
            local_names = self._get_cached_local_names(force_refresh=force_refresh)
            comfyui_model_names = self._get_cached_comfyui_model_names(force_refresh=force_refresh)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self.after(
                0,
                lambda: self._apply_model_status_refresh(
                    generation, local_names, comfyui_model_names, elapsed_ms, force_refresh
                ),
            )

        if force_refresh:
            self.set_status("Refreshing model status …")
        thread = threading.Thread(target=_worker, name="ModelStatusRefresh", daemon=True)
        self._model_status_refresh_thread = thread
        thread.start()

    def _apply_model_status_refresh(
        self,
        generation: int,
        local_names: set,
        comfyui_model_names: set,
        elapsed_ms: float,
        force_refresh: bool,
    ) -> None:
        if generation != self._model_status_refresh_generation:
            return
        cards = list(getattr(self, "_model_cards", []))
        self._continue_model_status_apply(
            generation,
            cards,
            local_names,
            comfyui_model_names,
            elapsed_ms,
            force_refresh,
            index=0,
        )

    def _continue_model_status_apply(
        self,
        generation: int,
        cards: list,
        local_names: set,
        comfyui_model_names: set,
        elapsed_ms: float,
        force_refresh: bool,
        index: int,
    ) -> None:
        if generation != self._model_status_refresh_generation:
            return
        end = min(index + 24, len(cards))
        for card in cards[index:end]:
            try:
                card.refresh_status(local_names=local_names, comfyui_model_names=comfyui_model_names)
                card.refresh_perf_badges()
            except Exception:
                pass
        if end < len(cards):
            self.after(
                1,
                lambda: self._continue_model_status_apply(
                    generation,
                    cards,
                    local_names,
                    comfyui_model_names,
                    elapsed_ms,
                    force_refresh,
                    index=end,
                ),
            )
            return
        self._model_detail_local_names = set(local_names) if local_names is not None else None
        self._model_detail_comfyui_model_names = set(comfyui_model_names) if comfyui_model_names is not None else None
        self._update_model_detail(local_names=local_names, comfyui_model_names=comfyui_model_names)
        if elapsed_ms > 250:
            logger.info(f"Models page status refresh completed in {elapsed_ms:.0f} ms.")
        if force_refresh:
            self.set_status("Model list refreshed.")

    # v5: TTL-cached local model names — avoids redundant /api/tags calls
    def _get_cached_local_names(self, force_refresh: bool = False) -> set:
        """
        Return the current set of locally-installed Ollama model names.
        Cached for 5 s to dedupe HTTP traffic from rapid UI refreshes.
        """
        if not self.ollama_ok:
            return set()
        now = time.time()
        if (not force_refresh) and self._local_names_cache:
            cached_at, names = self._local_names_cache
            if now - cached_at < 5.0:
                return names
        try:
            names = self.ollama.local_model_names()
        except Exception:
            names = set()
        self._local_names_cache = (now, names)
        return names

    def _get_cached_comfyui_model_names(self, force_refresh: bool = False) -> set:
        """Return available ComfyUI checkpoint/diffusion model filenames.

        Cached per refresh so the Models page does not issue one HTTP model-list
        probe per image card.
        """
        now = time.time()
        cache = getattr(self, "_comfyui_model_names_cache", None)
        if (not force_refresh) and cache:
            cached_at, names = cache
            if now - cached_at < 5.0:
                return names
        if self.comfyui_ok:
            try:
                data = self.comfyui.get_model_list()
                if isinstance(data, dict):
                    names = set(data.get("checkpoints", [])) | set(data.get("diffusion_models", []))
                else:
                    names = set(data or [])
            except Exception:
                names = set(self._local_comfyui_model_files())
        else:
            names = set(self._local_comfyui_model_files())
        self._comfyui_model_names_cache = (now, names)
        return names

    def _reload_catalog(self):
        """Re-read models_catalog.json and refresh the model cards."""
        self._catalog_models = catalog.load_catalog()
        self._toolbox_model_by_id_cache = None
        self._runnable_toolbox_titles_cache = None
        self._refresh_chat_model_selector()
        self._schedule_model_card_population()
        self._validate_image_recommended_settings()
        self._validate_image_perf_profile()
        count = len(self._catalog_models)
        self.set_status(f"Catalog reloaded — {count} model{'s' if count != 1 else ''} loaded.")
        logger.info(f"Catalog reloaded from {catalog.CATALOG_FILE} ({count} models).")

    def _validate_image_recommended_settings(self) -> None:
        """Warn if any Image Generation catalog entry is missing the
        `recommended_settings` block.

        Image-gen models should declare their recommended sampler/scheduler/
        steps/cfg/size in `models_catalog.json` so the UI applies the right
        defaults when the user picks the model. Entries without this block
        fall back to legacy filename-substring heuristics — works, but means
        any custom values from the docs go unused. This warning surfaces gaps
        so future model additions stay catalog-driven.
        """
        try:
            missing = []
            for m in self._catalog_models:
                if m.get("category") != "Image Generation":
                    continue
                if not isinstance(m.get("recommended_settings"), dict):
                    missing.append(m.get("id") or m.get("comfyui_model") or "?")
            if missing:
                ids = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
                logger.warning(
                    f"Catalog: {len(missing)} image-gen model(s) are missing "
                    f"'recommended_settings'; they will use heuristic defaults: "
                    f"{ids}. Add a 'recommended_settings' block (width/height/"
                    f"aspect/sampler/scheduler/steps/cfg/cfg_locked/family_label) "
                    f"per entry in models_catalog.json to make defaults explicit."
                )
        except Exception as exc:
            logger.debug(f"recommended_settings validation skipped: {exc}")

    def _validate_image_perf_profile(self) -> None:
        """v5.3: warn at startup/reload if any Image Generation entry lacks
        a ``perf_profile`` block. Without it, the Models page sectioning
        treats the model as ``general/alternative`` and shows no speed,
        quality, bucket, or fit badges. Encourages keeping new model
        additions catalog-driven end-to-end.
        """
        try:
            missing = []
            partial = []
            REQUIRED = {"speed_tier", "quality_tier", "category_bucket",
                        "recommendation"}
            for m in self._catalog_models:
                if m.get("category") != "Image Generation":
                    continue
                p = m.get("perf_profile")
                if not isinstance(p, dict):
                    missing.append(m.get("id") or m.get("comfyui_model") or "?")
                elif not REQUIRED.issubset(p.keys()):
                    short = [k for k in REQUIRED if k not in p]
                    partial.append(f"{m.get('id', '?')}({','.join(sorted(short))})")
            if missing:
                ids = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
                logger.warning(
                    f"Catalog: {len(missing)} image-gen model(s) are missing "
                    f"'perf_profile'; they will be ungrouped on the Models "
                    f"page: {ids}. Add a 'perf_profile' block (speed_tier/"
                    f"quality_tier/category_bucket/recommendation/speed_label/"
                    f"notes) per entry in models_catalog.json."
                )
            if partial:
                ids = ", ".join(partial[:10]) + (" …" if len(partial) > 10 else "")
                logger.warning(
                    f"Catalog: {len(partial)} image-gen model(s) have incomplete "
                    f"'perf_profile' blocks (missing required keys): {ids}."
                )
        except Exception as exc:
            logger.debug(f"perf_profile validation skipped: {exc}")

    def _open_catalog_file(self):
        """Open models_catalog.json in the system default editor."""
        path = str(catalog.CATALOG_FILE)
        if not catalog.CATALOG_FILE.exists():
            catalog.ensure_catalog_file()
        try:
            self._open_path(Path(path))
        except Exception as exc:
            messagebox.showerror("Cannot open file",
                                 f"Could not open {path}:\n{exc}", parent=self)

    def _reset_catalog_to_defaults(self):
        ok = messagebox.askyesno(
            "Reset catalog",
            "Overwrite models_catalog.json with the built-in default model list?\n"
            "Any models you added or modified will be lost.",
            parent=self,
        )
        if not ok:
            return
        if not catalog.save_catalog(list(catalog.MODELS)):
            messagebox.showerror("Reset failed", "Could not write models_catalog.json.", parent=self)
            return
        self._reload_catalog()
        self.set_status("Catalog reset to built-in defaults.")

    # ── Optional SKU file management ──────────────────────────────────────────

    def _reload_optional_skus(self):
        """Re-read optional SKUs and update all dependent state and UI."""
        sku_cfg = system_info.load_optional_sku_config()
        self._optional_sku_feature = sku_cfg.get("feature", {})
        self._optional_skus = sku_cfg.get("skus", [])
        self._optional_skus_enabled = bool(self._optional_skus)
        self._apply_optional_skus_to_modules()
        # Re-inject the detected SKU if it is not in the reloaded file
        if self._optional_sku:
            self._inject_local_sku(self._optional_sku)
        self._refresh_sku_filter_values()
        self._schedule_model_card_population()
        self._refresh_bench_profile_values(preserve_selection=True)
        count = len(self._optional_skus)
        self.set_status(f"SKU definitions reloaded — {count} entr{'ies' if count != 1 else 'y'}.")
        logger.info(f"SKU definitions reloaded from {system_info.OPTIONAL_SKUS_FILE} ({count} entries).")

    def _open_optional_skus_file(self):
        """Open the optional SKU file in the system default editor."""
        path = str(system_info.OPTIONAL_SKUS_FILE)
        if not system_info.OPTIONAL_SKUS_FILE.exists():
            messagebox.showinfo("SKU file not found", "SKU definition file is not available.", parent=self)
            return
        try:
            self._open_path(Path(path))
        except Exception as exc:
            messagebox.showerror("Cannot open file",
                                 f"Could not open {path}:\n{exc}", parent=self)

    def _reset_optional_skus_to_defaults(self):
        messagebox.showinfo(
            "No built-in defaults",
            "Optional SKU defaults are not stored in the app. Edit the private SKU file directly.",
            parent=self,
        )

    # ── Documentation ─────────────────────────────────────────────────────────

    def _open_docs(self):
        """Open the local documentation in the default web browser."""
        docs_path = Path(__file__).parent.parent / "docs" / "index.html"
        if docs_path.exists():
            webbrowser.open(docs_path.as_uri())
            self.set_status("Documentation opened in browser.")
            self.after(3000, lambda: self.set_status("Ready."))
        else:
            # Fall back to README if docs/index.html is missing
            readme = Path(__file__).parent.parent / "README.md"
            if readme.exists():
                self._open_path(readme)
                self.set_status("README opened in browser.")
                self.after(3000, lambda: self.set_status("Ready."))
            else:
                messagebox.showinfo(
                    "Documentation not found",
                    "Could not locate docs/index.html or README.md.",
                    parent=self,
                )

    # ── ComfyUI orphan cleanup ─────────────────────────────────────────────────

    def _kill_orphan_comfyui_processes(self) -> list:
        """Kill any python processes running ComfyUI/main.py that this instance did not start."""
        own_pid = (
            self.comfyui_process.pid
            if (self.comfyui_process and self.comfyui_process.poll() is None)
            else None
        )
        killed = []

        if sys.platform == "darwin":
            # macOS: use psutil to find ComfyUI processes
            try:
                import psutil
                for proc in psutil.process_iter(["pid", "cmdline"]):
                    try:
                        cmdline = proc.info.get("cmdline") or []
                        cmd_str = " ".join(cmdline)
                        if "ComfyUI" in cmd_str and "main.py" in cmd_str:
                            pid = proc.info["pid"]
                            if pid == own_pid:
                                continue
                            import signal
                            os.kill(pid, signal.SIGTERM)
                            logger.info(f"ComfyUI: killed orphan process PID {pid}")
                            killed.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                        pass
            except Exception as e:
                logger.warning(f"ComfyUI: orphan scan failed: {e}")
        else:
            # Windows: use PowerShell + taskkill
            import subprocess as _sp
            try:
                result = _sp.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_Process -Filter \"CommandLine LIKE '%ComfyUI%main.py%'\" "
                     "| Select-Object -ExpandProperty ProcessId"],
                    capture_output=True, text=True, timeout=10,
                )
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pid = int(line)
                    except ValueError:
                        continue
                    if pid == own_pid:
                        continue
                    try:
                        _sp.run(["taskkill", "/F", "/PID", str(pid)],
                                capture_output=True, timeout=5)
                        logger.info(f"ComfyUI: killed orphan process PID {pid}")
                        killed.append(pid)
                    except Exception as e:
                        logger.warning(f"ComfyUI: could not kill orphan PID {pid}: {e}")
            except Exception as e:
                logger.warning(f"ComfyUI: orphan scan failed: {e}")
        return killed

    def _startup_kill_orphan_comfyui(self):
        """Called once at startup to clean up any orphaned ComfyUI processes."""
        def _worker():
            t0 = time.perf_counter()
            killed = self._kill_orphan_comfyui_processes()
            logger.info(f"Startup: ComfyUI orphan scan took {(time.perf_counter() - t0) * 1000:.0f} ms")
            if killed:
                logger.info(f"ComfyUI: cleaned up {len(killed)} orphan process(es) at startup: PIDs {killed}")
            else:
                logger.debug("ComfyUI: no orphan processes found at startup")
        threading.Thread(target=_worker, daemon=True).start()

    # ── Low Resources startup check ──────────────────────────────────────────

    def _check_low_resources_startup(self):
        """At startup, if low resources mode is off but hardware looks constrained, offer to enable it."""
        if self.cfg.get("low_resources_mode"):
            return  # already enabled

        def _check():
            models_path = self.cfg.get("models_dir", ".")
            suggest, reason = resource_manager.should_suggest_low_resources(models_path)
            if suggest:
                self.after(0, lambda: self._prompt_low_resources(reason))

        threading.Thread(target=_check, daemon=True).start()

    def _prompt_low_resources(self, reason: str):
        """Show a dialog suggesting Low Resources Mode."""
        answer = messagebox.askyesno(
            "Enable Low Resources Mode?",
            f"This device appears to have limited resources:\n\n"
            f"  {reason}\n\n"
            f"Low Resources Mode checks disk space and RAM before\n"
            f"downloading or running models, and can automatically\n"
            f"free space in batch mode by removing unused models.\n\n"
            f"Enable Low Resources Mode?",
            parent=self,
        )
        if answer:
            self.cfg["low_resources_mode"] = True
            self._low_res_var.set(True)
            if not config.save(self.cfg):
                logger.error("Could not persist Low Resources Mode setting")
            logger.info("Low Resources Mode enabled (suggested at startup)")
            self.set_status("Low Resources Mode enabled.")

    # ── Optional SKU detection ────────────────────────────────────────────────

    def _detect_optional_device_async(self):
        def _detect():
            if not self._optional_skus_enabled or sys.platform == "darwin":
                sku = system_info.build_local_sku()
                self._sku_is_auto = True
            else:
                sku = system_info.detect_optional_sku()
                if sku is not None:
                    self._sku_is_auto = False
                else:
                    # Not a known optional SKU — build a synthetic SKU from local hardware
                    # so this machine still appears as a filter option
                    sku = system_info.build_local_sku()
                    self._sku_is_auto = True
            self._optional_sku = sku
            self.after(0, self._on_optional_device_detected)
        threading.Thread(target=_detect, daemon=True).start()

    def _apply_device_labels(self):
        """Replay detected SKU/device status into currently-built widgets."""
        if hasattr(self, "_device_banner_label") and self._device_banner_label is not None:
            self._device_banner_label.configure(
                text=self._device_display_text,
                text_color=self._device_display_color,
            )
        if hasattr(self, "_device_sidebar_label") and self._device_sidebar_label is not None:
            self._device_sidebar_label.configure(
                text=self._device_sidebar_text,
                text_color=self._device_sidebar_color,
            )

    def _on_optional_device_detected(self):
        sku = self._optional_sku
        if not sku:
            self._device_display_text = "Not detected — using all local hardware defaults"
            self._device_display_color = TEXT_MUTED
            self._device_sidebar_text = self._sku_text("not_detected_text", "This Device: not detected")
            self._device_sidebar_color = TEXT_MUTED
            self._apply_device_labels()
            return

        if self._sku_is_auto:
            # Local machine — rename to "This Device" for the filter button,
            # but keep the real hardware name for banner / sidebar display.
            real_name = sku["name"]
            sku["name"] = "This Device"
            display = "This Device"
            banner_color  = LINK_TEXT
            sidebar_text  = "This Device"
            sidebar_color = LINK_TEXT
            logger.info(f"Local machine detected: {real_name} → 'This Device' (auto-generated SKU)")
        else:
            # Optional SKU profile from the private SKU file. Hardware specs
            # (GPU model, VRAM, RAM, vCPU) are intentionally not exposed in
            # user-facing labels — only the profile name is shown.
            display = f"{self._sku_text('enabled_label', 'SKU')} {sku['name']}"
            banner_color  = SUCCESS_TEXT
            sidebar_text  = f"{self._sku_text('enabled_label', 'SKU')} {sku['name']}"
            sidebar_color = SUCCESS_TEXT
            logger.info(f"Optional SKU detected: {sku['name']} ({sku.get('vm_size_pattern', '')})")

        # If this SKU is not in the loaded list, inject it so it appears in the filter
        self._inject_local_sku(sku)

        # Auto-select this machine's SKU in the filter, then set category and refresh
        self._optional_filter_var.set(sku["name"])
        self._update_category_for_device()

        self._device_display_text = display
        self._device_display_color = banner_color
        self._device_sidebar_text = sidebar_text
        self._device_sidebar_color = sidebar_color
        self._apply_device_labels()
        if (
            hasattr(self, "_bench_profile_var")
            and not (getattr(self, "_bench_thread", None) and self._bench_thread.is_alive())
        ):
            self._bench_profile_var.set(self._bench_profile_for_sku(sku))
            self._apply_bench_profile_method_defaults()
        self._refresh_bench_profile_values(preserve_selection=True)

    def _update_category_for_device(self, vram_gb: int | None = None, ram_gb: int | None = None):
        """Pre-select the highest category this device/SKU can run; mark Image Generation if unavailable."""
        if vram_gb is None:
            dev      = self._optional_sku or {}
            vram_gb  = dev.get("vram_gb", 0)
            ram_gb   = dev.get("ram_gb", 0)
        dev_vram = vram_gb
        dev_ram  = ram_gb or 0

        # Check whether any image generation model is runnable.
        # CPU path (either user toggled CPU mode, or device has no GPU at all):
        # gate on RAM threshold and require an explicitly cpu_viable model.
        cpu_only_device = dev_vram == 0
        cpu_path = getattr(self, "_comfyui_force_cpu", False) or cpu_only_device
        if cpu_path:
            _avail_ram = (self._optional_sku or {}).get("ram_gb", 0) or dev_ram
            can_image_gen = (
                _avail_ram >= catalog.IMAGE_GEN_MIN_CPU_RAM_GB
                and any(
                    catalog.is_cpu_viable_image_model(m)
                    for m in self._catalog_models
                )
            )
        else:
            can_image_gen = dev_vram > 0 and any(
                m.get("backend") == "comfyui" and m.get("min_vram_gb", 0) <= dev_vram
                for m in self._catalog_models
            )

        # Style the Image Generation category button and sidebar nav button
        img_btn     = getattr(self, "_type_btns", {}).get("Image Gen") or self._cat_btns.get("Image Generation")
        img_nav_btn = self._nav_btns.get("image_gen")
        if can_image_gen:
            if img_btn is not None:
                img_btn.configure(text="Image Gen", text_color=TEXT_PRIMARY)
            if img_nav_btn is not None:
                img_nav_btn.configure(text_color=TEXT_PRIMARY)
            self._init_comfyui_if_needed()
        else:
            if img_btn is not None:
                img_btn.configure(text="Image Gen ✕", text_color=TEXT_DISABLED)
            if img_nav_btn is not None:
                img_nav_btn.configure(text_color=TEXT_DISABLED)
            if cpu_only_device:
                self._set_comfyui_status(
                    f"needs {catalog.IMAGE_GEN_MIN_CPU_RAM_GB} GB RAM for CPU mode", TEXT_MUTED
                )
            else:
                self._set_comfyui_status("no GPU", TEXT_MUTED)

        # Compute the set of runnable non-image categories, including local
        # adapter demos such as OCR, speech, embeddings, and document AI.
        text_cat_order = [
            "Ultra Small", "Small", "Medium", "Large", "Extra Large",
            "Vision", "Speech", "Embeddings", "Document AI",
        ]
        runnable_cats = {
            cat for cat in text_cat_order
            if any(
                m.get("category") == cat
                and m.get("backend") != "comfyui"
                and (
                    (dev_vram > 0 and m.get("min_vram_gb", 0) <= dev_vram)
                    or (dev_vram == 0 and m.get("min_ram_gb", 0) <= dev_ram)
                )
                for m in self._catalog_models
            )
        }

        if not self._user_set_cat_filter:
            if can_image_gen and runnable_cats == set(text_cat_order):
                # Can run everything — show All
                self._selected_cats = set()
            elif can_image_gen:
                self._selected_cats = runnable_cats | {"Image Generation"}
            else:
                self._selected_cats = runnable_cats
            logger.debug(f"_update_category_for_device: auto-set _selected_cats={self._selected_cats}")
        else:
            logger.debug(f"_update_category_for_device: user filter active — _selected_cats={self._selected_cats} preserved")

        self._refresh_cat_btn_states()
        self._schedule_model_card_population()

    # ── Toolbox page ──────────────────────────────────────────────────────────

    _TOOLBOX_WORKFLOWS = {
        "transcribe": {
            "title": "Transcribe",
            "description": "Turn local audio into text with Whisper. Pick a language or leave it on Auto-detect. First run may install packages or download model files.",
            "models": ["whisper-large-v3-turbo", "whisper-v3-turbo-gpu"],
            "input": "file",
            "filetypes": [("Audio", "*.wav *.mp3 *.m4a *.flac"), ("All files", "*.*")],
            "deps": ["soundfile", "torch", "transformers", "scipy", "accelerate", "safetensors", "huggingface_hub", "hf_xet"],
            "language_picker": "whisper",
        },
        "read": {
            "title": "Read image text",
            "description": "Best-effort OCR for screenshots or printed images using Florence-2. First run may install packages or download model files.",
            "models": ["florence-2-base", "trocr-large-printed", "trocr-base-printed"],
            "input": "file",
            "filetypes": [("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
            "deps": ["PIL", "torch", "transformers", "accelerate", "safetensors", "einops", "timm", "huggingface_hub", "hf_xet"],
        },
        "tables": {
            "title": "Extract table (fast)",
            "description": "Read a table out of an image using a fast multimodal Ollama model (MiniCPM-V). Returns tab-separated values you can paste into a spreadsheet. Requires Ollama to be running and the minicpm-v model to be pulled.",
            "models": ["minicpm-v-vision"],
            "input": "file",
            "filetypes": [("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
            "deps": [],
        },
        "extract_table": {
            "title": "Extract table (best)",
            "description": "High-fidelity table extraction with GOT-OCR 2.0. Returns LaTeX (saved alongside) plus a best-effort TSV. Slower and needs more VRAM than the fast option; first run downloads the model.",
            "models": ["got-ocr2"],
            "input": "file",
            "filetypes": [("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
            "deps": ["PIL", "torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_xet"],
        },
        "speak": {
            "title": "Speak",
            "description": "Generate a local WAV from typed text using the Piper offline TTS engine. Pick a voice / language from the dropdown. First run downloads the selected voice (~30-150 MB).",
            "models": ["piper-tts"],
            "input": "text",
            "sample": "Welcome. This voice was generated locally on your computer using a small offline model.",
            "deps": ["piper", "onnxruntime", "soundfile", "huggingface_hub", "hf_xet"],
            "language_picker": "piper",
        },
        "search": {
            "title": "Semantic search",
            "description": "Find the lines that best match your question by meaning - not just matching words. Type a question, add a few candidate sentences, and they're ranked by how closely each one answers it. First run may install packages or download model files.",
            "models": ["all-minilm"],
            "input": "search",
            "sample": "Query: how do I get my money back for a package that showed up late?\n\nOrders that arrive after the guaranteed date qualify for a full refund within five business days.\nYou can track your delivery using the link in your shipping confirmation email.\nOur customer service line is open weekdays from 9 a.m. to 6 p.m.\nGift cards can be reloaded online or at any store register.",
            "deps": ["sentence_transformers", "torch", "transformers", "huggingface_hub", "hf_xet"],
        },
        "describe": {
            "title": "Describe image",
            "description": "Caption or answer questions about an image. First run may install packages or download model files.",
            "models": ["florence-2-base"],
            "input": "file",
            "filetypes": [("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
            "deps": ["PIL", "torch", "transformers", "accelerate", "safetensors", "einops", "timm", "huggingface_hub", "hf_xet"],
        },
    }
    def _build_toolbox_page(self):
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        self._pages["toolbox"] = page
        self._toolbox_cards = {}
        self._selected_toolbox_workflow_id = "transcribe"
        self._toolbox_activity_visible = True

        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hdr, text="Toolbox", font=ctk.CTkFont(size=24, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            hdr,
            text="Run OCR, transcription, speech, embeddings, table detection, and image description locally.",
            text_color=TEXT_MUTED, anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        banner = ctk.CTkFrame(page, corner_radius=10, fg_color=INPUT_SURFACE, border_width=1, border_color=BORDER_STRONG)
        banner.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))
        banner.grid_columnconfigure(0, weight=1)
        self._toolbox_banner = banner
        self._toolbox_banner_label = ctk.CTkLabel(
            banner,
            text="Toolbox is ready. First runs may install packages or download model files.",
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self._toolbox_banner_label.grid(row=0, column=0, sticky="ew", padx=14, pady=8)
        self._toolbox_banner_progress = ctk.CTkProgressBar(banner, width=160, height=8, mode="indeterminate")
        self._toolbox_banner_progress.grid(row=0, column=1, padx=14, pady=8)
        self._toolbox_banner_progress.grid_remove()

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 14))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        self._toolbox_page_body = body

        browser = ctk.CTkScrollableFrame(body, width=350, fg_color="transparent")
        browser.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        browser.grid_columnconfigure(0, weight=1)
        self._toolbox_browser = browser

        detail = ctk.CTkFrame(body, corner_radius=14, fg_color=SURFACE_CARD, border_width=1, border_color=BORDER_STRONG)
        detail.grid(row=0, column=1, sticky="nsew")
        detail.grid_columnconfigure(0, weight=1)
        detail.grid_rowconfigure(6, weight=1)
        self._toolbox_detail_frame = detail

        row_index = 0
        groups = [
            ("Audio", ["transcribe", "speak"]),
            ("Vision & Documents", ["read", "tables", "extract_table", "describe"]),
            ("Text", ["search"]),
        ]
        for group_title, workflow_ids in groups:
            ctk.CTkLabel(
                browser, text=group_title.upper(),
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=TEXT_MUTED, anchor="w",
            ).grid(row=row_index, column=0, sticky="ew", padx=8, pady=(10 if row_index else 0, 4))
            row_index += 1
            for workflow_id in workflow_ids:
                row = self._build_toolbox_workflow_row(
                    browser, workflow_id, self._TOOLBOX_WORKFLOWS[workflow_id]
                )
                row.grid(row=row_index, column=0, sticky="ew", padx=4, pady=4)
                row_index += 1

        self._apply_toolbox_browser_density()
        self._build_toolbox_detail_panel(detail)
        self._select_toolbox_workflow(self._selected_toolbox_workflow_id, save_current=False)
        self._refresh_toolbox_cards()

    def _toolbox_model_entry(self, spec: dict) -> dict | None:
        catalog_key = id(self._catalog_models)
        cached = getattr(self, "_toolbox_model_by_id_cache", None)
        if not cached or cached[0] != catalog_key:
            cached = (catalog_key, {m.get("id"): m for m in self._catalog_models})
            self._toolbox_model_by_id_cache = cached
        by_id = cached[1]
        for mid in spec.get("models", []):
            entry = by_id.get(mid)
            if entry:
                return entry
        return None

    def _toolbox_can_run(self, entry: dict | None) -> tuple[bool, str]:
        if entry is None:
            return False, "model is not in the catalog"
        dev = self._optional_sku or {}
        ram_info = system_info.get_ram_info()
        ram_gb = float(dev.get("ram_gb", 0) or ram_info.get("total_gb", 0) or (ram_info.get("total_mb", 0) / 1024) or 0)
        vram_gb = float(dev.get("vram_gb", 0) or self._active_device_vram_gb() or 0)
        if ram_gb and entry.get("min_ram_gb", 0) > ram_gb:
            return False, f"needs {entry.get('min_ram_gb')} GB RAM"
        # Ollama-backed entries (ollama_tag set) let Ollama decide CPU vs
        # GPU split transparently, so the static min_vram_gb threshold
        # isn't useful here. Skip it.
        ollama_tag = (entry.get("ollama_tag") or "").strip()
        if not ollama_tag:
            if entry.get("min_vram_gb", 0) and vram_gb and entry.get("min_vram_gb", 0) > vram_gb:
                return False, f"needs {entry.get('min_vram_gb')} GB VRAM"
        return True, "ready"

    def _toolbox_missing_deps(self, spec: dict) -> list[str]:
        return [dep for dep in spec.get("deps", []) if importlib.util.find_spec(dep) is None]

    def _phase1_cache_dir(self) -> Path:
        return phase1_adapters.configure_local_hf_environment()

    def _phase1_model_cached(self, entry: dict) -> bool:
        repo = entry.get("hf_repo") or entry.get("onnx_repo")
        if not repo:
            return True
        cache_name = "models--" + repo.replace("/", "--")
        cache_dir = self._phase1_cache_dir()
        return (cache_dir / "hub" / cache_name).exists() or (cache_dir / cache_name).exists()

    def _runnable_toolbox_titles(self) -> list[str]:
        sku = self._optional_sku or {}
        cache_key = (
            id(self._catalog_models),
            sku.get("name"),
            sku.get("ram_gb"),
            sku.get("vram_gb"),
            self._active_device_vram_gb(),
        )
        now = time.monotonic()
        cached = getattr(self, "_runnable_toolbox_titles_cache", None)
        if cached and now - cached[0] < 30 and cached[1] == cache_key:
            return list(cached[2])
        titles = []
        for spec in self._TOOLBOX_WORKFLOWS.values():
            entry = self._toolbox_model_entry(spec)
            ok, _reason = self._toolbox_can_run(entry)
            if ok:
                titles.append(spec["title"])
        self._runnable_toolbox_titles_cache = (now, cache_key, list(titles))
        return titles

    def _build_toolbox_workflow_row(self, parent, workflow_id: str, spec: dict):
        row = ctk.CTkFrame(parent, corner_radius=12, fg_color=INPUT_SURFACE, border_width=1, border_color=BORDER_STRONG)
        row.grid_columnconfigure(0, weight=1)
        try:
            row.configure(takefocus=1)
        except Exception:
            pass
        title = ctk.CTkLabel(
            row, text=spec["title"], font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
        )
        title.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))
        status = ctk.CTkLabel(
            row, text="Checking", text_color=TEXT_MUTED, fg_color=SURFACE_CARD,
            corner_radius=10, padx=8, pady=2,
        )
        status.grid(row=0, column=1, sticky="e", padx=12, pady=(10, 0))
        desc = ctk.CTkLabel(
            row, text=self._toolbox_row_summary(workflow_id, spec),
            font=ctk.CTkFont(size=11), text_color=TEXT_MUTED,
            anchor="w", justify="left", wraplength=275,
        )
        desc.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 10))
        model = ctk.CTkLabel(
            row, text="", font=ctk.CTkFont(size=10), text_color=TEXT_MUTED,
            anchor="w", justify="left", wraplength=275,
        )
        model.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))

        for widget in (row, title, desc, model, status):
            widget.bind("<Button-1>", lambda _event, wid=workflow_id: self._select_toolbox_workflow(wid))
        row.bind("<Return>", lambda _event, wid=workflow_id: self._select_toolbox_workflow(wid))
        row.bind("<space>", lambda _event, wid=workflow_id: self._select_toolbox_workflow(wid))
        row.bind("<FocusIn>", lambda _event, frame=row: frame.configure(border_color=LINK_TEXT, border_width=2))
        row.bind(
            "<FocusOut>",
            lambda _event, wid=workflow_id, frame=row: frame.configure(
                border_color=self._IG_ACCENT if wid == self._selected_toolbox_workflow_id else BORDER_STRONG,
                border_width=2 if wid == self._selected_toolbox_workflow_id else 1,
            ),
        )

        path_var = ctk.StringVar(value="")
        self._toolbox_cards[workflow_id] = {
            "frame": row,
            "spec": spec,
            "title": title,
            "path_var": path_var,
            "text_value": spec.get("sample", ""),
            "result_text": "Final result appears here.",
            "activity_text": "Progress appears here.",
            "output_path": None,
            "status": status,
            "model": model,
            "summary": desc,
            "status_text": "Checking",
            "status_color": TEXT_MUTED,
            "readiness_text": "Readiness: checking ...",
            "readiness_color": TEXT_MUTED,
            "requirements_text": "",
            "action_text": f"Run {spec['title']}",
            "action_state": "normal",
        }
        return row

    def _toolbox_row_summary(self, workflow_id: str, spec: dict) -> str:
        summaries = {
            "transcribe": "Audio file -> local transcript",
            "speak": "Typed text -> generated WAV (offline Piper voice)",
            "read": "Image -> best-effort OCR text",
            "tables": "Image -> TSV via Ollama (minicpm-v)",
            "extract_table": "Image -> TSV + LaTeX via GOT-OCR 2.0",
            "describe": "Image -> caption or visual answer",
            "search": "Question + candidate lines -> ranked matches",
        }
        return summaries.get(workflow_id, spec.get("description", ""))

    def _toolbox_row_model_text(self, spec: dict, entry: dict | None) -> str:
        model_ids = [str(mid).strip() for mid in spec.get("models", []) if str(mid).strip()]
        if not model_ids:
            return ""
        if entry:
            primary = str(entry.get("name") or entry.get("id") or model_ids[0]).strip()
        else:
            primary = model_ids[0]
        # Keep compact "model-right" labels short so tiles stay dense.
        max_primary = 30
        if len(primary) > max_primary:
            primary = f"{primary[: max_primary - 3]}..."
        extras = max(0, len(model_ids) - 1)
        suffix = f" +{extras}" if extras > 0 else ""
        return f"Model: {primary}{suffix}"

    def _apply_toolbox_browser_density(self) -> None:
        title_font = ctk.CTkFont(size=12, weight="bold")
        summary_font = ctk.CTkFont(size=10)
        model_font = ctk.CTkFont(size=10)

        for parts in getattr(self, "_toolbox_cards", {}).values():
            frame = parts.get("frame")
            title = parts.get("title")
            status = parts.get("status")
            summary = parts.get("summary")
            model = parts.get("model")

            if self._widget_is_alive(frame):
                try:
                    frame.configure(corner_radius=10)
                    frame.grid_columnconfigure(0, weight=1)
                    frame.grid_columnconfigure(1, weight=0)
                except TclError:
                    pass

            if self._widget_is_alive(title):
                try:
                    title.configure(font=title_font)
                    title.grid_configure(padx=12, pady=(6, 0))
                except TclError:
                    pass

            if self._widget_is_alive(status):
                try:
                    status.grid_configure(padx=12, pady=(6, 0))
                except TclError:
                    pass

            if self._widget_is_alive(summary):
                try:
                    summary_wrap = 190
                    summary.configure(font=summary_font, wraplength=summary_wrap, anchor="w", justify="left")
                    summary.grid_configure(
                        row=1,
                        column=0,
                        columnspan=1,
                        sticky="ew",
                        padx=(12, 8),
                        pady=(1, 6),
                    )
                except TclError:
                    pass

            if self._widget_is_alive(model):
                try:
                    model.configure(font=model_font, wraplength=156, anchor="e", justify="right")
                    model.grid(
                        row=1,
                        column=1,
                        columnspan=1,
                        sticky="e",
                        padx=(8, 12),
                        pady=(1, 6),
                    )
                except TclError:
                    pass

    def _build_toolbox_detail_panel(self, detail):
        header = ctk.CTkFrame(detail, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(
            header, text="", font=ctk.CTkFont(size=22, weight="bold"), anchor="w"
        )
        title.grid(row=0, column=0, sticky="ew")
        status = ctk.CTkLabel(
            header, text="Checking", text_color=TEXT_MUTED, fg_color=INPUT_SURFACE,
            corner_radius=10, padx=10, pady=2,
        )
        status.grid(row=0, column=1, sticky="e", padx=(10, 0))
        help_btn = ctk.CTkButton(
            header, text="Help", width=64,
            **self._outline_button_style(),
            command=self._open_docs,
        )
        help_btn.grid(row=0, column=2, sticky="e", padx=(8, 0))
        model = ctk.CTkLabel(
            header, text="", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED, anchor="w"
        )
        model.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        description = ctk.CTkLabel(
            detail, text="", text_color=TEXT_MUTED, anchor="w",
            justify="left", wraplength=720,
        )
        description.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))

        req_row = ctk.CTkFrame(detail, fg_color="transparent")
        req_row.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))
        req_row.grid_columnconfigure((0, 1, 2), weight=1)
        packages = self._toolbox_requirement_chip(req_row, "Packages")
        packages.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        model_files = self._toolbox_requirement_chip(req_row, "Model files")
        model_files.grid(row=0, column=1, sticky="ew", padx=6)
        hardware = self._toolbox_requirement_chip(req_row, "Hardware")
        hardware.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        file_frame = ctk.CTkFrame(detail, fg_color="transparent")
        file_frame.grid_columnconfigure(0, weight=1)
        file_entry = ctk.CTkEntry(file_frame, placeholder_text="Choose a local file...")
        file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        file_browse = ctk.CTkButton(
            file_frame, text="Browse", width=86, **self._outline_button_style(),
            command=lambda: self._toolbox_browse_file(getattr(self, "_selected_toolbox_workflow_id", "")),
        )
        file_browse.grid(row=0, column=1)

        text_input = ctk.CTkTextbox(
            detail, height=110, wrap="word", fg_color=INPUT_SURFACE, border_color=BORDER_STRONG
        )

        # Optional language picker (used by Transcribe and Speak). One
        # frame holds a label + an option menu; the items in the menu
        # swap based on the active workflow. Hidden by default.
        from src import workflows as _wf

        language_frame = ctk.CTkFrame(detail, fg_color="transparent")
        language_frame.grid_columnconfigure(1, weight=1)
        language_label = ctk.CTkLabel(language_frame, text="Language:", anchor="w", text_color=TEXT_MUTED)
        language_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        whisper_choices = list(_wf.WHISPER_LANGUAGES.keys())
        piper_choices = list(_wf.PIPER_VOICES.keys())
        # StringVars persist user choice across panel re-renders.
        self._toolbox_transcribe_language = ctk.StringVar(value=whisper_choices[0])
        self._toolbox_speak_language = ctk.StringVar(value=piper_choices[0])
        language_menu = ctk.CTkOptionMenu(
            language_frame,
            values=whisper_choices,
            variable=self._toolbox_transcribe_language,
            width=240,
        )
        language_menu.grid(row=0, column=1, sticky="w")
        language_frame.grid_remove()

        action_row = ctk.CTkFrame(detail, fg_color="transparent")
        action_row.grid(row=4, column=0, sticky="ew", padx=18, pady=(10, 10))
        action_row.grid_columnconfigure(3, weight=1)
        run_btn = ctk.CTkButton(
            action_row, text="Run", width=150,
            **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
            command=lambda: self._run_toolbox_workflow(getattr(self, "_selected_toolbox_workflow_id", "")),
        )
        run_btn.grid(row=0, column=0, sticky="w")
        open_file = ctk.CTkButton(
            action_row, text="Open output", width=105, **self._outline_button_style(),
            command=lambda: self._open_toolbox_output_file(getattr(self, "_selected_toolbox_workflow_id", "")),
        )
        open_file.grid(row=0, column=1, padx=(8, 0))
        open_file.grid_remove()
        open_folder = ctk.CTkButton(
            action_row, text="Open folder", width=105, **self._outline_button_style(),
            command=lambda: self._open_toolbox_output_folder(getattr(self, "_selected_toolbox_workflow_id", "")),
        )
        open_folder.grid(row=0, column=2, padx=(8, 0))
        open_folder.grid_remove()

        ctk.CTkLabel(
            detail, text="Final result", font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        ).grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 4))
        output = ctk.CTkTextbox(detail, height=150, wrap="word", fg_color=INPUT_SURFACE, border_color=BORDER_STRONG)
        output.grid(row=6, column=0, sticky="nsew", padx=18, pady=(0, 10))
        output.insert("1.0", "Final result appears here.")

        activity_header = ctk.CTkFrame(detail, fg_color="transparent")
        activity_header.grid(row=7, column=0, sticky="ew", padx=18, pady=(0, 4))
        activity_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            activity_header, text="Activity details", font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w")
        activity_toggle = ctk.CTkButton(
            activity_header,
            text="Hide activity" if self._toolbox_activity_visible else "Show activity",
            width=115,
            **self._outline_button_style(), command=self._toggle_toolbox_activity,
        )
        activity_toggle.grid(row=0, column=1, sticky="e")
        activity = ctk.CTkTextbox(detail, height=95, wrap="word", fg_color=INPUT_SURFACE, border_color=BORDER_STRONG)
        activity.grid(row=8, column=0, sticky="ew", padx=18, pady=(0, 16))
        if not self._toolbox_activity_visible:
            activity.grid_remove()
        activity.insert("1.0", "Progress appears here.")

        self._toolbox_detail = {
            "title": title,
            "status": status,
            "model": model,
            "description": description,
            "packages": packages,
            "model_files": model_files,
            "hardware": hardware,
            "file_frame": file_frame,
            "file_entry": file_entry,
            "file_browse": file_browse,
            "text_input": text_input,
            "language_frame": language_frame,
            "language_label": language_label,
            "language_menu": language_menu,
            "run": run_btn,
            "open_file": open_file,
            "open_folder": open_folder,
            "output": output,
            "activity": activity,
            "activity_toggle": activity_toggle,
        }

    def _toolbox_requirement_chip(self, parent, title: str):
        chip = ctk.CTkLabel(
            parent, text=f"{title}: checking", text_color=TEXT_MUTED,
            fg_color=INPUT_SURFACE, corner_radius=10, padx=10, pady=8,
            anchor="w", justify="left",
        )
        return chip

    def _toggle_toolbox_activity(self):
        self._toolbox_activity_visible = not bool(getattr(self, "_toolbox_activity_visible", False))
        detail = getattr(self, "_toolbox_detail", {})
        activity = detail.get("activity")
        toggle = detail.get("activity_toggle")
        if activity is None or toggle is None:
            return
        if self._toolbox_activity_visible:
            activity.grid()
            toggle.configure(text="Hide activity")
        else:
            activity.grid_remove()
            toggle.configure(text="Show activity")

    def _save_toolbox_detail_state(self):
        workflow_id = getattr(self, "_selected_toolbox_workflow_id", None)
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        detail = getattr(self, "_toolbox_detail", None)
        if not parts or not detail:
            return
        spec = parts["spec"]
        if spec.get("input") == "file":
            parts["path_var"].set(detail["file_entry"].get().strip())
        else:
            parts["text_value"] = detail["text_input"].get("1.0", "end").strip()

    def _select_toolbox_workflow(self, workflow_id: str, save_current: bool = True):
        if workflow_id not in getattr(self, "_toolbox_cards", {}):
            return
        if save_current:
            self._save_toolbox_detail_state()
        self._selected_toolbox_workflow_id = workflow_id
        for wid, parts in self._toolbox_cards.items():
            selected = wid == workflow_id
            parts["frame"].configure(
                fg_color=("#dbeafe", "#1f3658") if selected else INPUT_SURFACE,
                border_color=self._IG_ACCENT if selected else BORDER_STRONG,
                border_width=2 if selected else 1,
            )
        self._render_toolbox_detail(workflow_id)

    def _render_toolbox_detail(self, workflow_id: str):
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        detail = getattr(self, "_toolbox_detail", None)
        if not parts or not detail:
            return
        spec = parts["spec"]
        detail["title"].configure(text=spec["title"])
        detail["description"].configure(text=spec["description"])
        detail["status"].configure(
            text=parts.get("status_text", "Checking"),
            text_color=parts.get("status_color", TEXT_MUTED),
        )
        entry = self._toolbox_model_entry(spec)
        models = ", ".join(spec.get("models", []))
        detail["model"].configure(text=f"Models: {models}" + (f"  |  Selected: {entry.get('name')}" if entry else ""))

        input_kind = spec.get("input")
        if input_kind == "file":
            detail["text_input"].grid_remove()
            detail["file_frame"].grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 0))
            detail["file_entry"].delete(0, "end")
            detail["file_entry"].insert(0, parts["path_var"].get())
            browse_text = "Browse audio" if workflow_id == "transcribe" else "Browse image"
            detail["file_browse"].configure(text=browse_text)
        else:
            detail["file_frame"].grid_remove()
            detail["text_input"].grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 0))
            detail["text_input"].delete("1.0", "end")
            detail["text_input"].insert("1.0", parts.get("text_value", spec.get("sample", "")))

        # Language / voice picker row. Visible for transcribe (Whisper
        # languages) and speak (Piper voices); hidden otherwise.
        picker_kind = spec.get("language_picker")
        from src import workflows as _wf

        if picker_kind == "whisper":
            detail["language_label"].configure(text="Language:")
            detail["language_menu"].configure(
                values=list(_wf.WHISPER_LANGUAGES.keys()),
                variable=self._toolbox_transcribe_language,
            )
            detail["language_frame"].grid(row=10, column=0, sticky="ew", padx=18, pady=(8, 0))
        elif picker_kind == "piper":
            detail["language_label"].configure(text="Voice:")
            detail["language_menu"].configure(
                values=list(_wf.PIPER_VOICES.keys()),
                variable=self._toolbox_speak_language,
            )
            detail["language_frame"].grid(row=10, column=0, sticky="ew", padx=18, pady=(8, 0))
        else:
            detail["language_frame"].grid_remove()

        self._set_textbox_value(detail["output"], parts.get("result_text", "Final result appears here."))
        self._set_textbox_value(detail["activity"], parts.get("activity_text", "Progress appears here."))
        self._apply_toolbox_requirement_chips(workflow_id)
        self._apply_toolbox_run_action(workflow_id)
        self._update_toolbox_output_action_widgets(workflow_id)

    def _set_textbox_value(self, textbox, text: str):
        textbox.delete("1.0", "end")
        textbox.insert("1.0", text)

    def _toolbox_install_in_progress(self) -> bool:
        return getattr(self, "_toolbox_install_workflow_id", None) is not None

    def _toolbox_workflow_in_progress(self) -> bool:
        return getattr(self, "_toolbox_active_workflow_id", None) is not None

    def _toolbox_busy(self) -> bool:
        return self._toolbox_install_in_progress() or self._toolbox_workflow_in_progress()

    def _clear_toolbox_install_state(self, workflow_id: str):
        if self._toolbox_install_workflow_id in (None, workflow_id):
            self._toolbox_install_thread = None
            self._toolbox_install_workflow_id = None

    def _clear_toolbox_workflow_state(self, workflow_id: str, token: int | None = None):
        if token is not None and token != self._toolbox_workflow_token:
            return
        if self._toolbox_active_workflow_id in (None, workflow_id):
            self._toolbox_workflow_thread = None
            self._toolbox_active_workflow_id = None

    def _set_toolbox_banner(self, text: str, color=TEXT_MUTED, active: bool = False) -> None:
        if not hasattr(self, "_toolbox_banner_label"):
            return
        self._toolbox_banner_label.configure(text=text, text_color=color)
        progress = getattr(self, "_toolbox_banner_progress", None)
        if progress is None:
            return
        if active:
            progress.grid()
            progress.start()
        else:
            progress.stop()
            progress.grid_remove()

    def _set_toolbox_activity(self, workflow_id: str, text: str, append: bool = False) -> None:
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        if not parts:
            return
        if append:
            existing = parts.get("activity_text", "")
            if existing == "Progress appears here.":
                existing = ""
            parts["activity_text"] = existing + text
        else:
            parts["activity_text"] = text
        if getattr(self, "_selected_toolbox_workflow_id", None) == workflow_id:
            detail = getattr(self, "_toolbox_detail", {})
            activity = detail.get("activity")
            if activity is not None:
                self._set_textbox_value(activity, parts["activity_text"])
                activity.see("end")

    def _set_toolbox_result(self, workflow_id: str, text: str) -> None:
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        if not parts:
            return
        parts["result_text"] = text
        if getattr(self, "_selected_toolbox_workflow_id", None) == workflow_id:
            detail = getattr(self, "_toolbox_detail", {})
            output = detail.get("output")
            if output is not None:
                self._set_textbox_value(output, text)

    def _set_toolbox_status(self, workflow_id: str, text: str, color=TEXT_MUTED) -> None:
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        if not parts:
            return
        parts["status_text"] = text
        parts["status_color"] = color
        parts["status"].configure(text=text, text_color=color)
        if getattr(self, "_selected_toolbox_workflow_id", None) == workflow_id:
            detail = getattr(self, "_toolbox_detail", {})
            status = detail.get("status")
            if status is not None:
                status.configure(text=text, text_color=color)

    def _set_toolbox_output_actions(self, workflow_id: str, output_path: Path | str | None) -> None:
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        if not parts:
            return
        path = Path(output_path) if output_path else None
        if path and not path.exists():
            self._set_toolbox_result(
                workflow_id,
                f"{parts.get('result_text', '').rstrip()}\n\nOutput file was reported but was not found: {path}",
            )
            path = None
        parts["output_path"] = path
        self._update_toolbox_output_action_widgets(workflow_id)

    def _update_toolbox_output_action_widgets(self, workflow_id: str) -> None:
        if getattr(self, "_selected_toolbox_workflow_id", None) != workflow_id:
            return
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        detail = getattr(self, "_toolbox_detail", {})
        if not parts or not detail:
            return
        path = parts.get("output_path")
        widgets = [detail["open_file"], detail["open_folder"]]
        if path:
            label = "Open audio" if workflow_id == "speak" else "Open output"
            detail["open_file"].configure(text=label)
            for widget in widgets:
                widget.grid()
        else:
            for widget in widgets:
                widget.grid_remove()

    def _open_toolbox_output_file(self, workflow_id: str) -> None:
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        path = parts.get("output_path") if parts else None
        if not path:
            return
        path = Path(path)
        if not path.exists():
            message = f"Output file was not found: {path}"
            self._set_toolbox_status(workflow_id, "Missing output", ERROR_TEXT)
            self._set_toolbox_result(workflow_id, message)
            self._set_toolbox_output_actions(workflow_id, None)
            return
        try:
            self._open_path(path)
        except OSError as exc:
            self._set_toolbox_status(workflow_id, "Open failed", ERROR_TEXT)
            self._set_toolbox_result(workflow_id, f"Could not open output file: {exc}")

    def _open_toolbox_output_folder(self, workflow_id: str) -> None:
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        path = parts.get("output_path") if parts else None
        if not path:
            return
        folder = Path(path).parent
        if not folder.exists():
            message = f"Output folder was not found: {folder}"
            self._set_toolbox_status(workflow_id, "Missing folder", ERROR_TEXT)
            self._set_toolbox_result(workflow_id, message)
            self._set_toolbox_output_actions(workflow_id, None)
            return
        try:
            self._open_path(folder)
        except OSError as exc:
            self._set_toolbox_status(workflow_id, "Open failed", ERROR_TEXT)
            self._set_toolbox_result(workflow_id, f"Could not open output folder: {exc}")

    def _refresh_toolbox_cards(self):
        installing = self._toolbox_install_in_progress()
        running = self._toolbox_workflow_in_progress()
        active_install = getattr(self, "_toolbox_install_workflow_id", None)
        active_workflow = getattr(self, "_toolbox_active_workflow_id", None)
        for workflow_id, parts in getattr(self, "_toolbox_cards", {}).items():
            entry = self._toolbox_model_entry(parts["spec"])
            missing = self._toolbox_missing_deps(parts["spec"])
            parts["model"].configure(
                text=self._toolbox_row_model_text(parts["spec"], entry)
            )
            if installing:
                if workflow_id != active_install:
                    self._set_toolbox_status(workflow_id, "Waiting", WARN_TEXT)
                    parts["readiness_text"] = "Readiness: another Toolbox install is running."
                else:
                    self._set_toolbox_status(workflow_id, "Installing", WARN_TEXT)
                    parts["readiness_text"] = "Readiness: installing optional Toolbox packages."
                parts["readiness_color"] = WARN_TEXT
                parts["packages_text"] = "Packages: install in progress"
                parts["packages_color"] = WARN_TEXT
                parts["model_files_text"] = "Model files: pending"
                parts["model_files_color"] = TEXT_MUTED
                parts["hardware_text"] = "Hardware: pending"
                parts["hardware_color"] = TEXT_MUTED
                parts["action_text"] = "Toolbox busy"
                parts["action_state"] = "disabled"
                continue
            if running:
                if workflow_id == active_workflow:
                    if parts.get("status_text") in ("Checking", "Ready", "Done", "Unavailable", "Waiting"):
                        self._set_toolbox_status(workflow_id, "Running", WARN_TEXT)
                    parts["readiness_text"] = "Readiness: running this workflow."
                else:
                    self._set_toolbox_status(workflow_id, "Waiting", WARN_TEXT)
                    parts["readiness_text"] = "Readiness: another Toolbox workflow is running."
                parts["readiness_color"] = WARN_TEXT
                parts["packages_text"] = "Packages: ready" if not missing else f"Packages: missing {', '.join(missing)}"
                parts["packages_color"] = SUCCESS_TEXT if not missing else WARN_TEXT
                parts["model_files_text"] = "Model files: pending"
                parts["model_files_color"] = TEXT_MUTED
                parts["hardware_text"] = "Hardware: pending"
                parts["hardware_color"] = TEXT_MUTED
                parts["action_text"] = "Toolbox busy"
                parts["action_state"] = "disabled"
                continue
            if missing:
                self._set_toolbox_status(workflow_id, "Needs install", WARN_TEXT)
                parts["readiness_text"] = f"Readiness: missing packages: {', '.join(missing)}"
                parts["readiness_color"] = WARN_TEXT
                parts["packages_text"] = f"Packages: missing {', '.join(missing)}"
                parts["packages_color"] = WARN_TEXT
                parts["model_files_text"] = "Model files: checked after packages install"
                parts["model_files_color"] = TEXT_MUTED
                parts["hardware_text"] = "Hardware: checked after packages install"
                parts["hardware_color"] = TEXT_MUTED
                parts["action_text"] = "Install required packages"
                parts["action_state"] = "normal"
                continue
            ok, reason = self._toolbox_can_run(entry)
            self._set_toolbox_status(workflow_id, "Ready" if ok else "Unavailable", SUCCESS_TEXT if ok else WARN_TEXT)
            parts["readiness_text"] = f"Readiness: {(entry.get('name') if ok and entry else reason)}"
            parts["readiness_color"] = SUCCESS_TEXT if ok else WARN_TEXT
            parts["packages_text"] = "Packages: ready"
            parts["packages_color"] = SUCCESS_TEXT
            if entry and ok:
                cached = self._phase1_model_cached(entry)
                parts["model_files_text"] = "Model files: cached" if cached else "Model files: first run downloads"
                parts["model_files_color"] = SUCCESS_TEXT if cached else WARN_TEXT
            else:
                parts["model_files_text"] = "Model files: unavailable"
                parts["model_files_color"] = WARN_TEXT
            parts["hardware_text"] = "Hardware: ready" if ok else f"Hardware: {reason}"
            parts["hardware_color"] = SUCCESS_TEXT if ok else WARN_TEXT
            parts["action_text"] = f"Run {parts['spec']['title']}"
            parts["action_state"] = "normal" if ok else "disabled"
        self._apply_toolbox_browser_density()
        selected = getattr(self, "_selected_toolbox_workflow_id", None)
        if selected:
            self._apply_toolbox_requirement_chips(selected)
            self._apply_toolbox_run_action(selected)
            self._update_toolbox_output_action_widgets(selected)

    def _apply_toolbox_requirement_chips(self, workflow_id: str):
        if getattr(self, "_selected_toolbox_workflow_id", None) != workflow_id:
            return
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        detail = getattr(self, "_toolbox_detail", {})
        if not parts or not detail:
            return
        detail["packages"].configure(
            text=parts.get("packages_text", "Packages: checking"),
            text_color=parts.get("packages_color", TEXT_MUTED),
        )
        detail["model_files"].configure(
            text=parts.get("model_files_text", "Model files: checking"),
            text_color=parts.get("model_files_color", TEXT_MUTED),
        )
        detail["hardware"].configure(
            text=parts.get("hardware_text", "Hardware: checking"),
            text_color=parts.get("hardware_color", TEXT_MUTED),
        )

    def _apply_toolbox_run_action(self, workflow_id: str):
        if getattr(self, "_selected_toolbox_workflow_id", None) != workflow_id:
            return
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        detail = getattr(self, "_toolbox_detail", {})
        if not parts or not detail:
            return
        detail["run"].configure(
            text=parts.get("action_text", f"Run {parts['spec']['title']}"),
            state=parts.get("action_state", "normal"),
        )

    def _toolbox_browse_file(self, workflow_id: str):
        parts = self._toolbox_cards.get(workflow_id)
        if not parts:
            return
        path = filedialog.askopenfilename(title=f"Select file for {parts['spec']['title']}", filetypes=parts["spec"].get("filetypes"))
        if path:
            parts["path_var"].set(path)
            if getattr(self, "_selected_toolbox_workflow_id", None) == workflow_id:
                detail = getattr(self, "_toolbox_detail", {})
                entry = detail.get("file_entry")
                if entry is not None:
                    entry.delete(0, "end")
                    entry.insert(0, path)

    def open_toolbox_for_model(self, model: dict):
        target = None
        model_id = model.get("id")
        for wid, spec in self._TOOLBOX_WORKFLOWS.items():
            if model_id in spec.get("models", []):
                target = wid
                break
        self._switch_page("toolbox")
        if target:
            self.after(100, lambda: self._toolbox_focus_workflow(target))

    def _toolbox_focus_workflow(self, workflow_id: str):
        parts = getattr(self, "_toolbox_cards", {}).get(workflow_id)
        if not parts:
            return
        self._select_toolbox_workflow(workflow_id)
        detail = getattr(self, "_toolbox_detail", {})
        output = detail.get("output")
        if self._widget_is_alive(output):
            try:
                output.focus_set()
            except TclError:
                pass

    def _run_toolbox_workflow(self, workflow_id: str):
        parts = self._toolbox_cards.get(workflow_id)
        if not parts:
            return
        self._select_toolbox_workflow(workflow_id)
        self._save_toolbox_detail_state()
        if self._toolbox_busy():
            active = self._toolbox_install_workflow_id or self._toolbox_active_workflow_id or "another workflow"
            message = "Wait for the current Toolbox install or workflow to finish before starting another one."
            logger.info(
                f"Toolbox workflow request for {workflow_id} skipped; active workflow: {active}",
                category=logger.CATEGORY_TOOLBOX,
            )
            self._set_toolbox_status(workflow_id, "Busy", WARN_TEXT)
            self._set_toolbox_activity(workflow_id, message)
            messagebox.showinfo("Toolbox busy", message, parent=self)
            return
        spec = parts["spec"]
        missing = self._toolbox_missing_deps(spec)
        if missing:
            self._install_toolbox_requirements(workflow_id, missing)
            return
        entry = self._toolbox_model_entry(spec)
        ok, reason = self._toolbox_can_run(entry)
        if not ok or entry is None:
            messagebox.showinfo("Toolbox workflow unavailable", reason, parent=self)
            return
        # Ollama-backed entries (ollama_tag set) skip the HF cache check
        # and instead gate on Ollama being up + the tag being pulled.
        ollama_tag = (entry.get("ollama_tag") or "").strip()
        if ollama_tag:
            if not self.ollama.is_running():
                messagebox.showinfo(
                    "Ollama not running",
                    "This workflow uses an Ollama-hosted model. Start Ollama "
                    "(from the Models tab or the Ollama tray icon) and try again.",
                    parent=self,
                )
                return
            try:
                model_local = self.ollama.is_model_local(ollama_tag)
            except Exception:
                model_local = False
            if not model_local:
                # v2026.06.01.8 (Ron, 2026-06-01): the legacy "Open the
                # Models tab" hint sent users on a hunt for vision
                # models (minicpm-v / gemma3-vision) that they actually
                # install most naturally from Image Gen > Vision picker
                # — and routing through one extra page is unnecessary
                # when we already have the catalog dict in hand and a
                # tested in-app downloader. Offer to download right
                # here instead. This works for every Ollama-backed
                # Toolbox workflow (extract table fast, etc.).
                model_name = entry.get("name") or ollama_tag
                size_gb = entry.get("size_gb") or 0
                size_hint = (
                    f"\n\nDownload size: ~{size_gb:.1f} GB."
                    if size_gb else ""
                )
                if messagebox.askyesno(
                    "Download model now?",
                    f"This workflow needs the Ollama model '{model_name}'.\n"
                    "It is not on this machine yet."
                    f"{size_hint}\n\n"
                    "Download it now? The Toolbox workflow will not "
                    "start until the download finishes — re-run this "
                    "workflow once the status shows the download is "
                    "complete.",
                    parent=self,
                ):
                    self.start_download(entry)
                return
        input_text = ""
        input_path = None
        if spec["input"] == "file":
            input_path = Path(parts["path_var"].get().strip())
            if not input_path.exists():
                messagebox.showinfo("Choose a file", "Select a local file first.", parent=self)
                return
        else:
            input_text = parts.get("text_value", "").strip()
            if not input_text:
                messagebox.showinfo("Enter input", "Enter text for this workflow first.", parent=self)
                return
        # HF model cache check only applies to non-Ollama entries.
        download_required = (not ollama_tag) and (not self._phase1_model_cached(entry))
        if download_required:
            repo = entry.get("hf_repo") or entry.get("onnx_repo") or entry.get("name", "this model")
            if not messagebox.askyesno(
                "Download model files",
                f"First run for {entry.get('name', repo)} needs a one-time Hugging Face model download.\n\n"
                f"Repository: {repo}\n"
                f"Cache: {self._phase1_cache_dir()}\n\n"
                "The detail panel will show status updates while the model is downloading/loading. This can take several minutes on cloud VMs.\n\n"
                "Continue?",
                parent=self,
            ):
                return
        self._toolbox_workflow_token += 1
        token = self._toolbox_workflow_token
        self._toolbox_active_workflow_id = workflow_id
        self._set_toolbox_output_actions(workflow_id, None)
        self._set_toolbox_status(workflow_id, "Downloading model" if download_required else "Running", WARN_TEXT)
        self._set_toolbox_banner(f"{spec['title']} is running ...", WARN_TEXT, active=True)
        self._set_toolbox_result(workflow_id, "Final result appears here when the workflow finishes.")
        logger.info(f"Toolbox workflow started: {spec['title']}", category=logger.CATEGORY_TOOLBOX)
        if download_required:
            repo = entry.get("hf_repo") or entry.get("onnx_repo") or entry.get("name", "this model")
            self._set_toolbox_activity(
                workflow_id,
                f"Downloading/loading {entry.get('name', repo)} from Hugging Face...\n"
                f"Repository: {repo}\n"
                f"Cache: {self._phase1_cache_dir()}\n"
                "First run can take several minutes. The result will appear here when ready.\n\n",
            )
        else:
            self._set_toolbox_activity(workflow_id, "Running...\n")
        self._refresh_toolbox_cards()

        def _progress(msg: str):
            log_key = f"{workflow_id}:log"
            if self._toolbox_last_progress.get(log_key) != msg:
                self._toolbox_last_progress[log_key] = msg
                logger.info(f"{spec['title']}: {msg}", category=logger.CATEGORY_TOOLBOX)
            def _update(m=msg):
                if token != self._toolbox_workflow_token:
                    return
                if self._toolbox_last_progress.get(workflow_id) == m:
                    return
                self._set_toolbox_status(workflow_id, m[:24], WARN_TEXT)
                self._toolbox_last_progress[workflow_id] = m
                self._set_toolbox_activity(workflow_id, f"{m}\n", append=True)
                self._set_toolbox_banner(m, WARN_TEXT, active=True)
            self.after(0, _update)

        def _worker():
            try:
                from src import workflows
                out_dir = Path(__file__).parent.parent / "toolbox_outputs"
                if workflow_id == "transcribe":
                    lang_label = getattr(self, "_toolbox_transcribe_language", None)
                    label_value = lang_label.get() if lang_label is not None else None
                    language = workflows.WHISPER_LANGUAGES.get(label_value)
                    result = workflows.transcribe(input_path, entry, progress_cb=_progress, language=language)
                elif workflow_id == "read":
                    result = workflows.read_image(input_path, entry, progress_cb=_progress)
                elif workflow_id == "tables":
                    result = workflows.extract_table_ollama(input_path, entry, output_dir=out_dir, progress_cb=_progress)
                elif workflow_id == "extract_table":
                    result = workflows.extract_table_got(input_path, entry, output_dir=out_dir, progress_cb=_progress)
                elif workflow_id == "speak":
                    voice_var = getattr(self, "_toolbox_speak_language", None)
                    voice_label = voice_var.get() if voice_var is not None else None
                    result = workflows.synthesize(input_text, entry, output_dir=out_dir, progress_cb=_progress, language=voice_label)
                elif workflow_id == "search":
                    lines = [line.strip() for line in input_text.splitlines() if line.strip()]
                    query = lines[0].replace("Query:", "").strip() if lines else "local AI"
                    corpus = [line for line in lines[1:] if not line.lower().startswith("query:")]
                    result = workflows.embed_and_rank(query, corpus or lines, entry, progress_cb=_progress)
                else:
                    result = workflows.describe(input_path, entry, progress_cb=_progress)
                self.after(0, lambda r=result, t=token: self._toolbox_workflow_done(workflow_id, r, t))
            except Exception as exc:
                self.after(0, lambda e=exc, t=token: self._toolbox_workflow_failed(workflow_id, e, t))

        thread = threading.Thread(target=_worker, name=f"ToolboxWorkflow-{workflow_id}", daemon=True)
        self._toolbox_workflow_thread = thread
        thread.start()

    def _install_toolbox_requirements(self, workflow_id: str, missing: list[str]):
        parts = self._toolbox_cards.get(workflow_id)
        if not parts:
            return
        self._select_toolbox_workflow(workflow_id)
        if self._toolbox_busy():
            active = self._toolbox_install_workflow_id or self._toolbox_active_workflow_id or "another workflow"
            message = (
                "Toolbox is already running an install or workflow. "
                "Wait for it to finish before starting another Toolbox action."
            )
            logger.info(
                f"Toolbox dependency install request for {workflow_id} skipped; active action: {active}",
                category=logger.CATEGORY_TOOLBOX,
            )
            self._set_toolbox_status(workflow_id, "Busy", WARN_TEXT)
            self._set_toolbox_activity(workflow_id, message)
            messagebox.showinfo("Toolbox busy", message, parent=self)
            return
        full_package_set = [
            "transformers>=4.57.0",
            "accelerate>=1.12.0",
            "safetensors>=0.7.0",
            "torch>=2.3.0",
            "sentence-transformers>=5.1.0",
            "soundfile>=0.13.0",
            "scipy>=1.16.0",
            "pillow>=10.0.0",
            "librosa>=0.11.0",
            "einops>=0.8.0",
            "timm>=1.0.0",
            "sentencepiece>=0.2.0",
            "huggingface-hub>=0.34.0,<1.0",
            "hf_xet>=1.2.0",
            "torchvision>=0.18.0",
            "diffusers>=0.36.0",
            "pyttsx3>=2.99",
            "peft>=0.18.0",
            "backoff>=2.2.0",
            "piper-tts>=1.2.0",
            "onnxruntime>=1.23.0",
        ]
        dep_to_package = {
            "accelerate": "accelerate>=1.12.0",
            "safetensors": "safetensors>=0.7.0",
            "torch": "torch>=2.3.0",
            "sentence_transformers": "sentence-transformers>=5.1.0",
            "soundfile": "soundfile>=0.13.0",
            "scipy": "scipy>=1.16.0",
            "PIL": "pillow>=10.0.0",
            "librosa": "librosa>=0.11.0",
            "einops": "einops>=0.8.0",
            "timm": "timm>=1.0.0",
            "sentencepiece": "sentencepiece>=0.2.0",
            "huggingface_hub": "huggingface-hub>=0.34.0,<1.0",
            "hf_xet": "hf_xet>=1.2.0",
            "torchvision": "torchvision>=0.18.0",
            "diffusers": "diffusers>=0.36.0",
            "pyttsx3": "pyttsx3>=2.99",
            "peft": "peft>=0.18.0",
            "backoff": "backoff>=2.2.0",
            "transformers": "transformers>=4.57.0",
            "piper": "piper-tts>=1.2.0",
            "onnxruntime": "onnxruntime>=1.23.0",
        }
        unknown_missing = [dep for dep in missing if dep not in dep_to_package]
        if unknown_missing:
            packages_to_install = list(full_package_set)
            install_scope = "the full optional Toolbox package set"
        else:
            selected = {dep_to_package[dep] for dep in missing}
            # Keep the strict hub pin in every Toolbox install path.
            selected.add("huggingface-hub>=0.34.0,<1.0")
            packages_to_install = sorted(selected)
            install_scope = "required Toolbox packages"
        if not messagebox.askyesno(
            "Install Toolbox packages",
            f"This will install {install_scope} in the current Python environment.\n\n"
            f"Missing packages: {', '.join(missing)}\n\nContinue?",
            parent=self,
        ):
            return
        self._set_toolbox_status(workflow_id, "Installing", WARN_TEXT)
        self._set_toolbox_output_actions(workflow_id, None)
        self._set_toolbox_result(workflow_id, "Install result appears here when package setup finishes.")
        self._set_toolbox_activity(workflow_id, "Installing required packages. See Logs for details.\n")
        self._set_toolbox_banner("Installing Toolbox packages ...", WARN_TEXT, active=True)

        def _worker():
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--no-input",
                    "--disable-pip-version-check",
                    *packages_to_install,
                ]
                logger.info(
                    "Toolbox dependency install started: " + " ".join(cmd),
                    category=logger.CATEGORY_TOOLBOX,
                )
                popen_kw = {}
                if sys.platform == "win32":
                    popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).parent.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **popen_kw,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info("[toolbox install] " + line, category=logger.CATEGORY_TOOLBOX)
                        def _update(msg=line[:80]):
                            self._set_toolbox_status(workflow_id, "Installing", WARN_TEXT)
                            self._set_toolbox_activity(workflow_id, f"{msg}\n", append=True)
                            self._set_toolbox_banner(msg, WARN_TEXT, active=True)
                        self.after(0, _update)
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"pip install failed with exit code {code}")
                check_kw = {"capture_output": True, "text": True, "timeout": 120}
                if sys.platform == "win32":
                    check_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
                check = subprocess.run([sys.executable, "-m", "pip", "check"], **check_kw)
                if check.returncode != 0:
                    detail = (check.stderr or check.stdout or "").strip()
                    raise RuntimeError(f"pip check failed after Toolbox install: {detail}")
                logger.info("Toolbox dependency install completed.", category=logger.CATEGORY_TOOLBOX)
                self.after(0, lambda: self._toolbox_install_done(workflow_id))
            except Exception as exc:
                logger.error(f"Toolbox dependency install failed: {exc}", category=logger.CATEGORY_TOOLBOX)
                self.after(0, lambda e=exc: self._toolbox_install_failed(workflow_id, e))

        thread = threading.Thread(target=_worker, daemon=True)
        self._toolbox_install_workflow_id = workflow_id
        self._toolbox_install_thread = thread
        self._refresh_toolbox_cards()
        thread.start()

    def _toolbox_install_done(self, workflow_id: str):
        self._clear_toolbox_install_state(workflow_id)
        importlib.invalidate_caches()
        parts = self._toolbox_cards.get(workflow_id)
        if not parts:
            self._refresh_toolbox_cards()
            return
        self._set_toolbox_status(workflow_id, "Install complete", SUCCESS_TEXT)
        self._set_toolbox_activity(workflow_id, "\nInstall complete. Refreshing Toolbox package state...\n", append=True)
        missing = self._toolbox_missing_deps(parts["spec"])
        if missing:
            self._set_toolbox_result(
                workflow_id,
                "Install finished, but these packages still appear missing: "
                f"{', '.join(missing)}. Try Install required packages again or check Logs.",
            )
        else:
            self._set_toolbox_result(workflow_id, "Install complete. You can run the workflow now.")
        self._set_toolbox_banner("Toolbox package install complete.", SUCCESS_TEXT, active=False)
        logger.info(f"Toolbox install complete: {workflow_id}", category=logger.CATEGORY_TOOLBOX)
        self._refresh_toolbox_cards()

    def _toolbox_install_failed(self, workflow_id: str, exc: Exception):
        self._clear_toolbox_install_state(workflow_id)
        self._set_toolbox_banner("Toolbox package install failed.", ERROR_TEXT, active=False)
        self._set_toolbox_status(workflow_id, "Failed", ERROR_TEXT)
        self._set_toolbox_result(workflow_id, str(exc))
        logger.error(f"Toolbox install failed: {workflow_id}: {exc}", category=logger.CATEGORY_TOOLBOX)
        self._refresh_toolbox_cards()

    def _toolbox_workflow_done(self, workflow_id: str, result, token: int | None = None):
        if token is not None and token != self._toolbox_workflow_token:
            return
        self._clear_toolbox_workflow_state(workflow_id, token)
        parts = self._toolbox_cards.get(workflow_id)
        if not parts:
            return
        self._set_toolbox_status(workflow_id, "Done", SUCCESS_TEXT)
        result_text = result.output_text
        if result.output_path and str(result.output_path) not in result.output_text:
            result_text = f"{result_text.rstrip()}\n\nSaved: {result.output_path}"
        self._set_toolbox_result(workflow_id, result_text)
        self._set_toolbox_output_actions(workflow_id, result.output_path)
        self._set_toolbox_banner(f"{parts['spec']['title']} finished.", SUCCESS_TEXT, active=False)
        logger.info(f"Toolbox workflow completed: {parts['spec']['title']}", category=logger.CATEGORY_TOOLBOX)
        self._refresh_toolbox_cards()

    def _toolbox_workflow_failed(self, workflow_id: str, exc: Exception, token: int | None = None):
        if token is not None and token != self._toolbox_workflow_token:
            return
        self._clear_toolbox_workflow_state(workflow_id, token)
        parts = self._toolbox_cards.get(workflow_id)
        if not parts:
            return
        self._set_toolbox_status(workflow_id, "Failed", ERROR_TEXT)
        self._set_toolbox_result(workflow_id, str(exc))
        self._set_toolbox_banner(f"{parts['spec']['title']} failed.", ERROR_TEXT, active=False)
        logger.error(f"Toolbox workflow failed: {parts['spec']['title']}: {exc}", category=logger.CATEGORY_TOOLBOX)
        self._refresh_toolbox_cards()

    # ── Chat page ─────────────────────────────────────────────────────────────

    def _configured_chat_font_family(self) -> str:
        family = str(self.cfg.get("chat_font_family", CHAT_TEXT_FONT_FAMILY)).strip()
        return family if family in CHAT_FONT_CHOICES else CHAT_TEXT_FONT_FAMILY

    def _chat_text_font(self) -> "ctk.CTkFont":
        return ctk.CTkFont(family=self._chat_font_family, size=CHAT_TEXT_FONT_SIZE)

    def _configure_chat_text_tags(self) -> None:
        if not hasattr(self, "_chat_display") or self._chat_display is None:
            return
        tb = self._chat_display._textbox
        label_font = (self._chat_font_family, CHAT_TEXT_FONT_SIZE, "bold")
        italic_font = (self._chat_font_family, CHAT_TEXT_FONT_SIZE, "italic")
        tb.tag_config("user_label", foreground="#5dade2", font=label_font)
        tb.tag_config("assistant_label", foreground="#82e0aa", font=label_font)
        tb.tag_config("thinking", foreground="#888888", font=italic_font)
        tb.tag_config("error_text", foreground="#e57373")
        tb.tag_config("system_text", foreground="#9a9a9a", font=italic_font)
        tb.tag_config("switch", foreground="#c39bd3", font=label_font)

    def _on_chat_font_changed(self, family: str) -> None:
        if family not in CHAT_FONT_CHOICES:
            family = CHAT_TEXT_FONT_FAMILY
        self._chat_font_family = family
        self.cfg["chat_font_family"] = family
        if not config.save(self.cfg):
            logger.error("Could not persist chat font selection")
        font = self._chat_text_font()
        if hasattr(self, "_chat_display") and self._chat_display is not None:
            self._chat_display.configure(font=font)
            self._configure_chat_text_tags()
        if hasattr(self, "_input_box") and self._input_box is not None:
            self._input_box.configure(font=font)
        self.set_status(f"Chat font set to {family}.")

    def _chat_model_entries(self) -> list[dict]:
        """Text/vision models that can be selected from the Chat page.

        v5.5.11 (Ron, 2026-05-26): Sorted fastest -> slowest using size_gb
        as a speed proxy (smaller models load and infer faster on the same
        hardware). Category and name are tiebreakers only — DO NOT reorder
        the sort key tuple, the fastest-first ordering is contractually
        required by the dropdown UX and by ModelDropdownSpeedSortContractTests.

        v5.5.12 (Ron, 2026-05-27): size_gb of 0, None, or a non-numeric value
        is treated as "size unknown" and pushed to the END of the dropdown
        via ``float('inf')``. The previous implementation returned 0.0 for
        missing/zero values, which floated 10 catalog entries (Llama 3.3 70B,
        Qwen3 30B-A3B, Phi-4 14B, Mistral Nemo, Mistral Small 3.2 24B,
        Qwen2.5 Coder 7B, Qwen2.5-VL 32B/7B/3B, DeepSeek-R1 Distill 1.5B)
        to the TOP of the dropdown — putting the heaviest model (Llama 3.3
        70B) first and burying the actual fastest (Qwen 2.5 0.5B at 0.4 GB).
        Pinned by ``ModelDropdownSpeedSortContractTests
        .test_chat_dropdown_pushes_zero_size_gb_to_end``.
        """
        def _size_gb(model: dict) -> float:
            try:
                val = float(model.get("size_gb") or 0)
            except (TypeError, ValueError):
                return float("inf")
            return val if val > 0 else float("inf")

        models = [
            m for m in getattr(self, "_catalog_models", [])
            if catalog.is_chat_selectable_model(m)
        ]
        return sorted(
            models,
            key=lambda m: (
                _size_gb(m),
                str(m.get("category") or ""),
                str(m.get("name") or "").lower(),
            ),
        )

    def _chat_model_label(self, model: dict) -> str:
        size = model.get("size_gb")
        try:
            size_text = f"{float(size):g} GB"
        except (TypeError, ValueError):
            size_text = "? GB"
        category = model.get("category") or "Model"
        return f"{model.get('name', 'Unnamed model')}  —  {category}, {size_text}"

    def _refresh_chat_model_selector(self) -> None:
        entries = self._chat_model_entries()
        self._chat_model_by_label = {self._chat_model_label(m): m for m in entries}
        labels = list(self._chat_model_by_label) or ["No chat models available"]

        if not hasattr(self, "_chat_model_var"):
            self._chat_model_var = ctk.StringVar(value=labels[0])
        elif self.active_model and self.active_model.get("backend") != "comfyui":
            self._set_chat_selector_to_model(self.active_model)
        elif self._chat_model_var.get() not in labels:
            self._chat_model_var.set(labels[0])

        menu = getattr(self, "_chat_model_menu", None)
        if menu is not None and not self._widget_is_alive(menu):
            self._chat_model_menu = None
            menu = None
        if menu is not None:
            try:
                menu.configure(values=labels)
            except TclError:
                self._chat_model_menu = None

        load_btn = getattr(self, "_chat_load_btn", None)
        if load_btn is not None and not self._widget_is_alive(load_btn):
            self._chat_load_btn = None
            load_btn = None
        if load_btn is not None:
            try:
                load_btn.configure(state="normal" if entries else "disabled")
            except TclError:
                self._chat_load_btn = None

    def _set_chat_selector_to_model(self, model: dict) -> None:
        if not hasattr(self, "_chat_model_var"):
            return
        target_id = model.get("id")
        target_tag = model.get("ollama_tag")
        for label, entry in getattr(self, "_chat_model_by_label", {}).items():
            if entry.get("id") == target_id or entry.get("ollama_tag") == target_tag:
                self._chat_model_var.set(label)
                return

    def _selected_chat_model(self) -> dict | None:
        return getattr(self, "_chat_model_by_label", {}).get(self._chat_model_var.get())

    def _load_selected_chat_model(self) -> None:
        if self._chat_thinking:
            self.set_status("Wait for the current response to finish before switching models.")
            return
        model = self._selected_chat_model()
        if not model:
            self.set_status("No chat model selected.")
            return
        backend = self._backend_var.get()
        tag = model.get("ollama_tag", "")
        if "Ollama" in backend and tag:
            if not self.ollama_ok:
                messagebox.showerror(
                    "Ollama not running",
                    "Ollama must be running to load this model.",
                    parent=self,
                )
                return
            try:
                if not self.ollama.is_model_local(tag):
                    download = messagebox.askyesno(
                        "Model not installed",
                        f"'{model['name']}' is not installed locally yet.\n\n"
                        "Download it now from the Chat page?",
                        parent=self,
                    )
                    if download:
                        self._start_chat_download_after_primary_action(model)
                    return
            except OllamaError as exc:
                messagebox.showerror("Ollama error", str(exc), parent=self)
                return
        self.load_model_for_chat(model)

    def _build_chat_page(self):
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["chat"] = page
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(page, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        toolbar.grid_columnconfigure(1, weight=1)

        self._refresh_chat_model_selector()
        ctk.CTkLabel(toolbar, text="Model:", font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 4))
        self._chat_model_menu = ctk.CTkOptionMenu(
            toolbar,
            values=list(getattr(self, "_chat_model_by_label", {})) or ["No chat models available"],
            variable=self._chat_model_var,
            width=360,
            dynamic_resizing=False,
            **self._option_menu_style(),
        )
        self._chat_model_menu.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self._chat_load_btn = ctk.CTkButton(
            toolbar, text="Load selected", width=110,
            **self._solid_button_style("#1f6aa5", "#1f538d"),
            command=self._load_selected_chat_model,
        )
        self._chat_load_btn.grid(row=0, column=2, padx=(0, 16))

        ctk.CTkLabel(toolbar, text="Loaded:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, padx=(0, 4), pady=(8, 0))
        self._active_model_label = ctk.CTkLabel(
            toolbar, text="None loaded",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=LINK_TEXT
        )
        self._active_model_label.grid(row=1, column=1, sticky="w", padx=(0, 16), pady=(8, 0))

        ctk.CTkLabel(toolbar, text="Backend:", font=ctk.CTkFont(size=12)).grid(row=0, column=3, padx=(0, 4))
        backend_menu = ctk.CTkOptionMenu(
            toolbar,
            values=["GPU/CPU (Ollama)", "CPU only (Ollama)", "OpenVINO (GPU/NPU)", "NPU/OpenVINO (ONNX)"],
            variable=self._backend_var,
            width=190,
            **self._option_menu_style(),
        )
        backend_menu.grid(row=0, column=4, sticky="w")

        ctk.CTkButton(
            toolbar, text="Free VRAM", width=90,
            **self._outline_button_style(),
            command=self._free_comfyui_vram,
        ).grid(row=0, column=5, padx=(8, 0))

        ctk.CTkButton(
            toolbar, text="Clear chat", width=90,
            **self._outline_button_style(),
            command=self._clear_chat,
        ).grid(row=0, column=6, padx=(8, 0))

        ctk.CTkLabel(toolbar, text="Font:", font=ctk.CTkFont(size=12)).grid(row=1, column=2, padx=(0, 4), pady=(8, 0))
        self._chat_font_var = ctk.StringVar(value=self._chat_font_family)
        self._chat_font_menu = ctk.CTkOptionMenu(
            toolbar,
            values=CHAT_FONT_CHOICES,
            variable=self._chat_font_var,
            width=155,
            command=self._on_chat_font_changed,
            **self._option_menu_style(),
        )
        self._chat_font_menu.grid(row=1, column=3, sticky="w", padx=(0, 16), pady=(8, 0))

        ctk.CTkLabel(toolbar, text="Reply tokens:", font=ctk.CTkFont(size=12)).grid(
            row=1, column=4, padx=(0, 4), pady=(8, 0)
        )
        self._max_tokens_entry = ctk.CTkEntry(
            toolbar,
            textvariable=self._max_tokens_var,
            width=86,
        )
        self._max_tokens_entry.grid(row=1, column=5, sticky="w", padx=(0, 8), pady=(8, 0))
        HelpTooltip(
            self._max_tokens_entry,
            "Use Max to let the loaded model fill its available response context. "
            "Enter a number when you want a shorter cap or need to retry a clipped answer with a larger limit.",
            "Reply tokens",
        )
        ctk.CTkButton(
            toolbar, text="Prompt ideas", width=112,
            **self._outline_button_style(text_color=LINK_TEXT),
            command=self._open_chat_prompt_ideas,
        ).grid(row=1, column=6, sticky="w", padx=(8, 0), pady=(8, 0))

        # Chat display
        self._chat_display = ctk.CTkTextbox(
            page, wrap="word", state="disabled",
            font=self._chat_text_font(),
            spacing1=4, spacing3=6,
        )
        self._chat_display.grid(row=1, column=0, sticky="nsew", padx=16, pady=4)

        # v5: empty-state placeholder text shown when no model is loaded
        self._chat_display.configure(state="normal")
        self._chat_display.insert("end",
            "\n  No model loaded yet.\n\n"
            "  → Pick a model and backend above, then click 'Load selected'.\n"
            "  → If an Ollama model is not installed yet, LocalAI can download it from here.\n\n"
        )
        self._chat_display.configure(state="disabled")

        # Configure on the inner tk.Text because CTkTextbox.tag_config blocks
        # the `font` kwarg.
        self._configure_chat_text_tags()

        # Input area
        input_row = ctk.CTkFrame(page, fg_color="transparent")
        input_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(4, 12))
        input_row.grid_columnconfigure(0, weight=1)

        self._input_box = ctk.CTkTextbox(
            input_row, height=68, wrap="word",
            font=self._chat_text_font(),
            spacing1=3, spacing3=3,
        )
        self._input_box.grid(row=0, column=0, sticky="ew")
        self._input_box.bind("<Return>", self._on_enter_key)
        self._input_box.bind("<Shift-Return>", lambda e: None)  # allow newlines

        btn_col = ctk.CTkFrame(input_row, fg_color="transparent")
        btn_col.grid(row=0, column=1, sticky="n", padx=(8, 0))
        btn_col.grid_columnconfigure((0, 1), weight=1)

        self._send_btn = ctk.CTkButton(
            btn_col, text="Send", width=82, state="disabled",
            **self._solid_button_style("#1f6aa5", "#1f538d"),
            command=self._send_message,
        )
        self._send_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))

        self._stop_btn = ctk.CTkButton(
            btn_col, text="Stop", width=82,
            **self._solid_button_style("#7a2a2a", "#601f1f"),
            command=self._stop_generation, state="disabled"
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        self._copy_last_response_btn = ctk.CTkButton(
            btn_col, text="Copy reply", width=82,
            command=self._copy_last_chat_response, state="disabled",
            **self._outline_button_style(),
        )
        self._copy_last_response_btn.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self._copy_last_exchange_btn = ctk.CTkButton(
            btn_col, text="Copy both", width=82,
            command=self._copy_last_chat_exchange, state="disabled",
            **self._outline_button_style(),
        )
        self._copy_last_exchange_btn.grid(row=1, column=1, sticky="ew")

        self._chat_ready_label = ctk.CTkLabel(
            page,
            text="Pick a model and backend above, then click Load selected.",
            font=ctk.CTkFont(size=11), text_color=WARN_TEXT,
            anchor="w",
        )
        self._chat_ready_label.grid(row=3, column=0, sticky="w", padx=16, pady=(0, 2))

        self._chat_first_token_progress = ctk.CTkProgressBar(page, height=6, mode="indeterminate")
        self._chat_first_token_progress.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 6))
        self._chat_first_token_progress.grid_remove()

        self._update_chat_readiness()
        self._update_chat_copy_buttons()

    def _on_enter_key(self, event):
        if not (event.state & 0x1):  # Shift not held
            self._send_message()
            return "break"  # prevent newline insertion

    def _response_token_budget_display(self) -> str:
        if str(self.cfg.get("response_token_mode") or "max").lower() != "custom":
            return "Max"
        try:
            value = int(self.cfg.get("max_tokens") or 0)
        except (TypeError, ValueError):
            return "Max"
        return str(value) if value > 0 else "Max"

    def _parse_response_token_budget(self, raw: object) -> tuple[int | None, str]:
        text = str(raw or "").strip().lower()
        if not text or text in {"max", "maximum", "fill", "full", "context", "0", "-1"}:
            return 0, ""
        try:
            value = int(text.replace(",", ""))
        except ValueError:
            return None, "Reply tokens must be Max or a whole number."
        if value <= 0:
            return 0, ""
        if value > CHAT_RESPONSE_TOKEN_MAX:
            return None, f"Reply tokens must be {CHAT_RESPONSE_TOKEN_MAX:,} or less."
        return value, ""

    def _sync_response_token_budget_from_ui(self) -> int | None:
        raw = self._max_tokens_var.get() if hasattr(self, "_max_tokens_var") else self._response_token_budget_display()
        budget, error = self._parse_response_token_budget(raw)
        if budget is None:
            self.set_status(error)
            if hasattr(self, "_chat_ready_label") and self._chat_ready_label is not None:
                self._chat_ready_label.configure(text=error, text_color=ERROR_TEXT)
            return None
        previous_mode = self.cfg.get("response_token_mode")
        previous_budget = self.cfg.get("max_tokens")
        self.cfg["response_token_mode"] = "custom" if budget > 0 else "max"
        self.cfg["max_tokens"] = budget
        if hasattr(self, "_max_tokens_var"):
            self._max_tokens_var.set(str(budget) if budget > 0 else "Max")
        changed = previous_mode != self.cfg["response_token_mode"] or previous_budget != budget
        if changed and not config.save(self.cfg):
            logger.error("Could not persist chat response token budget.")
        return budget

    def _context_length_for_active_model(self) -> int:
        try:
            return int((self.active_model or {}).get("context_length") or 0)
        except (TypeError, ValueError):
            return 0

    def _max_new_tokens_for_prompt(self, budget: int, prompt: str) -> int:
        if budget > 0:
            return budget
        context_length = self._context_length_for_active_model()
        if context_length <= 0:
            return CHAT_RESPONSE_TOKEN_ONNX_FALLBACK
        prompt_estimate = max(1, len(prompt) // 4)
        return max(1, min(CHAT_RESPONSE_TOKEN_MAX, context_length - prompt_estimate - 16))

    def _update_chat_readiness(self):
        ready = bool(self.active_model) and not self._chat_thinking
        if hasattr(self, "_send_btn") and self._send_btn is not None:
            self._send_btn.configure(state="normal" if ready else "disabled")
        if hasattr(self, "_chat_ready_label") and self._chat_ready_label is not None:
            if self._chat_thinking and self.active_model:
                self._chat_ready_label.configure(
                    text=f"Starting {self.active_model.get('name', 'model')} … waiting for the first token.",
                    text_color=WARN_TEXT,
                )
            elif self.active_model:
                self._chat_ready_label.configure(
                    text=f"Ready: {self.active_model.get('name', 'model loaded')}",
                    text_color=SUCCESS_TEXT,
                )
            else:
                self._chat_ready_label.configure(
                    text="Pick a model and backend above, then click Load selected.",
                    text_color=WARN_TEXT,
                )
        if hasattr(self, "_active_model_label") and self._active_model_label is not None:
            if self.active_model:
                self._active_model_label.configure(text=self._active_model_display)
            else:
                self._active_model_label.configure(text="None loaded")
        self._update_chat_copy_buttons()

    def _start_chat_first_token_feedback(self) -> None:
        """Show first-token latency feedback until streaming starts."""
        self._chat_first_token_started_at = time.monotonic()
        if hasattr(self, "_chat_first_token_progress") and self._chat_first_token_progress is not None:
            self._chat_first_token_progress.grid()
            self._chat_first_token_progress.start()
        self._update_chat_first_token_feedback()

    def _update_chat_first_token_feedback(self) -> None:
        if not self._chat_thinking or self._chat_first_token_started_at is None:
            return
        elapsed = int(max(0, time.monotonic() - self._chat_first_token_started_at))
        model_name = self.active_model.get("name", "model") if self.active_model else "model"
        message = (
            f"Starting {model_name} … waiting for first token ({elapsed}s). "
            "First run can load weights into memory/VRAM."
        )
        if hasattr(self, "_chat_ready_label") and self._chat_ready_label is not None:
            self._chat_ready_label.configure(text=message, text_color=WARN_TEXT)
        self.set_status(message)
        self._chat_first_token_timer_id = self.after(1000, self._update_chat_first_token_feedback)

    def _stop_chat_first_token_feedback(self, status_text: str | None = None) -> None:
        timer_id = self._chat_first_token_timer_id
        self._chat_first_token_timer_id = None
        self._chat_first_token_started_at = None
        if timer_id:
            try:
                self.after_cancel(timer_id)
            except Exception:
                pass
        if hasattr(self, "_chat_first_token_progress") and self._chat_first_token_progress is not None:
            self._chat_first_token_progress.stop()
            self._chat_first_token_progress.grid_remove()
        if status_text:
            if hasattr(self, "_chat_ready_label") and self._chat_ready_label is not None:
                self._chat_ready_label.configure(text=status_text, text_color=INFO_TEXT)
            self.set_status(status_text)

    def _last_chat_exchange(self) -> tuple[str, str] | None:
        """Return the most recent completed user prompt and assistant response."""
        response_index = None
        response_text = ""
        for idx in range(len(self.chat_history) - 1, -1, -1):
            item = self.chat_history[idx]
            if item.get("role") == "assistant" and str(item.get("content", "")).strip():
                response_index = idx
                response_text = str(item.get("content", "")).strip()
                break
        if response_index is None:
            return None

        prompt_text = ""
        for idx in range(response_index - 1, -1, -1):
            item = self.chat_history[idx]
            if item.get("role") == "user" and str(item.get("content", "")).strip():
                prompt_text = str(item.get("content", "")).strip()
                break
        return prompt_text, response_text

    def _update_chat_copy_buttons(self) -> None:
        exchange = self._last_chat_exchange()
        state = "normal" if exchange else "disabled"
        if hasattr(self, "_copy_last_response_btn") and self._copy_last_response_btn is not None:
            self._copy_last_response_btn.configure(state=state)
        if hasattr(self, "_copy_last_exchange_btn") and self._copy_last_exchange_btn is not None:
            self._copy_last_exchange_btn.configure(state=state)

    def _copy_text_to_clipboard(self, text: str, success_message: str) -> None:
        if not text.strip():
            self.set_status("Nothing to copy yet.")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
            self.set_status(success_message)
        except Exception as exc:
            self.set_status(f"Copy failed: {exc}")

    def _copy_last_chat_response(self) -> None:
        exchange = self._last_chat_exchange()
        if not exchange:
            self.set_status("No completed chat response to copy yet.")
            return
        self._copy_text_to_clipboard(exchange[1], "Last chat response copied.")

    def _copy_last_chat_exchange(self) -> None:
        exchange = self._last_chat_exchange()
        if not exchange:
            self.set_status("No completed prompt/response pair to copy yet.")
            return
        prompt, response = exchange
        text = f"Prompt:\n{prompt}\n\nResponse:\n{response}" if prompt else response
        self._copy_text_to_clipboard(text, "Last prompt and response copied.")

    # ── System page ───────────────────────────────────────────────────────────

    def _build_system_page(self):
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["system"] = page
        page.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            page, text="System Resources",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 12))

        self._sys_frame = ctk.CTkScrollableFrame(page)
        self._sys_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self._sys_frame.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)

        self._sys_widgets = {}
        rows = [
            ("cpu",     "CPU Usage"),
            ("ram",     "RAM"),
            ("storage", "Disk (models drive)"),
        ]
        for i, (key, label) in enumerate(rows):
            ctk.CTkLabel(self._sys_frame, text=label, font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
                         ).grid(row=i * 3, column=0, columnspan=2, sticky="w", padx=8, pady=(12, 0))
            bar = ctk.CTkProgressBar(self._sys_frame)
            bar.set(0)
            bar.grid(row=i * 3 + 1, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
            lbl = ctk.CTkLabel(self._sys_frame, text="", font=ctk.CTkFont(size=11), anchor="w", text_color=TEXT_MUTED)
            lbl.grid(row=i * 3 + 2, column=0, columnspan=2, sticky="w", padx=8)
            self._sys_widgets[key] = (bar, lbl)

        # GPU section (populated dynamically)
        self._gpu_start_row = len(rows) * 3
        self._gpu_labels = []

        # NPU section
        self._npu_label = ctk.CTkLabel(self._sys_frame, text="NPU: detecting …",
                                       font=ctk.CTkFont(size=13), anchor="w", text_color=TEXT_MUTED)
        self._npu_label.grid(row=self._gpu_start_row + 10, column=0, columnspan=2, sticky="w", padx=8, pady=(16, 4))

        # OpenVINO GenAI availability
        if OV_GENAI_AVAILABLE:
            ov_devs_list = available_ov_devices()
            ov_devs = ", ".join(ov_devs_list) if ov_devs_list else "none"
            ov_txt = f"OpenVINO GenAI: Available — devices: {ov_devs}"
            ov_color = SUCCESS_TEXT
        else:
            ov_txt = "OpenVINO GenAI: Not installed (run: pip install openvino openvino-genai)"
            ov_color = WARN_TEXT
        ctk.CTkLabel(self._sys_frame, text=ov_txt, font=ctk.CTkFont(size=11),
                     anchor="w", text_color=ov_color, wraplength=600
                     ).grid(row=self._gpu_start_row + 11, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        # ONNX/DirectML availability (fallback for non-Intel hardware)
        if DIRECTML_AVAILABLE:
            onnx_txt = "ONNX Runtime DirectML: Available (GPU/NPU acceleration ready)"
            onnx_color = SUCCESS_TEXT
        elif ONNX_AVAILABLE:
            onnx_txt = "ONNX Runtime: Available (CPU only)"
            onnx_color = WARN_TEXT
        else:
            onnx_txt = "ONNX Runtime: Not installed"
            onnx_color = TEXT_MUTED
        ctk.CTkLabel(self._sys_frame, text=onnx_txt, font=ctk.CTkFont(size=11),
                     anchor="w", text_color=onnx_color, wraplength=600
                     ).grid(row=self._gpu_start_row + 12, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        self._update_system_page()

    def _update_system_page(self):
        try:
            summary = system_info.get_system_summary(config.models_dir(self.cfg))
        except Exception:
            return

        # CPU
        cpu_pct = summary["cpu_percent"] / 100
        bar, lbl = self._sys_widgets["cpu"]
        bar.set(min(cpu_pct, 1.0))
        lbl.configure(text=f"{summary['cpu_percent']:.1f}%  —  {summary['platform'][:60]}")

        # RAM
        ram = summary["ram"]
        ram_pct = ram["percent"] / 100
        bar, lbl = self._sys_widgets["ram"]
        bar.set(min(ram_pct, 1.0))
        lbl.configure(text=(
            f"Used {_fmt_gb(ram['used_mb'] / 1024)} / "
            f"{_fmt_gb(ram['total_mb'] / 1024)}  —  "
            f"{_fmt_gb(ram['available_mb'] / 1024)} free"
        ))

        # Storage
        st = summary["storage"]
        st_pct = (st["used_gb"] / max(st["total_gb"], 0.001))
        bar, lbl = self._sys_widgets["storage"]
        bar.set(min(st_pct, 1.0))
        lbl.configure(text=(
            f"Free: {_fmt_gb(st['free_gb'])}  /  "
            f"Total: {_fmt_gb(st['total_gb'])}"
        ))

        # GPUs
        for w in self._gpu_labels:
            w.destroy()
        self._gpu_labels.clear()

        gpus = summary["gpus"]
        row_base = self._gpu_start_row
        if gpus:
            for gi, gpu in enumerate(gpus):
                r = row_base + gi * 3
                # v5.5.11 (Ron, 2026-05-26): Windows integrated GPUs (Intel
                # Iris/Arc Graphics, AMD Radeon Graphics, Snapdragon Adreno)
                # are tagged as unified_memory in system_info: they share
                # system RAM through DXGI Shared GPU Memory rather than
                # owning a real dedicated pool. Label them clearly so users
                # don't read "VRAM 0 / 16 GB" as physically dedicated VRAM.
                is_unified_igpu = bool(gpu.get("unified_memory")) and gpu.get("vendor") != "Apple"
                header_suffix = "  (shared with system RAM)" if is_unified_igpu else ""
                lbl_h = ctk.CTkLabel(
                    self._sys_frame,
                    text=f"GPU: {gpu['name']}  ({gpu['vendor']}){header_suffix}",
                    font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
                )
                lbl_h.grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(12, 0))
                bar = ctk.CTkProgressBar(self._sys_frame)
                vram_pct = gpu["vram_used_mb"] / max(gpu["vram_total_mb"], 1)
                bar.set(min(vram_pct, 1.0))
                bar.grid(row=r + 1, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
                if is_unified_igpu:
                    dedicated_mb = int(gpu.get("dedicated_vram_mb") or 0)
                    detail_text = (
                        f"Shared GPU memory: "
                        f"{_fmt_gb(gpu['vram_used_mb'] / 1024)} used / "
                        f"{_fmt_gb(gpu['vram_total_mb'] / 1024)} total system RAM  —  "
                        f"{_fmt_gb(gpu['vram_free_mb'] / 1024)} available"
                    )
                    if dedicated_mb > 0:
                        detail_text += f"  ({_fmt_gb(dedicated_mb / 1024)} dedicated carve-out)"
                else:
                    detail_text = (
                        f"VRAM used {_fmt_gb(gpu['vram_used_mb'] / 1024)} / "
                        f"{_fmt_gb(gpu['vram_total_mb'] / 1024)}  —  "
                        f"{_fmt_gb(gpu['vram_free_mb'] / 1024)} free"
                    )
                lbl_d = ctk.CTkLabel(
                    self._sys_frame,
                    text=detail_text,
                    font=ctk.CTkFont(size=11), anchor="w", text_color=TEXT_MUTED
                )
                lbl_d.grid(row=r + 2, column=0, columnspan=2, sticky="w", padx=8)
                self._gpu_labels.extend([lbl_h, bar, lbl_d])
        else:
            lbl = ctk.CTkLabel(
                self._sys_frame, text="No discrete GPU detected (using CPU / iGPU).",
                font=ctk.CTkFont(size=12), anchor="w", text_color=TEXT_MUTED
            )
            lbl.grid(row=row_base, column=0, columnspan=2, sticky="w", padx=8, pady=(12, 0))
            self._gpu_labels.append(lbl)

        # NPU
        npus = summary["npus"]
        if npus:
            txt = "NPU detected: " + ", ".join(f"{n['name']} ({n['status']})" for n in npus)
            color = SUCCESS_TEXT
        else:
            txt = "No NPU detected (AI PC feature — Snapdragon X / Intel Lunar Lake / AMD Strix)."
            color = TEXT_MUTED
        self._npu_label.configure(text=txt, text_color=color)

    # ── Logs page ─────────────────────────────────────────────────────────────

    def _build_logs_page(self):
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["logs"] = page
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(page, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        toolbar.grid_columnconfigure(6, weight=1)

        ctk.CTkLabel(toolbar, text="Log Level:", font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 4))
        self._log_level_var = ctk.StringVar(value="INFO")
        ctk.CTkOptionMenu(
            toolbar, values=["DEBUG", "INFO", "WARNING", "ERROR"],
            variable=self._log_level_var, width=110,
            command=lambda _: self._refresh_logs(),
            **self._option_menu_style(),
        ).grid(row=0, column=1)
        ctk.CTkLabel(toolbar, text="Category:", font=ctk.CTkFont(size=12)).grid(row=0, column=2, padx=(12, 4))
        self._log_category_var = ctk.StringVar(value="ALL")
        ctk.CTkOptionMenu(
            toolbar, values=["ALL", *logger.CATEGORIES],
            variable=self._log_category_var, width=140,
            command=lambda _: self._refresh_logs(),
            **self._option_menu_style(),
        ).grid(row=0, column=3)
        ctk.CTkLabel(toolbar, text="Search:", font=ctk.CTkFont(size=12)).grid(row=0, column=4, padx=(12, 4))
        self._log_search_var = ctk.StringVar(value="")
        log_search = ctk.CTkEntry(toolbar, textvariable=self._log_search_var, width=220)
        log_search.grid(row=0, column=5, sticky="w")
        self._log_search_var.trace_add("write", lambda *_: self._refresh_logs())
        ctk.CTkButton(toolbar, text="Copy Visible", width=95, command=self._copy_visible_logs, **self._outline_button_style()).grid(row=0, column=7, padx=(8, 0))
        ctk.CTkButton(toolbar, text="Export", width=70, command=self._export_visible_logs, **self._outline_button_style()).grid(row=0, column=8, padx=(8, 0))
        ctk.CTkButton(toolbar, text="Clear", width=70, command=self._clear_logs, **self._outline_button_style()).grid(row=0, column=9, padx=(8, 0))

        self._log_box = ctk.CTkTextbox(
            page, state="disabled", wrap="word",
            font=ctk.CTkFont(family="Consolas", size=11),
        )
        self._log_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))

    def _format_log_entry(self, entry: dict) -> str:
        category = str(entry.get("category") or logger.CATEGORY_SYSTEM)
        return f"[{entry['time']}] [{entry['level']:<7}] [{category:<10}] {entry['msg']}\n"

    def _log_progress_milestone(
        self,
        key: str,
        message: str,
        *,
        category: str,
        level: str = "INFO",
        completed: int = 0,
        total: int = 0,
        status: str = "",
        percent_step: int = 5,
        min_seconds: float = 15.0,
    ) -> None:
        state = self.__dict__.setdefault("_progress_log_state", {})
        previous = state.get(key, {})
        now = time.time()
        should_log = False
        bucket = previous.get("bucket")
        if total > 0:
            pct = max(0.0, min(100.0, (completed / total) * 100.0))
            bucket = int(pct // max(1, percent_step)) * max(1, percent_step)
            if completed >= total:
                bucket = 100
            should_log = bucket != previous.get("bucket")
        elif status and status != previous.get("status"):
            should_log = True
        if not should_log and now - float(previous.get("time", 0.0)) >= min_seconds:
            should_log = True
        if not should_log:
            return
        state[key] = {"bucket": bucket, "status": status, "time": now}
        log_fn = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error,
        }.get(level.upper(), logger.info)
        log_fn(message, category=category)

    def _refresh_logs(self):
        level_var = self.__dict__.get("_log_level_var")
        category_var = self.__dict__.get("_log_category_var")
        search_var = self.__dict__.get("_log_search_var")
        if level_var is None or search_var is None:
            return
        if not self._widget_is_alive(getattr(self, "_log_box", None)):
            self._log_box = None
            return
        category = category_var.get() if category_var is not None else "ALL"
        entries = logger.get_entries(
            level_var.get(),
            category=None if category == "ALL" else category,
        )
        search = search_var.get().strip().lower()
        try:
            self._log_box.configure(state="normal")
            self._log_box.delete("1.0", "end")
            for e in entries:
                line = self._format_log_entry(e)
                if search and search not in line.lower():
                    continue
                self._log_box.insert("end", line)
            self._log_box.configure(state="disabled")
            self._log_box.see("end")
        except TclError:
            self._log_box = None

    def _clear_logs(self):
        logger.clear()
        if not self._widget_is_alive(getattr(self, "_log_box", None)):
            self._log_box = None
            return
        try:
            self._log_box.configure(state="normal")
            self._log_box.delete("1.0", "end")
            self._log_box.configure(state="disabled")
        except TclError:
            self._log_box = None

    def _visible_log_text(self) -> str:
        if not self._widget_is_alive(getattr(self, "_log_box", None)):
            self._log_box = None
            return ""
        try:
            self._log_box.configure(state="normal")
            text = self._log_box.get("1.0", "end").rstrip()
            self._log_box.configure(state="disabled")
            return text
        except TclError:
            self._log_box = None
            return ""

    def _copy_visible_logs(self):
        text = self._visible_log_text()
        self.clipboard_clear()
        self.clipboard_append(text)
        self.set_status("Visible logs copied to clipboard.")

    def _export_visible_logs(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Export visible logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(self._visible_log_text() + "\n", encoding="utf-8")
            self.set_status(f"Logs exported: {path}")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)

    def _on_log_entry(self, entry: dict):
        if getattr(self, "_closing", False):
            return
        self._pending_log_entries.append(entry)
        if not self._log_flush_scheduled:
            self._log_flush_scheduled = True
            self.after(100, self._flush_log_entries)

    def _flush_log_entries(self):
        self._log_flush_scheduled = False
        if getattr(self, "_closing", False):
            self._pending_log_entries = []
            return
        entries = self._pending_log_entries
        self._pending_log_entries = []
        if not entries:
            return
        if not self._widget_is_alive(getattr(self, "_log_box", None)):
            self._log_box = None
            return
        for entry in entries:
            self._append_log_entry(entry, scroll=False)
        if self._widget_is_alive(getattr(self, "_log_box", None)):
            try:
                self._log_box.see("end")
            except TclError:
                self._log_box = None

    def _append_log_entry(self, entry: dict, scroll: bool = True):
        """Append a single log entry if it passes current level/category/search filters."""
        level_var = self.__dict__.get("_log_level_var")
        category_var = self.__dict__.get("_log_category_var")
        if level_var is None or not hasattr(self, "_log_box"):
            return  # UI not yet built
        if not self._widget_is_alive(self._log_box):
            self._log_box = None
            return
        _level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        min_level = _level_order.get(level_var.get(), 1)
        if _level_order.get(entry.get("level", "INFO"), 1) < min_level:
            return
        selected_category = category_var.get() if category_var is not None else "ALL"
        if selected_category != "ALL":
            entry_category = str(entry.get("category") or logger.CATEGORY_SYSTEM)
            if entry_category != selected_category:
                return
        line = self._format_log_entry(entry)
        search = self._log_search_var.get().strip().lower() if hasattr(self, "_log_search_var") else ""
        if search and search not in line.lower():
            return
        try:
            self._log_box.configure(state="normal")
            self._log_box.insert("end", line)
            self._log_box.configure(state="disabled")
            if scroll:
                self._log_box.see("end")
        except TclError:
            self._log_box = None

    # ── Settings page ─────────────────────────────────────────────────────────

    # ── Benchmark page ──────────────────────────────────────────────────────

    def _bench_required_deps_for_model(self, model: dict) -> list[str]:
        if not model.get("phase1_adapter"):
            return []
        return phase1_adapters.required_dependencies_for_model(model)

    def _bench_missing_deps_for_model(self, model: dict) -> list[str]:
        return phase1_adapters.missing_dependencies_for_model(model)

    def _bench_preflight_classify_impossible(
        self, model_ids: list[str] | set[str],
    ) -> list[dict]:
        """Return models from *model_ids* that are physically impossible.

        v5.5.4: deliberately conservative classification used by the
        Force-All pre-flight dialog. A model is flagged ONLY when there is
        no realistic way for it to load or download on this host:

        * ``min_ram_gb > 2.0 * total_ram_gb`` (even with paging, the model
          can't load — e.g. a 240 GB model on a 32 GB cloud VM).
        * ``size_gb > free_disk_gb`` (the download itself can't fit, so the
          run will fail at pull time).

        Things that "might fit with Force-All" (image-gen on CPU rigs,
        models a little over the RAM gate, etc.) are intentionally NOT
        flagged here because the whole point of Force-All is to attempt
        them.  This list is what the dialog offers to uncheck so the user
        can dismiss with the maximum runnable selection.

        Returns a list of ``{"id", "name", "reason"}`` dicts in the same
        order as ``self._catalog_models``.
        """
        if not model_ids:
            return []
        id_set = {str(mid) for mid in model_ids}
        capacity = self._bench_profile_capacity()
        total_ram_gb = float(capacity.get("total_ram_gb", 0) or 0)
        try:
            from src import resource_manager as _rm
            free_disk_gb = float(_rm.get_free_disk_gb(self.cfg.get("models_dir", ".")))
        except Exception:
            free_disk_gb = 0.0
        impossible: list[dict] = []
        for model in self._catalog_models:
            mid = str(model.get("id") or "")
            if mid not in id_set:
                continue
            name = str(model.get("name") or mid)
            min_ram = float(model.get("min_ram_gb", 0) or 0)
            size_gb = float(model.get("size_gb", 0) or 0)
            if min_ram > 0 and total_ram_gb > 0 and min_ram > 2.0 * total_ram_gb:
                impossible.append({
                    "id": mid,
                    "name": name,
                    "reason": (
                        f"needs {min_ram:.0f} GB RAM but this host only has "
                        f"{total_ram_gb:.0f} GB (over 2x — no way to page in)"
                    ),
                })
                continue
            if size_gb > 0 and free_disk_gb > 0 and size_gb > free_disk_gb:
                impossible.append({
                    "id": mid,
                    "name": name,
                    "reason": (
                        f"download is {size_gb:.1f} GB but only "
                        f"{free_disk_gb:.1f} GB free on disk"
                    ),
                })
        return impossible

    def _bench_default_models_for(self, profile: str, run_mode: str) -> set[str]:
        """Return the resolved default-tick model set for a SKU profile + run mode.

        SKU display names + per-SKU model sets are owned by ``skus.json``
        (see ``system_info.BENCHMARK_SKU_PROFILES`` and
        ``system_info.resolve_bench_models``). Unknown profile (e.g.
        "This Device") or unknown run mode returns the empty set, which the
        caller treats as "no observed evidence for this profile".
        """
        if not profile or profile == "This Device":
            return set()
        if run_mode not in ("quick", "extended"):
            return set()
        field = "bench_quick_models" if run_mode == "quick" else "bench_extended_models"
        target = profile.casefold()
        for sku in system_info.BENCHMARK_SKU_PROFILES:
            if str(sku.get("name") or "").casefold() == target:
                resolved = sku.get(field)
                if isinstance(resolved, (set, frozenset)):
                    return set(resolved)
                if isinstance(resolved, (list, tuple)):
                    return {str(x) for x in resolved}
                return set()
        return set()

    def _bench_profile_has_default_models(self, profile: str, run_mode: str) -> bool:
        """Return True when the SKU profile has *any* default-tick models for
        the given run mode (i.e. it appears in the loaded SKU catalog AND has
        a non-empty resolved set). Drives the "observed data exists" branch
        in ``_bench_default_selected_for_model``.
        """
        return bool(self._bench_default_models_for(profile, run_mode))

    def _bench_observed_success_for_profile(self, model: dict, capacity: dict) -> bool:
        """Return True if the model is in the default-tick set for the active
        run mode + profile.

        v5.5.12 SKU decoupling (Ron, 2026-05-28) — the per-SKU sets now live
        in ``skus.json`` and are resolved via
        ``_bench_default_models_for``. The method name is kept for back-compat
        with the existing ``_bench_model_available_for_profile`` /
        ``_bench_default_selected_for_model`` call sites.
        """
        profile = str(capacity.get("profile") or "")
        if not profile or profile == "This Device":
            return False
        model_id = str(model.get("id") or "")
        run_mode = self._active_run_mode()
        return model_id in self._bench_default_models_for(profile, run_mode)

    def _bench_default_fit_for_model(
        self,
        model: dict,
        *,
        available_ram_gb: float,
        total_ram_gb: float,
        vram_capacity_gb: float,
        has_gpu: bool,
    ) -> tuple[bool, str]:
        """Return whether the model should be selected by default for this machine capacity."""
        if model.get("phase1_adapter"):
            missing = self._bench_missing_deps_for_model(model)
            if missing:
                return False, f"missing {', '.join(missing[:3])}"

        min_ram = float(model.get("min_ram_gb") or 0)
        min_vram = float(model.get("min_vram_gb") or 0)

        # Catalog/SKU fit is based on physical capacity, not the transient free
        # RAM left after Windows, LocalAI, and a previously loaded model. A
        # 16 GB cloud VM often reports ~15.9 GB, so round to the advertised tier.
        capacity_ram_gb = max(total_ram_gb, round(total_ram_gb))

        cpu_viable = min_ram <= capacity_ram_gb
        gpu_viable = has_gpu and min_vram > 0 and min_vram <= vram_capacity_gb
        if cpu_viable or gpu_viable:
            return True, "OK"

        if has_gpu and min_vram > vram_capacity_gb and min_vram > 0:
            return False, f"needs {min_vram:g} GB VRAM; {vram_capacity_gb:.0f} GB installed"
        return False, f"needs {min_ram:g} GB RAM; {capacity_ram_gb:.0f} GB installed"

    def _bench_model_available_for_profile(
        self,
        model: dict,
        capacity: dict,
        *,
        allow_oversize: bool = False,
        force_all: bool = False,
    ) -> tuple[bool, str]:
        """Return whether a model row should be selectable for this profile.

        ``force_all`` (v5.5.2+) bypasses *capacity* gates (RAM/VRAM fit), but
        NEVER bypasses the hard gates above it: a model with no backend or
        with missing optional dependencies is still unavailable. The runner
        layer would skip such rows even if we forced them on, so showing them
        as runnable would lie to the user.
        """
        methods = self._bench_methods_for_ui(model)
        if not methods:
            return False, "no benchmark backend"
        if model.get("phase1_adapter"):
            missing = self._bench_missing_deps_for_model(model)
            if missing:
                return False, f"missing {', '.join(missing[:3])}"
        # v5.5.1 UX fix: image-gen models always show in the list, but on
        # non-GPU systems they're only checkable when Force All is on.
        # This is the per-row replacement for the removed
        # "Include image-gen models" checkbox.
        if self._is_image_model_ui(model):
            if not capacity.get("has_gpu") and not force_all:
                return False, "image gen requires GPU (use Force All to override)"
        if self._bench_observed_success_for_profile(model, capacity):
            return True, "successful prior SKU run"

        fit_ok, fit_reason = self._bench_default_fit_for_model(
            model,
            available_ram_gb=capacity["available_ram_gb"],
            total_ram_gb=capacity["total_ram_gb"],
            vram_capacity_gb=capacity["vram_capacity_gb"],
            has_gpu=capacity["has_gpu"],
        )
        if fit_ok:
            return True, "OK"
        if force_all:
            return True, f"Force-All: {fit_reason}"
        if allow_oversize:
            return True, f"override: {fit_reason}"
        return False, fit_reason

    def _bench_default_selected_for_model(self, model: dict, capacity: dict) -> bool:
        """Return whether a selectable model should be checked by default."""
        if not self._bench_methods_for_ui(model):
            return False
        if model.get("phase1_adapter"):
            missing = self._bench_missing_deps_for_model(model)
            if missing:
                return False
        # v5.5.1: image-gen rows never default-checked without a GPU.
        # The user can still tick them after enabling Force All, which is
        # exactly the "rare CPU-only image-gen run" path Ron called out.
        if self._is_image_model_ui(model) and not bool(capacity.get("has_gpu")):
            return False

        profile = str(capacity.get("profile") or "")
        run_mode = self._active_run_mode()
        # v5.5.12 SKU decoupling: the per-SKU verified-passers table is
        # resolved from skus.json via ``_bench_default_models_for``.
        # A SKU with any default-tick models for the active mode is treated
        # as "observed evidence exists" — only models in that resolved set
        # default-tick; everything else is fits-but-unticked (force-only).
        observed_profile_has_data = self._bench_profile_has_default_models(profile, run_mode)
        observed_success = self._bench_observed_success_for_profile(model, capacity)
        fit_ok, _reason = self._bench_default_fit_for_model(
            model,
            available_ram_gb=capacity["available_ram_gb"],
            total_ram_gb=capacity["total_ram_gb"],
            vram_capacity_gb=capacity["vram_capacity_gb"],
            has_gpu=capacity["has_gpu"],
        )
        if observed_success:
            return True
        if not fit_ok:
            return False

        if not capacity.get("is_sku") or profile == "This Device":
            # v2026.06.01.8 Quick-mode fallback fix (Ron, 2026-06-01).
            # The synthetic "This Device" profile is the only entry in
            # the SKU dropdown when skus.json is absent — and ``is_sku``
            # is False in that mode. Returning True for every fitting
            # model on Quick auto-ticks 40+ rows on a capable GPU and
            # silently breaks the "Quick is a sub-3-minute smoke set"
            # contract. Quick now restricts to the lean fallback
            # baseline that mirrors the skus.json ``quick_chat_ultra_small``
            # + ``quick_image_smallest`` sets, so Quick behaves the same
            # whether or not skus.json is loaded. Extended on This Device
            # keeps the legacy "everything that fits is default-ticked"
            # behavior — there is no per-SKU verified-passer set to
            # defer to without skus.json, and Extended is the right
            # place for "run every model my hardware can handle".
            if run_mode == "quick":
                fallback_ids = _bench_quick_fallback_model_ids(
                    has_gpu=bool(capacity.get("has_gpu"))
                )
                return str(model.get("id") or "") in fallback_ids
            return True
        if observed_profile_has_data:
            # Per Ron 2026-05-24 recalibration: a SKU with verified
            # data only auto-ticks models in its mode-specific set
            # (Quick or Extended). Everything else must be manually
            # ticked or Force-All'd — auto-ticking unknown rows would
            # break the "all default-checked tests are expected to
            # pass" contract for both modes.
            return False

        recommended_raw = [
            str(item) for item in system_info.get_recommended_skus_for_model(model.get("id"), model)
        ]
        recommended_for = [item.lower() for item in recommended_raw]
        if profile.lower() in recommended_for:
            return True
        for recommended_profile in recommended_raw:
            spec = next(
                (s for s in self._bench_profile_specs() if str(s.get("name", "")).lower() == recommended_profile.lower()),
                None,
            )
            if not spec:
                continue
            rec_ram = float(spec.get("ram_gb", 0) or 0)
            rec_vram = float(spec.get("vram_gb", 0) or 0)
            if rec_ram <= float(capacity.get("total_ram_gb") or 0):
                if rec_vram <= 0 or (
                    bool(capacity.get("has_gpu")) and rec_vram <= float(capacity.get("vram_capacity_gb") or 0)
                ):
                    return True

        # Conservative fallback for CPU-only profiles: tiny/small CPU models
        # remain safe defaults even when their catalog recommendations predate
        # the public Benchmark profile names.
        if not capacity.get("has_gpu"):
            category = str(model.get("category") or "")
            min_ram = float(model.get("min_ram_gb") or 0)
            if category in {"Ultra Small", "Small"} and min_ram <= min(8.0, float(capacity.get("total_ram_gb") or 0)):
                return True
        return False

    def _bench_profile_specs(self) -> list[dict]:
        """Return Benchmark profile specs sourced from ``skus.json``.

        v5.5.12 SKU decoupling: the previous public/optional split is gone —
        ``system_info.BENCHMARK_SKU_PROFILES`` is loaded from the same JSON
        and kept in sync with ``self._optional_skus``. We still iterate
        ``self._optional_skus`` to pick up any session-only injections
        (e.g. ``_inject_local_sku`` for an unmatched local hardware
        fingerprint) that may not have round-tripped through
        ``_apply_optional_skus_to_modules``.
        """
        specs = [dict(s) for s in system_info.BENCHMARK_SKU_PROFILES]
        if getattr(self, "_optional_skus_enabled", False):
            for sku in getattr(self, "_optional_skus", []):
                name = sku.get("name")
                if not name:
                    continue
                existing = next((s for s in specs if s.get("name") == name), None)
                if existing is not None:
                    existing.update(sku)
                else:
                    specs.append(dict(sku))
        if not specs:
            # Last-ditch fallback when skus.json is missing or unreadable:
            # one synthetic "This Device" entry built from local hardware so
            # the benchmark UI still has at least one selectable profile.
            specs = system_info.get_benchmark_sku_profiles()
        return specs

    def _bench_profile_values(self) -> list[str]:
        values: list[str] = [str(s["name"]) for s in self._bench_profile_specs() if s.get("name")]
        values.append("This Device")
        seen: set[str] = set()
        return [v for v in values if not (v in seen or seen.add(v))]

    def _bench_profile_for_sku(self, sku: dict | None) -> str:
        if not sku:
            return "This Device"
        profile_name = system_info.benchmark_sku_profile_name(
            sku.get("cpu", 0),
            sku.get("ram_gb", 0),
            sku.get("vram_gb", 0),
        )
        values = self._bench_profile_values()
        if profile_name in values:
            return profile_name
        sku_name = sku.get("name")
        if sku_name in values:
            return str(sku_name)
        return "This Device"

    def _bench_default_profile_name(self) -> str:
        values = self._bench_profile_values()
        try:
            active = self._optional_filter_var.get()
        except Exception:
            active = ""
        if active in values and active != "This Device":
            return active
        sku_profile = self._bench_profile_for_sku(getattr(self, "_optional_sku", None))
        if sku_profile in values:
            return sku_profile
        if active in values:
            return active
        return values[0]

    def _bench_profile_capacity(self, profile_name: str | None = None) -> dict:
        profile = profile_name
        if profile is None and hasattr(self, "_bench_profile_var"):
            profile = self._bench_profile_var.get()
        profile = profile or self._bench_default_profile_name()

        sku_spec = next((s for s in self._bench_profile_specs() if s.get("name") == profile), None)
        if sku_spec:
            ram_gb = float(sku_spec.get("ram_gb", 0) or 0)
            vram_gb = float(sku_spec.get("vram_gb", 0) or 0)
            return {
                "profile": sku_spec.get("name", profile),
                "available_ram_gb": ram_gb,
                "total_ram_gb": ram_gb,
                "vram_capacity_gb": vram_gb,
                "has_gpu": vram_gb > 0,
                "is_sku": True,
            }

        gpus = system_info.get_gpu_info()
        ram = system_info.get_ram_info()
        total_ram_gb = float(ram.get("total_mb", 0) or 0) / 1024
        vram_gb = max((_g.get("vram_total_mb", 0) for _g in gpus), default=0) / 1024
        return {
            "profile": "This Device",
            "available_ram_gb": float(ram.get("available_mb", 0) or 0) / 1024,
            "total_ram_gb": total_ram_gb,
            "vram_capacity_gb": vram_gb,
            "has_gpu": vram_gb > 0,
            "is_sku": False,
        }

    def _bench_capacity_label(self, capacity: dict | None = None) -> str:
        # Intentionally strips RAM/VRAM specs — SKU hardware specs
        # must not be exposed in user-facing UI, logs, or status messages.
        cap = capacity or self._bench_profile_capacity()
        return str(cap.get("profile") or "Benchmark profile")

    def _bench_methods_for_ui(self, model: dict) -> list[str]:
        run_mode = self._active_run_mode()
        if model.get("phase1_adapter"):
            # Utility / Toolbox models are intentionally excluded from
            # benchmark runs. They are exercised through the Toolbox tab
            # instead. Returning [] hides them from the checklist entirely.
            return []
        methods: list[str] = []
        if model.get("ollama_tag"):
            methods.append("ollama")
        if model.get("onnx_repo"):
            methods.append("onnx")
        # v5.5.3 B1 contract: image-gen rows ALWAYS appear in the list under
        # Extended run mode — including on CPU-only profiles where the
        # rows render disabled with a "GPU required (use Force All to
        # override)" reason and become user-checkable when Force All is on.
        # Earlier rounds gated this on ``capacity.get("has_gpu")``, which
        # made the rows invisible on CPU rigs and turned the Force-All
        # bypass at _bench_model_available_for_profile into dead code.
        #
        # v5.5.8 (Ron, 2026-05-24): Quick mode on GPU now also surfaces
        # image-gen rows so the curated Quick image row (playground v2.5
        # on the higher-VRAM GPU SKUs / realistic-vision-v6 on the entry
        # GPU SKU) can default-tick and run.
        if not methods and self._is_image_model_ui(model):
            if run_mode in {"extended", "quick"}:
                methods.append("image")
        return methods

    def _bench_toggle_value(self, attr: str, default: bool) -> bool:
        var = self.__dict__.get(attr)
        if var is None:
            return default
        try:
            return bool(var.get())
        except (AttributeError, TclError, TypeError):
            return default

    def _bench_method_fits_capacity(
        self,
        model: dict,
        method: str,
        capacity: dict,
        *,
        allow_oversize: bool = False,
    ) -> bool:
        min_ram = float(model.get("min_ram_gb") or 0)
        min_vram = float(model.get("min_vram_gb") or 0)
        capacity_ram_gb = max(
            float(capacity.get("total_ram_gb") or 0),
            round(float(capacity.get("total_ram_gb") or 0)),
        )
        total_vram_gb = float(capacity.get("vram_capacity_gb") or 0)
        has_gpu = bool(capacity.get("has_gpu")) and total_vram_gb > 0

        if method == "image":
            # v5.5.6+: mirror BatchRunner._image_gen_supported — Force All /
            # allow_oversize lets CPU-only profiles attempt image-gen.
            # They'll be slow and most will OOM, but the smallest models
            # (e.g. Realistic Vision v6 @ 2 GB) succeed on 16+ core CPU systems
            # with enough RAM, and the adaptive OOM ceiling auto-skips
            # bigger ones once the first failure pins the wall.
            # v5.5.9 (Ron, 2026-05-26): Snapdragon X is the one exception
            # — torch-directml has no Windows-ARM64 wheel, so ComfyUI
            # startup crashes with the ``torch_library_impl ...
            # _torchaudio.pyd`` popup regardless of Force All. Always
            # reject image-gen on Snapdragon.
            if is_snapdragon_arm64():
                return False
            if not has_gpu:
                if not allow_oversize:
                    return False
                # CPU image-gen under Force All: same RAM gate as text rows.
                # min_vram is irrelevant here (no GPU); rely on the RAM
                # capacity gate and let the runtime ceiling handle OOMs.
                return capacity_ram_gb >= min_ram or allow_oversize
            return min_vram <= 0 or total_vram_gb + 0.001 >= min_vram or allow_oversize

        if method == "onnx_openvino":
            if not has_gpu:
                return False
            return capacity_ram_gb >= min_ram or allow_oversize

        if method in {"ollama_gpu", "onnx_directml"}:
            if not has_gpu:
                return False
            if min_vram <= 0:
                return False
            return total_vram_gb >= min_vram or allow_oversize

        return capacity_ram_gb >= min_ram or allow_oversize

    def _bench_methods_for_run_ui(
        self,
        model: dict,
        *,
        capacity: dict | None = None,
        allow_oversize: bool | None = None,
    ) -> list[str]:
        """Mirror BatchRunner._methods_for using the current Benchmark toggles."""
        if model.get("benchmark_skip_reason"):
            return []
        skipped_methods = _benchmark_skip_methods(model)
        capacity = capacity or self._bench_profile_capacity()
        if allow_oversize is None:
            allow_oversize = self._bench_toggle_value("_bench_resource_override_var", False)
        run_mode = self._active_run_mode()

        if self._is_image_model_ui(model):
            # v5.5.8 (Ron, 2026-05-24): Quick mode runs ONE curated image
            # row on GPU SKUs (playground v2.5 / realistic-vision-v6 —
            # see ``bench_quick_models`` per SKU in skus.json).
            # Extended runs every image-gen row that fits. Both gate on
            # capacity.
            if (
                run_mode in {"extended", "quick"}
                and self._bench_method_fits_capacity(
                    model, "image", capacity, allow_oversize=allow_oversize
                )
            ):
                return ["image"]
            return []

        if model.get("phase1_adapter"):
            # Utility models never produce benchmark methods — they live in
            # the Toolbox tab instead. Keep this gate even if the back-compat
            # _bench_utility_var var is toggled on elsewhere.
            return []

        methods: list[str] = []
        if model.get("ollama_tag"):
            if (
                self._bench_toggle_value("_bench_gpu_var", bool(capacity.get("has_gpu")))
                and self._bench_method_fits_capacity(
                    model, "ollama_gpu", capacity, allow_oversize=allow_oversize
                )
            ):
                methods.append("ollama_gpu")
            if (
                self._bench_toggle_value("_bench_cpu_var", True)
                and self._bench_method_fits_capacity(
                    model, "ollama_cpu", capacity, allow_oversize=allow_oversize
                )
            ):
                methods.append("ollama_cpu")

        if (
            model.get("onnx_repo")
            and self._bench_toggle_value("_bench_onnx_var", ONNX_AVAILABLE)
            and ONNX_AVAILABLE
        ):
            if OPENVINO_AVAILABLE and self._bench_method_fits_capacity(
                model, "onnx_openvino", capacity, allow_oversize=allow_oversize
            ):
                methods.append("onnx_openvino")
            if DIRECTML_AVAILABLE and self._bench_method_fits_capacity(
                model, "onnx_directml", capacity, allow_oversize=allow_oversize
            ):
                methods.append("onnx_directml")
            if self._bench_method_fits_capacity(
                model, "onnx_cpu", capacity, allow_oversize=allow_oversize
            ):
                methods.append("onnx_cpu")
        return [method for method in methods if method not in skipped_methods]

    @staticmethod
    def _is_image_model_ui(model: dict) -> bool:
        """Mirror BatchRunner._is_image_model — image-gen models have a
        ComfyUI artifact or the explicit comfyui backend."""
        if not isinstance(model, dict):
            return False
        if (model.get("backend") or "").lower() == "comfyui":
            return True
        if model.get("comfyui_model") or model.get("comfyui_model_url"):
            return True
        category = (model.get("category") or "").strip().lower()
        return category == "image generation"

    def _apply_bench_profile_method_defaults(self) -> None:
        if not hasattr(self, "_bench_gpu_var"):
            return
        capacity = self._bench_profile_capacity()
        has_gpu = bool(capacity.get("has_gpu"))
        self._bench_gpu_var.set(has_gpu)
        self._bench_cpu_var.set(True)
        if hasattr(self, "_bench_gpu_check"):
            self._bench_gpu_check.configure(
                state="normal" if has_gpu else "disabled",
                text="Ollama GPU" if has_gpu else "Ollama GPU (unavailable for profile)",
            )
        # v5.5.1 UX fix: ``_bench_image_var`` is no longer a user-facing
        # toggle. It stays True so legacy snapshot/restore paths behave as if
        # image-gen was always included. Per-row enable/default-check state
        # for image-gen models is computed by _bench_model_available_for_profile
        # (GPU presence + Force All) and _bench_default_selected_for_model.
        if hasattr(self, "_bench_image_var"):
            self._bench_image_var.set(True)

    def _refresh_bench_profile_values(self, preserve_selection: bool = True) -> None:
        if not hasattr(self, "_bench_profile_var"):
            return
        values = self._bench_profile_values()
        if hasattr(self, "_bench_profile_segbtn") and self._bench_profile_segbtn is not None:
            self._bench_profile_segbtn.configure(values=values)
        if self._bench_profile_var.get() not in values:
            self._bench_profile_var.set(self._bench_default_profile_name())
        if hasattr(self, "_bench_checklist_scroll"):
            self._render_bench_model_checklist(preserve_selection=preserve_selection)

    def _on_bench_profile_changed(self, _value: str | None = None) -> None:
        self._apply_bench_profile_method_defaults()
        self._render_bench_model_checklist(preserve_selection=False)
        self.set_status(f"Benchmark profile set to {self._bench_capacity_label()}.")

    def _render_bench_model_checklist(self, preserve_selection: bool = False) -> None:
        if not hasattr(self, "_bench_checklist_scroll"):
            return
        checklist = self._bench_checklist_scroll
        old_selected = set()
        have_prior_state = (
            preserve_selection
            and hasattr(self, "_bench_model_vars")
            and bool(self._bench_model_vars)
        )
        if have_prior_state:
            old_selected = {mid for mid, var in self._bench_model_vars.items() if var.get()}
        elif preserve_selection:
            # Page was built before the async catalog finished loading, so
            # _bench_model_vars exists but is empty. Preserving that "selection"
            # would leave every model unchecked on the first real render. Fall
            # back to fit-based defaults so the current system's tests are
            # selected automatically.
            preserve_selection = False
        for child in checklist.winfo_children():
            child.destroy()

        num_cols = 3
        for c in range(num_cols):
            checklist.grid_columnconfigure(c, weight=1)

        category_order = [
            "Ultra Small", "Small", "Medium", "Large", "Extra Large",
            "Vision", "Speech", "Embeddings", "Document AI", "Image Generation",
        ]
        category_colors = {
            "Ultra Small":      INFO_TEXT,
            "Small":            SUCCESS_TEXT,
            "Medium":           WARN_TEXT,
            "Large":            WARN_TEXT,
            "Extra Large":      ERROR_TEXT,
            "Vision":           INFO_TEXT,
            "Speech":           LINK_TEXT,
            "Embeddings":       SUCCESS_TEXT,
            "Document AI":      WARN_TEXT,
            "Image Generation": LINK_TEXT,
        }

        capacity = self._bench_profile_capacity()
        allow_oversize = bool(
            hasattr(self, "_bench_resource_override_var")
            and self._bench_resource_override_var.get()
        )
        # v5.5.2: Force-All lifts the per-row capacity gate at render time so
        # the user can actually SEE the rows light up + check. Without this,
        # the checklist still grays out every too-big-for-the-SKU model and
        # the "All" preset (called by _on_bench_force_all_toggle) silently
        # leaves them unchecked - defeating the whole point of Force-All on
        # small SKUs. The runner's own allow_oversize + smart-skip remains
        # the source of truth for what actually runs.
        force_all = bool(
            self.__dict__.get("_bench_force_all_var") is not None
            and self._bench_force_all_var.get()
        )

        by_cat: dict[str, list] = {}
        for model in sorted(self._catalog_models, key=lambda m: m.get("size_gb", 0)):
            if not self._bench_methods_for_ui(model):
                continue
            by_cat.setdefault(model.get("category", "Other"), []).append(model)

        self._bench_model_vars: dict[str, ctk.BooleanVar] = {}
        self._bench_model_by_id: dict[str, dict] = {
            m["id"]: m for m in self._catalog_models if m.get("id")
        }
        self._bench_model_disabled_ids: set[str] = set()
        self._bench_model_oversize_ids: set[str] = set()
        # Per-category state: header button, body frame, checkbox var, ordered
        # model id list. Reused by the toggle/preset/footer helpers below.
        self._bench_category_state: dict[str, dict] = {}
        self._bench_category_select_vars: dict[str, ctk.BooleanVar] = {}
        # Guard so the per-category select-all trace doesn't fight individual-row
        # traces while we're propagating a bulk change.
        self._bench_category_sync_in_progress: set[str] = set()
        self._bench_selection_sync_in_progress = False
        # Honor the user's prior expand/collapse choices across re-renders.
        prior_collapsed: set[str] = getattr(self, "_bench_collapsed_categories", set())
        if not isinstance(prior_collapsed, set):
            prior_collapsed = set()
        # First render: collapse every group by default so the page is scannable.
        # _bench_collapsed_categories starts as None on the first call.
        first_render = not hasattr(self, "_bench_collapsed_categories")
        self._bench_collapsed_categories = (
            set(by_cat.keys()) if first_render else prior_collapsed & set(by_cat.keys())
        )
        # Visual banding: alternate category card backgrounds so users can
        # eyeball where one category ends and the next begins.  In post-v5.3.6,
        # each card also carries the same border chrome the Models page
        # ``ModelListRow`` uses (``border_width=1``, ``border_color=BORDER_STRONG``,
        # ``corner_radius=6``) so the benchmark "table" rows visually match
        # the model list "table" rows.  Ron's request: the per-category card
        # should read as a row of the benchmark table the way Aya Expanse /
        # DeepSeek-R1 rows read as rows of the model list.
        banded_colors = (SURFACE_INNER, SURFACE_CARD)
        grid_row = 0
        visible_idx = 0
        for cat in category_order:
            if cat not in by_cat:
                continue
            color = category_colors.get(cat, "gray70")
            category_models = by_cat[cat]
            cat_ids: list[str] = []

            # Per-category card frame provides the visual band + rounded corners
            # + the same BORDER_STRONG outline the model list rows use.
            band_color = banded_colors[visible_idx % len(banded_colors)]
            visible_idx += 1
            card = ctk.CTkFrame(
                checklist,
                fg_color=band_color,
                corner_radius=6,
                border_width=1,
                border_color=BORDER_STRONG,
            )
            card.grid(
                row=grid_row, column=0, columnspan=num_cols,
                sticky="ew", padx=(0, 6), pady=(2, 4),
            )
            card.grid_columnconfigure(2, weight=1)
            grid_row += 1

            # Header row: [select-all checkbox] [▾ label · counts · preview].
            # The checkbox toggles every eligible (non-disabled) row in this
            # category; clicking the label/arrow toggles collapse/expand.
            select_all_var = ctk.BooleanVar(value=False)
            self._bench_category_select_vars[cat] = select_all_var

            select_all_cb = ctk.CTkCheckBox(
                card, text="", variable=select_all_var,
                width=20, checkbox_width=18, checkbox_height=18,
                command=lambda c=cat: self._toggle_bench_category_selection(c),
            )
            select_all_cb.grid(row=0, column=0, sticky="w", padx=(8, 4), pady=(4, 2))

            header_btn = ctk.CTkButton(
                card, text="",
                anchor="w",
                fg_color="transparent",
                hover_color=("#dcdcdc", "#3a3a3a"),
                text_color=color,
                font=ctk.CTkFont(size=11, weight="bold"),
                height=24,
                corner_radius=6,
                command=lambda c=cat: self._toggle_bench_category(c),
            )
            header_btn.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(4, 2))

            body = ctk.CTkFrame(card, fg_color="transparent")
            body.grid(
                row=1, column=0, columnspan=3,
                sticky="ew", padx=(28, 8), pady=(0, 6),
            )
            for c in range(num_cols):
                body.grid_columnconfigure(c, weight=1)

            for j, model in enumerate(category_models):
                observed_default = self._bench_observed_success_for_profile(model, capacity)
                run_mode_label = self._active_run_mode()
                available, availability_reason = self._bench_model_available_for_profile(
                    model, capacity, allow_oversize=allow_oversize, force_all=force_all,
                )
                fit_ok, fit_reason = self._bench_default_fit_for_model(
                    model,
                    available_ram_gb=capacity["available_ram_gb"],
                    total_ram_gb=capacity["total_ram_gb"],
                    vram_capacity_gb=capacity["vram_capacity_gb"],
                    has_gpu=capacity["has_gpu"],
                )
                disabled = not available
                default_selected = (
                    (model["id"] in old_selected)
                    if preserve_selection
                    else self._bench_default_selected_for_model(model, capacity)
                )
                var = ctk.BooleanVar(value=bool(default_selected and not disabled))
                # Re-render the header/footer once when a per-row checkbox flips.
                var.trace_add(
                    "write",
                    lambda *_a, c=cat: self._on_bench_model_selection_changed(c),
                )
                self._bench_model_vars[model["id"]] = var
                cat_ids.append(model["id"])
                methods = ", ".join(self._bench_methods_for_ui(model))
                label = f"{model['name']} ({model['size_gb']}G · {methods})"
                if disabled:
                    self._bench_model_disabled_ids.add(model["id"])
                    label += f" · {availability_reason}"
                elif not fit_ok:
                    self._bench_model_oversize_ids.add(model["id"])
                    label += f" · Advanced: {fit_reason}"
                elif default_selected:
                    # v5.5.8: badge text reflects which set sourced the
                    # default tick — Quick set, Extended set, or the
                    # catalog recommended_for fallback (used only on
                    # profiles without verified results data).
                    if observed_default:
                        label += (
                            " · Quick set"
                            if run_mode_label == "quick"
                            else " · Extended set"
                        )
                    else:
                        label += " · Recommended"
                cb = ctk.CTkCheckBox(
                    body, text=label, variable=var,
                    font=ctk.CTkFont(size=10), checkbox_width=16, checkbox_height=16,
                    text_color=TEXT_DISABLED if disabled else color,
                    state="disabled" if disabled else "normal",
                )
                if disabled:
                    # v5.5.3 a11y: skip disabled rows in Tab traversal.
                    # Without takefocus=False, CTk's outer CTkFrame keeps the
                    # row in keyboard focus order, so non-GPU users would
                    # Tab through every disabled image-gen row before
                    # reaching the next interactive control.
                    try:
                        cb.configure(takefocus=False)
                    except Exception:
                        pass
                    try:
                        cb._canvas.configure(takefocus=0)
                    except Exception:
                        pass
                cb.grid(row=j // num_cols, column=j % num_cols,
                        sticky="w", padx=(0, 6), pady=1)

            self._bench_category_state[cat] = {
                "header_btn": header_btn,
                "body": body,
                "card": card,
                "ids": cat_ids,
                "color": color,
                "select_all_cb": select_all_cb,
            }
            # Render header text and apply the collapse state.
            self._update_bench_category_header(cat)
            if cat in self._bench_collapsed_categories:
                body.grid_remove()

        # Refresh global selection footer + preset-button enable state.
        self._update_bench_selection_footer()

    def _on_bench_model_selection_changed(self, category: str) -> None:
        if self.__dict__.get("_bench_selection_sync_in_progress", False):
            return
        self._update_bench_category_header(category)
        self._update_bench_selection_footer()

    def _toggle_bench_category_selection(self, category: str) -> None:
        """Bulk-select or deselect every eligible model in a category.

        Disabled rows (missing backend / over capacity without override) stay
        unchecked. The select-all checkbox's own state is the source of truth
        for the intent: True means "select all eligible", False means "clear".
        """
        state = self._bench_category_state.get(category) if hasattr(
            self, "_bench_category_state"
        ) else None
        if not state:
            return
        var = self._bench_category_select_vars.get(category)
        if var is None:
            return
        intent_selected = bool(var.get())
        disabled_ids = getattr(self, "_bench_model_disabled_ids", set())
        # Guard so the per-row traces don't fight us mid-loop.
        self._bench_category_sync_in_progress.add(category)
        prior_sync = self.__dict__.get("_bench_selection_sync_in_progress", False)
        self._bench_selection_sync_in_progress = True
        try:
            for mid in state.get("ids", []):
                row_var = self._bench_model_vars.get(mid)
                if row_var is None:
                    continue
                if mid in disabled_ids:
                    row_var.set(False)
                    continue
                row_var.set(intent_selected)
        finally:
            self._bench_category_sync_in_progress.discard(category)
            self._bench_selection_sync_in_progress = prior_sync
        # One explicit header/footer refresh after the bulk change.
        self._update_bench_category_header(category)
        self._update_bench_selection_footer()

    def _toggle_bench_category(self, category: str) -> None:
        """Expand or collapse a single category group in the checklist."""
        state = self.__dict__.get("_bench_category_state", {}).get(category)
        if not state:
            return
        collapsed = category in self._bench_collapsed_categories
        if collapsed:
            self._bench_collapsed_categories.discard(category)
            state["body"].grid()
        else:
            self._bench_collapsed_categories.add(category)
            state["body"].grid_remove()
        self._update_bench_category_header(category)

    def _set_all_bench_categories(self, *, collapsed: bool) -> None:
        """Bulk expand or collapse every category group."""
        if not hasattr(self, "_bench_category_state"):
            return
        for category, state in self._bench_category_state.items():
            if collapsed:
                self._bench_collapsed_categories.add(category)
                state["body"].grid_remove()
            else:
                self._bench_collapsed_categories.discard(category)
                state["body"].grid()
            self._update_bench_category_header(category)

    def _update_bench_category_header(self, category: str) -> None:
        """Refresh the per-category header label (chevron, count, preview) and
        the select-all checkbox state (checked when all eligible rows are
        selected, unchecked otherwise)."""
        state = self.__dict__.get("_bench_category_state", {}).get(category)
        if not state:
            return
        ids = state.get("ids", [])
        total = len(ids)
        disabled_ids = getattr(self, "_bench_model_disabled_ids", set())
        eligible_ids = [mid for mid in ids if mid not in disabled_ids]
        selected = sum(
            1 for mid in ids
            if self._bench_model_vars.get(mid) is not None
            and self._bench_model_vars[mid].get()
        )
        collapsed = category in getattr(self, "_bench_collapsed_categories", set())
        chevron = "▸" if collapsed else "▾"
        # Build a compact preview of model names so the user can see the group
        # at a glance even when collapsed. Cap at ~70 chars to avoid wrapping.
        model_by_id = self.__dict__.get("_bench_model_by_id")
        if not model_by_id:
            model_by_id = {m["id"]: m for m in self._catalog_models if m.get("id")}
        preview_names = [
            str((model_by_id.get(mid) or {}).get("name") or mid)
            for mid in ids[:6]
        ]
        preview = ", ".join(preview_names)
        if len(ids) > 6:
            preview += ", ..."
        max_len = 80
        if len(preview) > max_len:
            preview = preview[: max_len - 1] + "…"
        # v5.5.4 product-designer (P2-2): on non-GPU profiles with Force All
        # off, every Image Generation row in the checklist is disabled with a
        # "image gen requires GPU (use Force All to override)" per-row reason,
        # but the *group* header didn't communicate the same. Sighted users
        # could see the dimmed rows; screen-reader users navigating by
        # header only heard "Image Generation, 0 of N selected". Append a
        # short "(GPU required)" marker to the header so both audiences
        # learn the gating up-front.
        gpu_required_marker = ""
        try:
            if category == "Image Generation":
                cap = self._bench_profile_capacity()
                has_gpu = bool(cap.get("has_gpu"))
                force_all_var = self.__dict__.get("_bench_force_all_var")
                force_all = bool(force_all_var is not None and force_all_var.get())
                if not has_gpu and not force_all:
                    gpu_required_marker = "   ·   (GPU required)"
        except Exception:
            gpu_required_marker = ""
        text = f"  {chevron}  {category}   ·   {selected} of {total} selected{gpu_required_marker}   ·   {preview}"
        try:
            state["header_btn"].configure(text=text)
        except Exception:
            pass
        # Sync the category select-all checkbox: True iff every eligible row
        # is selected. This update must NOT re-fire the bulk handler, so we
        # set the var while suppressing the command callback by writing to it
        # only when the desired value differs from the current value.
        try:
            select_var = getattr(self, "_bench_category_select_vars", {}).get(category)
            if select_var is not None:
                desired = bool(eligible_ids) and all(
                    self._bench_model_vars.get(mid) is not None
                    and self._bench_model_vars[mid].get()
                    for mid in eligible_ids
                )
                if select_var.get() != desired:
                    # Setting the BooleanVar updates the visual state of the
                    # CTkCheckBox without invoking its `command` callback —
                    # only user clicks do that.
                    select_var.set(desired)
        except Exception:
            pass

    def _update_bench_selection_footer(self) -> None:
        """Refresh the running selection summary below the checklist."""
        if self.__dict__.get("_bench_selection_sync_in_progress", False):
            return
        label = self.__dict__.get("_bench_selection_footer")
        if label is None:
            return
        try:
            total_models = len(self._bench_model_vars)
            selected_ids = [
                mid for mid, var in self._bench_model_vars.items() if var.get()
            ]
            selected_count = len(selected_ids)
            run_mode = self._active_run_mode()
            capacity = self._bench_profile_capacity()
            allow_oversize = bool(
                self.__dict__.get("_bench_resource_override_var") is not None
                and self._bench_resource_override_var.get()
            )
            # Quick = 1 case per (model x method); Extended chat/utility = 3
            # samples; Extended image = 3 samples. Approximate to keep the
            # footer cheap to compute on every checkbox toggle.
            id_to_model = self.__dict__.get("_bench_model_by_id") or {
                m["id"]: m for m in self._catalog_models if m.get("id")
            }
            runnable_count = 0
            cases = 0
            for mid in selected_ids:
                model = id_to_model.get(mid) or {}
                methods = self._bench_methods_for_run_ui(
                    model, capacity=capacity, allow_oversize=allow_oversize
                )
                if methods:
                    runnable_count += 1
                per_sample = 3 if run_mode == "extended" else 1
                # Utility/ASR/embedding adapters always run a single fixture
                # even in Extended mode (matches _iter_samples_for).
                if model.get("phase1_adapter"):
                    per_sample = 1
                cases += len(methods) * per_sample
            mode_label = {
                "extended": "Extended",
            }.get(run_mode, "Quick")
            text = (
                f"Selected: {selected_count} of {total_models} models  ·  "
                f"Runnable now: {runnable_count}  ·  "
                f"~{cases} cases  ·  {mode_label} mode"
            )
            label.configure(text=text)
        except Exception:
            pass

    def _bench_apply_preset(self, preset: str) -> None:
        """Apply a selection preset across the entire checklist."""
        if not hasattr(self, "_bench_model_vars"):
            return
        capacity = self._bench_profile_capacity()
        # v5.5.2: When Force-All is on, the "all" preset must bypass both the
        # disabled and oversize gates so every catalog model actually ends up
        # selected. Without this short-circuit the preset would silently drop
        # every row that would otherwise be marked unavailable for the SKU,
        # which is exactly the behaviour Force-All exists to override.
        force_all = bool(
            self.__dict__.get("_bench_force_all_var") is not None
            and self._bench_force_all_var.get()
        )
        allow_oversize = bool(
            self.__dict__.get("_bench_resource_override_var") is not None
            and self._bench_resource_override_var.get()
        )
        id_to_model = {m["id"]: m for m in self._catalog_models}
        prior_sync = self.__dict__.get("_bench_selection_sync_in_progress", False)
        self._bench_selection_sync_in_progress = True
        try:
            for mid, var in self._bench_model_vars.items():
                bypass_row_gates = (preset == "all") and force_all
                if not bypass_row_gates and mid in self._bench_model_disabled_ids:
                    var.set(False)
                    continue
                model = id_to_model.get(mid) or {}
                if preset == "none":
                    var.set(False)
                elif preset == "all":
                    if (
                        not bypass_row_gates
                        and not allow_oversize
                        and mid in self._bench_model_oversize_ids
                    ):
                        var.set(False)
                    else:
                        var.set(True)
                elif preset == "results":
                    var.set(bool(self._bench_observed_success_for_profile(model, capacity)))
                elif preset == "defaults":
                    var.set(
                        bool(self._bench_default_selected_for_model(model, capacity))
                        and mid not in self._bench_model_disabled_ids
                    )
        finally:
            self._bench_selection_sync_in_progress = prior_sync
        # BooleanVar traces are suppressed during the bulk write; refresh once.
        for category in self.__dict__.get("_bench_category_state", {}):
            self._update_bench_category_header(category)
        self._update_bench_selection_footer()

    def _build_benchmark_page(self):
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["benchmark"] = page
        # row 0 = title, row 1 = options (collapsible), row 2 = log (expands),
        # row 3 = action bar (pinned to bottom, always visible).
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)

        # Detect ONNX availability once; model defaults come from the active benchmark profile.
        _has_onnx = importlib.util.find_spec("onnxruntime") is not None
        self._bench_profile_var = ctk.StringVar(value=self._bench_default_profile_name())
        _bench_capacity = self._bench_profile_capacity()

        # Title row (row 0) — title + collapse toggle
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(12, 0))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hdr, text="Batch Benchmark",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        self._bench_opts_toggle_btn = ctk.CTkButton(
            hdr, text="▲ Hide Options", width=130, height=26,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._toggle_bench_opts,
        )
        self._bench_opts_toggle_btn.grid(row=0, column=1, sticky="e")

        # ── Options frame (row 1) ────────────────────────────────────────
        self._bench_opts_frame = ctk.CTkFrame(page)
        opts = self._bench_opts_frame
        opts.grid(row=1, column=0, sticky="ew", padx=20, pady=(8, 6))
        opts.grid_columnconfigure(1, weight=1)

        # Prompt
        ctk.CTkLabel(opts, text="Prompt:", font=ctk.CTkFont(size=12)
                     ).grid(row=0, column=0, sticky="e", padx=(12, 6), pady=4)
        self._bench_prompt_var = ctk.StringVar(value=DEFAULT_PROMPT)
        ctk.CTkEntry(opts, textvariable=self._bench_prompt_var, width=500
                     ).grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 12), pady=4)

        # Timeout + Max failures on same row
        ctk.CTkLabel(opts, text="Timeout (sec):", font=ctk.CTkFont(size=12)
                     ).grid(row=1, column=0, sticky="e", padx=(12, 6), pady=4)
        self._bench_timeout_var = ctk.IntVar(value=300)
        ctk.CTkEntry(opts, textvariable=self._bench_timeout_var, width=80
                     ).grid(row=1, column=1, sticky="w", pady=4)

        ctk.CTkLabel(opts, text="Max failures:", font=ctk.CTkFont(size=12)
                     ).grid(row=1, column=2, sticky="e", padx=(20, 6), pady=4)
        self._bench_maxfail_var = ctk.IntVar(value=10)
        ctk.CTkEntry(opts, textvariable=self._bench_maxfail_var, width=60
                     ).grid(row=1, column=3, sticky="w", padx=(0, 12), pady=4)

        ctk.CTkLabel(opts, text="Benchmark profile:", font=ctk.CTkFont(size=12)
                     ).grid(row=2, column=0, sticky="e", padx=(12, 6), pady=4)
        self._bench_profile_segbtn = ctk.CTkSegmentedButton(
            opts,
            values=self._bench_profile_values(),
            variable=self._bench_profile_var,
            command=self._on_bench_profile_changed,
        )
        self._bench_profile_segbtn.grid(row=2, column=1, columnspan=3, sticky="w", padx=(0, 12), pady=4)

        # Run mode (Quick = one shared prompt; Extended = full sample set)
        ctk.CTkLabel(opts, text="Run mode:", font=ctk.CTkFont(size=12)
                     ).grid(row=3, column=0, sticky="e", padx=(12, 6), pady=4)
        self._bench_run_mode_var = ctk.StringVar(value="Quick")
        self._bench_run_mode_segbtn = ctk.CTkSegmentedButton(
            opts,
            values=["Quick", "Extended"],
            variable=self._bench_run_mode_var,
            command=self._on_bench_run_mode_changed,
        )
        self._bench_run_mode_segbtn.grid(row=3, column=1, sticky="w", padx=(0, 12), pady=4)
        # v5.5.1 UX fix: the "Include image-gen models" checkbox is gone —
        # image-gen rows are ALWAYS rendered in the model list. Their
        # checkable state is gated by GPU presence (per-row, in
        # _bench_model_available_for_profile) and overridable via Force All.
        # We keep ``_bench_image_var`` as a True constant so legacy call sites
        # that still read it (snapshot/restore / Force-All toggle) behave as
        # if image-gen was always included.
        self._bench_image_var = ctk.BooleanVar(value=True)

        # Method toggles (row 4)
        toggles = ctk.CTkFrame(opts, fg_color="transparent")
        toggles.grid(row=4, column=0, columnspan=4, sticky="w", padx=12, pady=4)

        self._bench_gpu_var = ctk.BooleanVar(value=bool(_bench_capacity.get("has_gpu")))
        self._bench_gpu_check = ctk.CTkCheckBox(
            toggles, text="Ollama GPU", variable=self._bench_gpu_var,
            command=self._on_bench_method_toggle,
        )
        self._bench_gpu_check.grid(row=0, column=0, padx=(0, 16))
        self._bench_cpu_var = ctk.BooleanVar(value=True)
        self._bench_cpu_check = ctk.CTkCheckBox(
            toggles, text="Ollama CPU", variable=self._bench_cpu_var,
            command=self._on_bench_method_toggle,
        )
        self._bench_cpu_check.grid(row=0, column=1, padx=(0, 16))
        self._bench_onnx_var = ctk.BooleanVar(value=_has_onnx)
        self._bench_onnx_check = ctk.CTkCheckBox(
            toggles, text="ONNX (OpenVINO/DirectML + CPU)",
            variable=self._bench_onnx_var,
            command=self._on_bench_method_toggle,
        )
        self._bench_onnx_check.grid(row=0, column=2, padx=(0, 16))
        # Utility / Toolbox demos are intentionally excluded from benchmark
        # runs. Keep _bench_utility_var / _bench_phase1_var as harmless
        # back-compat False stubs so existing call sites (CLI generation,
        # start_benchmark, automation) keep working without rendering a
        # Utility checkbox the user could click.
        self._bench_utility_var = ctk.BooleanVar(value=False)
        self._bench_phase1_var = self._bench_utility_var
        self._bench_cleanup_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(toggles, text="Cleanup after each model", variable=self._bench_cleanup_var
                        ).grid(row=0, column=3, padx=(0, 12))
        # v5.5.1 UX fix: removed the "Advanced: allow oversize models"
        # checkbox. It was a confusing power-user knob whose only useful
        # behaviour (run a model the SKU can't actually fit) is already
        # covered by Force All. Force All toggles ``_bench_resource_override_var``
        # internally so the runner's allow_oversize wiring stays intact;
        # users see one clear escape hatch instead of two overlapping ones.
        self._bench_resource_override_var = ctk.BooleanVar(value=False)
        # v5.5.0 UX fix: Force All checkbox previously lived at row=0 col=5
        # with the label "Force All (best-effort baseline)" — at 1280×800
        # the checkbox sat at x=1421 (off-screen to the right past the
        # viewport). Moved to a fresh row (row=1) so the option remains
        # visible without a disclosure/popover (per product-designer +
        # a11y review: it's a primary power-user control, must not be
        # hidden behind a disclosure). The descriptive caption that used
        # to live on row=1 now sits next to the checkbox on the same row.
        self._bench_force_all_var = ctk.BooleanVar(value=False)
        self._bench_force_all_check = ctk.CTkCheckBox(
            toggles,
            # v5.5.4 a11y (P2-A): UIA exposes the checkbox label as its
            # AccessibleName. The neighbouring caption (next column) is
            # purely visual — screen readers don't link it via
            # ``aria-describedby``. So pack the most-important behaviour
            # (capacity override + the CPU image-gen enable) into the
            # label itself. Kept short enough to stay inside the 1280×800
            # viewport (the v5.5.0 fix moved this to row=1; the descriptive
            # caption still lives next to it for sighted users).
            text="Force All (ignores capacity)",
            variable=self._bench_force_all_var,
            command=self._on_bench_force_all_toggle,
        )
        self._bench_force_all_check.grid(row=1, column=0, padx=(0, 8), pady=(4, 0), sticky="w")
        # v5.5.1+ a11y: screen readers + tooltip-less CTk surfaces need the
        # behaviour to live in the visible label.  Stays a regression-critical row
        # compliant — no host SKU specs mentioned.
        ctk.CTkLabel(
            toggles,
            text="(best-effort baseline — selects every model + method, ignores capacity, keeps going on failures; enables image-gen on CPU rigs)",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=1, columnspan=5, sticky="w", padx=(0, 12), pady=(4, 0))
        self._apply_bench_profile_method_defaults()

        # ── Model selector — grouped, collapsible, with presets ────────────
        model_frame = ctk.CTkFrame(opts, fg_color="transparent")
        model_frame.grid(row=7, column=0, columnspan=4, sticky="ew", padx=12, pady=(2, 8))
        model_frame.grid_columnconfigure(5, weight=1)
        model_frame.grid_rowconfigure(1, weight=1)

        # Toolbar row 0: "Models:" + presets + All/None.
        toolbar = ctk.CTkFrame(model_frame, fg_color="transparent")
        toolbar.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 4))
        toolbar.grid_columnconfigure(7, weight=1)

        ctk.CTkLabel(toolbar, text="Models:", font=ctk.CTkFont(size=12)
                     ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ctk.CTkLabel(
            toolbar, text="Presets:", font=ctk.CTkFont(size=10), text_color=TEXT_MUTED,
        ).grid(row=0, column=1, sticky="w", padx=(0, 4))
        ctk.CTkButton(
            toolbar, text="Profile defaults", width=120, height=22,
            font=ctk.CTkFont(size=10),
            **self._outline_button_style(),
            command=lambda: self._bench_apply_preset("defaults"),
        ).grid(row=0, column=2, padx=(0, 4))
        ctk.CTkButton(
            toolbar, text="Results successes", width=130, height=22,
            font=ctk.CTkFont(size=10),
            **self._outline_button_style(),
            command=lambda: self._bench_apply_preset("results"),
        ).grid(row=0, column=3, padx=(0, 4))
        ctk.CTkButton(
            toolbar, text="All", width=44, height=22,
            font=ctk.CTkFont(size=10),
            **self._outline_button_style(),
            command=lambda: self._bench_apply_preset("all"),
        ).grid(row=0, column=4, padx=(0, 4))
        ctk.CTkButton(
            toolbar, text="None", width=50, height=22,
            font=ctk.CTkFont(size=10),
            **self._outline_button_style(),
            command=lambda: self._bench_apply_preset("none"),
        ).grid(row=0, column=5, padx=(0, 12))

        # Expand-all / Collapse-all chips on the right side of the toolbar.
        ctk.CTkButton(
            toolbar, text="▾ Expand all", width=110, height=22,
            font=ctk.CTkFont(size=10),
            **self._outline_button_style(),
            command=lambda: self._set_all_bench_categories(collapsed=False),
        ).grid(row=0, column=6, padx=(0, 4))
        ctk.CTkButton(
            toolbar, text="▸ Collapse all", width=110, height=22,
            font=ctk.CTkFont(size=10),
            **self._outline_button_style(),
            command=lambda: self._set_all_bench_categories(collapsed=True),
        ).grid(row=0, column=7, sticky="w", padx=(0, 4))

        # 3-column collapsible checklist; bounded height keeps action buttons
        # below from scrolling off-screen as the catalog grows.
        self._bench_checklist_scroll = ctk.CTkScrollableFrame(
            model_frame,
            height=360,
            fg_color="transparent",
            border_width=0,
        )
        self._bench_checklist_scroll.grid(
            row=1, column=0, columnspan=6, sticky="nsew", pady=(2, 0)
        )

        # Selection summary footer (selected count, ~cases, run mode, profile).
        self._bench_selection_footer = ctk.CTkLabel(
            model_frame,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=INFO_TEXT,
            anchor="w",
        )
        self._bench_selection_footer.grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(6, 0)
        )

        self._render_bench_model_checklist(preserve_selection=False)

        # ── Output log (row 2) ────────────────────────────────────────────
        # Log expands to fill space between the (collapsible) options and the
        # static action bar pinned to the bottom of the page.
        self._bench_log = ctk.CTkTextbox(page, font=ctk.CTkFont(family="Consolas", size=11))
        self._bench_log.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 6))
        self._bench_log.configure(state="disabled")

        # ── Static action bar (row 3) ─────────────────────────────────────
        # Pinned to the bottom so Start / Stop / Retry / Open / Copy CLI are
        # always reachable, even with Options collapsed or during a long run.
        btn_frame = ctk.CTkFrame(page, fg_color=SURFACE_INNER, corner_radius=8)
        btn_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 12))

        self._bench_start_btn = ctk.CTkButton(
            btn_frame, text="Start Benchmark", width=160,
            **self._solid_button_style("#2d8a4e", "#236b3c"),
            command=self._start_benchmark,
        )
        self._bench_start_btn.grid(row=0, column=0, padx=(10, 8), pady=8)

        self._bench_stop_btn = ctk.CTkButton(
            btn_frame, text="Stop", width=80, state="disabled",
            **self._solid_button_style(self._IG_DANGER, "#8a2424"),
            command=self._stop_benchmark,
        )
        self._bench_stop_btn.grid(row=0, column=1, padx=(0, 8), pady=8)

        self._bench_retry_btn = ctk.CTkButton(
            btn_frame, text="Retry Failed", width=120,
            **self._outline_button_style(),
            command=self._retry_failed_benchmark,
        )
        self._bench_retry_btn.grid(row=0, column=2, padx=(0, 8), pady=8)

        self._bench_open_btn = ctk.CTkButton(
            btn_frame, text="Open Report Folder", width=160,
            **self._outline_button_style(),
            command=self._open_bench_output,
        )
        self._bench_open_btn.grid(row=0, column=3, padx=(0, 8), pady=8)

        # Open the latest HTML report directly in the user's browser.
        self._bench_open_html_btn = ctk.CTkButton(
            btn_frame, text="Open HTML Report", width=160,
            **self._outline_button_style(),
            command=self._open_latest_bench_html,
        )
        self._bench_open_html_btn.grid(row=0, column=4, padx=(0, 8), pady=8)

        # Copy CLI command button
        self._bench_cli_btn = ctk.CTkButton(
            btn_frame, text="Copy CLI Command", width=160,
            **self._outline_button_style(),
            command=self._copy_bench_cli,
        )
        self._bench_cli_btn.grid(row=0, column=5, padx=(0, 10), pady=8)

        # Internal state
        self._bench_thread: Optional[threading.Thread] = None
        self._bench_stop_event = threading.Event()
        self._bench_runner: Optional[BatchRunner] = None
        self._bench_opts_collapsed: bool = False

    def _toggle_bench_opts(self):
        """User-facing show/hide toggle. Derives intent from the actual
        widget state to avoid the v5.5.0 two-click bug where the cached
        ``_bench_opts_collapsed`` flag drifted out of sync with reality
        (e.g., after a page rebuild or after ``_start_benchmark``'s
        auto-collapse mutated the frame but not the flag)."""
        currently_visible = self._bench_opts_visible()
        self._set_bench_opts_visible(not currently_visible)

    def _bench_opts_visible(self) -> bool:
        """Ground-truth visibility for the benchmark options panel.

        ``winfo_ismapped()`` returns the actual grid-mapped state, which
        is the only reliable source after page rebuilds and auto-collapse.
        Returns False when the frame doesn't exist yet (build hasn't run)
        so callers don't crash during initial wiring."""
        frame = self.__dict__.get("_bench_opts_frame")
        if frame is None:
            return False
        try:
            return bool(frame.winfo_ismapped())
        except Exception:
            return False

    def _set_bench_opts_visible(self, visible: bool) -> None:
        """Single source of truth for showing/hiding the options panel.

        Updates the frame mapping, button label, and the cached
        ``_bench_opts_collapsed`` mirror (kept for back-compat with any
        external readers). a11y review: the button label MUST flip on
        every code path that toggles visibility, including the
        auto-collapse in ``_start_benchmark`` — that's the only
        screen-reader signal Tk offers since there's no aria-expanded
        equivalent in MSAA/UIA exposure of CTkButton."""
        frame = self.__dict__.get("_bench_opts_frame")
        btn = self.__dict__.get("_bench_opts_toggle_btn")
        if frame is None:
            return
        try:
            if visible:
                frame.grid()
            else:
                frame.grid_remove()
        except Exception:
            pass
        if btn is not None:
            try:
                btn.configure(text="▲ Hide Options" if visible else "▼ Show Options")
            except Exception:
                pass
        self._bench_opts_collapsed = not visible

    def _active_run_mode(self) -> str:
        """Return the runner mode token ('quick' | 'extended') from the UI var."""
        value = self.__dict__.get("_bench_run_mode_var")
        if value is None:
            return "quick"
        try:
            label = (value.get() or "").strip().lower()
        except Exception:
            return "quick"
        if label == "extended":
            return "extended"
        return "quick"

    def _on_bench_run_mode_changed(self, _value=None):
        """Refresh defaults so the model checklist includes image-gen models
        when Extended is active on a GPU profile, and update the
        footer's estimated case count for the new mode.

        v5.5.0 UX fix: ``preserve_selection=False`` so that switching mode
        re-applies that mode's defaults. Modes are opinionated presets, not
        filters over a persistent selection — switching to Extended should
        mean Extended, switching to Quick should mean Quick. The footer count
        is the user's confirmation that "Extended mode applied — N models
        selected"; no toast/dialog needed.
        """
        try:
            self._apply_bench_profile_method_defaults()
            self._render_bench_model_checklist(preserve_selection=False)
            self._update_bench_selection_footer()
        except Exception:
            pass

    def _on_bench_method_toggle(self) -> None:
        """Cheap refresh for Benchmark method toggles.

        Rebuilding the full checklist on every CPU/GPU/ONNX/Image click makes
        the page feel frozen. The selected rows stay put; this refreshes the
        runnable-model and case counts that determine what Start will run.
        """
        self._update_bench_selection_footer()

    # v5.5.1: _on_bench_image_toggle removed — Include Image Gen checkbox gone.

    def _on_bench_force_all_toggle(self) -> None:
        """When Force-All is enabled, snapshot the current selections, then
        flip every method on, oversize on, and select every catalog model.
        When Force-All is disabled, restore the snapshot so unticking the box
        does not destroy the user's prior carefully curated model selection.

        v5.5.1 product-designer fix: previously, unticking Force-All left the
        ballooned model set in place, silently nuking whatever the user had
        selected before. That was a footgun — users investigating the new
        checkbox would tick it, see ~60 rows appear, untick to back out, and
        find their 5-model selection gone with no undo. The snapshot lives
        on ``self._bench_force_all_snapshot`` and is restored on the down-toggle.
        """
        try:
            force_all = bool(self._bench_force_all_var.get())
        except Exception:
            force_all = False
        if force_all:
            # Snapshot current state so unticking restores it.
            snapshot: dict[str, object] = {}
            # v5.5.4 (SQT P3-3): drop ``_bench_image_var`` from the snapshot
            # tuple — it has been a True-constant since v5.5.1 (the user-
            # facing "Include Image Gen" checkbox went away), so storing it
            # was a no-op round-trip that misled future readers into thinking
            # there was still a user toggle to restore. ``_bench_resource_
            # override_var`` is the lone internal-only toggle Force All
            # actually flips.
            for attr in (
                "_bench_gpu_var", "_bench_cpu_var", "_bench_onnx_var",
                "_bench_resource_override_var",
            ):
                var = self.__dict__.get(attr)
                if var is not None:
                    try:
                        snapshot[attr] = bool(var.get())
                    except Exception:
                        snapshot[attr] = None
            model_vars = self.__dict__.get("_bench_model_vars") or {}
            model_snapshot: dict[str, bool] = {}
            for mid, mvar in model_vars.items():
                try:
                    model_snapshot[mid] = bool(mvar.get())
                except Exception:
                    model_snapshot[mid] = False
            snapshot["_model_vars"] = model_snapshot
            self._bench_force_all_snapshot = snapshot

            # Quietly flip the helper toggles so the run actually exercises
            # every backend rather than honouring stale per-method skips.
            # v5.5.4: ``_bench_image_var`` is a True constant initialized at
            # _build_bench_page so we don't set it here either — Force All
            # only needs to flip the three backend toggles plus the resource
            # override below.
            for attr in ("_bench_gpu_var", "_bench_cpu_var", "_bench_onnx_var"):
                var = self.__dict__.get(attr)
                if var is not None:
                    try:
                        var.set(True)
                    except Exception:
                        pass
            override = self.__dict__.get("_bench_resource_override_var")
            if override is not None:
                try:
                    override.set(True)
                except Exception:
                    pass
            # v5.5.1: there's no longer a visible "Advanced: allow oversize"
            # checkbox to lock — Force All is the single escape hatch.
            # Select every catalog model via the existing "All" preset path
            # so the runner sees the broadest possible set. _bench_apply_preset
            # already honours the oversize override we just set above.
            try:
                self._bench_apply_preset("all")
            except Exception:
                pass
        else:
            # Restore the snapshot if present so the user gets back exactly
            # what they had before they investigated the Force-All checkbox.
            snapshot = self.__dict__.pop("_bench_force_all_snapshot", None)
            if snapshot:
                for attr in (
                    "_bench_gpu_var", "_bench_cpu_var", "_bench_onnx_var",
                    "_bench_resource_override_var",
                ):
                    var = self.__dict__.get(attr)
                    prior = snapshot.get(attr)
                    if var is not None and prior is not None:
                        try:
                            var.set(bool(prior))
                        except Exception:
                            pass
                model_snapshot = snapshot.get("_model_vars") or {}
                model_vars = self.__dict__.get("_bench_model_vars") or {}
                for mid, mvar in model_vars.items():
                    if mid in model_snapshot:
                        try:
                            mvar.set(bool(model_snapshot[mid]))
                        except Exception:
                            pass
            # v5.5.1: no oversize companion checkbox to re-enable.
        # Always refresh the runnable counts so the footer text matches the
        # new state regardless of direction.
        try:
            self._render_bench_model_checklist(preserve_selection=True)
        except Exception:
            self._update_bench_selection_footer()

    def _open_latest_bench_html(self):
        """Open the most recent benchmark HTML report in the user's browser."""
        from src.batch_report import (
            LATEST_EXTENDED_HTML_NAME,
            LATEST_QUICK_HTML_NAME,
            LEGACY_HTML_NAME,
        )
        out = Path(os.path.dirname(os.path.abspath(__file__))).parent / "benchmark_results"
        if not out.exists():
            messagebox.showinfo(
                "No reports yet",
                f"No benchmark output folder exists yet:\n{out}",
                parent=self,
            )
            return
        mode = self._active_run_mode()
        preferred_alias = {
            "extended": LATEST_EXTENDED_HTML_NAME,
        }.get(mode, LATEST_QUICK_HTML_NAME)
        candidates = [
            out / preferred_alias,
            out / LATEST_EXTENDED_HTML_NAME,
            out / LATEST_QUICK_HTML_NAME,
            out / LEGACY_HTML_NAME,
        ]
        for path in candidates:
            if path.exists():
                self._open_path(path)
                return
        # Fall back to the newest .html in the folder.
        try:
            html_files = sorted(
                out.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True,
            )
        except OSError:
            html_files = []
        if html_files:
            self._open_path(html_files[0])
            return
        messagebox.showinfo(
            "No HTML report yet",
            "No HTML benchmark report has been generated yet. Run a benchmark first.",
            parent=self,
        )

    def _bench_select_models(self, select: bool):
        disabled = getattr(self, "_bench_model_disabled_ids", set())
        oversize = getattr(self, "_bench_model_oversize_ids", set())
        prior_sync = self.__dict__.get("_bench_selection_sync_in_progress", False)
        self._bench_selection_sync_in_progress = True
        try:
            for mid, var in self._bench_model_vars.items():
                var.set(bool(select and mid not in disabled and mid not in oversize))
        finally:
            self._bench_selection_sync_in_progress = prior_sync
        for category in self.__dict__.get("_bench_category_state", {}):
            self._update_bench_category_header(category)
        self._update_bench_selection_footer()

    def _get_selected_model_ids(self) -> list[str]:
        """Return selected Benchmark-visible model IDs."""
        disabled = getattr(self, "_bench_model_disabled_ids", set())
        return [
            mid for mid, var in self._bench_model_vars.items()
            if var.get() and mid not in disabled
        ]

    def _get_runnable_selected_model_ids(
        self,
        selected_ids: list[str] | None = None,
        *,
        capacity: dict | None = None,
        allow_oversize: bool | None = None,
    ) -> list[str]:
        selected_ids = selected_ids if selected_ids is not None else self._get_selected_model_ids()
        if not selected_ids:
            return []
        capacity = capacity or self._bench_profile_capacity()
        if allow_oversize is None:
            allow_oversize = self._bench_toggle_value("_bench_resource_override_var", False)
        id_to_model = self.__dict__.get("_bench_model_by_id") or {
            m["id"]: m for m in self._catalog_models if m.get("id")
        }
        runnable: list[str] = []
        for mid in selected_ids:
            model = id_to_model.get(mid)
            if not model:
                runnable.append(mid)
                continue
            has_backend = bool(
                model.get("phase1_adapter")
                or model.get("ollama_tag")
                or model.get("onnx_repo")
                or self._is_image_model_ui(model)
            )
            if not has_backend:
                # Static tests use minimal fake catalog rows without backend
                # metadata; real Benchmark-visible rows always have one.
                runnable.append(mid)
                continue
            if self._bench_methods_for_run_ui(
                model,
                capacity=capacity,
                allow_oversize=allow_oversize,
            ):
                runnable.append(mid)
        return runnable

    def _get_bench_cli_args(self) -> str:
        """Build the equivalent CLI command from current GUI settings."""
        parts = ["python run_batch.py"]
        prompt = self._bench_prompt_var.get()
        if prompt != DEFAULT_PROMPT:
            parts.append(f'--prompt "{prompt}"')
        timeout = self._bench_timeout_var.get()
        if timeout != 300:
            parts.append(f"--timeout {timeout}")
        maxfail = self._bench_maxfail_var.get()
        if maxfail != 10:
            parts.append(f"--max-failures {maxfail}")
        if not self._bench_gpu_var.get():
            parts.append("--skip-gpu")
        if not self._bench_cpu_var.get():
            parts.append("--skip-cpu")
        if not self._bench_onnx_var.get():
            parts.append("--skip-onnx")
        low_resources = bool(self.cfg.get("low_resources_mode"))
        cleanup_for_run = self._bench_cleanup_var.get() or low_resources
        if cleanup_for_run:
            parts.append("--cleanup")
        if low_resources:
            parts.append("--low-resources")
            if not self._bench_cleanup_var.get():
                parts.append("--cleanup-downloaded-only")
        # Utility / Toolbox demos are always excluded from benchmark runs;
        # no --skip-utility flag needed (it's the default).
        if self.__dict__.get("_bench_resource_override_var") and self._bench_resource_override_var.get():
            parts.append("--allow-oversize")
        if self.__dict__.get("_bench_force_all_var") and self._bench_force_all_var.get():
            parts.append("--force-all")
        run_mode = self._active_run_mode()
        if run_mode != "quick":
            parts.append(f"--run-mode {run_mode}")
        # v5.5.1: --skip-image is no longer emitted. The "Include image-gen
        # models" checkbox is gone; image-gen rows are always considered
        # runnable in Extended when the model fits the profile (or
        # Force All is on). If the user wants no image cases, they simply
        # leave all image-gen rows unchecked in the model list — the runner
        # sees an empty image selection and runs nothing.
        selected = self._get_selected_model_ids()
        if selected:
            parts.append(f"--models {' '.join(selected)}")
        capacity = self._bench_profile_capacity()
        if capacity.get("is_sku"):
            parts.append(f"--capacity-ram-gb {capacity['total_ram_gb']:g}")
            if capacity.get("has_gpu"):
                parts.append(f"--capacity-vram-gb {capacity['vram_capacity_gb']:g}")
            else:
                parts.append("--capacity-no-gpu")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results")
        parts.append(f'--output "{output_dir}"')
        return " ".join(parts)

    def _copy_bench_cli(self):
        selected = self._get_selected_model_ids()
        if not selected:
            messagebox.showinfo("No models selected", "Select at least one model to benchmark.", parent=self)
            self.set_status("Select at least one model before copying the benchmark CLI command.")
            return
        cmd = self._get_bench_cli_args()
        self.clipboard_clear()
        self.clipboard_append(cmd)
        self.set_status(
            f"CLI command copied for {len(selected)} model(s) on {self._bench_capacity_label()}."
        )

    def _open_bench_output(self):
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results")
        out = os.path.normpath(out)
        if os.path.exists(out):
            self._open_path(Path(out))
        else:
            messagebox.showinfo("No reports yet", f"Output folder does not exist yet:\n{out}", parent=self)

    def _bench_log_append(self, text: str):
        """v5: now coalesces via a buffer. Direct callers still work."""
        self._enqueue_bench_log(text)

    def _log_benchmark_chunk(self, text: str) -> None:
        if not text:
            return
        buffered = self.__dict__.get("_bench_logger_line_buf", "") + text
        lines = buffered.splitlines(keepends=True)
        pending = ""
        for line in lines:
            if line.endswith(("\n", "\r")):
                clean = line.strip()
                if clean:
                    logger.debug(clean, category=logger.CATEGORY_BENCHMARK)
            else:
                pending = line
        self._bench_logger_line_buf = pending[-1000:]

    def _enqueue_bench_log(self, text: str) -> None:
        """
        v5 bench-log batching: accumulate stdout chunks and flush every 100 ms.
        On a 10K-line benchmark this cuts UI updates from 10K to ~100,
        saving 10-30 s of UI-thread contention.
        Safe to call from any thread.
        """
        self._log_benchmark_chunk(text)
        self._bench_log_buf.append(text)
        if not self._bench_log_flush_scheduled:
            self._bench_log_flush_scheduled = True
            self.after(100, self._flush_bench_log)

    def _flush_bench_log(self) -> None:
        """Drain the bench-log buffer to the textbox in one UI update."""
        self._bench_log_flush_scheduled = False
        if not self._bench_log_buf:
            return
        batch = "".join(self._bench_log_buf)
        self._bench_log_buf.clear()
        self._bench_log.configure(state="normal")
        self._bench_log.insert("end", batch)
        self._bench_log.see("end")
        self._bench_log.configure(state="disabled")

    def _set_comfyui_start_failure_reason(self, reason: str) -> None:
        """Thread-safe writer for ``_comfyui_last_start_failure_reason``.

        Called from any thread that touches the ComfyUI start path
        (benchmark worker, Image Gen UI thread).  First non-empty reason
        wins WITHIN a single start attempt — the caller resets the field
        via ``_reset_comfyui_start_failure_reason`` before kicking off a
        new attempt so we never leak a reason from a previous attempt.
        """
        with self._comfyui_last_start_failure_lock:
            if not self._comfyui_last_start_failure_reason and reason:
                self._comfyui_last_start_failure_reason = reason

    def _reset_comfyui_start_failure_reason(self) -> None:
        with self._comfyui_last_start_failure_lock:
            self._comfyui_last_start_failure_reason = ""

    def _get_comfyui_start_failure_reason(self) -> str:
        with self._comfyui_last_start_failure_lock:
            return self._comfyui_last_start_failure_reason

    def _bench_force_cpu_for_comfyui(self) -> Optional[bool]:
        """Return benchmark-specific CPU/GPU mode intent for ComfyUI.

        Benchmarks should follow the selected benchmark profile, not the Image
        Gen page's CPU-mode toggle. GPU benchmark profiles therefore force
        non-CPU launch flags; CPU-only benchmark profiles force `--cpu`.
        """
        try:
            capacity = self._bench_profile_capacity()
        except Exception:
            return None
        return not bool((capacity or {}).get("has_gpu"))

    def _bench_ensure_comfyui_ready(self, timeout: int = 180, model: dict | None = None):
        """Worker-thread-safe ComfyUI readiness probe for the benchmark runner.

        Called by ``BatchRunner._run_image_comfyui`` before generating an image
        and again ONCE on a crash. Mirrors the Image Gen page's start logic
        (probe → start subprocess → poll until /system_stats responds) without
        touching any Tk widgets, so it is safe to call from the benchmark
        worker thread. It leaves warm ComfyUI instances alone unless the
        benchmark model declares required launch flags (for example SDXL Low
        VRAM's ``--lowvram``) that are missing from the active process.

        v5.3.6+ (cold-start fix): returns ``(ok: bool, reason: str)``
        so the benchmark log surfaces *why* ComfyUI didn't start, not the
        generic "ComfyUI is not running and could not be started" placeholder.
        ``BatchRunner._ensure_comfyui_running_for_run`` accepts both the new
        2-tuple and the legacy ``bool`` for back-compat; we always return the
        2-tuple form.

        The default timeout was bumped 60 → 180 s because cloud VMs (roaming
        profile container + Defender real-time scan + vGPU partition + cold
        torch / CUDA init) routinely need >60 s for the first /system_stats
        response.  Warm runs short-circuit on the initial ``is_running()``
        probe and pay zero of this extra budget.
        """
        try:
            client = getattr(self, "comfyui", None)
            if client is None:
                return False, "No ComfyUI client on app (image-gen disabled)"
            bench_force_cpu = self._bench_force_cpu_for_comfyui()
            try:
                is_running = client.is_running()
            except Exception:
                is_running = False
            if is_running:
                if not self._image_model_launch_flags_need_restart(
                    model,
                    force_cpu_override=bench_force_cpu,
                ):
                    return True, ""
                mode_label = (
                    "CPU mode"
                    if bench_force_cpu is True
                    else "GPU/auto mode"
                    if bench_force_cpu is False
                    else "current mode"
                )
                logger.info(
                    "ComfyUI: restarting for benchmark launch flags "
                    f"{self._comfyui_model_launch_flags(model)} ({mode_label})",
                    category=logger.CATEGORY_COMFYUI,
                )
                self._stop_comfyui_for_restart(
                    reason="benchmark launch flag change", kill_orphans=True
                )

            # Reset and call into the start path; the start path stamps
            # `_comfyui_last_start_failure_reason` with a specific reason on
            # any failure so we can surface it instead of "could not be started".
            def _start_for_benchmark() -> bool:
                if bench_force_cpu is None:
                    return self._start_comfyui_process(model)
                return self._start_comfyui_process(
                    model,
                    force_cpu_override=bench_force_cpu,
                )

            self._reset_comfyui_start_failure_reason()
            try:
                started = _start_for_benchmark()
            except Exception as exc:
                reason = f"_start_comfyui_process raised: {exc}"
                logger.warning(reason)
                return False, reason
            if not started:
                reason = (
                    self._get_comfyui_start_failure_reason()
                    or "ComfyUI subprocess failed to start (no specific reason recorded)"
                )
                return False, reason

            poll_start = time.perf_counter()
            interval = 2.0
            deadline = max(5, int(timeout))
            log_file = Path(__file__).parent.parent / "comfyui.log"
            restart_attempted = False
            while True:
                elapsed = time.perf_counter() - poll_start
                if elapsed >= deadline:
                    break
                # Early-death detector: if the subprocess exited before
                # /system_stats came up, bail with the exit code + log tail
                # instead of waiting out the full deadline.  This collapses
                # what used to be a silent 180 s wait into an immediate
                # actionable error (e.g. CUDA OOM, missing DLL).
                proc = getattr(self, "comfyui_process", None)
                if proc is not None:
                    try:
                        exit_code = proc.poll()
                    except (OSError, AttributeError, ValueError) as poll_exc:
                        # poll() is documented to raise only when the proc
                        # handle is invalid; treat that as "exit code unknown"
                        # but log so we don't silently swallow a real bug.
                        logger.warning(f"ComfyUI proc.poll() raised: {poll_exc}")
                        exit_code = None
                    if exit_code is not None:
                        tail = self._tail_comfyui_log(log_file, lines=40)
                        signal_line = self._comfyui_startup_signal_line(tail)
                        reason = (
                            f"ComfyUI subprocess (PID {proc.pid}) exited with "
                            f"code {exit_code} during startup — startup signal: "
                            f"{signal_line}"
                        )
                        logger.error(
                            f"{reason}\n--- last 40 lines of {log_file} ---\n{tail}"
                        )
                        if not restart_attempted:
                            restart_attempted = True
                            logger.warning(
                                "ComfyUI exited during benchmark startup; "
                                "retrying once before failing"
                            )
                            self._stop_comfyui_for_restart(
                                reason="benchmark startup early exit",
                                kill_orphans=True,
                            )
                            self._reset_comfyui_start_failure_reason()
                            try:
                                restarted = _start_for_benchmark()
                            except Exception as restart_exc:
                                return (
                                    False,
                                    f"{reason}; restart attempt raised: {restart_exc}",
                                )
                            if not restarted:
                                start_reason = (
                                    self._get_comfyui_start_failure_reason()
                                    or "ComfyUI subprocess failed to restart (no specific reason recorded)"
                                )
                                return (
                                    False,
                                    f"{reason}; restart attempt failed: {start_reason}",
                                )
                            poll_start = time.perf_counter()
                            continue
                        return False, reason
                try:
                    if client.is_running():
                        self.comfyui_ok = True
                        return True, ""
                except Exception:
                    pass
                time.sleep(interval)

            # Polling timeout — gather all the diagnostics we can so the
            # user has something to ship to support.
            proc = getattr(self, "comfyui_process", None)
            proc_alive = bool(proc and proc.poll() is None)
            exit_code = proc.poll() if proc else None
            tail = self._tail_comfyui_log(log_file, lines=40)
            elapsed_s = int(time.perf_counter() - poll_start)
            reason = (
                f"ComfyUI subprocess started but didn't respond on "
                f"/system_stats within {elapsed_s}s — process alive: "
                f"{proc_alive}, exit code: {exit_code}; see comfyui.log"
            )
            logger.error(
                f"{reason}\n--- last 40 lines of {log_file} ---\n{tail}"
            )
            return False, reason
        except Exception as exc:
            reason = f"_bench_ensure_comfyui_ready unexpected failure: {exc}"
            logger.warning(reason)
            return False, reason

    def _tail_comfyui_log(self, log_file: Path, lines: int = 40) -> str:
        """Return the last ``lines`` lines of ``comfyui.log`` for diagnostics.

        Safe-by-design: any read error (missing file, permission denied,
        log handle still held open by Popen) returns an empty string so
        the caller's error-building code keeps working.  Streams via a
        ``deque(maxlen=lines)`` so a long-running benchmark with a 100 MB
        ``comfyui.log`` does not blow up RAM when we tail it.
        """
        try:
            if not log_file.exists():
                return ""
            from collections import deque
            with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
                tail = deque(fh, maxlen=lines)
            return "".join(tail)
        except Exception as exc:
            logger.debug(f"_tail_comfyui_log({log_file}) suppressed: {exc}")
            return ""

    def _comfyui_startup_signal_line(self, tail: str) -> str:
        """Pick the most actionable line from a ComfyUI startup log tail."""
        lines = [str(line).strip() for line in str(tail or "").splitlines() if str(line).strip()]
        if not lines:
            return "(no log output)"

        # First pass: explicit error-style signatures.
        strong_markers = (
            "traceback",
            "error",
            "exception",
            "failed",
            "importerror",
            "modulenotfounderror",
            "runtimeerror",
            "valueerror",
            "typeerror",
            "oserror",
            "assertionerror",
            "not allowed with argument",
            "no module named",
            "dll load failed",
            "address already in use",
            "permission denied",
        )
        for line in reversed(lines):
            lower = line.lower()
            if any(marker in lower for marker in strong_markers):
                return line

        # Second pass: ignore known startup-noise lines.
        noisy_prefixes = (
            "warning: you need pytorch",
            "found comfy_kitchen backend",
            "checkpoint files will always be loaded safely",
            "total vram",
            "pytorch version:",
            "set vram state to:",
            "device:",
            "using async weight offloading",
            "enabled pinned memory",
            "using pytorch attention",
            "python version:",
            "comfyui version:",
            "comfyui frontend version:",
            "[prompt server] web root:",
            "comfyui-gguf:",
            "import times for custom nodes:",
            "context impl",
            "will assume non-transactional ddl.",
            "assets scan(",
            "starting server",
            "to see the gui go to:",
        )
        for line in reversed(lines):
            lower = line.lower()
            if not any(lower.startswith(prefix) for prefix in noisy_prefixes):
                return line

        return lines[-1]

    def _bench_recent_partial_run(
        self,
        output_dir: Path,
        window_hours: float = 12.0,
    ) -> Optional[BatchReport]:
        """Find the most recent benchmark report within ``window_hours``.

        v5.5.6+ — used by the Resume Today's Run flow on Start Benchmark.
        Returns the most recent report whose ``start_time`` falls inside a
        rolling window (default 12h per Ron's spec — handles overnight
        managed-VM reboots without bringing back stale week-old reports).
        Returns ``None`` when no eligible report exists.
        """
        try:
            json_path = find_latest_report_json(output_dir)
        except Exception:
            json_path = None
        if not json_path or not json_path.exists():
            return None
        try:
            report = BatchReport.load_json(json_path)
        except Exception:
            return None
        # Parse start_time (ISO 8601 like "2026-05-23T13:42:07") and gate
        # by the rolling window.  Local naive datetime is fine — both
        # sides come from the same machine's local clock.
        start_text = (report.start_time or "").strip()
        if not start_text:
            return None
        parsed: Optional[datetime] = None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(start_text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            try:
                parsed = datetime.fromisoformat(start_text)
            except ValueError:
                return None
        cutoff = datetime.now() - timedelta(hours=float(window_hours))
        if parsed < cutoff:
            return None
        return report

    def _start_benchmark(self):
        # v5.5.0 UX fix: immediate-disable + bench-log status so the user gets
        # feedback BEFORE the sync pre-flight (selection, capacity, oversize
        # confirmation, low-resources disk-space prompt). Reset on every
        # early-return validation path AFTER any blocking messagebox returns.
        # Access via __dict__.get so unit tests that bypass __init__ via
        # object.__new__(App) don't trip CTk's __getattr__ recursion when
        # the button hasn't been built yet.
        self._immediate_disable_btn(
            self.__dict__.get("_bench_start_btn"),
            text="Starting…",
            status_setter=self.__dict__.get("_enqueue_bench_log"),
            status_text="Starting benchmark…\n",
        )
        if self._bench_thread and self._bench_thread.is_alive():
            messagebox.showinfo("Running", "A benchmark is already in progress.", parent=self)
            self._reset_bench_start_btn()
            return

        # Get selected models from checklist
        selected_model_ids = self._get_selected_model_ids()
        if len(selected_model_ids) == 0:
            messagebox.showinfo("No models selected", "Select at least one model to benchmark.", parent=self)
            self._reset_bench_start_btn()
            return

        output_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "benchmark_results"
        capacity = self._bench_profile_capacity()

        # v5.5.6+ Resume Today's Run: if a benchmark report from the last
        # 12 hours exists with the same run mode, offer to skip the
        # already-completed combos instead of redoing them.  Resume lives
        # on Start Benchmark (NOT Retry — Retry has a fixed "rerun the
        # failures" meaning that we shouldn't redefine).  This is a 3-way
        # prompt (Resume / Start Fresh / Cancel); declining or cancelling
        # falls through to the normal Start Benchmark path.
        resume_from_report: Optional[BatchReport] = None
        resume_skip_combos: set[tuple[str, str, int]] = set()
        try:
            recent = self._bench_recent_partial_run(output_dir, window_hours=12.0)
        except Exception:
            recent = None
        if recent is not None and len(recent.results) > 0:
            current_mode = self._active_run_mode()
            if (recent.run_mode or "").lower() == (current_mode or "").lower():
                completed = recent.get_completed_combos()
                failed_combos = recent.get_failed_combos()
                done_n = len(completed)
                failed_n = len(failed_combos)
                if done_n or failed_n:
                    when = recent.start_time or "earlier today"
                    if failed_n:
                        retry_line = (
                            f"  • {failed_n} test(s) failed and will be re-run on Resume\n"
                        )
                    else:
                        retry_line = ""
                    choice = messagebox.askyesnocancel(
                        "Resume today's benchmark run?",
                        (
                            f"Found an unfinished benchmark from {when} "
                            f"({recent.run_mode} mode):\n"
                            f"  • {done_n} test(s) already passed\n"
                            f"{retry_line}"
                            "\n"
                            "Yes  (Resume): skip what already passed; re-run any failures plus anything not yet attempted\n"
                            "No   (Start fresh): archive that run and start over\n"
                            "Cancel: don't start anything"
                        ),
                        parent=self,
                    )
                    if choice is None:
                        self._reset_bench_start_btn()
                        return
                    if choice is True:
                        resume_from_report = recent
                        resume_skip_combos = completed

        allow_oversize = bool(
            hasattr(self, "_bench_resource_override_var")
            and self._bench_resource_override_var.get()
        )
        # v5.5.4: Read force_all NOW (was read at line ~9088 just before
        # BatchRunner) so the new pre-flight impossible-models dialog can
        # branch on it. Force-All implies allow_oversize at the runner
        # layer (`BatchRunner.__init__: self.allow_oversize = bool(...)
        # or self.force_all`), so `_get_runnable_selected_model_ids` needs
        # to know not to over-filter when Force-All is on.
        force_all = bool(
            self.__dict__.get("_bench_force_all_var")
            and self._bench_force_all_var.get()
        )
        model_ids = self._get_runnable_selected_model_ids(
            selected_model_ids,
            capacity=capacity,
            allow_oversize=allow_oversize or force_all,
        )
        if len(model_ids) == 0:
            messagebox.showinfo(
                "No runnable benchmark cases",
                "Enable at least one benchmark method that applies to the selected models.",
                parent=self,
            )
            self._reset_bench_start_btn()
            return

        # v5.5.4: Force-All pre-flight — classify selected models into
        # "absolutely impossible to run on this host" (download won't fit,
        # or model needs more than 2x total RAM). Offer the user three
        # options instead of letting those models burn pull-time / RAM
        # only to fail. The "Yes — uncheck impossible" path is the
        # recommended default so dismissing the dialog leaves the maximum
        # runnable set selected. Things that "might work" under Force-All
        # (image-gen on CPU, slightly-over-the-line RAM) are intentionally
        # NOT classified as impossible — that's the whole point of Force
        # All. See ``_bench_preflight_classify_impossible``.
        if force_all:
            impossible = self._bench_preflight_classify_impossible(model_ids)
            if impossible:
                preview_lines = [
                    f"  - {item['name']}: {item['reason']}"
                    for item in impossible[:8]
                ]
                if len(impossible) > 8:
                    preview_lines.append(f"  - ... and {len(impossible) - 8} more")
                choice = messagebox.askyesnocancel(
                    "Force All — some selections won't fit",
                    (
                        f"Force All is on, but {len(impossible)} of your selected "
                        "model(s) appear physically impossible on this hardware "
                        "(needs >2x your RAM, or download exceeds free disk):\n\n"
                        + "\n".join(preview_lines)
                        + "\n\nYes  (recommended): Uncheck these and run the rest"
                        + "\nNo:                  Run all anyway (impossible ones will fail fast)"
                        + "\nCancel:              Don't start the benchmark"
                    ),
                    parent=self,
                )
                if choice is None:
                    self._reset_bench_start_btn()
                    return
                if choice is True:
                    impossible_ids = {item["id"] for item in impossible}
                    bench_vars = self.__dict__.get("_bench_model_vars") or {}
                    for mid in impossible_ids:
                        var = bench_vars.get(mid)
                        if var is not None:
                            try:
                                var.set(False)
                            except Exception:
                                pass
                    model_ids = [mid for mid in model_ids if mid not in impossible_ids]
                    if len(model_ids) == 0:
                        messagebox.showinfo(
                            "Nothing left to run",
                            "After unchecking the impossible models, no runnable "
                            "models remain. Select smaller models or run on a "
                            "larger profile.",
                            parent=self,
                        )
                        self._reset_bench_start_btn()
                        return

        if allow_oversize:
            advanced_details = []
            for model in self._catalog_models:
                if model.get("id") not in model_ids:
                    continue
                fit_ok, reason = self._bench_default_fit_for_model(
                    model,
                    available_ram_gb=capacity["available_ram_gb"],
                    total_ram_gb=capacity["total_ram_gb"],
                    vram_capacity_gb=capacity["vram_capacity_gb"],
                    has_gpu=capacity["has_gpu"],
                )
                if not fit_ok:
                    advanced_details.append((
                        str(model.get("name") or model.get("id") or "Unknown model"),
                        str(model.get("id") or ""),
                        reason,
                    ))
            if advanced_details:
                preview_lines = [
                    f"  - {name} ({mid}): {reason}" if mid else f"  - {name}: {reason}"
                    for name, mid, reason in advanced_details[:8]
                ]
                if len(advanced_details) > 8:
                    preview_lines.append(f"  - ... and {len(advanced_details) - 8} more")
                proceed = messagebox.askyesno(
                    "Run oversized benchmark models?",
                    (
                        f"{len(advanced_details)} selected model(s) exceed "
                        f"{self._bench_capacity_label(capacity)}:\n\n"
                        + "\n".join(preview_lines)
                        + "\n\nThese may download large files, run for hours, or fail on this profile.\n\nContinue?"
                    ),
                    parent=self,
                )
                if not proceed:
                    self._reset_bench_start_btn()
                    return

        low_resources = bool(self.cfg.get("low_resources_mode"))
        cleanup_for_run = self._bench_cleanup_var.get() or low_resources
        cleanup_downloaded_only = low_resources and not self._bench_cleanup_var.get()

        # Low Resources Mode: pre-batch space assessment. When cleanup is active,
        # assess peak rolling download size instead of all missing downloads at once.
        if low_resources and model_ids:
            uses_ollama = self._bench_gpu_var.get() or self._bench_cpu_var.get()
            batch_models = [
                m for m in self._catalog_models
                if m["id"] in model_ids and uses_ollama and m.get("ollama_tag")
            ]
            assessment = resource_manager.assess_batch_space(
                batch_models,
                self.ollama,
                self.cfg.get("models_dir", "."),
                cleanup_after_each_model=cleanup_for_run,
            )
            if not assessment["ok"]:
                required_gb = assessment.get("required_with_headroom_gb", assessment["needed_gb"])
                footprint_note = (
                    f"The selected run needs ~{assessment['needed_gb']:.1f} GB of total downloads,\n"
                    f"but only ~{required_gb:.1f} GB free at one time with rolling cleanup.\n"
                    if cleanup_for_run and assessment.get("needed_gb", 0) > assessment.get("required_gb", 0)
                    else f"The batch run needs ~{assessment['needed_gb']:.1f} GB of downloads.\n"
                )
                if not assessment["possible"]:
                    messagebox.showerror(
                        "Not Enough Disk Space",
                        footprint_note +
                        f"You have {assessment['free_gb']:.1f} GB free.\n\n"
                        f"Even after deleting all removable models "
                        f"({assessment['deletable_gb']:.1f} GB), there still\n"
                        f"isn't enough space. Free up disk and try again.",
                        parent=self,
                    )
                    self._reset_bench_start_btn()
                    return
                else:
                    proceed = messagebox.askyesno(
                        "Batch Run Requires Space Cleanup",
                        footprint_note +
                        f"You have {assessment['free_gb']:.1f} GB free "
                        f"(shortfall: {assessment['shortfall_gb']:.1f} GB).\n\n"
                        f"To make room, LocalAI may need to delete previously\n"
                        f"downloaded models during the run.\n\n"
                        f"Deletion priority:\n"
                        f"  1. Empty Trash / purge disk caches\n"
                        f"  2. Vision models (largest first)\n"
                        f"  3. Image generation models (largest first)\n"
                        f"  4. Chat models not in this batch (largest first)\n"
                        f"  5. Newly downloaded benchmark models after each run\n\n"
                        f"Models will only be deleted as needed.\n"
                        f"Deleted models can be re-downloaded later.\n\n"
                        f"Continue?",
                        parent=self,
                    )
                    if not proceed:
                        self._reset_bench_start_btn()
                        return

        self._bench_stop_event.clear()
        self._bench_retry_mode = False
        self._bench_log.configure(state="normal")
        self._bench_log.delete("1.0", "end")
        self._bench_log.configure(state="disabled")
        self._bench_log_append(
            f"Benchmark profile: {self._bench_capacity_label(capacity)}\n"
            f"Selected models: {len(model_ids)}\n\n"
        )
        logger.info(
            f"Benchmark starting: profile={capacity['profile']}, "
            f"selected_models={len(model_ids)}, "
            f"ram_gb={capacity['total_ram_gb']:g}, "
            f"vram_gb={capacity['vram_capacity_gb']:g}, "
            f"has_gpu={capacity['has_gpu']}",
            category=logger.CATEGORY_BENCHMARK,
        )

        # Collapse options to maximise log space. Route via
        # ``_set_bench_opts_visible(False)`` so the button label flips even
        # on auto-collapse (a11y review: every code path that mutates the
        # panel visibility must update the SR-visible label).
        if self._bench_opts_visible():
            self._set_bench_opts_visible(False)

        self._bench_start_btn.configure(state="disabled")
        self._bench_stop_btn.configure(state="normal")
        self._bench_retry_btn.configure(state="disabled")
        self.set_status("Benchmark running ...")

        force_all = bool(
            self.__dict__.get("_bench_force_all_var")
            and self._bench_force_all_var.get()
        )

        runner = BatchRunner(
            prompt=self._bench_prompt_var.get(),
            timeout=self._bench_timeout_var.get(),
            cleanup=cleanup_for_run,
            model_ids=model_ids,
            skip_gpu=not self._bench_gpu_var.get(),
            skip_cpu=not self._bench_cpu_var.get(),
            skip_onnx=not self._bench_onnx_var.get(),
            skip_phase1=True,
            max_failures=self._bench_maxfail_var.get(),
            output_dir=output_dir,
            models_dir=Path(self.cfg.get("models_dir", "models")),
            low_resources_mode=self.cfg.get("low_resources_mode", False),
            comfyui_client=self.comfyui,
            ensure_comfyui_ready=self._bench_ensure_comfyui_ready,
            prepare_image_model=self._bench_prepare_image_model,
            cleanup_downloaded_only=cleanup_downloaded_only,
            capacity_ram_gb=capacity["total_ram_gb"],
            capacity_vram_gb=capacity["vram_capacity_gb"],
            capacity_has_gpu=capacity["has_gpu"],
            allow_oversize=allow_oversize,
            force_all=force_all,
            run_mode=self._active_run_mode(),
            # v5.5.3 (SQT P1): _bench_image_var is a True-constant since
            # the "Include Image Gen" checkbox was removed in v5.5.1.
            # Per-row selection is the single source of truth — image-gen
            # is never globally skipped. The prior expression silently
            # inverted to ``skip_image=True`` when the var hadn't been
            # built yet (headless / pre-bench-page-init), suppressing
            # image-gen exactly when we wanted it visible.
            skip_image=False,
            # v5.5.6+ Resume Today's Run: when the user accepted the
            # resume prompt, hand the prior report's completed combos to
            # the runner so they're silently filtered out of the planned
            # run.  No-op when not resuming.
            skip_combos=resume_skip_combos,
            # Reuse the prior report's file_stem so the resumed run
            # overwrites the same JSON/HTML pair (no orphan reports
            # cluttering benchmark_results/).
            report_file_stem=(resume_from_report.file_stem if resume_from_report else None),
        )
        self._bench_runner = runner
        if resume_from_report is not None:
            self._bench_log_append(
                f"Resuming today's run — skipping {len(resume_skip_combos)} already-completed combo(s).\n"
            )
        self._bench_log_append(
            f"Report file stem: {runner.report.file_stem}\n\n"
        )

        # Redirect print output to the GUI log
        import io
        import contextlib

        def _run():
            log_buffer = io.StringIO()
            try:
                # Monkey-patch print for this thread by capturing stdout
                old_stdout = sys.stdout
                sys.stdout = _ThreadWriter(old_stdout, self._enqueue_bench_log)
                report = runner.run()
                sys.stdout = old_stdout
            except Exception as e:
                self._enqueue_bench_log(f"\nError: {e}\n")
            finally:
                sys.stdout = sys.__stdout__
                # v5.5.6+ Resume Today's Run: append new results back into
                # the prior report and overwrite the JSON/HTML on disk so
                # the report shows the full picture (prior + just-finished
                # combos).  Wrapped in try/except so a merge failure never
                # masks the just-finished work — the runner's own JSON is
                # always saved incrementally during the run regardless.
                if resume_from_report is not None:
                    try:
                        resume_from_report.append_resume_results(runner.report)
                        resume_from_report.save_json(output_dir)
                        resume_from_report.save_html(output_dir)
                    except Exception as merge_err:
                        self._enqueue_bench_log(
                            f"\nResume merge warning: {merge_err}\n"
                        )
                self.after(0, self._benchmark_done)

        self._bench_thread = threading.Thread(target=_run, daemon=True)
        self._bench_thread.start()

    def _stop_benchmark(self):
        if self._bench_runner:
            self._bench_runner.request_stop()
            if self._bench_runner.specific_combos is not None:
                self._bench_log_append(
                    "\n--- Stop requested for retry; original report will be merged when retry exits ---\n"
                )
                self._bench_stop_btn.configure(state="disabled")
                self.set_status("Stopping benchmark retry and preserving original report ...")
                return
            try:
                json_path, html_path = self._bench_runner.save_partial()
                if json_path and html_path:
                    self._bench_log_append(
                        f"\n--- Stop requested; partial results saved ---\n"
                        f"JSON: {json_path}\nHTML: {html_path}\n"
                    )
                    self._bench_retry_btn.configure(state="normal")
                else:
                    self._bench_log_append("\n--- Stop requested; no completed runs to save yet ---\n")
            except Exception as exc:
                self._bench_log_append(f"\n--- Stop requested; partial save failed: {exc} ---\n")
        self._bench_stop_btn.configure(state="disabled")
        self.set_status("Stopping benchmark and saving partial results ...")

    def _benchmark_done(self):
        # v5.5.0 UX fix: restore the original button labels alongside state.
        # _immediate_disable_btn may have flipped Start to "Starting…"; if
        # we only reset state="normal" the label stays stuck and confuses
        # the next click.
        self._bench_start_btn.configure(state="normal", text="Start Benchmark")
        self._bench_stop_btn.configure(state="disabled")
        self._bench_retry_btn.configure(state="normal", text="Retry Failed")
        stopped = bool(self._bench_runner and self._bench_runner._interrupted)
        retry_run = bool(self._bench_runner and self._bench_runner.specific_combos is not None)
        if stopped:
            if retry_run:
                self.set_status("Benchmark retry stopped. Original report preserved; collected retry results merged.")
                self._bench_log_append("\n--- Benchmark retry stopped; original report preserved ---\n")
            else:
                self.set_status("Benchmark stopped. Partial results saved; retry is available for failures.")
                self._bench_log_append("\n--- Benchmark stopped; partial report saved ---\n")
        else:
            results = self._bench_runner.report.results if self._bench_runner else []
            failed = [r for r in results if not r.success] if results else []
            failed_count = len(failed)
            stem = self._bench_runner.report.file_stem if self._bench_runner else ""
            if failed_count > 0 and stem:
                # v5.5.1+: surface the partial-failure diagnostic sidecars so
                # users don't have to grep the activity log to know they
                # exist.  Per product-designer review — these files are the
                # whole point of the new Force-All + smart-skip work.
                self.set_status(
                    f"Benchmark complete with {failed_count} failure(s). "
                    "Diagnostics saved next to the HTML report."
                )
                self._bench_log_append(
                    f"\n--- Benchmark finished with {failed_count} failure(s) ---\n"
                    f"JSON/HTML stem: {stem}\n"
                    f"Failure diagnostics written: {stem}_failures.txt, "
                    f"{stem}_env.txt, {stem}_run.log\n"
                    "Use 'Open Report Folder' to read them, "
                    "or 'Open HTML Report' for the themed view.\n"
                )
            else:
                self.set_status("Benchmark complete. Open the HTML report or check the report folder.")
                if results and stem:
                    self._bench_log_append(
                        "\n--- Benchmark finished ---\n"
                        f"JSON/HTML stem: {stem}\n"
                        "Use 'Open HTML Report' for the themed results page.\n"
                    )
                else:
                    self._bench_log_append("\n--- Benchmark finished ---\n")
        self._refresh_model_cards()

    def _retry_failed_benchmark(self):
        # v5.5.0 UX fix: immediate-disable + bench-log status so the user gets
        # feedback BEFORE the sync pre-flight (report load, failure scan).
        # Reset on every early-return validation path AFTER any blocking
        # messagebox returns.
        self._immediate_disable_btn(
            self.__dict__.get("_bench_retry_btn"),
            text="Starting…",
            status_setter=self.__dict__.get("_enqueue_bench_log"),
            status_text="Starting retry…\n",
        )
        if self._bench_thread and self._bench_thread.is_alive():
            messagebox.showinfo("Running", "A benchmark is already in progress.", parent=self)
            self._reset_bench_retry_btn()
            return

        output_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "benchmark_results"
        json_path = find_latest_report_json(output_dir)

        if json_path is None or not json_path.exists():
            messagebox.showinfo("No previous run", f"No benchmark results found in:\n{output_dir}", parent=self)
            self._reset_bench_retry_btn()
            return

        try:
            prev_report = BatchReport.load_json(json_path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load previous results:\n{e}", parent=self)
            self._reset_bench_retry_btn()
            return

        failed = prev_report.get_failed_combos()
        if not failed:
            messagebox.showinfo("No failures", "All tests in the previous run passed. Nothing to retry.", parent=self)
            self._reset_bench_retry_btn()
            return
        capacity = self._bench_profile_capacity()
        allow_oversize = bool(
            hasattr(self, "_bench_resource_override_var")
            and self._bench_resource_override_var.get()
        )
        force_all = bool(
            self.__dict__.get("_bench_force_all_var")
            and self._bench_force_all_var.get()
        )

        # Set up UI
        self._bench_stop_event.clear()
        self._bench_retry_mode = True
        self._bench_log.configure(state="normal")
        self._bench_log.delete("1.0", "end")
        self._bench_log.configure(state="disabled")

        self._bench_start_btn.configure(state="disabled")
        self._bench_stop_btn.configure(state="normal")
        self._bench_retry_btn.configure(state="disabled")

        self._bench_log_append(
            f"Benchmark profile: {self._bench_capacity_label(capacity)}\n"
            f"Retrying {len(failed)} failed test(s) from:\n{json_path}\n"
        )
        for mid, method in failed:
            self._bench_log_append(f"  {mid} / {method}\n")
        self._bench_log_append("\n")
        self.set_status(f"Retrying {len(failed)} failed benchmark(s) ...")

        runner = BatchRunner(
            prompt=self._bench_prompt_var.get(),
            timeout=self._bench_timeout_var.get(),
            cleanup=self._bench_cleanup_var.get() or bool(self.cfg.get("low_resources_mode")),
            max_failures=self._bench_maxfail_var.get(),
            output_dir=output_dir,
            models_dir=Path(self.cfg.get("models_dir", "models")),
            specific_combos=failed,
            low_resources_mode=self.cfg.get("low_resources_mode", False),
            cleanup_downloaded_only=bool(self.cfg.get("low_resources_mode")) and not self._bench_cleanup_var.get(),
            prepare_image_model=self._bench_prepare_image_model,
            capacity_ram_gb=capacity["total_ram_gb"],
            capacity_vram_gb=capacity["vram_capacity_gb"],
            capacity_has_gpu=capacity["has_gpu"],
            allow_oversize=allow_oversize,
            force_all=force_all,
            report_file_stem=prev_report.file_stem,
            run_mode=prev_report.run_mode or self._active_run_mode(),
            # v5.5.3 (SQT P1): see _start_benchmark — image-gen is never
            # globally skipped; per-row selection is the gate.
            skip_image=False,
            comfyui_client=self.comfyui,
            ensure_comfyui_ready=self._bench_ensure_comfyui_ready,
            skip_phase1=True,
        )
        self._bench_runner = runner

        def _run():
            try:
                old_stdout = sys.stdout
                sys.stdout = _ThreadWriter(old_stdout, self._enqueue_bench_log)
                runner.run()
                sys.stdout = old_stdout
            except Exception as e:
                self._enqueue_bench_log(f"\nError: {e}\n")
            finally:
                sys.stdout = sys.__stdout__
                # Always merge whatever results were collected (even partial)
                # with the original report and save
                try:
                    prev_report.merge(runner.report)
                    json_out = prev_report.save_json(output_dir)
                    html_out = prev_report.save_html(output_dir)
                    sys.stdout = _ThreadWriter(sys.__stdout__, self._enqueue_bench_log)
                    print("\n--- MERGED RESULTS ---")
                    prev_report.print_summary()
                    print("Merged reports saved to:")
                    print(f"  JSON: {json_out}")
                    print(f"  HTML: {html_out}")
                    sys.stdout = sys.__stdout__
                except Exception as e:
                    self._enqueue_bench_log(f"\nMerge error: {e}\n")
                self.after(0, self._benchmark_done)

        self._bench_thread = threading.Thread(target=_run, daemon=True)
        self._bench_thread.start()

    def _build_settings_page(self):
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["settings"] = page
        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(page, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="Configure connectivity, storage, behavior, and maintenance tools.",
            text_color=TEXT_MUTED,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        def make_card(row_index: int, title: str, subtitle: str = ""):
            card = ctk.CTkFrame(
                scroll,
                corner_radius=14,
                fg_color=SURFACE_CARD,
                border_width=1,
                border_color=BORDER_STRONG,
            )
            card.grid(row=row_index, column=0, sticky="ew", padx=20, pady=(0, 12))
            card.grid_columnconfigure(0, weight=1)

            card_header = ctk.CTkFrame(card, fg_color="transparent")
            card_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
            card_header.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                card_header,
                text=title,
                font=ctk.CTkFont(size=15, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
            if subtitle:
                ctk.CTkLabel(
                    card_header,
                    text=subtitle,
                    font=ctk.CTkFont(size=11),
                    text_color=TEXT_MUTED,
                    anchor="w",
                    wraplength=760,
                ).grid(row=1, column=0, sticky="w", pady=(2, 0))

            card_body = ctk.CTkFrame(card, fg_color="transparent")
            card_body.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
            card_body.grid_columnconfigure(0, weight=1)
            return card_body

        def setting_row(
            parent,
            row_index: int,
            label: str,
            widget_fn,
            *,
            help_text: str | None = None,
            help_title: str | None = None,
            hint_text: str | None = None,
        ):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.grid(row=row_index, column=0, sticky="ew", pady=(0, 8))
            row.grid_columnconfigure(0, minsize=220)
            row.grid_columnconfigure(1, weight=1)

            label_cell = ctk.CTkFrame(row, fg_color="transparent")
            label_cell.grid(row=0, column=0, sticky="ne", padx=(0, 12))
            label_cell.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                label_cell,
                text=label,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="e",
            ).grid(row=0, column=0, sticky="e")
            if help_text:
                help_btn = ctk.CTkButton(
                    label_cell,
                    text="?",
                    width=22,
                    height=22,
                    **self._outline_button_style(corner_radius=11, text_color=TEXT_MUTED),
                )
                help_btn.grid(row=0, column=1, sticky="e", padx=(6, 0))
                HelpTooltip(help_btn, help_text, help_title or label.rstrip(":"))

            field_cell = ctk.CTkFrame(row, fg_color="transparent")
            field_cell.grid(row=0, column=1, sticky="nw")
            field_cell.grid_columnconfigure(0, weight=1)
            widget_fn(field_cell)

            if hint_text:
                ctk.CTkLabel(
                    field_cell,
                    text=hint_text,
                    font=ctk.CTkFont(size=10),
                    text_color=TEXT_MUTED,
                    anchor="w",
                    wraplength=520,
                    justify="left",
                ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        def maintenance_block(parent, row_index: int, title: str, location_text: str):
            block = ctk.CTkFrame(
                parent,
                corner_radius=10,
                fg_color=INPUT_SURFACE,
                border_width=1,
                border_color=BORDER_STRONG,
            )
            block.grid(row=row_index, column=0, sticky="ew", pady=(0, 10))
            block.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                block,
                text=title,
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
            location_label = ctk.CTkLabel(
                block,
                text=location_text,
                font=ctk.CTkFont(size=11),
                text_color=TEXT_MUTED,
                anchor="w",
                wraplength=740,
                justify="left",
            )
            location_label.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))
            button_row = ctk.CTkFrame(block, fg_color="transparent")
            button_row.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 10))
            return location_label, button_row

        self._host_var = ctk.StringVar(value=self.cfg["ollama_host"])
        self._comfyui_host_var = ctk.StringVar(
            value=self.cfg.get("comfyui_host", "http://127.0.0.1:8188")
        )
        self._models_dir_var = ctk.StringVar(value=self.cfg["models_dir"])
        self._comfyui_dir_var = ctk.StringVar(value=self.cfg.get("comfyui_dir", ""))

        ollama_env_value = os.environ.get("OLLAMA_MODELS", "").strip()
        self._ollama_models_dir_var = ctk.StringVar(value=ollama_env_value)
        self._ollama_models_dir_default_hint = str(constrained_env._default_ollama_models_dir())
        self._ollama_models_dir_saved_value = ollama_env_value

        # Ollama card
        ollama_card = make_card(
            1,
            "Ollama",
            "Connection, daemon storage location, and LLM model maintenance.",
        )

        def build_ollama_host_row(parent):
            parent.grid_columnconfigure(0, weight=1)
            ctk.CTkEntry(parent, textvariable=self._host_var, width=420).grid(
                row=0, column=0, sticky="w"
            )
            ctk.CTkButton(
                parent,
                text="Test Ollama",
                width=120,
                **self._outline_button_style(),
                command=self._test_ollama_settings,
            ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        setting_row(
            ollama_card,
            0,
            "Ollama Host:",
            build_ollama_host_row,
        )
        setting_row(
            ollama_card,
            1,
            "Ollama Models Directory:",
            lambda parent: ctk.CTkEntry(
                parent,
                textvariable=self._ollama_models_dir_var,
                width=520,
                placeholder_text=f"(default: {self._ollama_models_dir_default_hint})",
            ).grid(row=0, column=0, sticky="w"),
            help_text=(
                "Used by the Ollama daemon for model blob storage. Changing this affects every app on this machine "
                "that talks to Ollama (Open-WebUI, Continue.dev, and other front-ends), not just LocalAI. "
                "The daemon must be restarted to pick up changes."
            ),
            help_title="Ollama Models Directory",
            hint_text=(
                "Leave blank to restore Ollama defaults (~/.ollama/models)." + (
                    " On macOS, export OLLAMA_MODELS in your shell profile and restart Ollama."
                    if sys.platform == "darwin"
                    else ""
                )
            ),
        )

        def build_ollama_model_actions(parent):
            ctk.CTkButton(
                parent,
                text="Open Ollama Models Folder",
                **self._outline_button_style(),
                command=lambda: self._open_folder(self._get_ollama_models_path()),
            ).grid(row=0, column=0, sticky="w", padx=(0, 8))
            ctk.CTkButton(
                parent,
                text="Delete all downloaded LLM models",
                **self._solid_button_style(self._IG_DANGER, "#8a2424"),
                command=self._delete_all_models,
            ).grid(row=0, column=1, sticky="w")

        setting_row(
            ollama_card,
            2,
            "LLM models:",
            build_ollama_model_actions,
            hint_text=f"Current model store: {self._get_ollama_models_path()}",
        )

        # ComfyUI card
        comfyui_card = make_card(
            2,
            "ComfyUI",
            "Connection, installation location, and image-model maintenance.",
        )

        def build_comfyui_host_row(parent):
            parent.grid_columnconfigure(0, weight=1)
            ctk.CTkEntry(parent, textvariable=self._comfyui_host_var, width=420).grid(
                row=0, column=0, sticky="w"
            )
            ctk.CTkButton(
                parent,
                text="Test ComfyUI",
                width=120,
                **self._outline_button_style(),
                command=self._test_comfyui_settings,
            ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        setting_row(
            comfyui_card,
            0,
            "ComfyUI Host:",
            build_comfyui_host_row,
        )
        setting_row(
            comfyui_card,
            1,
            "ComfyUI Directory:",
            lambda parent: ctk.CTkEntry(
                parent,
                textvariable=self._comfyui_dir_var,
                width=520,
            ).grid(row=0, column=0, sticky="w"),
        )

        def build_comfyui_model_actions(parent):
            ctk.CTkButton(
                parent,
                text="Open ComfyUI Models Folder",
                **self._outline_button_style(),
                command=lambda: self._open_folder(self._get_comfyui_models_path()),
            ).grid(row=0, column=0, sticky="w", padx=(0, 8))
            ctk.CTkButton(
                parent,
                text="Delete all downloaded image models",
                **self._solid_button_style(self._IG_DANGER, "#8a2424"),
                command=self._delete_comfyui_models,
            ).grid(row=0, column=1, sticky="w")

        setting_row(
            comfyui_card,
            2,
            "Image models:",
            build_comfyui_model_actions,
            hint_text=(
                f"Current image-model location: {self._get_comfyui_models_path() or '<not found>'}"
            ),
        )

        # General card
        general = make_card(
            3,
            "General",
            "Shared storage and runtime behavior defaults.",
        )
        theme_mode = config.normalize_theme_mode(self.cfg.get("theme_mode")).title()
        self._theme_var = ctk.StringVar(value=theme_mode)
        self._autostart_var = ctk.BooleanVar(value=self.cfg.get("auto_start_ollama", True))
        self._temp_var = ctk.DoubleVar(value=self.cfg.get("temperature", 0.7))

        setting_row(
            general,
            0,
            "Models Directory:",
            lambda parent: ctk.CTkEntry(
                parent,
                textvariable=self._models_dir_var,
                width=520,
            ).grid(row=0, column=0, sticky="w"),
        )
        # Verify & Repair — always-available user-triggered scan that re-uses
        # the same engine the startup self-healing hooks call.
        setting_row(
            general,
            1,
            "Verify & Repair:",
            lambda parent: ctk.CTkButton(
                parent,
                text="Run now",
                width=160,
                command=self._open_verify_repair_dialog,
                **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
            ).grid(row=0, column=0, sticky="w"),
            help_text=(
                "Scans every storage location for orphan Ollama blobs, legacy "
                "ONNX paths, broken config entries, low disk space, and any "
                "interrupted migration. Nothing is changed without an explicit "
                "click on each finding."
            ),
            help_title="Verify & Repair",
            hint_text=(
                "Runs the same reconciliation LocalAI does at startup, but on "
                "demand."
            ),
        )
        setting_row(
            general,
            2,
            "Theme:",
            lambda parent: ctk.CTkOptionMenu(
                parent,
                values=["System", "Dark", "Light"],
                variable=self._theme_var,
                width=180,
                **self._option_menu_style(),
            ).grid(row=0, column=0, sticky="w"),
            help_text=(
                "System follows the current OS light/dark setting. "
                "Dark and Light use editable app palettes from config.json."
            ),
        )
        setting_row(
            general,
            3,
            "Auto-start Ollama:",
            lambda parent: ctk.CTkSwitch(parent, variable=self._autostart_var, text="").grid(
                row=0, column=0, sticky="w"
            ),
        )
        setting_row(
            general,
            4,
            "Low Resources Mode:",
            lambda parent: ctk.CTkSwitch(parent, variable=self._low_res_var, text="").grid(
                row=0, column=0, sticky="w"
            ),
            hint_text=(
                "Checks disk space and RAM before downloads/runs. "
                "Batch mode may remove downloaded models to free space."
            ),
        )
        setting_row(
            general,
            5,
            "Temperature (Response creativity):",
            lambda parent: ctk.CTkSlider(
                parent,
                from_=0,
                to=1,
                variable=self._temp_var,
                number_of_steps=20,
                width=280,
            ).grid(row=0, column=0, sticky="w"),
            help_text=(
                "Lower values make responses more predictable and focused. "
                "Higher values make responses more varied, but can increase mistakes."
            ),
            help_title="Response creativity",
        )

        save_row = ctk.CTkFrame(general, fg_color="transparent")
        save_row.grid(row=6, column=0, sticky="w", pady=(2, 0))
        ctk.CTkButton(
            save_row,
            text="Save Settings",
            command=self._save_settings,
            **self._solid_button_style("#1f6aa5", "#1f538d"),
        ).pack(side="left")

        self._settings_status_lbl = ctk.CTkLabel(
            general,
            text="Settings are validated before saving.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            anchor="w",
            wraplength=760,
            justify="left",
        )
        self._settings_status_lbl.grid(row=7, column=0, sticky="w", pady=(6, 0))

        # Catalog and SKU card
        maintenance = make_card(
            4,
            "Catalog and SKU definitions",
            "Manage catalog metadata and optional SKU profiles.",
        )

        _catalog_lbl, cat_btns = maintenance_block(
            maintenance,
            0,
            "Model catalog",
            f"Catalog file: {catalog.CATALOG_FILE}",
        )
        ctk.CTkButton(
            cat_btns,
            text="Reload Catalog",
            **self._outline_button_style(),
            command=self._reload_catalog,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            cat_btns,
            text="Open Catalog File",
            **self._outline_button_style(),
            command=self._open_catalog_file,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            cat_btns,
            text="Reset catalog to built-in defaults",
            **self._solid_button_style(self._IG_DANGER, "#8a2424"),
            command=self._reset_catalog_to_defaults,
        ).pack(side="left")

        sku_file_exists = system_info.OPTIONAL_SKUS_FILE.exists()
        sku_location_text = (
            f"Definition file: {system_info.OPTIONAL_SKUS_FILE}"
            if sku_file_exists
            else "Definition file: <not found>"
        )
        _sku_lbl, sku_btns = maintenance_block(
            maintenance,
            1,
            "SKU Definitions",
            sku_location_text,
        )
        ctk.CTkButton(
            sku_btns,
            text="Reload SKUs",
            state="normal" if sku_file_exists else "disabled",
            **self._outline_button_style(),
            command=self._reload_optional_skus,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            sku_btns,
            text="Open SKU File",
            state="normal" if sku_file_exists else "disabled",
            **self._outline_button_style(),
            command=self._open_optional_skus_file,
        ).pack(side="left")

        # Installed-but-not-in-catalog models — Settings reconciliation panel.
        self._build_uncatalogued_panel(scroll, row_index=5)

        ctk.CTkLabel(
            scroll,
            text="LocalAI Studio  —  Ron Martinsen  —  March 2026",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            anchor="center",
        ).grid(row=6, column=0, sticky="ew", padx=20, pady=(16, 8))

    def _get_ollama_models_path(self) -> str:
        """Return the Ollama models directory path."""
        # OLLAMA_MODELS env var overrides default
        env = os.environ.get("OLLAMA_MODELS")
        if env and Path(env).is_dir():
            return env
        default = Path.home() / ".ollama" / "models"
        return str(default)

    def _get_comfyui_models_path(self) -> Optional[str]:
        """Return the ComfyUI models directory path, or None."""
        comfyui = self._comfyui_installed_path()
        if comfyui:
            models = comfyui / "models"
            if models.is_dir():
                return str(models)
        return None

    def _open_folder(self, path: Optional[str]):
        """Open a folder in the system file explorer."""
        if not path or not Path(path).is_dir():
            messagebox.showinfo("Folder not found",
                                f"The folder does not exist:\n{path or '(not configured)'}",
                                 parent=self)
            return
        self._open_path(Path(path))

    def _open_path(self, path: Path):
        """Open a file/folder with the platform default application."""
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _delete_comfyui_models(self):
        """Delete all downloaded image generation models from ComfyUI."""
        models_path = self._get_comfyui_models_path()
        if not models_path:
            messagebox.showinfo("ComfyUI not found",
                                "ComfyUI is not installed — no image models to delete.",
                                parent=self)
            return

        # Count checkpoint and diffusion model files
        p = Path(models_path)
        model_files = []
        for subdir in ("checkpoints", "diffusion_models"):
            d = p / subdir
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file() and f.suffix.lower() in (
                        ".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin"
                    ):
                        model_files.append(f)

        if not model_files:
            messagebox.showinfo("No models found",
                                "No image generation models found in ComfyUI.",
                                parent=self)
            return

        names = "\n  ".join(f.name for f in model_files[:10])
        if len(model_files) > 10:
            names += f"\n  … and {len(model_files) - 10} more"

        ok = messagebox.askyesno(
            "Delete Image Models",
            f"This will delete {len(model_files)} image model(s) from ComfyUI:\n\n"
            f"  {names}\n\n"
            "You will need to re-download them to use them again.\n\nContinue?",
            parent=self,
        )
        if not ok:
            return

        deleted = 0
        for f in model_files:
            try:
                f.unlink()
                logger.info(f"Deleted image model: {f.name}")
                deleted += 1
            except Exception as e:
                logger.error(f"Could not delete {f.name}: {e}")

        self.set_status(f"Deleted {deleted} image model(s).")
        self._refresh_model_cards()

    def _validate_settings_inputs(self) -> tuple[bool, list[str], list[str]]:
        from urllib.parse import urlparse

        errors: list[str] = []
        warnings: list[str] = []

        for label, value in (
            ("Ollama Host", self._host_var.get().strip()),
            ("ComfyUI Host", self._comfyui_host_var.get().strip()),
        ):
            parsed = urlparse(value)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                errors.append(f"{label} must be a valid http(s) URL, such as http://localhost:11434.")

        models_dir = self._models_dir_var.get().strip()
        if not models_dir:
            errors.append("Models Directory is required.")
        else:
            try:
                p = Path(models_dir).expanduser()
                if p.exists() and not p.is_dir():
                    errors.append("Models Directory points to a file, not a folder.")
                elif not p.exists() and not p.parent.exists():
                    errors.append("Models Directory parent folder does not exist.")
            except OSError as exc:
                errors.append(f"Models Directory is invalid: {exc}")

        comfyui_dir = self._comfyui_dir_var.get().strip()
        if comfyui_dir:
            try:
                p = Path(comfyui_dir).expanduser()
                if p.exists() and not p.is_dir():
                    errors.append("ComfyUI Directory points to a file, not a folder.")
                elif not p.exists():
                    warnings.append("ComfyUI Directory does not exist yet; setup or restart may be needed.")
                elif not (p / "main.py").exists():
                    warnings.append("ComfyUI Directory exists but main.py was not found.")
            except OSError as exc:
                errors.append(f"ComfyUI Directory is invalid: {exc}")

        ollama_dir = self._ollama_models_dir_var.get().strip() if hasattr(self, "_ollama_models_dir_var") else ""
        if ollama_dir:
            try:
                p = Path(ollama_dir).expanduser()
                if p.exists() and not p.is_dir():
                    errors.append("Ollama Models Directory points to a file, not a folder.")
                elif not p.exists() and not p.parent.exists():
                    errors.append("Ollama Models Directory parent folder does not exist.")
            except OSError as exc:
                errors.append(f"Ollama Models Directory is invalid: {exc}")

        return not errors, errors, warnings

    def _set_settings_status(self, text: str, color=TEXT_MUTED):
        if hasattr(self, "_settings_status_lbl") and self._settings_status_lbl is not None:
            self._settings_status_lbl.configure(text=text, text_color=color)
        self.set_status(text)

    def _test_ollama_settings(self):
        host = self._host_var.get().strip()
        self._set_settings_status(f"Testing Ollama at {host} …", WARN_TEXT)

        def _test():
            client = OllamaClient(host)
            if client.is_running():
                version = client.version()
                self.after(0, lambda: self._set_settings_status(
                    f"Ollama reachable at {host} (v{version}).", SUCCESS_TEXT
                ))
            else:
                self.after(0, lambda: self._set_settings_status(
                    f"Ollama did not respond at {host}.", ERROR_TEXT
                ))

        threading.Thread(target=_test, daemon=True).start()

    def _test_comfyui_settings(self):
        host = self._comfyui_host_var.get().strip()
        self._set_settings_status(f"Testing ComfyUI at {host} …", WARN_TEXT)

        def _test():
            client = ComfyUIClient(host)
            if client.is_running():
                self.after(0, lambda: self._set_settings_status(
                    f"ComfyUI reachable at {host}.", SUCCESS_TEXT
                ))
            else:
                self.after(0, lambda: self._set_settings_status(
                    f"ComfyUI did not respond at {host}.", ERROR_TEXT
                ))

        threading.Thread(target=_test, daemon=True).start()

    def _save_settings(self):
        ok, errors, warnings = self._validate_settings_inputs()
        if not ok:
            msg = "\n".join(errors)
            self._set_settings_status(msg, ERROR_TEXT)
            messagebox.showerror("Settings validation failed", msg, parent=self)
            return

        # Drive-mismatch confirmation: if any of the three storage dirs
        # is on a different drive than the app itself, warn the user
        # once. Cancel restores the previous saved values for any
        # changed entries and aborts the save.
        if sys.platform == "win32":
            try:
                app_drive = Path(__file__).resolve().parent.parent.drive.upper()
            except Exception:
                app_drive = ""
            mismatches: list[str] = []
            for label, value in (
                ("Models Directory", self._models_dir_var.get().strip()),
                ("ComfyUI Directory", self._comfyui_dir_var.get().strip()),
                ("Ollama Models Directory",
                 self._ollama_models_dir_var.get().strip()
                 if hasattr(self, "_ollama_models_dir_var") else ""),
            ):
                if not value:
                    continue
                try:
                    drv = Path(value).expanduser().resolve().drive.upper()
                except Exception:
                    drv = ""
                if app_drive and drv and drv != app_drive:
                    mismatches.append(f"  • {label}: {value}  (on {drv})")
            if mismatches:
                proceed = messagebox.askokcancel(
                    "Storage on a different drive",
                    "One or more storage folders are on a different drive "
                    f"than LocalAI Studio (app drive: {app_drive}):\n\n"
                    + "\n".join(mismatches)
                    + "\n\nThis is supported, but keep in mind:\n"
                    "• If the other drive disappears (USB unplugged, "
                    "network share offline) the models will be unavailable.\n"
                    "• Ollama is a shared daemon — every app on this PC "
                    "that talks to Ollama will see the new model "
                    "location too.\n\n"
                    "Click OK to save anyway, or Cancel to revert.",
                    parent=self,
                )
                if not proceed:
                    # Revert StringVars to whatever was last saved.
                    self._models_dir_var.set(self.cfg.get("models_dir", ""))
                    self._comfyui_dir_var.set(self.cfg.get("comfyui_dir", ""))
                    if hasattr(self, "_ollama_models_dir_var"):
                        self._ollama_models_dir_var.set(
                            getattr(self, "_ollama_models_dir_saved_value", "")
                        )
                    self._set_settings_status("Save cancelled.", WARN_TEXT)
                    return

        previous_theme_mode = config.normalize_theme_mode(self.cfg.get("theme_mode"))
        current_page = self._current_page
        self.cfg["ollama_host"] = self._host_var.get()
        self.cfg["comfyui_host"] = self._comfyui_host_var.get()
        self.cfg["comfyui_dir"] = self._comfyui_dir_var.get()
        self.cfg["models_dir"] = self._models_dir_var.get()
        self.cfg["theme_mode"] = config.normalize_theme_mode(self._theme_var.get())
        self.cfg["theme_palettes"] = config.normalize_theme_palettes(self.cfg.get("theme_palettes"))
        self.cfg["dark_mode"] = self.cfg["theme_mode"] == "dark"
        self.cfg["auto_start_ollama"] = self._autostart_var.get()
        self.cfg["low_resources_mode"] = self._low_res_var.get()
        self.cfg["toolbox_left_column_mode"] = "normal"
        self.cfg["temperature"] = round(self._temp_var.get(), 2)
        try:
            Path(self.cfg["models_dir"]).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"Could not create Models Directory: {exc}"
            self._set_settings_status(msg, ERROR_TEXT)
            messagebox.showerror("Settings validation failed", msg, parent=self)
            return

        if not config.save(self.cfg):
            msg = "Settings could not be saved. Check localai.log for details."
            self._set_settings_status(msg, ERROR_TEXT)
            messagebox.showerror("Save failed", msg, parent=self)
            return
        self.ollama = OllamaClient(self.cfg["ollama_host"])
        self.comfyui = ComfyUIClient(self.cfg["comfyui_host"])
        # Keep comfyui_path.bat in sync for batch file compatibility
        comfyui_dir_val = self._comfyui_dir_var.get()
        if comfyui_dir_val:
            self._sync_comfyui_path_bat(Path(comfyui_dir_val))
        _apply_theme_palette_globals(self.cfg)
        ctk.set_appearance_mode(self.cfg["theme_mode"].title())
        theme_changed = self.cfg["theme_mode"] != previous_theme_mode
        if theme_changed:
            self._rebuild_ui_for_theme_change(current_page)
        elif "toolbox" in self._pages:
            self._apply_toolbox_browser_density()
            self._refresh_toolbox_cards()

        # Persist OLLAMA_MODELS to the user environment if changed.
        # setx is Windows-only; on macOS the helper is a no-op and the
        # user is told to export OLLAMA_MODELS manually (see Settings
        # row tooltip). The daemon must be restarted by the user — we
        # do not auto-restart it because it's shared infrastructure.
        ollama_extra_warn = ""
        if hasattr(self, "_ollama_models_dir_var"):
            new_ollama_val = self._ollama_models_dir_var.get().strip()
            previous_ollama_val = getattr(self, "_ollama_models_dir_saved_value", "")
            if new_ollama_val != previous_ollama_val:
                if sys.platform == "win32":
                    try:
                        if new_ollama_val:
                            Path(new_ollama_val).expanduser().mkdir(parents=True, exist_ok=True)
                            if self._set_user_env_var("OLLAMA_MODELS", new_ollama_val):
                                os.environ["OLLAMA_MODELS"] = new_ollama_val
                                ollama_extra_warn = (
                                    "OLLAMA_MODELS updated — restart the Ollama "
                                    "daemon (or sign out / sign in) for the change "
                                    "to take effect across every app on this PC."
                                )
                            else:
                                ollama_extra_warn = (
                                    "OLLAMA_MODELS save failed — see localai.log."
                                )
                        else:
                            if self._delete_user_env_var("OLLAMA_MODELS"):
                                os.environ.pop("OLLAMA_MODELS", None)
                                ollama_extra_warn = (
                                    "OLLAMA_MODELS cleared — restart the Ollama "
                                    "daemon to return to the default location."
                                )
                            else:
                                ollama_extra_warn = (
                                    "OLLAMA_MODELS clear failed — see localai.log."
                                )
                    except OSError as exc:
                        ollama_extra_warn = (
                            f"OLLAMA_MODELS not applied: {exc}"
                        )
                else:
                    ollama_extra_warn = (
                        "On this platform, export OLLAMA_MODELS in your "
                        "shell profile manually and restart Ollama for the "
                        "change to take effect."
                    )
                self._ollama_models_dir_saved_value = new_ollama_val

        status_parts = ["Settings saved."]
        if warnings:
            status_parts.append(" ".join(warnings))
        if ollama_extra_warn:
            status_parts.append(ollama_extra_warn)
        status_color = WARN_TEXT if (warnings or ollama_extra_warn) else SUCCESS_TEXT
        self._set_settings_status(" ".join(status_parts), status_color)
        logger.info("Settings saved.")

    # ── Navigation ────────────────────────────────────────────────────────────

    _current_page: str = "models"

    def _init_comfyui_if_needed(self):
        """Run ComfyUI migration and startup check once — deferred until GPU confirmed or tab visited."""
        if self._comfyui_initialized:
            return
        self._comfyui_initialized = True
        self.after(0, self._migrate_comfyui_if_needed)
        self.after(100, self._check_comfyui_async)

    def _wire_nav_rail_arrow_keys(self) -> None:
        """Bind Up/Down/Home/End on every nav button so users can move through
        the left rail with arrow keys after the first Tab into it.

        Tab/Shift-Tab still traverse the buttons individually (handled by
        ``a11y.install()``), so this is purely an extra convenience pattern
        — the WAI-ARIA "menu" / "tablist" pattern allows either.
        """
        try:
            pages = list(self._nav_btns.keys())
            buttons = list(self._nav_btns.values())
        except Exception:
            return
        if not buttons:
            return

        def _focus(idx: int) -> None:
            idx = max(0, min(len(buttons) - 1, idx))
            try:
                canvas = getattr(buttons[idx], "_canvas", None)
                if canvas is not None:
                    canvas.focus_set()
            except Exception:
                pass

        def _activate(idx: int) -> None:
            try:
                self._switch_page(pages[idx])
            except Exception:
                pass

        for i, btn in enumerate(buttons):
            canvas = getattr(btn, "_canvas", None)
            if canvas is None:
                continue
            canvas.bind("<Down>", lambda _e, ii=i: (_focus(ii + 1), "break")[1], add="+")
            canvas.bind("<Up>", lambda _e, ii=i: (_focus(ii - 1), "break")[1], add="+")
            canvas.bind("<Home>", lambda _e: (_focus(0), "break")[1], add="+")
            canvas.bind("<End>", lambda _e: (_focus(len(buttons) - 1), "break")[1], add="+")

        # Global Ctrl+1..9 to jump to a page (escape any focused field).
        try:
            a11y.bind_app_shortcuts(
                self,
                page_switcher=self._switch_page,
                pages=[(p, p.title()) for p in pages],
            )
        except Exception:
            pass

        # Make Tab / Shift-Tab actually move focus. Without this, CTk's
        # Canvas-drawn widgets never get Tab navigation because no widget
        # in the bindtag chain consumes the event to call tk_focusNext.
        try:
            a11y.install_global_focus_traversal(self)
        except Exception:
            pass

        # Park initial focus on the first nav button so the focus ring is
        # visible from launch and arrow keys work without requiring a click.
        try:
            first_nav = next(iter(self._nav_btns.values()), None)
            if first_nav is not None:
                # CTkButton's inner canvas is the actual focus target.
                target = getattr(first_nav, "_canvas", first_nav)
                self.after(50, lambda t=target: t.focus_set())
        except Exception:
            pass

    def _switch_page(self, page: str):
        if page == "docs":
            self._open_docs()
            return  # don't change the active page or highlight the button

        # v5: lazy-build the page on first visit
        if page not in self._pages and page in self._page_builders:
            t0 = time.time()
            self._page_builders[page]()
            logger.debug(f"Built '{page}' page on first visit in {(time.time()-t0)*1000:.0f} ms")

        self._current_page = page
        for name, frame in self._pages.items():
            frame.grid_remove()
        # v2026.06.01.10: pages live in row=1 of self._content because
        # the optional incomplete-setup banner occupies row=0. The banner
        # is hidden by default and only shows when _refresh_setup_banner()
        # detects a broken install (missing Ollama / ComfyUI / CUDA torch
        # on an NVIDIA box). Page content auto-shifts down when it shows.
        self._pages[page].grid(row=1, column=0, sticky="nsew")

        for name, btn in self._nav_btns.items():
            active = name == page
            btn.configure(
                fg_color=self._IG_ACCENT if active else INPUT_SURFACE,
                text_color=self._IG_ACCENT_TEXT if active else TEXT_PRIMARY,
                hover_color=BUTTON_SECONDARY_HOVER,
            )

        if page == "home":
            self._refresh_home_page()
        elif page == "models":
            if not getattr(self, "_models_page_just_built", False):
                self._refresh_model_cards()
        elif page == "toolbox":
            self._refresh_toolbox_cards()
        elif page == "image_gen":
            self._init_comfyui_if_needed()
            self._img_refresh_comfyui(start_if_needed=False)
        elif page == "system":
            self._update_system_page()
        elif page == "logs":
            self._refresh_logs()

    # ── Incomplete-setup banner (v2026.06.01.10) ──────────────────────────────
    #
    # When setup.bat fails silently (e.g. window auto-closed before user saw
    # the error, or user clicked through prompts without noticing CUDA wasn't
    # installed on their NVIDIA box), the app launches into a degraded state:
    # no Ollama, no ComfyUI, CPU-only torch. Today this is logged as quiet
    # WARNING lines nobody reads. The banner below makes it loud and visible:
    # a yellow strip at the top of every page that lists what's broken, with
    # a button that opens a modal with re-run-setup guidance.
    #
    # Detection signals (all "best effort" — never raise to the UI thread):
    #   1. self.ollama_ok is False after _check_ollama_async finishes
    #   2. self._comfyui_installed_path() is None (ComfyUI absent on disk)
    #   3. self._pytorch_cuda_missing_on_nvidia is True (NVIDIA + CPU torch)
    #
    # Refresh hook points (where we call self._refresh_setup_warning_banner):
    #   - End of _apply_gpu_detection_result (sets the NVIDIA-CPU flag)
    #   - End of _check_ollama_async (sets self.ollama_ok)
    #   - End of _try_start_ollama (may flip self.ollama_ok)

    def _build_setup_warning_banner(self) -> None:
        """Create the incomplete-setup warning banner.

        Banner is constructed once inside ``self._content`` at row=0 and
        kept hidden via ``grid_remove()`` until
        :meth:`_refresh_setup_warning_banner` detects a broken-install
        signal. Page widgets live at row=1 so they auto-shift down when
        the banner shows. Safe to call before any async startup check
        completes — banner stays hidden until there is something to say.
        """
        try:
            container = ctk.CTkFrame(
                self._content,
                corner_radius=0,
                fg_color=("#fff4ce", "#3a2f0d"),
                border_width=1,
                border_color=WARN_TEXT,
            )
            container.grid(row=0, column=0, sticky="ew")
            container.grid_columnconfigure(1, weight=1)
            container.grid_remove()

            icon = ctk.CTkLabel(
                container,
                text="⚠",
                text_color=WARN_TEXT,
                font=ctk.CTkFont(size=18, weight="bold"),
            )
            icon.grid(row=0, column=0, padx=(14, 8), pady=8, sticky="w")

            label = ctk.CTkLabel(
                container,
                text="",
                text_color=WARN_TEXT,
                anchor="w",
                justify="left",
                wraplength=900,
            )
            label.grid(row=0, column=1, padx=(0, 12), pady=8, sticky="ew")

            details_btn = ctk.CTkButton(
                container,
                text="Show details",
                width=120,
                command=self._show_setup_warning_details,
            )
            details_btn.grid(row=0, column=2, padx=(0, 14), pady=8, sticky="e")

            self._setup_warning_banner = container
            self._setup_warning_label = label
            self._setup_warning_button = details_btn
        except Exception as exc:
            # The banner is purely advisory — never let a UI build error
            # block the app from launching.
            logger.debug(f"Incomplete-setup banner build failed: {exc}")
            self._setup_warning_banner = None
            self._setup_warning_label = None
            self._setup_warning_button = None

    def _detect_incomplete_setup_state(self) -> list[str]:
        """Return a list of human-readable issues, empty when setup is OK.

        Each entry is a short sentence ready to display in the banner /
        details modal. Order matters — Ollama first because users hit it
        first; ComfyUI second (image gen); GPU acceleration third
        (performance, not correctness).
        """
        issues: list[str] = []
        try:
            if not getattr(self, "ollama_ok", False):
                issues.append(
                    "Ollama isn't installed or isn't running — chat, vision, "
                    "and embedding models will not work."
                )
        except Exception:
            pass
        try:
            if self._comfyui_installed_path() is None:
                issues.append(
                    "ComfyUI isn't installed — image generation is unavailable."
                )
        except Exception:
            pass
        try:
            if getattr(self, "_pytorch_cuda_missing_on_nvidia", False):
                issues.append(
                    "An NVIDIA GPU is present but PyTorch is CPU-only — image "
                    "generation will run on the CPU (very slow). Re-run setup "
                    "to install CUDA PyTorch."
                )
        except Exception:
            pass
        return issues

    def _refresh_setup_warning_banner(self) -> None:
        """Show or hide the banner based on current setup-state detection.

        Idempotent — safe to call repeatedly from any async completion
        hook. Quietly no-ops if the banner widget hasn't been built yet
        (e.g. during early startup before _build_ui finishes) or has
        been destroyed by a theme rebuild.
        """
        banner = getattr(self, "_setup_warning_banner", None)
        label = getattr(self, "_setup_warning_label", None)
        if banner is None or label is None:
            return
        try:
            if not banner.winfo_exists():
                return
        except Exception:
            return

        issues = self._detect_incomplete_setup_state()
        if not issues:
            try:
                banner.grid_remove()
            except Exception:
                pass
            return

        if len(issues) == 1:
            summary = "Setup looks incomplete: " + issues[0]
        else:
            summary = (
                f"Setup looks incomplete ({len(issues)} issues) — "
                "click \u201cShow details\u201d for the full list and how to fix."
            )
        try:
            label.configure(text=summary)
            banner.grid()
        except Exception:
            pass

    def _show_setup_warning_details(self) -> None:
        """Open a modal listing all detected issues + remediation steps."""
        try:
            issues = self._detect_incomplete_setup_state()
            if not issues:
                return

            top = ctk.CTkToplevel(self)
            top.title("Incomplete setup detected")
            top.transient(self)
            top.geometry("640x420")
            try:
                top.grab_set()
            except Exception:
                pass

            frame = ctk.CTkFrame(top, corner_radius=0, fg_color="transparent")
            frame.pack(fill="both", expand=True, padx=20, pady=20)

            header = ctk.CTkLabel(
                frame,
                text="The app started, but the install isn't complete.",
                font=ctk.CTkFont(size=15, weight="bold"),
                anchor="w",
                justify="left",
            )
            header.pack(anchor="w", pady=(0, 12))

            issues_box = ctk.CTkTextbox(frame, height=180, wrap="word")
            issues_box.pack(fill="both", expand=True)
            for i, msg in enumerate(issues, 1):
                issues_box.insert("end", f"{i}. {msg}\n\n")
            issues_box.configure(state="disabled")

            hint = ctk.CTkLabel(
                frame,
                text=(
                    "How to fix:\n"
                    " \u2022 Close this app, then double-click setup.bat "
                    "again. It now auto-captures everything to setup.log "
                    "next to the app — share that file if you need help.\n"
                    " \u2022 If GPU acceleration is missing on an NVIDIA box, "
                    "run fix_nvidia_pytorch.bat (or re-run setup.bat and answer "
                    "\u201cy\u201d when prompted)."
                ),
                anchor="w",
                justify="left",
                wraplength=600,
            )
            hint.pack(anchor="w", pady=(12, 12))

            btn = ctk.CTkButton(frame, text="Close", width=120, command=top.destroy)
            btn.pack(anchor="e")
        except Exception as exc:
            logger.debug(f"Incomplete-setup details dialog failed: {exc}")

    # ── Ollama management ─────────────────────────────────────────────────────

    def _check_ollama_async(self):
        def _check():
            if self.ollama.is_running():
                self.ollama_ok = True
                ver = self.ollama.version()
                self.after(0, lambda: self._ollama_status_label.configure(
                    text=f"Ollama v{ver} running", text_color=SUCCESS_TEXT
                ))
                logger.info(f"Ollama running (v{ver}) at {self.cfg['ollama_host']}")
            else:
                self.ollama_ok = False
                if self.cfg.get("auto_start_ollama"):
                    self.after(0, lambda: self._ollama_status_label.configure(
                        text="Starting Ollama …", text_color=WARN_TEXT
                    ))
                    self._try_start_ollama()
                else:
                    self.after(0, lambda: self._ollama_status_label.configure(
                        text="Ollama not running.\nInstall from ollama.com",
                        text_color=TEXT_MUTED
                    ))
                    logger.warning("Ollama is not running. Download it from https://ollama.com")
            # v2026.06.01.10: refresh the in-app incomplete-setup banner
            # whenever the Ollama probe completes. self.ollama_ok is now
            # authoritative, so banner can show/hide accordingly.
            self.after(0, self._refresh_setup_warning_banner)

        threading.Thread(target=_check, daemon=True).start()

    def _try_start_ollama(self):
        def _start():
            try:
                popen_kwargs = {
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                subprocess.Popen(["ollama", "serve"], **popen_kwargs)
                time.sleep(3)
                if self.ollama.is_running():
                    self.ollama_ok = True
                    self.after(0, lambda: self._ollama_status_label.configure(
                        text=f"Ollama started", text_color=SUCCESS_TEXT
                    ))
                    logger.info("Ollama started successfully.")
                else:
                    self.after(0, lambda: self._ollama_status_label.configure(
                        text="Ollama not found.\nGet it at ollama.com", text_color=TEXT_MUTED
                    ))
                    logger.warning("Could not start Ollama. Is it installed?")
            except FileNotFoundError:
                self.after(0, lambda: self._ollama_status_label.configure(
                    text="Ollama not installed.\nGet it at ollama.com", text_color=TEXT_MUTED
                ))
                logger.warning("Ollama executable not found. Download from https://ollama.com")
            # v2026.06.01.10: refresh the incomplete-setup banner after the
            # auto-start attempt finishes — self.ollama_ok may have flipped.
            self.after(0, self._refresh_setup_warning_banner)
        threading.Thread(target=_start, daemon=True).start()

    # ── ComfyUI management ────────────────────────────────────────────────────

    def _sync_comfyui_path_bat(self, comfyui_path: Path):
        """Write comfyui_path.bat to keep batch files working (Windows only)."""
        if sys.platform != "win32":
            return
        bat_file = Path(__file__).parent.parent / "comfyui_path.bat"
        try:
            bat_file.write_text(
                f"set LOCALAI_COMFYUI={comfyui_path}\n", encoding="utf-8"
            )
        except OSError:
            pass

    def _migrate_comfyui_if_needed(self):
        """Detect ComfyUI at any legacy location and offer migration to the standard path."""
        pending = self._pending_comfyui_migration()
        if pending is None:
            return
        self._prompt_comfyui_migration(pending)

    def _pending_comfyui_migration(self) -> Optional[dict]:
        """Return migration context (old_path, new_path) if a ComfyUI move is offerable, else None.

        Pure check — no dialog, no filesystem mutation, no config save. Used
        by both the single-purpose dialog (`_migrate_comfyui_if_needed`)
        and the unified composite first-run prompt
        (`_check_storage_relocation_on_startup`).
        """
        from src.config import _default_data_dir
        app_root = Path(__file__).parent.parent
        new_path = _default_data_dir() / "ComfyUI"
        cfg_path = Path(self.cfg.get("comfyui_dir", "")) if self.cfg.get("comfyui_dir") else None

        # Skip if config already points to the canonical location
        if cfg_path and cfg_path == new_path:
            return None

        # Skip if the canonical location already has a valid install
        if (new_path / "main.py").exists():
            return None

        # Determine the legacy location to migrate from, in priority order:
        #   1. The path currently in config IF it has a real install
        #      (this picks up the post-v5.3.7 case where config points
        #      at the old %LOCALAPPDATA%\LocalAI\ComfyUI default).
        #   2. The old exact embedded path <app_root>/ComfyUI.
        old_path = None
        if cfg_path and (cfg_path / "main.py").exists():
            old_path = cfg_path
        if old_path is None:
            candidate = app_root / "ComfyUI"
            if (candidate / "main.py").exists() and candidate != new_path:
                old_path = candidate
        if old_path is None:
            return None
        return {"old_path": old_path, "new_path": new_path}

    def _prompt_comfyui_migration(self, pending: dict) -> None:
        """Show the single-purpose ComfyUI Yes/No/Cancel dialog and apply."""
        old_path = pending["old_path"]
        new_path = pending["new_path"]
        answer = messagebox.askyesnocancel(
            "Migrate ComfyUI?",
            f"ComfyUI was found at:\n"
            f"  {old_path}\n\n"
            f"The standard location is:\n"
            f"  {new_path}\n\n"
            f"Move ComfyUI to the standard location?\n\n"
            f"Yes = Move files (recommended)\n"
            f"No = Keep at current location (update config only)\n"
            f"Cancel = Decide later",
            parent=self,
        )
        if answer is None:  # Cancel
            return
        self._apply_comfyui_migration(pending, move=bool(answer))

    def _apply_comfyui_migration(self, pending: dict, *, move: bool) -> None:
        """Apply a ComfyUI migration with the given choice (move=True or update-config-only).

        DO NOT REGRESS: Windows ``shutil.move(src, dst)`` semantics nest the
        source INTO ``dst`` as a subdirectory when ``dst`` already exists
        as a directory.  That bug burned 73 GB of model re-downloads on a
        high-VRAM system in v5.4.6 because ``setup.bat`` (and the Z-Image
        support installer) pre-create empty scaffold subdirs at the
        canonical destination.  All destination-handling MUST flow through
        the 4-branch dispatch below.  See the regression-critical DO NOT REGRESS row
        "Migration apply must use 4-branch destination dispatch".
        """
        old_path = pending["old_path"]
        new_path = pending["new_path"]
        if move:
            import shutil
            from src import migration as _migration
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                branch_taken = self._move_with_safe_dest_handling(
                    old_path, new_path, label="ComfyUI"
                )
                # POST-MOVE VERIFICATION — never log success before the
                # destination actually contains the install.  The earlier
                # log lied for 15 seconds before the second attempt
                # surfaced the real error; that anti-pattern is banned.
                if not (new_path / "main.py").exists():
                    raise RuntimeError(
                        f"Post-move sentinel missing: {new_path}\\main.py "
                        f"does not exist after a {branch_taken!r} migration "
                        f"of ComfyUI from {old_path}. Refusing to advertise "
                        "success and leaving config pointed at the old path."
                    )
                self.cfg["comfyui_dir"] = str(new_path)
                if not config.save(self.cfg):
                    logger.error("Could not persist ComfyUI migration target")
                self._sync_comfyui_path_bat(new_path)
                messagebox.showinfo(
                    "Migration Complete",
                    f"ComfyUI moved to:\n{new_path}",
                    parent=self,
                )
                logger.info(
                    f"Migrated ComfyUI from {old_path} to {new_path} "
                    f"({branch_taken} branch)"
                )
            except Exception as e:
                messagebox.showerror(
                    "Migration Failed",
                    f"Could not move ComfyUI:\n{e}\n\nUsing current location instead.",
                    parent=self,
                )
                self.cfg["comfyui_dir"] = str(old_path)
                if not config.save(self.cfg):
                    logger.error("Could not persist ComfyUI fallback path")
                self._sync_comfyui_path_bat(old_path)
                logger.error(f"ComfyUI migration failed: {e}")
        else:
            self.cfg["comfyui_dir"] = str(old_path)
            if not config.save(self.cfg):
                logger.error("Could not persist ComfyUI path")
            self._sync_comfyui_path_bat(old_path)
            logger.info(f"User chose to keep ComfyUI at {old_path}")

    def _move_with_safe_dest_handling(
        self, src: Path, dst: Path, *, label: str = "directory"
    ) -> str:
        """Move ``src`` to ``dst`` using the 4-branch destination dispatch.

        Returns the name of the branch taken (``"absent"``, ``"empty"``, or
        ``"merge"``) for logging and tests.  Raises ``RuntimeError`` if
        ``dst`` is a non-directory file (we refuse to silently overwrite a
        user's file).  Raises on merge errors so the caller can log + fall
        back instead of advertising a success that did not happen.

        Branches:

        * **absent** — ``dst`` does not exist.  Plain ``shutil.move`` (the
          fast, atomic-on-same-volume path).
        * **non-directory file at dst** — ``RuntimeError``.  We will not
          clobber a file that the user (or another tool) put there.
        * **empty scaffold** — ``dst`` exists as a directory but contains
          zero files anywhere in its tree (e.g. ``setup.bat`` pre-created
          ``<dst>\\models\\checkpoints`` as an empty placeholder).  Safe to
          ``shutil.rmtree(dst)`` then ``shutil.move(src, dst)``.
        * **has content** — ``dst`` exists with real files.  Route through
          :func:`src.migration.safe_merge_directory` (``robocopy /E /XO``
          semantics: dest-wins on conflict), then ``shutil.rmtree(src)``
          on a clean run.  Errors from the merge surface so we never delete
          the source after a partial merge.
        """
        import shutil
        from src import migration as _migration
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            raise RuntimeError(
                f"Source {label} does not exist: {src}"
            )
        if not dst.exists():
            shutil.move(str(src), str(dst))
            return "absent"
        if dst.is_file() or (not dst.is_dir()):
            raise RuntimeError(
                f"Refusing to move {label}: destination {dst} exists and is "
                "not a directory.  Move or remove the file at that path "
                "and try again."
            )
        if _migration.is_empty_tree(dst):
            shutil.rmtree(str(dst))
            shutil.move(str(src), str(dst))
            return "empty"
        # has content — merge instead of nest
        merge_result = _migration.safe_merge_directory(src, dst)
        if merge_result["errors"]:
            preview = "; ".join(merge_result["errors"][:5])
            raise RuntimeError(
                f"safe_merge_directory reported {len(merge_result['errors'])} "
                f"error(s) merging {label} from {src} into {dst} "
                f"(first 5: {preview}).  Source NOT removed."
            )
        # Clean run — remove the now-merged source.
        try:
            shutil.rmtree(str(src))
        except OSError as exc:
            raise RuntimeError(
                f"Merge of {label} into {dst} succeeded "
                f"({merge_result['copied']} copied / {merge_result['skipped']} "
                f"skipped) but source {src} could not be removed: {exc}.  "
                "Remove it manually."
            ) from exc
        return "merge"

    def _migrate_models_dir_if_needed(self):
        """Detect models dir at a legacy location and offer migration to the standard path."""
        pending = self._pending_models_dir_migration()
        if pending is None:
            return
        if pending.get("auto_adopt"):
            # Silent adoption — old path empty or missing.
            self.cfg["models_dir"] = str(pending["new_default"])
            if not config.save(self.cfg):
                logger.error("Could not persist default models directory")
            logger.info(
                f"Models dir updated to {pending['new_default']} "
                "(old path was empty or missing)"
            )
            return
        self._prompt_models_dir_migration(pending)

    def _pending_models_dir_migration(self) -> Optional[dict]:
        """Return migration context if models_dir should be relocated, else None.

        Returned dict shape:
            {"current": Path, "new_default": Path, "auto_adopt": bool}

        ``auto_adopt`` True means the old path is empty/missing so the new
        default can be silently adopted without bothering the user; False
        means there's real ONNX/OpenVINO content that warrants a prompt.
        """
        from src.config import _default_data_dir
        new_default = _default_data_dir() / "models"
        current = Path(self.cfg.get("models_dir", ""))
        if current == new_default:
            return None

        def _has_models(p: Path) -> bool:
            for sub in ("onnx", "ov"):
                d = p / sub
                if d.is_dir() and any(d.rglob("*")):
                    return True
            return False

        auto_adopt = not current.exists() or not _has_models(current)
        return {"current": current, "new_default": new_default, "auto_adopt": auto_adopt}

    def _prompt_models_dir_migration(self, pending: dict) -> None:
        """Show the single-purpose Models Yes/No/Cancel dialog and apply."""
        current = pending["current"]
        new_default = pending["new_default"]
        answer = messagebox.askyesnocancel(
            "Move Models Directory?",
            f"ONNX/OpenVINO models were found at:\n"
            f"  {current}\n\n"
            f"The new default models location is:\n"
            f"  {new_default}\n\n"
            f"What would you like to do?\n\n"
            f"Yes    = Move model files to the new location\n"
            f"No     = Keep files where they are, update config only\n"
            f"Cancel = Decide later",
            parent=self,
        )
        if answer is None:  # Cancel
            return
        self._apply_models_dir_migration(pending, move=bool(answer))

    def _apply_models_dir_migration(self, pending: dict, *, move: bool) -> None:
        """Apply a models_dir migration with the given choice.

        DO NOT REGRESS: same 4-branch destination dispatch as
        :meth:`_apply_comfyui_migration`.  Per-subdir, not whole-tree,
        because only the ``onnx`` and ``ov`` subtrees are owned by this
        function — other subdirs (checkpoints, loras, …) belong to other
        owners and must not be moved here.
        """
        current = pending["current"]
        new_default = pending["new_default"]
        if move:
            try:
                new_default.mkdir(parents=True, exist_ok=True)
                branches: dict[str, str] = {}
                for sub in ("onnx", "ov"):
                    src = current / sub
                    if not src.exists():
                        continue
                    dst = new_default / sub
                    branch_taken = self._move_with_safe_dest_handling(
                        src, dst, label=f"models/{sub}"
                    )
                    # Verify post-move: at least one file must now be at
                    # the destination tree (the source had files because
                    # _pending_models_dir_migration's _has_models gate
                    # already filtered for that).
                    if not any(dst.rglob("*")):
                        raise RuntimeError(
                            f"Post-move verification failed for models/{sub}: "
                            f"{dst} is empty after a {branch_taken!r} migration."
                        )
                    branches[sub] = branch_taken
                self.cfg["models_dir"] = str(new_default)
                if not config.save(self.cfg):
                    logger.error("Could not persist migrated models directory")
                messagebox.showinfo(
                    "Migration Complete",
                    f"Models moved to:\n{new_default}",
                    parent=self,
                )
                logger.info(
                    f"Migrated models dir from {current} to {new_default} "
                    f"(branches: {branches or 'no-op'})"
                )
            except Exception as e:
                messagebox.showerror(
                    "Migration Failed",
                    f"Could not move models:\n{e}\n\nUsing current location instead.",
                    parent=self,
                )
                logger.error(f"Models dir migration failed: {e}")
        else:
            self.cfg["models_dir"] = str(current)
            if not config.save(self.cfg):
                logger.error("Could not persist models directory")
            logger.info(f"User chose to keep models at {current} (config remapped)")

    # ── Ollama models directory migration ────────────────────────────────
    #
    # Off-profile relocation for the Ollama daemon's models directory.
    # The Ollama daemon is shared infrastructure: changing OLLAMA_MODELS
    # redirects every Ollama-talking app on the box (Open-WebUI,
    # Continue.dev, other LLM front-ends), so this migration is ALWAYS
    # opt-in and never silently applied. Regression-critical contract:
    # "Ollama migration must never auto-apply without explicit user
    #  consent because the daemon is shared infrastructure."

    def _migrate_ollama_models_dir_if_needed(self):
        """Offer to relocate the Ollama models directory off the user profile.

        Skips silently when:
          * ``OLLAMA_MODELS`` is already set (user has already customised).
          * ``cfg["ollama_offer_relocation"] == "declined"`` (user said No
            in a previous run; we never nag them again).
          * No blobs exist at ``%USERPROFILE%\\.ollama\\models`` (nothing
            to relocate, no need to bother).
          * Not on Windows (the roaming profile container failure mode
            this targets is Windows-only). On macOS the user can still
            relocate manually via the Settings row.

        Otherwise prompts Yes/No/Decide-later with a dialog that names
        the shared-daemon implication explicitly. Yes → ``setx
        OLLAMA_MODELS`` at USER scope + optional blob copy + "restart
        Ollama" non-blocking warning. No → persist
        ``cfg["ollama_offer_relocation"] = "declined"``.
        """
        pending = self._pending_ollama_models_dir_migration()
        if pending is None:
            return
        self._prompt_ollama_models_dir_migration(pending)

    def _pending_ollama_models_dir_migration(self) -> Optional[dict]:
        """Return migration context if Ollama relocation is offerable, else None.

        Returned dict shape:
            {"old_dir": Path, "new_target": Path}
        """
        if sys.platform != "win32":
            return None
        if os.environ.get("OLLAMA_MODELS"):
            return None
        if str(self.cfg.get("ollama_offer_relocation", "")).strip().lower() == "declined":
            return None
        old_dir = Path.home() / ".ollama" / "models"
        # Single Path.exists() check — keeps the warm path cheap so we
        # don't add measurable startup latency on machines without
        # Ollama installed.
        if not (old_dir / "blobs").exists():
            return None
        new_target = constrained_env._default_ollama_models_dir()
        return {"old_dir": old_dir, "new_target": new_target}

    def _prompt_ollama_models_dir_migration(self, pending: dict) -> None:
        """Show the single-purpose Ollama Yes/No/Decide-later dialog and apply."""
        old_dir = pending["old_dir"]
        new_target = pending["new_target"]
        answer = messagebox.askyesnocancel(
            "Move Ollama models directory?",
            "LocalAI noticed Ollama is storing model blobs inside your "
            "Windows user profile:\n"
            f"  {old_dir}\n\n"
            "If your user profile has limited space (common on corporate "
            "or managed machines and on any roamed profile), large "
            "Ollama models can fill it up and downloads will fail with "
            "\"no space left on device\" even when your disk shows "
            "plenty of room.\n\n"
            "LocalAI can point Ollama at a folder next to the app instead:\n"
            f"  {new_target}\n\n"
            "Important: OLLAMA_MODELS is shared infrastructure. After "
            "the Ollama daemon restarts, every other app on this PC that "
            "talks to Ollama (Open-WebUI, Continue.dev, other LLM "
            "front-ends) will also start reading models from the new "
            "directory.\n\n"
            "Yes    = Update OLLAMA_MODELS now (you'll be asked about "
            "copying existing blobs)\n"
            "No     = Leave Ollama where it is (don't ask again)\n"
            "Cancel = Decide later (ask me again next time)",
            parent=self,
        )
        if answer is None:  # Cancel — ask again next launch
            return
        if not answer:  # No — record declined so we never nag again
            self.cfg["ollama_offer_relocation"] = "declined"
            if not config.save(self.cfg):
                logger.error("Could not persist ollama_offer_relocation flag")
            logger.info("User declined Ollama models dir relocation; will not ask again")
            return
        # Yes — do the relocation
        self._apply_ollama_models_dir_migration(pending)

    def _apply_ollama_models_dir_migration(self, pending: dict) -> None:
        """Perform OLLAMA_MODELS relocation (setx USER scope + optional copy + restart prompt)."""
        old_dir = pending["old_dir"]
        new_target = pending["new_target"]
        try:
            new_target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "Relocation failed",
                f"Could not create {new_target}:\n{exc}\n\n"
                "OLLAMA_MODELS was not changed.",
                parent=self,
            )
            logger.error(f"Ollama relocation: mkdir failed for {new_target}: {exc}")
            return
        ok = self._set_user_env_var("OLLAMA_MODELS", str(new_target))
        if not ok:
            messagebox.showerror(
                "Relocation failed",
                "setx OLLAMA_MODELS failed. The Ollama models directory "
                "was not changed. Check the log for details.",
                parent=self,
            )
            return
        os.environ["OLLAMA_MODELS"] = str(new_target)
        logger.info(f"Set OLLAMA_MODELS={new_target} at USER scope")
        # Optional blob copy.
        if (old_dir / "blobs").exists():
            copy_choice = messagebox.askyesno(
                "Copy existing Ollama blobs?",
                f"Copy existing model blobs from\n  {old_dir}\nto\n  {new_target}?\n\n"
                "This can take several minutes per GB. If you skip, the "
                "old blobs stay where they are and Ollama re-downloads "
                "any model you ask it to pull.",
                parent=self,
            )
            if copy_choice:
                self._robocopy_ollama_blobs(old_dir, new_target)
        # Non-blocking restart instruction.
        messagebox.showwarning(
            "Restart Ollama",
            "OLLAMA_MODELS has been updated.\n\n"
            "For the change to take effect, restart Ollama:\n"
            "  1. Right-click the llama tray icon → Quit Ollama\n"
            "  2. Start menu → Ollama\n\n"
            "Other Ollama-using apps on this PC will also pick up the "
            "new directory after the daemon restarts.",
            parent=self,
        )

    def _set_user_env_var(self, name: str, value: str) -> bool:
        """Set a USER-scope environment variable via ``setx`` (Windows only).

        Returns True on success, False otherwise. ``setx`` writes the
        registry (HKCU\\Environment) and persists across sessions; it
        does NOT affect already-running processes — callers must
        instruct the user to restart the affected daemon.
        """
        if sys.platform != "win32":
            logger.warning(f"_set_user_env_var: skipping setx on non-Windows for {name}")
            return False
        try:
            result = subprocess.run(
                ["setx", name, value],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            if result.returncode != 0:
                logger.error(
                    f"setx {name} failed (rc={result.returncode}): {result.stderr.strip()}"
                )
                return False
            return True
        except Exception as exc:
            logger.error(f"setx {name} raised: {exc}")
            return False

    def _delete_user_env_var(self, name: str) -> bool:
        """Delete a USER-scope environment variable via ``reg delete`` (Windows only).

        Used by the Settings "clear OLLAMA_MODELS" path to restore the
        Ollama daemon's default behavior. ``reg delete`` returns exit
        code 1 when the value doesn't exist; we treat that as success
        because the user-visible outcome is the same.
        """
        if sys.platform != "win32":
            logger.warning(f"_delete_user_env_var: skipping reg delete on non-Windows for {name}")
            return False
        try:
            result = subprocess.run(
                ["reg", "delete", r"HKCU\Environment", "/F", "/V", name],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            # rc=0 → deleted; rc=1 → value didn't exist (still OK).
            if result.returncode in (0, 1):
                return True
            logger.error(
                f"reg delete {name} failed (rc={result.returncode}): {result.stderr.strip()}"
            )
            return False
        except Exception as exc:
            logger.error(f"reg delete {name} raised: {exc}")
            return False

    def _robocopy_ollama_blobs(self, src: Path, dst: Path) -> None:
        """Copy Ollama blobs from ``src`` to ``dst`` using robocopy."""
        if sys.platform != "win32":
            return
        try:
            result = subprocess.run(
                [
                    "robocopy", str(src), str(dst),
                    "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/R:1", "/W:1",
                ],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            # robocopy exit codes: 0/1 = success, 2 = extra files, 3 = both;
            # 8+ = at least one file failed.
            if result.returncode >= 8:
                messagebox.showwarning(
                    "Blob copy reported errors",
                    f"robocopy returned exit code {result.returncode}. "
                    "Some files may not have been copied. Check the log "
                    "for details.",
                    parent=self,
                )
                logger.warning(
                    f"robocopy {src} → {dst} rc={result.returncode}: {result.stderr.strip()}"
                )
            else:
                logger.info(f"robocopy {src} → {dst} succeeded (rc={result.returncode})")
        except Exception as exc:
            messagebox.showerror(
                "Blob copy failed",
                f"Could not copy Ollama blobs:\n{exc}",
                parent=self,
            )
            logger.error(f"robocopy {src} → {dst} raised: {exc}")

    # ── Unified storage-relocation first-run prompt ──────────────────────

    def _check_storage_relocation_on_startup(self):
        """Orchestrator called at app startup. Decides between unified vs single-purpose dialogs.

        When 2+ of (models, ComfyUI, Ollama) want to fire on the same
        launch, fold them into ONE composite dialog with three checkboxes
        (default-checked) instead of three sequential modal popups.
        Fallback to existing single-purpose dialogs when only one is
        pending. ComfyUI's lazy-init path still calls
        ``_migrate_comfyui_if_needed`` later for users who never hit the
        startup composite (idempotent — no-ops once migrated).

        Honors ``LOCALAI_SUPPRESS_STARTUP_PROMPTS=1`` as a non-interactive
        escape hatch so headless smoke tests and CI runs don't hang on a
        modal dialog that nobody can dismiss.
        """
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            logger.info(
                "Storage relocation check skipped "
                "(LOCALAI_SUPPRESS_STARTUP_PROMPTS=1)"
            )
            return
        try:
            prompting = self._collect_storage_relocation_prompts()
        except Exception as exc:
            logger.error(f"Storage relocation check failed: {exc}")
            return
        self._apply_storage_relocation_prompts(prompting)

    def _check_storage_relocation_on_startup_async(self):
        """Run startup storage probes off the Tk thread, then marshal dialogs back."""
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            logger.info(
                "Storage relocation check skipped "
                "(LOCALAI_SUPPRESS_STARTUP_PROMPTS=1)"
            )
            return

        def _worker():
            try:
                prompting = self._collect_storage_relocation_prompts()
            except Exception as exc:
                logger.error(f"Storage relocation check failed: {exc}")
                return
            self.after(0, lambda p=prompting: self._apply_storage_relocation_prompts(p))

        threading.Thread(
            target=_worker,
            name="StartupStorageRelocation",
            daemon=True,
        ).start()

    def _collect_storage_relocation_prompts(self) -> list[tuple[str, dict]]:
        """Return pending storage-relocation prompts; safe to call on a worker thread."""
        try:
            models_pending = self._pending_models_dir_migration()
            comfyui_pending = self._pending_comfyui_migration()
            ollama_pending = self._pending_ollama_models_dir_migration()
        except Exception as exc:
            logger.error(f"Storage relocation check failed: {exc}")
            return []

        # Auto-adopt the models default silently when there's nothing to
        # move — that case never warrants a dialog regardless of how
        # many other migrations are pending.
        prompting = []
        if models_pending is not None and models_pending.get("auto_adopt"):
            prompting.append(("models_auto_adopt", models_pending))
            models_pending = None

        if models_pending is not None:
            prompting.append(("models", models_pending))
        if comfyui_pending is not None:
            prompting.append(("comfyui", comfyui_pending))
        if ollama_pending is not None:
            prompting.append(("ollama", ollama_pending))
        return prompting

    def _apply_storage_relocation_prompts(self, prompting: list[tuple[str, dict]]) -> None:
        promptable: list[tuple[str, dict]] = []
        for key, pending in prompting:
            if key == "models_auto_adopt":
                self._apply_models_dir_auto_adopt(pending)
            else:
                promptable.append((key, pending))
        prompting = promptable

        if len(prompting) >= 2:
            self._prompt_unified_storage_relocation(prompting)
            return
        # 0 or 1 pending — fall through to the existing single-purpose dialogs.
        for key, pending in prompting:
            if key == "models":
                self._prompt_models_dir_migration(pending)
            elif key == "comfyui":
                self._prompt_comfyui_migration(pending)
            elif key == "ollama":
                self._prompt_ollama_models_dir_migration(pending)

    def _apply_models_dir_auto_adopt(self, pending: dict) -> None:
        self.cfg["models_dir"] = str(pending["new_default"])
        if not config.save(self.cfg):
            logger.error("Could not persist default models directory")
        logger.info(
            f"Models dir updated to {pending['new_default']} "
            "(old path was empty or missing)"
        )

    def _app_drive_is_removable(self) -> bool:
        """Return True if the app path lives on a removable drive (Windows).

        Uses ``GetDriveTypeW`` (returns 2 for removable, 3 for fixed,
        4 for network, 5 for CD-ROM, 6 for RAM disk). Returns False on
        non-Windows and on any probe failure — a removable note that
        doesn't fire is strictly better than a false-positive one.
        """
        if sys.platform != "win32":
            return False
        try:
            import ctypes
            app_root = Path(__file__).parent.parent
            drive = app_root.drive or ""
            if not drive:
                return False
            # GetDriveTypeW wants a trailing backslash.
            root = drive.rstrip("\\") + "\\"
            kernel32 = ctypes.windll.kernel32
            kernel32.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetDriveTypeW.restype = ctypes.c_uint
            return kernel32.GetDriveTypeW(root) == 2
        except Exception:
            return False

    def _app_drive_is_reboot_volatile(self) -> bool:
        """Return True when the app root lives on a removable / RAM / unknown drive.

        Used by the migration UI's "your target drive may be wiped between
        reboots" warning.  Wraps :func:`src.migration.is_reboot_volatile_drive`
        so callers don't need to know about the underlying Win32 API.
        """
        try:
            return _migration.is_reboot_volatile_drive(Path(__file__).parent.parent)
        except Exception:
            return False

    # ── v5.3.10 self-healing scans ───────────────────────────────────────

    def _resume_pending_migration_if_any_async(self) -> None:
        """Scan for resumable migration state off the UI thread."""
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return

        def _worker() -> None:
            try:
                state = self._pending_migration_state_for_prompt()
            except Exception as exc:
                logger.error(f"Resume-migration scan failed: {exc}")
                return
            if state is not None:
                self.after(0, lambda s=state: self._prompt_migration_resume(s))

        threading.Thread(
            target=_worker,
            name="StartupMigrationResumeScan",
            daemon=True,
        ).start()

    def _resume_pending_migration_if_any(self) -> None:
        """If a prior launch left a migration state file, prompt the user.

        Honors ``LOCALAI_SUPPRESS_STARTUP_PROMPTS=1``.  PENDING / PRE_FLIGHT
        states silently fall through to the normal startup flow; only the
        COPYING / VERIFYING phases (where a partial target exists) prompt.
        """
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return
        try:
            state = self._pending_migration_state_for_prompt()
            if state is not None:
                self._prompt_migration_resume(state)
        except Exception as exc:
            logger.error(f"Resume-migration scan failed: {exc}")

    def _pending_migration_state_for_prompt(self):
        app_root = Path(__file__).parent.parent
        state = _migration.find_resumable_state(app_root)
        if state is None:
            return None
        phase = _migration.MigrationPhase(state.phase)
        if phase not in _migration.USER_PROMPT_RESUME_PHASES:
            # PENDING / PRE_FLIGHT — silently drop the stale lock file,
            # nothing was written to the target yet.
            _migration.clear_state(app_root)
            logger.info(
                f"Discarded pre-copy migration state ({state.kind}, phase={state.phase})"
            )
            return None
        return state

    def _prompt_migration_resume(self, state) -> None:
        """Three-way Resume / Roll back / Decide-later dialog."""
        try:
            answer = messagebox.askyesnocancel(
                "Unfinished migration found",
                "A previous attempt to move "
                f"{state.kind} was interrupted at the "
                f"{state.phase.replace('_', ' ')} stage.\n\n"
                f"From: {state.source}\n"
                f"To:   {state.target}\n\n"
                "Yes    = Resume (recommended)\n"
                "No     = Roll back (discard partial copy, keep existing files)\n"
                "Cancel = Decide later",
                parent=self,
            )
        except Exception as exc:
            logger.error(f"Resume-migration dialog failed: {exc}")
            return
        if answer is None:
            return  # Decide later
        if answer is False:
            # Roll back: delete partial target + clear state.
            try:
                target = Path(state.target)
                if target.exists():
                    import shutil as _sh
                    _sh.rmtree(str(target), ignore_errors=True)
                _migration.clear_state(Path(__file__).parent.parent)
                messagebox.showinfo(
                    "Rolled back",
                    "The interrupted migration was rolled back. Your existing "
                    f"{state.kind} files are still at:\n  {state.source}",
                    parent=self,
                )
            except Exception as exc:
                logger.error(f"Resume rollback failed: {exc}")
            return
        # Resume — re-derive plan from state and run the engine.
        try:
            kind_to_sentinel = {
                "comfyui": _migration.has_comfyui_sentinel,
                "models": _migration.has_models_sentinel,
                "ollama": _migration.has_ollama_sentinel,
            }
            sentinel = kind_to_sentinel.get(state.kind, lambda _p: True)
            plan = _migration.MigrationPlan(
                kind=state.kind,
                source=Path(state.source),
                target=Path(state.target),
                sentinel_check=sentinel,
                is_ollama=(state.kind == "ollama"),
            )
            self._run_migration_engine_modal(plan, resume_from=state)
        except Exception as exc:
            logger.error(f"Resume execution failed: {exc}")
            messagebox.showerror(
                "Resume failed",
                f"Could not resume the migration:\n{exc}",
                parent=self,
            )

    def _heal_orphan_ollama_blobs_async(self) -> None:
        """Scan orphan Ollama blobs off the UI thread, then prompt on Tk."""
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return

        def _worker() -> None:
            try:
                finding = self._orphan_ollama_blob_finding_for_prompt()
            except Exception as exc:
                logger.error(f"Orphan-blob heal scan failed: {exc}")
                return
            if finding is not None:
                root, blob_count = finding
                self.after(
                    0,
                    lambda r=root, c=blob_count: self._prompt_orphan_blob_recovery(r, c),
                )

        threading.Thread(
            target=_worker,
            name="StartupOrphanOllamaBlobScan",
            daemon=True,
        ).start()

    def _heal_orphan_ollama_blobs(self) -> None:
        """Detect Ollama directories with blobs/ but no manifests/ and offer Recover/Roll-back.

        This is the v5.3.10 self-healing path for today's disaster: a partial
        Ollama migration that left blobs at the target but no manifests, so
        Ollama itself believed it had nothing while the disk was still
        wedged.  We never silently delete blobs — we always present the user
        with the choice.

        Honors ``LOCALAI_SUPPRESS_STARTUP_PROMPTS=1``.  The actual scan logic
        lives in :func:`src.migration.scan_orphan_ollama_blobs` so the
        Settings → Storage → "Verify & Repair" button can share it.
        """
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return
        try:
            finding = self._orphan_ollama_blob_finding_for_prompt()
            if finding is None:
                return
            root, blob_count = finding
            self._prompt_orphan_blob_recovery(root, blob_count)
        except Exception as exc:
            logger.error(f"Orphan-blob heal scan failed: {exc}")

    def _orphan_ollama_blob_finding_for_prompt(self) -> tuple[Path, int] | None:
        findings = _migration.scan_orphan_ollama_blobs()
        if not findings:
            return None
        # Only prompt for the first orphan we find per startup.
        return findings[0]

    def _prompt_orphan_blob_recovery(self, root: Path, blob_count: int) -> None:
        """Show a Recover / Discard / Decide-later dialog for orphan Ollama blobs."""
        try:
            answer = messagebox.askyesnocancel(
                "Orphan Ollama blobs found",
                f"LocalAI found {blob_count} Ollama blobs at:\n"
                f"  {root}\n\n"
                "…but no matching model manifests. This usually means a "
                "previous migration was interrupted before the manifests "
                "were copied across, so Ollama can't see the models even "
                "though the blobs are taking up disk space.\n\n"
                "Yes    = Try to recover from the original Ollama directory\n"
                "No     = Delete the orphan blobs (frees space, irreversible)\n"
                "Cancel = Decide later",
                parent=self,
            )
        except Exception as exc:
            logger.error(f"Orphan-blob dialog failed: {exc}")
            return
        if answer is None:
            return
        if answer is False:
            confirm = messagebox.askyesno(
                "Delete orphan blobs?",
                f"Delete {blob_count} orphan blobs at:\n  {root / 'blobs'}\n\n"
                "This cannot be undone.",
                icon="warning",
                parent=self,
            )
            if not confirm:
                return
            try:
                import shutil as _sh
                _sh.rmtree(str(root / "blobs"), ignore_errors=True)
                messagebox.showinfo(
                    "Orphan blobs removed",
                    f"Removed orphan blobs at:\n  {root / 'blobs'}",
                    parent=self,
                )
                logger.info(f"Removed orphan Ollama blobs at {root / 'blobs'}")
            except Exception as exc:
                logger.error(f"Orphan-blob delete failed: {exc}")
            return
        # Recover: re-run the Ollama migration from the home location.
        home_models = Path.home() / ".ollama" / "models"
        if not (home_models / "blobs").is_dir():
            messagebox.showwarning(
                "Source not found",
                f"Could not find a source Ollama directory at:\n  {home_models}\n\n"
                "Nothing to recover from.",
                parent=self,
            )
            return
        try:
            plan = _migration.MigrationPlan(
                kind="ollama",
                source=home_models,
                target=root,
                sentinel_check=_migration.has_ollama_sentinel,
                is_ollama=True,
            )
            self._run_migration_engine_modal(plan)
        except Exception as exc:
            logger.error(f"Orphan-blob recovery failed: {exc}")
            messagebox.showerror(
                "Recovery failed",
                f"Could not recover:\n{exc}",
                parent=self,
            )

    def _heal_legacy_onnx_paths_async(self) -> None:
        """Scan legacy ONNX paths off the UI thread, then show any info dialog on Tk."""
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return

        def _worker() -> None:
            try:
                offenders = self._legacy_onnx_offenders_for_startup()
            except Exception as exc:
                logger.error(f"Legacy-ONNX heal scan failed: {exc}")
                return
            if offenders:
                self.after(0, lambda o=offenders: self._show_legacy_onnx_paths(o))

        threading.Thread(
            target=_worker,
            name="StartupLegacyOnnxScan",
            daemon=True,
        ).start()

    def _heal_legacy_onnx_paths(self) -> None:
        """Detect catalog ONNX entries pointing at %LOCALAPPDATA%\\LocalAI\\* and offer cleanup.

        v5.3.10 self-healing for the second arm of today's recovery: 14.77 GB
        of ONNX models inside the user profile that the catalog was pointing
        at directly.  We log + show a single info dialog naming the affected
        rows; the user manually migrates them from Settings → Models.

        Honors ``LOCALAI_SUPPRESS_STARTUP_PROMPTS=1``.  The actual scan logic
        lives in :func:`src.migration.scan_legacy_onnx_paths` so the
        Settings → Storage → "Verify & Repair" button can share it.
        """
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return
        try:
            offenders = self._legacy_onnx_offenders_for_startup()
            if offenders:
                self._show_legacy_onnx_paths(offenders)
        except Exception as exc:
            logger.error(f"Legacy-ONNX heal scan failed: {exc}")

    def _legacy_onnx_offenders_for_startup(self) -> list[dict]:
        if sys.platform != "win32":
            return []
        return _migration.scan_legacy_onnx_paths(
            catalog_entries=self._catalog_entries_for_reconciliation(),
        )

    def _show_legacy_onnx_paths(self, offenders: list[dict]) -> None:
        top = "\n".join(
            f"{e.get('name') or e.get('id') or '?'}  ({e.get('path')})"
            for e in offenders[:5]
        )
        more = f"\n…and {len(offenders) - 5} more" if len(offenders) > 5 else ""
        messagebox.showinfo(
            "Models found inside your user profile",
            "LocalAI found models still living under your Windows user "
            "profile (LOCALAPPDATA\\LocalAI). On cloud VMs and managed "
            "machines that drive is often size-capped, which can break "
            "downloads later.\n\n"
            f"Affected:\n{top}{more}\n\n"
            "Open Settings → Models → Storage to relocate them "
            "alongside the app.",
            parent=self,
        )
        logger.info(
            f"Legacy ONNX scan found {len(offenders)} catalog entries under user profile"
        )

    def _process_scheduled_deletes_after_startup(self) -> None:
        """Drain any ``*.deleteme`` directories left over from a prior migration."""
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return
        try:
            app_root = Path(__file__).parent.parent
            deleted = _migration.process_scheduled_deletes(app_root)
            if deleted:
                logger.info(f"Scheduled-delete cleanup removed: {deleted}")
        except Exception as exc:
            logger.error(f"Scheduled-delete cleanup failed: {exc}")

    def _process_scheduled_deletes_after_startup_async(self) -> None:
        """Drain scheduled deletes without blocking initial tab navigation."""
        if os.environ.get("LOCALAI_SUPPRESS_STARTUP_PROMPTS") == "1":
            return

        def _worker() -> None:
            self._process_scheduled_deletes_after_startup()

        threading.Thread(
            target=_worker,
            name="StartupScheduledDeleteCleanup",
            daemon=True,
        ).start()

    # ── Verify & Repair (user-triggered Settings → Storage button) ───────

    def _catalog_entries_for_reconciliation(self) -> list[dict]:
        """Return the catalog rows used by storage reconciliation scans."""
        explicit_entries = getattr(self, "catalog_entries", None)
        if explicit_entries is not None:
            return list(explicit_entries)
        return list(getattr(self, "_catalog_models", []) or [])

    def _collect_verify_repair_findings(self) -> list:
        """Run every reconciliation scan and return a flat list of Findings.

        Shares the underlying scanners with the startup self-healing hooks
        (``scan_orphan_ollama_blobs``, ``scan_legacy_onnx_paths``,
        ``scan_config_coherence``, ``scan_disk_space``,
        ``find_resumable_state``).  The Fix / Ignore callables here re-use
        the same handlers the startup hooks call so the user gets identical
        recovery behaviour from the button.
        """
        findings: list = []
        cfg = getattr(self, "cfg", {}) or {}
        app_root = Path(__file__).parent.parent

        # 1. Orphan Ollama blobs.
        try:
            for root, blob_count in _migration.scan_orphan_ollama_blobs():
                findings.append(_migration.Finding(
                    kind="orphan_blobs",
                    severity=_migration.SEVERITY_ACTION,
                    summary=f"Ollama blobs without manifests at {root}",
                    detail=(
                        f"{blob_count} blob file{'s' if blob_count != 1 else ''} "
                        "present but no model manifests — Ollama can't see "
                        "these models even though the disk space is used."
                    ),
                    fix_callable=(lambda r=root, c=blob_count:
                                  self._prompt_orphan_blob_recovery(r, c)),
                    fix_label="Fix",
                    ignore_label="Ignore",
                ))
        except Exception as exc:
            logger.error(f"Verify & Repair: orphan-blob scan failed: {exc}")

        # 2. Legacy ONNX paths under %LOCALAPPDATA%\LocalAI.
        try:
            catalog_entries = self._catalog_entries_for_reconciliation()
            for entry in _migration.scan_legacy_onnx_paths(
                catalog_entries=catalog_entries,
            ):
                p = entry.get("path")
                name = entry.get("name") or entry.get("id") or "?"
                findings.append(_migration.Finding(
                    kind="legacy_onnx",
                    severity=_migration.SEVERITY_ACTION,
                    summary=f"Legacy ONNX directory: {name}",
                    detail=(
                        f"{p} lives under your Windows user profile. "
                        "On cloud VMs and managed machines that drive is "
                        "often size-capped."
                    ),
                    fix_callable=(lambda src=p: self._fix_legacy_onnx_path(src)),
                    fix_label="Move to canonical",
                    ignore_label="Ignore",
                ))
        except Exception as exc:
            logger.error(f"Verify & Repair: legacy-ONNX scan failed: {exc}")

        # 3. Config-vs-filesystem coherence.
        try:
            for entry in _migration.scan_config_coherence(cfg):
                key = entry["key"]
                value = entry["value"]
                reason = entry["reason"]
                findings.append(_migration.Finding(
                    kind="config_coherence",
                    severity=_migration.SEVERITY_ACTION,
                    summary=f"Config path \"{key}\" is broken",
                    detail=f"{value} — {reason}",
                    fix_callable=(lambda k=key: self._repair_config_path(k)),
                    fix_label="Repair",
                    ignore_label="Ignore",
                ))
        except Exception as exc:
            logger.error(f"Verify & Repair: config-coherence scan failed: {exc}")

        # 4. Disk-space audit.
        try:
            for entry in _migration.scan_disk_space(cfg):
                drive = entry.get("drive") or "?"
                free_gb = entry.get("free_gb", 0.0)
                paths = entry.get("paths") or []
                findings.append(_migration.Finding(
                    kind="disk_space",
                    severity=_migration.SEVERITY_WARN,
                    summary=f"Low free space on {drive} ({free_gb:.1f} GB free)",
                    detail=(
                        "Configured paths on this drive: "
                        + ", ".join(paths)
                        + ". A future download or migration may fail."
                    ),
                ))
        except Exception as exc:
            logger.error(f"Verify & Repair: disk-space scan failed: {exc}")

        # 5. Legacy / stale HuggingFace cache locations.
        try:
            for entry in _migration.scan_legacy_hf_cache(app_root=app_root):
                src = entry["source"]
                size_gb = float(entry.get("size_gb", 0.0))
                label = entry.get("label", str(src))
                findings.append(_migration.Finding(
                    kind="legacy_hf_cache",
                    severity=_migration.SEVERITY_ACTION,
                    summary=f"{label} ({size_gb:.2f} GB)",
                    detail=(
                        f"{src} should live at {entry['destination']} so it "
                        "follows the v5.3.10 cache redirection and is not "
                        "trapped inside your roaming user profile."
                    ),
                    fix_callable=(lambda s=src, d=entry["destination"]:
                                  self._fix_legacy_hf_cache(s, d)),
                    fix_label="Migrate",
                    ignore_label="Ignore",
                ))
        except Exception as exc:
            logger.error(f"Verify & Repair: legacy-hf-cache scan failed: {exc}")

        # 6. Pending migration state file.
        try:
            state = _migration.find_resumable_state(app_root)
            if state is not None:
                findings.append(_migration.Finding(
                    kind="pending_migration",
                    severity=_migration.SEVERITY_ACTION,
                    summary="Previous move was interrupted",
                    detail=(
                        f"Phase: {state.phase}. Source: {state.source}. "
                        f"Target: {state.target}."
                    ),
                    fix_callable=self._resume_pending_migration_if_any,
                    fix_label="Resume",
                    ignore_label="Decide later",
                ))
        except Exception as exc:
            logger.error(f"Verify & Repair: pending-migration scan failed: {exc}")

        return findings

    def _fix_legacy_onnx_path(self, source: Path) -> None:
        """Offer to move a legacy ONNX directory to ``<models_dir>\\onnx\\<name>``."""
        try:
            source = Path(source)
            models_dir = Path(self.cfg.get("models_dir", ""))
            if not models_dir or not str(models_dir):
                messagebox.showerror(
                    "Models directory not configured",
                    "Configure a Models directory in Settings before moving "
                    "legacy ONNX content.",
                    parent=self,
                )
                return
            target = models_dir / "onnx" / source.name
            if target.exists():
                messagebox.showinfo(
                    "Target already exists",
                    f"A directory at {target} already exists. Remove or rename "
                    "it before moving the legacy copy.",
                    parent=self,
                )
                return
            confirm = messagebox.askyesno(
                "Move legacy ONNX directory?",
                f"Move:\n  {source}\nto:\n  {target}\n\n"
                "The original location will be replaced with a `.deleteme` "
                "rename and cleaned up on the next launch.",
                parent=self,
            )
            if not confirm:
                return
            plan = _migration.MigrationPlan(
                kind="models",
                source=source,
                target=target,
                sentinel_check=lambda p: Path(p).is_dir(),
            )
            self._run_migration_engine_modal(plan)
        except Exception as exc:
            logger.error(f"Legacy ONNX move failed: {exc}")
            messagebox.showerror(
                "Move failed",
                f"Could not move {source}:\n{exc}",
                parent=self,
            )

    def _fix_legacy_hf_cache(self, source: Path, destination: Path) -> None:
        """Migrate a legacy HF cache directory into the canonical location.

        The destination (``<app>/.cache/huggingface``) is often pre-created (and
        empty) by the v5.3.10 batch-file environment block before the user even
        opens Settings → Verify & Repair.  Treat that empty shell as "target
        does not exist" by scrubbing the empty subdirs first, then run the
        standard MigrationEngine.

        When the destination already has real content we cannot rename the
        source on top of it, but refusing the merge (the pre-v5.3.7 behavior)
        stranded the user with a dead-end dialog asking them to run robocopy
        from cmd — which they cannot do for permissions / locked-down profile
        reasons.  Offer to perform that exact safe merge in-app
        (:func:`_migration.safe_merge_directory`, which mirrors
        ``robocopy /E /XO``: destination files always win on conflict, source
        files newer than destination overwrite, source files missing in dest
        are copied across).  After a clean merge we remove the source dir
        outright — it lives outside ``app_root`` so the ``.deleteme`` sweeper
        cannot reach it, and leaving it as a half-empty husk re-triggers the
        same Verify & Repair finding on next launch.
        """
        try:
            source = Path(source)
            destination = Path(destination)
            if not source.exists() or not source.is_dir():
                return
            if destination.exists():
                try:
                    has_files = any(
                        p.is_file() for p in destination.rglob("*")
                    )
                except OSError:
                    has_files = True
                if has_files:
                    self._merge_legacy_hf_cache_into_populated_dest(
                        source, destination
                    )
                    return
                try:
                    import shutil
                    shutil.rmtree(destination)
                except OSError as exc:
                    logger.error(f"Could not clear empty destination {destination}: {exc}")
                    messagebox.showerror(
                        "Cannot prepare destination",
                        f"{destination} could not be cleared: {exc}",
                        parent=self,
                    )
                    return
            confirm = messagebox.askyesno(
                "Migrate legacy HuggingFace cache?",
                f"Move:\n  {source}\nto:\n  {destination}\n\n"
                "Same-drive moves are nearly instant.  The original location "
                "will be replaced with a `.deleteme` rename and cleaned up "
                "on the next launch.",
                parent=self,
            )
            if not confirm:
                return
            plan = _migration.MigrationPlan(
                kind="hf_cache",
                source=source,
                target=destination,
                sentinel_check=lambda p: Path(p).is_dir(),
            )
            self._run_migration_engine_modal(plan)
        except Exception as exc:
            logger.error(f"Legacy HF cache migration failed: {exc}")
            messagebox.showerror(
                "Migration failed",
                f"Could not move {source}:\n{exc}",
                parent=self,
            )

    def _merge_legacy_hf_cache_into_populated_dest(
        self, source: Path, destination: Path
    ) -> None:
        """Confirm + perform a safe robocopy-/XO-equivalent merge in Python.

        Asks the user once, then calls :func:`_migration.safe_merge_directory`
        (destination always wins on conflict).  On success the source dir is
        removed so the Verify & Repair row clears on the next scan; on any
        error the source is left alone and the user sees the count of
        failures so they can investigate before re-running.
        """
        try:
            try:
                src_bytes = sum(
                    f.stat().st_size
                    for f in source.rglob("*")
                    if f.is_file()
                )
            except OSError:
                src_bytes = 0
            src_mb = src_bytes / 1_048_576.0
            src_label = (
                f"{src_mb:.1f} MB" if src_mb < 1024 else f"{src_mb / 1024:.2f} GB"
            )
            confirm = messagebox.askyesno(
                "Merge stranded HuggingFace cache?",
                f"Source:\n  {source}\n  ({src_label})\n\n"
                f"Destination:\n  {destination}\n  (already has files)\n\n"
                "Mirrors `robocopy /E /XO`:\n"
                "  • Files only in source → copied across.\n"
                "  • Files in both → destination wins (anything you "
                "downloaded since is kept).\n"
                "  • Source files newer than destination → overwrite "
                "destination.\n\n"
                "After a clean merge the source directory is removed so "
                "this finding clears.  Proceed?",
                parent=self,
            )
            if not confirm:
                return
            result = _migration.safe_merge_directory(source, destination)
            copied = int(result.get("copied", 0))
            skipped = int(result.get("skipped", 0))
            errors = list(result.get("errors", []))
            if errors:
                preview = "\n  ".join(errors[:5])
                more = (
                    f"\n  …and {len(errors) - 5} more"
                    if len(errors) > 5
                    else ""
                )
                logger.warning(
                    f"Safe-merge of {source} -> {destination} hit "
                    f"{len(errors)} error(s); source left in place."
                )
                messagebox.showwarning(
                    "Merge finished with errors",
                    f"Copied {copied}, preserved {skipped}, "
                    f"failed {len(errors)}.\n\n"
                    f"Errors:\n  {preview}{more}\n\n"
                    "Source directory was NOT removed so you can re-run "
                    "after investigating.",
                    parent=self,
                )
                return
            try:
                import shutil
                shutil.rmtree(str(source))
            except OSError as exc:
                logger.warning(
                    f"Safe-merge succeeded but could not remove source "
                    f"{source}: {exc}"
                )
                messagebox.showwarning(
                    "Merge succeeded, source not removed",
                    f"Copied {copied}, preserved {skipped}.\n\n"
                    f"Source directory could not be removed: {exc}\n\n"
                    "Delete it manually when convenient.",
                    parent=self,
                )
                return
            logger.info(
                f"Safe-merge of {source} -> {destination} complete: "
                f"copied={copied}, preserved={skipped}; source removed."
            )
            messagebox.showinfo(
                "Cache merged",
                f"Copied {copied} new/updated files into "
                f"{destination}.\nPreserved {skipped} destination files "
                "(your downloads since the original move).\n\n"
                "Source directory removed.",
                parent=self,
            )
        except Exception as exc:
            logger.error(f"Safe-merge of {source} -> {destination} failed: {exc}")
            messagebox.showerror(
                "Merge failed",
                f"Could not merge {source} into {destination}:\n{exc}",
                parent=self,
            )

    def _repair_config_path(self, key: str) -> None:
        """Rewrite a phantom config path back to its canonical default."""
        try:
            if key == "comfyui_dir":
                app_root = Path(__file__).parent.parent
                canonical = app_root / "ComfyUI"
                self.cfg["comfyui_dir"] = str(canonical)
            elif key == "models_dir":
                app_root = Path(__file__).parent.parent
                canonical = app_root / "models"
                canonical.mkdir(parents=True, exist_ok=True)
                self.cfg["models_dir"] = str(canonical)
            else:
                logger.warning(f"Repair config: unknown key {key}")
                return
            if config.save(self.cfg):
                messagebox.showinfo(
                    "Config repaired",
                    f"Reset \"{key}\" to its canonical location:\n  "
                    f"{self.cfg.get(key)}",
                    parent=self,
                )
            else:
                messagebox.showerror(
                    "Could not save config",
                    "config.save() returned False — check the log for details.",
                    parent=self,
                )
        except Exception as exc:
            logger.error(f"Repair config {key} failed: {exc}")
            messagebox.showerror(
                "Repair failed",
                f"Could not repair {key}:\n{exc}",
                parent=self,
            )

    def _open_verify_repair_dialog(self) -> None:
        """User-triggered handler for Settings → Storage → "Verify & Repair"."""
        try:
            findings = self._collect_verify_repair_findings()
        except Exception as exc:
            logger.error(f"Verify & Repair collection failed: {exc}")
            messagebox.showerror(
                "Verify & Repair failed",
                f"Could not run the scan:\n{exc}",
                parent=self,
            )
            return
        self._render_verify_repair_dialog(findings)

    def _render_verify_repair_dialog(self, findings: list) -> None:
        """Show a CTkToplevel with one row per Finding (or an all-clear card)."""
        try:
            win = ctk.CTkToplevel(self)
            win.title("Verify & Repair")
            win.geometry("760x520")
            win.transient(self)
            try:
                win.grab_set()
            except Exception:
                pass
            win.grid_columnconfigure(0, weight=1)
            win.grid_rowconfigure(1, weight=1)

            header = ctk.CTkFrame(win, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
            header.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                header,
                text="Verify & Repair",
                font=ctk.CTkFont(size=18, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                header,
                text=(
                    "Scans every storage location and surfaces anything LocalAI "
                    "would normally only catch at startup. Fixes are explicit — "
                    "nothing is deleted or rewritten without your click."
                ),
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=11),
                anchor="w",
                wraplength=720,
                justify="left",
            ).grid(row=1, column=0, sticky="w", pady=(2, 0))

            body = ctk.CTkScrollableFrame(win, fg_color="transparent")
            body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(4, 8))
            body.grid_columnconfigure(0, weight=1)

            if not findings:
                clean = ctk.CTkFrame(body, corner_radius=10, fg_color=INPUT_SURFACE)
                clean.grid(row=0, column=0, sticky="ew", pady=8)
                clean.grid_columnconfigure(0, weight=1)
                ctk.CTkLabel(
                    clean,
                    text="All checks passed — no action needed.",
                    font=ctk.CTkFont(size=13, weight="bold"),
                    anchor="w",
                ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
                ctk.CTkLabel(
                    clean,
                    text=(
                        "Orphan-blob check, legacy ONNX scan, config coherence, "
                        "disk-space audit and pending-migration state are all "
                        "healthy."
                    ),
                    text_color=TEXT_MUTED,
                    font=ctk.CTkFont(size=11),
                    anchor="w",
                    wraplength=680,
                    justify="left",
                ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))
            else:
                for idx, f in enumerate(findings):
                    self._render_finding_row(body, idx, win, f)

            footer = ctk.CTkFrame(win, fg_color="transparent")
            footer.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
            footer.grid_columnconfigure(0, weight=1)
            ctk.CTkButton(
                footer,
                text="Close",
                command=win.destroy,
                **self._outline_button_style(),
            ).grid(row=0, column=1, sticky="e")

            # Fix All — only shown when there is more than one auto-fixable
            # finding, otherwise the per-row "Fix" buttons are already a
            # one-click flow.  Walks every fix_callable in severity order
            # (action → warn → info) then re-runs the scan so cleared
            # findings disappear without the user having to re-open the
            # dialog.  Addresses the "requires multiple Settings visits and
            # 4 manual button clicks" complaint that v5.4.7 left open.
            fixable = [f for f in findings if getattr(f, "fix_callable", None)]
            if len(fixable) >= 2:
                def _do_fix_all(items=tuple(fixable), w=win):
                    failures: list[str] = []
                    for f in items:
                        try:
                            f.fix_callable()
                        except Exception as exc:
                            failures.append(f"• {f.summary}: {exc}")
                            logger.warning(
                                f"Fix All: {f.summary!r} fixer raised: {exc}"
                            )
                    try:
                        w.destroy()
                    except Exception:
                        pass
                    if failures:
                        try:
                            messagebox.showwarning(
                                "Fix All — partial",
                                "Some fixes raised errors:\n\n"
                                + "\n".join(failures[:8])
                                + (
                                    f"\n\n… and {len(failures) - 8} more"
                                    if len(failures) > 8
                                    else ""
                                )
                                + "\n\nThe scan will reopen so you can see "
                                "what is still outstanding.",
                                parent=self,
                            )
                        except Exception:
                            pass
                    try:
                        self.after(50, self._open_verify_repair_dialog)
                    except Exception:
                        pass

                ctk.CTkButton(
                    footer,
                    text=f"Fix All ({len(fixable)})",
                    command=_do_fix_all,
                    **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
                ).grid(row=0, column=2, sticky="e", padx=(8, 0))
        except Exception as exc:
            logger.error(f"Verify & Repair dialog failed: {exc}")
            messagebox.showerror(
                "Verify & Repair",
                f"Could not show the report:\n{exc}",
                parent=self,
            )

    def _render_finding_row(self, parent, row_index: int, dialog, finding) -> None:
        """Render a single Finding row inside the Verify & Repair dialog."""
        sev = getattr(finding, "severity", _migration.SEVERITY_WARN)
        accent = (
            self._IG_DANGER
            if sev == _migration.SEVERITY_ACTION
            else "#b8870a"
            if sev == _migration.SEVERITY_WARN
            else self._IG_HERO
        )
        card = ctk.CTkFrame(
            parent,
            corner_radius=10,
            fg_color=INPUT_SURFACE,
            border_width=1,
            border_color=BORDER_STRONG,
        )
        card.grid(row=row_index, column=0, sticky="ew", pady=6)
        card.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(card, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        title_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            title_row,
            text=finding.summary,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
            text_color=accent,
            wraplength=560,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        if finding.detail:
            ctk.CTkLabel(
                card,
                text=finding.detail,
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=11),
                anchor="w",
                wraplength=680,
                justify="left",
            ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))

        if finding.fix_callable or finding.ignore_callable:
            btn_row = ctk.CTkFrame(card, fg_color="transparent")
            btn_row.grid(row=2, column=0, sticky="e", padx=12, pady=(0, 10))
            if finding.fix_callable:
                def _do_fix(fn=finding.fix_callable, w=dialog):
                    try:
                        fn()
                    finally:
                        try:
                            w.destroy()
                        except Exception:
                            pass
                        # Re-open the dialog with refreshed findings so the
                        # user can see whether the issue cleared.
                        try:
                            self.after(50, self._open_verify_repair_dialog)
                        except Exception:
                            pass
                ctk.CTkButton(
                    btn_row,
                    text=finding.fix_label or "Fix",
                    command=_do_fix,
                    **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
                ).pack(side="right", padx=(8, 0))
            if finding.ignore_callable or finding.ignore_label:
                def _do_ignore(fn=finding.ignore_callable):
                    try:
                        if fn:
                            fn()
                    except Exception as exc:
                        logger.warning(f"Ignore callback raised: {exc}")
                ctk.CTkButton(
                    btn_row,
                    text=finding.ignore_label or "Ignore",
                    command=_do_ignore,
                    **self._outline_button_style(),
                ).pack(side="right")

    # ── Uncatalogued reconciliation (Settings → Models maintenance) ──────

    def _list_installed_ollama_tags(self) -> tuple[list[dict], Optional[str]]:
        """Return ``(rows, error)`` from ``ollama list``.

        Each row: ``{"tag": str, "size": str, "id": str}``.  On any failure
        an error string is returned and ``rows`` is empty.  We shell out to
        ``ollama list`` rather than the HTTP API so the size column matches
        what the user sees in their terminal.
        """
        try:
            proc = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            return [], "Ollama daemon not running — start it from Settings → Ollama to reconcile."
        except subprocess.TimeoutExpired:
            return [], "Ollama daemon timed out — try again in a moment."
        except OSError as exc:
            return [], f"Could not run ollama list: {exc}"
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            first = err[0] if err else f"exit code {proc.returncode}"
            return [], "Ollama daemon not running — start it from Settings → Ollama to reconcile." if "connect" in first.lower() else f"ollama list failed: {first}"
        rows: list[dict] = []
        for raw in (proc.stdout or "").splitlines():
            line = raw.rstrip()
            if not line:
                continue
            # Header line: "NAME    ID    SIZE    MODIFIED" — skip.
            up = line.upper()
            if up.startswith("NAME") and "SIZE" in up:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tag = parts[0]
            size = ""
            ident = ""
            # Conservative column split: NAME ID SIZE_NUM SIZE_UNIT MODIFIED…
            if len(parts) >= 4:
                ident = parts[1]
                size = parts[2] + (" " + parts[3] if not parts[3].isdigit() else "")
            elif len(parts) == 3:
                size = parts[1] + " " + parts[2]
            rows.append({"tag": tag, "size": size, "id": ident})
        return rows, None

    def _build_uncatalogued_panel(self, parent, *, row_index: int) -> None:
        """Build the Settings → Models "Installed but not in catalog" card.

        Reconciles installed Ollama tags and on-disk ONNX directories
        against ``models_catalog.json``.  Never auto-deletes; every delete
        requires explicit confirmation.
        """
        card = ctk.CTkFrame(
            parent,
            corner_radius=14,
            fg_color=SURFACE_CARD,
            border_width=1,
            border_color=BORDER_STRONG,
        )
        card.grid(row=row_index, column=0, sticky="ew", padx=20, pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Installed but not in catalog",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text=(
                "These models are installed on this machine but are not listed in your models_catalog.json. "
                "Use Delete to free space, or leave them alone if you use them via "
                "another tool. Nothing is removed without an explicit confirmation."
            ),
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
            wraplength=720,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        body.grid_columnconfigure(0, weight=1)

        self._uncatalogued_body_ref = body
        self._refresh_uncatalogued_lists_async(body)

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="e", padx=14, pady=(0, 12))
        ctk.CTkButton(
            actions,
            text="Refresh",
            command=lambda: self._refresh_uncatalogued_lists_async(self._uncatalogued_body_ref),
            **self._outline_button_style(),
        ).pack(side="right")

    def _refresh_uncatalogued_lists_async(self, body) -> None:
        """Refresh the uncatalogued model panel without blocking Settings navigation."""
        if not self._widget_is_alive(body):
            return
        generation = self.__dict__.get("_uncatalogued_refresh_generation", 0) + 1
        self._uncatalogued_refresh_generation = generation
        self._show_uncatalogued_loading(body)

        def _worker() -> None:
            try:
                state = self._collect_uncatalogued_lists()
            except Exception as exc:
                logger.error(f"Uncatalogued model scan failed: {exc}")
                state = {"fatal_error": str(exc)}
            self.after(
                0,
                lambda g=generation, s=state: self._apply_uncatalogued_lists_result(body, g, s),
            )

        threading.Thread(
            target=_worker,
            name="SettingsUncataloguedScan",
            daemon=True,
        ).start()

    def _apply_uncatalogued_lists_result(self, body, generation: int, state: dict) -> None:
        if generation != self.__dict__.get("_uncatalogued_refresh_generation", 0):
            return
        if not self._widget_is_alive(body):
            return
        self._render_uncatalogued_lists(body, state)

    def _show_uncatalogued_loading(self, body) -> None:
        for child in list(body.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass
        ctk.CTkLabel(
            body,
            text="Scanning installed models in the background …",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(6, 0), pady=(4, 10))

    def _collect_uncatalogued_lists(self) -> dict:
        catalog_entries = self._catalog_entries_for_reconciliation()
        installed, err = self._list_installed_ollama_tags()
        installed_tags = [r["tag"] for r in installed]
        tag_meta = {r["tag"]: r for r in installed}
        uncatalogued_tags = (
            []
            if err
            else _migration.scan_uncatalogued_ollama_tags(installed_tags, catalog_entries)
        )

        models_dir = Path(self.cfg.get("models_dir", "") or ".")
        try:
            uncatalogued_dirs = _migration.scan_uncatalogued_onnx_dirs(
                models_dir, catalog_entries,
            )
        except Exception as exc:
            logger.error(f"Uncatalogued ONNX scan failed: {exc}")
            uncatalogued_dirs = []

        return {
            "ollama_error": err,
            "tag_meta": tag_meta,
            "uncatalogued_tags": uncatalogued_tags,
            "uncatalogued_dirs": uncatalogued_dirs,
        }

    def _render_uncatalogued_lists(self, body, state: dict) -> None:
        """Render or re-render the two uncatalogued lists inside *body*."""
        for child in list(body.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass

        fatal_error = state.get("fatal_error")
        if fatal_error:
            ctk.CTkLabel(
                body,
                text=f"Could not scan installed models: {fatal_error}",
                text_color=ERROR_TEXT,
                font=ctk.CTkFont(size=11),
                anchor="w",
                wraplength=680,
                justify="left",
            ).grid(row=0, column=0, sticky="w", padx=(6, 0), pady=(4, 10))
            return

        # Ollama section.
        ollama_lbl = ctk.CTkLabel(
            body,
            text="Ollama tags not in catalog",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        ollama_lbl.grid(row=0, column=0, sticky="w", pady=(4, 4))

        err = state.get("ollama_error")
        if err:
            ctk.CTkLabel(
                body,
                text=err,
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=11),
                anchor="w",
                wraplength=680,
                justify="left",
            ).grid(row=1, column=0, sticky="w", padx=(6, 0), pady=(0, 10))
        else:
            uncatalogued_tags = state.get("uncatalogued_tags", [])
            tag_meta = state.get("tag_meta", {})
            if not uncatalogued_tags:
                ctk.CTkLabel(
                    body,
                    text="Everything installed is in your catalog ✓",
                    text_color=TEXT_MUTED,
                    font=ctk.CTkFont(size=11),
                    anchor="w",
                ).grid(row=1, column=0, sticky="w", padx=(6, 0), pady=(0, 10))
            else:
                rows_frame = ctk.CTkFrame(body, fg_color="transparent")
                rows_frame.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
                rows_frame.grid_columnconfigure(0, weight=1)
                for i, tag in enumerate(uncatalogued_tags):
                    size = (tag_meta.get(tag) or {}).get("size") or ""
                    row = ctk.CTkFrame(rows_frame, fg_color="transparent")
                    row.grid(row=i, column=0, sticky="ew", pady=2)
                    row.grid_columnconfigure(1, weight=1)
                    ctk.CTkLabel(
                        row, text=tag, font=ctk.CTkFont(size=12, weight="bold"),
                        anchor="w",
                    ).grid(row=0, column=0, sticky="w")
                    ctk.CTkLabel(
                        row, text=size, text_color=TEXT_MUTED,
                        font=ctk.CTkFont(size=11), anchor="w",
                    ).grid(row=0, column=1, sticky="w", padx=(12, 0))
                    ctk.CTkButton(
                        row, text="Delete", width=80,
                        command=lambda t=tag: self._confirm_delete_ollama_tags([t]),
                        **self._solid_button_style(self._IG_DANGER, "#8a2424"),
                    ).grid(row=0, column=2, sticky="e", padx=(8, 0))
                bulk = ctk.CTkButton(
                    body,
                    text=f"Delete all not-in-catalog Ollama tags ({len(uncatalogued_tags)})",
                    command=lambda lst=list(uncatalogued_tags):
                        self._confirm_delete_ollama_tags(lst),
                    **self._solid_button_style(self._IG_DANGER, "#8a2424"),
                )
                bulk.grid(row=2, column=0, sticky="w", padx=(6, 0), pady=(2, 10))

        # ONNX section.
        onnx_lbl = ctk.CTkLabel(
            body,
            text="ONNX directories not in catalog",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        onnx_lbl.grid(row=3, column=0, sticky="w", pady=(8, 4))

        uncatalogued_dirs = state.get("uncatalogued_dirs", [])
        if not uncatalogued_dirs:
            ctk.CTkLabel(
                body,
                text="Everything installed is in your catalog ✓",
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=11),
                anchor="w",
            ).grid(row=4, column=0, sticky="w", padx=(6, 0), pady=(0, 10))
        else:
            rows_frame = ctk.CTkFrame(body, fg_color="transparent")
            rows_frame.grid(row=4, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
            rows_frame.grid_columnconfigure(0, weight=1)
            for i, p in enumerate(uncatalogued_dirs):
                row = ctk.CTkFrame(rows_frame, fg_color="transparent")
                row.grid(row=i, column=0, sticky="ew", pady=2)
                row.grid_columnconfigure(1, weight=1)
                ctk.CTkLabel(
                    row, text=p.name, font=ctk.CTkFont(size=12, weight="bold"),
                    anchor="w",
                ).grid(row=0, column=0, sticky="w")
                ctk.CTkLabel(
                    row, text=str(p), text_color=TEXT_MUTED,
                    font=ctk.CTkFont(size=10), anchor="w", wraplength=520,
                    justify="left",
                ).grid(row=0, column=1, sticky="w", padx=(12, 0))
                ctk.CTkButton(
                    row, text="Delete", width=80,
                    command=lambda d=p: self._confirm_delete_onnx_dirs([d]),
                    **self._solid_button_style(self._IG_DANGER, "#8a2424"),
                ).grid(row=0, column=2, sticky="e", padx=(8, 0))
            bulk_o = ctk.CTkButton(
                body,
                text=f"Delete all not-in-catalog ONNX directories ({len(uncatalogued_dirs)})",
                command=lambda lst=list(uncatalogued_dirs):
                    self._confirm_delete_onnx_dirs(lst),
                **self._solid_button_style(self._IG_DANGER, "#8a2424"),
            )
            bulk_o.grid(row=5, column=0, sticky="w", padx=(6, 0), pady=(2, 10))

    def _confirm_delete_ollama_tags(self, tags: list[str]) -> None:
        """Confirm + run ``ollama rm`` on each tag in *tags*. No-op on empty input."""
        tags = [t for t in (tags or []) if isinstance(t, str) and t.strip()]
        if not tags:
            return
        listing = "\n".join(f"  • {t}" for t in tags[:20])
        more = f"\n…and {len(tags) - 20} more" if len(tags) > 20 else ""
        confirm = messagebox.askyesno(
            "Delete not-in-catalog Ollama tags?",
            f"This will run `ollama rm` for the following "
            f"{len(tags)} tag{'s' if len(tags) != 1 else ''}:\n\n{listing}{more}\n\n"
            "This frees disk space and cannot be undone. Continue?",
            icon="warning",
            parent=self,
        )
        if not confirm:
            return
        failures: list[str] = []
        for tag in tags:
            try:
                proc = subprocess.run(
                    ["ollama", "rm", tag],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if proc.returncode != 0:
                    failures.append(f"{tag}: {(proc.stderr or proc.stdout or '').strip()[:120]}")
                else:
                    logger.info(f"Deleted not-in-catalog Ollama tag: {tag}")
            except Exception as exc:
                failures.append(f"{tag}: {exc}")
        if failures:
            messagebox.showwarning(
                "Some deletes failed",
                "Some tags could not be removed:\n\n" + "\n".join(failures),
                parent=self,
            )
        else:
            messagebox.showinfo(
                "Deleted",
                f"Removed {len(tags)} not-in-catalog Ollama tag"
                f"{'s' if len(tags) != 1 else ''}.",
                parent=self,
            )
        body = getattr(self, "_uncatalogued_body_ref", None)
        if body is not None:
            try:
                self._refresh_uncatalogued_lists_async(body)
            except Exception:
                pass

    def _confirm_delete_onnx_dirs(self, dirs: list[Path]) -> None:
        """Confirm + ``shutil.rmtree`` each directory. No-op on empty input."""
        dirs = [Path(d) for d in (dirs or []) if d]
        if not dirs:
            return
        listing = "\n".join(f"  • {d}" for d in dirs[:20])
        more = f"\n…and {len(dirs) - 20} more" if len(dirs) > 20 else ""
        confirm = messagebox.askyesno(
            "Delete not-in-catalog ONNX directories?",
            f"This will permanently delete the following "
            f"{len(dirs)} director{'ies' if len(dirs) != 1 else 'y'}:\n\n"
            f"{listing}{more}\n\n"
            "This frees disk space and cannot be undone. Continue?",
            icon="warning",
            parent=self,
        )
        if not confirm:
            return
        failures: list[str] = []
        import shutil as _shutil
        for d in dirs:
            try:
                _shutil.rmtree(str(d))
                logger.info(f"Deleted not-in-catalog ONNX directory: {d}")
            except Exception as exc:
                failures.append(f"{d}: {exc}")
        if failures:
            messagebox.showwarning(
                "Some deletes failed",
                "Some directories could not be removed:\n\n" + "\n".join(failures),
                parent=self,
            )
        else:
            messagebox.showinfo(
                "Deleted",
                f"Removed {len(dirs)} uncatalogued director"
                f"{'ies' if len(dirs) != 1 else 'y'}.",
                parent=self,
            )
        body = getattr(self, "_uncatalogued_body_ref", None)
        if body is not None:
            try:
                self._refresh_uncatalogued_lists_async(body)
            except Exception:
                pass

    # ── Migration engine wrapper ─────────────────────────────────────────

    def _run_migration_engine_modal(self, plan, *, resume_from=None) -> bool:
        """Run a MigrationEngine plan inline with a CTk progress dialog.

        Returns True on success, False otherwise.  All exceptions are caught
        and surfaced via messagebox so the caller never has to wrap us.
        """
        from src.migration_ui import (
            MigrationProgressDialog,
            PreflightFailureDialog,
        )
        app_root = Path(__file__).parent.parent

        def _commit_config(new_target: Path) -> bool:
            try:
                if plan.kind == "comfyui":
                    self.cfg["comfyui_dir"] = str(new_target)
                    ok = config.save(self.cfg)
                    if ok:
                        try:
                            self._sync_comfyui_path_bat(new_target)
                        except Exception:
                            pass
                    return ok
                if plan.kind == "models":
                    self.cfg["models_dir"] = str(new_target)
                    return config.save(self.cfg)
                if plan.kind == "ollama":
                    if not self._set_user_env_var("OLLAMA_MODELS", str(new_target)):
                        return False
                    os.environ["OLLAMA_MODELS"] = str(new_target)
                    return True
            except Exception as exc:
                logger.error(f"Migration commit failed for {plan.kind}: {exc}")
                return False
            return True

        engine = _migration.MigrationEngine(
            plan, app_root=app_root, config_commit=_commit_config,
            resume_from=resume_from,
        )
        pre = engine.preflight()
        if not pre.ok:
            try:
                PreflightFailureDialog(self, pre)
            except Exception:
                messagebox.showerror(
                    "Migration can't start", pre.reason or "Pre-flight check failed.",
                    parent=self,
                )
            return False
        if pre.target_is_reboot_volatile:
            confirm = messagebox.askyesno(
                "Target drive may be wiped on reboot",
                f"The destination drive ({Path(plan.target).drive or plan.target}) "
                "looks like a removable or RAM disk. On some cloud VMs and "
                "VMs these volumes are wiped between sessions.\n\n"
                "Continue anyway?",
                icon="warning", parent=self,
            )
            if not confirm:
                return False
        done_holder: dict[str, Any] = {"ok": False, "exc": None, "ready": False}

        def _on_done(ok, exc):
            done_holder["ok"] = ok
            done_holder["exc"] = exc
            done_holder["ready"] = True
        try:
            dialog = MigrationProgressDialog(self, engine, on_done=_on_done)
            dialog.start()
            # Drive the Tk main loop until the dialog finishes.
            while not done_holder["ready"]:
                try:
                    self.update()
                except Exception:
                    break
                time.sleep(0.05)
        except Exception as exc:
            logger.error(f"Migration UI failed: {exc}")
            return False
        if done_holder["exc"] is not None and not isinstance(done_holder["exc"], _migration.MigrationCancelled):
            messagebox.showerror(
                "Migration failed",
                f"Could not complete migration:\n{done_holder['exc']}",
                parent=self,
            )
        return bool(done_holder["ok"])

    def _prompt_unified_storage_relocation(self, prompting: list) -> None:
        """Composite dialog: three checkboxes (default-checked) + Yes/No/Decide-later.

        ``prompting`` is a list of ``(key, pending)`` pairs where key is
        one of ``"models"``, ``"comfyui"``, ``"ollama"``.
        """
        keys = {key: pending for key, pending in prompting}

        dlg = ctk.CTkToplevel(self)
        dlg.title("Move LocalAI storage off your user profile?")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        # Body copy
        intro_lines = [
            "LocalAI now recommends keeping its storage next to the app "
            "instead of inside your Windows user profile.",
            "",
            "If your user profile has limited space (common on corporate "
            "or managed machines and on any roamed profile), large "
            "models can fill it up and downloads will fail with \"no "
            "space left on device\" even when your disk shows plenty of "
            "room.",
            "",
            "Pick which directories you'd like to move:",
        ]
        if self._app_drive_is_removable():
            intro_lines.insert(
                0,
                "Heads up: LocalAI is running from a removable drive — "
                "its drive letter can change between sessions, which "
                "may affect anything you migrate to it.",
            )
            intro_lines.insert(1, "")

        intro = "\n".join(intro_lines)
        ctk.CTkLabel(
            dlg, text=intro, justify="left", anchor="w", wraplength=520,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(16, 8))

        # Build a checkbox per pending migration. Default-checked.
        vars_by_key: dict[str, ctk.BooleanVar] = {}
        row_index = 1
        labels = {
            "models": (
                "Move LocalAI models directory",
                lambda p: f"  {p['current']}\n  → {p['new_default']}",
            ),
            "comfyui": (
                "Move ComfyUI install",
                lambda p: f"  {p['old_path']}\n  → {p['new_path']}",
            ),
            "ollama": (
                "Move Ollama models directory (affects every Ollama app on this PC)",
                lambda p: f"  {p['old_dir']}\n  → {p['new_target']}",
            ),
        }
        for key in ("models", "comfyui", "ollama"):
            if key not in keys:
                continue
            label_text, detail_fn = labels[key]
            var = ctk.BooleanVar(value=True)
            vars_by_key[key] = var
            ctk.CTkCheckBox(
                dlg, text=label_text, variable=var,
            ).grid(row=row_index, column=0, columnspan=2, sticky="w", padx=24, pady=(8, 0))
            row_index += 1
            ctk.CTkLabel(
                dlg, text=detail_fn(keys[key]), justify="left", anchor="w",
                font=ctk.CTkFont(size=10), text_color=TEXT_MUTED,
            ).grid(row=row_index, column=0, columnspan=2, sticky="w", padx=44, pady=(0, 4))
            row_index += 1

        if "ollama" in keys:
            ctk.CTkLabel(
                dlg,
                text=(
                    "Note: changing the Ollama models directory updates "
                    "OLLAMA_MODELS for your user account. After the "
                    "Ollama daemon restarts, every other app on this "
                    "PC that talks to Ollama (Open-WebUI, Continue.dev, "
                    "other LLM front-ends) will also start reading from "
                    "the new directory."
                ),
                justify="left", anchor="w", wraplength=520,
                font=ctk.CTkFont(size=10), text_color=TEXT_MUTED,
            ).grid(row=row_index, column=0, columnspan=2, sticky="w", padx=24, pady=(4, 8))
            row_index += 1

        # Button row
        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.grid(row=row_index, column=0, columnspan=2, sticky="e", padx=20, pady=(12, 16))
        choice = {"value": "later"}

        def _on_yes():
            choice["value"] = "yes"
            dlg.destroy()

        def _on_no():
            choice["value"] = "no"
            dlg.destroy()

        def _on_later():
            choice["value"] = "later"
            dlg.destroy()

        ctk.CTkButton(
            btn_row, text="Move checked items",
            **self._solid_button_style("#1f6aa5", "#1f538d"),
            command=_on_yes,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_row, text="Don't ask again",
            **self._outline_button_style(),
            command=_on_no,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_row, text="Decide later",
            **self._outline_button_style(),
            command=_on_later,
        ).pack(side="right")

        dlg.protocol("WM_DELETE_WINDOW", _on_later)
        dlg.wait_window()

        decision = choice["value"]
        if decision == "later":
            return
        for key, pending in prompting:
            checked = vars_by_key.get(key, ctk.BooleanVar(value=False)).get() if decision == "yes" else False
            if decision == "no":
                # User said "Don't ask again" — apply No semantics to each
                # (record declined for Ollama; remap config for the others
                # so we don't loop on next launch).
                if key == "models":
                    self._apply_models_dir_migration(pending, move=False)
                elif key == "comfyui":
                    self._apply_comfyui_migration(pending, move=False)
                elif key == "ollama":
                    self.cfg["ollama_offer_relocation"] = "declined"
                    if not config.save(self.cfg):
                        logger.error("Could not persist ollama_offer_relocation flag")
                    logger.info("User declined Ollama relocation via composite dialog")
                continue
            # decision == "yes"
            if not checked:
                # The user unchecked this item in the composite — record
                # No-equivalent semantics so we don't re-prompt.
                if key == "models":
                    self._apply_models_dir_migration(pending, move=False)
                elif key == "comfyui":
                    self._apply_comfyui_migration(pending, move=False)
                elif key == "ollama":
                    self.cfg["ollama_offer_relocation"] = "declined"
                    if not config.save(self.cfg):
                        logger.error("Could not persist ollama_offer_relocation flag")
                continue
            if key == "models":
                self._apply_models_dir_migration(pending, move=True)
            elif key == "comfyui":
                self._apply_comfyui_migration(pending, move=True)
            elif key == "ollama":
                self._apply_ollama_models_dir_migration(pending)

    def _comfyui_installed_path(self) -> Optional[Path]:
        """Return the ComfyUI installation path, checking config first."""
        # 1. Check config.json (authoritative source)
        cfg_dir = self.cfg.get("comfyui_dir", "")
        if cfg_dir:
            p = Path(cfg_dir)
            if (p / "main.py").exists():
                self._sync_comfyui_path_bat(p)
                return p

        # 2. Legacy: read comfyui_path.bat (Windows only)
        if sys.platform == "win32":
            bat = Path(__file__).parent.parent / "comfyui_path.bat"
            if bat.exists():
                try:
                    text = bat.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        line = line.strip()
                        if line.upper().startswith("SET LOCALAI_COMFYUI="):
                            p = Path(line.split("=", 1)[1].strip())
                            if (p / "main.py").exists():
                                if cfg_dir != str(p):
                                    self.cfg["comfyui_dir"] = str(p)
                                    if not config.save(self.cfg):
                                        logger.error("Could not persist repaired ComfyUI path")
                                self._sync_comfyui_path_bat(p)
                                return p
                except Exception:
                    pass

        # 3. Legacy: check old embedded location
        old_path = Path(__file__).parent.parent / "ComfyUI"
        if (old_path / "main.py").exists():
            configured = Path(cfg_dir) if cfg_dir else None
            if configured is not None and configured != old_path:
                self.cfg["comfyui_dir"] = str(old_path)
                if not config.save(self.cfg):
                    logger.error("Could not persist repaired ComfyUI path")
                self._sync_comfyui_path_bat(old_path)
            return old_path

        # 4. v2026.06.01.9: sibling-of-app fallback. ``setup.bat`` line ~388
        # (``%~dp0..\\ComfyUI``) installs ComfyUI here when an earlier app
        # tree was discovered next door. When v.N+1 is unzipped on top of
        # v.N the new app dir has no ``config.json`` or ``comfyui_path.bat``
        # yet, but the sibling ComfyUI is fully functional. Without this
        # fallback the benchmark and Image Gen tab both report "ComfyUI not
        # installed at expected paths" for a working install — fix is to
        # match setup.bat's own search order.
        sibling_path = Path(__file__).parent.parent.parent / "ComfyUI"
        if (sibling_path / "main.py").exists():
            self.cfg["comfyui_dir"] = str(sibling_path)
            if not config.save(self.cfg):
                logger.error("Could not persist repaired ComfyUI path")
            self._sync_comfyui_path_bat(sibling_path)
            return sibling_path

        return None

    def _check_comfyui_async(self):
        def _check():
            installed = self._comfyui_installed_path()
            if not installed:
                self.after(0, lambda: self._set_comfyui_status(
                    "not installed\nRun setup.sh to install" if sys.platform == "darwin" else "not installed\nRun setup.bat to install", TEXT_MUTED
                ))
                logger.info("ComfyUI not installed — image generation unavailable.")
                return

            if not self.comfyui.is_running():
                self.after(0, lambda: self._set_comfyui_status(
                    "installed — click Generate to start", TEXT_MUTED
                ))
                self.comfyui_ok = False
                logger.info("ComfyUI installed; startup deferred until Generate or Analyze→Prompt.")
                return

            self.comfyui_ok = True
            try:
                stats = self.comfyui.get_system_stats()
                gpu = stats.get("devices", [{}])[0].get("name", "GPU")
            except Exception:
                gpu = "GPU"
            self.after(0, lambda g=gpu: self._set_comfyui_status(
                f"running ({g})", SUCCESS_TEXT
            ))
            self.after(0, lambda: self._set_image_gen_enabled(True))
            self.after(500, self._refresh_model_cards)
            logger.info(f"ComfyUI running at {self.comfyui.host}")
            return
        threading.Thread(target=_check, daemon=True).start()

    def _set_status_threadsafe(self, message: str) -> None:
        if threading.current_thread() is threading.main_thread():
            self.set_status(message)
        else:
            self.after(0, lambda m=message: self.set_status(m))

    def _update_idletasks_if_ui_thread(self) -> None:
        if threading.current_thread() is threading.main_thread():
            self.update_idletasks()

    def _ensure_gguf_support(self, comfyui_path: Path, *, prompt: bool = True) -> bool:
        """Install ComfyUI-GGUF custom node and Flux CLIP/VAE files if missing.

        Returns True if everything is ready, False if installation failed or
        was cancelled.
        """
        custom_nodes_dir = comfyui_path / "custom_nodes"
        gguf_node_dir = custom_nodes_dir / "ComfyUI-GGUF"

        needs_restart = False

        # 1. Install ComfyUI-GGUF custom node
        if not gguf_node_dir.exists():
            proceed = True
            if prompt:
                proceed = messagebox.askyesno(
                    "Install GGUF Support",
                    "GGUF models require the ComfyUI-GGUF custom node.\n\n"
                    "Install it now? (Required to use this model)",
                    parent=self,
                )
            if not proceed:
                return False

            logger.info("Installing ComfyUI-GGUF custom node...", category=logger.CATEGORY_IMAGE_GEN)
            self._set_status_threadsafe("Downloading GGUF support for ComfyUI...")
            try:
                import shutil, zipfile, io, requests
                custom_nodes_dir.mkdir(parents=True, exist_ok=True)

                zip_url = "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip"
                logger.info(f"Downloading ComfyUI-GGUF from {zip_url}", category=logger.CATEGORY_IMAGE_GEN)
                resp = requests.get(zip_url, timeout=60)
                resp.raise_for_status()

                tmp_dir = custom_nodes_dir / ".ComfyUI-GGUF.download"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    zf.extractall(tmp_dir)

                extracted = tmp_dir / "ComfyUI-GGUF-main"
                if extracted.exists():
                    shutil.move(str(extracted), str(gguf_node_dir))
                shutil.rmtree(tmp_dir, ignore_errors=True)

                logger.info("ComfyUI-GGUF custom node installed successfully", category=logger.CATEGORY_IMAGE_GEN)
                needs_restart = True
            except Exception as e:
                try:
                    import shutil as _shutil
                    _shutil.rmtree(custom_nodes_dir / ".ComfyUI-GGUF.download", ignore_errors=True)
                except Exception:
                    pass
                logger.error(f"Failed to install GGUF custom node: {e}", category=logger.CATEGORY_IMAGE_GEN)
                if prompt:
                    messagebox.showerror(
                        "Installation Failed",
                        f"Could not install GGUF support:\n\n{e}\n\n"
                        "Check your internet connection and try again.",
                        parent=self,
                    )
                return False

        # 1b. Ensure the `gguf` Python package is installed.
        #     ComfyUI-GGUF's ops.py does `import gguf` at load time — if the package
        #     is missing, the node silently fails and UnetLoaderGGUF never registers.
        import importlib.util as _ilu
        if _ilu.find_spec("gguf") is None:
            logger.info(
                "Installing gguf Python package (required by ComfyUI-GGUF)…",
                category=logger.CATEGORY_IMAGE_GEN,
            )
            self._set_status_threadsafe("Installing gguf package…")
            try:
                import subprocess as _sp
                run_kw = {"capture_output": True, "text": True, "timeout": 300}
                if sys.platform == "win32":
                    run_kw["creationflags"] = _sp.CREATE_NO_WINDOW
                result = _sp.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--no-input",
                        "--disable-pip-version-check",
                        "gguf>=0.13.0",
                        "PyYAML>=5.1",
                        "tqdm>=4.27",
                    ],
                    **run_kw,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "pip install failed")
                logger.info("gguf package installed successfully", category=logger.CATEGORY_IMAGE_GEN)
                needs_restart = True
            except Exception as e:
                logger.error(f"Failed to install gguf package: {e}", category=logger.CATEGORY_IMAGE_GEN)
                if prompt:
                    messagebox.showerror(
                        "Installation Failed",
                        f"Could not install the gguf Python package:\n\n{e}\n\n"
                        "Run manually: pip install gguf>=0.13.0 PyYAML>=5.1 tqdm>=4.27",
                        parent=self,
                    )
                return False

        # 2. Ensure Flux CLIP encoders exist (needed by the GGUF Flux workflow)
        clip_dir = comfyui_path / "models" / "clip"
        clip_dir.mkdir(parents=True, exist_ok=True)
        flux_clip_files = {
            "t5xxl_fp8_e4m3fn.safetensors": (
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
                "t5xxl_fp8_e4m3fn.safetensors"
            ),
            "clip_l.safetensors": (
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
                "clip_l.safetensors"
            ),
        }
        missing_clips = {
            name: url for name, url in flux_clip_files.items()
            if not (clip_dir / name).exists()
        }

        # 3. Ensure Flux VAE exists
        vae_dir = comfyui_path / "models" / "vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        flux_vae_file = "ae.safetensors"
        flux_vae_url = (
            "https://huggingface.co/sirorable/flux-ae-vae/resolve/main/ae.safetensors"
        )
        missing_vae = not (vae_dir / flux_vae_file).exists()

        if missing_clips or missing_vae:
            parts = []
            if missing_clips:
                parts.append(f"CLIP encoders: {', '.join(missing_clips)}")
            if missing_vae:
                parts.append(f"VAE: {flux_vae_file}")
            proceed = True
            if prompt:
                proceed = messagebox.askyesno(
                    "Download Flux Support Files",
                    f"Flux GGUF models need additional support files:\n\n"
                    + "\n".join(f"  • {p}" for p in parts) +
                    "\n\nDownload them now? (One-time, ~5 GB total)",
                    parent=self,
                )
            if not proceed:
                return False

            import requests
            for name, url in missing_clips.items():
                dest = clip_dir / name
                logger.info(f"Downloading Flux CLIP: {name}", category=logger.CATEGORY_IMAGE_GEN)
                self._set_status_threadsafe(f"Downloading {name} …")
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    r.raise_for_status()
                    _last_idle = 0.0
                    def _progress(_downloaded):
                        nonlocal _last_idle
                        now = time.monotonic()
                        if now - _last_idle >= 0.25:
                            _last_idle = now
                            self._update_idletasks_if_ui_thread()
                    self._download_stream_to_path(r, dest, progress_cb=_progress)
                    logger.info(f"Downloaded {name}", category=logger.CATEGORY_IMAGE_GEN)
                except Exception as e:
                    logger.error(f"Failed to download {name}: {e}", category=logger.CATEGORY_IMAGE_GEN)
                    if prompt:
                        messagebox.showerror(
                            "Download Failed",
                            f"Could not download {name}:\n\n{e}",
                            parent=self,
                        )
                    return False

            if missing_vae:
                dest = vae_dir / flux_vae_file
                logger.info(f"Downloading Flux VAE: {flux_vae_file}", category=logger.CATEGORY_IMAGE_GEN)
                self._set_status_threadsafe(f"Downloading {flux_vae_file} …")
                try:
                    r = requests.get(flux_vae_url, stream=True, timeout=30)
                    r.raise_for_status()
                    _last_idle = 0.0
                    def _progress(_downloaded):
                        nonlocal _last_idle
                        now = time.monotonic()
                        if now - _last_idle >= 0.25:
                            _last_idle = now
                            self._update_idletasks_if_ui_thread()
                    self._download_stream_to_path(r, dest, progress_cb=_progress)
                    logger.info(f"Downloaded {flux_vae_file}", category=logger.CATEGORY_IMAGE_GEN)
                except Exception as e:
                    logger.error(f"Failed to download {flux_vae_file}: {e}", category=logger.CATEGORY_IMAGE_GEN)
                    if prompt:
                        messagebox.showerror(
                            "Download Failed",
                            f"Could not download {flux_vae_file}:\n\n{e}",
                            parent=self,
                        )
                    return False

        if needs_restart and prompt:
            messagebox.showinfo(
                "GGUF Support Installed",
                "ComfyUI-GGUF custom node and Flux support files installed!\n\n"
                "ComfyUI will be restarted after download to load the new node.",
                parent=self,
            )

        return True

    def _download_stream_to_path(
        self,
        response,
        dest: Path,
        *,
        stop_event: Optional[threading.Event] = None,
        progress_cb=None,
        chunk_size: int = 1_048_576,
    ) -> None:
        """Write a streaming HTTP response to dest safely using dest.part."""
        partial = dest.with_name(dest.name + ".part")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if partial.exists():
                partial.unlink()
            downloaded = 0
            with open(partial, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if stop_event and stop_event.is_set():
                        raise RuntimeError("Download cancelled")
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded)
            os.replace(partial, dest)
        except Exception:
            try:
                if partial.exists():
                    partial.unlink()
            except OSError:
                pass
            raise

    # ── Chroma custom-node content ────────────────────────────────────────────
    _CHROMA_NODE_CODE = '''\
import torch

class ChromaLatentToImage:
    """Convert Chroma x0 pixel-space latent output to a ComfyUI IMAGE.

    Chroma x0 outputs [B, 3, H, W] pixel-space data from KSampler instead of
    the standard 16-channel Flux latent. The standard Flux VAE
    (ae.safetensors) cannot decode it. This node converts it directly to a
    ComfyUI IMAGE tensor ([B, H, W, 3], float32, values in [0, 1]).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "LocalAI"

    def convert(self, samples):
        t = samples["samples"].float()  # [B, 3, H, W]
        # x0 prediction outputs are typically normalised to [-1, 1].
        # Clamp and remap to [0, 1] for image display.
        t = t.clamp(-1.0, 1.0)
        t = (t + 1.0) / 2.0
        t = t.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        return (t,)


NODE_CLASS_MAPPINGS = {"ChromaLatentToImage": ChromaLatentToImage}
NODE_DISPLAY_NAME_MAPPINGS = {"ChromaLatentToImage": "Chroma: Latent To Image"}
'''

    def _ensure_chroma_support(self, comfyui_path: Path) -> bool:
        """Write the ChromaLatentToImage custom node if it is not already present.

        Returns True if the node is ready.  Returns True even if it was just
        written (caller is responsible for restarting ComfyUI when needed).
        Sets self._chroma_node_needs_restart = True when a restart is required.
        """
        node_dir = comfyui_path / "custom_nodes" / "ComfyUI-LocalAI-Chroma"
        init_file = node_dir / "__init__.py"
        if init_file.exists():
            return True
        try:
            node_dir.mkdir(parents=True, exist_ok=True)
            init_file.write_text(self._CHROMA_NODE_CODE, encoding="utf-8")
            logger.info(
                "ChromaLatentToImage custom node written — ComfyUI restart required",
                category=logger.CATEGORY_IMAGE_GEN,
            )
            self._chroma_node_needs_restart = True
        except Exception as e:
            logger.error(f"Failed to write Chroma custom node: {e}", category=logger.CATEGORY_IMAGE_GEN)
            return False
        return True

    def _ensure_flux_clip_vae(self, comfyui_path: Path, *, prompt: bool = True) -> bool:
        """Ensure Flux CLIP encoders and VAE are present for non-GGUF Flux models.

        The Comfy-Org flux1-dev.safetensors is UNET-only (no CLIP, no VAE).
        Downloads external CLIP (t5xxl + clip_l) and VAE (ae.safetensors).
        Returns True if ready, False if download failed or cancelled.
        """
        # Collect all missing support files
        missing_files = {}  # {dest_path: url}

        clip_dir = comfyui_path / "models" / "clip"
        clip_dir.mkdir(parents=True, exist_ok=True)
        flux_clip_files = {
            "t5xxl_fp8_e4m3fn.safetensors": (
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
                "t5xxl_fp8_e4m3fn.safetensors"
            ),
            "clip_l.safetensors": (
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
                "clip_l.safetensors"
            ),
        }
        for name, url in flux_clip_files.items():
            if not (clip_dir / name).exists():
                missing_files[clip_dir / name] = url

        vae_dir = comfyui_path / "models" / "vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        vae_file = "ae.safetensors"
        vae_url = "https://huggingface.co/sirorable/flux-ae-vae/resolve/main/ae.safetensors"
        if not (vae_dir / vae_file).exists():
            missing_files[vae_dir / vae_file] = vae_url

        if not missing_files:
            return True

        names = [p.name for p in missing_files]
        proceed = True
        if prompt:
            proceed = messagebox.askyesno(
                "Download Flux Support Files",
                f"Flux models need additional support files:\n\n"
                + "\n".join(f"  \u2022 {n}" for n in names) +
                "\n\nDownload them now? (One-time, ~5 GB total)",
                parent=self,
            )
        if not proceed:
            return False

        import requests
        for dest, url in missing_files.items():
            name = dest.name
            logger.info(f"Downloading Flux support file: {name}", category=logger.CATEGORY_IMAGE_GEN)
            self._set_status_threadsafe(f"Downloading {name} \u2026")
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                _last_idle = 0.0
                def _progress(_downloaded):
                    nonlocal _last_idle
                    now = time.monotonic()
                    if now - _last_idle >= 0.25:
                        _last_idle = now
                        self._update_idletasks_if_ui_thread()
                self._download_stream_to_path(r, dest, progress_cb=_progress)
                logger.info(f"Downloaded {name}", category=logger.CATEGORY_IMAGE_GEN)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}", category=logger.CATEGORY_IMAGE_GEN)
                if prompt:
                    messagebox.showerror(
                        "Download Failed",
                        f"Could not download {name}:\n\n{e}",
                        parent=self,
                    )
                return False
        return True

    def _ensure_z_image_support(self, comfyui_path: Path, *, prompt: bool = True) -> bool:
        """Ensure Z-Image support files (Qwen text encoder + Flux VAE) are present.

        Both Z-Image and Z-Image Turbo require:
          • qwen_3_4b_fp8_mixed.safetensors → models/text_encoders/
          • ae.safetensors (Flux VAE)       → models/vae/
        Returns True if ready, False if download failed or was cancelled.
        """
        missing_files = {}  # {dest_path: url}

        te_dir = comfyui_path / "models" / "text_encoders"
        te_dir.mkdir(parents=True, exist_ok=True)
        qwen_file = "qwen_3_4b_fp8_mixed.safetensors"
        qwen_url = (
            "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
            "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors"
        )
        if not (te_dir / qwen_file).exists():
            missing_files[te_dir / qwen_file] = qwen_url

        vae_dir = comfyui_path / "models" / "vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        vae_file = "ae.safetensors"
        vae_url = "https://huggingface.co/sirorable/flux-ae-vae/resolve/main/ae.safetensors"
        if not (vae_dir / vae_file).exists():
            missing_files[vae_dir / vae_file] = vae_url

        if not missing_files:
            return True

        names = [p.name for p in missing_files]
        proceed = True
        if prompt:
            proceed = messagebox.askyesno(
                "Download Z-Image Support Files",
                "Z-Image needs additional support files:\n\n"
                + "\n".join(f"  \u2022 {n}" for n in names)
                + "\n\nDownload them now? (One-time, ~6 GB total)",
                parent=self,
            )
        if not proceed:
            return False

        import requests
        for dest, url in missing_files.items():
            name = dest.name
            logger.info(f"Downloading Z-Image Turbo support file: {name}", category=logger.CATEGORY_IMAGE_GEN)
            self._set_status_threadsafe(f"Downloading {name} \u2026")
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                _last_idle = 0.0
                def _progress(_downloaded):
                    nonlocal _last_idle
                    now = time.monotonic()
                    if now - _last_idle >= 0.25:
                        _last_idle = now
                        self._update_idletasks_if_ui_thread()
                self._download_stream_to_path(r, dest, progress_cb=_progress)
                logger.info(f"Downloaded {name}", category=logger.CATEGORY_IMAGE_GEN)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}", category=logger.CATEGORY_IMAGE_GEN)
                if prompt:
                    messagebox.showerror(
                        "Download Failed",
                        f"Could not download {name}:\n\n{e}",
                        parent=self,
                    )
                return False
        return True

    def _ensure_image_model_runtime_support(self, model_filename: str, *, prompt: bool = True) -> bool:
        """Ensure support files/custom nodes for a local ComfyUI model before generation."""
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            return False
        lower = (model_filename or "").lower()
        is_gguf = lower.endswith(".gguf")
        is_chroma = "chroma" in lower
        is_z_image = "z_image" in lower
        is_flux = "flux" in lower or is_chroma or is_gguf
        if is_gguf:
            return self._ensure_gguf_support(comfyui_path, prompt=prompt)
        if is_z_image:
            return self._ensure_z_image_support(comfyui_path, prompt=prompt)
        if is_chroma:
            if not self._ensure_chroma_support(comfyui_path):
                return False
            return self._ensure_flux_clip_vae(comfyui_path, prompt=prompt)
        if is_flux:
            return self._ensure_flux_clip_vae(comfyui_path, prompt=prompt)
        return True

    def _image_model_runtime_support_missing_items(self, model_filename: str) -> list[str]:
        """Return support packages/files that must be prepared before this model can run."""
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
            import importlib.util as _ilu
            if _ilu.find_spec("gguf") is None:
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

    def _image_model_runtime_needs_restart(self, model_filename: str) -> bool:
        """Return True when ComfyUI is running but missing nodes required by this model."""
        try:
            if not self.comfyui.is_running():
                return False
            lower = (model_filename or "").lower()
            if lower.endswith(".gguf") and not self.comfyui.has_gguf_node():
                return True
            if "chroma" in lower and not self.comfyui.has_chroma_node():
                return True
        except Exception as exc:
            logger.debug(f"Image model runtime support probe failed: {exc}", category=logger.CATEGORY_IMAGE_GEN)
        return False

    def _prepare_image_model_support_async(self, model_filename: str, missing_items: list[str]) -> None:
        self._img_support_prep_in_progress = True
        self._set_image_generate_button_running(True)
        if hasattr(self, "_img_stop_btn"):
            self._img_stop_btn.configure(state="disabled")
        if hasattr(self, "_img_save_btn"):
            self._img_save_btn.configure(state="disabled")
        details = ", ".join(missing_items)
        self._img_set_status(f"Preparing image model support: {details} ...", color=WARN_TEXT)
        logger.info(
            f"Preparing Image Gen support for {model_filename}: {details}",
            category=logger.CATEGORY_IMAGE_GEN,
        )
        self._img_safe_clear_display(
            "Preparing Image Gen support files ...\n\n"
            "Generation will continue automatically when support is ready.",
            WARN_TEXT,
        )
        self._img_show_progress(mode="indeterminate", color=WARN_TEXT)

        def _worker():
            ok = False
            err = ""
            try:
                ok = self._ensure_image_model_runtime_support(model_filename, prompt=False)
                if not ok:
                    err = "Could not prepare required Image Gen support files."
            except Exception as exc:
                err = str(exc)
            self.after(0, lambda: self._image_model_support_prepared(ok, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _image_model_support_prepared(self, ok: bool, err: str = "") -> None:
        self._img_support_prep_in_progress = False
        if not ok:
            self._set_image_generate_button_running(False)
            self._img_stop_progress()
            self._img_set_status(f"Image model support setup failed: {err}", color=ERROR_TEXT)
            self._img_safe_clear_display(f"Image model support setup failed:\n\n{err}", ERROR_TEXT)
            logger.error(f"Image model support setup failed: {err}", category=logger.CATEGORY_IMAGE_GEN)
            return
        self._img_set_status("Image model support ready. Starting generation ...", color=SUCCESS_TEXT)
        logger.info("Image model support ready; starting queued generation", category=logger.CATEGORY_IMAGE_GEN)
        self.after(100, self._start_image_generation)

    def _ensure_selected_image_checkpoint_present(self, model_filename: str) -> Optional[bool]:
        """Silently auto-download a missing Image Gen checkpoint, mirroring
        the chat-model UX where Ollama models auto-pull on first use.

        Returns:
            None  — nothing to do (file present, no catalog entry, or no URL),
                    caller should continue with generation as normal.
            True  — download started in the background; caller MUST return and
                    let ``_image_checkpoint_downloaded`` re-enter Generate.
            False — caught a permanent error (caller has already surfaced the
                    error message and reset the button state) and should return.

        v2026.06.01.9: previously a user could pick a downloadable catalog
        model in the dropdown and click Generate, only to fail with a
        ComfyUI-side "model not found" error and no recovery path inside the
        app. The Benchmark tab already auto-pulls via
        ``_bench_prepare_image_model`` — this method reuses the same helper
        from the Image Gen tab so both surfaces behave the same.
        """
        if not model_filename:
            return None
        entry = self._find_catalog_entry_for_model(model_filename)
        if not entry or not entry.get("comfyui_model_url"):
            return None
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            return None
        try:
            target_path = self._comfyui_model_download_target(entry, comfyui_path)
        except Exception as exc:
            logger.debug(
                f"Could not resolve image checkpoint target for {model_filename}: {exc}",
                category=logger.CATEGORY_IMAGE_GEN,
            )
            return None
        if target_path.exists():
            return None
        self._download_image_checkpoint_async(entry)
        return True

    def _download_image_checkpoint_async(self, model_entry: dict) -> None:
        """Background worker that pulls a missing Image Gen checkpoint via
        ``_bench_prepare_image_model`` (same helper the Benchmark tab uses)
        and re-enters ``_start_image_generation`` on success.
        """
        self._img_checkpoint_download_in_progress = True
        self._set_image_generate_button_running(True)
        if hasattr(self, "_img_stop_btn"):
            self._img_stop_btn.configure(state="disabled")
        if hasattr(self, "_img_save_btn"):
            self._img_save_btn.configure(state="disabled")
        friendly = model_entry.get("name") or model_entry.get("comfyui_model") or "checkpoint"
        size_gb = model_entry.get("size_gb")
        size_hint = f" (~{size_gb:.1f} GB)" if isinstance(size_gb, (int, float)) and size_gb else ""
        self._img_set_status(
            f"Downloading {friendly}{size_hint}…",
            color=WARN_TEXT,
        )
        logger.info(
            f"Image Gen tab: auto-downloading checkpoint {model_entry.get('comfyui_model', '?')}",
            category=logger.CATEGORY_IMAGE_GEN,
        )
        self._img_safe_clear_display(
            f"Downloading {friendly}{size_hint}…\n\n"
            "Generation will start automatically when the download completes.",
            WARN_TEXT,
        )
        self._img_show_progress(mode="indeterminate", color=WARN_TEXT)

        def _worker():
            ok = False
            err = ""
            try:
                ok, err = self._bench_prepare_image_model(model_entry, self._img_stop_event)
            except Exception as exc:
                ok = False
                err = str(exc) or "Unknown error during image checkpoint download"
            self.after(0, lambda: self._image_checkpoint_downloaded(ok, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _image_checkpoint_downloaded(self, ok: bool, err: str = "") -> None:
        self._img_checkpoint_download_in_progress = False
        if not ok:
            self._set_image_generate_button_running(False)
            self._img_stop_progress()
            message = err or "Could not download the selected image checkpoint."
            self._img_set_status(f"Image checkpoint download failed: {message}", color=ERROR_TEXT)
            self._img_safe_clear_display(
                f"Image checkpoint download failed:\n\n{message}",
                ERROR_TEXT,
            )
            logger.error(
                f"Image Gen tab auto-download failed: {message}",
                category=logger.CATEGORY_IMAGE_GEN,
            )
            return
        self._img_set_status("Checkpoint ready. Starting generation…", color=SUCCESS_TEXT)
        logger.info(
            "Image Gen tab: checkpoint download finished; resuming Generate",
            category=logger.CATEGORY_IMAGE_GEN,
        )
        self.after(100, self._start_image_generation)

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
        run_kw = {"capture_output": True, "text": True, "timeout": 30}
        if sys.platform == "win32":
            run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            [python_exe, "-c", check_code],
            **run_kw,
        )
        if result.returncode == 0:
            return []
        if result.returncode == 1:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"dependency probe failed with exit code {result.returncode}")

    def _ensure_comfyui_core_dependencies(self, python_exe: str) -> bool:
        """Repair small ComfyUI runtime deps without rerunning torch-heavy requirements.

        v5.3.6+ diagnostics: on any failure path, stamps
        ``self._comfyui_last_start_failure_reason`` with the specific reason
        (probe-raised / pip-install-failed / still-missing-after-install)
        including the last line of pip's stderr where applicable, so the
        benchmark log surfaces the *actual* cause instead of "ComfyUI is not
        running and could not be started".
        """
        cache = self.__dict__.get("_comfyui_dependency_ok_by_python", {})
        cache_key = str(python_exe)
        if cache.get(cache_key):
            return True
        modules = list(COMFYUI_CORE_PYTHON_DEPS)
        try:
            missing = self._missing_comfyui_python_modules(python_exe, modules)
        except Exception as e:
            reason = f"ComfyUI dependency probe failed: {e}"
            logger.error(f"Could not verify ComfyUI Python dependencies: {e}", category=logger.CATEGORY_COMFYUI)
            self._set_comfyui_start_failure_reason(reason)
            return False

        if not missing:
            cache[cache_key] = True
            self._comfyui_dependency_ok_by_python = cache
            return True
        cache.pop(cache_key, None)
        self._comfyui_dependency_ok_by_python = cache

        lock = self.__dict__.get("_comfyui_dependency_lock")
        if lock is None:
            lock = threading.Lock()
            self._comfyui_dependency_lock = lock
        with lock:
            try:
                missing = self._missing_comfyui_python_modules(python_exe, modules)
            except Exception as e:
                reason = f"ComfyUI dependency probe failed after lock: {e}"
                logger.error(f"Could not verify ComfyUI Python dependencies: {e}", category=logger.CATEGORY_COMFYUI)
                self._set_comfyui_start_failure_reason(reason)
                return False
            if not missing:
                cache[cache_key] = True
                self._comfyui_dependency_ok_by_python = cache
                return True

            packages = sorted({COMFYUI_CORE_PYTHON_DEPS[module] for module in missing})
            logger.warning(
                "ComfyUI Python dependencies missing "
                f"({', '.join(missing)}); installing {', '.join(packages)}",
                category=logger.CATEGORY_COMFYUI,
            )
            self.after(0, lambda: self._set_comfyui_status("installing missing Python deps …", WARN_TEXT))
            run_kw = {"capture_output": True, "text": True, "timeout": 300}
            if sys.platform == "win32":
                run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
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
                **run_kw,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                logger.error(
                    f"Failed to install ComfyUI Python dependencies: {detail[-1000:]}",
                    category=logger.CATEGORY_COMFYUI,
                )
                # Last non-empty line of pip stderr is usually the actionable
                # "ERROR: …" or "No matching distribution found for …" line.
                last = ""
                for line in reversed(detail.splitlines()):
                    if line.strip():
                        last = line.strip()
                        break
                self._set_comfyui_start_failure_reason(
                    f"ComfyUI dependency install failed: {last or 'pip exited with code ' + str(result.returncode)}"
                )
                return False

            try:
                still_missing = self._missing_comfyui_python_modules(python_exe, modules)
            except Exception as e:
                reason = f"ComfyUI dependency re-probe after install failed: {e}"
                logger.error(
                    f"Could not verify ComfyUI Python dependencies after install: {e}",
                    category=logger.CATEGORY_COMFYUI,
                )
                self._set_comfyui_start_failure_reason(reason)
                return False
            if still_missing:
                logger.error(
                    "ComfyUI Python dependencies still missing after install: "
                    + ", ".join(still_missing),
                    category=logger.CATEGORY_COMFYUI,
                )
                self._set_comfyui_start_failure_reason(
                    f"ComfyUI dependencies still missing after install attempt: "
                    f"{', '.join(still_missing)}"
                )
                cache.pop(cache_key, None)
                self._comfyui_dependency_ok_by_python = cache
                return False
            logger.info("ComfyUI Python dependencies verified", category=logger.CATEGORY_COMFYUI)
            cache[cache_key] = True
            self._comfyui_dependency_ok_by_python = cache
            return True

    def _comfyui_model_download_target(self, model: dict, comfyui_path: Path) -> Path:
        """Return the ComfyUI destination path for a catalog image model."""
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
                logger.info(f"Moving GGUF model from checkpoints/ to diffusion_models/: {filename}")
                shutil.move(str(legacy_file), str(output_file))
        return output_file

    def _bench_prepare_image_model(self, model: dict, stop_event: Optional[threading.Event] = None) -> tuple[bool, str]:
        """Ensure benchmark image models and runtime support are present before queueing."""
        filename = model.get("comfyui_model", "")
        url = model.get("comfyui_model_url", "")
        if not filename:
            return False, "No comfyui_model filename defined for this catalog entry"

        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            return False, "ComfyUI not installed at expected paths"

        missing_support = self._image_model_runtime_support_missing_items(filename)
        if not self._ensure_image_model_runtime_support(filename, prompt=False):
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
                import requests
                logger.info(
                    f"Benchmark downloading ComfyUI model: {filename}",
                    category=logger.CATEGORY_IMAGE_GEN,
                )
                logger.info(
                    f"Benchmark ComfyUI download destination: {output_file}",
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

    def download_comfyui_model(self, model: dict):
        """Automatically download a ComfyUI image model."""
        # v5.4: Mirror the user-added install gate from start_download so
        # neither the chat nor the image-gen path can bypass it.
        if model.get("user_added") and not self._user_added_model_can_install(model):
            messagebox.showinfo(
                "Can't download automatically",
                f"'{model.get('name') or model.get('id')}' was added from "
                "Hugging Face for reference, but LocalAI doesn't have a "
                "one-click downloader for its backend. To install it, use "
                "the source link in the detail pane to grab the files "
                "manually, or look for a single-file (.safetensors or "
                ".gguf) release of the same model.",
                parent=self,
            )
            return
        url = model.get("comfyui_model_url", "")
        filename = model.get("comfyui_model", "")

        if not url or not filename:
            manual_url = model.get("comfyui_manual_url", "")
            if manual_url:
                msg = (
                    f"'{model['name']}' must be downloaded manually.\n\n"
                    f"Once downloaded, rename the file to:\n"
                    f"  {filename}\n\n"
                    f"and place it in your ComfyUI models/checkpoints folder.\n\n"
                    f"Open the download page in your browser?"
                )
                if messagebox.askyesno("Manual Download Required", msg, parent=self):
                    import webbrowser
                    webbrowser.open(manual_url)
            else:
                messagebox.showinfo(
                    "Manual Download Required",
                    f"'{model['name']}' must be downloaded manually.\n"
                    f"Check the model description for instructions.\n\n"
                    f"Once downloaded, place '{filename}' in:\n"
                    f"  ComfyUI/models/checkpoints/",
                    parent=self,
                )
            return

        # Check if ComfyUI is installed
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            messagebox.showerror(
                "ComfyUI not installed",
                "ComfyUI must be installed to download image models.\n"
                "Run setup.sh to install ComfyUI." if sys.platform == "darwin" else "Run setup.bat to install ComfyUI.",
                parent=self,
            )
            return

        output_file = self._comfyui_model_download_target(model, comfyui_path)
        model_dir = output_file.parent

        # Flux models need CLIP encoders + VAE; GGUF also needs the GGUF custom node.
        # Z-Image models (z_image_bf16, z_image_turbo_bf16) need Qwen text encoder + Flux VAE.
        _lower = filename.lower()
        is_gguf = _lower.endswith(".gguf")
        is_flux = 'flux' in _lower or 'chroma' in _lower
        is_z_image = 'z_image' in _lower
        is_chroma = 'chroma' in _lower
        if is_gguf:
            if not self._ensure_gguf_support(comfyui_path):
                return
        elif is_z_image:
            if not self._ensure_z_image_support(comfyui_path):
                return
        elif is_chroma:
            self._ensure_chroma_support(comfyui_path)
            if not self._ensure_flux_clip_vae(comfyui_path):
                return
        elif is_flux:
            if not self._ensure_flux_clip_vae(comfyui_path):
                return

        # Check if already downloaded
        if output_file.exists():
            proceed = messagebox.askyesno(
                "File exists",
                f"'{filename}' already exists.\n\nDownload again?",
                parent=self,
            )
            if not proceed:
                # Model already downloaded — go to Image Gen page
                self.open_image_gen_for_model(model)
                self._img_refresh_comfyui()
                return

        # Check disk space (estimate 5-10 GB for typical checkpoint)
        st = system_info.get_storage_info(model_dir)
        if st["free_gb"] < 5:
            messagebox.showerror(
                "Not enough disk space",
                f"Downloading '{model['name']}' needs ~5-10 GB.\n"
                f"Only {st['free_gb']:.1f} GB free on disk.\n"
                "Free up space and try again.",
                parent=self,
            )
            return

        # Check if download already in progress
        if self._download_thread and self._download_thread.is_alive():
            messagebox.showinfo(
                "Download in progress",
                "Wait for the current download to finish.",
                parent=self
            )
            return

        self._stop_event.clear()
        self._show_progress(True)
        self.set_status(f"Downloading {model['name']} …")
        logger.info(f"Starting ComfyUI model download: {filename}")
        logger.info(f"URL: {url}")
        logger.info(f"Destination: {output_file}")

        def _do_download():
            try:
                import requests
                # Start download with streaming
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                last_ui_update = 0.0

                def _progress(done_bytes: int):
                    nonlocal downloaded, last_ui_update
                    downloaded = done_bytes

                    # Throttle UI updates to at most 2 per second so the
                    # callback queue doesn't grow to millions of entries.
                    now = time.monotonic()
                    if now - last_ui_update >= 0.5:
                        last_ui_update = now
                        if total_size > 0:
                            pct = downloaded / total_size
                            pct_str = f"{pct * 100:.0f}%  ({_fmt_bytes(downloaded)} / {_fmt_bytes(total_size)})"
                        else:
                            pct = 0
                            pct_str = f"{_fmt_bytes(downloaded)} downloaded"
                        msg = f"Downloading {filename}: {pct_str}"
                        self.after(0, lambda m=msg, p=pct: (
                            self.set_status(m),
                            self._progress_bar.set(min(p, 1.0)),
                        ))

                self._download_stream_to_path(
                    response, output_file, stop_event=self._stop_event,
                    progress_cb=_progress,
                )

                logger.info(f"Download complete: {filename}")
                self.after(0, lambda: self._comfyui_download_done(model, True))

            except RuntimeError as e:
                if str(e) == "Download cancelled":
                    logger.info("Download cancelled by user")
                    self.after(0, lambda: self._comfyui_download_done(
                        model, False, "Download cancelled"
                    ))
                    return
                logger.error(f"ComfyUI model download failed: {e}")
                self.after(0, lambda err=str(e): self._comfyui_download_done(
                    model, False, err
                ))
            except Exception as e:
                logger.error(f"ComfyUI model download failed: {e}")
                self.after(0, lambda err=str(e): self._comfyui_download_done(
                    model, False, err
                ))

        self._download_thread = threading.Thread(target=_do_download, daemon=True)
        self._download_thread.start()

    def _comfyui_download_done(self, model: dict, success: bool, error: str = ""):
        """Handle completion of ComfyUI model download."""
        self._show_progress(False)

        if success:
            self.set_status(f"'{model['name']}' downloaded successfully.")
            logger.info(f"ComfyUI model ready: {model.get('comfyui_model', '')}")

            # Check if this was a GGUF model - might need to restart ComfyUI to load custom node
            filename = model.get('comfyui_model', '')
            is_gguf = filename.lower().endswith('.gguf')

            # Restart ComfyUI if needed (for GGUF custom node to load)
            if is_gguf and self.comfyui_process:
                self.set_status("Restarting ComfyUI to load GGUF support…")
                logger.info("Restarting ComfyUI to load GGUF custom node")

                def _restart():
                    # Terminate current ComfyUI
                    if self.comfyui_process and self.comfyui_process.poll() is None:
                        try:
                            self.comfyui_process.terminate()
                            self.comfyui_process.wait(timeout=10)
                        except Exception:
                            self.comfyui_process.kill()
                        self.comfyui_process = None

                    time.sleep(2)
                    # Restart with refresh
                    self.after(0, self._img_refresh_comfyui)

                threading.Thread(target=_restart, daemon=True).start()
            elif self.comfyui.is_running():
                # Just refresh model list
                self.set_status("Refreshing ComfyUI models…")
                threading.Thread(target=lambda: (
                    time.sleep(2),  # Give file system a moment
                    self.after(0, self._img_refresh_comfyui)
                ), daemon=True).start()
            else:
                messagebox.showinfo(
                    "Download complete",
                    f"'{model['name']}' downloaded successfully!\n\n"
                    "Go to the Image Generation page and click Refresh to use it.",
                    parent=self
                )
        else:
            self.set_status("Download failed.")
            messagebox.showerror(
                "Download failed",
                f"Failed to download '{model['name']}':\n\n{error}\n\n"
                "Check your internet connection and try again.",
                parent=self
            )

        # Refresh model cards to update status
        for card in self._model_cards:
            if card.model == model:
                card.refresh_status(comfyui_model_names=self._get_cached_comfyui_model_names(force_refresh=True))

        # v5.5.0 UX fix: also refresh the right-pane detail view if the
        # just-installed model is the currently-selected row. Without this,
        # the install state on the right pane stays stale until the user
        # clicks elsewhere and clicks back — the "click away and back"
        # workaround Ron reported. Per perf review: call
        # ``_update_model_detail`` directly instead of routing through
        # ``_refresh_visible_model_status_from_snapshots`` (the per-card
        # refresh above already handled the card list).
        try:
            selected_id = getattr(self, "_selected_model_id", None)
            if selected_id and selected_id == model.get("id"):
                self._update_model_detail(
                    comfyui_model_names=self._get_cached_comfyui_model_names(force_refresh=False),
                )
        except Exception:
            # The detail pane may not be built yet if the install completes
            # before the Models page has been visited — degrade silently.
            pass

        # v5.5.4 safety net for the install-staleness Ron reported: even
        # with the immediate refresh above, ComfyUI's ``/get_model_list``
        # HTTP endpoint sometimes returns a cached list that doesn't yet
        # see the just-downloaded file (a few seconds of OS-level fs
        # propagation + ComfyUI's own watcher latency). Schedule a second
        # forced refresh ~3 s later so the right pane and card row catch
        # up without the user having to click another row. Cheap and
        # idempotent — if ComfyUI already saw the file on the first pass,
        # the second pass paints the same state and nothing changes.
        try:
            self.after(3000, self._refresh_model_cards)
        except Exception:
            pass

    def open_image_gen_for_model(self, model: dict):
        """Navigate to the Image Gen page and pre-select *model*."""
        # Use the catalog's friendly name so it matches the dropdown's display labels.
        # If ComfyUI hasn't refreshed yet the dropdown is still showing a placeholder;
        # the next _img_on_comfyui_ready call will reconcile this against the file list.
        display_name = model.get("name") or model.get("comfyui_model", "")
        self._switch_page("image_gen")
        self._img_model_var.set(display_name)

    def _replace_textbox_text(self, textbox, text: str) -> None:
        textbox.delete("1.0", "end")
        textbox.insert("1.0", text)

    def _apply_chat_demo_prompt(self, model: dict | None) -> None:
        if not model or not hasattr(self, "_input_box") or self._input_box is None:
            return
        if self._chat_thinking:
            return
        demo = model_demos.get_model_demo(model)
        self._replace_textbox_text(self._input_box, demo["primary"])
        self.set_status(f"Loaded a {demo['feature']} sample prompt for {model.get('name', 'the model')}.")

    def _apply_image_demo_prompt(self, model: dict | None) -> None:
        if not model or not hasattr(self, "_img_prompt") or self._img_prompt is None:
            return
        demo = model_demos.get_model_demo(model)
        self._replace_textbox_text(self._img_prompt, demo["primary"])
        if hasattr(self, "_img_prompt_status") and self._img_prompt_status is not None:
            self._img_prompt_status.configure(
                text=f"Loaded a {demo['feature']} sample for {model.get('name', 'this model')}. Edit it or click Generate."
            )

    def run_model_demo(self, model: dict):
        """Run the catalog's one-click utility demo for OCR, speech, embeddings, and document AI."""
        demo = model_demos.get_model_demo(model)
        self.set_status(f"Running {demo['feature']} demo for {model.get('name', 'model')} ...")

        def _run():
            try:
                from src.phase1_adapters import run_transformers_adapter

                root = Path(__file__).parent.parent / "demo_results"
                result = run_transformers_adapter(model, root)
                self.after(0, lambda r=result: self._model_demo_done(model, demo, r))
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda: (
                    self.set_status("Model demo failed."),
                    messagebox.showerror(
                        "Model demo failed",
                        f"{model.get('name', 'Model')} could not run its sample demo.\n\n{msg}",
                        parent=self,
                    ),
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _model_demo_done(self, model: dict, demo: dict, result: dict) -> None:
        output = str(result.get("output_text") or "Demo completed.")
        artifact = result.get("image") or result.get("audio")
        if artifact:
            try:
                webbrowser.open(Path(artifact).resolve().as_uri())
            except Exception:
                pass
        self.set_status(f"{demo['feature']} demo complete for {model.get('name', 'model')}.")
        messagebox.showinfo(
            "Demo complete",
            f"{model.get('name', 'Model')} — {demo['feature']}\n\n{output[:1800]}",
            parent=self,
        )

    # ── Image Generation page ─────────────────────────────────────────────────

    # v5.1: shared visual constants for the Image Gen page. Keeps the design
    # system honest — every card uses the same accents/radii/typography.
    _IG_CARD_FG       = SURFACE_CARD               # outer card surface
    _IG_INNER_FG      = SURFACE_INNER              # inner control surface
    _IG_BORDER        = BORDER_STRONG              # AA-visible control/card border
    _IG_ACCENT        = "#5aa2ff"                   # primary brand blue
    _IG_ACCENT_HOVER  = "#3f86d8"
    _IG_HERO          = "#2a6a4a"                   # generate-button green
    _IG_HERO_HOVER    = "#1f5036"
    _IG_SUCCESS       = "#2d8a4e"
    _IG_SUCCESS_HOVER = "#236b3c"
    _IG_DANGER        = "#a83a3a"
    _IG_ACCENT_TEXT   = ("#0f172a", "#0f172a")
    _IG_DANGER_TEXT   = ("#ffffff", "#ffffff")
    _IG_DISABLED_FG   = ("#cbd5e1", "#475569")
    _IG_DISABLED_TEXT = ("#111827", "#f9fafb")
    _IG_MUTED_FG      = TEXT_MUTED
    _IG_SUBHEAD_FG    = TEXT_SECONDARY
    _IG_BADGE_BG_REC  = ("#cdebd1", "#1e4a28")     # Recommended chip
    _IG_BADGE_BG_ACC  = ("#dde8ff", "#1f3260")     # Highest-Accuracy chip
    _IG_BADGE_BG_EFF  = ("#fff0d6", "#4a3a1a")     # Efficient-Multimodal chip
    _IG_RADIUS        = 14
    _IG_GAP           = 14

    def _option_menu_style(self) -> dict:
        """Shared contrast-safe styling for selected/dropdown option menus."""
        return {
            "fg_color": INPUT_SURFACE,
            "text_color": TEXT_PRIMARY,
            "button_color": self._IG_ACCENT,
            "button_hover_color": self._IG_ACCENT_HOVER,
            "dropdown_fg_color": INPUT_SURFACE,
            "dropdown_text_color": TEXT_PRIMARY,
            "dropdown_hover_color": ("#e8f1ff", "#334155"),
        }

    def _outline_button_style(self, corner_radius: int = 8, text_color=None) -> dict:
        """Palette-backed outline buttons avoid transparent rounded-corner artifacts."""
        return {
            "fg_color": INPUT_SURFACE,
            "hover_color": ("#eef4ff", "#334155"),
            "border_width": 1,
            "border_color": self._IG_BORDER,
            "text_color": text_color or TEXT_PRIMARY,
            "text_color_disabled": TEXT_DISABLED,
            "corner_radius": corner_radius,
        }

    def _solid_button_style(self, fg_color: str, hover_color: str, text_color=("#ffffff", "#ffffff")) -> dict:
        return {
            "fg_color": fg_color,
            "hover_color": hover_color,
            "text_color": text_color,
            "text_color_disabled": self._IG_DISABLED_TEXT,
            "border_width": 1,
            "border_color": fg_color,
            "corner_radius": 8,
        }

    def _set_image_generate_button_running(self, running: bool) -> None:
        if not hasattr(self, "_img_generate_btn") or self._img_generate_btn is None:
            return
        if running:
            self._img_generate_btn.configure(
                state="disabled",
                fg_color=self._IG_DISABLED_FG,
                text_color_disabled=self._IG_DISABLED_TEXT,
            )
        else:
            # v5.5.0 UX fix: restore the original button label on re-enable.
            # _immediate_disable_btn may have flipped the text to "Starting…";
            # _set_image_generate_button_running(False) is the canonical place
            # to restore "✨  Generate".
            self._img_generate_btn.configure(
                state="normal",
                text="✨  Generate",
                fg_color=self._IG_HERO,
                hover_color=self._IG_HERO_HOVER,
                text_color=("#ffffff", "#ffffff"),
                text_color_disabled=self._IG_DISABLED_TEXT,
            )

    def _immediate_disable_btn(
        self,
        btn,
        *,
        text: Optional[str] = None,
        status_setter=None,
        status_text: Optional[str] = None,
    ) -> None:
        """Surgical immediate-disable for hero buttons (v5.5.0 UX fix).

        Generate / Start Benchmark / Retry Failed buttons MUST give the user
        immediate click feedback. This helper:
          1. Flips ``state="disabled"`` and optionally updates the button text
             (e.g., "Starting…").
          2. Optionally updates the page's designated screen-reader status
             surface (``_img_set_canvas_status`` for Image Gen,
             ``_enqueue_bench_log`` for Benchmark) so AT users also get the
             signal — Tk has no aria-live equivalent on Windows MSAA.
          3. Flushes the paint queue via ``update_idletasks()`` so the user
             sees the state change *before* any synchronous pre-flight
             validation runs.

        Per a11y review: focus is NOT moved when the button disables; Tk
        leaves focus in place, which is the correct behaviour. Per design
        review: no Unicode spinner, no color-shift state — the disabled grey
        + "Starting…" label is sufficient affordance.
        """
        if btn is None:
            return
        try:
            kwargs = {"state": "disabled"}
            if text is not None:
                kwargs["text"] = text
            btn.configure(**kwargs)
        except Exception:
            pass
        if status_setter is not None and status_text:
            try:
                status_setter(status_text)
            except Exception:
                pass
        try:
            self.update_idletasks()
        except Exception:
            pass

    def _reset_bench_start_btn(self) -> None:
        """Restore the Start Benchmark button label/state after early return."""
        btn = self.__dict__.get("_bench_start_btn")
        if btn is None:
            return
        try:
            btn.configure(state="normal", text="Start Benchmark")
        except Exception:
            pass

    def _reset_bench_retry_btn(self) -> None:
        """Restore the Retry Failed button label/state after early return."""
        btn = self.__dict__.get("_bench_retry_btn")
        if btn is None:
            return
        try:
            btn.configure(state="normal", text="Retry Failed")
        except Exception:
            pass

    def _default_negative_prompt_for_entry(self, entry: dict | None) -> str:
        if not entry:
            return ""
        prompts = self.cfg.get("default_negative_prompts", {})
        if not isinstance(prompts, dict):
            return ""
        for key in (entry.get("id"), entry.get("comfyui_model"), entry.get("name")):
            if isinstance(key, str) and key in prompts:
                value = prompts.get(key, "")
                return value.strip() if isinstance(value, str) else ""
        return ""

    def _selected_model_default_negative_prompt(self) -> str:
        return self._default_negative_prompt_for_entry(self._selected_image_model_catalog_entry())

    def _apply_selected_model_negative_prompt(self) -> None:
        if not hasattr(self, "_img_neg_prompt") or self._img_neg_prompt is None:
            return
        negative = self._selected_model_default_negative_prompt()
        self._img_neg_prompt.delete(0, "end")
        if negative:
            self._img_neg_prompt.insert(0, negative)
        try:
            self._img_neg_prompt.configure(placeholder_text=negative or "Negative prompt ignored for this model")
        except Exception:
            pass

    def _apply_img2img_default_prompts(self) -> None:
        """Pre-populate the positive + negative prompt with the documented
        img2img sample so the user has a known-good starting point the
        moment they tick "Use reference image for generation".

        Strings live as ``IMG2IMG_DEFAULT_POSITIVE`` / ``IMG2IMG_DEFAULT_NEGATIVE``
        at module top so the prompt builder in ``docs/image-gen-guide.html``
        section 5.3 can stay byte-identical (the doc test pins both directions).
        Users who want a different man/woman/age/lighting/clothing combo are
        directed to the doc's prompt builder via the page status line.
        """
        if hasattr(self, "_img_prompt") and self._img_prompt is not None:
            self._replace_textbox_text(self._img_prompt, IMG2IMG_DEFAULT_POSITIVE)
        if hasattr(self, "_img_neg_prompt") and self._img_neg_prompt is not None:
            self._img_neg_prompt.delete(0, "end")
            self._img_neg_prompt.insert(0, IMG2IMG_DEFAULT_NEGATIVE)
            try:
                self._img_neg_prompt.configure(placeholder_text=IMG2IMG_DEFAULT_NEGATIVE)
            except Exception:
                pass

    def _build_image_page_snapdragon_unsupported(self):
        """Static panel shown on Windows ARM64 (Snapdragon X) Image Generation.

        ComfyUI's startup imports ``torchaudio`` for its audio nodes, and
        ``torch-directml`` has no Windows-ARM64 wheel on PyPI. Both make image
        generation unrunnable on Snapdragon X. Instead of letting the user
        click Generate and watch ComfyUI crash with an
        ``Entry Point Not Found: torch_library_impl`` Windows popup, we render
        a clear "unsupported" explanation with what *does* still work on
        Snapdragon (chat, vision, ONNX, embeddings, speech).
        """
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["image_gen"] = page
        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(
            page,
            fg_color=self._IG_INNER_FG if hasattr(self, "_IG_INNER_FG") else "transparent",
            corner_radius=14,
        )
        card.grid(row=0, column=0, padx=40, pady=40, sticky="n")
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text="Image Generation is unavailable on Snapdragon X",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, padx=24, pady=(24, 6), sticky="w")

        ctk.CTkLabel(
            card,
            text=(
                "ComfyUI uses PyTorch + DirectML for image generation, and the "
                "torch-directml package on PyPI ships x64-only wheels — there "
                "is no Windows-ARM64 wheel today. As a result, ComfyUI cannot "
                "start on a Snapdragon X system and image generation is "
                "disabled."
            ),
            font=ctk.CTkFont(size=13), wraplength=620, justify="left",
        ).grid(row=1, column=0, padx=24, pady=(0, 14), sticky="w")

        ctk.CTkLabel(
            card,
            text="What still works on this machine",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=2, column=0, padx=24, pady=(8, 4), sticky="w")
        ctk.CTkLabel(
            card,
            text=(
                "  •  Chat models via Ollama (Windows-ARM64 builds available)\n"
                "  •  Vision, embeddings, document and speech models via ONNX Runtime\n"
                "  •  Benchmark page (Quick / Extended), Models page, Playground"
            ),
            font=ctk.CTkFont(size=12), justify="left",
        ).grid(row=3, column=0, padx=24, pady=(0, 14), sticky="w")

        ctk.CTkLabel(
            card,
            text="If you have an Intel or AMD AI PC",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=4, column=0, padx=24, pady=(8, 4), sticky="w")
        ctk.CTkLabel(
            card,
            text=(
                "DirectML image generation IS supported on x64 Intel / AMD AI PCs. "
                "If you reach this panel on an x64 system the architecture detection is "
                "wrong — please file an issue. If torch and torchaudio have drifted out "
                "of ABI sync on a supported system (symptom: \"Entry Point Not Found: "
                "torch_library_impl\"), run fix_directml_pytorch.bat from the install "
                "folder to realign them."
            ),
            font=ctk.CTkFont(size=12), wraplength=620, justify="left",
        ).grid(row=5, column=0, padx=24, pady=(0, 24), sticky="w")

    def _build_image_page(self):
        """v5.1 Apple-designer two-zone Image Gen page.

        Zone A (top card on the left, scrollable column): **Generate** — the
        image-generation flow (model, prompt, settings, hero Generate button).

        Zone B (second card on the left): **Vision-assisted prompting** — the
        optional reference-image → vision-model → Analyze-to-prompt helper.

        Right column: the generated image canvas + an info strip.

        v5.5.9 (Ron, 2026-05-26): On Windows ARM64 (Snapdragon X), torch-directml
        is x64-only on PyPI so ComfyUI cannot start. We render a static
        "unsupported on Snapdragon ARM64" panel instead of the full UI so the
        user sees a friendly explanation rather than the
        ``torch_library_impl could not be located in _torchaudio.pyd`` popup
        from ComfyUI's startup torchaudio import.
        """
        if is_snapdragon_arm64():
            self._build_image_page_snapdragon_unsupported()
            return
        page = ctk.CTkFrame(self._content, corner_radius=0, fg_color="transparent")
        self._pages["image_gen"] = page
        page.grid_rowconfigure(1, weight=1)
        # Left column = scrollable controls (fixed width). Right column = canvas.
        page.grid_columnconfigure(0, weight=0, minsize=540)
        page.grid_columnconfigure(1, weight=1, minsize=380)

        # ── HEADER STRIP ────────────────────────────────────────────────────
        # Light, clean — title + tiny subtitle + ComfyUI status pill on the
        # right with a kebab-style row of actions. No big buttons up here so the
        # eye lands on the content cards below.
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=20, pady=(16, 4))
        hdr.grid_columnconfigure(0, weight=1)

        title_col = ctk.CTkFrame(hdr, fg_color="transparent")
        title_col.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_col, text="Image Generation",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="top", anchor="w")
        ctk.CTkLabel(
            title_col, text="Create with ComfyUI — or describe an image and let a vision model write the prompt for you.",
            font=ctk.CTkFont(size=12), text_color=self._IG_SUBHEAD_FG,
        ).pack(side="top", anchor="w", pady=(2, 0))

        # Right-aligned status + action cluster
        action_col = ctk.CTkFrame(hdr, fg_color="transparent")
        action_col.grid(row=0, column=1, sticky="e")

        self._img_comfyui_status = ctk.CTkLabel(
            action_col, text="ComfyUI: checking …",
            font=ctk.CTkFont(size=11), text_color=self._IG_SUBHEAD_FG,
            fg_color=self._IG_INNER_FG, corner_radius=10,
            padx=12, pady=6,
        )
        self._img_comfyui_status.pack(side="left", padx=(0, 8))

        # v5: replay any ComfyUI status that arrived before the Image page was lazily built
        if self._last_comfyui_status is not None:
            try:
                text, color = self._last_comfyui_status
                self._img_comfyui_status.configure(text=f"ComfyUI: {text}", text_color=color)
            except Exception:
                pass

        ctk.CTkButton(
            action_col, text="Restart", width=72, height=28,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._img_restart_comfyui,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            action_col, text="Free VRAM", width=82, height=28,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._free_comfyui_vram,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            action_col, text="Help", width=58, height=28,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._open_image_gen_guide,
        ).pack(side="left")

        # CPU-only banner: thin pill below the header on CPU-only hardware.
        if self.gpu_info.gpu_type == "cpu" and not getattr(self, "_gpu_detection_pending", False):
            banner = ctk.CTkFrame(
                page, fg_color=("#fff4dc", "#3a2a1a"),
                corner_radius=10, border_width=1, border_color=("#f0d088", "#5a4828"),
            )
            banner.grid(row=2, column=0, columnspan=2, sticky="ew",
                        padx=20, pady=(0, 8))
            ctk.CTkLabel(
                banner,
                text=(
                    "⚠  CPU-only mode — no GPU detected.  "
                    "Use SD 1.5 at 512×512 (~5–8 min/image).  "
                    "SDXL and Flux are not viable on CPU."
                ),
                font=ctk.CTkFont(size=11),
                text_color=WARN_TEXT,
                anchor="w",
            ).pack(side="left", padx=14, pady=8)

        # ── LEFT COLUMN: scrollable container holding the two zone cards ────
        left = ctk.CTkScrollableFrame(
            page, corner_radius=0, fg_color="transparent",
        )
        left.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(8, 16))
        left.grid_columnconfigure(0, weight=1)

        # Internal state — created here so widgets can bind to them.
        self._img_ref_path: Optional[str] = None
        self._img_ref_photo = None
        self._img_img2img_path: Optional[str] = None
        self._img_img2img_photo = None
        self._img_img2img_var = ctk.BooleanVar(value=False)
        # v5.5.14: pull denoise + aspect-match defaults from persisted config so
        # the user's last choice survives restarts. Config defaults: 0.55 / True.
        try:
            _init_denoise = float(self.cfg.get("img2img_denoise", 0.55))
        except (TypeError, ValueError):
            _init_denoise = 0.55
        _init_denoise = max(0.05, min(1.0, _init_denoise))
        self._img_denoise_var = ctk.DoubleVar(value=_init_denoise)
        self._img_match_aspect_var = ctk.BooleanVar(
            value=bool(self.cfg.get("img2img_match_aspect", True)),
        )
        self._img_friendly_to_filename: dict[str, str] = {}
        self._img_vision_cards: dict[str, dict] = {}   # cat_id -> {frame, badge, ...}
        self._img_prompt_collapsed: bool = False  # imagegen-prompt-collapse default

        # Build the two zone cards.
        gen_card = self._build_image_zone_generate(left)
        gen_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, self._IG_GAP))

        vis_card = self._build_image_zone_vision(left)
        vis_card.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 8))

        # ── RIGHT COLUMN: image display canvas + info strip ─────────────────
        display_card = ctk.CTkFrame(
            page, corner_radius=self._IG_RADIUS, fg_color=self._IG_CARD_FG,
        )
        display_card.grid(row=1, column=1, sticky="nsew",
                          padx=(10, 20), pady=(8, 16))
        # v5.5.0 UX fix: the canvas (display_frame) lives at row=3, not row=2.
        # Row=2 is the optional CPU banner. Previously, grid_rowconfigure(2,
        # weight=1) gave the CPU banner unbounded growth and pushed the Save
        # Image info_row (row=4) below the viewport at 1280×800. Canvas gets
        # weight=1; info_row is pinned at row=4 with weight=0, minsize=44 so
        # the Save Image button is always visible whether the banner is shown
        # or not.
        display_card.grid_rowconfigure(3, weight=1)
        display_card.grid_rowconfigure(4, weight=0, minsize=44)
        display_card.grid_columnconfigure(0, weight=1)

        # Eyebrow + title for the canvas
        canvas_hdr = ctk.CTkFrame(display_card, fg_color="transparent")
        canvas_hdr.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        ctk.CTkLabel(
            canvas_hdr, text="CANVAS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self._IG_MUTED_FG,
        ).pack(side="left")
        ctk.CTkLabel(
            canvas_hdr, text="  ·  preview of the generated image",
            font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG,
        ).pack(side="left")

        self._build_image_prompt_editor(display_card).grid(
            row=1, column=0, sticky="ew", padx=14, pady=(0, 12)
        )

        self._img_cpu_banner = ctk.CTkLabel(
            display_card,
            text="CPU mode — expect 60–180 s per image with SD 1.5 models on a small CPU SKU.",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#fff3cd", "#5c4a1f"),
            text_color=WARN_TEXT,
            corner_radius=8,
            anchor="w",
            padx=12,
            pady=8,
        )
        self._img_cpu_banner.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        self._img_cpu_banner.grid_remove()

        display_frame = ctk.CTkFrame(
            display_card, corner_radius=10, fg_color=self._IG_INNER_FG,
        )
        display_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 14))
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)

        self._img_display = ctk.CTkLabel(
            display_frame,
            text=(
                "Your generated image will appear here.\n\n"
                "Tip: edit the prompt above, then click Generate."
            ),
            font=ctk.CTkFont(size=12),
            text_color=self._IG_MUTED_FG, justify="center",
        )
        self._img_display.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

        # Progress lives in the canvas so generation activity appears where
        # the image or final error will replace it. Vision Analyze→Prompt uses
        # this same row; the left cards are controls only.
        progress_row = ctk.CTkFrame(display_frame, fg_color="transparent")
        progress_row.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 18))
        progress_row.grid_columnconfigure(0, weight=1)
        progress_row.grid_remove()
        self._img_progress_row = progress_row

        self._img_progress_bar = ctk.CTkProgressBar(progress_row, height=10)
        self._img_progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._img_progress_bar.configure(mode="indeterminate")

        self._img_elapsed_lbl = ctk.CTkLabel(
            progress_row, text="0s", font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG,
        )
        self._img_elapsed_lbl.grid(row=0, column=1, sticky="e")

        self._img_status_lbl = ctk.CTkLabel(
            progress_row, text="", font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
            wraplength=720, justify="left",
        )
        self._img_status_lbl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # Info strip + Save action (hidden until an image exists).
        # v5.5.0 UX fix: explicit SURFACE_INNER background + rounded corners
        # so the action strip is visibly distinct from the canvas above it
        # (matches the Benchmark action-bar pattern). Without the contrasting
        # surface the Save Image button floats ambiguously against the dark
        # canvas in dark theme.
        info_row = ctk.CTkFrame(
            display_card, fg_color=self._IG_INNER_FG, corner_radius=8,
        )
        info_row.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))
        info_row.grid_columnconfigure(0, weight=1)

        self._img_info_lbl = ctk.CTkLabel(
            info_row, text="",
            font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG, anchor="w", justify="left",
        )
        self._img_info_lbl.grid(row=0, column=0, sticky="w", padx=(12, 0), pady=8)

        self._img_save_btn = ctk.CTkButton(
            info_row, text="Save Image …", width=120, height=30,
            state="disabled",
            **self._outline_button_style(),
            command=self._save_generated_image,
        )
        self._img_save_btn.grid(row=0, column=1, sticky="e", padx=(8, 12), pady=6)

        # ── Internal state for generation threading ─────────────────────────
        self._img_stop_event = threading.Event()
        self._img_thread: Optional[threading.Thread] = None
        self._img_waiting_for_comfyui_generation: bool = False
        self._analyze_stop_event: threading.Event = threading.Event()
        self._analyze_thread: Optional[threading.Thread] = None
        self._analyze_gen_id: int = 0
        self._analyze_elapsed_start: float = 0.0
        self._analyze_elapsed_timer_id: Optional[str] = None
        self._img_analyze_status = self._img_status_lbl
        self._img_analyze_progress = self._img_progress_bar
        self._img_analyze_elapsed_lbl = self._img_elapsed_lbl
        self._img_bytes: Optional[bytes] = None
        self._img_photo = None
        self._img_gen_id: int = 0
        self._img_last_params: dict = {}
        self._img_elapsed_start: float = 0.0
        self._img_elapsed_timer_id: Optional[str] = None

        # Initial sync — sets readiness label, model dropdown, vision picker visuals
        self._on_img_model_changed()
        self._refresh_image_readiness()
        try:
            self._refresh_vision_picker_ui()
        except Exception:
            pass
        self._refresh_vision_model_ui()

    def _build_image_prompt_editor(self, parent) -> "ctk.CTkFrame":
        prompt_card = ctk.CTkFrame(
            parent, corner_radius=10, fg_color=self._IG_INNER_FG,
            border_width=1, border_color=self._IG_BORDER,
        )
        prompt_card.grid_columnconfigure(0, weight=1)

        prompt_label_row = ctk.CTkFrame(prompt_card, fg_color="transparent")
        prompt_label_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        prompt_label_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            prompt_label_row, text="Prompt",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
        ).grid(row=0, column=0, sticky="w")
        # Collapse toggle sits between the label and Prompt ideas so the user
        # can always re-expand without the prompt textbox being visible.
        # The a11y layer (src/a11y.py) auto-wires Tab focus + Enter/Space
        # activation + the focus ring on this CTkButton at construction time.
        self._img_prompt_collapse_btn = ctk.CTkButton(
            prompt_label_row,
            text="\u25be Hide prompt",
            height=24, width=120,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(text_color=LINK_TEXT),
            command=self._toggle_img_prompt_collapsed,
        )
        self._img_prompt_collapse_btn.grid(row=0, column=1, sticky="e", padx=(0, 6))
        ctk.CTkButton(
            prompt_label_row, text="Prompt ideas", height=24, width=110,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(text_color=LINK_TEXT),
            command=self._open_image_prompts,
        ).grid(row=0, column=2, sticky="e")

        self._img_prompt = ctk.CTkTextbox(
            prompt_card, height=98, wrap="word",
            fg_color=INPUT_SURFACE, border_color=self._IG_BORDER,
            border_width=1, corner_radius=8,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12),
        )
        self._img_prompt.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        self._img_prompt.bind("<Control-Return>", lambda _event: self._start_image_generation())

        self._img_prompt_status = ctk.CTkLabel(
            prompt_card,
            text="Vision prompts appear here immediately after analysis.",
            font=ctk.CTkFont(size=10),
            text_color=self._IG_MUTED_FG,
            anchor="w",
        )
        self._img_prompt_status.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))

        # Re-apply collapsed state in case a future default flips it (or a
        # restart-persisted flag is added per plan §6). Today this is a no-op
        # because the flag defaults to False, but it keeps the contract honest.
        if getattr(self, "_img_prompt_collapsed", False):
            self._apply_img_prompt_collapsed(True)

        return prompt_card

    # ── Image-gen prompt collapse / expand (plan: imagegen-prompt-collapse) ──
    def _apply_img_prompt_collapsed(self, collapsed: bool) -> None:
        """Hide/show the prompt textbox + status row and keep the toggle button
        label in sync so it can never drift. Every collapse/expand — manual or
        automatic — routes through this single method. Do NOT call
        ``grid_remove()`` / ``grid()`` on ``_img_prompt`` from anywhere else.
        """
        self._img_prompt_collapsed = bool(collapsed)
        txt = getattr(self, "_img_prompt", None)
        status = getattr(self, "_img_prompt_status", None)
        btn = getattr(self, "_img_prompt_collapse_btn", None)
        if collapsed:
            if txt is not None:
                try:
                    txt.grid_remove()
                except Exception:
                    pass
            if status is not None:
                try:
                    status.grid_remove()
                except Exception:
                    pass
            if btn is not None:
                try:
                    btn.configure(text="\u25b8 Show prompt")
                except Exception:
                    pass
        else:
            if txt is not None:
                try:
                    txt.grid()
                except Exception:
                    pass
            if status is not None:
                try:
                    status.grid()
                except Exception:
                    pass
            if btn is not None:
                try:
                    btn.configure(text="\u25be Hide prompt")
                except Exception:
                    pass

    def _toggle_img_prompt_collapsed(self) -> None:
        """Manual toggle bound to the collapse button."""
        self._apply_img_prompt_collapsed(
            not getattr(self, "_img_prompt_collapsed", False)
        )

    def _img_autocollapse_prompt_if_unmaximized(self) -> None:
        """After a successful render, collapse the prompt to free vertical
        room for the freshly generated image — but ONLY when the window is
        NOT maximized. ``self.state() == "zoomed"`` is Tk's maximized signal
        on Windows. Routes through ``_apply_img_prompt_collapsed`` so the
        toggle button label stays in sync. No-op if already collapsed or
        maximized. Wrapped in try/except because some Tk states can raise on
        ``state()`` — failing closed (no collapse) is the safe default.
        """
        try:
            if getattr(self, "_img_prompt_collapsed", False):
                return
            try:
                window_state = self.state()
            except Exception:
                window_state = ""
            if window_state == "zoomed":
                return
            self._apply_img_prompt_collapsed(True)
        except Exception:
            pass

    # ── ZONE A — Generate ────────────────────────────────────────────────────
    def _build_image_zone_generate(self, parent) -> "ctk.CTkFrame":
        """Build the GENERATE card: model picker → prompts → settings → hero CTA."""
        card = ctk.CTkFrame(parent, corner_radius=self._IG_RADIUS, fg_color=self._IG_CARD_FG)
        card.grid_columnconfigure(0, weight=1)

        # Eyebrow + title + subtitle
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 4))
        ctk.CTkLabel(
            head, text="GENERATE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self._IG_MUTED_FG,
        ).pack(side="top", anchor="w")
        ctk.CTkLabel(
            head, text="Image generation",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="top", anchor="w", pady=(2, 0))
        ctk.CTkLabel(
            head, text="Pick a model, describe what you want, and click Generate.",
            font=ctk.CTkFont(size=12),
            text_color=self._IG_SUBHEAD_FG,
        ).pack(side="top", anchor="w", pady=(2, 0))

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 18))
        body.grid_columnconfigure(0, weight=1)

        # ── Model selector ──────────────────────────────────────────────
        model_row = ctk.CTkFrame(body, fg_color="transparent")
        model_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        model_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            model_row, text="Image model",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        self._img_model_var = ctk.StringVar(value="")
        self._img_model_menu = ctk.CTkOptionMenu(
            model_row, variable=self._img_model_var,
            values=["(no models — ComfyUI offline)"],
            width=320, dynamic_resizing=False,
            **self._option_menu_style(),
        )
        self._img_model_menu.grid(row=0, column=1, sticky="ew")
        self._img_model_var.trace_add("write", self._on_img_model_changed)

        # ── Negative prompt (compact, secondary) ────────────────────────
        ctk.CTkLabel(
            body, text="Negative prompt   (optional)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
        ).grid(row=3, column=0, sticky="w", pady=(4, 4))
        self._img_neg_prompt = ctk.CTkEntry(
            body, placeholder_text="Model-specific default",
            placeholder_text_color=TEXT_MUTED,
            fg_color=self._IG_INNER_FG, border_color=self._IG_BORDER,
            border_width=1, corner_radius=8,
        )
        self._img_neg_prompt.grid(row=4, column=0, sticky="ew", pady=(0, 14))

        # ── Settings grid (denser, grouped) ─────────────────────────────
        settings = ctk.CTkFrame(
            body, fg_color=self._IG_INNER_FG, corner_radius=10,
            border_width=1, border_color=self._IG_BORDER,
        )
        settings.grid(row=5, column=0, sticky="ew", pady=(0, 14))
        settings.grid_columnconfigure((0, 1), weight=1)

        # Eyebrow for settings
        ctk.CTkLabel(
            settings, text="Generation settings",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 6))

        def _setting_label(parent, text):
            return ctk.CTkLabel(
                parent, text=text, font=ctk.CTkFont(size=10),
                text_color=self._IG_MUTED_FG, anchor="w",
            )

        # Row 1: Size (W/H) + Aspect preset
        size_box = ctk.CTkFrame(settings, fg_color="transparent")
        size_box.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 6))
        _setting_label(size_box, "Size").pack(side="top", anchor="w")
        size_row = ctk.CTkFrame(size_box, fg_color="transparent")
        size_row.pack(side="top", fill="x", pady=(2, 0))
        self._img_width_var = ctk.StringVar(value="512")
        self._img_height_var = ctk.StringVar(value="512")
        ctk.CTkLabel(size_row, text="W", font=ctk.CTkFont(size=10),
                     text_color=self._IG_MUTED_FG).pack(side="left", padx=(0, 2))
        ctk.CTkEntry(
            size_row, textvariable=self._img_width_var, width=68,
            fg_color=INPUT_SURFACE, border_color=self._IG_BORDER,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(size_row, text="H", font=ctk.CTkFont(size=10),
                     text_color=self._IG_MUTED_FG).pack(side="left", padx=(0, 2))
        ctk.CTkEntry(
            size_row, textvariable=self._img_height_var, width=68,
            fg_color=INPUT_SURFACE, border_color=self._IG_BORDER,
        ).pack(side="left", padx=(0, 12))

        self._aspect_presets = {
            "Square 1:1": (1024, 1024),
            "Landscape 16:9": (1344, 768),
            "Landscape 3:2": (1216, 832),
            "Portrait 9:16": (768, 1344),
            "Portrait 2:3": (832, 1216),
            "Wide 21:9": (1536, 640),
            "SD 512": (512, 512),
            "SD 512×768": (512, 768),
            "SD 768×512": (768, 512),
            "SD 768": (768, 768),
        }
        self._img_aspect_var = ctk.StringVar(value="Square 1:1")
        ctk.CTkOptionMenu(
            size_row, variable=self._img_aspect_var,
            values=list(self._aspect_presets.keys()), width=160,
            dynamic_resizing=False,
            command=self._on_aspect_changed,
            **self._option_menu_style(),
        ).pack(side="left", padx=(0, 0))

        # Row 2: Sampler / Scheduler
        sampler_box = ctk.CTkFrame(settings, fg_color="transparent")
        sampler_box.grid(row=2, column=0, sticky="ew", padx=(14, 6), pady=(6, 6))
        _setting_label(sampler_box, "Sampler").pack(side="top", anchor="w")
        self._img_sampler_var = ctk.StringVar(value="euler")
        ctk.CTkOptionMenu(
            sampler_box, variable=self._img_sampler_var,
            values=["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde",
                    "dpmpp_sde", "dpmpp_3m_sde", "ddim", "uni_pc"],
            width=180,
            dynamic_resizing=False,
            **self._option_menu_style(),
        ).pack(side="top", anchor="w", pady=(2, 0))

        scheduler_box = ctk.CTkFrame(settings, fg_color="transparent")
        scheduler_box.grid(row=2, column=1, sticky="ew", padx=(6, 14), pady=(6, 6))
        _setting_label(scheduler_box, "Scheduler").pack(side="top", anchor="w")
        self._img_scheduler_var = ctk.StringVar(value="normal")
        ctk.CTkOptionMenu(
            scheduler_box, variable=self._img_scheduler_var,
            values=["normal", "karras", "exponential", "simple", "sgm_uniform", "beta"],
            width=160,
            dynamic_resizing=False,
            **self._option_menu_style(),
        ).pack(side="top", anchor="w", pady=(2, 0))

        # Row 3: Steps / CFG (+ CFG lock pill)
        steps_box = ctk.CTkFrame(settings, fg_color="transparent")
        steps_box.grid(row=3, column=0, sticky="ew", padx=(14, 6), pady=(2, 6))
        _setting_label(steps_box, "Steps").pack(side="top", anchor="w")
        self._img_steps_var = ctk.StringVar(value="20")
        ctk.CTkEntry(
            steps_box, textvariable=self._img_steps_var, width=86,
            fg_color=INPUT_SURFACE, border_color=self._IG_BORDER,
        ).pack(side="top", anchor="w", pady=(2, 0))

        cfg_box = ctk.CTkFrame(settings, fg_color="transparent")
        cfg_box.grid(row=3, column=1, sticky="ew", padx=(6, 14), pady=(2, 6))
        cfg_label_row = ctk.CTkFrame(cfg_box, fg_color="transparent")
        cfg_label_row.pack(side="top", fill="x")
        _setting_label(cfg_label_row, "CFG").pack(side="left")
        self._img_cfg_lock_lbl = ctk.CTkLabel(
            cfg_label_row, text="", font=ctk.CTkFont(size=10),
            text_color=WARN_TEXT,
        )
        self._img_cfg_lock_lbl.pack(side="left", padx=(8, 0))
        self._img_cfg_var = ctk.StringVar(value="7.0")
        self._img_cfg_entry = ctk.CTkEntry(
            cfg_box, textvariable=self._img_cfg_var, width=86,
            fg_color=INPUT_SURFACE, border_color=self._IG_BORDER,
        )
        self._img_cfg_entry.pack(side="top", anchor="w", pady=(2, 0))

        # Row 4: Seed
        seed_box = ctk.CTkFrame(settings, fg_color="transparent")
        seed_box.grid(row=4, column=0, columnspan=2, sticky="ew",
                      padx=14, pady=(2, 8))
        _setting_label(seed_box, "Seed   (-1 = random)").pack(side="top", anchor="w")
        seed_row = ctk.CTkFrame(seed_box, fg_color="transparent")
        seed_row.pack(side="top", fill="x", pady=(2, 0))
        self._img_seed_var = ctk.StringVar(value="-1")
        self._img_seed_entry = ctk.CTkEntry(
            seed_row, textvariable=self._img_seed_var, width=140,
            fg_color=INPUT_SURFACE, border_color=self._IG_BORDER,
        )
        self._img_seed_entry.pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            seed_row, text="🎲 Random", width=86, height=28,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=lambda: self._img_seed_var.set("-1"),
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            seed_row, text="📋 Copy", width=78, height=28,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._copy_seed_to_clipboard,
        ).pack(side="left")

        # Row 5: CPU mode
        _cpu_only_hw = self.gpu_info.gpu_type == "cpu"
        self._img_cpu_mode_var = ctk.BooleanVar(value=self._comfyui_force_cpu)
        _cpu_label = (
            "Force CPU mode  ·  no GPU detected — SD 1.5 recommended"
            if _cpu_only_hw
            else "Force CPU mode  ·  slower, but no VRAM pressure"
        )
        self._img_cpu_mode_cb = ctk.CTkCheckBox(
            settings, text=_cpu_label,
            variable=self._img_cpu_mode_var,
            font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG,
            border_color=self._IG_BORDER,
            fg_color=self._IG_ACCENT, hover_color=self._IG_ACCENT_HOVER,
            command=self._on_img_cpu_mode_changed,
        )
        if _cpu_only_hw:
            self._img_cpu_mode_cb.configure(state="disabled")
        self._img_cpu_mode_cb.grid(row=5, column=0, columnspan=2, sticky="w",
                                   padx=14, pady=(2, 14))

        # ── Reference image generation (img2img) ────────────────────────
        self._img_img2img_frame = ctk.CTkFrame(
            body, fg_color=self._IG_INNER_FG, corner_radius=10,
            border_width=1, border_color=self._IG_BORDER,
        )
        self._img_img2img_frame.grid(row=6, column=0, sticky="ew", pady=(0, 14))
        self._img_img2img_frame.grid_columnconfigure(1, weight=1)

        self._img_img2img_cb = ctk.CTkCheckBox(
            self._img_img2img_frame,
            text="Use reference image for generation",
            variable=self._img_img2img_var,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._IG_SUBHEAD_FG,
            border_color=self._IG_BORDER,
            fg_color=self._IG_ACCENT, hover_color=self._IG_ACCENT_HOVER,
            command=self._on_img_img2img_mode_changed,
        )
        self._img_img2img_cb.grid(row=0, column=0, columnspan=2, sticky="w",
                                  padx=14, pady=(12, 4))

        self._img_img2img_thumb = ctk.CTkLabel(
            self._img_img2img_frame, text="",
            width=58, height=58,
            fg_color=("gray85", "gray22"), corner_radius=8,
            text_color=self._IG_MUTED_FG,
            font=ctk.CTkFont(size=10),
        )
        self._img_img2img_thumb.grid(row=1, column=0, sticky="nw", padx=14, pady=(4, 12))

        ref_col = ctk.CTkFrame(self._img_img2img_frame, fg_color="transparent")
        ref_col.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=(4, 12))
        ref_col.grid_columnconfigure(0, weight=1)
        self._img_img2img_filename_lbl = ctk.CTkLabel(
            ref_col, text="No image selected",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_SECONDARY, anchor="w",
            wraplength=330,
        )
        self._img_img2img_filename_lbl.grid(row=0, column=0, sticky="ew")

        ref_btns = ctk.CTkFrame(ref_col, fg_color="transparent")
        ref_btns.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ctk.CTkButton(
            ref_btns, text="Browse image...", width=120, height=28,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._browse_img2img_reference_image,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            ref_btns, text="Clear", width=68, height=28,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._clear_img2img_reference_image,
        ).pack(side="left")

        denoise_row = ctk.CTkFrame(ref_col, fg_color="transparent")
        denoise_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        denoise_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            denoise_row, text="Strength",
            font=ctk.CTkFont(size=10),
            text_color=self._IG_MUTED_FG,
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._img_denoise_slider = ctk.CTkSlider(
            denoise_row, from_=0.1, to=1.0, number_of_steps=90,
            variable=self._img_denoise_var,
            command=lambda _value: self._update_img2img_denoise_label(),
        )
        self._img_denoise_slider.grid(row=0, column=1, sticky="ew")
        self._img_denoise_lbl = ctk.CTkLabel(
            denoise_row, text="0.75",
            width=42, font=ctk.CTkFont(size=10),
            text_color=self._IG_SUBHEAD_FG,
        )
        self._img_denoise_lbl.grid(row=0, column=2, sticky="e", padx=(8, 0))

        # v5.5.14: legend strip teaches the user what the dimensionless
        # strength slider actually means — most users guess wrong otherwise.
        self._img_denoise_legend = ctk.CTkLabel(
            ref_col,
            text=(
                "0.30 subtle restyle  ·  0.55 restyle (default)  ·  "
                "0.85 heavy restyle  ·  1.00 ignore reference"
            ),
            font=ctk.CTkFont(size=10),
            text_color=self._IG_MUTED_FG,
            anchor="w", justify="left",
            wraplength=360,
        )
        self._img_denoise_legend.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        # v5.5.14: aspect-match snaps W×H to the closest SDXL bucket of the
        # reference's aspect at queue time, fixing the "center-crop cuts off
        # the head" failure mode when reference is portrait but target is square.
        self._img_match_aspect_cb = ctk.CTkCheckBox(
            ref_col,
            text="Match reference aspect (recommended)",
            variable=self._img_match_aspect_var,
            font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG,
            border_color=self._IG_BORDER,
            fg_color=self._IG_ACCENT, hover_color=self._IG_ACCENT_HOVER,
            command=self._on_img2img_match_aspect_toggle,
        )
        self._img_match_aspect_cb.grid(row=4, column=0, sticky="w", pady=(6, 0))

        # ── Hero CTA: Generate (+ Stop) ─────────────────────────────────
        cta_row = ctk.CTkFrame(body, fg_color="transparent")
        cta_row.grid(row=1, column=0, sticky="ew", pady=(2, 12))
        cta_row.grid_columnconfigure(0, weight=1)

        self._img_generate_btn = ctk.CTkButton(
            cta_row, text="✨  Generate", height=44,
            state="normal",
            font=ctk.CTkFont(size=14, weight="bold"),
            **self._solid_button_style(self._IG_HERO, self._IG_HERO_HOVER),
            command=self._start_image_generation,
        )
        self._img_generate_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self._img_stop_btn = ctk.CTkButton(
            cta_row, text="■ Stop", width=88, height=44,
            state="disabled",
            font=ctk.CTkFont(size=12, weight="bold"),
            **self._solid_button_style(self._IG_DANGER, "#8a2424"),
            command=self._stop_image_generation,
        )
        self._img_stop_btn.grid(row=0, column=1, sticky="e")

        # Ready checklist (orange when blocked, green when good to go)
        self._img_ready_lbl = ctk.CTkLabel(
            body, text="", font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
            wraplength=470, justify="left",
        )
        self._img_ready_lbl.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        return card

    # ── ZONE B — Vision-assisted prompting ────────────────────────────────────
    def _build_image_zone_vision(self, parent) -> "ctk.CTkFrame":
        """Build the VISION card: reference image → model picker → Analyze."""
        card = ctk.CTkFrame(parent, corner_radius=self._IG_RADIUS, fg_color=self._IG_CARD_FG)
        card.grid_columnconfigure(0, weight=1)

        # Header
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 4))
        ctk.CTkLabel(
            head, text="VISION-ASSISTED PROMPTING",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self._IG_MUTED_FG,
        ).pack(side="top", anchor="w")
        ctk.CTkLabel(
            head, text="Describe an image to build a prompt",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="top", anchor="w", pady=(2, 0))
        ctk.CTkLabel(
            head,
            text="Upload a reference photo and let a vision (multimodal) model write a Stable Diffusion prompt for you. Optional — skip this and write your own prompt above.",
            font=ctk.CTkFont(size=12),
            text_color=self._IG_SUBHEAD_FG, justify="left", wraplength=360,
        ).pack(side="top", anchor="w", pady=(2, 0))

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 18))
        body.grid_columnconfigure(0, weight=1)

        # ── Reference image row ─────────────────────────────────────────
        ref_label = ctk.CTkLabel(
            body, text="Reference image",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
        )
        ref_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

        ref_panel = ctk.CTkFrame(
            body, fg_color=self._IG_INNER_FG, corner_radius=10,
            border_width=1, border_color=self._IG_BORDER,
        )
        ref_panel.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        ref_panel.grid_columnconfigure(1, weight=1)

        self._img_ref_thumb = ctk.CTkLabel(
            ref_panel, text="",
            width=72, height=72,
            fg_color=("gray85", "gray22"), corner_radius=8,
            text_color=self._IG_MUTED_FG,
            font=ctk.CTkFont(size=11),
        )
        self._img_ref_thumb.grid(row=0, column=0, sticky="nw", padx=12, pady=12)

        ref_text_col = ctk.CTkFrame(ref_panel, fg_color="transparent")
        ref_text_col.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=12)
        ref_text_col.grid_columnconfigure(0, weight=1)
        self._img_ref_filename_lbl = ctk.CTkLabel(
            ref_text_col, text="No image selected",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_SECONDARY, anchor="w",
            wraplength=300,
        )
        self._img_ref_filename_lbl.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            ref_text_col, text="PNG or JPEG - auto-resized to 800 px max before analysis",
            font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
            wraplength=300,
        ).grid(row=1, column=0, sticky="ew", pady=(2, 0))

        ref_btn_row = ctk.CTkFrame(ref_text_col, fg_color="transparent")
        ref_btn_row.grid(row=2, column=0, sticky="w", pady=(8, 0))
        browse_btn = ctk.CTkButton(
            ref_btn_row, text="Browse image...", width=128, height=30,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._browse_reference_image,
        )
        browse_btn.pack(side="left", padx=(0, 6))
        clear_btn = ctk.CTkButton(
            ref_btn_row, text="Clear", width=72, height=30,
            font=ctk.CTkFont(size=11),
            **self._outline_button_style(),
            command=self._clear_reference_image,
        )
        clear_btn.pack(side="left")

        for widget in (ref_panel, self._img_ref_thumb, ref_text_col, self._img_ref_filename_lbl):
            widget.bind("<Button-1>", lambda _e: self._browse_reference_image())
            try:
                widget.configure(cursor="hand2")
            except Exception:
                pass

        # ── Vision model picker ─────────────────────────────────────────
        picker_label = ctk.CTkLabel(
            body, text="Vision model",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
        )
        picker_label.grid(row=2, column=0, sticky="w", pady=(0, 4))

        picker = ctk.CTkFrame(body, fg_color="transparent")
        picker.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        picker.grid_columnconfigure(0, weight=1)

        self._build_vision_picker(picker)

        # Help text under the picker — updated dynamically in _refresh_vision_picker_ui
        self._img_vision_help_lbl = ctk.CTkLabel(
            body, text="",
            font=ctk.CTkFont(size=11),
            text_color=self._IG_SUBHEAD_FG, anchor="w",
            wraplength=360, justify="left",
        )
        self._img_vision_help_lbl.grid(row=4, column=0, sticky="w", pady=(0, 12))

        # ── Analyze CTA ─────────────────────────────────────────────────
        # v2026.06.01.8 state-machine layout (Ron, 2026-06-01):
        # Exactly ONE primary action is visible at a time, chosen by
        # ``_refresh_vision_model_ui`` based on the selected vision
        # model's install state and any in-flight operation.
        # Visible button → state:
        #   Download                → selected model is not installed
        #   Stop ("Cancel download") → selected model is currently being pulled
        #   Analyze → Prompt          → selected model is installed and idle
        #   Stop                      → an analyze is running
        # The legacy three-button row caused the "Analyze → Promp" /
        # "nalyze → Prompt" clipping in the screenshot Ron flagged
        # because the row was sized for one button at a time but always
        # rendered all three. Now that only one is visible at a time
        # the surviving button gets full width via column 0 weight and
        # the clip is gone by construction. The single column also lets
        # us drop the per-button left/right padding tricks.
        analyze_row = ctk.CTkFrame(body, fg_color="transparent")
        analyze_row.grid(row=5, column=0, sticky="ew", pady=(2, 6))
        analyze_row.grid_columnconfigure(0, weight=1)

        self._img_analyze_btn = ctk.CTkButton(
            analyze_row, text="Analyze → Prompt", height=40,
            state="normal",
            font=ctk.CTkFont(size=13, weight="bold"),
            **self._solid_button_style(self._IG_ACCENT, self._IG_ACCENT_HOVER, self._IG_ACCENT_TEXT),
            command=self._analyze_reference_image,
        )
        self._img_analyze_btn.grid(row=0, column=0, sticky="ew")

        self._img_analyze_get_model_btn = ctk.CTkButton(
            analyze_row, text="⬇ Download", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            **self._solid_button_style(self._IG_ACCENT, self._IG_ACCENT_HOVER, self._IG_ACCENT_TEXT),
            command=self._download_vision_model,
        )
        # Hidden by default; ``_refresh_vision_model_ui`` shows it when
        # the selected vision model is not installed locally.
        self._img_analyze_get_model_btn.grid(row=0, column=0, sticky="ew")
        self._img_analyze_get_model_btn.grid_remove()

        # The Stop button does double duty: cancels an in-flight
        # vision-model pull when one is running, otherwise cancels an
        # in-flight Analyze → Prompt. ``_stop_vision_action`` dispatches
        # so the button only needs one command binding.
        self._img_analyze_stop_btn = ctk.CTkButton(
            analyze_row, text="■ Stop", height=40,
            state="disabled",
            font=ctk.CTkFont(size=13, weight="bold"),
            **self._solid_button_style(self._IG_DANGER, "#8a2424", self._IG_DANGER_TEXT),
            command=self._stop_vision_action,
        )
        self._img_analyze_stop_btn.grid(row=0, column=0, sticky="ew")
        self._img_analyze_stop_btn.grid_remove()

        return card

    # ── Vision picker widgets ─────────────────────────────────────────────────
    def _build_vision_picker(self, parent) -> None:
        """Create the vision-model selector. Each row is clickable; the selected
        one gets an accent border + dot. Rows live in
        ``self._img_vision_cards[catalog_id]``.
        """
        entries = self._vision_picker_entries()
        # Defensive: if the catalog is missing the picker entries, just show a
        # one-line note so users still see *something*.
        if not entries:
            ctk.CTkLabel(
                parent,
                text="Vision models are not configured in this catalog.",
                font=ctk.CTkFont(size=11), text_color=self._IG_SUBHEAD_FG,
            ).grid(row=0, column=0, sticky="w")
            return

        self._img_vision_cards = {}
        for row, entry in enumerate(entries):
            cat_id = entry.get("id", "")
            badge_text = entry.get("recommendation_badge") or ""
            card = ctk.CTkFrame(
                parent, corner_radius=12,
                fg_color=self._IG_INNER_FG,
                border_width=1, border_color=self._IG_BORDER,
            )
            card.grid(row=row, column=0, sticky="ew", padx=0, pady=(0, 6))
            card.grid_columnconfigure(1, weight=1)

            dot = ctk.CTkLabel(
                card, text="○", width=18,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=self._IG_MUTED_FG,
            )
            dot.grid(row=0, column=0, rowspan=3, sticky="n", padx=(10, 6), pady=(11, 0))

            header = ctk.CTkFrame(card, fg_color="transparent")
            header.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(8, 0))
            header.grid_columnconfigure(0, weight=1)

            # Model name
            name_lbl = ctk.CTkLabel(
                header, text=entry.get("name", cat_id),
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w", justify="left",
            )
            name_lbl.grid(row=0, column=0, sticky="w")

            badge = None
            if badge_text:
                badge_fg = (
                    self._IG_BADGE_BG_REC if "Recommend" in badge_text else
                    self._IG_BADGE_BG_ACC if "Accuracy" in badge_text else
                    self._IG_BADGE_BG_EFF
                )
                badge = ctk.CTkLabel(
                    header, text=badge_text,
                    font=ctk.CTkFont(size=9, weight="bold"),
                    text_color=TEXT_PRIMARY,
                    fg_color=badge_fg, corner_radius=8,
                    padx=8, pady=2,
                )
                badge.grid(row=0, column=1, sticky="e", padx=(8, 0))

            # Tradeoff line
            note = entry.get("tradeoff_note") or ""
            note_lbl = ctk.CTkLabel(
                card, text=note,
                font=ctk.CTkFont(size=10),
                text_color=self._IG_SUBHEAD_FG,
                anchor="w", justify="left", wraplength=330,
            )
            note_lbl.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(2, 2))

            # Installed/Not-installed pill at the bottom
            status_lbl = ctk.CTkLabel(
                card, text=" · ",
                font=ctk.CTkFont(size=10), text_color=self._IG_MUTED_FG,
                anchor="w",
            )
            status_lbl.grid(row=2, column=1, sticky="w", padx=(0, 10), pady=(0, 8))

            # Bind click to whole card and labels
            def _bind_click(widget, cid=cat_id):
                widget.bind("<Button-1>", lambda _e, c=cid: self._set_selected_vision_model(c))
                # Hand cursor on hover
                try:
                    widget.configure(cursor="hand2")
                except Exception:
                    pass

            widgets = [card, header, dot, name_lbl, note_lbl, status_lbl]
            if badge is not None:
                widgets.append(badge)
            for w in widgets:
                _bind_click(w)

            self._img_vision_cards[cat_id] = {
                "frame": card,
                "dot": dot,
                "name": name_lbl,
                "status": status_lbl,
                "note": note_lbl,
                "entry": entry,
            }

    def _refresh_vision_picker_ui(self) -> None:
        """Repaint vision picker cards: selected one gets accent border + dot;
        each card shows Installed / Not installed pill at the bottom."""
        if not getattr(self, "_img_vision_cards", None):
            return
        selected_tag = self._get_selected_vision_tag()
        local_names = self._get_cached_local_names()

        selected_entry = None
        for cat_id, parts in self._img_vision_cards.items():
            entry = parts["entry"]
            tag = entry.get("ollama_tag", "")
            is_selected = (tag == selected_tag)
            if is_selected:
                selected_entry = entry
            installed = _ollama_tag_is_local(tag, local_names)

            # Border + dot reflect selection
            try:
                parts["frame"].configure(
                    border_color=self._IG_ACCENT if is_selected else self._IG_BORDER,
                    border_width=2 if is_selected else 1,
                )
            except Exception:
                pass
            try:
                parts["dot"].configure(
                    text="●" if is_selected else "○",
                    text_color=self._IG_ACCENT if is_selected else self._IG_MUTED_FG,
                )
            except Exception:
                pass
            # Installed status pill at card bottom
            try:
                if installed:
                    parts["status"].configure(
                        text="● Installed",
                        text_color=("#1f6e2e", "#7fd28b"),
                    )
                else:
                    size_gb = entry.get("size_gb")
                    size_txt = f"  ·  ~{size_gb:.1f} GB" if size_gb else ""
                    parts["status"].configure(
                        text=f"○ Not installed{size_txt}",
                        text_color=self._IG_MUTED_FG,
                    )
            except Exception:
                pass

        # Update the long help line under the picker (what's selected, what it's good for).
        if hasattr(self, "_img_vision_help_lbl") and self._img_vision_help_lbl is not None:
            if selected_entry:
                badge = selected_entry.get("recommendation_badge") or ""
                note = selected_entry.get("tradeoff_note") or ""
                tag = selected_entry.get("ollama_tag", "")
                installed_now = _ollama_tag_is_local(tag, local_names)
                badge_text = f"{badge} — " if badge else ""
                tail = "" if installed_now else "  ·  Click Download to install."
                self._img_vision_help_lbl.configure(
                    text=f"{badge_text}{note}{tail}",
                )
            else:
                self._img_vision_help_lbl.configure(text="")

    def _clear_reference_image(self) -> None:
        """Drop the currently-loaded reference image and reset the thumbnail row."""
        self._img_ref_path = None
        self._img_ref_photo = None
        try:
            if hasattr(self, "_img_ref_thumb") and self._img_ref_thumb is not None:
                self._img_ref_thumb.configure(image=None, text="")
            if hasattr(self, "_img_ref_filename_lbl") and self._img_ref_filename_lbl is not None:
                self._img_ref_filename_lbl.configure(text="No image selected")
        except Exception:
            pass
        # Re-evaluate analyze button (will disable if no ref)
        self._refresh_vision_model_ui()

    def _selected_img2img_entry(self) -> Optional[dict]:
        display_name = self._img_model_var.get() if hasattr(self, "_img_model_var") else ""
        model_name = self._img_friendly_to_filename.get(display_name, display_name)
        if not model_name:
            return None
        return self._find_catalog_entry_for_model(model_name)

    def _selected_model_supports_img2img(self) -> bool:
        entry = self._selected_img2img_entry()
        return bool(entry and entry.get("supports_img2img"))

    def _update_img2img_denoise_label(self) -> None:
        try:
            value = float(self._img_denoise_var.get())
        except Exception:
            value = 0.55
        if hasattr(self, "_img_denoise_lbl") and self._img_denoise_lbl is not None:
            self._img_denoise_lbl.configure(text=f"{value:.2f}")
        # Persist the user's choice so it survives restarts. Cheap — the
        # config writer debounces, and the slider only fires on release.
        try:
            self.cfg["img2img_denoise"] = round(max(0.05, min(1.0, value)), 2)
            config.save(self.cfg)
        except Exception as exc:
            logger.debug(f"img2img_denoise persist failed: {exc}")

    def _on_img2img_match_aspect_toggle(self) -> None:
        val = bool(self._img_match_aspect_var.get())
        self.cfg["img2img_match_aspect"] = val
        try:
            if not config.save(self.cfg):
                logger.warning("img2img_match_aspect: failed to persist")
        except Exception as exc:
            logger.debug(f"img2img_match_aspect persist failed: {exc}")

    def _refresh_img2img_controls(self) -> None:
        if not hasattr(self, "_img_img2img_cb") or self._img_img2img_cb is None:
            return
        entry = self._selected_img2img_entry()
        supports = bool(entry and entry.get("supports_img2img"))
        workflows = entry.get("img2img_workflows", {}) if entry else {}
        try:
            if isinstance(workflows, dict) and "denoise_default" in workflows:
                self._img_denoise_var.set(float(workflows.get("denoise_default", 0.55)))
                self._update_img2img_denoise_label()
            self._img_img2img_cb.configure(
                state="normal" if supports else "disabled",
                text=(
                    "Use reference image for generation"
                    if supports else
                    "Reference image generation not supported by this model"
                ),
            )
            if not supports and self._img_img2img_var.get():
                self._img_img2img_var.set(False)
            control_state = "normal" if supports and self._img_img2img_var.get() else "disabled"
            for widget_name in (
                "_img_denoise_slider",
                "_img_img2img_thumb",
                "_img_img2img_filename_lbl",
                "_img_match_aspect_cb",
            ):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.configure(state=control_state)
        except Exception as exc:
            logger.debug(f"Img2img control refresh failed: {exc}")
        self._refresh_image_readiness()

    def _on_img_img2img_mode_changed(self) -> None:
        if self._img_img2img_var.get() and not self._selected_model_supports_img2img():
            self._img_img2img_var.set(False)
            self._img_set_status(
                "Reference image generation is available for SD/SDXL checkpoint models only.",
                color=WARN_TEXT,
            )
            self._refresh_img2img_controls()
            return
        self._refresh_img2img_controls()

        # v5.5.14 (Ron, 2026-05-29): swap the positive + negative prompt
        # together with the mode. Ticking the box loads a documented sample
        # so the user always has a known-good ref-image prompt; unticking
        # restores the selected model's normal demo prompt + negative.
        # Users who want a different man/woman/age/light/clothes combo are
        # directed to the prompt builder in image-gen-guide.html §5.3.
        if self._img_img2img_var.get():
            self._apply_img2img_default_prompts()
            self._img_set_status(
                "Loaded the default reference-image sample prompt. "
                "See the help doc's prompt builder for other man/woman, "
                "age, lighting, and clothing combos.",
            )
        else:
            model = self._selected_image_model_catalog_entry()
            if model:
                self._apply_image_demo_prompt(model)
            self._apply_selected_model_negative_prompt()

    def _browse_img2img_reference_image(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Generation Reference Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self._img_img2img_path = path
        if hasattr(self, "_img_img2img_filename_lbl") and self._img_img2img_filename_lbl is not None:
            self._img_img2img_filename_lbl.configure(text=Path(path).name)
        if PIL_AVAILABLE:
            try:
                img = _PIL_Image.open(path)
                img.thumbnail((58, 58), _PIL_Image.LANCZOS)
                self._img_img2img_photo = _PIL_ImageTk.PhotoImage(img)
                self._img_img2img_thumb.configure(image=self._img_img2img_photo, text="")
            except Exception:
                self._img_img2img_thumb.configure(image=None, text="?")
        else:
            self._img_img2img_thumb.configure(text=Path(path).name[:10])
        self._refresh_image_readiness()

    def _clear_img2img_reference_image(self) -> None:
        self._img_img2img_path = None
        self._img_img2img_photo = None
        try:
            self._img_img2img_thumb.configure(image=None, text="")
            self._img_img2img_filename_lbl.configure(text="No image selected")
        except Exception:
            pass
        self._refresh_image_readiness()

    # ── Analyze → Prompt (img2prompt) ─────────────────────────────────────────

    # v5.1: legacy class attribute kept for any downstream code or saved configs
    # that may still reference it. The active vision tag is now read from
    # cfg["default_vision_model_id"] via _get_selected_vision_tag(), which falls
    # back to "gemma3:4b" when no preference is set.
    VISION_MODEL         = "gemma3:4b"
    LEGACY_VISION_MODEL  = "llama3.2-vision:11b"
    DEFAULT_VISION_TAG   = "gemma3:4b"
    VISION_MIN_VRAM_GB   = 8   # 11B Q4 needs ~7 GB; SKUs below this get num_gpu=0

    # v5.1: the three selectable vision options exposed in the Image Gen page
    # picker. Order matters — first one is the recommended default. Each entry
    # is the catalog id (NOT the ollama tag — see _get_vision_catalog_entry()).
    VISION_PICKER_CATALOG_IDS = (
        "gemma3:4b-vision",       # Recommended
        "gemma3:12b-vision",      # Highest Accuracy
        "minicpm-v-vision",       # Efficient Multimodal
    )

    def _get_vision_catalog_entry(self, catalog_id: str) -> Optional[dict]:
        """Look up a vision-model catalog entry by its catalog id."""
        for m in self._catalog_models:
            if m.get("id") == catalog_id:
                return m
        return None

    def _vision_picker_entries(self) -> list[dict]:
        """Return the catalog entries displayed in the Image Gen vision picker,
        in the order they should be shown. Missing entries are skipped silently."""
        entries = []
        for cid in self.VISION_PICKER_CATALOG_IDS:
            entry = self._get_vision_catalog_entry(cid)
            if entry is not None:
                entries.append(entry)
        return sorted(entries, key=lambda m: (float(m.get("min_vram_gb") or 0), str(m.get("name") or "")))

    def _get_selected_vision_tag(self) -> str:
        """Return the ollama tag of the currently selected vision model.

        Resolution order:
          1) cfg["default_vision_model_id"] → catalog entry → ollama_tag
          2) DEFAULT_VISION_TAG ("gemma3:4b") if the saved id is missing or invalid
        """
        try:
            wanted_id = (self.cfg.get("default_vision_model_id") or "").strip()
        except Exception:
            wanted_id = ""
        if wanted_id:
            entry = self._get_vision_catalog_entry(wanted_id)
            if entry:
                tag = (entry.get("ollama_tag") or "").strip()
                if tag:
                    return tag
            # Some legacy installs may have persisted the bare ollama tag rather
            # than the catalog id. Honor that so settings don't silently flip.
            for m in self._catalog_models:
                if m.get("ollama_tag") == wanted_id and "vision" in (m.get("tags") or []):
                    return wanted_id
        return self.DEFAULT_VISION_TAG

    def _get_selected_vision_entry(self) -> Optional[dict]:
        """Return the catalog entry matching the selected vision tag, or None."""
        tag = self._get_selected_vision_tag()
        for m in self._catalog_models:
            if m.get("ollama_tag") == tag and "vision" in (m.get("tags") or []):
                return m
        return None

    def _set_selected_vision_model(self, catalog_id: str) -> None:
        """Persist the user's vision model selection and refresh the picker UI."""
        entry = self._get_vision_catalog_entry(catalog_id)
        if entry is None:
            logger.warning(f"Vision picker: unknown catalog id '{catalog_id}'")
            return
        self.cfg["default_vision_model_id"] = catalog_id
        if not config.save(self.cfg):
            logger.error("Vision picker: failed to persist default_vision_model_id")
        logger.info(f"Vision picker: selected {catalog_id} ({entry.get('ollama_tag')})")
        # Repaint the picker cards + analyze/download CTAs.
        try:
            self._refresh_vision_picker_ui()
        except Exception as exc:
            logger.debug(f"Vision picker UI refresh deferred: {exc}")
        self._refresh_vision_model_ui()

    _ANALYZE_SYSTEM_PROMPT = (
        "You are a Stable Diffusion prompt engineer. "
        "Describe ONLY what is visually present in the image using 15-25 short tags. "
        "Include: subject (shape, material, texture, color), background elements, "
        "lighting (direction, quality, color temperature), "
        "depth of field / bokeh, camera angle, atmosphere and mood. "
        "Do NOT include abstract concepts, categories, brands, history, culture, "
        "industry terms, or anything that is not directly visible. "
        "Bad example: 'coffee culture, coffee marketing, coffee industry'. "
        "Good example: 'ceramic coffee cup, latte art heart, dark glossy saucer, "
        "dramatic side lighting, warm amber highlights, shallow depth of field, "
        "soft bokeh, moody atmosphere, cinematic, photorealistic'. "
        "Output ONLY a comma-separated list of tags — no sentences, no explanation."
    )

    def _is_vision_model_downloading(self, vision_tag: str) -> bool:
        """Return True when the currently-selected vision model is the
        target of an in-flight Ollama pull. Used by
        ``_refresh_vision_model_ui`` to switch the analyze-row state
        machine to the "downloading" state.
        """
        thread = getattr(self, "_download_thread", None)
        if thread is None or not thread.is_alive():
            return False
        active = getattr(self, "_active_download_tag", "") or ""
        return bool(vision_tag) and active == vision_tag

    def _is_vision_analyze_running(self) -> bool:
        """Return True when an Analyze → Prompt worker is in flight and
        has not been signaled to stop. Used by the vision-row state
        machine to choose between Analyze and Stop.
        """
        thread = getattr(self, "_analyze_thread", None)
        if thread is None or not thread.is_alive():
            return False
        stop_event = getattr(self, "_analyze_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            return False
        return True

    def _stop_vision_action(self):
        """Single dispatcher behind the analyze-row Stop button. Cancels
        whichever vision-side operation is currently in flight for the
        selected model:
          * Ollama pull → set the shared download stop_event so
            ``OllamaClient.pull_model`` aborts cleanly.
          * Analyze → Prompt → delegate to ``_stop_analyze`` (which sets
            the analyze stop_event and reflects the disabled state on
            the button).
        Falling through with neither operation active is a no-op — the
        button is hidden in that state, so reaching here would only
        happen via a keyboard shortcut on a stale handle.
        """
        vision_tag = self._get_selected_vision_tag()
        if self._is_vision_model_downloading(vision_tag):
            try:
                self._stop_event.set()
            except Exception as exc:
                logger.debug(f"Vision Stop: failed to signal download stop event: {exc}")
            self._img_analyze_stop_btn.configure(
                state="disabled",
                fg_color=self._IG_DISABLED_FG,
                text_color_disabled=self._IG_DISABLED_TEXT,
            )
            try:
                self.set_status(f"Cancelling download of {vision_tag} …")
            except Exception:
                pass
            logger.info(
                f"Vision picker: cancel download requested for {vision_tag}"
            )
            return
        if self._is_vision_analyze_running():
            self._stop_analyze()
            return
        logger.debug("Vision Stop pressed with no active operation; ignored.")

    def _refresh_vision_model_ui(self):
        """Update the analyze-row state machine for the selected vision
        model. Exactly ONE of the three buttons is visible at a time:

          * Analyze running → [Stop] only
          * Download running for the selected tag → [Stop] only
          * Selected model not installed → [Download] only
          * Selected model installed (idle) → [Analyze → Prompt] only

        v2026.06.01.8 (Ron, 2026-06-01): replaces the legacy
        always-three-buttons layout. The legacy layout caused the
        "Analyze → Promp" clipping seen on narrower vision panels
        because all three buttons fought for the same row width, and
        showed an enabled Analyze button even when the selected model
        was not installed.
        """
        # Repaint vision picker cards first so the installed pills
        # match whatever just changed (download finished, ollama just
        # came up, selection changed, etc.).
        try:
            self._refresh_vision_picker_ui()
        except Exception:
            pass
        if not hasattr(self, "_img_analyze_btn"):
            return

        vision_tag = self._get_selected_vision_tag()
        try:
            has_model = self.ollama_ok and self.ollama.is_model_local(vision_tag)
        except Exception:
            has_model = False
        downloading = self._is_vision_model_downloading(vision_tag)
        analyzing = self._is_vision_analyze_running()

        def _show_only(widget):
            """Hide the other two analyze-row buttons and show this one."""
            for btn in (
                self._img_analyze_btn,
                self._img_analyze_get_model_btn,
                self._img_analyze_stop_btn,
            ):
                if btn is widget:
                    btn.grid()
                else:
                    btn.grid_remove()

        if analyzing:
            _show_only(self._img_analyze_stop_btn)
            self._img_analyze_stop_btn.configure(
                text="■ Stop",
                state="normal",
                fg_color=self._IG_DANGER,
                hover_color="#8a2424",
                text_color=self._IG_DANGER_TEXT,
            )
            return
        if downloading:
            _show_only(self._img_analyze_stop_btn)
            self._img_analyze_stop_btn.configure(
                text="■ Cancel download",
                state="normal",
                fg_color=self._IG_DANGER,
                hover_color="#8a2424",
                text_color=self._IG_DANGER_TEXT,
            )
            if hasattr(self, "_img_analyze_status"):
                entry = self._get_selected_vision_entry()
                display_name = (entry or {}).get("name") or vision_tag
                self._img_set_canvas_status(
                    f"Downloading {display_name} …", TEXT_MUTED
                )
            return
        if has_model:
            _show_only(self._img_analyze_btn)
            self._img_analyze_btn.configure(
                state="normal",
                fg_color=self._IG_ACCENT,
                hover_color=self._IG_ACCENT_HOVER,
                text_color=self._IG_ACCENT_TEXT,
            )
            if hasattr(self, "_img_analyze_status"):
                self._img_set_canvas_status("")
            return
        # Selected vision model is not installed locally.
        _show_only(self._img_analyze_get_model_btn)
        if hasattr(self, "_img_analyze_status"):
            entry = self._get_selected_vision_entry()
            display_name = (entry or {}).get("name") or vision_tag
            self._img_set_canvas_status(
                f"{display_name} not installed — click Download to get it",
                WARN_TEXT,
            )

    def _browse_reference_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        self._img_ref_path = path

        # Update the filename label so users can see exactly which file is loaded
        if hasattr(self, "_img_ref_filename_lbl") and self._img_ref_filename_lbl is not None:
            try:
                self._img_ref_filename_lbl.configure(text=Path(path).name)
            except Exception:
                pass

        # Show thumbnail
        if PIL_AVAILABLE:
            try:
                img = _PIL_Image.open(path)
                img.thumbnail((64, 64), _PIL_Image.LANCZOS)
                self._img_ref_photo = _PIL_ImageTk.PhotoImage(img)
                self._img_ref_thumb.configure(image=self._img_ref_photo, text="")
            except Exception:
                self._img_ref_thumb.configure(image=None, text="?")
        else:
            self._img_ref_thumb.configure(text=Path(path).name[:10])

        # Enable analyze button only if vision model is available
        self._refresh_vision_model_ui()

    def _download_vision_model(self):
        vision_tag = self._get_selected_vision_tag()
        model_dict = next(
            (m for m in self._catalog_models if m.get("ollama_tag") == vision_tag),
            None,
        )
        if model_dict is None:
            logger.error(f"_download_vision_model: {vision_tag} not found in catalog")
            self._img_set_canvas_status(
                f"{vision_tag} not found in catalog — check models_catalog.json",
                ERROR_TEXT,
            )
            return
        logger.info(f"_download_vision_model: queuing download for {vision_tag}")
        started = self.start_download(model_dict)
        # v2026.06.01.8: flip the analyze-row from Download → Cancel
        # download immediately so the user sees the new state without
        # waiting for the next paint tick. If start_download bailed
        # early (e.g. user declined a resource warning), the refresh
        # sees no live download and leaves the Download button up.
        if started:
            try:
                self._refresh_vision_model_ui()
            except Exception as exc:
                logger.debug(f"Vision picker refresh after start_download failed: {exc}")

    def _analyze_reference_image(self):
        if not self._img_ref_path:
            self._img_set_canvas_status("Choose a reference image first.", WARN_TEXT)
            return

        # v5: One-time-per-session warning that ComfyUI will be paused. Vision
        # analysis takes 30-60 s and triggers a ComfyUI restart afterwards to
        # avoid a model-corruption bug. Users with the box ticked never see
        # this dialog again until the app restarts.
        if not getattr(self, '_vision_warning_acknowledged', False):
            proceed = messagebox.askyesno(
                "Pause ComfyUI for vision analysis?",
                "Analyze → Prompt loads a vision model that needs all of your "
                "GPU's memory.\n\n"
                "What happens next:\n"
                "  • ComfyUI is stopped and unloaded (5-10 s)\n"
                "  • The vision model analyses the reference image (30-60 s)\n"
                "  • ComfyUI stays paused so you can analyze more images\n"
                "  • Generate restarts ComfyUI when you are ready to render\n\n"
                "Image generation will be unavailable during analysis. "
                "Continue?",
                parent=self,
            )
            if not proceed:
                return
            self._vision_warning_acknowledged = True

        # Re-verify vision model is still available before starting
        vision_tag = self._get_selected_vision_tag()
        try:
            if not self.ollama.is_model_local(vision_tag):
                self._img_set_canvas_status(
                    f"{vision_tag} not installed — click Download to get it",
                    WARN_TEXT,
                )
                return
        except Exception as probe_exc:
            self._img_set_canvas_status("Ollama not responding — is it running?", ERROR_TEXT)
            logger.error(f"Analyze→Prompt: Ollama probe failed: {probe_exc}")
            return

        # Cancel any previous run
        if self._analyze_thread and self._analyze_thread.is_alive():
            self._analyze_stop_event.set()
            self._analyze_thread.join(timeout=2)

        self._analyze_stop_event.clear()
        self._analyze_gen_id += 1
        analyze_gid = self._analyze_gen_id

        # UI: running state
        self._img_analyze_btn.configure(
            state="disabled",
            fg_color=self._IG_DISABLED_FG,
            text_color_disabled=self._IG_DISABLED_TEXT,
        )
        self._img_analyze_stop_btn.configure(
            state="normal",
            fg_color=self._IG_DANGER,
            hover_color="#8a2424",
            text_color=self._IG_DANGER_TEXT,
        )
        self._img_set_canvas_status("Analyzing …", TEXT_MUTED)
        self._img_progress_row.grid()
        self._img_analyze_progress.configure(mode="indeterminate", progress_color=INFO_TEXT)
        self._img_analyze_progress.grid()
        self._img_analyze_elapsed_lbl.grid()
        self._img_analyze_progress.start()
        self._analyze_elapsed_start = time.monotonic()
        self._analyze_tick_elapsed(analyze_gid)

        def _worker():
            # Track whether we shut ComfyUI down so we know to restart it
            _comfyui_was_ok = False
            try:
                if analyze_gid != self._analyze_gen_id:
                    return
                logger.info(f"Analyze→Prompt: worker started — image={self._img_ref_path}")

                # Phase 1: read + resize + encode
                logger.debug("Analyze→Prompt: reading image from disk")
                with open(self._img_ref_path, "rb") as f:
                    raw = f.read()
                size_kb = len(raw) // 1024
                logger.info(f"Analyze→Prompt: image read — {size_kb} KB")

                # Resize to max 800px on longest side — reduces payload and speeds up Ollama
                pil_img = _PIL_Image.open(io.BytesIO(raw))
                orig_w, orig_h = pil_img.size
                _MAX_DIM = 800
                if max(orig_w, orig_h) > _MAX_DIM:
                    pil_img.thumbnail((_MAX_DIM, _MAX_DIM), _PIL_Image.LANCZOS)
                    if pil_img.mode not in ("RGB", "L"):
                        pil_img = pil_img.convert("RGB")
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=85)
                    raw = buf.getvalue()
                    logger.info(f"Analyze→Prompt: resized {orig_w}x{orig_h} → {pil_img.width}x{pil_img.height} — {len(raw)//1024} KB")
                else:
                    logger.info(f"Analyze→Prompt: image {orig_w}x{orig_h} — already ≤{_MAX_DIM}px, no resize")

                img_b64 = base64.b64encode(raw).decode("utf-8")
                b64_kb = len(img_b64) // 1024
                logger.info(f"Analyze→Prompt: base64 encoded — {b64_kb} KB payload")

                messages = [
                    {"role": "system", "content": self._ANALYZE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Analyze this image and generate the prompt.",
                        "images": [img_b64],
                    },
                ]

                # Phase 2: terminate ComfyUI entirely to free all VRAM for the vision model.
                # free_vram() via the /free API only returns memory that PyTorch's allocator
                # has previously committed — before any generation it has nothing to return,
                # so Ollama sees insufficient VRAM and silently falls back to CPU (0 tokens).
                # Killing the process guarantees the OS reclaims every byte ComfyUI held.
                _comfyui_was_ok = self.comfyui_ok or bool(
                    self.comfyui_process and self.comfyui_process.poll() is None
                ) or self.comfyui.is_running()
                if _comfyui_was_ok:
                    logger.info("Analyze→Prompt: stopping ComfyUI to release all VRAM for vision model")
                    self.after(0, lambda gid=analyze_gid: gid == self._analyze_gen_id and self._img_set_canvas_status(
                        "Stopping ComfyUI to free VRAM for vision model…", TEXT_MUTED
                    ))
                    self.after(0, lambda gid=analyze_gid: gid == self._analyze_gen_id and self._set_comfyui_status(
                        "stopped (vision model)", WARN_TEXT
                    ))
                    if self.comfyui_process and self.comfyui_process.poll() is None:
                        try:
                            self.comfyui_process.terminate()
                            self.comfyui_process.wait(timeout=8)
                        except Exception:
                            try:
                                self.comfyui_process.kill()
                            except Exception:
                                pass
                    self._kill_orphan_comfyui_processes()
                    self.comfyui_process = None
                    self._close_comfyui_log_handle()
                    self.comfyui_ok = False
                    self.after(0, lambda gid=analyze_gid: gid == self._analyze_gen_id and self._set_image_gen_enabled(False))
                    logger.info("Analyze→Prompt: ComfyUI stopped — waiting for OS to reclaim VRAM")
                    time.sleep(2)

                # Phase 3: send to Ollama and wait for first token
                # Force CPU on SKUs with insufficient VRAM for the vision model.
                # Without this, Ollama sees the CUDA GPU and thrashes between CPU/GPU
                # resulting in an effective hang (observed: >3000s on A10-4Q, 4 GB VRAM).
                _sku_vram = (self._optional_sku or {}).get("vram_gb", 0)
                if not _sku_vram:
                    try:
                        gpus = system_info.get_gpu_info()
                        _sku_vram = max((g.get("vram_total_mb", 0) for g in gpus), default=0) / 1024
                    except Exception:
                        _sku_vram = 0
                _vision_entry = self._get_selected_vision_entry()
                # Use the entry's own min_vram_gb when available — smaller Gemma 3 4B
                # is fine at 4 GB while the legacy llama vision needed 8+.
                _vision_min_vram = (
                    (_vision_entry or {}).get("min_vram_gb")
                    or self.VISION_MIN_VRAM_GB
                )
                _vision_num_gpu = 0 if _sku_vram < _vision_min_vram else -1
                if _vision_num_gpu == 0:
                    logger.info(
                        f"Analyze→Prompt: VRAM={_sku_vram} GB < {_vision_min_vram} GB"
                        f" — forcing CPU inference for {vision_tag}"
                    )
                logger.info(f"Analyze→Prompt: sending to Ollama model={vision_tag}")
                _mode_label = "CPU mode" if _vision_num_gpu == 0 else "GPU mode"
                self.after(0, lambda ml=_mode_label, gid=analyze_gid: gid == self._analyze_gen_id and self._img_set_canvas_status(
                    f"Sent to Ollama — waiting for model… ({ml})", TEXT_MUTED
                ))

                tokens = []
                token_count = 0
                first_token_time: Optional[float] = None

                # Heartbeat: update status every 5s, log every 10s while waiting for first token
                _last_log_elapsed = [0]

                def _waiting_update():
                    if analyze_gid != self._analyze_gen_id:
                        return
                    if token_count == 0 and not self._analyze_stop_event.is_set():
                        elapsed = int(time.monotonic() - self._analyze_elapsed_start)
                        self._img_set_canvas_status(
                            f"Waiting for Ollama… {elapsed}s (loading model or processing image)",
                            WARN_TEXT,
                        )
                        if elapsed - _last_log_elapsed[0] >= 10:
                            logger.info(f"Analyze→Prompt: still waiting — {elapsed}s elapsed, no tokens yet")
                            _last_log_elapsed[0] = elapsed
                        self.after(5000, lambda gid=analyze_gid: gid == self._analyze_gen_id and _waiting_update())

                self.after(5000, lambda gid=analyze_gid: gid == self._analyze_gen_id and _waiting_update())

                # 15-25 tags × ~5 tokens each = up to ~125 tokens; leave headroom so the
                # final tag (often the realism cue like "photorealistic") doesn't get cut.
                _MAX_TOKENS = 240
                for token in self.ollama.chat_stream(
                    vision_tag, messages, temperature=0.3,
                    num_gpu=_vision_num_gpu,
                    stop_event=self._analyze_stop_event,
                    num_predict=_MAX_TOKENS,
                ):
                    if analyze_gid != self._analyze_gen_id:
                        return
                    if self._analyze_stop_event.is_set():
                        logger.info(f"Analyze→Prompt: cancelled by user after {token_count} tokens")
                        self.after(0, lambda gid=analyze_gid: self._analyze_cancelled(gid))
                        return

                    if token_count == 0:
                        first_token_time = time.monotonic()
                        wait_secs = round(first_token_time - self._analyze_elapsed_start, 1)
                        logger.info(f"Analyze→Prompt: first token received after {wait_secs}s "
                                    f"({'cold load' if wait_secs > 5 else 'warm'})")
                        self.after(0, lambda gid=analyze_gid: gid == self._analyze_gen_id and self._img_set_canvas_status("Receiving…", TEXT_MUTED))

                    tokens.append(token)
                    token_count += 1

                    if token_count % 5 == 0:
                        preview = "".join(tokens)[-50:]
                        self.after(0, lambda n=token_count, p=preview, gid=analyze_gid: gid == self._analyze_gen_id and self._img_set_canvas_status(
                            f"Receiving… {n} tokens — {p}", TEXT_MUTED
                        ))

                    if token_count % 10 == 0:
                        logger.info(f"Analyze→Prompt: {token_count} tokens received")

                    # Hard cap — stop consuming if model ignores num_predict
                    if token_count >= _MAX_TOKENS:
                        logger.warning(f"Analyze→Prompt: hit hard cap of {_MAX_TOKENS} tokens — stopping stream")
                        break

                prompt_text = "".join(tokens).strip()
                if not prompt_text:
                    logger.warning("Analyze→Prompt: Ollama returned an empty response")
                    self.after(0, lambda gid=analyze_gid: self._analyze_error(
                        "Ollama returned an empty response — the vision model may have failed to process the image",
                        gid,
                    ))
                    return
                if first_token_time:
                    stream_secs = round(time.monotonic() - first_token_time, 1)
                    logger.info(f"Analyze→Prompt: stream finished — {token_count} tokens "
                                f"in {stream_secs}s — result: {prompt_text[:120]}")
                else:
                    logger.warning("Analyze→Prompt: completed but received 0 tokens — empty response")
                self.after(0, lambda pt=prompt_text, gid=analyze_gid: self._apply_analyzed_prompt(pt, gid))

            except Exception as exc:
                import traceback
                full_msg = str(exc)
                tb = traceback.format_exc()
                logger.error(f"Analyze→Prompt EXCEPTION: {full_msg}")
                logger.error(f"Analyze→Prompt TRACEBACK: {tb}")
                self.after(0, lambda m=full_msg, gid=analyze_gid: self._analyze_error(m, gid))

            finally:
                # Keep ComfyUI stopped after vision so users can analyze several
                # references in a row. Generate restarts it when they are ready.
                if _comfyui_was_ok:
                    logger.info("Analyze→Prompt: ComfyUI restart deferred for additional vision prompts")
                    self.after(0, lambda gid=analyze_gid: gid == self._analyze_gen_id and self._mark_comfyui_deferred_after_vision())

        self._analyze_thread = threading.Thread(target=_worker, daemon=True)
        self._analyze_thread.start()
        # v2026.06.01.8: now that the thread is alive, refresh the
        # vision picker UI so the state machine flips the analyze-row
        # to the Stop button. The direct ``configure(state="disabled")``
        # calls above keep the button-state-based ``_analyze_tick_elapsed``
        # gating intact (it reads ``_img_analyze_btn.cget("state")``).
        try:
            self._refresh_vision_model_ui()
        except Exception as exc:
            logger.debug(f"Vision picker refresh on analyze start failed: {exc}")

    def _stop_analyze(self):
        """User clicked Stop — signal the worker thread to abort."""
        self._analyze_stop_event.set()
        self._img_analyze_stop_btn.configure(
            state="disabled",
            fg_color=self._IG_DISABLED_FG,
            text_color_disabled=self._IG_DISABLED_TEXT,
        )
        self._img_set_canvas_status("Stopping …", WARN_TEXT)
        logger.info("Analyze→Prompt: stop requested by user")

    def _analyze_cancelled(self, analyze_gid: Optional[int] = None):
        """Called on main thread after worker detects stop event."""
        if analyze_gid is not None and analyze_gid != self._analyze_gen_id:
            return
        self._analyze_cleanup_ui()
        self._img_set_canvas_status("Cancelled", WARN_TEXT)

    def _analyze_tick_elapsed(self, analyze_gid: Optional[int] = None):
        """Update the elapsed-time label while analysis is running."""
        if analyze_gid is not None and analyze_gid != self._analyze_gen_id:
            return
        if not self._analyze_stop_event.is_set() and self._img_analyze_btn.cget("state") == "disabled":
            elapsed = int(time.monotonic() - self._analyze_elapsed_start)
            self._img_analyze_elapsed_lbl.configure(text=f"{elapsed}s")
            self._analyze_elapsed_timer_id = self.after(1000, lambda gid=analyze_gid: self._analyze_tick_elapsed(gid))

    def _analyze_cleanup_ui(self):
        """Reset all analyze UI controls to idle state."""
        self._refresh_vision_model_ui()
        self._img_analyze_stop_btn.configure(
            state="disabled",
            fg_color=self._IG_DISABLED_FG,
            text_color_disabled=self._IG_DISABLED_TEXT,
        )
        self._img_analyze_progress.stop()
        self._img_analyze_progress.grid_remove()
        self._img_analyze_elapsed_lbl.configure(text="")
        self._img_analyze_elapsed_lbl.grid_remove()
        if self._analyze_elapsed_timer_id:
            self.after_cancel(self._analyze_elapsed_timer_id)
            self._analyze_elapsed_timer_id = None

    def _apply_analyzed_prompt(self, prompt_text: str, analyze_gid: Optional[int] = None):
        if analyze_gid is not None and analyze_gid != self._analyze_gen_id:
            return
        self._img_prompt.delete("1.0", "end")
        self._img_prompt.insert("1.0", prompt_text)
        self._img_prompt.see("1.0")
        if hasattr(self, "_img_prompt_status") and self._img_prompt_status is not None:
            self._img_prompt_status.configure(
                text="Updated from vision analysis. Analyze another image or click Generate.",
                text_color=SUCCESS_TEXT,
            )
        self._analyze_cleanup_ui()
        self._img_set_canvas_status(
            "Prompt updated above the canvas. You can analyze another image before generating.",
            SUCCESS_TEXT,
        )

    def _analyze_error(self, msg: str, analyze_gid: Optional[int] = None):
        if analyze_gid is not None and analyze_gid != self._analyze_gen_id:
            return
        self._analyze_cleanup_ui()
        display = msg if len(msg) <= 120 else msg[:117] + "…"
        self._img_set_canvas_status(f"Error: {display}", ERROR_TEXT)
        logger.error(f"Analyze→Prompt display error: {msg}")

    def _comfyui_model_launch_flags(self, model: dict | None = None) -> list[str]:
        """Return model-specific ComfyUI startup flags for Image Gen/benchmarks."""
        if model is None:
            try:
                model = self._selected_image_model_catalog_entry()
            except Exception:
                model = None
        raw = (model or {}).get("comfyui_launch_flags") or []
        if isinstance(raw, str):
            raw = [raw]
        flags: list[str] = []
        for flag in raw:
            text = str(flag or "").strip()
            if text and text not in flags:
                flags.append(text)
        return flags

    def _comfyui_effective_launch_flags(
        self,
        model: dict | None = None,
        force_cpu_override: Optional[bool] = None,
    ) -> list[str]:
        gpu_flags = list(self.gpu_info.get_comfyui_flags())
        if force_cpu_override is True:
            gpu_flags = ["--cpu"]
        elif force_cpu_override is False:
            gpu_flags = [f for f in gpu_flags if f != "--cpu"]
        elif self._comfyui_force_cpu and "--cpu" not in gpu_flags:
            gpu_flags = ["--cpu"]   # override CUDA/DirectML/MPS flags entirely
        for flag in self._comfyui_model_launch_flags(model):
            if flag not in gpu_flags:
                gpu_flags.append(flag)
        return gpu_flags

    def _image_model_launch_flags_need_restart(
        self,
        model: dict | None = None,
        force_cpu_override: Optional[bool] = None,
    ) -> bool:
        required = set(self._comfyui_model_launch_flags(model))
        try:
            if not self.comfyui.is_running():
                return False
        except Exception:
            return False
        proc = getattr(self, "comfyui_process", None)
        try:
            owns_live_process = bool(proc and proc.poll() is None)
        except Exception:
            owns_live_process = False
        active = set(getattr(self, "_comfyui_current_launch_flags", []) or []) if owns_live_process else set()
        if force_cpu_override is not None:
            if ("--cpu" in active) != bool(force_cpu_override):
                return True
        if not required:
            return False
        return not required.issubset(active)

    def _stop_comfyui_for_restart(self, *, reason: str, kill_orphans: bool = False) -> None:
        proc = getattr(self, "comfyui_process", None)
        if proc and proc.poll() is None:
            try:
                logger.info(
                    f"ComfyUI: terminating owned process PID {proc.pid} for {reason}",
                    category=logger.CATEGORY_COMFYUI,
                )
                proc.terminate()
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                    logger.info(
                        "ComfyUI: owned process killed (force)",
                        category=logger.CATEGORY_COMFYUI,
                    )
                except Exception as exc:
                    logger.warning(
                        f"ComfyUI: could not kill owned process for {reason}: {exc}",
                        category=logger.CATEGORY_COMFYUI,
                    )
        self.comfyui_process = None
        self._comfyui_current_launch_flags = []
        self._close_comfyui_log_handle()
        if kill_orphans:
            orphans = self._kill_orphan_comfyui_processes()
            if orphans:
                time.sleep(1)
        try:
            self.comfyui.reconnect()
        except Exception:
            pass

    def _start_comfyui_process(
        self,
        model: dict | None = None,
        force_cpu_override: Optional[bool] = None,
    ) -> bool:
        """
        Start ComfyUI in a subprocess with appropriate GPU flags.

        Returns:
            True if started successfully, False otherwise.

        On failure, stamps ``self._comfyui_last_start_failure_reason`` with a
        specific reason (not-installed / dep-install-failed / popen-failed)
        so callers (esp. the benchmark worker via ``_bench_ensure_comfyui_ready``)
        can surface it in the user-visible error string instead of the generic
        "ComfyUI is not running and could not be started" placeholder.
        """
        # Check if already running
        if self.comfyui_process and self.comfyui_process.poll() is None:
            if self._image_model_launch_flags_need_restart(
                model,
                force_cpu_override=force_cpu_override,
            ):
                self._stop_comfyui_for_restart(
                    reason="launch flag change", kill_orphans=True
                )
            else:
                logger.debug("ComfyUI process already running", category=logger.CATEGORY_COMFYUI)
                return True

        # Check if ComfyUI is installed
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            reason = "ComfyUI not installed at expected paths (config.json comfyui_dir, comfyui_path.bat, or ./ComfyUI)"
            logger.warning(reason, category=logger.CATEGORY_COMFYUI)
            self._set_comfyui_start_failure_reason(reason)
            return False

        # Get Python executable
        python_exe = sys.executable  # fallback to current Python

        if sys.platform == "win32":
            python_path_file = Path(__file__).parent.parent / "python_path.bat"
            if python_path_file.exists():
                try:
                    with open(python_path_file, 'r') as f:
                        for line in f:
                            if line.startswith("set LOCALAI_PYTHON="):
                                python_exe = line.split("=", 1)[1].strip()
                                break
                except Exception as e:
                    logger.debug(f"Could not read python_path.bat: {e}", category=logger.CATEGORY_COMFYUI)

        # v5.3.6+ cold-start diagnostics: log the install path + python before
        # the slow dep-install / subprocess.Popen so support can correlate a
        # failure with the actual paths involved.
        logger.info(f"ComfyUI install path: {comfyui_path}", category=logger.CATEGORY_COMFYUI)
        logger.info(f"ComfyUI python: {python_exe}", category=logger.CATEGORY_COMFYUI)

        if not self._ensure_comfyui_core_dependencies(python_exe):
            if not self._get_comfyui_start_failure_reason():
                self._set_comfyui_start_failure_reason(
                    "ComfyUI core dependency check/install failed — see ComfyUI status logs"
                )
            return False

        # Ensure LocalAI custom nodes (e.g. Chroma passthrough) are in place before
        # ComfyUI loads — write them proactively so no restart is needed on first use.
        self._chroma_node_needs_restart = False
        self._ensure_chroma_support(comfyui_path)

        # Build command with GPU and catalog-specific launch flags.
        if force_cpu_override is None:
            gpu_flags = self._comfyui_effective_launch_flags(model)
        else:
            gpu_flags = self._comfyui_effective_launch_flags(
                model,
                force_cpu_override=force_cpu_override,
            )
        main_py = comfyui_path / "main.py"
        cmd = [python_exe, str(main_py), "--listen", "127.0.0.1"]
        cmd.extend(gpu_flags)

        # Start process
        try:
            mode_note = " (CPU mode)" if "--cpu" in gpu_flags else ""
            logger.info(f"Starting ComfyUI{mode_note} with {self.gpu_info} acceleration...", category=logger.CATEGORY_COMFYUI)
            # Bumped INFO so support reports include the exact launch command
            # without users having to flip a debug switch.
            logger.info(f"ComfyUI launch cmd: {cmd}", category=logger.CATEGORY_COMFYUI)

            # Redirect output to log file — keep handle open so subprocess can write
            log_file = Path(__file__).parent.parent / "comfyui.log"
            self._close_comfyui_log_handle()
            self._comfyui_log_handle = open(log_file, 'w')
            popen_kw = {}
            if sys.platform == "win32":
                popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            # v5.5.6+: when ComfyUI is launched in CPU mode, saturate all
            # logical cores by exporting OMP/MKL/NUMEXPR thread counts via
            # env vars BEFORE Popen — PyTorch+OpenMP on Windows defaults to
            # physical-cores-only when SMT/Hyper-Threading is on, which on
            # a 32-CPU cloud VM manifests as image-gen using ~50% of
            # logical processors with the rest idle.  Setting these env
            # vars before the child's torch import wins the race and the
            # child uses every logical core (typical 1.5-2x throughput on
            # 16/32-CPU SKUs).  No effect when --cpu isn't in the launch
            # flags (GPU mode is GPU-bound, not CPU-bound).
            env_for_popen = None
            if "--cpu" in gpu_flags:
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
                    f"ComfyUI CPU thread saturation: OMP_NUM_THREADS={logical} "
                    f"(MKL/NUMEXPR/OPENBLAS mirror)",
                    category=logger.CATEGORY_COMFYUI,
                )
            self.comfyui_process = subprocess.Popen(
                cmd,
                stdout=self._comfyui_log_handle,
                stderr=subprocess.STDOUT,
                env=env_for_popen,
                **popen_kw,
            )
            self._comfyui_current_launch_flags = list(gpu_flags)
            logger.info(
                f"ComfyUI subprocess started (PID {self.comfyui_process.pid}); "
                f"waiting for /system_stats…",
                category=logger.CATEGORY_COMFYUI,
            )
            logger.info(f"ComfyUI log file: {log_file}", category=logger.CATEGORY_COMFYUI)
            self._close_comfyui_log_handle()
            return True
        except Exception as e:
            reason = f"ComfyUI subprocess.Popen failed: {e}"
            logger.error(reason, category=logger.CATEGORY_COMFYUI)
            self._set_comfyui_start_failure_reason(reason)
            self._comfyui_current_launch_flags = []
            self._close_comfyui_log_handle()
            return False

    def _close_comfyui_log_handle(self):
        handle = getattr(self, "_comfyui_log_handle", None)
        if handle:
            try:
                handle.close()
            except Exception:
                pass
            self._comfyui_log_handle = None

    def _free_comfyui_vram(self):
        """Ask ComfyUI to unload all models and free VRAM."""
        # Warn if image generation is running — don't block, but log it
        if self._img_thread and self._img_thread.is_alive():
            logger.warning(
                "Free VRAM: image generation is in progress — VRAM release may interrupt it",
                category=logger.CATEGORY_COMFYUI,
            )
            self._img_set_status("Warning: Free VRAM requested during generation — generation may fail", color=WARN_TEXT)

        # Use the image gen page status line so we don't stomp on analyze→prompt status
        self._img_set_status("Freeing ComfyUI VRAM …", color=WARN_TEXT)
        logger.info("ComfyUI: Free VRAM requested by user", category=logger.CATEGORY_COMFYUI)

        def _free():
            # Always attempt the API call — don't gate on comfyui_ok flag
            try:
                ok = self.comfyui.free_vram()
            except Exception as e:
                logger.error(f"ComfyUI Free VRAM: exception: {e}", category=logger.CATEGORY_COMFYUI)
                ok = False

            if ok:
                self.comfyui_ok = True  # if it responded, it's running
                self.after(0, lambda: (
                    self._img_set_status("VRAM freed — models unloaded.", color=SUCCESS_TEXT),
                    logger.info("ComfyUI: VRAM freed successfully", category=logger.CATEGORY_COMFYUI),
                ))
            else:
                self.after(0, lambda: (
                    self._img_set_status(
                        "Free VRAM failed — ComfyUI may not be running. Try Restart.", color=ERROR_TEXT
                    ),
                    logger.warning("ComfyUI: Free VRAM failed — ComfyUI not responding", category=logger.CATEGORY_COMFYUI),
                ))

        threading.Thread(target=_free, daemon=True).start()

    def _on_img_cpu_mode_changed(self):
        """Toggle ComfyUI between GPU and CPU mode and restart."""
        self._comfyui_force_cpu = self._img_cpu_mode_var.get()
        logger.info(
            f"Image Gen: CPU mode {'enabled' if self._comfyui_force_cpu else 'disabled'} — restarting ComfyUI",
            category=logger.CATEGORY_COMFYUI,
        )
        self._update_category_for_device()   # re-evaluate which models are visible
        if hasattr(self, "_img_cpu_banner") and self._img_cpu_banner is not None:
            show_canvas_cpu_banner = self._comfyui_force_cpu and self.gpu_info.gpu_type != "cpu"
            if show_canvas_cpu_banner:
                self._img_cpu_banner.grid()
            else:
                self._img_cpu_banner.grid_remove()
        if hasattr(self, "_img_model_menu") and self._img_model_menu is not None:
            self._populate_image_model_menu(self._local_comfyui_model_files())
            self._on_img_model_changed()
        self._img_restart_comfyui()

    def _img_restart_comfyui(self):
        """Kill ComfyUI (including orphans) and restart it on a background thread."""
        # Cancel any in-progress image generation before killing the server
        if self._img_thread and self._img_thread.is_alive():
            logger.info("ComfyUI Restart: cancelling in-progress image generation", category=logger.CATEGORY_COMFYUI)
            self._img_stop_event.set()

        self._set_comfyui_status("restarting …", WARN_TEXT)
        self._set_image_gen_enabled(False)
        self._img_set_status("Restarting ComfyUI …", color=WARN_TEXT)
        logger.info("ComfyUI: Restart requested by user", category=logger.CATEGORY_COMFYUI)

        def _do_restart():
            # Step 1: terminate the process we own
            if self.comfyui_process and self.comfyui_process.poll() is None:
                try:
                    logger.info(
                        f"ComfyUI: terminating owned process PID {self.comfyui_process.pid}",
                        category=logger.CATEGORY_COMFYUI,
                    )
                    self.comfyui_process.terminate()
                    self.comfyui_process.wait(timeout=8)
                    logger.info("ComfyUI: owned process terminated cleanly", category=logger.CATEGORY_COMFYUI)
                except Exception:
                    try:
                        self.comfyui_process.kill()
                        logger.info("ComfyUI: owned process killed (force)", category=logger.CATEGORY_COMFYUI)
                    except Exception as e:
                        logger.warning(f"ComfyUI: could not kill owned process: {e}", category=logger.CATEGORY_COMFYUI)
            self.comfyui_process = None
            self._close_comfyui_log_handle()

            # Step 2: kill any orphans not tracked by this instance
            orphans = self._kill_orphan_comfyui_processes()
            if orphans:
                import time as _time
                _time.sleep(1)  # brief pause to let OS release the port

            # Step 3: reset state and restart
            self.comfyui_ok = False
            self.comfyui.reconnect()
            logger.info("ComfyUI: starting fresh instance", category=logger.CATEGORY_COMFYUI)
            self.after(0, lambda: self._img_refresh_comfyui(start_if_needed=True))

        threading.Thread(target=_do_restart, daemon=True).start()

    def _img_refresh_comfyui(self, start_if_needed: bool = False, kill_orphans: bool = False):
        def _check():
            installed = self._comfyui_installed_path()
            if not installed:
                self.after(0, self._img_on_comfyui_offline)
                return

            # When called after analyze→prompt (or any forced restart), kill any orphan
            # ComfyUI processes first — otherwise is_running() may return True for a stale
            # process from a previous session that still has its node execution cache in RAM
            # (that cache is what caused cross-session image contamination: coffee→puppy blends).
            if kill_orphans:
                if self.comfyui_process and self.comfyui_process.poll() is None:
                    try:
                        logger.info(
                            f"ComfyUI: terminating owned process PID {self.comfyui_process.pid} before restart",
                            category=logger.CATEGORY_COMFYUI,
                        )
                        self.comfyui_process.terminate()
                        self.comfyui_process.wait(timeout=8)
                    except Exception:
                        try:
                            self.comfyui_process.kill()
                        except Exception as exc:
                            logger.warning(
                                f"ComfyUI: could not kill owned process before restart: {exc}",
                                category=logger.CATEGORY_COMFYUI,
                            )
                    self.comfyui_process = None
                    self._close_comfyui_log_handle()
                orphans = self._kill_orphan_comfyui_processes()
                if orphans:
                    time.sleep(1)  # brief pause so OS releases the port

            # Probe-only refreshes keep ComfyUI stopped until the user clicks
            # Generate or explicitly restarts it.
            if not self.comfyui.is_running():
                if not start_if_needed:
                    self.comfyui_ok = False
                    self.after(0, self._img_on_comfyui_idle)
                    return
                self.after(0, lambda: (
                    self._set_comfyui_status(f"starting ({self.gpu_info}) — please wait …", WARN_TEXT),
                    self._set_image_gen_enabled(False),
                    self._img_set_status("Waiting for ComfyUI to start …", color=WARN_TEXT),
                ))
                logger.info("ComfyUI: waiting for startup /system_stats", category=logger.CATEGORY_COMFYUI)
                if not self._start_comfyui_process():
                    self.after(0, lambda: (
                        setattr(self, "_img_waiting_for_comfyui_generation", False),
                        setattr(self, "_pending_generation_after_comfyui_restart", False),
                        self._img_stop_progress(),
                        self._set_comfyui_status("failed to start — check logs", ERROR_TEXT),
                        self._set_image_gen_enabled(False),
                        self._img_set_status("ComfyUI failed to start", color=ERROR_TEXT),
                    ))
                    self.comfyui_ok = False
                    return

            # Give ComfyUI up to 60s to start
            max_wait = 60
            poll_interval = 5
            waited = 0

            while True:
                if self.comfyui.is_running():
                    self.comfyui_ok = True
                    try:
                        models = self.comfyui.get_model_list()
                    except ComfyUIError:
                        models = []
                    self.after(0, lambda m=models: self._img_on_comfyui_ready(m))
                    return
                if waited >= max_wait:
                    break
                time.sleep(poll_interval)
                waited += poll_interval
                remaining = max_wait - waited
                self.after(0, lambda r=remaining: self._set_comfyui_status(
                    f"starting — waiting … ({r}s remaining)", WARN_TEXT
                ))
                logger.info(
                    f"ComfyUI: still waiting for /system_stats ({remaining}s remaining)",
                    category=logger.CATEGORY_COMFYUI,
                )
            # Close log handle so file is flushed
            self._close_comfyui_log_handle()
            self.comfyui_ok = False
            # Check if process crashed and show log details
            if self.comfyui_process and self.comfyui_process.poll() is not None:
                exit_code = self.comfyui_process.poll()
                crash_hint = ""
                try:
                    log_file = Path(__file__).parent.parent / "comfyui.log"
                    if log_file.exists():
                        lines = log_file.read_text(errors="replace").strip().splitlines()
                        tail = lines[-8:] if len(lines) > 8 else lines
                        crash_hint = "\n".join(tail)
                except Exception:
                    pass
                msg = f"crashed (code {exit_code})"
                if crash_hint:
                    msg += f"\n{crash_hint}"
                self.after(0, lambda m=msg: self._set_comfyui_status(m, ERROR_TEXT))
                logger.error(f"ComfyUI crashed (code {exit_code}): {crash_hint}", category=logger.CATEGORY_COMFYUI)
            else:
                self.after(0, self._img_on_comfyui_offline)
        threading.Thread(target=_check, daemon=True).start()

    def _set_comfyui_status(self, text: str, color: str):
        """Update ComfyUI status in both the sidebar footer and the Image Gen page — single source of truth.

        v5: stores the last status so the Image Gen page can apply it when
        lazily built. Guards against widgets not yet existing.
        """
        self._last_comfyui_status = (text, color)
        if hasattr(self, "_comfyui_status_label") and self._comfyui_status_label is not None:
            self._comfyui_status_label.configure(text=f"ComfyUI: {text}", text_color=color)
        if hasattr(self, "_img_comfyui_status") and self._img_comfyui_status is not None:
            # v5.1 header pill includes the "ComfyUI: " prefix so the label reads
            # as a complete sentence with a colored dot/status word.
            self._img_comfyui_status.configure(text=f"ComfyUI: {text}", text_color=color)

    def _set_image_gen_enabled(self, enabled: bool):
        """Track ComfyUI readiness without disabling the Generate action.

        The Image Gen nav tab is ALWAYS clickable — users need to be able to read
        the page, pick a vision model, and review prompts even before ComfyUI is
        running. Generate is also always clickable while idle; it validates the
        current state and starts or restarts ComfyUI only when needed.
        """
        self._image_gen_enabled = enabled
        # Defensive: if a prior call disabled the nav button, force it back on.
        nav_btn = self._nav_btns.get("image_gen") if hasattr(self, "_nav_btns") else None
        if nav_btn:
            try:
                nav_btn.configure(state="normal", text_color=TEXT_PRIMARY)
            except Exception:
                pass
        if hasattr(self, "_img_generate_btn") and self._img_generate_btn is not None:
            if getattr(self, "_img_waiting_for_comfyui_generation", False):
                self._set_image_generate_button_running(True)
            elif self._img_thread is not None and self._img_thread.is_alive():
                self._set_image_generate_button_running(True)
            else:
                self._set_image_generate_button_running(False)
        self._refresh_image_readiness()

    def _image_model_is_selected(self) -> bool:
        if not hasattr(self, "_img_model_var"):
            return False
        model = (self._img_model_var.get() or "").strip()
        return bool(model and not model.startswith("("))

    def _refresh_image_readiness(self):
        if not hasattr(self, "_img_ready_lbl") or self._img_ready_lbl is None:
            return
        if hasattr(self, "_img_generate_btn") and self._img_generate_btn is not None:
            if getattr(self, "_img_waiting_for_comfyui_generation", False):
                self._set_image_generate_button_running(True)
            elif self._img_thread is not None and self._img_thread.is_alive():
                self._set_image_generate_button_running(True)
            else:
                self._set_image_generate_button_running(False)
        blockers = []
        if self._vision_comfyui_deferred_restart:
            blockers.append("ComfyUI paused for vision; Generate will restart it")
        elif not self.comfyui_ok or not self._image_gen_enabled:
            blockers.append("ComfyUI is not ready")
        if not self._image_model_is_selected():
            blockers.append("select or download an image model")
        if getattr(self, "_img_img2img_var", None) is not None and self._img_img2img_var.get():
            if not self._selected_model_supports_img2img():
                blockers.append("selected model does not support reference-image generation")
            elif not self._img_img2img_path:
                blockers.append("choose a reference image for generation")
        if blockers:
            self._img_ready_lbl.configure(
                text="Ready checklist: " + " / ".join(blockers),
                text_color=WARN_TEXT,
            )
        else:
            self._img_ready_lbl.configure(
                text="Ready: enter a prompt, then Generate.",
                text_color=SUCCESS_TEXT,
            )
        # v2026.06.01.10: image-readiness can transition when ComfyUI is
        # newly installed (which is exactly when we want to clear the
        # "ComfyUI not installed" entry from the incomplete-setup banner).
        # Refresh here so post-install completion drives the banner.
        try:
            self._refresh_setup_warning_banner()
        except Exception:
            pass

    def _mark_comfyui_deferred_after_vision(self) -> None:
        self._vision_comfyui_deferred_restart = True
        self._set_comfyui_status("paused for vision; Generate will restart it", WARN_TEXT)
        self._set_image_gen_enabled(False)
        self._refresh_image_readiness()

    def _local_comfyui_model_files(self) -> list[str]:
        """Return locally downloaded ComfyUI model filenames without starting ComfyUI."""
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path:
            return []
        files: list[str] = []
        for subdir in ("checkpoints", "diffusion_models"):
            model_dir = comfyui_path / "models" / subdir
            if not model_dir.exists():
                continue
            for path in model_dir.iterdir():
                if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".pth", ".gguf"}:
                    files.append(path.name)
        return sorted(set(files), key=str.lower)

    def _image_model_friendly_name(self, filename: str) -> str:
        """Return the catalog display name for a ComfyUI checkpoint filename.

        Falls back to the filename itself for files that aren't in the catalog
        (e.g. user-supplied checkpoints), so nothing ever disappears from the list.
        """
        if not filename:
            return filename
        for m in self._catalog_models:
            if m.get("comfyui_model") == filename:
                return m.get("name") or filename
        return filename

    def _catalog_image_model_filenames_with_url(self) -> list[str]:
        """Return ComfyUI checkpoint filenames for catalog Image Generation
        entries that have an automatic-download URL configured. Used by the
        Image Gen dropdown so users can pick a model even before its
        checkpoint is on disk — the file is auto-pulled on Generate.

        v2026.06.01.9 (Image Gen auto-pull): previously the dropdown was
        sourced purely from ComfyUI's on-disk checkpoint scan, which meant
        a fresh ComfyUI install (no models pulled) showed "(no checkpoints
        found)" and Generate refused. Now the dropdown unions on-disk
        names with downloadable catalog entries so the user can pick any
        curated image-gen model and have it silently downloaded on
        Generate, matching the Ollama chat-model UX.
        """
        return [
            m.get("comfyui_model") or ""
            for m in self._catalog_models
            if m.get("category") == "Image Generation"
            and m.get("comfyui_model")
            and m.get("comfyui_model_url")
        ]

    def _populate_image_model_menu(self, models: list[str]) -> None:
        # v2026.06.01.9: union the on-disk checkpoint list with downloadable
        # catalog image-gen entries so users can pick (and silently auto-pull)
        # a curated model even before it's been installed. The existing CPU /
        # iGPU filters below still apply, and the existing fastest-first sort
        # and user-checkpoint disambiguation are unchanged — a catalog model
        # not yet on disk is treated identically to one that is. The Generate
        # path handles the actual download (``_ensure_selected_image_model_present``).
        catalog_downloadable = self._catalog_image_model_filenames_with_url()
        seen_union: set[str] = set()
        unioned: list[str] = []
        for fn in list(models) + catalog_downloadable:
            if not fn or fn in seen_union:
                continue
            seen_union.add(fn)
            unioned.append(fn)
        models = unioned
        if getattr(self, "_comfyui_force_cpu", False):
            cpu_files = {
                m.get("comfyui_model")
                for m in self._catalog_models
                if catalog.is_cpu_viable_image_model(m)
            }
            models = [m for m in models if m in cpu_files]
        # v5.5.12 (Ron, 2026-05-27): On Windows-integrated GPUs (Intel Arc/UHD,
        # AMD Radeon Graphics, Snapdragon Adreno), DirectX TDR kills any GPU
        # kernel that runs longer than ~2s. Heavy SDXL/Flux UNets routinely
        # take 6-15s per step on iGPUs even with plenty of shared memory, so
        # the GPU is killed mid-generation with "GPU device instance has been
        # suspended". Filter the dropdown to iGPU-viable models (SD 1.5 family
        # + step-reduced SDXL Lightning). Catalog-flagged via ``igpu_viable``.
        # Apple Silicon is also unified_memory but has no TDR, so it's exempt
        # (handled by ``_windows_unified_igpu`` cache, which is False on macOS).
        # Per Ron, no in-page escape hatch — users probe heavy models via the
        # Benchmark page's "Force All" toggle. Non-catalog user-installed
        # checkpoints stay visible: we can't make an informed call on them,
        # and hiding them silently would be worse UX than letting them attempt
        # generation and surface the friendly TDR dialog on failure.
        elif getattr(self, "_windows_unified_igpu", False):
            igpu_files = {
                m.get("comfyui_model")
                for m in self._catalog_models
                if catalog.is_igpu_viable_image_model(m)
            }
            catalog_filenames = {
                m.get("comfyui_model")
                for m in self._catalog_models
                if m.get("comfyui_model")
            }
            models = [
                m for m in models
                if m in igpu_files or m not in catalog_filenames
            ]
        if models:
            # Show catalog-friendly names in the dropdown but keep a reverse map so
            # we can resolve back to the actual filename for ComfyUI calls.
            # If two filenames map to the same friendly name (shouldn't happen with
            # the current catalog, but be defensive), disambiguate by appending the filename.
            seen: dict[str, int] = {}
            triples: list[tuple[float, str, str]] = []  # (size_gb_sort, display_label, filename)
            for fn in models:
                label = self._image_model_friendly_name(fn)
                if label in seen:
                    label = f"{label} ({fn})"
                seen[label] = 1
                # v5.5.11 (Ron, 2026-05-26): Sort image dropdown fastest -> slowest.
                # Use catalog size_gb as a speed proxy (smaller models load and
                # generate faster on the same hardware). Models missing from the
                # catalog sort to the end so user-installed checkpoints don't
                # mask the curated fastest options at the top.
                entry = self._find_catalog_entry_for_model(fn)
                try:
                    size_gb = float(entry.get("size_gb")) if entry else float("inf")
                except (TypeError, ValueError):
                    size_gb = float("inf")
                triples.append((size_gb, label, fn))
            triples.sort(key=lambda t: (t[0], t[1].lower()))
            pairs: list[tuple[str, str]] = [(t[1], t[2]) for t in triples]
            unique_friendly = [p[0] for p in pairs]
            self._img_friendly_to_filename = {p[0]: p[1] for p in pairs}
            self._img_model_menu.configure(values=unique_friendly)
            current = self._img_model_var.get()
            if not current or current not in unique_friendly:
                self._img_model_var.set(unique_friendly[0])
        else:
            self._img_friendly_to_filename = {}
            self._img_model_menu.configure(values=["(no checkpoints found)"])
            self._img_model_var.set("(no checkpoints found)")

    def _img_on_comfyui_ready(self, models: list[str]):
        self._vision_comfyui_deferred_restart = False
        # Clear any queued prompts left over from a previous LocalAI session.
        # This prevents a stale generation (e.g. coffee) from running after the
        # new session has already submitted its own prompt (e.g. puppy).
        self.comfyui.clear_queue()
        self._set_comfyui_status(f"connected — {len(models)} checkpoint(s)", SUCCESS_TEXT)
        logger.info(f"ComfyUI connected with {len(models)} checkpoint(s)", category=logger.CATEGORY_COMFYUI)
        # Clear any "Waiting for ComfyUI …" message left over from startup (#bug: stale orange status).
        if not self._pending_generation_after_comfyui_restart:
            self._img_set_canvas_status("")
        self._set_image_gen_enabled(True)
        # Refresh model cards so ComfyUI models show correct status
        self.after(200, self._refresh_model_cards)
        self._populate_image_model_menu(models)
        self._refresh_image_readiness()
        if self._pending_generation_after_comfyui_restart:
            self._pending_generation_after_comfyui_restart = False
            self._img_waiting_for_comfyui_generation = False
            self.after(100, self._start_image_generation)

    def _img_on_comfyui_idle(self):
        if not getattr(self, "_img_waiting_for_comfyui_generation", False):
            self._pending_generation_after_comfyui_restart = False
        self._set_comfyui_status("installed — click Generate to start", TEXT_MUTED)
        if hasattr(self, "_img_model_menu") and self._img_model_menu is not None:
            self._populate_image_model_menu(self._local_comfyui_model_files())
        self._set_image_gen_enabled(False)
        if hasattr(self, "_img_set_status"):
            self._img_set_status("ComfyUI will start when you click Generate.", color=TEXT_MUTED)

    def _img_on_comfyui_offline(self):
        self._pending_generation_after_comfyui_restart = False
        self._img_waiting_for_comfyui_generation = False
        if self._comfyui_installed_path():
            if self.comfyui_process and self.comfyui_process.poll() is not None:
                exit_code = self.comfyui_process.poll()
                msg = f"exited (code {exit_code}) — click Restart to relaunch"
                logger.error(f"ComfyUI process exited with code {exit_code}", category=logger.CATEGORY_COMFYUI)
            else:
                msg = "not responding — click Restart to relaunch"
                logger.warning("ComfyUI not responding", category=logger.CATEGORY_COMFYUI)
        else:
            msg = "not installed — run setup.sh to install" if sys.platform == "darwin" else "not installed — run setup.bat to install"
            logger.warning("ComfyUI not installed", category=logger.CATEGORY_COMFYUI)
        self._set_comfyui_status(msg, ERROR_TEXT)
        if hasattr(self, "_img_model_menu") and self._img_model_menu is not None:
            self._img_model_menu.configure(values=["(ComfyUI offline)"])
            self._img_model_var.set("(ComfyUI offline)")
        self._set_image_gen_enabled(False)
        if hasattr(self, "_img_set_status"):
            try:
                self._img_set_status("Waiting for ComfyUI …", color=WARN_TEXT)
            except (AttributeError, Exception):
                pass

    def _on_aspect_changed(self, preset_name: str):
        """Set width/height from aspect ratio preset."""
        if preset_name in self._aspect_presets:
            w, h = self._aspect_presets[preset_name]
            self._img_width_var.set(w)
            self._img_height_var.set(h)

    def _find_catalog_entry_for_model(self, comfyui_filename: str) -> dict | None:
        """Look up the catalog entry whose `comfyui_model` matches the given
        filename. Used by `_on_img_model_changed` to pull per-model
        `recommended_settings`. Returns None if the filename isn't found in
        the active catalog (e.g. a user-installed model not yet catalogued).
        """
        if not comfyui_filename:
            return None
        try:
            for m in self._catalog_models:
                if m.get("comfyui_model") == comfyui_filename:
                    return m
        except Exception:
            pass
        return None

    def _apply_recommended_settings(self, rec: dict) -> None:
        """Apply a `recommended_settings` block from the catalog to the UI vars.

        Field semantics (every key is optional; missing keys are left alone):
          width / height — exact pixel dimensions
          aspect         — must match a key in _aspect_presets ("Square 1:1",
                           "SD 512×768", etc); width/height take precedence
                           afterwards so the dropdown's preset-driven w/h is
                           overridden with the exact catalog values
          sampler        — must match a value in the sampler dropdown
          scheduler      — must match a value in the scheduler dropdown
          steps / cfg    — numeric defaults
          cfg_locked     — True for CFG-distilled families (Flux, Z-Image,
                           Chroma, AuraFlow, SD-Turbo, SDXL-Turbo, SDXL-Lightning)
          family_label   — short string used in the "🔒 X requires CFG≈1.0" badge

        Setting order is important: aspect first (its change-handler will set
        width/height to the preset), then width/height (overrides preset),
        then sampler/scheduler/steps/cfg. The CFG lock is applied last and
        controlled exclusively by `cfg_locked` — never by string heuristics.
        """
        try:
            aspect = rec.get("aspect")
            if aspect and aspect in self._aspect_presets:
                self._img_aspect_var.set(aspect)
            w = rec.get("width")
            h = rec.get("height")
            if w is not None:
                self._img_width_var.set(str(int(w)))
            if h is not None:
                self._img_height_var.set(str(int(h)))
            sampler = rec.get("sampler")
            if sampler:
                self._img_sampler_var.set(str(sampler))
            scheduler = rec.get("scheduler")
            if scheduler:
                self._img_scheduler_var.set(str(scheduler))
            steps = rec.get("steps")
            if steps is not None:
                self._img_steps_var.set(str(int(steps)))
            cfg = rec.get("cfg")
            if cfg is not None:
                self._img_cfg_var.set(f"{float(cfg):g}")
        except Exception as exc:
            logger.debug(f"Recommended-settings apply failed: {exc}")

    def _on_img_model_changed(self, *_args):
        """Adjust UI defaults when the selected model changes.

        v5.1 behaviour:
        1. Look up the model's `recommended_settings` block in the catalog
           (configurable in models_catalog.json — single source of truth).
        2. If present, apply those exact values + honor `cfg_locked` for the
           CFG-distilled family lock badge.
        3. If absent, fall back to the legacy filename-substring heuristics so
           user-installed/uncatalogued models still get sensible defaults.
        """
        display_name = self._img_model_var.get()
        # Translate friendly label back to the actual filename — the family-detection
        # below relies on substrings like "flux", "z_image", "sd_xl" that only appear
        # in the filename, not in the catalog display name.
        model_name = self._img_friendly_to_filename.get(display_name)
        if not model_name:
            # Dropdown map not populated yet (e.g. opened from a model card before
            # ComfyUI's first refresh). Fall back to the catalog so the right family
            # defaults still apply.
            for m in self._catalog_models:
                if m.get("name") == display_name:
                    model_name = m.get("comfyui_model") or display_name
                    break
            if not model_name:
                model_name = display_name
        lower = model_name.lower()
        self._apply_selected_model_negative_prompt()

        # ── 1. Catalog-driven path: pull the per-model recommended_settings
        catalog_entry = self._find_catalog_entry_for_model(model_name)
        rec = catalog_entry.get("recommended_settings") if catalog_entry else None
        used_catalog_recs = False

        if isinstance(rec, dict) and rec:
            self._apply_recommended_settings(rec)
            used_catalog_recs = True
            cfg_locked = bool(rec.get("cfg_locked"))
            family_label = rec.get("family_label") or "this model"
            try:
                if cfg_locked:
                    self._img_cfg_entry.configure(state="disabled")
                    self._img_cfg_lock_lbl.configure(
                        text=f"🔒 {family_label} requires CFG≈{self._img_cfg_var.get()}",
                    )
                else:
                    self._img_cfg_entry.configure(state="normal")
                    self._img_cfg_lock_lbl.configure(text="")
            except Exception:
                pass
            self._clamp_image_params_for_igpu()
            self._apply_image_demo_prompt(catalog_entry)
            self._refresh_img2img_controls()
            # v5.5.14 (Ron, 2026-05-29): if img2img mode survived the refresh
            # (new model supports it), override the freshly-loaded model demo
            # prompt with the documented img2img defaults so the user always
            # sees the ref-image sample while ref-image mode is on.
            if hasattr(self, "_img_img2img_var") and self._img_img2img_var.get():
                self._apply_img2img_default_prompts()
            self._refresh_image_readiness()
            return

        # ── 2. Heuristic fallback: filename-substring detection for catalogue
        #       gaps (user-installed models that aren't in models_catalog.json).
        logger.debug(
            f"No recommended_settings for {model_name!r} — falling back to "
            f"family heuristics. Add a 'recommended_settings' block in "
            f"models_catalog.json for this model to make defaults explicit."
        )

        # Flux-architecture: native Flux GGUF/safetensors and Flux fine-tunes
        is_flux = lower.endswith('.gguf') or 'flux' in lower
        is_chroma = 'chroma' in lower
        is_schnell = 'schnell' in lower
        # Z-Image family: DiT with Qwen text encoder (turbo=8 steps, base=20 steps)
        is_z_image = 'z_image' in lower
        is_z_image_turbo = 'z_image_turbo' in lower
        # HunyuanImage uses flow-matching like Flux
        is_hunyuan = 'hunyuan' in lower
        # SDXL-family: base SDXL, Juggernaut XL (exclude z_image which is not SDXL)
        is_sdxl = (
            not is_z_image and (
                'sd_xl' in lower or 'sdxl' in lower
                or 'juggernaut-xl' in lower or 'juggernaut_xl' in lower
            )
        )
        is_sdxl_fast = is_sdxl and ('lightning' in lower or 'turbo' in lower)
        # SD 1.5 fine-tunes (Realistic Vision, etc.)
        is_sd15_finetune = 'realistic_vision' in lower or 'realistic-vision' in lower

        if is_chroma:
            # Chroma x0 pixel-space flow models require CFG 1.0.
            self._img_cfg_var.set("1.0")
            self._img_width_var.set("1024")
            self._img_height_var.set("1024")
            self._img_sampler_var.set("euler")
            self._img_scheduler_var.set("simple")
            self._img_aspect_var.set("Square 1:1")
            self._img_steps_var.set("28")
        elif is_flux:
            # Flux: CFG 1.0, euler + simple
            self._img_cfg_var.set("1.0")
            self._img_width_var.set("1024")
            self._img_height_var.set("1024")
            self._img_sampler_var.set("euler")
            self._img_scheduler_var.set("simple")
            self._img_aspect_var.set("Square 1:1")
            self._img_steps_var.set("4" if is_schnell else "20")
        elif is_z_image:
            # Z-Image family: DiT with Qwen encoder, res_multistep sampler
            self._img_cfg_var.set("1.0")
            self._img_width_var.set("1024")
            self._img_height_var.set("1024")
            self._img_sampler_var.set("euler")
            self._img_scheduler_var.set("simple")
            self._img_aspect_var.set("Square 1:1")
            self._img_steps_var.set("8" if is_z_image_turbo else "20")
        elif is_hunyuan:
            # HunyuanImage: flow-matching, similar to Flux
            self._img_cfg_var.set("1.0")
            self._img_width_var.set("1024")
            self._img_height_var.set("1024")
            self._img_sampler_var.set("euler")
            self._img_scheduler_var.set("simple")
            self._img_aspect_var.set("Square 1:1")
            self._img_steps_var.set("20")
        elif is_sdxl_fast:
            # SDXL Turbo/Lightning checkpoints are CFG-distilled; UI must match backend CFG=1.0.
            self._img_steps_var.set("8")
            self._img_cfg_var.set("1.0")
            self._img_width_var.set("1024")
            self._img_height_var.set("1024")
            self._img_sampler_var.set("euler")
            self._img_scheduler_var.set("sgm_uniform")
            self._img_aspect_var.set("Square 1:1")
        elif is_sdxl:
            # SDXL-family: 1024×1024, dpmpp_2m + karras, 30 steps, CFG 7.0
            self._img_steps_var.set("30")
            self._img_cfg_var.set("7.0")
            self._img_width_var.set("1024")
            self._img_height_var.set("1024")
            self._img_sampler_var.set("dpmpp_2m")
            self._img_scheduler_var.set("karras")
            self._img_aspect_var.set("Square 1:1")
        elif not model_name.startswith("("):
            # SD 1.5 and fine-tunes (Realistic Vision, etc.): 512×512
            self._img_steps_var.set("25" if is_sd15_finetune else "20")
            self._img_cfg_var.set("7.0")
            self._img_width_var.set("512")
            self._img_height_var.set("512")
            self._img_sampler_var.set("euler_ancestral" if is_sd15_finetune else "euler")
            self._img_scheduler_var.set("karras" if is_sd15_finetune else "normal")
            self._img_aspect_var.set("SD 512")

        # v5: Lock CFG entry when the selected model family ignores CFG
        # (Flux, Z-Image, Chroma, Hunyuan — all use distillation / flow matching
        #  that mandates CFG=1.0; values >1 silently overflow VRAM 4-9×).
        try:
            lock_cfg = bool(is_flux or is_z_image or is_chroma or is_hunyuan or is_sdxl_fast)
            if lock_cfg:
                self._img_cfg_entry.configure(state="disabled")
                family = (
                    "Chroma" if is_chroma else
                    "Z-Image" if is_z_image else
                    "Hunyuan" if is_hunyuan else
                    "SDXL Lightning/Turbo" if is_sdxl_fast else
                    "Flux"
                )
                self._img_cfg_lock_lbl.configure(
                    text=f"🔒 {family} requires CFG≈{self._img_cfg_var.get()}",
                )
            else:
                self._img_cfg_entry.configure(state="normal")
                self._img_cfg_lock_lbl.configure(text="")
        except Exception:
            pass
        self._clamp_image_params_for_igpu()
        self._apply_image_demo_prompt(catalog_entry)
        self._refresh_img2img_controls()
        # v5.5.14 (Ron, 2026-05-29): same img2img-mode override as the
        # catalog branch above — keep the documented ref-image sample
        # visible whenever img2img is on after the refresh.
        if hasattr(self, "_img_img2img_var") and self._img_img2img_var.get():
            self._apply_img2img_default_prompts()
        self._refresh_image_readiness()

    def _clamp_image_params_for_igpu(self) -> None:
        """Clamp image-gen defaults to TDR-safe ranges on Windows-iGPU SKUs.

        v5.5.12 (Ron, 2026-05-27): Defaults like 30 steps at 768×512 (Realistic
        Vision V6) or 1024×1024 (SDXL) reliably TDR on Intel Arc Graphics, AMD
        Radeon Graphics, and Snapdragon Adreno because DirectX kills any GPU
        kernel running longer than ~2s. Clamping happens AFTER ``_apply_recommended_settings``
        / heuristic defaults, so users see safe values pre-populated in the UI
        and a successful first generation. Steps capped at 10; width / height
        capped at 512. CFG / sampler / scheduler unchanged — they don't shift
        per-kernel duration meaningfully. Users can manually override the
        clamped values if they want to probe the TDR ceiling.

        Apple Silicon iGPUs are also unified_memory but have no TDR
        (``_windows_unified_igpu`` is False), so this is a no-op there.
        """
        if not getattr(self, "_windows_unified_igpu", False):
            return
        try:
            steps_raw = self._img_steps_var.get() or "10"
            steps = int(steps_raw)
            if steps > 10:
                self._img_steps_var.set("10")
        except (ValueError, AttributeError, TypeError):
            pass
        for var_name in ("_img_width_var", "_img_height_var"):
            try:
                var = getattr(self, var_name, None)
                if var is None:
                    continue
                val_raw = var.get() or "512"
                val = int(val_raw)
                if val > 512:
                    var.set("512")
            except (ValueError, AttributeError, TypeError):
                pass

    def _copy_seed_to_clipboard(self):
        """v5: Copy the last-used seed to the clipboard. Falls back to the
        seed entry value if no generation has run yet."""
        try:
            params = getattr(self, '_img_last_params', None) or {}
            seed = params.get("seed")
            if seed is None or seed == -1:
                seed = self._img_seed_var.get()
            self.clipboard_clear()
            self.clipboard_append(str(seed))
            self.update()  # required for Windows clipboard to commit
            self.set_status(f"Seed {seed} copied to clipboard.")
        except Exception as e:
            self.set_status(f"Copy seed failed: {e}")

    def _start_image_generation(self):
        # v5.5.0 UX fix: immediate-disable + canvas-status update so the user
        # gets feedback BEFORE the sync pre-flight validation runs. The button
        # text flips to "Starting…", the canvas status surface (the a11y SR
        # signal — Tk has no aria-live equivalent) reads the same. On every
        # early-return validation path below we call
        # ``_set_image_generate_button_running(False)`` AFTER the messagebox
        # (the modal is blocking, so reset must follow it — otherwise the
        # button looks stuck during the dialog).
        self._immediate_disable_btn(
            self.__dict__.get("_img_generate_btn"),
            text="Starting…",
            status_setter=self.__dict__.get("_img_set_canvas_status"),
            status_text="Starting generation…",
        )
        # ── Cancel / wait for any previous generation thread ──────────────
        if getattr(self, "_img_support_prep_in_progress", False):
            self._img_set_status("Image model support is still being prepared; generation will continue automatically.", color=WARN_TEXT)
            self._set_image_generate_button_running(False)
            return
        if getattr(self, "_img_checkpoint_download_in_progress", False):
            self._img_set_status(
                "Image checkpoint is still downloading; generation will continue automatically.",
                color=WARN_TEXT,
            )
            self._set_image_generate_button_running(False)
            return
        if getattr(self, "_img_waiting_for_comfyui_generation", False):
            self._img_set_status("ComfyUI is still starting; generation will continue automatically.", color=WARN_TEXT)
            self._img_show_progress(mode="indeterminate", color=WARN_TEXT)
            self._set_image_generate_button_running(False)
            return
        if self._img_thread is not None and self._img_thread.is_alive():
            self._img_stop_event.set()
            self._img_thread.join(timeout=5)

        model_display = self._img_model_var.get()
        if not model_display or model_display.startswith("("):
            messagebox.showinfo("No model selected", "Select a checkpoint model first.", parent=self)
            self._set_image_generate_button_running(False)
            return
        # Resolve the catalog-friendly label back to the actual filename ComfyUI expects.
        model_filename = self._img_friendly_to_filename.get(model_display, model_display)
        img2img_enabled = bool(
            getattr(self, "_img_img2img_var", None) is not None
            and self._img_img2img_var.get()
        )
        img2img_ref_path = None
        img2img_denoise = 0.55
        if img2img_enabled:
            if not self._selected_model_supports_img2img():
                messagebox.showinfo(
                    "Reference image not supported",
                    "The selected model does not support reference-image generation yet.",
                    parent=self,
                )
                self._set_image_generate_button_running(False)
                return
            img2img_ref_path = self._img_img2img_path
            if not img2img_ref_path or not Path(img2img_ref_path).exists():
                messagebox.showinfo(
                    "Reference image required",
                    "Choose a reference image before generating in reference-image mode.",
                    parent=self,
                )
                self._set_image_generate_button_running(False)
                return
            try:
                img2img_denoise = float(self._img_denoise_var.get())
            except Exception:
                img2img_denoise = 0.55

        positive = self._img_prompt.get("1.0", "end").strip()
        if not positive:
            messagebox.showinfo("Empty prompt", "Enter a description of the image to generate.", parent=self)
            self._set_image_generate_button_running(False)
            return

        # ── Content filter ────────────────────────────────────────────────
        blocked_term = content_filter.check_prompt(positive)
        if blocked_term:
            messagebox.showerror(
                "Prompt blocked",
                "Your prompt contains content that violates the usage policy.\n\n"
                "Image generation is intended for professional and appropriate use only.\n\n"
                "Edit your prompt and try again.",
                parent=self,
            )
            logger.warning(f"Image prompt blocked (matched: {blocked_term})", category=logger.CATEGORY_IMAGE_GEN)
            self._set_image_generate_button_running(False)
            return

        negative = self._img_neg_prompt.get().strip()
        blocked_neg = content_filter.check_prompt(negative) if negative else None
        if blocked_neg:
            messagebox.showerror(
                "Prompt blocked",
                "Your negative prompt contains content that violates the usage policy.\n\n"
                "Edit your prompt and try again.",
                parent=self,
            )
            logger.warning(f"Negative prompt blocked (matched: {blocked_neg})", category=logger.CATEGORY_IMAGE_GEN)
            self._set_image_generate_button_running(False)
            return

        # v2026.06.01.9: auto-pull the selected checkpoint silently if it's
        # a downloadable catalog entry that isn't on disk yet. Matches the
        # chat-tab UX (Ollama models auto-pull on first use). Returns True
        # when a background download started — we MUST return here and let
        # ``_image_checkpoint_downloaded`` re-enter Generate when ready.
        ensure_result = self._ensure_selected_image_checkpoint_present(model_filename)
        if ensure_result is True:
            return
        if ensure_result is False:
            return

        try:
            width = int(self._img_width_var.get() or 512)
            height = int(self._img_height_var.get() or 512)
            steps = int(self._img_steps_var.get() or 20)
            cfg = float(self._img_cfg_var.get() or 7.0)
            seed = int(self._img_seed_var.get() or -1)
            sampler = self._img_sampler_var.get()
            scheduler = self._img_scheduler_var.get()
        except ValueError:
            messagebox.showerror("Invalid settings",
                                 "Width, Height, Steps, and Seed must be numbers.",
                                 parent=self)
            self._set_image_generate_button_running(False)
            return

        # v5.5.14: when "Match reference aspect" is on and a reference image
        # is loaded, snap the workflow's W×H to the closest standard SDXL
        # bucket for the reference's aspect. This fixes the "ImageScale
        # crop=center cuts off the head" failure mode you get when the
        # reference is portrait but the target size is square. The user's
        # selected dimensions still drive the "native" scale (1024 for SDXL,
        # 512 for SD1.5) — only the aspect ratio is overridden.
        if (
            img2img_enabled
            and img2img_ref_path
            and bool(getattr(self, "_img_match_aspect_var", None) and self._img_match_aspect_var.get())
        ):
            try:
                from PIL import Image as _PIL_AspectImage
                with _PIL_AspectImage.open(img2img_ref_path) as _ref:
                    _rw, _rh = _ref.size
                # Pick native from the user's selected long edge — keeps
                # SD1.5 (512-family) and SDXL (1024-family) both working.
                _native = 1024 if max(width, height) >= 768 else 512
                from src import comfyui_client as _cc_mod
                snapped_w, snapped_h = _cc_mod.snap_to_aspect_bucket(
                    _rw, _rh, native=_native,
                )
                if (snapped_w, snapped_h) != (width, height):
                    logger.info(
                        f"Reference aspect-match: {width}x{height} → "
                        f"{snapped_w}x{snapped_h} (ref {_rw}x{_rh}, "
                        f"aspect {_rw/_rh:.3f})",
                        category=logger.CATEGORY_IMAGE_GEN,
                    )
                    width, height = snapped_w, snapped_h
            except Exception as _aspect_exc:
                logger.debug(f"Reference aspect-match skipped: {_aspect_exc}")

        missing_runtime_support = self._image_model_runtime_support_missing_items(model_filename)
        if missing_runtime_support:
            proceed = messagebox.askyesno(
                "Prepare Image Gen Support",
                "This local model needs additional ComfyUI support before it can run:\n\n"
                + "\n".join(f"  - {item}" for item in missing_runtime_support)
                + "\n\nPrepare it now? Generation will continue automatically when ready.",
                parent=self,
            )
            if not proceed:
                self._set_image_generate_button_running(False)
                return
            self._prepare_image_model_support_async(model_filename, missing_runtime_support)
            return

        if not self._ensure_image_model_runtime_support(model_filename, prompt=False):
            messagebox.showerror(
                "Image model support failed",
                "LocalAI could not prepare the required ComfyUI support for this model. Check localai.log and comfyui.log for details.",
                parent=self,
            )
            self._set_image_generate_button_running(False)
            return

        # ── Start/restart ComfyUI only after the user has a valid render request.
        self.comfyui.reconnect()  # drop stale HTTP keep-alives
        catalog_entry = self._find_catalog_entry_for_model(model_filename)
        needs_runtime_restart = self._image_model_runtime_needs_restart(model_filename)
        needs_launch_flag_restart = self._image_model_launch_flags_need_restart(catalog_entry)
        needs_restart = (
            self._vision_comfyui_deferred_restart
            or needs_runtime_restart
            or needs_launch_flag_restart
        )
        if needs_restart or not self.comfyui.is_running():
            self.comfyui_ok = False
            self._pending_generation_after_comfyui_restart = True
            self._img_waiting_for_comfyui_generation = True
            self._set_image_generate_button_running(True)
            self._img_stop_btn.configure(state="disabled")
            self._img_save_btn.configure(state="disabled")
            if needs_restart:
                status_label = "restarting to load model support ..." if needs_runtime_restart else "restarting for image generation ..."
                self._set_comfyui_status(status_label, WARN_TEXT)
                self._img_set_status(
                    "Restarting ComfyUI, then generation will start automatically ...",
                    color=WARN_TEXT,
                )
                wait_label = "Restarting ComfyUI to load model support" if needs_runtime_restart else "Restarting ComfyUI for image generation"
            else:
                self._set_comfyui_status("starting for image generation ...", WARN_TEXT)
                self._img_set_status(
                    "Starting ComfyUI, then generation will start automatically ...",
                    color=WARN_TEXT,
                )
                wait_label = "Starting ComfyUI for image generation"
            self._img_safe_clear_display(
                f"{wait_label} ...\n\nYour prompt is queued and will run automatically when the image engine is ready.",
                WARN_TEXT,
            )
            self._img_show_progress(mode="indeterminate", color=WARN_TEXT)
            if self._img_elapsed_timer_id:
                self.after_cancel(self._img_elapsed_timer_id)
                self._img_elapsed_timer_id = None
            self._img_elapsed_start = time.time()
            self._img_tick_elapsed()
            self._img_refresh_comfyui(start_if_needed=True, kill_orphans=needs_restart)
            return
        self.comfyui_ok = True
        self._vision_comfyui_deferred_restart = False
        self._img_waiting_for_comfyui_generation = False

        # ── Bump generation ID so stale callbacks from old threads are ignored
        self._img_gen_id += 1
        gen_id = self._img_gen_id

        # Store generation params for logging, save filename, and display
        self._img_last_params = {
            "model": model_filename,
            "width": width, "height": height,
            "steps": steps, "cfg": cfg, "seed": seed,
            "sampler": sampler, "scheduler": scheduler,
            "prompt": positive,
            "negative_prompt": negative,
            "reference_image": Path(img2img_ref_path).name if img2img_ref_path else "",
            "denoise": img2img_denoise if img2img_ref_path else None,
        }
        # Derive a short SKU label from the detected optional SKU.
        sku_name = self._optional_sku.get("name", "") if self._optional_sku else ""
        self._img_last_params["sku"] = sku_name

        # Log generation start with settings (not the prompt)
        logger.info(
            f"Image generation started: model={model_filename}, "
            f"{width}x{height}, steps={steps}, cfg={cfg}, "
            f"sampler={sampler}/{scheduler}, seed={seed}"
            + (f", img2img_denoise={img2img_denoise:.2f}" if img2img_ref_path else "")
            + (f", sku={sku_name}" if sku_name else ""),
            category=logger.CATEGORY_IMAGE_GEN,
        )

        self._img_stop_event.clear()
        self._img_bytes = None
        self._set_image_generate_button_running(True)
        self._img_stop_btn.configure(state="normal")
        self._img_save_btn.configure(state="disabled")
        # Clear image from label BEFORE releasing the photo reference —
        # otherwise GC destroys the pyimage while the label still references it,
        # causing "_tkinter.TclError: image "pyimage…" doesn't exist" on redraw.
        est_str, est_risky = self._timing_estimate(model_filename, steps)
        display_lines = f"Preparing: {model_display}\n{width}x{height}, {steps} steps"
        if est_str:
            display_lines += f"\n\n{est_str}"
        self._img_safe_clear_display(display_lines, WARN_TEXT if est_risky else TEXT_MUTED)
        self._img_set_status("Preparing …", color=INFO_TEXT)

        # Show progress bar in indeterminate mode until first step event arrives
        self._img_show_progress(mode="indeterminate", color=INFO_TEXT)
        if self._img_elapsed_timer_id:
            self.after_cancel(self._img_elapsed_timer_id)
            self._img_elapsed_timer_id = None
        self._img_elapsed_start = time.time()
        self._img_step_times: list = []   # reset per-generation timing
        self._img_tick_elapsed()

        def _run():
            try:
                # Unload any selectable vision model that might still be in VRAM —
                # check all picker candidates plus the legacy llama vision tag so we
                # release VRAM regardless of which one the user last analyzed with.
                _vision_tags = {self._get_selected_vision_tag(), self.LEGACY_VISION_MODEL}
                _vision_tags.update(
                    e.get("ollama_tag", "")
                    for e in self._vision_picker_entries()
                    if e.get("ollama_tag")
                )
                _vision_tags.discard("")
                _running = {m.get("name", "") for m in self.ollama.running_models()}
                for _vtag in list(_vision_tags):
                    if _ollama_tag_is_local(_vtag, _running):
                        self.after(0, lambda t=_vtag: self._img_update_progress(
                            f"Freeing VRAM — unloading {t} …", gen_id))
                        logger.info(f"Image generation: freeing VRAM — unloading {_vtag}", category=logger.CATEGORY_IMAGE_GEN)
                        try:
                            unload_ok = self.ollama.unload_model(_vtag)
                        except Exception:
                            unload_ok = False
                        if unload_ok:
                            logger.info(f"Image generation: {_vtag} unloaded", category=logger.CATEGORY_IMAGE_GEN)

                _last_logged_step = [-1]  # last step number logged
                _last_logged_s    = [0]   # fallback: last elapsed-second logged
                _step_times: list = []    # (step_num, elapsed_s) for ETA + telemetry

                def _progress(msg: str):
                    import re as _re
                    display_msg = msg
                    step_m = _re.search(r"step (\d+)/(\d+) \((\d+)%\) — (\d+)s", msg)
                    if step_m:
                        step, total, pct, elapsed_s = (int(step_m.group(i)) for i in (1, 2, 3, 4))
                        _step_times.append((step, elapsed_s))
                        # After step 2 we have a pure-inference delta — compute ETA
                        if len(_step_times) >= 2:
                            _deltas = [_step_times[i][1] - _step_times[i-1][1]
                                       for i in range(1, len(_step_times))]
                            _avg = sum(_deltas) / len(_deltas)
                            eta_s = int(_avg * (total - step))
                            projected_s = elapsed_s + eta_s
                            eta_part = f"  ETA ~{_fmt_duration(eta_s)}"
                            if projected_s > _GEN_TIMEOUT_S:
                                eta_part += " ⚠"
                            display_msg = msg + eta_part
                        if step > _last_logged_step[0]:
                            _last_logged_step[0] = step
                            logger.info(
                                f"Image generation: step {step}/{total} ({pct}%) — {elapsed_s}s",
                                category=logger.CATEGORY_IMAGE_GEN,
                            )
                    else:
                        # No step data — log once every 10 s so the log shows it's alive
                        elapsed_m = _re.search(r"(\d+)s", msg)
                        if elapsed_m:
                            elapsed_s = int(elapsed_m.group(1))
                            if elapsed_s - _last_logged_s[0] >= 10:
                                logger.info(f"Image generation: {msg}", category=logger.CATEGORY_IMAGE_GEN)
                                _last_logged_s[0] = elapsed_s
                    self.after(0, lambda m=display_msg, gid=gen_id: self._img_update_progress(m, gid))

                img_bytes = self.comfyui.generate_image(
                    model_filename=model_filename,
                    positive_prompt=positive,
                    negative_prompt=negative,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg,
                    seed=seed,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    reference_image_path=img2img_ref_path,
                    denoise=img2img_denoise,
                    progress_cb=_progress,
                    stop_event=self._img_stop_event,
                )
                # Save timing telemetry for future pre-flight estimates.
                # load_s = elapsed at step 1 minus one pure-inference step.
                # step_s = average of step-to-step deltas (step 2 onward).
                if len(_step_times) >= 2:
                    _deltas = [_step_times[i][1] - _step_times[i-1][1]
                               for i in range(1, len(_step_times))]
                    _avg_step = sum(_deltas) / len(_deltas)
                    _load = max(1.0, _step_times[0][1] - _avg_step)
                    self.after(0, lambda ls=_load, ss=_avg_step, gid=gen_id: (
                        self._timing_save_entry(model_filename, ls, ss) if gid == self._img_gen_id else None
                    ))
                self.after(0, lambda b=img_bytes, gid=gen_id: (
                    self._img_generation_done(b) if gid == self._img_gen_id else None
                ))
            except ComfyUIError as e:
                err = str(e)
                self.after(0, lambda m=err, gid=gen_id: (
                    self._img_generation_failed(m) if gid == self._img_gen_id else None
                ))
            except Exception as e:
                err = str(e)
                self.after(0, lambda m=err, gid=gen_id: (
                    self._img_generation_failed(m) if gid == self._img_gen_id else None
                ))

        self._img_thread = threading.Thread(target=_run, daemon=True)
        self._img_thread.start()

    def _img_safe_clear_display(self, text: str, text_color=TEXT_MUTED):
        """Clear the display label's image without triggering pyimage GC crash.

        The key insight: customtkinter's CTkLabel._draw() accesses the
        underlying tkinter PhotoImage when reconfiguring colours/text.
        If the CTkImage (self._img_photo) has already been GC'd, the
        PhotoImage it created ("pyimage1") no longer exists and _draw()
        crashes.  We must tell the label to drop its image reference
        FIRST, while the photo object is still alive.
        """
        try:
            self._img_display.configure(image=None, text=text, text_color=text_color)
        except Exception:
            # If the pyimage is already gone, destroy and recreate the label
            try:
                parent = self._img_display.master
                self._img_display.destroy()
                self._img_display = ctk.CTkLabel(
                    parent, text=text,
                    font=ctk.CTkFont(size=13), text_color=text_color,
                )
                self._img_display.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
            except Exception as e:
                logger.warning(f"Failed to recreate image display: {e}")
        self._img_photo = None  # NOW safe to release

    # ── Timing telemetry ──────────────────────────────────────────────────────

    def _timing_load(self) -> dict:
        """Load timing telemetry from disk. Returns empty dict on any error."""
        try:
            if _TIMING_FILE.exists():
                return json.loads(_TIMING_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {}

    def _timing_profile_key(self) -> str:
        sku = self._optional_sku or {}
        ram_gb = int(round(float(sku.get("ram_gb") or 0)))
        if not ram_gb:
            try:
                ram_gb = int(round(system_info.get_ram_info().get("total_mb", 0) / 1024))
            except Exception:
                ram_gb = 0
        if getattr(self, "_comfyui_force_cpu", False):
            mode = "cpu"
            vram_gb = 0
        else:
            mode = "gpu"
            vram_gb = int(round(float(self._active_device_vram_gb() or 0)))
        gpu_type = str(getattr(getattr(self, "gpu_info", None), "gpu_type", "") or "unknown")
        gpu = getattr(self, "gpu_info", None)
        gpu_name = str(
            getattr(gpu, "device_name", None)
            or getattr(gpu, "name", None)
            or "unknown"
        )
        gpu_name = re.sub(r"[^A-Za-z0-9._-]+", "_", gpu_name).strip("_") or "unknown"
        unified = "1" if (
            getattr(getattr(self, "gpu_info", None), "unified_memory", False)
            or sku.get("unified_memory")
        ) else "0"
        return f"{mode}|ram={ram_gb}|vram={vram_gb}|gpu={gpu_type}|name={gpu_name}|unified={unified}"

    def _timing_save_entry(self, model_key: str, load_s: float, step_s: float):
        """Append timing for the current hardware profile (keeps last N per model)."""
        data = self._timing_load()
        profile_key = self._timing_profile_key()
        profiles = data.setdefault("profiles", {})
        profile = profiles.setdefault(profile_key, {})
        entries = profile.get(model_key, [])
        entries.append({
            "load_s": round(load_s, 1),
            "step_s": round(step_s, 2),
            "profile": profile_key,
        })
        profile[model_key] = entries[-_TIMING_MAX_SAMPLES:]
        data["version"] = 2
        try:
            _TIMING_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            logger.warning(f"Could not save timing telemetry: {e}")

    def _timing_estimate(self, model_key: str, steps: int) -> tuple[str, bool]:
        """Return (estimate_string, is_risky) from historical telemetry.

        Returns ("", False) when no telemetry exists for this exact hardware
        profile and model — cross-system timings are intentionally ignored.
        is_risky is True when projected time exceeds the generation timeout.
        """
        data = self._timing_load()
        profile_key = self._timing_profile_key()
        entries = data.get("profiles", {}).get(profile_key, {}).get(model_key, [])
        if not entries:
            return "", False
        avg_load = sum(e["load_s"] for e in entries) / len(entries)
        avg_step = sum(e["step_s"] for e in entries) / len(entries)
        total_s  = int(avg_load + avg_step * steps)
        is_risky = total_s > _GEN_TIMEOUT_S and not getattr(self, "_comfyui_force_cpu", False)
        risk_note = "  ⚠ may time out" if is_risky else ""
        return (
            f"Est. ~{_fmt_duration(total_s)}"
            f"  (~{_fmt_duration(int(avg_load))} load"
            f" + {avg_step:.1f}s/step × {steps}){risk_note}"
        ), is_risky

    def _img_set_status(self, msg: str, color=TEXT_MUTED):
        self._img_set_canvas_status(msg, color)
        self.set_status(msg)

    def _img_set_canvas_status(self, msg: str, color=TEXT_MUTED):
        if hasattr(self, "_img_status_lbl") and self._img_status_lbl is not None:
            self._img_status_lbl.configure(text=msg, text_color=color)

    def _img_show_progress(self, *, mode: str = "indeterminate", color=INFO_TEXT, start: bool = True) -> None:
        if not hasattr(self, "_img_progress_row") or self._img_progress_row is None:
            return
        self._img_progress_row.grid()
        self._img_progress_bar.configure(mode=mode, progress_color=color)
        self._img_progress_bar.grid()
        self._img_elapsed_lbl.grid()
        if mode == "determinate":
            self._img_progress_bar.stop()
            self._img_progress_bar.set(0)
        elif start:
            self._img_progress_bar.start()

    def _img_tick_elapsed(self):
        """Update the elapsed-time label every second while generating."""
        if self._img_elapsed_start <= 0:
            return
        elapsed = int(time.time() - self._img_elapsed_start)
        mins, secs = divmod(elapsed, 60)
        txt = f"{mins}:{secs:02d}" if mins else f"{secs}s"
        self._img_elapsed_lbl.configure(text=txt)
        self._img_elapsed_timer_id = self.after(1000, self._img_tick_elapsed)

    def _img_stop_progress(self):
        """Stop the progress bar and elapsed timer."""
        self._img_elapsed_start = 0.0
        if self._img_elapsed_timer_id:
            self.after_cancel(self._img_elapsed_timer_id)
            self._img_elapsed_timer_id = None
        self._img_progress_bar.stop()
        self._img_progress_bar.grid_remove()
        self._img_elapsed_lbl.configure(text="")
        self._img_elapsed_lbl.grid_remove()
        self._img_progress_row.grid_remove()

    def _img_update_progress(self, msg: str, gen_id: int):
        """Update status and display area during generation (called from progress_cb)."""
        if gen_id != self._img_gen_id:
            return
        color = WARN_TEXT if "⚠" in msg else INFO_TEXT
        self._img_set_status(msg, color=color)

        # Switch progress bar from indeterminate spinner to determinate fill when step data arrives
        import re as _re
        step_m = _re.search(r"step (\d+)/(\d+)", msg)
        if step_m:
            self._img_progress_row.grid()
            self._img_progress_bar.grid()
            self._img_elapsed_lbl.grid()
            step, total = int(step_m.group(1)), int(step_m.group(2))
            frac = step / total if total > 0 else 0
            if self._img_progress_bar.cget("mode") != "determinate":
                self._img_progress_bar.stop()
                self._img_progress_bar.configure(mode="determinate")
            self._img_progress_bar.set(frac)

        # Update the big display area so the user sees activity
        elapsed = int(time.time() - self._img_elapsed_start) if self._img_elapsed_start > 0 else 0
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"
        p = getattr(self, '_img_last_params', {})
        model_name = p.get("model", "")
        settings = f"{p.get('width', '')}x{p.get('height', '')}, {p.get('steps', '')} steps"
        try:
            self._img_display.configure(
                text=f"Model: {model_name}\n{settings}\n\n{msg}\n\nElapsed: {time_str}",
                text_color=TEXT_MUTED,
            )
        except Exception:
            pass  # label may be mid-transition; non-critical

    def _img_generation_done(self, img_bytes: bytes):
        elapsed = int(time.time() - self._img_elapsed_start) if self._img_elapsed_start > 0 else 0
        self._img_stop_progress()
        self._img_waiting_for_comfyui_generation = False
        self._img_bytes = img_bytes
        self._set_image_gen_enabled(self.comfyui_ok)
        self._img_stop_btn.configure(state="disabled")
        self._img_save_btn.configure(state="normal")
        # imagegen-prompt-collapse: free vertical room for the freshly rendered
        # image when the window isn't maximized. No-op when maximized (there's
        # already room for both prompt and image) and no-op if already collapsed.
        self._img_autocollapse_prompt_if_unmaximized()
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"
        p = getattr(self, '_img_last_params', {})
        model_short = p.get("model", "unknown")
        self._img_set_status(
            f"Done — {model_short} in {time_str}.", color=SUCCESS_TEXT,
        )
        logger.info(
            f"Image generation complete: model={model_short}, "
            f"{p.get('width', '?')}x{p.get('height', '?')}, "
            f"steps={p.get('steps', '?')}, cfg={p.get('cfg', '?')}, "
            f"elapsed={time_str}",
            category=logger.CATEGORY_IMAGE_GEN,
        )

        # v5.1: write a compact info strip under the canvas so the user can see
        # the model, size, seed, and render time at a glance without scrolling
        # the status pill.
        if hasattr(self, "_img_info_lbl") and self._img_info_lbl is not None:
            try:
                seed_val = p.get("seed", -1)
                seed_disp = "Random" if seed_val == -1 else str(seed_val)
                info_parts = [
                    str(model_short),
                    f"{p.get('width', '?')}×{p.get('height', '?')}",
                    f"seed {seed_disp}",
                    f"{time_str} render",
                ]
                self._img_info_lbl.configure(text="  ·  ".join(info_parts))
            except Exception:
                pass

        if PIL_AVAILABLE:
            try:
                import io
                pil_img = _PIL_Image.open(io.BytesIO(img_bytes))
                pil_img.thumbnail((700, 500), _PIL_Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                       size=pil_img.size)
                # Safe-clear the label first to detach any stale pyimage,
                # then attach the new image on a clean label.
                self._img_safe_clear_display("", TEXT_MUTED)
                self._img_photo = ctk_img
                try:
                    self._img_display.configure(image=ctk_img, text="")
                except Exception:
                    # Label is still corrupt — recreate it
                    logger.warning("Recreating image display label after configure failure")
                    parent = self._img_display.master
                    self._img_display.destroy()
                    self._img_display = ctk.CTkLabel(
                        parent, text="", image=ctk_img,
                        font=ctk.CTkFont(size=13),
                    )
                    self._img_display.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
                return
            except Exception as e:
                logger.warning(f"Image display failed: {e}")

        # No PIL or display failed — use safe clear to avoid pyimage crash
        self._img_safe_clear_display(
            "Image generated successfully.\nUse 'Save Image' to view.",
            SUCCESS_TEXT,
        )

    def _img_generation_failed(self, error: str):
        self._img_stop_progress()
        self._img_waiting_for_comfyui_generation = False
        self._set_image_gen_enabled(self.comfyui_ok)
        self._img_stop_btn.configure(state="disabled")
        short_err = (error[:120] + " …") if len(error) > 120 else error
        self._img_set_status(f"FAILED: {short_err}", color=ERROR_TEXT)
        self._img_safe_clear_display(
            f"Generation failed:\n\n{error}", ERROR_TEXT,
        )
        logger.error(f"Image generation failed: {error}", category=logger.CATEGORY_IMAGE_GEN)

        # Detect CUDA / vGPU errors and suggest fixes
        err_lower = error.lower()
        # v5.5.12 (Ron, 2026-05-27): DXGI TDR on Windows-iGPU SKUs (Intel Arc
        # Graphics, AMD Radeon Graphics, Snapdragon Adreno). The GPU device
        # gets killed mid-generation when a single kernel exceeds Windows'
        # ~2s per-kernel timeout — independent from memory pressure. Heavy
        # SDXL/Flux UNets routinely take 6-15s per step on iGPUs and trigger
        # this even with plenty of shared GPU memory free. Dispatch BEFORE
        # the vbar/CUDA branches because the device-suspended string is
        # distinctive enough to be safe at the top, and Ron's failure case
        # is the most likely path users will hit here.
        if (
            "device instance has been suspended" in err_lower
            or "getdeviceremovedreason" in err_lower
            or "device removed" in err_lower
        ):
            messagebox.showerror(
                "Image generation timed out (Windows DXGI TDR)",
                "Image generation was interrupted because your GPU couldn't "
                "complete a single step within Windows' ~2-second per-kernel "
                "timeout (DXGI Timeout Detection and Recovery).\n\n"
                f"GPU: {self.gpu_info.device_name}\n\n"
                "This is not a memory issue — it's a per-kernel duration limit\n"
                "Windows enforces on integrated GPUs. Try in order:\n\n"
                "  1. Lower steps to 5–10\n"
                "  2. Use 512×512 resolution\n"
                "  3. Switch to a smaller model:\n"
                "       • Realistic Vision V6 (SD 1.5 family)\n"
                "       • SDXL Lightning (4-step distillation)\n\n"
                "If you want to probe what your iGPU can handle, the\n"
                "Benchmark page has a \"Force All\" toggle that bypasses\n"
                "the safety filter applied here.",
                parent=self,
            )
        elif "vbar allocation failed" in err_lower or (
            "operation not supported" in err_lower and self.gpu_info.is_vgpu
        ):
            messagebox.showerror(
                "vGPU Compatibility Error",
                "Image generation failed due to a CUDA limitation on\n"
                "virtual GPUs (vGPU).\n\n"
                f"GPU: {self.gpu_info.device_name}\n\n"
                "LocalAI launches ComfyUI with the required vGPU/MIG\n"
                "triple (--disable-dynamic-vram, --disable-cuda-malloc,\n"
                "--disable-async-offload). If the error persists after a\n"
                "fresh restart, you can also add extra ComfyUI flags via\n"
                "the LOCALAI_COMFYUI_EXTRA_FLAGS environment variable\n"
                "(e.g. --lowvram, --reserve-vram 1.5, --cache-none) and\n"
                "relaunch LocalAI.\n\n"
                "Please close and restart LocalAI to apply the fix.\n"
                "If the error persists after restart, report it as a bug.",
                parent=self,
            )
        elif "operation not supported" in err_lower or "cuda error" in err_lower:
            if sys.platform == "win32":
                steps = (
                    "1. Close and restart LocalAI (applies vGPU workarounds)\n"
                    "2. Run fix_nvidia_pytorch.bat to upgrade PyTorch to CUDA 12.8\n"
                    "3. Set LOCALAI_COMFYUI_EXTRA_FLAGS=--disable-cuda-malloc and\n"
                    "   relaunch — early-generation NVIDIA driver regressions\n"
                    "   sometimes need this even on bare metal.\n\n"
                )
            else:
                steps = (
                    "1. Close and restart LocalAI (applies vGPU workarounds)\n"
                    "2. Reinstall PyTorch: pip3 install torch torchvision torchaudio\n\n"
                )
            messagebox.showerror(
                "CUDA Error — PyTorch Upgrade May Help",
                "Image generation failed with a CUDA error.\n\n"
                "Try these fixes in order:\n"
                f"{steps}"
                f"GPU: {self.gpu_info.device_name}",
                parent=self,
            )

    def _stop_image_generation(self):
        self._img_stop_event.set()
        self._img_stop_btn.configure(state="disabled")
        self._img_set_status("Stopping …", color=WARN_TEXT)

    def _save_generated_image(self):
        if not self._img_bytes:
            return
        from tkinter import filedialog
        import re as _re

        # Build a descriptive default filename:
        # e.g. "DeviceLabel_dreamshaper_v8_512x512_20steps_cfg7.0.png"
        p = getattr(self, '_img_last_params', {})
        sku = p.get("sku", "").replace(" ", "_") or "LocalAI"
        model_base = Path(p.get("model", "image")).stem  # strip .safetensors etc
        model_base = _re.sub(r'[^\w\-.]', '_', model_base)  # filesystem-safe
        w, h = p.get("width", ""), p.get("height", "")
        steps = p.get("steps", "")
        cfg = p.get("cfg", "")
        default_name = f"{sku}_{model_base}_{w}x{h}_{steps}steps_cfg{cfg}.png"

        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Save generated image",
        )
        if path:
            try:
                # v5: Embed PNG metadata (iTXt chunks) describing the generation
                # parameters so the file is self-documenting. Falls back to a
                # raw byte write if Pillow can't decode the buffer for any
                # reason. Also drops a sidecar .json with the same data for
                # tools that don't read PNG metadata.
                try:
                    from PIL import Image as _PILImage, PngImagePlugin as _PngInfo
                    import io as _io
                    import json as _json
                    pil_img = _PILImage.open(_io.BytesIO(self._img_bytes))
                    info = _PngInfo.PngInfo()
                    metadata = {
                        "app": f"LocalAI Studio {APP_VERSION}",
                        "model": str(p.get("model", "")),
                        "width": str(p.get("width", "")),
                        "height": str(p.get("height", "")),
                        "steps": str(p.get("steps", "")),
                        "cfg": str(p.get("cfg", "")),
                        "seed": str(p.get("seed", "")),
                        "sampler": str(p.get("sampler", "")),
                        "scheduler": str(p.get("scheduler", "")),
                        "prompt": str(p.get("prompt", "")),
                        "negative_prompt": str(p.get("negative_prompt", "")),
                        "sku": str(p.get("sku", "")),
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    for k, v in metadata.items():
                        info.add_itxt(f"LocalAI:{k}", v, zip=True)
                    # Also drop a Stable-Diffusion-compatible "parameters" string
                    info.add_text("parameters", (
                        f"{metadata['prompt']}\n"
                        f"Negative prompt: {metadata['negative_prompt']}\n"
                        f"Steps: {metadata['steps']}, Sampler: {metadata['sampler']}, "
                        f"CFG scale: {metadata['cfg']}, Seed: {metadata['seed']}, "
                        f"Size: {metadata['width']}x{metadata['height']}, "
                        f"Model: {metadata['model']}"
                    ))
                    pil_img.save(path, format="PNG", pnginfo=info, optimize=True)
                    # Sidecar JSON for easy programmatic access
                    sidecar = Path(path).with_suffix(".json")
                    sidecar.write_text(
                        _json.dumps(metadata, indent=2), encoding="utf-8",
                    )
                except Exception as meta_exc:
                    logger.debug(f"PNG metadata embed skipped: {meta_exc}")
                    with open(path, "wb") as f:
                        f.write(self._img_bytes)
                self.set_status(f"Image saved: {path}")
                logger.info(f"Image saved: {Path(path).name}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e), parent=self)

    # ── Download ──────────────────────────────────────────────────────────────

    def start_download(self, model: dict) -> bool:
        # v5.4: Centralized install gate for user-added entries. This must
        # fire here (not just from _install_selected_model) because chat
        # primary-action path and row-level Download button both reach this
        # function without consulting the gate. Keeps the message consistent.
        if model.get("user_added") and not self._user_added_model_can_install(model):
            messagebox.showinfo(
                "Can't download automatically",
                f"'{model.get('name') or model.get('id')}' was added from "
                "Hugging Face for reference, but LocalAI doesn't have a "
                "one-click downloader for its backend. To install it, use "
                "the source link in the detail pane to grab the files "
                "manually, or look for an Ollama / ONNX build of the same "
                "model.",
                parent=self,
            )
            return False
        if not self.ollama_ok:
            messagebox.showerror(
                "Ollama not running",
                "Ollama must be running to download models.\n"
                "Download it from https://ollama.com and try again.",
                parent=self,
            )
            return False

        # Storage check (enhanced in Low Resources Mode)
        models_dir_path = config.models_dir(self.cfg)
        if self.cfg.get("low_resources_mode"):
            ok, reason = resource_manager.check_disk_for_download(
                model["size_gb"], models_dir_path,
            )
            if not ok:
                messagebox.showerror("Not enough disk space", reason, parent=self)
                return False
        else:
            st = system_info.get_storage_info(models_dir_path)
            needed_gb = model["size_gb"] * 1.1  # 10 % buffer
            if st["free_gb"] < needed_gb:
                messagebox.showerror(
                    "Not enough disk space",
                    f"Downloading '{model['name']}' needs ~{needed_gb:.1f} GB.\n"
                    f"Only {st['free_gb']:.1f} GB free on disk.\n"
                    "Free up space and try again.",
                    parent=self,
                )
                return False

        # Extra disk-check for size-capped user profiles.
        # ``system_info.get_storage_info`` above measures the drive that
        # holds ``models_dir_path`` (the LocalAI working dir), but Ollama
        # actually writes blobs to ``$OLLAMA_MODELS`` or
        # ``~/.ollama/models``, which on some managed/roamed Windows
        # profiles lives inside a profile container with its own hard
        # size cap. This extra check measures the *actual* Ollama models
        # drive and only blocks when that profile drive is the
        # bottleneck, so it never fires for healthy machines
        # (``precheck_ollama_pull`` returns None when the volume has room).
        ollama_skip = constrained_env.precheck_ollama_pull(model.get("size_gb", 0))
        if ollama_skip:
            messagebox.showerror(
                "Ollama models drive full",
                f"Downloading '{model['name']}' {ollama_skip}",
                parent=self,
            )
            return False

        # Hardware check (show warning, not block)
        gpus = system_info.get_gpu_info()
        gpu_idx = 0 if gpus else None
        ok, reason = system_info.can_run_model(model, gpu_index=gpu_idx)
        if not ok:
            # On unified memory (Apple Silicon), CPU and GPU share the same RAM —
            # "run it on CPU instead" is not a valid fallback.
            is_unified = gpus and gpus[0].get("unified_memory", False)
            if is_unified:
                hint = "This Mac uses unified memory (CPU and GPU share the same RAM pool)."
            else:
                hint = "You can still run it on CPU."
            proceed = messagebox.askyesno(
                "Resource warning",
                f"{reason}\n\n{hint}\n\nDownload anyway?",
                parent=self,
            )
            if not proceed:
                return False

        if self._download_thread and self._download_thread.is_alive():
            messagebox.showinfo("Download in progress", "Wait for the current download to finish.", parent=self)
            return False

        self._stop_event.clear()
        self._show_progress(True)
        self.set_status(f"Downloading {model['name']} …")
        self._active_download_tag = model["ollama_tag"]
        self.__dict__.setdefault("_progress_log_state", {}).pop(f"download:{model['ollama_tag']}", None)
        logger.info(f"Starting download: {model['ollama_tag']}", category=logger.CATEGORY_MODEL_PULL)

        def _do_download():
            try:
                self.ollama.pull_model(
                    model["ollama_tag"],
                    progress_cb=self._download_progress_cb,
                    stop_event=self._stop_event,
                )
                if model["ollama_tag"] not in self.cfg["downloaded_this_session"]:
                    self.cfg["downloaded_this_session"].append(model["ollama_tag"])
                if not config.save(self.cfg):
                    logger.error("Could not persist downloaded_this_session after model pull")
                self.after(0, lambda: self._download_done(model, success=True))
            except OllamaError as e:
                # Surface a disk-pressure hint in the failure message so
                # users on size-capped Windows profiles see the
                # set_ollama_models_dir.bat relocation workaround
                # (writes to <app_path>\Ollama by default) instead of
                # just the bare daemon error.
                # ``profile_aware_ollama_error`` is a noop on healthy
                # machines so this never regresses the local happy path.
                friendly = constrained_env.profile_aware_ollama_error(str(e))
                self.after(0, lambda err=friendly: self._download_done(model, success=False, error=err))

        self._download_thread = threading.Thread(target=_do_download, daemon=True)
        self._download_thread.start()
        return True

    def _download_progress_cb(self, status: str, completed: int, total: int):
        if total > 0:
            pct = completed / total
            pct_str = f"{pct * 100:.0f}%  ({_fmt_bytes(completed)} / {_fmt_bytes(total)})"
        else:
            pct = 0
            pct_str = status
        msg = f"Downloading: {status}  {pct_str}"
        tag = getattr(self, "_active_download_tag", "model")
        self._log_progress_milestone(
            f"download:{tag}",
            f"{tag}: {status}  {pct_str}",
            category=logger.CATEGORY_MODEL_PULL,
            completed=completed,
            total=total,
            status=status,
            percent_step=5,
            min_seconds=10.0,
        )
        self.after(0, lambda m=msg, p=pct: (
            self.set_status(m),
            self._progress_bar.set(min(p, 1.0)),
        ))

    def _download_done(self, model: dict, success: bool, error: str = ""):
        self._show_progress(False)
        if success:
            self.set_status(f"'{model['name']}' downloaded successfully.")
            logger.info(f"Download complete: {model['ollama_tag']}", category=logger.CATEGORY_MODEL_PULL)
            self._refresh_model_cards()
            self._refresh_chat_model_selector()
            pending = getattr(self, "_pending_chat_load_after_download", None)
            if pending and pending.get("model_id") == model.get("id"):
                self._pending_chat_load_after_download = None
                backend = pending.get("backend")
                if backend:
                    self._backend_var.set(backend)
                self.load_model_for_chat(model)
                return
            # If the downloaded model is any of the selectable vision models or the
            # legacy llama vision tag, refresh the vision picker + analyze CTAs.
            try:
                vision_tags = {self.LEGACY_VISION_MODEL}
                vision_tags.update(
                    e.get("ollama_tag", "")
                    for e in (self._vision_picker_entries() if hasattr(self, "_vision_picker_entries") else [])
                    if e.get("ollama_tag")
                )
                if model.get("ollama_tag") in vision_tags:
                    try:
                        self._refresh_vision_picker_ui()
                    except Exception:
                        pass
                    try:
                        self._refresh_vision_model_ui()
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            self.set_status(f"Download failed: {error}")
            logger.error(f"Download failed for {model['ollama_tag']}: {error}", category=logger.CATEGORY_MODEL_PULL)
            pending = getattr(self, "_pending_chat_load_after_download", None)
            if pending and pending.get("model_id") == model.get("id"):
                self._pending_chat_load_after_download = None
            # v2026.06.01.8: even on failure, the vision picker needs to
            # revert from "Cancel download" back to "Download" if the
            # failed model is a vision-picker entry — otherwise the
            # Stop button would stay visible after the failure dialog.
            try:
                vision_tags_fail = {self.LEGACY_VISION_MODEL}
                vision_tags_fail.update(
                    e.get("ollama_tag", "")
                    for e in (self._vision_picker_entries() if hasattr(self, "_vision_picker_entries") else [])
                    if e.get("ollama_tag")
                )
                if model.get("ollama_tag") in vision_tags_fail:
                    try:
                        self._refresh_vision_model_ui()
                    except Exception:
                        pass
            except Exception:
                pass
            messagebox.showerror("Download failed", error, parent=self)

    def _show_progress(self, show: bool):
        if show:
            self._progress_bar.grid()
            self._progress_bar.set(0)
        else:
            self._progress_bar.grid_remove()

    # ── Load model for chat ───────────────────────────────────────────────────

    def load_model_for_chat(self, model: dict):
        backend = self._backend_var.get()

        # Resource check
        gpus = system_info.get_gpu_info()
        gpu_idx = 0 if gpus and "Ollama" in backend else None
        ok, reason = system_info.can_run_model(model, gpu_index=gpu_idx)
        if not ok:
            proceed = messagebox.askyesno(
                "Resource warning",
                f"{reason}\n\nTry loading anyway?",
                parent=self,
            )
            if not proceed:
                return

        if "OpenVINO (GPU/NPU)" in backend:
            self._load_ov_model(model)
        elif "ONNX" in backend:
            self._load_onnx_model(model)
        else:
            force_cpu = "CPU only" in backend
            prev_model = self.active_model.get("name") if self.active_model else None
            self.active_model = model
            self.active_session = None
            self.active_model["_num_gpu"] = 0 if force_cpu else -1
            self._active_model_display = f"{model['name']}  [{backend}]"
            self._switch_page("chat")
            self._set_chat_selector_to_model(model)
            self._active_model_label.configure(text=self._active_model_display)
            self._update_chat_readiness()
            self.chat_history.clear()
            self._update_chat_copy_buttons()
            if prev_model and prev_model != model["name"]:
                self._append_chat("model_switch", f"{model['name']}  ({backend})")
            self._append_chat("system", f"Model loaded: {model['name']}  ({backend})\n"
                              f"Type your message and press Enter to chat.\n")
            self._apply_chat_demo_prompt(model)
            logger.info(f"Chat model set to: {model['ollama_tag']} (backend={backend})")

    def _load_ov_model(self, model: dict):
        if not OV_GENAI_AVAILABLE:
            messagebox.showerror(
                "OpenVINO GenAI not installed",
                "Install it with:\n  pip install openvino openvino-genai\n\n"
                "Or use the GPU/CPU (Ollama) backend instead.",
                parent=self,
            )
            return
        if not model.get("ov_repo"):
            messagebox.showerror(
                "No OpenVINO model available",
                f"'{model['name']}' does not have a pre-built OpenVINO model.\n"
                "Use the GPU/CPU (Ollama) or ONNX backend.",
                parent=self,
            )
            return

        preferred_device = pick_ov_device("GPU")
        model_dir = config.models_dir(self.cfg) / "ov" / model["id"].replace(":", "_")

        self._switch_page("chat")
        self._append_chat("system", f"Loading OpenVINO model {model['name']} …\n")
        self.set_status("Loading OpenVINO model …")
        logger.info(f"Loading OV: {model['ov_repo']} → {model_dir}")

        def _load():
            try:
                if not model_dir.exists() or not any(
                    f.suffix in (".xml", ".bin") for f in model_dir.iterdir()
                ):
                    if not HF_AVAILABLE:
                        raise OVError('huggingface_hub not installed.\nRun: pip install "huggingface-hub>=0.34.0,<1.0"')
                    self.after(0, lambda: self._append_chat(
                        "system", "Downloading OpenVINO model from HuggingFace …\n"
                    ))
                    download_ov_model(
                        model["ov_repo"],
                        model_dir,
                        progress_cb=lambda m: self.after(0, lambda msg=m: self.set_status(msg)),
                        stop_event=self._stop_event,
                    )

                self.after(0, lambda: self._append_chat(
                    "system", f"Compiling model for {preferred_device} …\n"
                ))
                session = OVModelSession(model_dir, preferred_device)
                actual_device = session._device
                provider_label = f"OpenVINO/{actual_device}"
                if actual_device != preferred_device:
                    self.after(0, lambda: self._append_chat(
                        "system",
                        f"Note: {preferred_device} unavailable for this model, using {actual_device}.\n"
                    ))
                ram_after = system_info.get_ram_info()
                self.after(0, lambda: self._onnx_load_done(model, session, ram_after, provider_label))
            except (OVError, Exception) as e:
                err = str(e)
                self.after(0, lambda msg=err: (
                    self._append_chat("error", f"Failed to load OpenVINO model: {msg}\n"),
                    self.set_status("OpenVINO load failed."),
                    logger.error(f"OV load failed: {msg}"),
                ))

        threading.Thread(target=_load, daemon=True).start()

    def _load_onnx_model(self, model: dict):
        if not ONNX_AVAILABLE:
            messagebox.showerror(
                "ONNX Runtime not installed",
                "Install it with:\n  pip install onnxruntime-directml\n\n"
                "Or use the GPU/CPU (Ollama) backend instead.",
                parent=self,
            )
            return
        if not model.get("onnx_repo"):
            messagebox.showerror(
                "No ONNX version available",
                f"'{model['name']}' does not have an ONNX model.\n"
                "Use the GPU/CPU (Ollama) backend.",
                parent=self,
            )
            return

        if OPENVINO_AVAILABLE:
            provider, provider_options = "OpenVINOExecutionProvider", {"device_type": "NPU"}
            provider_label = "OpenVINO/NPU"
            onnx_subfolder = model.get("onnx_openvino_subfolder") or model.get("onnx_subfolder", "")
        elif DIRECTML_AVAILABLE:
            provider, provider_options = "DmlExecutionProvider", {}
            provider_label = "DirectML/NPU"
            onnx_subfolder = model.get("onnx_subfolder", "")
        else:
            provider, provider_options = "CPUExecutionProvider", {}
            provider_label = "CPU"
            onnx_subfolder = (
                model.get("onnx_cpu_subfolder")
                or model.get("onnx_openvino_subfolder")
                or model.get("onnx_subfolder", "")
            )
        model_dir = config.models_dir(self.cfg) / "onnx" / model["id"].replace(":", "_")

        self._switch_page("chat")
        self._append_chat("system", f"Loading ONNX model {model['name']} …\n")
        self.set_status(f"Loading ONNX model …")
        logger.info(f"Loading ONNX: {model['onnx_repo']} / {onnx_subfolder}")

        def _load():
            try:
                def ensure_subfolder(subfolder: str) -> None:
                    subfolder_path = model_dir / subfolder if subfolder else model_dir
                    if subfolder_path.exists() and any(subfolder_path.iterdir()):
                        return
                    if not HF_AVAILABLE:
                        raise OnnxError(
                            "huggingface_hub not installed.\n"
                            'Run: pip install "huggingface-hub>=0.34.0,<1.0"'
                        )
                    self.after(0, lambda: self._append_chat(
                        "system", "Downloading ONNX model from HuggingFace …\n"
                    ))
                    download_onnx_model(
                        model["onnx_repo"],
                        subfolder,
                        model_dir,
                        progress_cb=lambda m: self.after(0, lambda msg=m: self.set_status(msg)),
                        stop_event=self._stop_event,
                    )

                load_provider = provider
                load_provider_options = provider_options
                load_provider_label = provider_label
                load_subfolder = onnx_subfolder

                ensure_subfolder(load_subfolder)
                use_genai = has_genai_config(model_dir, load_subfolder)
                if use_genai:
                    if not GENAI_AVAILABLE:
                        raise OnnxError(
                            "Model requires onnxruntime-genai (genai_config.json present), "
                            "but the package is not installed.\n"
                            "Run: pip install onnxruntime-genai-directml"
                        )
                    if load_provider == "OpenVINOExecutionProvider":
                        if DIRECTML_AVAILABLE:
                            load_provider = "DmlExecutionProvider"
                            load_provider_options = {}
                            load_provider_label = "DirectML/NPU"
                            load_subfolder = model.get("onnx_subfolder", "")
                        else:
                            load_provider = "CPUExecutionProvider"
                            load_provider_options = {}
                            load_provider_label = "CPU"
                            load_subfolder = (
                                model.get("onnx_cpu_subfolder")
                                or model.get("onnx_openvino_subfolder")
                                or load_subfolder
                            )
                        ensure_subfolder(load_subfolder)
                        use_genai = has_genai_config(model_dir, load_subfolder)

                self.after(0, lambda: self._append_chat(
                    "system", f"Loading weights onto {load_provider_label} …\n"
                ))
                if use_genai:
                    session = OnnxGenAISession(model_dir, subfolder=load_subfolder)
                else:
                    session = OnnxModelSession(
                        model_dir,
                        load_provider,
                        load_provider_options,
                        subfolder=load_subfolder,
                    )
                ram_after = system_info.get_ram_info()
                self.after(0, lambda: self._onnx_load_done(model, session, ram_after, load_provider_label))
            except (OnnxError, Exception) as e:
                err = str(e)
                self.after(0, lambda msg=err: (
                    self._append_chat("error", f"Failed to load ONNX model: {msg}\n"),
                    self.set_status("ONNX load failed."),
                    logger.error(f"ONNX load failed: {msg}", category=logger.CATEGORY_CHAT),
                ))

        threading.Thread(target=_load, daemon=True).start()

    def _onnx_load_done(
        self,
        model: dict,
        session: OnnxModelSession | OnnxGenAISession,
        ram_after: dict,
        provider_label: str,
    ):
        prev_model = self.active_model.get("name") if self.active_model else None
        self.active_model = model
        self.active_session = session
        self._active_model_display = f"{model['name']}  [{provider_label}]"
        self._set_chat_selector_to_model(model)
        self._active_model_label.configure(text=self._active_model_display)
        self._update_chat_readiness()
        self.chat_history.clear()
        self._update_chat_copy_buttons()
        if prev_model and prev_model != model["name"]:
            self._append_chat("model_switch", f"{model['name']}  ({provider_label})")
        avail = ram_after["available_mb"] / 1024
        used = ram_after["used_mb"] / 1024
        self._append_chat(
            "system",
            f"Model ready on {provider_label}.\n"
            f"RAM after load: {used:.1f} GB used, {avail:.1f} GB free.\n"
        )
        self._apply_chat_demo_prompt(model)
        self.set_status(f"{model['name']} loaded on {provider_label}.")
        logger.info(f"ONNX model loaded: {model['name']} on {provider_label}", category=logger.CATEGORY_CHAT)

    # ── Chat ──────────────────────────────────────────────────────────────────

    def _send_message(self):
        if not self.active_model:
            self.set_status("Pick a model and backend, then click Load selected.")
            self._update_chat_readiness()
            return

        text = self._input_box.get("1.0", "end").strip()
        if not text:
            return

        response_token_budget = self._sync_response_token_budget_from_ui()
        if response_token_budget is None:
            return

        self._input_box.delete("1.0", "end")
        self._append_chat("user", text)
        self.chat_history.append({"role": "user", "content": text})

        self._send_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._stop_event.clear()
        self._chat_generation_id += 1
        gen_id = self._chat_generation_id
        self._chat_thinking = True
        self._update_chat_readiness()
        self._start_chat_first_token_feedback()

        if self.active_session:
            self._stream_onnx(text, gen_id, response_token_budget)
        else:
            self._stream_ollama(gen_id, response_token_budget)

    def _stream_ollama(self, gen_id: int, response_token_budget: int):
        tag = self.active_model["ollama_tag"]
        num_gpu = self.active_model.get("_num_gpu", -1)
        temp = self.cfg.get("temperature", 0.7)
        history = list(self.chat_history)
        num_predict = response_token_budget if response_token_budget > 0 else -2
        keep_alive = str(self.active_model.get("ollama_keep_alive") or "").strip() or None
        try:
            num_ctx = int(self.active_model.get("ollama_num_ctx") or 0)
        except (TypeError, ValueError):
            num_ctx = 0
        if num_ctx <= 0:
            num_ctx = None

        buffer = [""]
        self._append_chat("assistant", "")  # placeholder
        insert_mark = self._chat_display.index("end-1c")

        def _stream():
            full = ""
            started = time.time()
            last_progress = started
            token_chunks = 0
            final_stats = None
            logger.info(f"Chat started: {tag}", category=logger.CATEGORY_CHAT)
            try:
                for token, stats in self.ollama.chat_stream_with_stats(
                    tag, history, num_gpu=num_gpu, temperature=temp,
                    num_predict=num_predict, stop_event=self._stop_event,
                    num_ctx=num_ctx, keep_alive=keep_alive,
                ):
                    if stats is not None:
                        final_stats = stats
                    if self._stop_event.is_set():
                        break
                    if not token:
                        continue
                    token_chunks += 1
                    now = time.time()
                    if token_chunks == 1:
                        logger.info(
                            f"Chat first token: {tag} after {now - started:.1f}s",
                            category=logger.CATEGORY_CHAT,
                        )
                    elif now - last_progress >= 15.0:
                        logger.info(
                            f"Chat streaming: {tag}, {len(full)} chars in {now - started:.0f}s",
                            category=logger.CATEGORY_CHAT,
                        )
                        last_progress = now
                    full += token
                    if gen_id == self._chat_generation_id:
                        self._enqueue_token(token, gen_id)
            except OllamaError as e:
                err = str(e)
                self.after(0, lambda m=err, gid=gen_id: gid == self._chat_generation_id and (
                    self._flush_tokens(),
                    self._clear_chat_response_placeholder(),
                    self._append_chat("error", f"\nError: {m}\n"),
                    logger.error(f"Chat error: {m}", category=logger.CATEGORY_CHAT),
                ))
            finally:
                elapsed = time.time() - started
                if self._stop_event.is_set():
                    logger.warning(
                        f"Chat stopped: {tag} after {elapsed:.1f}s and {len(full)} chars",
                        category=logger.CATEGORY_CHAT,
                    )
                elif final_stats:
                    load_s = float(final_stats.get("load_duration") or 0) / 1_000_000_000
                    prompt_s = float(final_stats.get("prompt_eval_duration") or 0) / 1_000_000_000
                    gen_s = float(final_stats.get("eval_duration") or 0) / 1_000_000_000
                    logger.info(
                        f"Chat complete: {tag}, {len(full)} chars in {elapsed:.1f}s "
                        f"(load {load_s:.2f}s, prompt {prompt_s:.2f}s, generate {gen_s:.2f}s)",
                        category=logger.CATEGORY_CHAT,
                    )
                else:
                    logger.info(
                        f"Chat complete: {tag}, {len(full)} chars in {elapsed:.1f}s",
                        category=logger.CATEGORY_CHAT,
                    )
                if gen_id == self._chat_generation_id:
                    self.chat_history.append({"role": "assistant", "content": full})
                self.after(0, lambda gid=gen_id: self._chat_done(gid))

        threading.Thread(target=_stream, daemon=True).start()

    def _stream_onnx(self, prompt: str, gen_id: int, response_token_budget: int):
        # Build a simple prompt from history
        full_prompt = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in self.chat_history
        ) + "\nAssistant:"

        session = self.active_session
        max_new_tokens = self._max_new_tokens_for_prompt(response_token_budget, full_prompt)

        def _stream():
            full = ""
            started = time.time()
            last_progress = started
            token_chunks = 0
            logger.info("ONNX chat started", category=logger.CATEGORY_CHAT)
            try:
                for token in session.generate_stream(
                    full_prompt, temperature=self.cfg.get("temperature", 0.7),
                    max_new_tokens=max_new_tokens,
                    stop_event=self._stop_event,
                ):
                    if self._stop_event.is_set():
                        break
                    token_chunks += 1
                    now = time.time()
                    if token_chunks == 1:
                        logger.info(
                            f"ONNX chat first token after {now - started:.1f}s",
                            category=logger.CATEGORY_CHAT,
                        )
                    elif now - last_progress >= 15.0:
                        logger.info(
                            f"ONNX chat streaming: {len(full)} chars in {now - started:.0f}s",
                            category=logger.CATEGORY_CHAT,
                        )
                        last_progress = now
                    full += token
                    if gen_id == self._chat_generation_id:
                        self._enqueue_token(token, gen_id)
            except OnnxError as e:
                err = str(e)
                self.after(0, lambda m=err, gid=gen_id: gid == self._chat_generation_id and (
                    self._flush_tokens(),
                    self._clear_chat_response_placeholder(),
                    self._append_chat("error", f"\nError: {m}\n"),
                    logger.error(f"ONNX chat error: {m}", category=logger.CATEGORY_CHAT),
                ))
            finally:
                elapsed = time.time() - started
                if self._stop_event.is_set():
                    logger.warning(
                        f"ONNX chat stopped after {elapsed:.1f}s and {len(full)} chars",
                        category=logger.CATEGORY_CHAT,
                    )
                else:
                    logger.info(
                        f"ONNX chat complete: {len(full)} chars in {elapsed:.1f}s",
                        category=logger.CATEGORY_CHAT,
                    )
                if gen_id == self._chat_generation_id:
                    self.chat_history.append({"role": "assistant", "content": full})
                self.after(0, lambda gid=gen_id: self._chat_done(gid))

        self._append_chat("assistant", "")
        threading.Thread(target=_stream, daemon=True).start()

    def _append_token(self, token: str):
        # v5: kept for compatibility; new code uses _enqueue_token / _flush_tokens
        self._enqueue_token(token)

    def _enqueue_token(self, token: str, gen_id: int | None = None) -> None:
        """
        v5 token batching: accumulate tokens in a buffer and flush every 50 ms.
        Reduces per-token UI updates from 200+/response to ~20/response.
        Safe to call from any thread.
        """
        if gen_id is None:
            gen_id = self._chat_generation_id
        if gen_id != self._chat_generation_id:
            return
        self._token_buf.append((gen_id, token))
        if not self._token_flush_scheduled:
            self._token_flush_scheduled = True
            self.after(50, self._flush_tokens)

    def _flush_tokens(self) -> None:
        """Drain the token buffer to the chat display in one UI update."""
        self._token_flush_scheduled = False
        if not self._token_buf:
            return
        current_gen = self._chat_generation_id
        batch = "".join(token for gen_id, token in self._token_buf if gen_id == current_gen)
        self._token_buf.clear()
        if not batch:
            return
        # Remove the placeholder only once. Later token batches must append.
        if getattr(self, "_chat_response_placeholder_active", False):
            self._stop_chat_first_token_feedback("Receiving response …")
            self._clear_chat_response_placeholder()
        self._chat_display.configure(state="normal")
        self._chat_display.insert("end", batch)
        self._chat_display.configure(state="disabled")
        self._chat_display.see("end")

    def _clear_chat_response_placeholder(self) -> None:
        if not getattr(self, "_chat_response_placeholder_active", False):
            return
        self._chat_display.configure(state="normal")
        try:
            self._chat_display.delete("assistant_body", "end-1c")
        except Exception:
            content = self._chat_display.get("1.0", "end")
            if content.rstrip().endswith("thinking …"):
                self._chat_display.delete("end-11c", "end-1c")
        self._chat_response_placeholder_active = False
        self._chat_display.configure(state="disabled")

    def _append_chat(self, role: str, text: str):
        self._chat_display.configure(state="normal")
        if role == "user":
            self._chat_display.insert("end", "\n")
            self._chat_display.insert("end", "You:", "user_label")
            self._chat_display.insert("end", f" {text}\n\n")
        elif role == "assistant":
            self._chat_display.insert("end", "Assistant:", "assistant_label")
            self._chat_display.insert("end", " ")
            # Mark where streamed tokens begin so the "thinking …"
            # placeholder can be replaced cleanly without trimming the label.
            self._chat_display.mark_set("assistant_body", "end-1c")
            self._chat_display.mark_gravity("assistant_body", "left")
            self._chat_response_placeholder_active = True
            self._chat_display.insert("end", "thinking …", "thinking")
        elif role == "system":
            self._chat_display.insert("end", f"[{text.rstrip()}]\n", "system_text")
        elif role == "error":
            self._chat_display.insert("end", text, "error_text")
        elif role == "model_switch":
            self._chat_display.insert(
                "end",
                f"\n{'─' * 60}\n  Switched to model: {text}\n{'─' * 60}\n\n",
                "switch",
            )
        self._chat_display.configure(state="disabled")
        self._chat_display.see("end")

    def _chat_done(self, gen_id: int | None = None):
        if gen_id is not None and gen_id != self._chat_generation_id:
            return
        self._flush_tokens()
        self._clear_chat_response_placeholder()
        self._stop_chat_first_token_feedback()
        self._chat_thinking = False
        self._chat_response_placeholder_active = False
        self._chat_display.configure(state="normal")
        self._chat_display.insert("end", "\n\n")
        self._chat_display.configure(state="disabled")
        self._update_chat_readiness()
        self._update_chat_copy_buttons()
        self._stop_btn.configure(state="disabled")
        self.set_status("Ready.")

    def _stop_generation(self):
        self._stop_event.set()
        self._chat_generation_id += 1
        self._token_buf.clear()
        self._stop_btn.configure(state="disabled")
        self._clear_chat_response_placeholder()
        self._stop_chat_first_token_feedback("Generation stopped.")
        self._chat_thinking = False
        self._chat_response_placeholder_active = False
        self._update_chat_readiness()

    def _clear_chat(self):
        self.chat_history.clear()
        self._chat_display.configure(state="normal")
        self._chat_display.delete("1.0", "end")
        self._chat_display.configure(state="disabled")
        self._update_chat_copy_buttons()

    # ── Delete models ─────────────────────────────────────────────────────────

    def delete_model(self, model: dict):
        tag = model.get("ollama_tag", "")
        try:
            self.ollama.delete_model(tag)
            if tag in self.cfg["downloaded_this_session"]:
                self.cfg["downloaded_this_session"].remove(tag)
            if not config.save(self.cfg):
                logger.error("Could not persist downloaded_this_session after delete")
            cache = getattr(self, "_local_names_cache", None)
            if cache:
                cached_at, names = cache
                self._local_names_cache = (
                    cached_at,
                    {name for name in names if not _ollama_tag_is_local(tag, {name})},
                )
            self._invalidate_model_status_refresh()
            detail_names = getattr(self, "_model_detail_local_names", None)
            if detail_names is not None:
                self._model_detail_local_names = {
                    name for name in detail_names if not _ollama_tag_is_local(tag, {name})
                }
            local_names, comfyui_model_names = self._latest_model_status_snapshots()
            self.set_status(f"Deleted: {model['name']}")
            logger.info(f"Model deleted: {tag}")
            self._refresh_chat_model_selector()
            self._refresh_visible_model_status_from_snapshots(
                local_names=local_names,
                comfyui_model_names=comfyui_model_names,
            )
            self._schedule_model_status_refresh(force_refresh=False)
        except OllamaError as e:
            messagebox.showerror("Delete failed", str(e), parent=self)
            logger.error(f"Delete failed for {tag}: {e}")

    def delete_comfyui_model(self, model: dict):
        model_filename = model.get("comfyui_model", "")
        comfyui_path = self._comfyui_installed_path()
        if not comfyui_path or not model_filename:
            messagebox.showerror("Delete failed",
                                 "Cannot locate ComfyUI installation or model filename.",
                                 parent=self)
            return
        deleted = False
        for subdir in ("checkpoints", "diffusion_models"):
            target = comfyui_path / "models" / subdir / model_filename
            if target.exists():
                try:
                    target.unlink()
                    deleted = True
                    logger.info(f"Deleted ComfyUI model file: {target}")
                except Exception as e:
                    messagebox.showerror("Delete failed", str(e), parent=self)
                    logger.error(f"Failed to delete {target}: {e}")
                    return
                break
        if deleted:
            cache = getattr(self, "_comfyui_model_names_cache", None)
            if cache:
                cached_at, names = cache
                self._comfyui_model_names_cache = (
                    cached_at,
                    {name for name in names if name != model_filename},
                )
            self._invalidate_model_status_refresh()
            detail_names = getattr(self, "_model_detail_comfyui_model_names", None)
            if detail_names is not None:
                self._model_detail_comfyui_model_names = {
                    name for name in detail_names if name != model_filename
                }
            local_names, comfyui_model_names = self._latest_model_status_snapshots()
            self.set_status(f"Deleted: {model['name']}")
            self._refresh_visible_model_status_from_snapshots(
                local_names=local_names,
                comfyui_model_names=comfyui_model_names,
            )
            self._schedule_model_status_refresh(force_refresh=False)
        else:
            messagebox.showerror("Delete failed",
                                 f"Model file '{model_filename}' not found on disk.",
                                 parent=self)
            logger.error(f"delete_comfyui_model: file not found — {model_filename}")

    def _delete_all_models(self):
        ok = messagebox.askyesno(
            "Delete ALL models",
            "This will delete every downloaded model from Ollama.\n"
            "You will need to re-download them to use them again.\n\nContinue?",
            parent=self,
        )
        if not ok:
            return
        try:
            local = self.ollama.list_local_models()
        except OllamaError:
            local = []
        for m in local:
            tag = m.get("name", "")
            try:
                self.ollama.delete_model(tag)
                logger.info(f"Deleted: {tag}")
            except OllamaError as e:
                logger.error(f"Could not delete {tag}: {e}")
        self._refresh_model_cards()
        self.set_status("All local models deleted.")

    # ── Resource monitor ──────────────────────────────────────────────────────

    def _start_resource_monitor(self):
        def _monitor():
            while not getattr(self, "_closing", False):
                time.sleep(5)
                if getattr(self, "_closing", False):
                    break
                if self._current_page == "system":
                    self.after(0, self._update_system_page)
        threading.Thread(target=_monitor, daemon=True).start()

    # ── Status bar ────────────────────────────────────────────────────────────

    def set_status(self, msg: str):
        if getattr(self, "_closing", False):
            return
        self._status_label.configure(text=msg)

    # ── Close / cleanup ───────────────────────────────────────────────────────

    def _on_close(self):
        if getattr(self, "_closing", False):
            return
        session_downloads = self.cfg.get("downloaded_this_session", [])

        if session_downloads:
            try:
                local = {m["name"] for m in self.ollama.list_local_models()}
                still_local = [t for t in session_downloads if _ollama_tag_is_local(t, local)]
            except Exception:
                still_local = session_downloads

            if still_local:
                names = "\n  • ".join(still_local)
                answer = messagebox.askyesnocancel(
                    "Free up disk space?",
                    f"These models were downloaded this session:\n  • {names}\n\n"
                    "Would you like to delete them to free up disk space?\n\n"
                    "Yes = delete them   No = keep them   Cancel = don't quit yet",
                    parent=self,
                )
                if answer is None:
                    return  # user cancelled — don't close
                if answer:
                    for tag in still_local:
                        try:
                            self.ollama.delete_model(tag)
                            logger.info(f"Cleanup: deleted {tag}")
                        except OllamaError:
                            pass

        self._closing = True
        self.cfg["downloaded_this_session"] = []
        if not config.save(self.cfg):
            logger.error("Shutdown: settings save failed")
        self._stop_event.set()
        if hasattr(self, "_img_stop_event"):
            self._img_stop_event.set()
        if hasattr(self, "_analyze_stop_event"):
            self._analyze_stop_event.set()

        # Terminate the ComfyUI process we own
        if self.comfyui_process and self.comfyui_process.poll() is None:
            logger.info(f"_on_close: terminating ComfyUI PID {self.comfyui_process.pid}")
            try:
                self.comfyui_process.terminate()
                self.comfyui_process.wait(timeout=5)
            except Exception:
                try:
                    self.comfyui_process.kill()
                except Exception:
                    pass
        # Belt-and-suspenders: sweep any orphan ComfyUI processes (zombie subprocesses
        # from earlier crashes or detached restarts can hold GPU VRAM after parent exit).
        try:
            self._kill_orphan_comfyui_processes()
        except Exception:
            pass
        self._close_comfyui_log_handle()
        logger.remove_listener(self._on_log_entry)
        self._stop_docs_http_server()

        self.destroy()










































































































# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Persistent application configuration stored as JSON.
"""

import json
import os
import sys  # noqa: F401  (kept for backward import compatibility)
from pathlib import Path
from typing import Optional

from src import logger as _log
from src.persistence import atomic_write_json


DEFAULT_THEME_PALETTES = {
    "light": {
        "text_primary": "#111827",
        "text_secondary": "#1f2937",
        "text_muted": "#374151",
        "text_disabled": "#4b5563",
        "link_text": "#064f9e",
        "info_text": "#005a9e",
        "warn_text": "#8a4b00",
        "success_text": "#176b2c",
        "error_text": "#b00020",
        "border_strong": "#6b7280",
        "surface_card": "#ebebeb",
        "surface_inner": "#f7f7f7",
        "input_surface": "#ffffff",
    },
    "dark": {
        "text_primary": "#f9fafb",
        "text_secondary": "#e5e7eb",
        "text_muted": "#d1d5db",
        "text_disabled": "#cbd5e1",
        "link_text": "#8ec5ff",
        "info_text": "#7cc7ff",
        "warn_text": "#ffb74d",
        "success_text": "#7ee787",
        "error_text": "#ff8a80",
        "border_strong": "#9ca3af",
        "surface_card": "#2b2b2b",
        "surface_inner": "#303030",
        "input_surface": "#3a3a3a",
    },
}

SD15_NEGATIVE_PROMPT = (
    "lowres, blurry, jpeg artifacts, low quality, worst quality, watermark, text, "
    "signature, logo, ugly, deformed, bad anatomy, extra fingers, missing fingers, "
    "mutated hands, bad proportions, cropped, out of frame, oversaturated"
)

ANIME_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry, oversaturated, simple background, sketch, monochrome"
)

SDXL_NEGATIVE_PROMPT = (
    "low quality, worst quality, jpeg artifacts, blurry, washed out, oversaturated, "
    "plastic skin, deformed, bad anatomy, extra fingers, mutated hands, watermark, "
    "text, signature, logo, ugly, cropped, out of frame"
)

PLAYGROUND_NEGATIVE_PROMPT = (
    SDXL_NEGATIVE_PROMPT
    + ", airbrushed skin, waxy skin, doll-like skin, porcelain skin, textureless skin, "
    + "plastic hair, helmet hair, smeared hair, textureless hair"
)

DEFAULT_NEGATIVE_PROMPTS_BY_MODEL = {
    "sd15": SD15_NEGATIVE_PROMPT,
    "realistic-vision-v6": SD15_NEGATIVE_PROMPT,
    "dreamlike-art-dreamlike-diffusion-1.0": SD15_NEGATIVE_PROMPT,
    "lykon-dreamshaper": SD15_NEGATIVE_PROMPT,
    "nuigurumi-basil-mix": ANIME_NEGATIVE_PROMPT,
    "gsdf-counterfeit-v3.0": ANIME_NEGATIVE_PROMPT,
    "xiaolxl-guofeng3": SD15_NEGATIVE_PROMPT,
    "stabilityai-sd-turbo": "",
    "sdxl-base": SDXL_NEGATIVE_PROMPT,
    "sdxl-lowvram": SDXL_NEGATIVE_PROMPT,
    "juggernaut-xl-v9": SDXL_NEGATIVE_PROMPT,
    "segmind-ssd-1b": SDXL_NEGATIVE_PROMPT,
    "playgroundai-playground-v2.5-1024px-aesthetic": PLAYGROUND_NEGATIVE_PROMPT,
    "cagliostrolab-animagine-xl-4.0": ANIME_NEGATIVE_PROMPT,
    "dataautogpt3-opendallev1.1": SDXL_NEGATIVE_PROMPT,
    "stabilityai-sdxl-turbo": "",
    "bytedance-sdxl-lightning": "",
    "flux1-schnell-q4": "",
    "flux1-dev-q4": "",
    "flux1-dev-fp16": "",
    "ostris-openflux.1": "",
    "freepik-flux.1-lite-8b-alpha": "",
    "z-image": "",
    "z-image-turbo": "",
    "fal-auraflow": "",
}


def _default_data_dir() -> Path:
    """Return the app-path-relative data directory root.

    Data lives next to the app per the regression-critical contract —
    never under ``%USERPROFILE%``, never under ``%LOCALAPPDATA%``, and
    never on a hardcoded drive letter. The "right" drive is always
    whatever drive the app itself was unpacked to; we derive it live
    from ``Path(__file__).parent.parent`` (this file lives at
    ``<app_path>/src/config.py``) so the answer follows the install,
    not the user profile or any drive letter baked in at build time.

    Returned on BOTH Windows and macOS so the data layout is identical
    across platforms: ``<app_path>/models``, ``<app_path>/ComfyUI``,
    ``<app_path>/Ollama``. The previous Windows ``%LOCALAPPDATA%`` /
    macOS ``~/Library/Application Support`` defaults were retired in
    post-v5.3.7 because they tripped the roaming profile container
    cap on constrained cloud VMs and forced roamed
    profiles to stream multi-GB model blobs.
    """
    return Path(__file__).parent.parent


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _is_legacy_profile_comfyui_dir(path: Path) -> bool:
    """Return True for retired profile-relative ComfyUI defaults."""
    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "LocalAI" / "ComfyUI")
    try:
        home = Path.home()
    except RuntimeError:
        home = None
    if home is not None:
        candidates.extend([
            home / "AppData" / "Local" / "LocalAI" / "ComfyUI",
            home / "Library" / "Application Support" / "LocalAI" / "ComfyUI",
        ])
    return any(_same_path(path, candidate) for candidate in candidates)


def _has_comfyui_install(path: Path) -> bool:
    return (path / "main.py").is_file()


def _candidate_comfyui_install_dirs(app_root: Path) -> list[Path]:
    """Return the ordered list of locations where ComfyUI may live next
    to a LocalAI app install. Matches the lookup order in ``setup.bat`` so
    a setup that picked the sibling location is honored even when the
    user has not re-run setup inside the current app tree (which is the
    common case when v.N+1 is unzipped on top of a v.N install: the
    fresh app dir has no ``config.json`` / ``comfyui_path.bat`` yet, but
    the old sibling ComfyUI is still on disk and fully usable).

    Order:
      1. ``<app_root>/ComfyUI``   (default install — child of app)
      2. ``<app_root>/../ComfyUI`` (sibling — matches ``%~dp0..\\ComfyUI``
                                    fallback in ``setup.bat``)
    """
    return [app_root / "ComfyUI", app_root.parent / "ComfyUI"]


def _resolve_existing_comfyui_install(app_root: Path) -> Optional[Path]:
    """Return the first existing ComfyUI install in the standard
    next-to-the-app search order, or ``None`` if neither candidate
    contains ``main.py``."""
    for candidate in _candidate_comfyui_install_dirs(app_root):
        if _has_comfyui_install(candidate):
            return candidate
    return None

DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "models_dir": "",          # filled in at first run as <app_path>/models
    "comfyui_dir": "",         # filled in at first run as <app_path>/ComfyUI
    "default_backend": "auto", # auto | gpu | npu | cpu
    "auto_start_ollama": True,
    "theme_mode": "system",     # system | dark | light
    "theme_palettes": DEFAULT_THEME_PALETTES,
    "dark_mode": False,         # legacy compatibility; theme_mode is authoritative
    "downloaded_this_session": [],  # list of ollama tags pulled this session
    "temperature": 0.7,
    "response_token_mode": "max",  # max | custom
    "max_tokens": 0,               # custom generated-token cap; 0 = fill context
    "low_resources_mode": False,
    "toolbox_left_column_mode": "normal",  # normal | compact
    "chat_font_family": "Segoe UI",
    "default_negative_prompts": DEFAULT_NEGATIVE_PROMPTS_BY_MODEL,
    # v5.1: which vision (multimodal) model the Image Gen page uses for
    # Analyze → Prompt. Defaults to Gemma 3 4B for best balance of accuracy
    # and speed. Persisted whenever the user changes it from the picker.
    "default_vision_model_id": "gemma3:4b-vision",
    # v5.5.14: reference-image (img2img) defaults persisted across restarts.
    # 0.55 is the sweet spot for "restyle while preserving composition" —
    # the historical 0.75 default essentially ignored the reference.
    # img2img_match_aspect snaps the workflow's W×H to the closest SDXL
    # bucket for the reference's aspect ratio at queue time, avoiding the
    # "center-crop cuts the head off" failure with portrait references on
    # a square target.
    "img2img_denoise": 0.55,
    "img2img_match_aspect": True,
}

CONFIG_FILE = Path(__file__).parent.parent / "config.json"
THEME_MODES = {"system", "dark", "light"}
TOOLBOX_LEFT_COLUMN_MODES = {"normal", "compact"}


def normalize_theme_mode(value: object) -> str:
    """Return a supported CustomTkinter appearance mode name."""
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in THEME_MODES:
            return mode
    return DEFAULT_CONFIG["theme_mode"]


def normalize_toolbox_left_column_mode(value: object) -> str:
    """Return a supported Toolbox left-column density mode."""
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in TOOLBOX_LEFT_COLUMN_MODES:
            return mode
    return DEFAULT_CONFIG["toolbox_left_column_mode"]


def normalize_theme_palettes(value: object) -> dict:
    """Merge user-editable light/dark palette overrides with safe defaults."""
    merged = {
        mode: dict(DEFAULT_THEME_PALETTES[mode])
        for mode in ("light", "dark")
    }
    if not isinstance(value, dict):
        return merged
    for mode in ("light", "dark"):
        overrides = value.get(mode)
        if not isinstance(overrides, dict):
            continue
        for key, default_value in DEFAULT_THEME_PALETTES[mode].items():
            candidate = overrides.get(key)
            if isinstance(candidate, str) and candidate.strip():
                merged[mode][key] = candidate.strip()
            else:
                merged[mode][key] = default_value
    return merged


def normalize_default_negative_prompts(value: object) -> dict:
    """Merge user overrides with the built-in ImageGenPrompts model defaults."""
    merged = dict(DEFAULT_NEGATIVE_PROMPTS_BY_MODEL)
    if not isinstance(value, dict):
        return merged
    for model_id, prompt in value.items():
        if isinstance(model_id, str) and isinstance(prompt, str):
            merged[model_id.strip()] = prompt.strip()
    return merged


def normalize_response_token_settings(cfg: dict) -> None:
    """Normalize chat response-length settings in-place."""
    mode = str(cfg.get("response_token_mode") or "max").strip().lower()
    if mode != "custom":
        cfg["response_token_mode"] = "max"
        cfg["max_tokens"] = 0
        return
    try:
        max_tokens = int(cfg.get("max_tokens") or 0)
    except (TypeError, ValueError):
        max_tokens = 0
    if max_tokens <= 0:
        cfg["response_token_mode"] = "max"
        cfg["max_tokens"] = 0
        return
    cfg["response_token_mode"] = "custom"
    cfg["max_tokens"] = max_tokens


def load() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    had_theme_mode = False
    loaded_ok = False
    saved_payload = None
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved_payload = saved
            loaded_ok = True
            had_theme_mode = "theme_mode" in saved
            cfg.update(saved)
        except json.JSONDecodeError as exc:
            _log.error(f"Config: {CONFIG_FILE} is invalid JSON ({exc}) — using defaults.")
        except OSError as exc:
            _log.error(f"Config: could not read {CONFIG_FILE} ({exc}) — using defaults.")
    cfg["theme_mode"] = normalize_theme_mode(cfg.get("theme_mode") if had_theme_mode else None)
    cfg["theme_palettes"] = normalize_theme_palettes(cfg.get("theme_palettes"))
    cfg["default_negative_prompts"] = normalize_default_negative_prompts(
        cfg.get("default_negative_prompts")
    )
    normalize_response_token_settings(cfg)
    cfg["toolbox_left_column_mode"] = normalize_toolbox_left_column_mode(
        cfg.get("toolbox_left_column_mode")
    )
    cfg.pop("default_negative_prompt", None)
    cfg["dark_mode"] = cfg["theme_mode"] == "dark"
    # Set default models dir to <app_path>/models (next to the app)
    if not cfg["models_dir"]:
        cfg["models_dir"] = str(_default_data_dir() / "models")
    # Set/default-repair ComfyUI dir to <app_path>/ComfyUI (next to the app).
    # Older v5.3.x setup builds wrote %LOCALAPPDATA%/Application Support paths;
    # on cloud VMs those profile paths are often missing or size-capped.
    #
    # v2026.06.01.9: when ``comfyui_dir`` is empty (e.g. a fresh app
    # extraction on top of an older install where setup.bat was never re-run
    # in the new tree), also probe the sibling location
    # (``<app_path>/../ComfyUI``) that ``setup.bat`` itself falls back to.
    # Without this, the resolver returns the child default path even when
    # ComfyUI is sitting one folder above the app, and the benchmark / image
    # gen tab report "ComfyUI not installed at expected paths" for a working
    # install.
    app_root = _default_data_dir()
    default_comfyui_dir = app_root / "ComfyUI"
    comfyui_dir_value = str(cfg.get("comfyui_dir") or "").strip()
    if not comfyui_dir_value:
        resolved = _resolve_existing_comfyui_install(app_root)
        cfg["comfyui_dir"] = str(resolved if resolved is not None else default_comfyui_dir)
    else:
        comfyui_path = Path(comfyui_dir_value)
        configured_missing = not _has_comfyui_install(comfyui_path)
        configured_is_legacy_profile = _is_legacy_profile_comfyui_dir(comfyui_path)
        if configured_missing or configured_is_legacy_profile:
            resolved = _resolve_existing_comfyui_install(app_root)
            if resolved is not None:
                cfg["comfyui_dir"] = str(resolved)
    if loaded_ok and saved_payload is not None:
        missing_default_key = any(key not in saved_payload for key in DEFAULT_CONFIG)
        if (
            missing_default_key
            or saved_payload.get("theme_palettes") != cfg["theme_palettes"]
            or saved_payload.get("default_negative_prompts") != cfg["default_negative_prompts"]
            or saved_payload.get("response_token_mode") != cfg["response_token_mode"]
            or saved_payload.get("max_tokens") != cfg["max_tokens"]
            or saved_payload.get("toolbox_left_column_mode") != cfg["toolbox_left_column_mode"]
            or saved_payload.get("models_dir") != cfg["models_dir"]
            or saved_payload.get("comfyui_dir") != cfg["comfyui_dir"]
        ):
            save(cfg)
    return cfg


def save(cfg: dict) -> bool:
    try:
        atomic_write_json(CONFIG_FILE, cfg, indent=2, ensure_ascii=False)
        return True
    except OSError as exc:
        _log.error(f"Config: could not write {CONFIG_FILE}: {exc}")
        return False


def models_dir(cfg: dict) -> Path:
    p = Path(cfg["models_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def comfyui_dir(cfg: dict) -> Path:
    """Return the configured ComfyUI directory (does NOT create it)."""
    return Path(cfg["comfyui_dir"])

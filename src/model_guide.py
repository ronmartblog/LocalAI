# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Build the consolidated Model Guide HTML page.

This module replaces four legacy docs (ChatPromptIdeas, ImageGenPrompts,
ModelDemoPrompts, model-value-props) with a single unified guide that:

* Uses the cool-blue dark/light theme (dark default).
* Renders every active catalog model as a collapsible card.
* Pulls demo prompts from src.model_demos.get_model_demo (which already merges
  src.sample_prompts.MODEL_DEMO_SAMPLE_OVERRIDES on top of the curated
  _IMAGE_PROMPTS / _UTILITY_DEMOS / _CHAT_OVERRIDES tables).
* Pulls recommended image-gen settings from models_catalog.json's
  ``recommended_settings`` block.
* Pulls negative prompts via src.model_demos._negative_prompt_for_model.
* Preserves every URL-param contract from the legacy docs
  (``?modelId`` / ``?model`` / ``?chatModel`` / ``?imageModel`` / ``?ollama``
  / ``?ollamaTag`` / ``?vram`` / ``?hardware`` / ``?gpu`` / ``?ram`` /
  ``?unified`` / ``?surface`` / ``?goal``) plus the ``#model-<slug>``
  fragment scheme so all four legacy app.py opener callsites keep working.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from src.model_demos import (
    _IMAGE_PROMPTS,
    _UTILITY_DEMOS,
    _chat_reply_limit_label,
    _clean,
    _gb_label,
    _negative_prompt_for_model,
    _sample_thumbnail_url,
    _token_label,
    doc_fragment,
    get_model_demo,
)
from src import system_info


def _ram_tier_gb(min_ram_gb) -> int | None:
    """Bucket a model's ``min_ram_gb`` into the public CPU RAM tier strip
    (16 / 32 / 64 GB). Returns None for models with no RAM hint or for
    surfaces that don't get a tier (image / speech / embed / doc).
    """
    try:
        mr = int(min_ram_gb or 0)
    except (TypeError, ValueError):
        return None
    if mr <= 0:
        return None
    if mr <= 16:
        return 16
    if mr <= 32:
        return 32
    return 64


def _card_ram_tier(model: dict) -> int | None:
    """Only chat + vision cards carry a CPU RAM tier (their CPU fallback
    is meaningful). Image-gen / speech / embed / document cards are GPU
    workloads where CPU RAM tiering would be misleading.
    """
    surface = _model_surface(model)
    if surface not in ("chat", "vision"):
        return None
    return _ram_tier_gb(model.get("min_ram_gb"))


def _render_sku_chips() -> str:
    """Emit hardware-filter chips: ``All`` + three CPU RAM tier chips
    (``cpu16`` / ``cpu32`` / ``cpu64``) + one chip per GPU SKU loaded from
    ``skus.json``.

    Each GPU chip uses the SKU's ``vram_gb`` as ``data-sku`` so it maps to
    the existing ``#hardware`` ``<select>`` options ("4", "8", "12", "16",
    "24") and the CSS dot-color stops at ``.sku-chip[data-sku="<vram>"]``.
    SKU display names come straight from the JSON — never hardcoded here.
    When no SKUs are loaded (file missing/broken) only the All + CPU
    tier chips are emitted; the rest of the filter UI keeps working via
    the hardware select.
    """
    chips: list[str] = [
        '<button class="sku-chip active" data-sku="all" type="button" aria-pressed="true">'
        '<span class="sku-dot" aria-hidden="true"></span>All '
        '<span class="sku-sub">all hardware</span></button>',
        '<button class="sku-chip" data-sku="cpu16" type="button" aria-pressed="false">'
        '<span class="sku-dot" aria-hidden="true"></span>CPU \u2264 16 GB '
        '<span class="sku-sub">RAM \u00b7 up to ~14B</span></button>',
        '<button class="sku-chip" data-sku="cpu32" type="button" aria-pressed="false">'
        '<span class="sku-dot" aria-hidden="true"></span>CPU \u2264 32 GB '
        '<span class="sku-sub">RAM \u00b7 up to ~30B</span></button>',
        '<button class="sku-chip" data-sku="cpu64" type="button" aria-pressed="false">'
        '<span class="sku-dot" aria-hidden="true"></span>CPU \u2264 64 GB '
        '<span class="sku-sub">RAM \u00b7 30B+ class</span></button>',
    ]
    seen_vram: set[int] = set()
    for sku in system_info.get_benchmark_sku_profiles():
        try:
            vram = int(round(float(sku.get("vram_gb") or 0)))
        except (TypeError, ValueError):
            continue
        if vram <= 0 or vram in seen_vram:
            continue
        seen_vram.add(vram)
        name = html.escape(str(sku.get("name") or ""))
        chips.append(
            f'<button class="sku-chip" data-sku="{vram}" type="button" aria-pressed="false">'
            f'<span class="sku-dot" aria-hidden="true"></span>{name} '
            f'<span class="sku-sub">{vram} GB</span></button>'
        )
    return "\n    ".join(chips)


SURFACES = [
    ("chat",   "Chat",             "Text LLMs through Ollama — code, reasoning, long-context, multilingual."),
    ("vision", "Vision",           "Multimodal models for image understanding and Analyze → Prompt."),
    ("image",  "Image generation", "ComfyUI checkpoints, distilled fast variants, and high-quality photoreal."),
    ("speech", "Speech",           "Local transcription (Whisper) and TTS (SpeechT5)."),
    ("embed",  "Embeddings",       "Fast CPU embedding for semantic search and retrieval."),
    ("doc",    "Document AI",      "OCR, table detection, and image captioning surfaced in Toolbox."),
]


def _model_surface(model: dict) -> str:
    """Return one of the six surface ids for this catalog entry."""
    mid = _clean(model.get("id")).lower()
    backend = _clean(model.get("backend")).lower()
    category = _clean(model.get("category")).lower()
    tags = {t.lower() for t in (model.get("tags") or [])}

    if backend == "comfyui" or category == "image generation" or mid in _IMAGE_PROMPTS:
        return "image"
    if category == "speech" or any(t in tags for t in ("speech", "tts", "whisper", "speech-to-text", "text-to-speech")) \
            or "whisper" in mid or "tts" in mid or "speecht5" in mid:
        return "speech"
    if category == "embeddings" or "embedding" in tags or "embed" in mid or "minilm" in mid:
        return "embed"
    if category == "document ai" or any(t in tags for t in ("ocr", "captioning", "document")) \
            or "trocr" in mid or "florence" in mid or "table-transformer" in mid:
        return "doc"
    if "vision" in tags or "multimodal" in tags or "vision" in category \
            or "vl" in mid.split(":")[0].split("-") or "vision" in mid:
        return "vision"
    return "chat"


def _model_goals(model: dict, surface: str) -> list[str]:
    """Derive a small set of goal tags used by the Goal filter dropdown."""
    mid = _clean(model.get("id")).lower()
    tags = {t.lower() for t in (model.get("tags") or [])}
    category = _clean(model.get("category")).lower()
    rec = " ".join(
        _clean(r).lower()
        for r in system_info.get_recommended_skus_for_model(model.get("id"), model)
    )
    showcase = _clean(model.get("showcase_prompt")).lower()
    desc = _clean(model.get("description")).lower()
    haystack = f"{mid} {category} {rec} {showcase} {desc} {' '.join(tags)}"

    goals: list[str] = []

    def hit(*needles: str) -> bool:
        return any(n in haystack for n in needles)

    if surface == "image":
        if hit("photoreal", "photo-real", "photo", "realistic", "sdxl", "juggernaut", "realistic-vision"):
            goals.append("photo")
        if hit("anime", "counterfeit", "booru", "illustration", "cartoon"):
            goals.append("anime")
        if hit("paint", "art", "chroma", "playground", "aesthetic", "artistic"):
            goals.append("art")
        if hit("schnell", "turbo", "lightning", "fast", "lowvram", "z-image-turbo", "q4"):
            goals.append("fast")
        if not goals:
            goals.append("photo")
        return goals

    if surface == "speech":
        if hit("whisper", "speech-to-text", "transcrib"):
            goals.append("transcribe")
        if hit("tts", "speecht5", "text-to-speech"):
            goals.append("tts")
        return goals or ["transcribe"]

    if surface == "embed":
        return ["retrieve"]

    if surface == "doc":
        return ["ocr"]

    if surface == "vision":
        goals.append("quick")
        if hit("chart", "screenshot", "ocr", "qwen2.5vl", "vl"):
            goals.append("ocr")
        if hit("reason"):
            goals.append("reasoning")
        return goals

    if hit("coder", "coding", "code"):
        goals.append("coding")
    if hit("reason", "math", "logic", "deepseek"):
        goals.append("reasoning")
    if hit("multilingual", "aya", "translat"):
        goals.append("multilingual")
    if hit("long-context", "long context", "128k", "131072", "mistral-nemo"):
        goals.append("long")
    if not goals:
        goals.append("quick")
    return goals


def _surface_label(surface: str) -> str:
    for sid, label, _desc in SURFACES:
        if sid == surface:
            return label
    return surface.title()


def _image_settings(model: dict) -> tuple[dict[str, str], bool]:
    """Return (settings_dict, is_cfg_locked) for an image-gen card."""
    rec = model.get("recommended_settings") or {}
    width = rec.get("width") or model.get("default_width") or 1024
    height = rec.get("height") or model.get("default_height") or 1024
    sampler = _clean(rec.get("sampler")) or _clean(model.get("default_sampler")) or "euler"
    scheduler = _clean(rec.get("scheduler")) or _clean(model.get("default_scheduler")) or "normal"
    steps = rec.get("steps") if rec.get("steps") is not None else model.get("default_steps", 20)
    cfg = rec.get("cfg") if rec.get("cfg") is not None else model.get("default_cfg", 7.0)
    cfg_locked = bool(rec.get("cfg_locked"))
    cfg_label = f"{cfg:g}" if isinstance(cfg, (int, float)) else _clean(cfg)
    if cfg_locked:
        cfg_label = f"{cfg_label} 🔒"
    settings: dict[str, str] = {
        "size": f"{width} × {height}",
        "sampler": sampler,
        "scheduler": scheduler,
        "steps": str(steps),
        "cfg": cfg_label,
    }
    # CFG-locked is already conveyed by the 🔒 on the cfg cell + the dedicated
    # "negative prompts ignored" panel below. Don't add a redundant tile.
    return settings, cfg_locked


def _escape_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _render_card(model: dict, surface: str) -> str:
    mid = _clean(model.get("id"))
    name = _clean(model.get("name")) or mid
    vendor = _clean(model.get("vendor")) or "Unknown"
    category = _clean(model.get("category")) or "Model"
    slug = doc_fragment(mid).removeprefix("model-")
    fragment = f"model-{slug}" if slug else f"model-{mid}"

    demo = get_model_demo(model)
    feature = _clean(demo.get("feature")) or "Local AI"
    why = _clean(demo.get("why"))
    samples = list(demo.get("samples", []))[:3]

    min_ram = _gb_label(model.get("min_ram_gb"), zero_label="any RAM")
    min_vram_raw = model.get("min_vram_gb") or 0
    try:
        min_vram_num = float(min_vram_raw)
    except (TypeError, ValueError):
        min_vram_num = 0.0
    min_vram_label = _gb_label(min_vram_raw, zero_label="CPU OK")
    context_label = _token_label(model.get("context_length"))

    fit_parts = [f"{min_ram} RAM", min_vram_label if min_vram_num > 0 else "CPU OK"]
    if min_vram_num > 0:
        fit_parts[1] = f"{min_vram_label} VRAM"
    fit = " · ".join(fit_parts)

    surface_label = _surface_label(surface)
    goals = _model_goals(model, surface)

    meta_chips = [f'<span><strong>Fit:</strong> {html.escape(fit)}</span>']
    if context_label and context_label != "Not listed":
        meta_chips.append(f'<span><strong>Context:</strong> {html.escape(context_label)}</span>')
    if surface == "chat":
        reply = _chat_reply_limit_label(model)
        meta_chips.append(f'<span><strong>Reply default:</strong> {html.escape(reply)}</span>')
    if surface == "image":
        showcase = _clean(model.get("showcase_prompt"))
        if showcase:
            meta_chips.append('<span><strong>Showcase prompt:</strong> available</span>')

    settings_html = ""
    notes_html = ""
    negative_html = ""

    if surface == "image":
        settings, _locked = _image_settings(model)
        settings_html = '<div class="settings-grid">' + "".join(
            f'<div class="setting"><div class="lbl">{html.escape(k)}</div>'
            f'<div class="val">{html.escape(str(v))}</div></div>'
            for k, v in settings.items()
        ) + "</div>"
        tradeoff = _clean(model.get("tradeoff_note"))
        if tradeoff:
            notes_html = f'<div class="notes">{html.escape(tradeoff)}</div>'

        negative_text, is_image_gen = _negative_prompt_for_model(mid)
        if is_image_gen:
            if negative_text:
                neg_attr = _escape_attr(negative_text)
                negative_html = (
                    '<div class="neg-block">'
                    '<div class="neg-title-row">'
                    '<div class="neg-label">Negative prompt (shared)</div>'
                    f'<button type="button" class="copy-btn" data-copy="{neg_attr}">📋 Copy</button>'
                    '</div>'
                    f'<div class="neg-text" title="Click to copy" data-copy="{neg_attr}">{html.escape(negative_text)}</div>'
                    '</div>'
                )
            else:
                negative_html = (
                    '<div class="neg-block neg-locked">'
                    '<div class="neg-label">CFG-locked family — negative prompts ignored</div>'
                    '<p class="neg-note">This model family runs at <strong>CFG&nbsp;=&nbsp;1.0</strong> '
                    '(Flux / Z-Image / Turbo / Lightning / Chroma). Negative prompts are silently '
                    'ignored by the workflow. Use precise positive language instead.</p>'
                    '</div>'
                )

    utility_note_html = ""
    if mid in _UTILITY_DEMOS:
        utility_note_html = (
            '<div class="notes">Run from Toolbox — this utility model is surfaced through a '
            'Toolbox workflow, not the Chat tab.</div>'
        )

    # Surfaces whose "samples" are workflow instructions, not pasteable prompts.
    # We render them as advisory notes (no copy button, no "Demo prompt #" title).
    workflow_surfaces = {"speech", "embed", "doc"}

    prompts_html_parts = []
    for idx, sample in enumerate(samples):
        sample_text = _clean(sample) if isinstance(sample, str) else _clean(str(sample))
        if not sample_text:
            continue
        if surface in workflow_surfaces:
            prompts_html_parts.append(
                f'<div class="notes workflow-note">{html.escape(sample_text)}</div>'
            )
            continue
        thumb_url = _sample_thumbnail_url(mid, idx) if surface == "image" else None
        thumb_html = ""
        if thumb_url:
            thumb_html = (
                f'<a class="prompt-thumb-link" href="{html.escape(thumb_url)}" target="_blank" '
                f'rel="noopener" aria-label="Sample render for {_escape_attr(name)} — open full size">'
                f'<img class="prompt-thumb" src="{html.escape(thumb_url)}" alt="" loading="lazy" '
                f'decoding="async" width="140" height="140"></a>'
            )
        body_class = "prompt-card with-thumb" if thumb_html else "prompt-card"
        title = f"Demo prompt {idx + 1}"
        text_attr = _escape_attr(sample_text)
        prompts_html_parts.append(
            f'<div class="{body_class}">'
            f'{thumb_html}'
            f'<div class="prompt-body">'
            f'<div class="prompt-title-row">'
            f'<div class="prompt-title">{html.escape(title)}</div>'
            f'<button type="button" class="copy-btn" data-copy="{text_attr}">📋 Copy</button>'
            f'</div>'
            f'<div class="prompt-text" title="Click to copy" data-copy="{text_attr}">{html.escape(sample_text)}</div>'
            f'</div>'
            f'</div>'
        )
    prompts_html = '<div class="prompts">' + "".join(prompts_html_parts) + "</div>"

    # Search index includes the negative prompt + workflow notes so a phrase
    # buried in a negative ("worst quality") still surfaces the card.
    negative_text_for_search, _ = _negative_prompt_for_model(mid)
    search_blob = " ".join([
        mid, name, vendor, category, feature, why, " ".join(goals),
        " ".join(t for t in (model.get("tags") or [])),
        negative_text_for_search or "",
        " ".join(_clean(s) if isinstance(s, str) else _clean(str(s)) for s in samples),
    ]).lower()

    ram_tier = _card_ram_tier(model)
    ram_attr = f'data-ram="{ram_tier}" ' if ram_tier else ''
    return (
        f'<article class="model-card" id="{html.escape(fragment)}" '
        f'data-model-id="{_escape_attr(mid)}" {ram_attr}data-surface="{html.escape(surface)}" '
        f'data-vram="{int(min_vram_num)}" data-goals="{html.escape(" ".join(goals))}" '
        f'data-category="{_escape_attr(category)}" data-collapsed="true" '
        f'data-search="{_escape_attr(search_blob)}">'
        '<div class="card-head">'
        '<div>'
        f'<div class="eyebrow">{html.escape(surface_label)} · {html.escape(vendor)} · {html.escape(category)}</div>'
        f'<div class="model-name">{html.escape(name)}<span class="model-id">{html.escape(mid)}</span></div>'
        '</div>'
        '<div class="card-head-meta">'
        f'<span class="feature">{html.escape(feature)}</span>'
        f'<button type="button" class="card-expand" aria-expanded="false" aria-controls="body-{html.escape(slug or mid)}">'
        '<span class="chev" aria-hidden="true">▸</span><span class="lbl">Expand</span></button>'
        '</div></div>'
        f'<p class="why">{html.escape(why)}</p>'
        f'<div class="meta">{"".join(meta_chips)}</div>'
        f'<div class="card-body" id="body-{html.escape(slug or mid)}">'
        f'{settings_html}'
        f'{notes_html}'
        f'{utility_note_html}'
        f'{prompts_html}'
        f'{negative_html}'
        '</div>'
        '</article>'
    )


def _classify_models(models: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {sid: [] for sid, _l, _d in SURFACES}
    for model in models:
        if not _clean(model.get("id")):
            continue
        surface = _model_surface(model)
        grouped.setdefault(surface, []).append(model)
    return grouped


def _render_rail(grouped: dict[str, list[dict]]) -> str:
    parts = []
    for index, (sid, label, _desc) in enumerate(SURFACES):
        models = grouped.get(sid, [])
        if not models:
            continue
        # Open every group — the rail is the "Jump to a model" index. Hidden
        # groups would force keyboard users to expand each one before they can
        # tab to the link they want.
        open_attr = " open"
        links = "".join(
            f'<a href="#model-{html.escape(doc_fragment(_clean(m.get("id"))).removeprefix("model-"))}" '
            f'data-target="model-{html.escape(doc_fragment(_clean(m.get("id"))).removeprefix("model-"))}">'
            f'{html.escape(_clean(m.get("name")) or _clean(m.get("id")))}</a>'
            for m in models
        )
        parts.append(
            f'<details class="rail-group" data-surface="{html.escape(sid)}"{open_attr}>'
            f'<summary><span>{html.escape(label)}</span>'
            f'<span class="rail-count">{len(models)}</span></summary>'
            f'<div class="rail-group-children">{links}</div>'
            '</details>'
        )
    return "".join(parts)


def _render_surface_sections(grouped: dict[str, list[dict]]) -> str:
    parts = []
    for sid, label, desc in SURFACES:
        models = grouped.get(sid, [])
        if not models:
            continue
        cards = "".join(_render_card(m, sid) for m in models)
        parts.append(
            f'<section class="surface-section" data-surface="{html.escape(sid)}" '
            f'id="surface-{html.escape(sid)}">'
            f'<div class="surface-head">'
            f'<span class="pill">{html.escape(label)}</span>'
            f'<h2 id="{html.escape(sid)}">{html.escape(label)}</h2>'
            f'<span class="desc">{html.escape(desc)}</span>'
            '</div>'
            f'<div class="model-grid">{cards}</div>'
            '</section>'
        )
    return "".join(parts)


def _render_surface_tabs(grouped: dict[str, list[dict]], total: int) -> str:
    parts = [
        '<button class="surface-tab active" data-surface="all" type="button" '
        'role="tab" aria-selected="true" tabindex="0">'
        f'All <span class="count">{total}</span></button>'
    ]
    for sid, label, _desc in SURFACES:
        n = len(grouped.get(sid, []))
        if not n:
            continue
        parts.append(
            f'<button class="surface-tab" data-surface="{html.escape(sid)}" type="button" '
            f'role="tab" aria-selected="false" tabindex="-1">'
            f'{html.escape(label)} <span class="count">{n}</span></button>'
        )
    return "".join(parts)


CSS = r"""
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg:        #1a1a2e;
  --surface:   #16213e;
  --surface-2: #1c2a4a;
  --card:      #0f3460;
  --card-alt:  #12294f;
  --accent:    #4f9cf9;
  --accent-2:  #7ad7ff;
  --accent-soft: rgba(79,156,249,.14);
  --accent-fg: #0d1117;
  --good:      #56d364;
  --warn:      #ffb347;
  --bad:       #ff7b72;
  --purple:    #c89bff;
  --text:      #e7edf5;
  --text-soft: #c9d1d9;
  --muted:     #9aa7bb;
  --border:    #2a3b58;
  --border-strong: #3d5278;
  --code-bg:   #0c1320;
  --shadow:    0 18px 48px rgba(0,0,0,.42);
  --c-ultra:  #6ec6ff;
  --c-small:  #69f0ae;
  --c-medium: #ffd740;
  --c-large:  #ff9800;
  --c-xl:     #ff5252;
  --c-vision: #c89bff;
  --c-image:  #7ad7ff;
  --c-speech: #ffd740;
  --c-embed:  #69f0ae;
  --c-doc:    #ff9800;
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg:        #f1f4fb;
  --surface:   #ffffff;
  --surface-2: #eaf0fa;
  --card:      #ffffff;
  --card-alt:  #f3f7fd;
  --accent:    #1864c4;
  --accent-2:  #0f5099;
  --accent-soft: rgba(24,100,196,.10);
  --accent-fg: #ffffff;
  --good:      #1f8a3a;
  --warn:      #b76b00;
  --bad:       #b91c1c;
  --purple:    #6b3fb7;
  --text:      #121b2e;
  --text-soft: #2e3a52;
  --muted:     #5c6a82;
  --border:    #d4dbe8;
  --border-strong: #a7b3c6;
  --code-bg:   #f3f7fd;
  --shadow:    0 14px 38px rgba(15,80,153,.14);
  --c-ultra:  #1864c4;
  --c-small:  #1f8a3a;
  --c-medium: #b76b00;
  --c-large:  #d2691e;
  --c-xl:     #b91c1c;
  --c-vision: #6b3fb7;
  --c-image:  #0f5099;
  --c-speech: #b76b00;
  --c-embed:  #1f8a3a;
  --c-doc:    #b91c1c;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
/* docnav-a11y fix (2026-05-30): the docnav-a11y JS at end of body owns scroll
   math. Per-section `scroll-margin-top` already positions native-anchor
   landings; combining it with a non-zero `scroll-padding-top` here doubled the
   offset and pushed sections ~half a screen below the sticky header. */
html { scroll-behavior: smooth; scroll-padding-top: 0; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body { font-family: 'Segoe UI', Aptos, Calibri, system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; font-size: 14.5px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: var(--code-bg); color: var(--accent-2); padding: 2px 6px; border-radius: 4px; font-size: .82em; font-family: Consolas, 'Cascadia Code', monospace; }

/* Skip-link for keyboard users — only visible when focused. */
.skip-link {
  position: absolute; left: 12px; top: -100px;
  background: var(--accent); color: var(--accent-fg);
  padding: 8px 14px; border-radius: 6px; font-weight: 600; z-index: 300;
  transition: top .15s;
}
.skip-link:focus { top: 12px; outline: none; }

/* WCAG 2.4.11 focus-visible style for every interactive element. */
:where(button, a, input, select, summary, [tabindex]):focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 6px;
}
:where(button, a, input, select, summary, [tabindex]):focus:not(:focus-visible) {
  outline: 0;
}

header.top {
  background: linear-gradient(135deg, var(--card) 0%, var(--surface) 100%);
  padding: 28px 36px 22px;
  border-bottom: 2px solid var(--accent);
  position: sticky; top: 0; z-index: 50;
}
.top-row { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap; }
.title-block .back-link {
  display: inline-block; color: var(--accent); font-size: .8rem; font-weight: 600;
  text-decoration: none; margin-bottom: 6px; padding: 2px 0;
}
.title-block .back-link:hover { text-decoration: underline; }
.title-block h1 { font-size: 1.7rem; color: #fff; font-weight: 700; }
[data-theme="light"] .title-block h1 { color: var(--text); }
.title-block p { color: var(--muted); font-size: .92rem; margin-top: 4px; max-width: 760px; }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.theme-toggle { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; cursor: pointer; font: inherit; font-size: .85rem; }
.theme-toggle:hover { border-color: var(--accent); }

nav.filter-bar {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 14px 36px; position: sticky; top: var(--header-h, 92px); z-index: 40;
  display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 10px; align-items: end;
}
.filter-field label { display: block; color: var(--muted); font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px; }
.filter-field input, .filter-field select {
  width: 100%; background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 7px; padding: 8px 10px;
  font: inherit;
}
.filter-field input:focus, .filter-field select:focus { border-color: var(--accent); }

.shell { display: grid; grid-template-columns: 280px 1fr; gap: 0; min-height: calc(100vh - var(--sticky-top, 168px)); }

aside.rail {
  background: var(--surface); border-right: 1px solid var(--border);
  padding: 20px 16px; position: sticky; top: var(--sticky-top, 168px);
  height: calc(100vh - var(--sticky-top, 168px)); overflow-y: auto;
  scrollbar-color: var(--border) transparent;
}
.rail-label { color: var(--muted); font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; padding: 0 8px 6px; }
.rail-count { background: var(--surface-2); color: var(--muted); font-size: .7rem; padding: 1px 8px; border-radius: 999px; }
details.rail-group { margin: 0 0 4px; }
details.rail-group > summary {
  list-style: none; cursor: pointer; padding: 7px 10px; border-radius: 6px;
  font-weight: 600; font-size: .9rem; color: var(--text);
  display: flex; align-items: center; gap: 8px; justify-content: space-between;
}
details.rail-group > summary::-webkit-details-marker { display: none; }
details.rail-group > summary:hover { background: var(--accent-soft); }
details.rail-group > summary::before { content: "▸"; color: var(--accent); transition: transform .2s; font-size: .85em; }
details.rail-group[open] > summary::before { transform: rotate(90deg); }
.rail-group-children { padding: 2px 0 6px 16px; display: grid; gap: 1px; }
.rail-group-children a { display: block; padding: 5px 10px; color: var(--muted); font-size: .82rem; border-radius: 5px; text-decoration: none; }
.rail-group-children a:hover { color: var(--accent-2); background: var(--accent-soft); }

.surface-tabs {
  display: flex; padding: 0 36px; background: var(--bg);
  border-bottom: 1px solid var(--border); gap: 0; align-items: stretch;
  position: sticky; top: var(--tabs-top, 168px); z-index: 30;
  overflow-x: auto; scrollbar-width: thin;
}
.surface-tab { background: transparent; border: none; color: var(--muted); padding: 14px 18px; cursor: pointer; font: inherit; font-size: .92rem; font-weight: 500; border-bottom: 2px solid transparent; white-space: nowrap; }
.surface-tab:hover { color: var(--text); }
.surface-tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.surface-tab[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.surface-tab .count { color: var(--muted); font-size: .8em; margin-left: 4px; }

main.content { padding: 28px 36px 80px; min-width: 0; }

.density-bar {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  background: linear-gradient(135deg, var(--card) 0%, var(--surface) 100%);
  border: 1px solid var(--border); border-left: 4px solid var(--accent);
  border-radius: 0 10px 10px 0; padding: 11px 16px; margin: 0 0 18px;
  box-shadow: 0 4px 14px rgba(0,0,0,.18);
}
[data-theme="light"] .density-bar { box-shadow: 0 4px 14px rgba(15,80,153,.06); }
.density-count { color: var(--text-soft); font-size: .9rem; }
.density-count strong { color: var(--accent-2); font-weight: 700; font-size: 1.05em; }
.density-spacer { flex: 1; }
.density-btn {
  background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 14px; font: inherit; font-size: .82rem;
  font-weight: 500; cursor: pointer; transition: background .15s, border-color .15s, color .15s;
}
.density-btn:hover { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-2); }

.system-banner {
  display: flex; align-items: center; gap: 14px;
  background: var(--surface); border: 1px solid var(--border);
  border-left: 4px solid var(--good); border-radius: 0 10px 10px 0;
  padding: 12px 18px; margin: 0 0 18px;
}
.system-banner[hidden] { display: none; }
.system-banner-icon { font-size: 1.4rem; }
.system-banner-body { flex: 1; }
.system-banner-title { color: var(--text); font-weight: 600; font-size: .95rem; }
.system-banner-subtitle { color: var(--muted); font-size: .85rem; margin-top: 2px; }
.system-banner-clear { background: transparent; color: var(--accent-2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; cursor: pointer; font: inherit; font-size: .85rem; }
.system-banner-clear:hover { background: var(--surface-2); border-color: var(--accent); }

.surface-section { margin: 0 0 36px; scroll-margin-top: var(--sticky-top, 168px); }
.surface-section[hidden] { display: none; }
.surface-head { display: flex; align-items: center; gap: 12px; margin: 24px 0 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.surface-head .pill { background: var(--accent); color: var(--accent-fg); font-size: .72rem; font-weight: 700; padding: 3px 11px; border-radius: 999px; text-transform: uppercase; letter-spacing: .05em; }
.surface-head h2 { font-size: 1.4rem; color: var(--text); }
.surface-head .desc { color: var(--muted); font-size: .88rem; margin-left: auto; }

.surface-section[data-surface="chat"]   .pill { background: var(--c-ultra);  color: var(--accent-fg); }
.surface-section[data-surface="vision"] .pill { background: var(--c-vision); color: var(--accent-fg); }
.surface-section[data-surface="image"]  .pill { background: var(--c-image);  color: var(--accent-fg); }
.surface-section[data-surface="speech"] .pill { background: var(--c-speech); color: var(--accent-fg); }
.surface-section[data-surface="embed"]  .pill { background: var(--c-embed);  color: var(--accent-fg); }
.surface-section[data-surface="doc"]    .pill { background: var(--c-doc);    color: var(--accent-fg); }

.model-grid { display: grid; gap: 14px; }
.model-card {
  background: var(--card); border: 1px solid var(--border);
  border-left: 4px solid var(--accent); border-radius: 12px;
  padding: 18px 22px; transition: box-shadow .15s, border-color .15s;
  scroll-margin-top: var(--sticky-top, 168px);
}
.surface-section[data-surface="chat"]   .model-card { border-left-color: var(--c-ultra); }
.surface-section[data-surface="vision"] .model-card { border-left-color: var(--c-vision); }
.surface-section[data-surface="image"]  .model-card { border-left-color: var(--c-image); }
.surface-section[data-surface="speech"] .model-card { border-left-color: var(--c-speech); }
.surface-section[data-surface="embed"]  .model-card { border-left-color: var(--c-embed); }
.surface-section[data-surface="doc"]    .model-card { border-left-color: var(--c-doc); }
.model-card:hover { box-shadow: var(--shadow); }
/* Target highlight fades after a few seconds so it stops being a permanent
   false-landmark. Driven by .target-fade class added by JS. */
.model-card.target { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); transition: box-shadow .8s ease-out, border-color .8s ease-out; }
.model-card.target.target-fade { border-color: var(--border); box-shadow: none; }
.model-card.hidden { display: none; }

.model-card[data-collapsed="true"] .card-body { display: none; }
.model-card[data-collapsed="true"] { padding-top: 14px; padding-bottom: 14px; }
.card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; }
.card-head-meta { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.card-expand {
  background: transparent; color: var(--accent-2);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 5px 10px; font: inherit; font-size: .78rem; font-weight: 600;
  cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
  white-space: nowrap;
}
.card-expand:hover { background: var(--accent-soft); border-color: var(--accent); }
.card-expand .chev { display: inline-block; transition: transform .2s; font-size: .9em; line-height: 1; }
.model-card:not([data-collapsed="true"]) .card-expand .chev { transform: rotate(90deg); }
/* Expand/Collapse label text lives directly in .lbl so screen readers expose
   it as the button's accessible name (CSS ::after content isn't reliably
   announced by NVDA/Narrator/VoiceOver). The text is updated in JS by
   setCardCollapsed(). */
.eyebrow { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 2px; }
.model-name { color: var(--text); font-size: 1.15rem; font-weight: 700; }
.model-id { color: var(--muted); font-weight: 400; font-size: .85rem; margin-left: 6px; }
.feature {
  background: var(--accent-soft); color: var(--accent-2); border: 1px solid var(--border);
  border-radius: 999px; padding: 4px 12px; font-size: .76rem; font-weight: 600; white-space: nowrap;
}
.why { color: var(--text-soft); font-size: .9rem; margin: 6px 0 10px; max-width: 780px; }

.meta { display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted); font-size: .78rem; margin-bottom: 10px; }
.meta span { background: var(--surface-2); padding: 3px 9px; border-radius: 5px; border: 1px solid var(--border); }
.meta strong { color: var(--text-soft); font-weight: 500; }

.settings-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 7px; margin: 10px 0 12px;
}
.setting { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; }
.setting .lbl { color: var(--muted); font-size: .68rem; text-transform: uppercase; letter-spacing: .04em; }
.setting .val { color: var(--text); font-size: .85rem; margin-top: 1px; }
.notes { background: var(--surface); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; padding: 8px 12px; font-size: .82rem; color: var(--muted); margin-bottom: 10px; }
/* workflow-note is the "this isn't a pasteable prompt, it's an instruction" variant
   used for Speech/Embeddings/Document AI cards. Slightly bigger / accent-tinted
   text so users don't try to copy-paste it. */
.notes.workflow-note { color: var(--text-soft); font-size: .9rem; line-height: 1.55; background: var(--card-alt); border-left-color: var(--accent-2); }

.prompts { display: grid; gap: 10px; }
.prompt-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
.prompt-card.with-thumb { display: flex; gap: 12px; align-items: stretch; }
.prompt-thumb-link { flex: 0 0 auto; display: block; line-height: 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--surface-2); }
.prompt-thumb-link:hover { border-color: var(--accent); }
.prompt-thumb { width: 140px; height: 140px; object-fit: cover; cursor: zoom-in; display: block; }
.prompt-body { flex: 1 1 auto; min-width: 0; }
.prompt-title-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 6px; }
.prompt-title { color: var(--text); font-size: .8rem; font-weight: 600; }
.prompt-text {
  background: var(--code-bg); color: var(--text-soft); font-family: Consolas, 'Cascadia Code', monospace;
  font-size: .82rem; line-height: 1.55; padding: 10px 12px; border-radius: 6px;
  border: 1px solid var(--border); cursor: pointer; word-break: break-word; white-space: pre-wrap;
  transition: border-color .15s, background .15s;
}
.prompt-text:hover { border-color: var(--accent); }
.prompt-text.copied { border-color: var(--good); background: rgba(86,211,100,.08); }
.copy-btn {
  background: var(--surface-2); color: var(--accent-2);
  border: 1px solid var(--border); border-radius: 5px;
  padding: 3px 10px; font: inherit; font-size: .72rem; font-weight: 600;
  cursor: pointer; white-space: nowrap; display: inline-flex; align-items: center; gap: 4px;
  transition: background .15s, border-color .15s, color .15s;
}
.copy-btn:hover { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
.copy-btn.copied { background: var(--good); color: #0d1117; border-color: var(--good); }
.neg-block { margin-top: 12px; background: var(--neg-bg, rgba(255,123,114,.10)); border: 1px solid var(--neg-border, rgba(255,123,114,.30)); border-left: 3px solid var(--bad); border-radius: 0 8px 8px 0; padding: 10px 12px; }
.neg-title-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 6px; }
.neg-label { display: inline-flex; align-items: center; gap: 6px; color: var(--bad); font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.neg-label::before { content: "⛔"; font-size: 1em; }
.neg-text {
  background: var(--code-bg); color: var(--text-soft); font-family: Consolas, 'Cascadia Code', monospace;
  font-size: .8rem; line-height: 1.5; padding: 9px 12px; border-radius: 6px; border: 1px solid var(--border);
  cursor: pointer; white-space: pre-wrap; word-break: break-word;
  transition: border-color .15s, background .15s;
}
.neg-text:hover { border-color: var(--bad); }
.neg-text.copied { border-color: var(--good); background: rgba(86,211,100,.08); }
.neg-locked .neg-note { color: var(--text-soft); font-size: .82rem; margin-top: 4px; }
[data-theme="light"] .neg-block { --neg-bg: rgba(185,28,28,.07); --neg-border: rgba(185,28,28,.22); }

.empty-state {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 30px; text-align: center; color: var(--muted); margin: 20px 0;
}
.empty-state[hidden] { display: none; }

/* Toast sits above the back-to-top FAB so they never overlap. */
#toast {
  position: fixed; bottom: 76px; left: 50%; transform: translateX(-50%) translateY(20px);
  background: var(--good); color: #0d1117; padding: 10px 22px; border-radius: 999px;
  font-weight: 600; font-size: .85rem; opacity: 0; pointer-events: none;
  transition: opacity .2s, transform .2s; z-index: 200;
}
#toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
#back-to-top {
  position: fixed; bottom: 24px; right: 24px;
  background: var(--accent); color: var(--accent-fg); border: none;
  padding: 11px 18px; border-radius: 999px; cursor: pointer;
  font-weight: 600; box-shadow: var(--shadow); opacity: 0; pointer-events: none;
  transition: opacity .2s, transform .2s; z-index: 200;
}
#back-to-top.show { opacity: 1; pointer-events: auto; }

/* === SKU filter strip (header) === */
.sku-strip {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 10px 0 4px; margin-top: 14px;
  border-top: 1px solid var(--border-soft, var(--border));
}
.sku-label {
  color: var(--muted); font-size: .7rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .07em; margin-right: 4px;
  white-space: nowrap;
}
.sku-chip {
  display: inline-flex; align-items: center; gap: 7px;
  background: var(--surface-2, var(--surface)); color: var(--text);
  border: 1px solid var(--border); border-radius: 999px;
  padding: 5px 12px; font: inherit; font-size: .82rem; font-weight: 600;
  cursor: pointer; transition: background .15s, border-color .15s, color .15s;
  white-space: nowrap;
}
.sku-chip:hover { border-color: var(--accent); color: var(--accent-2); }
.sku-chip.active {
  background: var(--accent); color: var(--accent-fg);
  border-color: var(--accent);
}
.sku-chip .sku-sub {
  color: var(--muted); font-weight: 500; font-size: .76rem;
  margin-left: 2px;
}
.sku-chip.active .sku-sub { color: var(--accent-fg); opacity: .85; }
.sku-chip .sku-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--c-cpu, var(--muted)); display: inline-block;
}
.sku-chip[data-sku="all"]    .sku-dot { background: transparent; border: 1px dashed var(--muted); }
.sku-chip[data-sku="cpu16"]  .sku-dot,
.sku-chip[data-sku="cpu32"]  .sku-dot,
.sku-chip[data-sku="cpu64"]  .sku-dot { background: #b07a3c; }
.sku-chip[data-sku="4"]      .sku-dot { background: #4f9cf9; }
.sku-chip[data-sku="8"]      .sku-dot { background: #7ad7ff; }
.sku-chip[data-sku="12"]     .sku-dot { background: #4dd0e1; }
.sku-chip[data-sku="16"]     .sku-dot { background: #4dd0e1; }
.sku-chip[data-sku="24"]     .sku-dot { background: #4caf50; }
.sku-chip.active .sku-dot { border-color: var(--accent-fg); }

@media (max-width: 900px) {
  .shell { grid-template-columns: 1fr; }
  aside.rail { display: none; }
  nav.filter-bar { grid-template-columns: 1fr 1fr; padding: 12px 16px; }
  header.top, main.content { padding-left: 16px; padding-right: 16px; }
  .title-block p { display: none; } /* keep header compact so sticky stack fits */
  .prompt-card.with-thumb { flex-direction: column; }
  .prompt-thumb { width: 100%; max-width: 240px; height: auto; }
  .sku-strip { gap: 6px; padding: 8px 0 2px; }
  .sku-chip { padding: 4px 10px; font-size: .78rem; }
  .sku-label { display: none; }
}

/* ── Compact header mode ──────────────────────────────────────────────────
   Toggled via #compact-toggle. State persists to localStorage and is
   applied by HEAD_SCRIPT before first paint so there is no flash of the
   tall header. The sticky-offset ResizeObserver picks the new height up
   automatically, so --header-h / --tabs-top / --sticky-top stay correct
   and rail-click + deep-link scrolling lands at the right place.

   Compact mode hides the subtitle, shrinks the H1 inline with the
   back-link + controls, trims header padding, and condenses the SKU
   chip strip. Estimated savings: ~160px of sticky stack height.        */
[data-compact="1"] header.top { padding: 6px 36px 6px; }
[data-compact="1"] .top-row { align-items: center; gap: 12px; }
[data-compact="1"] .title-block { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
[data-compact="1"] .title-block .back-link { margin-bottom: 0; font-size: .76rem; }
[data-compact="1"] .title-block h1 { font-size: 1.05rem; font-weight: 600; }
[data-compact="1"] .title-block p { display: none; }
[data-compact="1"] .sku-strip { padding: 4px 0 0; gap: 6px; }
[data-compact="1"] .sku-label { display: none; }
[data-compact="1"] .sku-chip { padding: 4px 10px; font-size: .78rem; }
[data-compact="1"] .sku-chip .sku-sub { display: none; }
.compact-toggle { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; cursor: pointer; font: inherit; font-size: .85rem; }
.compact-toggle:hover { border-color: var(--accent); }
"""


JS = r"""
const SURFACE_LABEL = {
  all: 'all surfaces', chat: 'Chat', vision: 'Vision', image: 'Image generation',
  speech: 'Speech', embed: 'Embeddings', doc: 'Document AI'
};

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function setCardCollapsed(card, collapsed) {
  card.dataset.collapsed = collapsed ? 'true' : 'false';
  const btn = card.querySelector('.card-expand');
  if (btn) {
    btn.setAttribute('aria-expanded', String(!collapsed));
    const lbl = btn.querySelector('.lbl');
    if (lbl) lbl.textContent = collapsed ? 'Expand' : 'Collapse';
  }
}

function wireCardExpanders() {
  document.querySelectorAll('.card-expand').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const card = btn.closest('.model-card');
      const isCollapsed = card.dataset.collapsed === 'true';
      setCardCollapsed(card, !isCollapsed);
    });
  });
}

function expandAllVisible() {
  document.querySelectorAll('.model-card:not(.hidden)').forEach(c => setCardCollapsed(c, false));
}

function collapseAllVisible() {
  document.querySelectorAll('.model-card:not(.hidden)').forEach(c => setCardCollapsed(c, true));
}

let toastTimer = null;
function showToast(msg) {
  const t = document.querySelector('#toast');
  if (!t) return;
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 1300);
}

function copyText(text, el) {
  navigator.clipboard.writeText(text).then(() => {
    if (el) { el.classList.add('copied'); setTimeout(() => el.classList.remove('copied'), 1200); }
    showToast('Copied!');
  });
}

function wireCopyButtons() {
  document.querySelectorAll('.prompt-text, .neg-text').forEach(el => {
    el.addEventListener('click', () => copyText(el.dataset.copy || '', el));
  });
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const body = btn.closest('.prompt-body, .neg-block');
      const target = body ? body.querySelector('.prompt-text, .neg-text') : btn;
      copyText(btn.dataset.copy || '', target);
      btn.classList.add('copied');
      setTimeout(() => btn.classList.remove('copied'), 1200);
    });
  });
}

function applyFilters(opts) {
  const fromSearch = !!(opts && opts.fromSearch);
  const q = (document.querySelector('#q').value || '').trim().toLowerCase();
  const surface = document.querySelector('#surface-filter').value;
  const hw = document.querySelector('#hardware').value;
  const goal = document.querySelector('#goal').value;
  const searchActive = q.length >= 2;

  let visible = 0;
  document.querySelectorAll('.model-card').forEach(card => {
    const matchSurface = surface === 'all' || card.dataset.surface === surface;
    const vram = parseInt(card.dataset.vram, 10) || 0;
    const ram  = parseInt(card.dataset.ram,  10) || 0;
    let matchHw = true;
    if (hw === 'cpu') matchHw = vram === 0;
    else if (hw.indexOf('cpu') === 0) matchHw = ram > 0 && ram <= parseInt(hw.slice(3), 10);
    else if (hw !== 'all') matchHw = vram <= parseInt(hw, 10);
    const matchGoal = goal === 'all' || (card.dataset.goals || '').split(' ').includes(goal);
    const matchSearch = !q || (card.dataset.search || '').includes(q);
    const show = matchSurface && matchHw && matchGoal && matchSearch;
    card.classList.toggle('hidden', !show);
    if (show) {
      visible++;
      // Only auto-expand on search input, never on hardware/goal/surface change.
      if (searchActive && fromSearch) setCardCollapsed(card, false);
    }
  });

  document.querySelectorAll('.surface-section').forEach(sec => {
    const any = sec.querySelector('.model-card:not(.hidden)');
    sec.hidden = !any || (surface !== 'all' && sec.dataset.surface !== surface);
  });

  document.querySelector('#empty-state').hidden = visible > 0;
  document.querySelector('#visible-count').textContent = visible;

  document.querySelectorAll('.surface-tab').forEach(t => {
    const active = t.dataset.surface === surface;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', String(active));
    t.setAttribute('tabindex', active ? '0' : '-1');
  });

  updateSystemBanner(hw, goal);
  updateTabsOverflow();
  syncSKUChips(hw);
  gateGoalOptions(hw);
}

function debounce(fn, ms) {
  let t = null;
  return function debounced() {
    const args = arguments;
    clearTimeout(t);
    t = setTimeout(() => fn.apply(null, args), ms);
  };
}

function updateSystemBanner(hw, goal) {
  const sb = document.querySelector('#system-banner');
  if (!sb) return;
  const sub = document.querySelector('#sb-subtitle');
  if (hw === 'all' && goal === 'all') { sb.hidden = true; return; }
  const parts = [];
  if (hw === 'cpu') parts.push('CPU only');
  else if (hw.indexOf('cpu') === 0) parts.push('CPU · ≤ ' + hw.slice(3) + ' GB RAM');
  else if (hw !== 'all') parts.push('≤ ' + hw + ' GB VRAM');
  if (goal !== 'all') {
    const goalText = document.querySelector('#goal option:checked').textContent;
    parts.push('Goal: ' + goalText);
  }
  sub.textContent = parts.join(' · ');
  sb.hidden = false;
}

function normalize(value) {
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function findCardFromValue(raw) {
  let decoded = '';
  try { decoded = decodeURIComponent(raw || ''); } catch (_) { decoded = raw || ''; }
  const target = normalize(decoded);
  if (!target) return null;
  const cards = Array.from(document.querySelectorAll('.model-card'));
  return cards.find(card =>
    normalize(card.dataset.modelId) === target ||
    normalize(card.id) === target ||
    normalize((card.id || '').replace(/^model-/, '')) === target
  );
}

/* Keep URL deep links reliable even when incoming hardware/surface/goal
   params would otherwise hide the target card. App entry points pass
   system filters by design; deep-link model context must win if there is a
   conflict so users always land on the requested card. */
function ensureDeepLinkCardVisible(card) {
  if (!card) return;
  const surfaceSel = document.querySelector('#surface-filter');
  const hwSel = document.querySelector('#hardware');
  const goalSel = document.querySelector('#goal');
  if (!surfaceSel || !hwSel || !goalSel) return;

  const cardSurface = card.dataset.surface || '';
  const cardVram = parseInt(card.dataset.vram, 10) || 0;
  const cardGoals = (card.dataset.goals || '').split(' ').filter(Boolean);

  if (surfaceSel.value !== 'all' && surfaceSel.value !== cardSurface) {
    surfaceSel.value = 'all';
  }

  const hw = hwSel.value;
  if (hw === 'cpu' && cardVram !== 0) {
    hwSel.value = 'all';
  } else if (hw.indexOf('cpu') === 0) {
    const cardRam = parseInt(card.dataset.ram, 10) || 0;
    const hwLimit = parseInt(hw.slice(3), 10) || 0;
    if (!cardRam || !hwLimit || cardRam > hwLimit) hwSel.value = 'all';
  } else if (hw !== 'all') {
    const hwLimit = parseInt(hw, 10) || 0;
    if (!hwLimit || cardVram > hwLimit) hwSel.value = 'all';
  }

  if (goalSel.value !== 'all' && !cardGoals.includes(goalSel.value)) {
    goalSel.value = 'all';
  }
}

let targetFadeTimer = null;
function focusDeepLinkCard(card) {
  setCardCollapsed(card, false);
  card.classList.add('target');
  // Same explicit-math + double-rAF scroll as wireRailLinks so deep-link
  // navigation (URL params, hash) lands the card just below the sticky
  // stack instead of partially hidden behind it.
  scrollCardIntoView(card);
  // Move keyboard focus into the card so AT / keyboard users can interact.
  card.setAttribute('tabindex', '-1');
  try { card.focus({ preventScroll: true }); } catch (_) { card.focus(); }
  // Fade the target ring after 3.5s so it stops being a permanent landmark.
  clearTimeout(targetFadeTimer);
  targetFadeTimer = setTimeout(() => {
    card.classList.add('target-fade');
    setTimeout(() => { card.classList.remove('target', 'target-fade'); }, 900);
  }, 3500);
}

function applyUrlParams() {
  const p = new URLSearchParams(location.search);
  const hw = p.get('hardware') || p.get('vram');
  if (hw) {
    let v = null;
    if (hw === 'cpu' || hw === 'cpu16' || hw === 'cpu32' || hw === 'cpu64') v = hw;
    else if (parseInt(hw, 10)) v = String(parseInt(hw, 10));
    if (v) document.querySelector('#hardware').value = v;
  }
  const surf = p.get('surface');
  if (surf) document.querySelector('#surface-filter').value = surf;
  const goal = p.get('goal');
  if (goal) document.querySelector('#goal').value = goal;

  const raw = p.get('modelId') || p.get('model') || p.get('chatModel') ||
              p.get('imageModel') || p.get('ollama') || p.get('ollamaTag') ||
              (window.location.hash || '').replace(/^#/, '').replace(/^model-/, '');
  if (raw) {
    const card = findCardFromValue(raw);
    if (card) ensureDeepLinkCardVisible(card);
  }
  applyFilters();
  if (raw) {
    setTimeout(() => {
      const card = findCardFromValue(raw);
      if (card) focusDeepLinkCard(card);
    }, 100);
  }
}

function handleHashDeepLink() {
  const raw = (window.location.hash || '').replace(/^#/, '').replace(/^model-/, '');
  if (!raw) return;
  const card = findCardFromValue(raw);
  if (!card) return;
  ensureDeepLinkCardVisible(card);
  applyFilters();
  focusDeepLinkCard(card);
}

function wireRailLinks() {
  document.querySelectorAll('aside.rail a[data-target]').forEach(a => {
    a.addEventListener('click', (ev) => {
      const card = document.getElementById(a.dataset.target);
      if (!card) return;
      // If the active surface filter would hide this card, switch back to
      // "All" so the rail link always lands on something visible. Without
      // this guard, clicking (for example) "Playground" while the "Vision"
      // tab is active is a silent no-op — the card stays display:none and
      // the browser anchor jump goes nowhere.
      const cardSurface = card.dataset.surface || '';
      const surfaceSel = document.querySelector('#surface-filter');
      if (surfaceSel && surfaceSel.value !== 'all' && surfaceSel.value !== cardSurface) {
        surfaceSel.value = 'all';
        applyFilters();
      }
      setCardCollapsed(card, false);
      // Prevent the default anchor jump because our smooth-scroll below is
      // the correct landing — the default jump would happen first and feel
      // jittery.
      ev.preventDefault();
      // Double rAF + explicit math: the previous single-rAF + scrollIntoView
      // version could land the card behind the sticky stack because (a) the
      // card-expand reflow from setCardCollapsed and (b) the surface-filter
      // reflow from applyFilters() above could both shift the card's
      // position after the rAF measurement was taken. Two rAFs guarantee
      // both reflows have settled before we read the rect; explicit math
      // using window.scrollY + getBoundingClientRect() honors the actual
      // runtime --sticky-top instead of relying on the browser to combine
      // scroll-margin + smooth-scroll correctly under sticky positioning.
      scrollCardIntoView(card);
    });
  });
}

/* Scroll the page so `card` lands just below the sticky stack.
   Always uses the runtime --sticky-top CSS variable (kept in sync by
   wireStickyOffsets' ResizeObserver) so the math is correct in compact
   header mode, after window resize, after the subtitle wraps, etc. */
function scrollCardIntoView(card) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const stickyTopRaw = getComputedStyle(document.documentElement)
      .getPropertyValue('--sticky-top').trim();
    const stickyTop = parseInt(stickyTopRaw, 10) || 168;
    const rect = card.getBoundingClientRect();
    const targetTop = rect.top + window.scrollY - stickyTop - 8;
    window.scrollTo({ top: Math.max(0, targetTop), behavior: 'smooth' });
  }));
}

function wireSurfaceTabs() {
  document.querySelectorAll('.surface-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const sid = btn.dataset.surface;
      document.querySelector('#surface-filter').value = sid;
      applyFilters();
      // After the filter applies, scroll the left rail so the matching
      // group lands near the top of the rail viewport. This keeps the
      // rail and the surface tabs in visual sync — without it the rail
      // stays scrolled to whatever group was previously visible, which
      // makes the rail feel disconnected from the tabs above.
      scrollRailToSurface(sid);
    });
  });
}

/* Scroll the left rail so the requested surface group lands at the top.
   When sid === 'all', scroll the rail back to its very top. */
function scrollRailToSurface(sid) {
  const rail = document.querySelector('aside.rail');
  if (!rail) return;
  if (sid === 'all') {
    rail.scrollTo({ top: 0, behavior: 'smooth' });
    return;
  }
  const group = rail.querySelector('details.rail-group[data-surface="' + sid + '"]');
  if (!group) return;
  if (!group.open) group.open = true;
  // Align the group's <summary> to the top of the rail viewport, accounting
  // for the rail's own padding. Using offsetTop relative to the scrolling
  // container (the rail itself, since it has overflow-y:auto) is the most
  // reliable cross-browser anchor.
  const targetTop = group.offsetTop - 8;
  rail.scrollTo({ top: Math.max(0, targetTop), behavior: 'smooth' });
}

/* SKU chips above the search bar mirror the #hardware dropdown.
   - Clicking a chip sets #hardware and re-applies filters.
   - applyFilters() keeps chip active state in sync (so the chips also reflect
     changes made via the dropdown).
   - Picking the CPU chip hides image-gen goals from the Goal dropdown, since
     CPU-only SKUs don't have a usable image-gen path. Other SKUs show
     all goals. */
const IMAGE_GEN_GOALS = ['photo','fast','anime','art'];

function wireSKUChips() {
  document.querySelectorAll('.sku-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const hw = chip.dataset.sku || 'all';
      const sel = document.querySelector('#hardware');
      sel.value = hw;
      applyFilters();
    });
  });
}

function syncSKUChips(hw) {
  document.querySelectorAll('.sku-chip').forEach(c => {
    const active = (c.dataset.sku || 'all') === hw;
    c.classList.toggle('active', active);
    c.setAttribute('aria-pressed', String(active));
  });
}

function gateGoalOptions(hw) {
  const sel = document.querySelector('#goal');
  if (!sel) return;
  const hideImage = hw === 'cpu' || hw.indexOf('cpu') === 0;
  let resetNeeded = false;
  Array.from(sel.options).forEach(opt => {
    const isImage = IMAGE_GEN_GOALS.includes(opt.value);
    const shouldHide = hideImage && isImage;
    opt.hidden = shouldHide;
    opt.disabled = shouldHide;
    if (shouldHide && sel.value === opt.value) resetNeeded = true;
  });
  if (resetNeeded) sel.value = 'all';
}

function wireThemeToggle() {
  const btn = document.querySelector('#theme-toggle');
  // Label shows the ACTION (what clicking will do), not the current state.
  const update = () => {
    const t = document.documentElement.getAttribute('data-theme');
    btn.textContent = t === 'dark' ? '☀ Light' : '🌙 Dark';
    btn.setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  };
  update();
  btn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
    update();
  });
}

function wireBackToTop() {
  const btn = document.querySelector('#back-to-top');
  window.addEventListener('scroll', () => {
    btn.classList.toggle('show', window.scrollY > 400);
  });
  btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
}

/* Compact-header toggle: shrinks the hero so cards get more screen
   real-estate. HEAD_SCRIPT restores the saved state before first paint to
   avoid a flash. Toggling adds/removes [data-compact="1"] on <html>; the
   sticky-offset ResizeObserver picks the new header height up automatically
   so --header-h / --tabs-top / --sticky-top all stay correct. */
function wireCompactToggle() {
  const btn = document.querySelector('#compact-toggle');
  if (!btn) return;
  const sync = () => {
    const on = document.documentElement.getAttribute('data-compact') === '1';
    btn.setAttribute('aria-pressed', String(on));
    btn.textContent = on ? '▴ Expand' : '▾ Compact';
  };
  sync();
  btn.addEventListener('click', () => {
    const on = document.documentElement.getAttribute('data-compact') === '1';
    if (on) {
      document.documentElement.removeAttribute('data-compact');
      try { localStorage.removeItem('localai-model-guide-compact'); } catch (_) {}
    } else {
      document.documentElement.setAttribute('data-compact', '1');
      try { localStorage.setItem('localai-model-guide-compact', '1'); } catch (_) {}
    }
    sync();
  });
}

/* Drive the --header-h and --sticky-top CSS variables from the actual rendered
   heights so the multi-bar sticky stack (header → filter-bar → surface-tabs)
   stays aligned when the header subtitle wraps at narrow widths. */
function wireStickyOffsets() {
  const header = document.querySelector('header.top');
  const filterBar = document.querySelector('nav.filter-bar');
  const surfaceTabs = document.querySelector('nav.surface-tabs');
  if (!header || !filterBar || !surfaceTabs) return;
  const update = () => {
    const headerH = Math.round(header.getBoundingClientRect().height);
    const filterH = Math.round(filterBar.getBoundingClientRect().height);
    const tabsH = Math.round(surfaceTabs.getBoundingClientRect().height);
    document.documentElement.style.setProperty('--header-h', headerH + 'px');
    document.documentElement.style.setProperty('--tabs-top',  (headerH + filterH) + 'px');
    document.documentElement.style.setProperty('--sticky-top', (headerH + filterH + tabsH) + 'px');
  };
  if ('ResizeObserver' in window) {
    const obs = new ResizeObserver(update);
    obs.observe(header);
    obs.observe(filterBar);
    obs.observe(surfaceTabs);
  } else {
    window.addEventListener('resize', update);
  }
  update();
}

function updateTabsOverflow() {
  // Intentionally a no-op for now. Reserved for the optional scroll-fade hint
  // (designer-review R2). Kept as a hook so wireSurfaceTabs / window resize
  // listeners can call it without conditional branching.
}

function init() {
  wireStickyOffsets();
  wireSurfaceTabs();
  wireSKUChips();
  wireRailLinks();
  wireCardExpanders();
  wireCopyButtons();
  wireThemeToggle();
  wireBackToTop();
  wireCompactToggle();

  // Debounce search so a fast typist doesn't trigger N expand+reflow passes per keystroke.
  document.querySelector('#q').addEventListener('input', debounce(() => applyFilters({ fromSearch: true }), 250));
  ['surface-filter','hardware','goal'].forEach(id =>
    document.querySelector('#' + id).addEventListener('change', () => applyFilters())
  );
  document.querySelector('#expand-all').addEventListener('click', expandAllVisible);
  document.querySelector('#collapse-all').addEventListener('click', collapseAllVisible);
  const sbClear = document.querySelector('#sb-clear');
  if (sbClear) sbClear.addEventListener('click', () => {
    document.querySelector('#hardware').value = 'all';
    document.querySelector('#goal').value = 'all';
    applyFilters();
  });

  const tabs = document.querySelector('nav.surface-tabs');
  if (tabs) tabs.addEventListener('scroll', updateTabsOverflow);
  window.addEventListener('resize', updateTabsOverflow);
  window.addEventListener('hashchange', handleHashDeepLink);

  applyUrlParams();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
"""


HEAD_SCRIPT = """
(() => {
  // Cool-blue dark is the default. The user can override with ?clawpilotTheme=light|dark
  // or by toggling the button. We do NOT auto-follow prefers-color-scheme: the cool-blue
  // dark theme is the brand default, but we honour an explicit query param.
  const param = new URLSearchParams(window.location.search).get('clawpilotTheme');
  if (param === 'light' || param === 'dark') {
    document.documentElement.setAttribute('data-theme', param);
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
  // Restore compact-header state from localStorage BEFORE first paint so
  // the tall hero doesn't flash before collapsing.
  try {
    if (localStorage.getItem('localai-model-guide-compact') === '1') {
      document.documentElement.setAttribute('data-compact', '1');
    }
  } catch (_) { /* localStorage may be blocked in some embed contexts */ }
})();
"""


# ===================================================================
# Rail / Table-of-Contents navigation accessibility + behaviour fix.
# Added 2026-05-30 by docs-nav fix pass. Self-contained and removable
# (guarded by window.__docNavA11y). Fixes across every doc page that
# includes it:
#   1. Clicking a rail item now scrolls the matching section to just
#      below the sticky header (used to land ~half a screen too low
#      because scroll-padding-top + scroll-margin-top doubled the
#      offset).
#   2. The clicked rail item immediately becomes the active item
#      (and aria-current="true") instead of lagging behind.
#   3. Full keyboard support: Tab/Shift+Tab between rail items,
#      Up/Down/Left/Right + Home/End to move focus, Enter or Space
#      to jump to a section.
#   4. Screen-reader semantics: rail is exposed as a navigation
#      landmark, the active item carries aria-current, and focus is
#      moved to the target section so AT announces it.
#   5. Works identically in light and dark themes (uses theme vars).
# Mirrors docs/index.html, docs/image-gen-guide.html,
# docs/model-summary.html, and the source-of-truth snippet at
# Hackathon/docs-nav-accessibility-fix.snippet.html.
# ===================================================================
DOCNAV_A11Y = r"""<style id="docnav-a11y-style">
  html { scroll-padding-top: 0 !important; }
  #rail a:focus-visible,
  aside.rail a:focus-visible {
    outline: 2px solid var(--accent, #4ea1ff);
    outline-offset: 2px;
    border-radius: 6px;
  }
  [tabindex="-1"]:focus,
  [tabindex="-1"]:focus-visible {
    outline: none !important;
    box-shadow: none !important;
  }
  #rail a[aria-current="true"],
  aside.rail a[aria-current="true"] { font-weight: 600; }
</style>
<script id="docnav-a11y-script">
(() => {
  if (window.__docNavA11y) return;
  window.__docNavA11y = true;

  const rail = document.querySelector('#rail') || document.querySelector('aside.rail');
  if (!rail) return;
  if (!rail.getAttribute('role')) rail.setAttribute('role', 'navigation');
  if (!rail.getAttribute('aria-label')) rail.setAttribute('aria-label', 'On this page');

  const links = Array.from(rail.querySelectorAll('a[href^="#"]')).filter(a => {
    const id = decodeURIComponent((a.getAttribute('href') || '').slice(1));
    return id && document.getElementById(id);
  });
  if (!links.length) return;

  const root = document.documentElement;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function offset() {
    const cs = getComputedStyle(root);
    let v = parseInt(cs.getPropertyValue('--scroll-offset'), 10);
    if (!v) v = parseInt(cs.getPropertyValue('--sticky-top'), 10);
    if (!v) {
      document.querySelectorAll('header, .toolbar, .filter-bar, .surface-tabs, nav').forEach(el => {
        const p = getComputedStyle(el).position;
        const r = el.getBoundingClientRect();
        if ((p === 'sticky' || p === 'fixed') && r.top <= 1 && r.bottom > 0) {
          v = Math.max(v || 0, Math.round(r.bottom));
        }
      });
    }
    return (v || 0) + 10;
  }

  function targetOf(link) {
    return document.getElementById(decodeURIComponent(link.getAttribute('href').slice(1)));
  }

  function setActive(link) {
    links.forEach(a => {
      const on = a === link;
      a.classList.toggle('active', on);
      if (on) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    });
  }

  function scrollToEl(el) {
    const y = el.getBoundingClientRect().top + window.scrollY - offset();
    window.scrollTo({ top: Math.max(0, y), behavior: reduce ? 'auto' : 'smooth' });
  }

  const LOCK_MS = reduce ? 0 : 800;
  let lockUntil = 0;

  function activate(link, moveFocus) {
    const el = targetOf(link);
    if (!el) return;
    const det = el.closest('details');
    if (det && !det.open) det.open = true;
    setActive(link);
    lockUntil = Date.now() + LOCK_MS;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      scrollToEl(el);
      try { history.replaceState(null, '', link.getAttribute('href')); } catch (e) {}
      if (moveFocus !== false) {
        if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');
        setTimeout(() => { try { el.focus({ preventScroll: true }); } catch (e) {} }, reduce ? 0 : 350);
      }
      setTimeout(() => { setActive(link); }, LOCK_MS + 60);
      setTimeout(spy, LOCK_MS + 260);
    }));
  }

  links.forEach((link, i) => {
    link.addEventListener('click', e => { e.preventDefault(); activate(link, true); });
    link.addEventListener('keydown', e => {
      switch (e.key) {
        case 'Enter':
        case ' ':
        case 'Spacebar':
          e.preventDefault(); activate(link, true); break;
        case 'ArrowDown':
        case 'ArrowRight':
          e.preventDefault(); links[(i + 1) % links.length].focus(); break;
        case 'ArrowUp':
        case 'ArrowLeft':
          e.preventDefault(); links[(i - 1 + links.length) % links.length].focus(); break;
        case 'Home':
          e.preventDefault(); links[0].focus(); break;
        case 'End':
          e.preventDefault(); links[links.length - 1].focus(); break;
      }
    });
  });

  const targets = links.map(l => ({ link: l, el: targetOf(l) })).filter(t => t.el);
  let spyScheduled = false;
  function spy() {
    spyScheduled = false;
    if (Date.now() < lockUntil) return;
    const line = offset() + 6;
    let current = null;
    targets.forEach(t => {
      const r = t.el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) return;
      if (r.top - line <= 0) current = t.link;
    });
    if (!current && targets.length) {
      const firstVisible = targets.find(t => {
        const r = t.el.getBoundingClientRect();
        return !(r.width === 0 && r.height === 0);
      });
      if (firstVisible) current = firstVisible.link;
    }
    if (current) setActive(current);
  }
  window.addEventListener('scroll', () => {
    if (spyScheduled) return;
    spyScheduled = true;
    requestAnimationFrame(spy);
  }, { passive: true });

  function fixHash() {
    if (!location.hash) return;
    const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (!el) return;
    const det = el.closest('details');
    if (det) det.open = true;
    requestAnimationFrame(() => requestAnimationFrame(() => scrollToEl(el)));
  }
  window.addEventListener('hashchange', fixHash);
  window.addEventListener('load', () => setTimeout(fixHash, 80));
  setTimeout(spy, 200);
})();
</script>"""


def build_model_guide_html(models: list[dict]) -> str:
    """Render the consolidated Model Guide page as an HTML string."""
    grouped = _classify_models(models)
    total = sum(len(v) for v in grouped.values())
    tabs_html = _render_surface_tabs(grouped, total)
    rail_html = _render_rail(grouped)
    sections_html = _render_surface_sections(grouped)
    sku_chips_html = _render_sku_chips()

    catalog_json = json.dumps([
        {"id": _clean(m.get("id")),
         "name": _clean(m.get("name")) or _clean(m.get("id")),
         "fragment": doc_fragment(_clean(m.get("id"))),
         "surface": _model_surface(m)}
        for m in models if _clean(m.get("id"))
    ])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalAI Studio — Model Guide</title>
<script>{HEAD_SCRIPT}</script>
<style>{CSS}</style>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to content</a>
<header class="top">
  <div class="top-row">
    <div class="title-block">
      <a class="back-link" href="index.html" aria-label="Back to LocalAI documentation">← Back to docs</a>
      <h1>Model Guide</h1>
      <p>Every catalog model ({total} total) — value props, recommended settings, click-to-copy demos, sample renders. Hardware-aware filters. Cards start collapsed — expand one at a time or use the bar below.</p>
    </div>
    <div class="controls">
      <button class="compact-toggle" id="compact-toggle" type="button" title="Toggle compact header (hide title + subtitle to maximize card space)" aria-pressed="false">▾ Compact</button>
      <button class="theme-toggle" id="theme-toggle" type="button" title="Toggle light/dark theme">🌙 Dark</button>
    </div>
  </div>
  <div class="sku-strip" role="group" aria-label="Filter by hardware — SKU or CPU-only">
    <span class="sku-label">Hardware SKU</span>
    {sku_chips_html}
  </div>
</header>

<nav class="filter-bar" aria-label="Filters">
  <div class="filter-field">
    <label for="q">Search</label>
    <input id="q" type="search" autocomplete="off" spellcheck="false"
           placeholder="Model, vendor, capability, prompt text, negative…">
  </div>
  <div class="filter-field">
    <label for="hardware">Hardware fit</label>
    <select id="hardware">
      <option value="all">All hardware</option>
      <option value="cpu16">CPU · ≤ 16 GB RAM</option>
      <option value="cpu32">CPU · ≤ 32 GB RAM</option>
      <option value="cpu64">CPU · ≤ 64 GB RAM</option>
      <option value="4">≤ 4 GB VRAM</option>
      <option value="8">≤ 8 GB VRAM</option>
      <option value="12">≤ 12 GB VRAM</option>
      <option value="16">≤ 16 GB VRAM</option>
      <option value="24">24+ GB VRAM</option>
    </select>
  </div>
  <div class="filter-field">
    <label for="goal">Goal</label>
    <select id="goal">
      <option value="all">All goals</option>
      <option value="quick">Quick chat / answers</option>
      <option value="reasoning">Reasoning / math</option>
      <option value="coding">Coding</option>
      <option value="long">Long-context analysis</option>
      <option value="multilingual">Multilingual</option>
      <option value="photo">Hyperrealistic photo</option>
      <option value="fast">Fast previews / drafts</option>
      <option value="anime">Anime / illustration</option>
      <option value="art">Painterly / artistic</option>
      <option value="ocr">OCR / document</option>
      <option value="transcribe">Audio transcription</option>
      <option value="tts">Text-to-speech</option>
      <option value="retrieve">Retrieval / similarity</option>
    </select>
  </div>
  <!-- The surface dropdown was redundant with the surface tab strip below;
       removed per designer review (N4). The hidden select keeps internal JS
       happy without putting a visible duplicate control on screen. -->
  <select id="surface-filter" hidden aria-hidden="true" tabindex="-1">
    <option value="all" selected>All surfaces</option>
    <option value="chat">Chat</option>
    <option value="vision">Vision</option>
    <option value="image">Image generation</option>
    <option value="speech">Speech</option>
    <option value="embed">Embeddings</option>
    <option value="doc">Document AI</option>
  </select>
</nav>

<nav class="surface-tabs" role="tablist" aria-label="Filter by model surface">
{tabs_html}
</nav>

<div class="shell">

  <aside class="rail" aria-label="Model index">
    <div class="rail-label">Jump to a model</div>
    <div id="rail-surfaces">{rail_html}</div>
  </aside>

  <main class="content" id="main-content" tabindex="-1">

    <section class="system-banner" id="system-banner" hidden aria-live="polite">
      <div class="system-banner-icon" aria-hidden="true">💻</div>
      <div class="system-banner-body">
        <div class="system-banner-title" id="sb-title">Showing models for your system</div>
        <div class="system-banner-subtitle" id="sb-subtitle"></div>
      </div>
      <button class="system-banner-clear" id="sb-clear" type="button">Show all models</button>
    </section>

    <div data-doc-extension-slot="model-value-hardware-profiles" hidden></div>

    <div class="density-bar" role="toolbar" aria-label="Card density">
      <div class="density-count"><strong id="visible-count">{total}</strong> of <span id="total-count">{total}</span> models</div>
      <div class="density-spacer"></div>
      <button class="density-btn" id="collapse-all" type="button" aria-label="Collapse every visible card">▸ Collapse all</button>
      <button class="density-btn" id="expand-all"   type="button" aria-label="Expand every visible card">▾ Expand all</button>
    </div>

    <div class="empty-state" id="empty-state" role="status" aria-live="polite" hidden>No models match these filters. Try clearing the search box or selecting a larger hardware profile.</div>

    <div id="surfaces">{sections_html}</div>
  </main>
</div>

<button id="back-to-top" type="button" aria-label="Back to top">↑ Top</button>
<div id="toast" role="status" aria-live="polite">Copied!</div>

<script id="catalog-data" type="application/json">{catalog_json}</script>
<script>{JS}</script>
<!-- localai-docs-extension.js injects an optional cloud-VM coverage table into
     the data-doc-extension-slot inside <main>. The slot lives on a dedicated
     <div> (not on .shell) so the inject's innerHTML replacement can't wipe
     the rail and main content. Page works fine if the script is absent. -->
<script src="localai-docs-extension.js" defer></script>
{DOCNAV_A11Y}
</body>
</html>
"""


def write_model_guide(models: list[dict], path: Path) -> None:
    """Render and write the Model Guide HTML to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_model_guide_html(models), encoding="utf-8")

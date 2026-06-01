# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Model-specific demo prompts and documentation helpers."""

from __future__ import annotations

import html
import json
from pathlib import Path

from src.sample_prompts import MODEL_DEMO_SAMPLE_OVERRIDES


def _clean(value: object) -> str:
    return str(value or "").strip()


def _slug(value: str) -> str:
    return "-".join(
        part for part in "".join(ch.lower() if ch.isalnum() else "-" for ch in value).split("-")
        if part
    )


def doc_fragment(model_id: str) -> str:
    slug = _slug(model_id)
    return f"model-{slug}" if slug else ""


def _gb_label(value: object, *, zero_label: str) -> str:
    text = _clean(value)
    if not text:
        return zero_label
    try:
        number = float(text)
    except ValueError:
        return text
    if number <= 0:
        return zero_label
    return f"{number:g} GB"


def _token_label(value: object) -> str:
    text = _clean(value)
    if not text:
        return "Not listed"
    try:
        number = int(float(text))
    except ValueError:
        return text
    if number <= 0:
        return "Not listed"
    return f"{number:,} tokens"


def _chat_reply_limit_label(model: dict) -> str:
    category = _clean(model.get("category"))
    backend = _clean(model.get("backend")).lower()
    is_chat = (
        backend != "comfyui"
        and category not in {"Image Generation", "Speech", "Embeddings", "Document AI"}
        and bool(model.get("ollama_tag") or model.get("onnx_repo") or model.get("ov_repo"))
    )
    if not is_chat:
        return "Not a chat surface"
    context = _token_label(model.get("context_length"))
    if context == "Not listed":
        return "Max / fill context; custom cap in Chat"
    return f"Max / fill context ({context}); custom cap in Chat"


_THUMBNAIL_DIR = Path(__file__).resolve().parents[1] / "docs" / "images" / "model_demos"


def _sample_thumbnail_url(model_id: str, index: int) -> str | None:
    """Return a relative thumbnail URL if a generated sample image exists.

    Thumbnails are produced by tools/build_demo_thumbnails.py and live under
    docs/images/model_demos/modeldemo-<model_id>-<n>.jpg (1-based n).
    """
    if not model_id:
        return None
    fname = f"modeldemo-{model_id}-{index + 1}.jpg"
    if (_THUMBNAIL_DIR / fname).exists():
        return f"images/model_demos/{fname}"
    return None


def _negative_prompt_for_model(model_id: str) -> tuple[str, bool]:
    """Return (negative_prompt_text, is_image_gen_model).

    is_image_gen_model means the model has any entry in `_IMAGE_PROMPTS`.
    An empty string with is_image_gen_model=True indicates the model belongs
    to a CFG=1.0 family (Flux / Z-Image / Turbo / Lightning / Chroma) where
    negative prompts are silently ignored by the workflow.
    """
    if model_id not in _IMAGE_PROMPTS:
        return "", False
    try:
        from src.config import DEFAULT_NEGATIVE_PROMPTS_BY_MODEL
    except Exception:
        return "", True
    return str(DEFAULT_NEGATIVE_PROMPTS_BY_MODEL.get(model_id, "") or ""), True


_IMAGE_PROMPTS: dict[str, list[str]] = {
    "bytedance-sdxl-lightning": [
        # Surfer inside a turquoise barrel
        "A surfer carving inside a glassy turquoise barrel wave at sunrise, spray catching the warm light, dynamic action sports photography, ultra-sharp 1024px composition",
        # Bengal tiger water burst
        "a Bengal tiger leaping through a wall of crystal-clear water droplets, hyper-detailed fur, slow-motion, dramatic side lighting, nature photography",
        # Tuscan wildflower meadow
        "A summer wildflower meadow with poppies, cornflowers, and daisies stretching to a row of distant cypress trees under a clear blue sky, sharp foreground focus, naturalistic Tuscan landscape photography",
    ],
    "flux1-dev-fp16": [
        # Lighthouse in a storm
        "A lone stone lighthouse on a rocky North Atlantic headland weathering a winter storm at twilight, waves exploding against the cliff base, the beacon beam cutting through rain, dramatic seascape photography, 8k detail",
        # Chef plating a tasting course
        "A fine-dining chef carefully placing a sprig of microgreens on a precisely composed seared scallop dish at the kitchen pass, warm overhead lights, shallow depth of field, editorial food photography",
        # Victorian spiral staircase
        "A wrought-iron spiral staircase rising through a Victorian library hall lined with ancient leather books, a stained-glass dome above casting colored light onto the marble floor, architectural interior photography",
    ],
    "flux1-dev-q4": [
        # Juniper bonsai study
        "A meticulously pruned juniper bonsai in a hand-thrown ceramic pot resting on a slate surface, soft north-window light, every needle and bark texture sharply rendered, traditional Japanese horticulture photography",
        # Solo cellist on stage
        "A solo cellist performing under a single spotlight on a dark theater stage, bow mid-stroke, expressive concentration, dramatic chiaroscuro concert photography",
        # Antique brass pocket watch
        "A heavily detailed brass antique pocket watch lying open on a worn leather journal, intricate gear movement visible, soft library lighting, macro photography",
    ],
    "flux1-schnell-q4": [
        # Hummingbird at trumpet flower
        "An iridescent ruby-throated hummingbird hovering beside a bright red trumpet flower, wings frozen mid-beat, soft bokeh garden background, ultra-sharp nature photography",
        # Dewdrop on a red maple leaf
        "A single jewel-like dewdrop on a vivid red Japanese maple leaf in early morning light, hyper-clear extreme macro photography with crisp edge detail and a soft green forest bokeh background",
        # Adventurer's brass compass and map
        "An antique brass compass and a folded canvas map laid across a weathered wooden table beside a steaming enamel mug of coffee at first light, soft golden sunrise spilling through a tent opening, warm earthy tones with crisp metal detail, evocative still-life adventure photography",
    ],
    "gsdf-counterfeit-v3.0": [
        # Cherry blossom suburban street
        "Anime-style illustration of a quiet Japanese suburban street lined with blooming cherry trees, petals drifting in the breeze, soft pastel colors, peaceful daytime scene, painterly background detail",
        # Cozy cat cafe interior
        "Anime-style cozy cat cafe interior with several sleeping cats on plush chairs, steaming teacups on small wooden tables, warm afternoon light through paper screens, gentle line art and detailed lighting",
        # Mountain shrine path in autumn
        "Anime-style illustration of a stone path leading up to a small mountain shrine surrounded by maple trees in autumn, lantern light glowing at dusk, painterly Studio Ghibli mood",
    ],
    "juggernaut-xl-v9": [
        # Equestrian and chestnut horse
        "A confident equestrian in a clean white riding shirt and tan breeches standing beside a chestnut horse in a sunlit paddock, natural skin texture and realistic horse coat detail, magazine-quality portrait photography",
        # Mechanic in a classic-car workshop
        "A focused mechanic in clean navy coveralls adjusting the engine of a classic European sports car inside a tidy workshop, soft overhead garage lighting, authentic skin and fabric textures, documentary-style photography",
        # Old fisherman mending a net
        "A weathered Portuguese fisherman mending a deep-blue net on a stone harbor wall at golden hour, ocean breeze in his white hair, deeply lined skin, documentary portrait photography",
    ],
    "playgroundai-playground-v2.5-1024px-aesthetic": [
        # Pastel still life with peonies
        "A pastel still life of fresh peonies, macarons, and a porcelain teapot on a marble table, soft pink and mint color palette, elegant lifestyle magazine photography",
        # Moody emerald armchair interior
        "Sumptuous moody magazine interior with a single emerald-green velvet armchair and a small black marble side table beneath a tall arched window, dramatic afternoon light spilling in from a sunlit garden beyond, dark navy plaster walls with subtle brass picture-rail detail, a lush potted fiddle-leaf fig in the corner, polished herringbone wood floor, magazine-quality interior aesthetic photography with rich saturated color",
        # Fantasy face-paint portrait with natural skin and textured hair
        "Close-up fantasy beauty portrait matching the supplied reference-image mood: a woman with luminous cyan-blue eyes and thick copper-auburn wavy hair framing her face, individual hair strands and flyaways clearly visible, elaborate symmetrical turquoise and metallic-gold face painting sweeping across her cheeks, nose bridge, temples, and forehead like an artful painted mask, glittering pigment strokes, natural matte skin with visible pores, peach fuzz, faint freckles, realistic skin texture, intense forward gaze, dark teal background, shallow depth of field, cinematic editorial color grading, not plastic, not airbrushed, richly detailed painterly-real aesthetic",
    ],
    "realistic-vision-v6": [
        # Bioluminescent sea-cave night scene
        "Cinematic photorealistic vertical composition inside a colossal black-basalt sea cave at night, turquoise bioluminescent waves rolling through a mirror-like tide pool, a thin waterfall dropping from a moonlit opening in the cave ceiling, silver mist, wet volcanic rock reflections, glowing sea foam, dramatic scale, deep shadows, crisp natural detail, no signs, no letters, no readable text, no people, no hands, no human skin, high-end environmental photography",
        # Cozy fireplace evening
        "A cozy evening scene of a crackling wood fire glowing inside a large stone fireplace, two empty cognac leather armchairs angled toward the flames with a soft wool throw blanket draped over one armrest, warm flickering firelight dancing across rustic wood-paneled walls and a worn Persian rug, deep atmospheric photorealistic interior photography with soft shadows and natural warm color temperature",
        # Sommelier in a candlelit cellar
        "A focused young sommelier in a tailored black vest and white shirt swirling a glass of deep ruby red wine inside a candlelit wine cellar lined with oak barrels and dusty bottles, warm amber lighting catching the wine and her dark hair, realistic skin texture and natural fabric folds, editorial restaurant photography",
    ],
    "sdxl-base": [
        # Tropical waterfall
        "A lush tropical waterfall cascading into a turquoise plunge pool surrounded by ferns and mossy basalt rocks, dappled afternoon sunlight breaking through the jungle canopy, photorealistic landscape photography, sharp focus, vivid natural color",
        # Antique bookshop interior
        "Interior of an old wooden bookshop with floor-to-ceiling shelves of leather-bound volumes, a brass reading lamp casting warm light on a green velvet armchair, dust motes drifting in shafts of afternoon sun, atmospheric architectural photography",
        # Vintage cafe-racer at dusk
        "A polished vintage 1960s British cafe-racer motorcycle parked on a wet cobblestone street at dusk, chrome reflections catching the neon glow of shop signs, light rain beading on the leather seat, cinematic urban photography",
    ],
    "sdxl-lowvram": [
        # Late-summer sunflower field
        "A vast field of sunflowers in late summer with rolling hills in the background, gentle breeze visible in petal motion, soft golden-hour lighting, naturalistic landscape painting style",
        # Cozy farmhouse kitchen
        "A small cozy farmhouse kitchen with a copper kettle steaming on a wood stove, a wicker basket of fresh bread on the table, morning sunlight through gingham curtains, warm rustic interior photography",
        # Vintage cafe-racer at dusk (SDXL Base checkpoint, low-VRAM profile)
        "A polished vintage 1960s British cafe-racer motorcycle parked on a wet cobblestone street at dusk, chrome reflections catching the neon glow of shop signs, light rain beading on the leather seat, cinematic urban photography",
    ],
    "z-image-turbo": [
        # Alpine mountain pass at sunrise
        "A winding alpine mountain pass road photographed from above at sunrise, mist rising from the valleys, snow-capped peaks in the distance, dramatic high-contrast landscape photography",
        # Coastal tide pool
        "A coastal tide pool at low tide revealing orange starfish, purple sea urchins, and emerald anemones in clear water, wet rocks reflecting the morning sky, vibrant natural-history photography",
        # Hot-air balloons over Cappadocia
        "Several colorful hot-air balloons drifting over the limestone fairy chimneys of Cappadocia at sunrise, soft warm light, sweeping aerial landscape photography",
    ],
}

_UTILITY_DEMOS = {
    "whisper-large-v3-turbo": {
        "feature": "Speech-to-text",
        "primary": "Transcribe the included local WAV clip, preserve punctuation, then summarize the spoken message in one sentence.",
        "samples": [
            "Transcribe a recorded product demo and return: transcript, speaker intent, and three follow-up action items.",
            "Convert a meeting voice note into a clean summary with decisions, dates, and owners.",
            "Transcribe a noisy customer support clip and mark any uncertain words with [unclear].",
        ],
        "why": "Whisper Large v3 Turbo is a strong speech-recognition demo because it turns raw audio into useful written notes quickly.",
    },
    "whisper-v3-turbo-gpu": {
        "feature": "GPU speech-to-text",
        "primary": "Use the accelerated path to transcribe the included WAV clip and return a concise demo-ready transcript.",
        "samples": [
            "Transcribe a fast voice memo and produce a crisp executive summary.",
            "Turn a product walkthrough audio clip into a checklist of demonstrated features.",
            "Transcribe a short training clip and identify three key terms the speaker emphasized.",
        ],
        "why": "The GPU variant shows how hardware acceleration can make speech transcription feel immediate.",
    },
    "speecht5-tts": {
        "feature": "Text-to-speech",
        "primary": "Generate a friendly voice saying: Your pickup is ready; bring identification and ask staff if you have questions.",
        "samples": [
            "Create a warm 10-second narration for a museum audio-guide welcome.",
            "Generate a crisp accessibility voiceover explaining a museum map kiosk.",
            "Create a short spoken alert: Your image generation batch is complete.",
        ],
        "why": "SpeechT5 adds spoken output so audio workflows are not limited to transcription.",
    },
    "piper-tts": {
        "feature": "Offline TTS (multi-language)",
        "primary": "Speak a short greeting using the default Piper voice and save the WAV locally.",
        "samples": [
            "Generate a Spanish welcome announcement for a self-checkout kiosk using a Piper Spanish voice.",
            "Read a one-sentence shipping update aloud in English using a male Piper voice.",
            "Create a 5-second French confirmation message and save the WAV in the toolbox_outputs folder.",
        ],
        "why": "Piper runs entirely on CPU with small per-voice ONNX files and ships dozens of languages.",
    },
    "got-ocr2": {
        "feature": "Document OCR + table extraction",
        "primary": "Extract a table from the included benchmark image and return both LaTeX and TSV.",
        "samples": [
            "OCR a scanned invoice page and return the line-item table as TSV.",
            "Convert a photographed financial summary into a LaTeX tabular block.",
            "Pull the cell text from a complex multi-row spreadsheet screenshot for downstream parsing.",
        ],
        "why": "GOT-OCR 2.0 returns structured output (LaTeX / TSV) that downstream tooling can parse.",
    },
    "all-minilm": {
        "feature": "Embeddings",
        "primary": "Embed three short customer-support sentences and show which two are semantically closest.",
        "samples": [
            "Compare support-ticket sentences and rank which two describe the same issue.",
            "Embed five feature requests and cluster similar requests into themes.",
            "Score whether a user's search query matches knowledge-base pages about returns, shipping, or warranties.",
        ],
        "why": "All-MiniLM is tiny, fast, and makes semantic search easy to demonstrate on modest hardware.",
    },
    "florence-2-base": {
        "feature": "Image captioning",
        "primary": "Caption the included benchmark image and describe the visible objects and layout.",
        "samples": [
            "Create alt text for a product screenshot with enough detail for accessibility.",
            "Describe the objects and layout in a dashboard screenshot.",
            "Turn a whiteboard photo into a short visual summary for a meeting recap.",
        ],
        "why": "Florence-2 Base shows lightweight local image understanding outside the chat UI.",
    },
    "table-transformer": {
        "feature": "Table-region detection",
        "primary": "Find table regions in the included benchmark image and save cropped regions for follow-up OCR.",
        "samples": [
            "Find tables in a scanned invoice image and save cropped table regions.",
            "Identify the table area in a photographed spreadsheet printout.",
            "Detect multiple tables in a report page before OCR extraction.",
        ],
        "why": "Table Transformer demonstrates document-layout intelligence, not just chat.",
    },
    "trocr-base-printed": {
        "feature": "OCR",
        "primary": "Read the text in the included benchmark image and return only the extracted text.",
        "samples": [
            "Extract printed text from a small screenshot and preserve line breaks.",
            "Read a printed label image and return a normalized plain-text version.",
            "OCR a simple form header and identify the most important field names.",
        ],
        "why": "TrOCR Base is the fast OCR demo for smaller machines.",
    },
    "trocr-large-printed": {
        "feature": "High-accuracy OCR",
        "primary": "Read the text in the included benchmark image, preserve capitalization, and flag any uncertain characters.",
        "samples": [
            "Extract printed text from a dense report crop and preserve punctuation.",
            "OCR a scanned product label and return fields as key-value pairs.",
            "Read a slide screenshot and summarize the extracted text in one sentence.",
        ],
        "why": "TrOCR Large shows the higher-quality OCR path when the user has a larger memory budget.",
    },
    "phi-4-multimodal": {
        "feature": "Multimodal reasoning",
        "primary": "Give three concise reasons a field-service assistant should understand both images and text.",
        "samples": [
            "Give three concise reasons a field-service assistant should understand both images and text.",
            "Compare an OCR-only workflow with a multimodal reasoning workflow and explain when each is the better demo.",
            "Create a short presenter script that introduces multimodal reasoning as a future-facing assistant capability.",
        ],
        "why": "Phi-4 Multimodal bridges vision with reasoning for richer image-and-text tasks.",
    },
}

_CHAT_OVERRIDES = {
    "qwen2.5:0.5b": ("Ultra-fast tiny assistant", "In 60 words or less, give a practical checklist for preparing a community workshop demo."),
    "gemma3:1b": ("Tiny writing helper", "Rewrite this rough launch note into a friendly, polished announcement under 120 words: the new assistant can summarize text, images, audio notes, and documents in one place."),
    "llama3.2:1b": ("Tiny reasoning", "Solve this short logic puzzle and explain only the key steps: three model demos run in sequence; OCR must run before summarization, speech must run after OCR, and image generation cannot run last. What is the order?"),
    "llama3.2:3b": ("Long-context small model", "Summarize this product idea into problem, user, value, risks, and MVP scope: a mobile app that connects neighborhood volunteers with time-sensitive community requests."),
    "phi4:mini": ("Compact STEM/code", "Write a small Python function that estimates whether a job fits a device from available and required memory values, then include three assert tests."),
    "deepseek-r1-1.5b": ("Small reasoning", "Think carefully but answer concisely: if a demo must prove speed, accuracy, and image understanding in five minutes, what sequence should I show and why?"),
    "nemotron-3-nano:4b": ("Efficient NVIDIA reasoning", "Final answer only. A museum loses power 20 minutes before opening. Create a crisp action plan in exactly five bullets: visitor safety, exhibit protection, staff roles, communication, and reopening check."),
    "nemotron-3-nano:4b-q8_0": ("Higher-quality NVIDIA reasoning", "Final answer only. Review this Python function for one bug and provide corrected code under 160 words: def median(values): values.sort(); return values[len(values)//2]"),
    "granite3.1-moe:latest": ("Fast business assistant", "Turn these notes into a crisp executive update with risks and next actions: launch date moved, supplier review pending, pilot users enthusiastic, support plan incomplete."),
    "mistral-nemo": ("Long-context writing", "Create a demo script for a presenter showing a new field-service assistant. Include opening hook, three feature beats, and closing line."),
    "aya-expanse:8b": ("Multilingual", "Translate this product promise into Spanish, French, Japanese, and Arabic, then note any phrase that should be localized: A field assistant turns messy notes into clear next steps."),
    "dolphin3:latest": ("Direct coding helper", "Given a bug report that filtering shows archived products in the active list, write a concise debugging plan and a likely JavaScript fix."),
    "falcon3:7b": ("Fast general assistant", "Create five punchy one-line demo captions for a helpful assistant, each focused on a different capability: chat, vision, OCR, speech, image generation."),
    "granite3.3:8b": ("Enterprise analysis", "Analyze this enterprise rollout plan for an internal assistant. Return benefits, governance concerns, and a 30-day pilot checklist."),
    "qwen2.5-coder-7b": ("Code generation", "Implement a dependency-free JavaScript function that filters product cards by selected category and available inventory. Include tests as console assertions."),
    "phi4": ("STEM reasoning", "Explain how quantization helps a large model fit on smaller hardware. Use an analogy, one formula-style memory estimate, and a caveat."),
    "qwen3-30b-a3b": ("MoE reasoning", "/no_think Explain why a mixture-of-experts model can feel larger than its active parameters. Include implications for speed, memory, and user experience."),
    "deepseek-r1:32b": ("Large reasoning", "Answer in exactly five concise bullets comparing three pilot ideas: automated triage, document summarization, and image-based inspection. Give one reason per recommendation."),
    "llama3.3": ("Large enterprise reasoning", "Draft a board-level summary explaining why an organization should run sensitive assistants close to its data, with benefits, risks, rollout phases, and success metrics."),
    "llama3.2-vision:11b": ("Vision chat", "If I upload a product screenshot, analyze layout, key UI affordances, missing context, and what a user should try next."),
    "gemma3:4b-vision": ("Small vision chat", "Describe an uploaded screenshot for a non-technical user, then suggest three quick improvements to make it clearer."),
    "gemma3:12b-vision": ("Balanced vision reasoning", "Analyze an uploaded chart or dashboard image. Extract visible metrics, infer the trend, and list questions to ask before making a decision."),
    "gemma3-27b": ("Large multimodal reasoning", "Review an uploaded architecture diagram. Explain the system, identify risks, and suggest a simpler alternative."),
    "minicpm-v-vision": ("Efficient vision", "Analyze an uploaded receipt or sign image and return structured JSON with visible text, objects, and confidence notes."),
    "mistral-small3.2": ("Large vision/writing", "Analyze an uploaded product screenshot and write a polished release-note paragraph plus three user-facing benefits."),
    "qwen2.5vl-3b": ("Small vision-language", "Look at an uploaded image and return a concise caption, visible text, and one likely user intent."),
    "qwen2.5vl-7b": ("Balanced vision-language", "Analyze an uploaded UI screenshot and produce a QA checklist: layout issues, text issues, accessibility issues, and suggested fixes."),
    "qwen2.5vl-32b": ("High-end vision reasoning", "Study an uploaded dense dashboard screenshot and produce a senior-analyst brief with insights, caveats, and recommended next actions."),
}


def _default_chat_demo(model: dict) -> dict:
    model_id = _clean(model.get("id"))
    name = _clean(model.get("name")) or model_id
    tags = " ".join(model.get("tags") or []).lower()
    category = _clean(model.get("category"))
    feature = "Local chat"
    primary = (
        f"Show why {name} is useful: create a concise, demo-ready answer "
        "with a recommendation, a tradeoff, and a next step."
    )
    if "coder" in model_id or "coding" in tags:
        feature = "Coding"
        primary = f"Use {name} to review a short Python function for bugs, edge cases, and a cleaner implementation."
    elif "vision" in category.lower() or "vl" in model_id or "vision" in tags:
        feature = "Vision chat"
        primary = f"Use {name} with an uploaded screenshot: describe what is visible, extract any text, and recommend what to improve."
    elif "reason" in model_id or "deepseek" in model_id:
        feature = "Reasoning"
        primary = f"Use {name} to solve a constrained planning problem step by step, then give a concise final recommendation."
    elif "aya" in model_id:
        feature = "Multilingual"
        primary = f"Use {name} to translate a product update into three languages while preserving a professional tone."
    elif "granite" in model_id:
        feature = "Enterprise analysis"
        primary = f"Use {name} to turn rough enterprise rollout notes into an executive summary with risks and next actions."
    return {
        "feature": feature,
        "primary": primary,
        "samples": [
            primary,
            f"Ask {name} for a structured comparison of two demo paths: fastest wow versus deepest quality.",
            f"Have {name} produce a presenter script that explains its own strength for a {category or 'general'} task.",
        ],
        "why": f"{name} is best demoed with a structured prompt that makes its {feature.lower()} behavior easy to judge.",
    }


def get_model_demo(model: dict) -> dict:
    model_id = _clean(model.get("id"))
    name = _clean(model.get("name")) or model_id
    if model_id in _UTILITY_DEMOS:
        demo = dict(_UTILITY_DEMOS[model_id])
    elif model_id in _IMAGE_PROMPTS or model.get("backend") == "comfyui":
        prompts = _IMAGE_PROMPTS.get(model_id)
        if isinstance(prompts, list) and prompts:
            samples = [str(p).strip() for p in prompts if str(p).strip()]
        elif isinstance(prompts, str) and prompts.strip():
            samples = [prompts.strip()]
        else:
            samples = []
        if not samples:
            fallback_primary = (
                f"A polished showcase image made with {name}: detailed cinematic "
                "composition, natural lighting, sharp focus, professional photography"
            )
            samples = [
                fallback_primary,
                f"A vibrant real-world scene that highlights {name}: detailed natural "
                "lighting, balanced composition, crisp color, magazine-style photography.",
                f"A photorealistic landscape rendered with {name}: dramatic atmospheric "
                "lighting, sharp depth of field, evocative scene with no logos or text.",
            ]
        demo = {
            "feature": "Image generation",
            "primary": samples[0],
            "samples": samples,
            "why": (
                f"{name} is included to showcase a distinct image-generation style or "
                "resource profile."
            ),
        }
    else:
        feature, primary = _CHAT_OVERRIDES.get(model_id, (None, None))
        demo = _default_chat_demo(model)
        if primary:
            demo["feature"] = feature or demo["feature"]
            demo["primary"] = primary
            prompt_prefix = "/no_think " if model_id.startswith("qwen3") else ""
            demo["samples"] = [
                primary,
                f"{prompt_prefix}Use {name} to create a demo script with opening hook, proof point, and closing line.",
                f"{prompt_prefix}Using {name}, compare two short demo formats — (A) a fast 1-minute live demo and (B) a polished 3-minute scripted demo — and recommend one in exactly 3 short bullets covering audience fit, the speed-versus-polish tradeoff, and one risk.",
            ]
            demo["why"] = f"{name} is a strong demo for {demo['feature'].lower()} because the prompt makes its best use clear."

    if model_id in MODEL_DEMO_SAMPLE_OVERRIDES:
        demo["samples"] = MODEL_DEMO_SAMPLE_OVERRIDES[model_id]
        demo["primary"] = demo["samples"][0]

    samples = [str(s).strip() for s in demo.get("samples", []) if str(s).strip()]
    while len(samples) < 3:
        samples.append(f"Show a unique demo for {name} focused on {demo.get('feature', 'assistant capability')}.")
    demo["samples"] = samples[:3]
    demo["primary"] = _clean(demo.get("primary")) or demo["samples"][0]
    demo["why"] = _clean(demo.get("why")) or f"{name} has a focused demo prompt."
    demo["feature"] = _clean(demo.get("feature")) or "Assistant"
    return demo


def build_demo_docs_html(models: list[dict]) -> str:
    rows = []
    nav = []
    for model in models:
        mid = _clean(model.get("id"))
        if not mid:
            continue
        demo = get_model_demo(model)
        frag = doc_fragment(mid)
        name = _clean(model.get("name")) or mid
        category = _clean(model.get("category")) or "Model"
        vendor = _clean(model.get("vendor")) or "Unknown"
        min_ram = _gb_label(model.get("min_ram_gb"), zero_label="Varies by backend")
        min_vram = _gb_label(model.get("min_vram_gb"), zero_label="Not required")
        context = _token_label(model.get("context_length"))
        reply_limit = _chat_reply_limit_label(model)
        utility_note = (
            '<p class="note">Run Demo / Toolbox workflow idea: this utility model is not a Chat prompt target.</p>'
            if mid in _UTILITY_DEMOS else ""
        )
        nav.append(
            f'<a href="#{html.escape(frag)}" data-target="{html.escape(frag, quote=True)}">'
            f'{html.escape(name)}</a>'
        )
        sample_parts = []
        for sample_idx, sample in enumerate(demo["samples"]):
            thumb_url = _sample_thumbnail_url(mid, sample_idx)
            if thumb_url:
                thumb_html = (
                    f'<a class="thumb-link" href="{html.escape(thumb_url)}" target="_blank" '
                    f'rel="noopener" aria-label="Sample render for {html.escape(name, quote=True)} '
                    f'— open full size in a new tab">'
                    f'<img class="sample-thumb" src="{html.escape(thumb_url)}" alt="" loading="lazy" '
                    f'decoding="async" width="180" height="180"></a>'
                )
                sample_classes = "sample sample-has-thumb"
            else:
                thumb_html = ""
                sample_classes = "sample"
            sample_parts.append(
                f'<div class="{sample_classes}">{thumb_html}'
                f'<div class="sample-body">'
                f'<button type="button" class="copy-btn" '
                f'data-copy="{html.escape(sample, quote=True)}">Copy</button>'
                f'<pre>{html.escape(sample)}</pre>'
                f'</div></div>'
            )
        samples = "\n".join(sample_parts)
        negative_text, is_image_gen = _negative_prompt_for_model(mid)
        if is_image_gen:
            if negative_text:
                negative_block = (
                    '<h3>Negative prompt (shared across all three samples)</h3>'
                    '<div class="sample negative-sample">'
                    '<div class="sample-body">'
                    f'<button type="button" class="copy-btn" '
                    f'data-copy="{html.escape(negative_text, quote=True)}">Copy</button>'
                    f'<pre>{html.escape(negative_text)}</pre>'
                    '</div></div>'
                )
            else:
                negative_block = (
                    '<h3>Negative prompt</h3>'
                    '<p class="note negative-note">This model family runs at '
                    '<strong>CFG&nbsp;=&nbsp;1.0</strong> (Flux, Z&#8209;Image, '
                    'Turbo, Lightning, Chroma). Negative prompts are silently '
                    'ignored by the workflow — guidance below 1.0 has no effect, '
                    'and adding a negative prompt will not change the render. '
                    'Use precise positive language instead.</p>'
                )
        else:
            negative_block = ""
        rows.append(f'''
      <section class="model-card" id="{html.escape(frag)}" data-model-id="{html.escape(mid)}" data-category="{html.escape(category)}" data-search="{html.escape((mid + " " + name + " " + category + " " + vendor + " " + demo["feature"]).lower(), quote=True)}">
        <div class="card-head">
          <div>
            <p class="eyebrow">{html.escape(category)} · {html.escape(vendor)}</p>
            <h2>{html.escape(name)}</h2>
          </div>
          <span class="feature">{html.escape(demo["feature"])}</span>
        </div>
        <p class="why">{html.escape(demo["why"])}</p>
        {utility_note}
        <div class="meta">
          <span>ID: <code>{html.escape(mid)}</code></span>
          <span>Min RAM: {html.escape(min_ram)}</span>
          <span>Min VRAM: {html.escape(min_vram)}</span>
          <span>Context: {html.escape(context)}</span>
          <span>Reply default: {html.escape(reply_limit)}</span>
        </div>
        <h3>Three click-to-copy wow samples</h3>
        {samples}
        {negative_block}
      </section>''')

    model_json = json.dumps([
        {"id": _clean(m.get("id")), "name": _clean(m.get("name")), "fragment": doc_fragment(_clean(m.get("id")))}
        for m in models if _clean(m.get("id"))
    ])
    nav_html = "\n".join(nav)
    cards_html = "\n".join(rows)
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalAI Studio Model Demo Prompts</title>
<script>
  (() => {{
    const param = new URLSearchParams(window.location.search).get("clawpilotTheme");
    const theme =
      param || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
  }})();
</script>
<style>
:root {{
  color-scheme: light;
  --cp-bg: #f7f4ef;
  --cp-bg-elevated: #fcfbf8;
  --cp-surface: #ffffff;
  --cp-surface-soft: #f5f5f5;
  --cp-border: #dedede;
  --cp-border-strong: #919191;
  --cp-text: #242424;
  --cp-text-muted: #5c5c5c;
  --cp-text-soft: #6f6f6f;
  --cp-accent: #b11f4b;
  --cp-accent-hover: #9a1a41;
  --cp-accent-soft: rgba(177, 31, 75, 0.08);
  --cp-accent-fg: #ffffff;
  --cp-success: #16a34a;
  --cp-danger: #dc2626;
  --cp-warning: #f59e0b;
  --cp-link: #0078d4;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.12);
  --cp-overlay: rgba(255, 255, 255, 0.8);
  --cp-panel: rgba(255, 255, 255, 0.86);
  --cp-panel-strong: rgba(255, 255, 255, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.55);
  --cp-highlight: rgba(177, 31, 75, 0.12);
}}
html[data-theme="dark"] {{
  color-scheme: dark;
  --cp-bg: #3d3b3a;
  --cp-bg-elevated: #343231;
  --cp-surface: #292929;
  --cp-surface-soft: #2e2e2e;
  --cp-border: #474747;
  --cp-border-strong: #5f5f5f;
  --cp-text: #dedede;
  --cp-text-muted: #919191;
  --cp-text-soft: #b0b0b0;
  --cp-accent: #fd8ea1;
  --cp-accent-hover: #fb7b91;
  --cp-accent-soft: rgba(253, 142, 161, 0.14);
  --cp-accent-fg: #1a1a1a;
  --cp-success: #4ade80;
  --cp-danger: #f87171;
  --cp-warning: #fbbf24;
  --cp-link: #4da6ff;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
  --cp-overlay: rgba(41, 41, 41, 0.88);
  --cp-panel: rgba(41, 41, 41, 0.72);
  --cp-panel-strong: rgba(41, 41, 41, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.04);
  --cp-highlight: rgba(253, 142, 161, 0.12);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; scroll-padding-top: 96px; }}
body {{ margin: 0; background: var(--cp-bg); color: var(--cp-text); font-family: "Segoe UI", Aptos, Calibri, -apple-system, BlinkMacSystemFont, sans-serif; }}
header {{ position: sticky; top: 0; z-index: 10; padding: 20px 28px; background: var(--cp-panel-strong); border-bottom: 1px solid var(--cp-border); }}
h1 {{ margin: 0 0 6px; font-size: 28px; }}
.sub {{ margin: 0; color: var(--cp-text-muted); }}
.toolbar {{ display: grid; grid-template-columns: 1fr 220px; gap: 10px; margin-top: 14px; }}
input, select {{ border: 1px solid var(--cp-border); background: var(--cp-surface); color: var(--cp-text); border-radius: 0.625rem; padding: 9px 10px; font: inherit; }}
.layout {{ display: grid; grid-template-columns: 290px 1fr; gap: 20px; padding: 20px 28px 40px; }}
aside {{ position: sticky; top: 112px; height: calc(100vh - 132px); overflow: auto; padding: 14px; background: var(--cp-surface); border: 1px solid var(--cp-border); border-radius: 16px; }}
aside a {{ display: block; padding: 6px 8px; color: var(--cp-link); text-decoration: none; border-radius: 0.625rem; font-size: 13px; }}
aside a:hover, aside a.active {{ background: var(--cp-accent-soft); }}
aside a.active {{ color: var(--cp-accent); font-weight: 700; }}
main {{ display: grid; gap: 14px; }}
.model-card {{ background: var(--cp-surface); border: 1px solid var(--cp-border); border-radius: 16px; padding: 18px; box-shadow: 0 0 2px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.14); }}
.model-card.target {{ border-color: var(--cp-accent); outline: 3px solid var(--cp-accent-soft); }}
.card-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; }}
.eyebrow {{ margin: 0 0 4px; color: var(--cp-text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
h2 {{ margin: 0; font-size: 20px; }}
h3 {{ margin: 16px 0 8px; font-size: 14px; color: var(--cp-text-muted); }}
.feature {{ background: var(--cp-accent-soft); color: var(--cp-accent); border: 1px solid var(--cp-border); border-radius: 999px; padding: 5px 10px; white-space: nowrap; font-size: 12px; font-weight: 700; }}
.why {{ color: var(--cp-text); }}
.note {{ margin: 10px 0; padding: 10px 12px; border-radius: 0.625rem; background: var(--cp-accent-soft); border: 1px solid var(--cp-border); color: var(--cp-text); font-size: 13px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px; color: var(--cp-text-muted); font-size: 12px; }}
code, pre {{ font-family: Consolas, "Courier New", Courier, monospace; }}
code {{ color: var(--cp-link); }}
.sample {{ display: flex; gap: 12px; align-items: flex-start; margin: 8px 0; }}
.sample-body {{ position: relative; flex: 1 1 auto; min-width: 0; }}
.thumb-link {{ flex: 0 0 auto; display: block; line-height: 0; border: 1px solid var(--cp-border); border-radius: 10px; overflow: hidden; background: var(--cp-surface-soft); text-decoration: none; }}
.thumb-link:hover {{ border-color: var(--cp-accent); }}
.thumb-link:focus-visible {{ outline: 2px solid var(--cp-accent); outline-offset: 2px; }}
.sample-thumb {{ display: block; width: 180px; height: 180px; object-fit: cover; cursor: zoom-in; }}
.negative-sample .sample-body pre {{ background: var(--cp-highlight); border-color: var(--cp-accent); }}
.negative-note {{ font-size: 13px; }}
pre {{ margin: 0; white-space: pre-wrap; background: var(--cp-surface-soft); color: var(--cp-text); border: 1px solid var(--cp-border); border-radius: 0.625rem; padding: 12px 86px 12px 12px; min-height: 52px; }}
.copy-btn {{ position: absolute; top: 8px; right: 8px; border: 1px solid var(--cp-border); background: var(--cp-accent); color: var(--cp-accent-fg); border-radius: 0.625rem; padding: 6px 10px; cursor: pointer; font: inherit; }}
.copy-btn:hover {{ background: var(--cp-accent-hover); }}
.hidden {{ display: none; }}
@media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} aside {{ position: static; height: auto; }} .toolbar {{ grid-template-columns: 1fr; }} }}
@media (max-width: 600px) {{ .sample {{ flex-direction: column; }} .sample-thumb {{ width: 100%; max-width: 240px; height: auto; }} }}
</style>
</head>
<body>
<header>
  <h1>LocalAI Studio model demo prompts</h1>
  <p class="sub">Every active catalog model has three copyable samples. Open with <code>?modelId=&lt;id&gt;</code>, <code>?model=&lt;id&gt;</code>, <code>?ollama=&lt;tag&gt;</code>, or a <code>#model-id</code> fragment to jump directly to a selected model.</p>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search models, features, categories, vendors...">
    <select id="category"><option value="">All categories</option></select>
  </div>
</header>
<div class="layout">
  <aside>
    <strong>Catalog models</strong>
    <nav>{nav_html}</nav>
  </aside>
  <main id="cards">{cards_html}</main>
</div>
<script>
const MODELS = {model_json};
const cards = Array.from(document.querySelectorAll('.model-card'));
const navLinks = Array.from(document.querySelectorAll('aside nav a[href^="#"]'));
const search = document.getElementById('search');
const category = document.getElementById('category');
const cats = Array.from(new Set(cards.map(card => card.dataset.category))).sort();
cats.forEach(cat => {{
  const opt = document.createElement('option');
  opt.value = cat;
  opt.textContent = cat;
  category.appendChild(opt);
}});
function normalize(value) {{
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
}}
function findTargetFromValue(raw) {{
  let decoded = '';
  try {{
    decoded = decodeURIComponent(raw || '');
  }} catch (_err) {{
    decoded = raw || '';
  }}
  const target = normalize(decoded);
  if (!target) return null;
  return cards.find(card => normalize(card.dataset.modelId) === target || normalize(card.id) === target || normalize(card.id.replace(/^model-/, '')) === target);
}}
function findTargetFromLocation() {{
  const params = new URLSearchParams(window.location.search);
  const raw = params.get('modelId') || params.get('model') || params.get('chatModel') || params.get('imageModel') || params.get('ollama') || params.get('ollamaTag') || (window.location.hash || '').replace(/^#/, '');
  return findTargetFromValue(raw);
}}
function applyFilters() {{
  const q = search.value.trim().toLowerCase();
  const cat = category.value;
  cards.forEach(card => {{
    const ok = (!q || card.dataset.search.includes(q) || card.textContent.toLowerCase().includes(q)) && (!cat || card.dataset.category === cat);
    card.classList.toggle('hidden', !ok);
  }});
}}
function clearTargetState() {{
  cards.forEach(card => card.classList.remove('target'));
  navLinks.forEach(link => link.classList.remove('active'));
}}
function focusTarget(target) {{
  if (!target) return;
  category.value = '';
  search.value = '';
  applyFilters();
  clearTargetState();
  target.classList.add('target');
  const activeLink = navLinks.find(link => (link.dataset.target || '').replace(/^#/, '') === target.id);
  if (activeLink) activeLink.classList.add('active');
  setTimeout(() => target.scrollIntoView({{ block: 'start' }}), 50);
}}
document.addEventListener('click', event => {{
  const btn = event.target.closest('.copy-btn');
  if (!btn) return;
  navigator.clipboard.writeText(btn.dataset.copy || '').then(() => {{
    const old = btn.textContent;
    btn.textContent = 'Copied';
    setTimeout(() => btn.textContent = old, 900);
  }});
}});
navLinks.forEach(link => {{
  link.addEventListener('click', event => {{
    const fragment = (link.dataset.target || link.getAttribute('href') || '').replace(/^#/, '');
    const target = document.getElementById(fragment);
    if (!target) return;
    event.preventDefault();
    if ((window.location.hash || '').replace(/^#/, '') !== fragment) {{
      history.pushState(null, '', '#' + fragment);
    }}
    focusTarget(target);
  }});
}});
window.addEventListener('hashchange', () => {{
  const target = findTargetFromValue((window.location.hash || '').replace(/^#/, ''));
  if (target) focusTarget(target);
}});
search.addEventListener('input', applyFilters);
category.addEventListener('change', applyFilters);
const target = findTargetFromLocation();
if (target) focusTarget(target);
</script>
</body>
</html>
'''


def write_demo_docs(models: list[dict], path: Path) -> None:
    path.write_text(build_demo_docs_html(models), encoding="utf-8")

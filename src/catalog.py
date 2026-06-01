# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Model catalog for LocalAI Studio.

Built-in MODELS list is the authoritative default. At runtime the app reads
models_catalog.json (project root) via load_catalog(); ensure_catalog_file()
creates that file from MODELS the first time the app runs.  Users can edit the
JSON to add, remove, or modify models without touching Python code.
"""

import json
from pathlib import Path

from src import logger as _log
from src.persistence import atomic_write_json

# Path to the user-editable catalog file (same directory as config.json)
CATALOG_FILE = Path(__file__).parent.parent / "models_catalog.json"

# Fields every model entry must supply.
# Image generation models use "comfyui_model" instead of "ollama_tag".
_REQUIRED_FIELDS = {
    "id", "name", "vendor", "category",
    "size_gb", "min_ram_gb", "min_vram_gb",
}
# Text models need ollama_tag; image models need comfyui_model; benchmark
# adapter models can use hf_repo/onnx_repo plus a non-Ollama backend.
# Validated separately in _validate_model.

_CATALOG_README = (
    "LocalAI Studio model catalog — add, remove, or edit models here. "
    "Each entry maps to a model card in the Models tab. "
    "Fields: id (unique key), name, vendor, category "
    "(Ultra Small|Small|Medium|Large|Extra Large|Vision|Speech|Embeddings|"
    "Document AI|Image Generation), "
    "description, parameters, size_gb (download size), min_ram_gb, "
    "min_vram_gb (set 0 for CPU-only), context_length, "
    "ollama_tag (ollama pull tag — required for text models), "
    "comfyui_model (checkpoint filename — required for image generation models), "
    "comfyui_model_url (direct download URL for the checkpoint, optional), "
    "learn_more_url (optional exact external model page), "
    "backend (omit or 'ollama' for text models; 'comfyui' for image models; "
    "or a benchmark adapter backend such as transformers, onnxruntime, "
    "onnx-genai, openvino, sentence-transformers/onnx, faster-whisper/onnx), "
    "hf_repo/onnx_repo/ov_repo (HuggingFace repo or null), hf_revision "
    "(required for trust_remote_code utility models AND for any user_added "
    "entry — always a resolved 40-char commit SHA, never a branch name), "
    "onnx_subfolder (optional), "
    "tags (list of strings), "
    "chat_selectable (optional bool override for Chat page visibility), "
    "ollama_num_ctx (optional per-model Ollama chat context cap), "
    "ollama_keep_alive (optional Ollama keep_alive value such as '30m'), "
    "benchmark_skip_reason (optional reason to remove a known-bad model from "
    "pass/fail benchmark suites). "
    "Image Generation models may set supports_img2img=true and img2img_workflows "
    "(denoise_default/min/max) to enable reference-image generation via ComfyUI. "
    "Image Generation models MUST also include recommended_settings AND "
    "perf_profile blocks (validated at startup). "
    "Optional Add-from-Hugging-Face fields: user_added=true (marks an entry "
    "added via the Models > + Add from Hugging Face dialog), requires_review=true "
    "(surfaces a banner in the detail pane when the compatibility verdict was "
    "warn or the user overrode unsupported), source_url (the original pasted "
    "Hugging Face / Ollama URL), added_at (ISO 8601 UTC timestamp). "
    "Optional SKU compatibility is computed at runtime when a private SKU file is present. "
    "By default merge_builtins=true adds new built-in defaults on app updates; "
    "set merge_builtins=false for JSON-only behavior, or list IDs in "
    "disabled_builtin_ids to hide selected built-in defaults. "
    "Reload takes effect via Settings > Reload Catalog (no restart needed). "
    "Unknown fields are preserved on reload; keep curation notes in external "
    "reports rather than in the shipped catalog."
)

_NON_CHAT_CATEGORIES = {"Image Generation", "Speech", "Embeddings", "Document AI"}
_CHAT_BACKENDS = {"", "ollama", "onnx", "onnxruntime", "onnx-genai", "openvino"}

MODELS = [{'id': 'qwen2.5:0.5b',
  'name': 'Qwen 2.5 0.5B',
  'vendor': 'Alibaba',
  'category': 'Ultra Small',
  'description': "Alibaba's tiniest Qwen model. Surprisingly capable for its size. Good for quick "
                 'tests on any hardware.',
  'parameters': '0.5B',
  'size_gb': 0.4,
  'min_ram_gb': 4,
  'min_vram_gb': 1,
  'context_length': 32768,
  'ollama_tag': 'qwen2.5:0.5b',
  'onnx_repo': None,
  'tags': ['tiny', 'fast', 'low-memory', 'chat']},
 {'id': 'llama3.2:1b',
  'name': 'Llama 3.2 1B',
  'vendor': 'Meta',
  'category': 'Ultra Small',
  'description': "Meta's compact 1B model. Strong reasoning for its size, runs well on CPU.",
  'parameters': '1B',
  'size_gb': 0.8,
  'min_ram_gb': 4,
  'min_vram_gb': 1,
  'context_length': 131072,
  'ollama_tag': 'llama3.2:1b',
  'onnx_repo': None,
  'tags': ['tiny', 'fast', 'low-memory', 'chat']},
 {'id': 'llama3.2:3b',
  'name': 'Llama 3.2 3B',
  'vendor': 'Meta',
  'category': 'Small',
  'description': "Meta's 3B model with a massive 128K context window. Great for document analysis "
                 'on modest hardware.',
  'parameters': '3B',
  'size_gb': 1.9,
  'min_ram_gb': 6,
  'min_vram_gb': 3,
  'context_length': 131072,
  'ollama_tag': 'llama3.2:3b',
  'onnx_repo': None,
  'tags': ['small', 'long-context', 'chat']},
 {'id': 'phi4:mini',
  'name': 'Phi-4 Mini 3.8B',
  'vendor': 'Phi',
  'category': 'Small',
  'description': "Latest Phi Mini, improved on Phi-3.5 across benchmarks. Great coding "
                 'and STEM performance.',
  'parameters': '3.8B',
  'size_gb': 2.5,
  'min_ram_gb': 8,
  'min_vram_gb': 0,
  'context_length': 16384,
  'ollama_tag': 'phi4-mini',
  'onnx_repo': None,
  'tags': ['coding', 'local-no-api', 'ollama', 'phase1-shortlist', 'reasoning', 'small', 'stem']},
 {'id': 'llama3.2-vision:11b',
  'name': 'Llama 3.2 Vision 11B',
  'vendor': 'Meta',
  'category': 'Vision',
  'description': "Meta's multimodal 11B model with image understanding. An older, heavier "
                 'alternative to Gemma 3 for the Analyze → Prompt feature on the Image Generation '
                 'page. Prefer Gemma 3 4B (Recommended) or Gemma 3 12B (Highest Accuracy) unless '
                 'you specifically want this model.',
  'parameters': '11B',
  'size_gb': 8.0,
  'min_ram_gb': 16,
  'min_vram_gb': 10,
  'context_length': 131072,
  'ollama_tag': 'llama3.2-vision:11b',
  'onnx_repo': None,
  'tags': ['vision', 'multimodal', 'image-analysis', 'analyze-prompt', 'legacy'],
  'supports_vision': True},
 {'id': 'gemma3:4b-vision',
  'name': 'Gemma 3 4B',
  'vendor': 'Google',
  'category': 'Vision',
  'description': "Google's Gemma 3 4B multimodal model. Best balance of accuracy and speed for "
                 'everyday image understanding and the Analyze → Prompt feature on the Image '
                 'Generation page. Recommended default for vision tasks.',
  'parameters': '4B',
  'size_gb': 3.3,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 131072,
  'ollama_tag': 'gemma3:4b',
  'onnx_repo': None,
  'benchmark_num_predict': 2048,
  'tags': ['vision', 'multimodal', 'image-analysis', 'analyze-prompt'],
  'supports_vision': True,
  'recommendation_badge': 'Recommended',
  'tradeoff_note': 'Best balance of accuracy and speed for most image tasks.',
  'speed_tier': 'fast',
  'accuracy_tier': 'high'},
 {'id': 'gemma3:12b-vision',
  'name': 'Gemma 3 12B',
  'vendor': 'Google',
  'category': 'Vision',
  'description': "Google's Gemma 3 12B multimodal model. Highest accuracy and prompt fidelity for "
                 'image understanding — slower than Gemma 3 4B but worth it when quality matters '
                 'more than latency.',
  'parameters': '12B',
  'size_gb': 8.1,
  'min_ram_gb': 16,
  'min_vram_gb': 0,
  'context_length': 131072,
  'ollama_tag': 'gemma3:12b',
  'onnx_repo': None,
  'benchmark_num_predict': 2048,
  'tags': ['accuracy',
           'analyze-prompt',
           'image-analysis',
           'local-no-api',
           'multimodal',
           'ollama',
           'phase1-shortlist',
           'vision'],
  'supports_vision': True,
  'recommendation_badge': 'Highest Accuracy',
  'tradeoff_note': 'Highest measured accuracy; slower than Gemma 3 4B.',
  'speed_tier': 'medium',
  'accuracy_tier': 'highest'},
 {'id': 'minicpm-v-vision',
  'name': 'MiniCPM-V',
  'vendor': 'OpenBMB',
  'category': 'Vision',
  'description': "OpenBMB's MiniCPM-V — efficient multimodal model with strong practical image "
                 'understanding and good OCR / multilingual behavior at lower VRAM than 12B-class '
                 'vision models.',
  'parameters': '8B',
  'size_gb': 5.5,
  'min_ram_gb': 12,
  'min_vram_gb': 6,
  'context_length': 32768,
  'ollama_tag': 'minicpm-v:latest',
  'onnx_repo': None,
  'tags': ['vision', 'multimodal', 'image-analysis', 'analyze-prompt', 'ocr', 'multilingual'],
  'supports_vision': True,
  'recommendation_badge': 'Efficient Multimodal',
  'tradeoff_note': 'Efficient multimodal model with good OCR and multilingual support.',
  'speed_tier': 'medium-fast',
  'accuracy_tier': 'high'},
 {'id': 'deepseek-r1:32b',
  'name': 'DeepSeek-R1 32B Distill Q4',
  'vendor': 'DeepSeek',
  'category': 'Large',
  'description': 'R1 reasoning at 32B scale, quantised to Q4. Near frontier-level reasoning '
                 'performance. Requires ~20 GB VRAM (24 GB GPU).',
  'parameters': '32B',
  'size_gb': 19.9,
  'min_ram_gb': 32,
  'min_vram_gb': 20,
  'context_length': 131072,
  'ollama_tag': 'deepseek-r1:32b',
  'onnx_repo': None,
  'tags': ['coding',
           'large',
           'local-no-api',
           'math',
           'ollama',
           'phase1-shortlist',
           'reasoning']},
 {'id': 'sdxl-base',
  'name': 'SDXL 1.0 Base',
  'vendor': 'Stability AI',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'Stable Diffusion XL base model. Much higher quality than SD 1.5, especially for '
                 '1024×1024. Requires 8 GB VRAM (8 GB+ GPU).\n'
                 '\n'
                 'Speed/quality: balanced reference SDXL at 1024². ~17–22 s on 24 GB VRAM. Best '
                 'baseline for high-res output.',
  'parameters': '3.5B UNet',
  'size_gb': 6.5,
  'min_ram_gb': 16,
  'min_vram_gb': 8,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'sd_xl_base_1.0.safetensors',
  'comfyui_model_url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors',
  'default_width': 1024,
  'default_height': 1024,
  'tags': ['comfyui',
           'diffusion',
           'high-quality',
           'image-generation',
           'local-no-api',
           'phase1-shortlist',
           'sdxl'],
  'comfyui_family': 'sdxl',
  'showcase_prompt': 'a majestic golden eagle in mid-flight soaring over snow-capped Alpine '
                     'mountains at golden hour, cinematic lighting, sharp focus, photorealistic, '
                     '8k detail, magazine cover',
  'hf_rank': 2,
  'tradeoff_note': 'Speed/quality: balanced reference SDXL at 1024². ~17–22 s on 24 GB VRAM. Best '
                   'baseline for high-res output.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'dpmpp_2m',
                           'scheduler': 'karras',
                           'steps': 30,
                           'cfg': 7.0,
                           'cfg_locked': False,
                           'family_label': 'SDXL'},
  'perf_profile': {'speed_tier': 'balanced',
                   'quality_tier': 'great',
                   'category_bucket': 'general',
                   'recommendation': 'recommended',
                   'speed_label': '~15s',
                   'notes': 'Solid SDXL workhorse — runs almost anywhere with 8 GB+ VRAM.'},
  'supports_img2img': True,
  'img2img_min_vram_gb': 8,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'flux1-schnell-q4',
  'name': 'Flux.1 Schnell Q4',
  'vendor': 'Black Forest Labs',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'Flux.1 Schnell quantised to Q4. State-of-the-art image quality at high speed. '
                 'Prompt following far exceeds SD/SDXL. Requires 8 GB VRAM (8 GB+ GPU).\n'
                 '\n'
                 'Speed/quality: SPEED — 1–4 step distilled Flux Schnell, Q4 quant. ~6–15 s at '
                 '1024². Lower fidelity than Dev.',
  'parameters': '12B (Q4)',
  'size_gb': 6.6,
  'min_ram_gb': 16,
  'min_vram_gb': 8,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'flux1-schnell-Q4_K_S.gguf',
  'comfyui_model_url': 'https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_K_S.gguf',
  'default_width': 1024,
  'default_height': 1024,
  'tags': ['comfyui',
           'fast',
           'flux',
           'high-quality',
           'image-generation',
           'local-no-api',
           'phase1-shortlist'],
  'tradeoff_note': 'Speed/quality: SPEED — 1–4 step distilled Flux Schnell, Q4 quant. ~6–15 s at '
                   '1024². Lower fidelity than Dev.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'euler',
                           'scheduler': 'simple',
                           'steps': 4,
                           'cfg': 1.0,
                           'cfg_locked': True,
                           'family_label': 'Flux'},
  'perf_profile': {'speed_tier': 'fast',
                   'quality_tier': 'excellent',
                   'category_bucket': 'speed',
                   'recommendation': 'top_pick',
                   'speed_label': '~6s',
                   'notes': 'Fast Flux — near-Dev quality in 4 steps; outstanding speed/quality.'},
  'supports_img2img': False,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'flux1-dev-q4',
  'name': 'Flux.1 Dev Q4',
  'vendor': 'Black Forest Labs',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'Flux.1 Dev quantised to Q4. Highest quality of the Flux family — slower than '
                 'Schnell but more detailed outputs. Requires 10 GB VRAM (12 GB+ GPU).\n'
                 '\n'
                 'Speed/quality: MEMORY — 4-bit Flux.1 Dev. Runs in ~10 GB VRAM with mild quality '
                 'loss vs FP16. ~45–70 s.',
  'parameters': '12B (Q4)',
  'size_gb': 10.0,
  'min_ram_gb': 20,
  'min_vram_gb': 10,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'flux1-dev-Q4_K_S.gguf',
  'comfyui_model_url': 'https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf',
  'default_width': 1024,
  'default_height': 1024,
  'tags': ['image-generation', 'flux', 'high-quality', 'detailed'],
  'comfyui_family': 'flux_gguf',
  'showcase_prompt': 'A cozy bookshop interior at twilight, a calico cat sitting in the window '
                     'with a wooden sign that reads "Open Late" hanging beside it, golden hour '
                     'light spilling between bookshelves, photorealistic, shallow depth of field',
  'hf_rank': 26,
  'tradeoff_note': 'Speed/quality: MEMORY — 4-bit Flux.1 Dev. Runs in ~10 GB VRAM with mild '
                   'quality loss vs FP16. ~45–70 s.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'euler',
                           'scheduler': 'simple',
                           'steps': 24,
                           'cfg': 1.0,
                           'cfg_locked': True,
                           'family_label': 'Flux'},
  'perf_profile': {'speed_tier': 'balanced',
                   'quality_tier': 'sota',
                   'category_bucket': 'quality',
                   'recommendation': 'recommended',
                   'speed_label': '~25s',
                   'notes': 'Best quality you can fit in 10 GB; Flux Dev quantized to Q4.'},
  'supports_img2img': False,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'flux1-dev-fp16',
  'name': 'Flux.1 Dev FP16',
  'vendor': 'Black Forest Labs',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'Flux.1 Dev at full FP16 precision. Maximum quality — the reference '
                 'implementation. Needs ~24 GB VRAM (24 GB GPU only).\n'
                 '\n'
                 'Speed/quality: QUALITY (slow + heavy) — full-precision Flux.1 Dev. ~60–90 s, '
                 'requires ~24 GB VRAM. Best Flux output.',
  'parameters': '12B (FP16)',
  'size_gb': 23.8,
  'min_ram_gb': 32,
  'min_vram_gb': 24,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'flux1-dev.safetensors',
  'comfyui_model_url': 'https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev.safetensors',
  'default_width': 1024,
  'default_height': 1024,
  'tags': ['comfyui',
           'flux',
           'fp16',
           'image-generation',
           'local-no-api',
           'maximum-quality',
           'phase1-shortlist'],
  'tradeoff_note': 'Speed/quality: QUALITY (slow + heavy) — full-precision Flux.1 Dev. ~60–90 s, '
                   'requires ~24 GB VRAM. Best Flux output.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'euler',
                           'scheduler': 'simple',
                           'steps': 28,
                           'cfg': 1.0,
                           'cfg_locked': True,
                           'family_label': 'Flux'},
  'perf_profile': {'speed_tier': 'slow',
                   'quality_tier': 'sota',
                   'category_bucket': 'quality',
                   'recommendation': 'top_pick',
                   'speed_label': '~45s',
                   'notes': 'Flagship Flux at full precision — needs 24 GB VRAM.'},
  'supports_img2img': False,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'juggernaut-xl-v9',
  'name': 'Juggernaut XL v9',
  'vendor': 'RunDiffusion',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': "RunDiffusion's premier SDXL photorealism fine-tune. Exceptional skin texture, "
                 'natural lighting, and photojournalistic quality at 1024×1024. Best SDXL model '
                 'for executive demos. Runs on 8 GB VRAM.\n'
                 '\n'
                 'Speed/quality: QUALITY — SDXL photoreal fine-tune. ~20–25 s. Top pick for '
                 'cinematic realism.',
  'parameters': '3.5B UNet',
  'size_gb': 6.6,
  'min_ram_gb': 16,
  'min_vram_gb': 8,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',
  'comfyui_model_url': 'https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',
  'default_width': 1024,
  'default_height': 1024,
  'tags': ['image-generation', 'sdxl', 'photorealism', 'high-quality', 'portrait'],
  'tradeoff_note': 'Speed/quality: QUALITY — SDXL photoreal fine-tune. ~20–25 s. Top pick for '
                   'cinematic realism.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'dpmpp_2m_sde',
                           'scheduler': 'karras',
                           'steps': 32,
                           'cfg': 6.5,
                           'cfg_locked': False,
                           'family_label': 'SDXL'},
  'perf_profile': {'speed_tier': 'balanced',
                   'quality_tier': 'excellent',
                   'category_bucket': 'photo',
                   'recommendation': 'top_pick',
                   'speed_label': '~18s',
                   'notes': 'Top SDXL photoreal fine-tune — portraits, scenes, products.'},
  'supports_img2img': True,
  'img2img_min_vram_gb': 8,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'realistic-vision-v6',
  'name': 'Realistic Vision v6.0',
  'vendor': 'SG161222',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'The definitive SD 1.5 photorealism fine-tune. Produces strikingly realistic '
                 'portraits and scenes at 512–768 px. Fast generation, 4 GB VRAM, enormous LoRA '
                 'compatibility. No VAE baked in — pair with vae-ft-mse-840000 for best color.\n'
                 '\n'
                 'Speed/quality: balanced SD1.5 photoreal fine-tune. ~5–10 s on 24 GB VRAM. Best '
                 'SD1.5 choice for portraits.',
  'parameters': '860M UNet',
  'size_gb': 2.0,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'Realistic_Vision_V6.0_NV_B1_fp16.safetensors',
  'comfyui_model_url': 'https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_fp16.safetensors',
  'default_width': 512,
  'default_height': 512,
  'tags': ['image-generation', 'sd15', 'photorealism', 'portrait', 'fast'],
  'igpu_viable': True,
  'tradeoff_note': 'Speed/quality: balanced SD1.5 photoreal fine-tune. ~5–10 s on 24 GB VRAM. Best '
                   'SD1.5 choice for portraits.',
  'recommended_settings': {'width': 512,
                           'height': 768,
                           'aspect': 'SD 512×768',
                           'sampler': 'dpmpp_2m',
                           'scheduler': 'karras',
                           'steps': 30,
                           'cfg': 6.0,
                           'cfg_locked': False,
                           'family_label': 'SD 1.5'},
  'perf_profile': {'speed_tier': 'fast',
                   'quality_tier': 'great',
                   'category_bucket': 'photo',
                   'recommendation': 'top_pick',
                   'speed_label': '~4s',
                   'notes': 'Tiny SD 1.5 photoreal — runs anywhere; very fast iteration.'},
  'supports_img2img': True,
  'img2img_min_vram_gb': 4,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True},
  'cpu_viable': True,
  'expected_cpu_time_label': '~90s on small CPU'},
 {'id': 'z-image-turbo',
  'name': 'Z-Image Turbo',
  'vendor': 'Tongyi-MAI / Comfy-Org',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': "Z-Image's distilled fast variant, repackaged by Comfy-Org for native ComfyUI "
                 'use. Generates high-quality photorealistic images in just 8 steps using a Qwen 3 '
                 'text encoder and Flux VAE — blazing fast with strong prompt adherence. 12 GB '
                 'VRAM. Support files (~6 GB for Qwen encoder + VAE) are downloaded '
                 'automatically.\n'
                 '\n'
                 'Speed/quality: SPEED — distilled Z-Image. ~5–10 s, near-base quality.',
  'parameters': 'DiT (bf16)',
  'size_gb': 12.3,
  'min_ram_gb': 24,
  'min_vram_gb': 12,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'z_image_turbo_bf16.safetensors',
  'comfyui_model_url': 'https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors',
  'comfyui_model_dir': 'diffusion_models',
  'default_width': 1024,
  'default_height': 1024,
  'tags': ['image-generation', 'fast', 'turbo', 'high-quality', 'photorealism'],
  'tradeoff_note': 'Speed/quality: SPEED — distilled Z-Image. ~5–10 s, near-base quality.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'euler',
                           'scheduler': 'simple',
                           'steps': 8,
                           'cfg': 1.0,
                           'cfg_locked': True,
                           'family_label': 'Z-Image'},
  'perf_profile': {'speed_tier': 'fast',
                   'quality_tier': 'excellent',
                   'category_bucket': 'speed',
                   'recommendation': 'top_pick',
                   'speed_label': '~8s',
                   'notes': 'Best speed/quality balance — 8-step distilled Z-Image.'},
  'supports_img2img': False,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'bytedance-sdxl-lightning',
  'name': 'SDXL Lightning',
  'vendor': 'ByteDance',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'ByteDance distilled SDXL — sharp photorealism in 8 steps via the full ComfyUI '
                 'checkpoint. (SDXL Lightning, HF #14, ♥ 2,148)\n'
                 '\n'
                 'Speed/quality: SPEED — 2/4/8-step SDXL distillation. ~3–8 s at 1024². '
                 'Near-base quality with the right step count.',
  'parameters': '',
  'size_gb': 6.9,
  'min_ram_gb': 16,
  'min_vram_gb': 8,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'sdxl_lightning_8step.safetensors',
  'comfyui_model_url': 'https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors',
  'comfyui_model_dest': 'checkpoints',
  'comfyui_family': 'sdxl_lightning',
  'default_width': 1024,
  'default_height': 1024,
  'default_steps': 8,
  'default_cfg': 1.0,
  'default_sampler': 'euler',
  'default_scheduler': 'sgm_uniform',
  'showcase_prompt': 'a Bengal tiger leaping through a wall of crystal-clear water droplets, '
                     'hyper-detailed fur, slow-motion, dramatic side lighting, nature photography',
  'hf_repo': 'ByteDance/SDXL-Lightning',
  'hf_rank': 14,
  'tags': ['image-generation', 'diffusion', 'sdxl_lightning'],
  'igpu_viable': True,
  'tradeoff_note': 'Speed/quality: SPEED — 2/4/8-step SDXL distillation using the full ComfyUI '
                   'checkpoint. ~3–8 s at 1024². Near-base quality with the right step count.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'euler',
                           'scheduler': 'sgm_uniform',
                           'steps': 8,
                           'cfg': 1.0,
                           'cfg_locked': True,
                           'family_label': 'SDXL Lightning'},
  'perf_profile': {'speed_tier': 'fast',
                   'quality_tier': 'great',
                   'category_bucket': 'speed',
                   'recommendation': 'recommended',
                   'speed_label': '~4s',
                   'notes': '8-step Lightning UNet — keeps SDXL quality at distill speed.'},
  'supports_img2img': True,
  'img2img_min_vram_gb': 8,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'playgroundai-playground-v2.5-1024px-aesthetic',
  'name': 'playground v2.5 1024px aesthetic',
  'vendor': 'playgroundai',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'Playground v2.5 — SDXL-arch retrained from scratch for aesthetics. Top-tier '
                 'color and contrast. (Stable Diffusion XL, HF #55, ♥ 763)\n'
                 '\n'
                 'Speed/quality: QUALITY — SDXL-based aesthetic fine-tune. ~20–28 s. Excellent for '
                 'stylized photo-art.',
  'parameters': '',
  'size_gb': 6.5,
  'min_ram_gb': 16,
  'min_vram_gb': 8,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'playground-v2.5-1024px-aesthetic.fp16.safetensors',
  'comfyui_model_url': 'https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/resolve/main/playground-v2.5-1024px-aesthetic.fp16.safetensors',
  'comfyui_model_dest': 'checkpoints',
  'comfyui_family': 'sdxl',
  'default_width': 1024,
  'default_height': 1024,
  'default_steps': 28,
  'default_cfg': 3.0,
  'default_sampler': 'dpmpp_2m',
  'default_scheduler': 'karras',
  'showcase_prompt': 'a hyperrealistic close-up of an iridescent beetle on a dewdrop-covered leaf, '
                     'macro photography, soft morning light, shallow DOF, National Geographic '
                     'style',
  'hf_repo': 'playgroundai/playground-v2.5-1024px-aesthetic',
  'hf_rank': 55,
  'tags': ['image-generation', 'diffusion', 'sdxl'],
  'tradeoff_note': 'Speed/quality: QUALITY — SDXL-based aesthetic fine-tune. ~20–28 s. Excellent '
                   'for stylized photo-art.',
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'dpmpp_2m',
                           'scheduler': 'karras',
                           'steps': 30,
                           'cfg': 3.0,
                           'cfg_locked': False,
                           'family_label': 'Playground v2.5'},
  'perf_profile': {'speed_tier': 'balanced',
                   'quality_tier': 'excellent',
                   'category_bucket': 'photo',
                   'recommendation': 'top_pick',
                   'speed_label': '~18s',
                   'notes': 'Aesthetic SDXL successor — CFG=3 by design; vibrant results.'},
  'supports_img2img': True,
  'img2img_min_vram_gb': 8,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'gsdf-counterfeit-v3.0',
  'name': 'Counterfeit V3.0',
  'vendor': 'gsdf',
  'category': 'Image Generation',
  'backend': 'comfyui',
  'description': 'Counterfeit V3.0 — last in the seminal SD 1.5 anime series. (Stable Diffusion '
                 '1.5, HF #74, ♥ 562)\n'
                 '\n'
                 'Speed/quality: balanced SD1.5 anime/illustration merge. ~5–10 s.',
  'parameters': '',
  'size_gb': 4.0,
  'min_ram_gb': 16,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': '',
  'comfyui_model': 'Counterfeit-V3.0_fp16.safetensors',
  'comfyui_model_url': 'https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fp16.safetensors',
  'comfyui_model_dest': 'checkpoints',
  'comfyui_family': 'sd15',
  'default_width': 512,
  'default_height': 768,
  'default_steps': 30,
  'default_cfg': 7.0,
  'default_sampler': 'dpmpp_2m',
  'default_scheduler': 'karras',
  'showcase_prompt': 'masterpiece, best quality, beautiful anime girl with long flowing pink hair, '
                     'sakura petals, traditional shrine background, kimono with intricate floral '
                     'pattern, soft natural light',
  'hf_repo': 'gsdf/Counterfeit-V3.0',
  'hf_rank': 74,
  'tags': ['image-generation', 'diffusion', 'sd15'],
  'igpu_viable': True,
  'tradeoff_note': 'Speed/quality: balanced SD1.5 anime/illustration merge. ~5–10 s.',
  'recommended_settings': {'width': 512,
                           'height': 768,
                           'aspect': 'SD 512×768',
                           'sampler': 'dpmpp_2m_sde',
                           'scheduler': 'karras',
                           'steps': 30,
                           'cfg': 7.0,
                           'cfg_locked': False,
                           'family_label': 'SD 1.5'},
  'perf_profile': {'speed_tier': 'fast',
                   'quality_tier': 'great',
                   'category_bucket': 'art',
                   'recommendation': 'alternative',
                   'speed_label': '~5s',
                   'notes': 'Polished anime SD 1.5 v3 — clean lineart and color.'},
  'supports_img2img': True,
  'img2img_min_vram_gb': 4,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True},
  'cpu_viable': True,
  'expected_cpu_time_label': '~90s on small CPU'},
 {'id': 'sdxl-lowvram',
  'name': 'SDXL Low VRAM',
  'vendor': 'Stability AI',
  'category': 'Image Generation',
  'description': 'Low-VRAM SDXL profile for small-GPU demonstrations using the SDXL Base '
                 'checkpoint.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 16,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'benchmark_skip_reason': 'SDXL Low VRAM benchmark path is covered by sdxl-base; the '
                           '--lowvram cold-start on 24 GB SKUs is non-deterministic '
                           '(ComfyUI subprocess exits during pytorch attention init).',
  'tags': ['comfyui --lowvram', 'local-no-api', 'phase1-shortlist'],
  'comfyui_model': 'sd_xl_base_1.0.safetensors',
  'comfyui_model_url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors',
  'default_width': 1024,
  'default_height': 1024,
  'recommended_settings': {'width': 1024,
                           'height': 1024,
                           'aspect': 'Square 1:1',
                           'sampler': 'dpmpp_2m',
                           'scheduler': 'karras',
                           'steps': 30,
                           'cfg': 7.0,
                           'cfg_locked': False,
                           'family_label': 'SDXL'},
  'perf_profile': {'speed_tier': 'balanced',
                   'quality_tier': 'great',
                   'category_bucket': 'general',
                   'recommendation': 'recommended',
                   'speed_label': '~15s',
                   'notes': 'Solid SDXL workhorse — runs almost anywhere with 8 GB+ VRAM.'},
  'backend': 'comfyui',
  'comfyui_launch_flags': ['--lowvram'],
  'supports_img2img': True,
  'img2img_min_vram_gb': 4,
  'img2img_workflows': {'denoise_default': 0.75,
                        'denoise_min': 0.1,
                        'denoise_max': 1.0,
                        'supports_denoise_override': True}},
 {'id': 'whisper-large-v3-turbo',
  'name': 'Whisper Large v3 Turbo',
  'vendor': 'OpenAI',
  'category': 'Speech',
  'description': 'Offline transcription benchmark Phase 1 benchmark adapter entry; requires the '
                 'matching app/runtime surface before exposing as an end-user workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': 'openai/whisper-large-v3-turbo',
  'tags': ['faster-whisper-onnx', 'local-no-api', 'phase1-shortlist'],
  'backend': 'faster-whisper/onnx',
  'hf_repo': 'openai/whisper-large-v3-turbo',
  'phase1_adapter': True},
 {'id': 'deepseek-r1-1.5b',
  'name': 'DeepSeek-R1 Distill 1.5B',
  'vendor': 'DeepSeek',
  'category': 'Small',
  'description': 'Watch chain-of-thought style reasoning on tiny hardware',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 4,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': 'deepseek-r1:1.5b',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'gemma3-27b',
  'name': 'Gemma 3 27B',
  'vendor': 'Google',
  'category': 'Vision',
  'description': 'Google flagship local multimodal model',
  'parameters': '27.4B',
  'size_gb': 16.2,
  'min_ram_gb': 32,
  'min_vram_gb': 17,
  'context_length': 131072,
  'ollama_tag': 'gemma3:27b',
  'benchmark_num_predict': 2048,
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'llama3.3',
  'name': 'Llama 3.3 70B',
  'vendor': 'Meta',
  'category': 'Extra Large',
  'description': 'Largest local general model in pack',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 64,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': 'llama3.3',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'mistral-nemo',
  'name': 'Mistral Nemo',
  'vendor': 'Mistral/NVIDIA',
  'category': 'Small',
  'description': '128K long-context benchmark',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 16,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': 'mistral-nemo',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'mistral-small3.2',
  'name': 'Mistral Small 3.2 24B',
  'vendor': 'Mistral',
  'category': 'Vision',
  'description': 'Vision + function calling + long context',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 32,
  'min_vram_gb': 15,
  'context_length': 0,
  'ollama_tag': 'mistral-small3.2',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'phi4',
  'name': 'Phi-4 14B',
  'vendor': 'Phi',
  'category': 'Small',
  'description': 'Phi flagship general local model',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 16,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': 'phi4',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'qwen2.5-coder-7b',
  'name': 'Qwen2.5 Coder 7B',
  'vendor': 'Qwen',
  'category': 'Medium',
  'description': 'Local coding assistant baseline',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 12,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': 'qwen2.5-coder:7b',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'qwen2.5vl-32b',
  'name': 'Qwen2.5-VL 32B',
  'vendor': 'Qwen',
  'category': 'Vision',
  'description': 'Best open chart/document/screenshot reasoning',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 32,
  'min_vram_gb': 20,
  'context_length': 0,
  'ollama_tag': 'qwen2.5vl:32b',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'qwen2.5vl-3b',
  'name': 'Qwen2.5-VL 3B',
  'vendor': 'Qwen',
  'category': 'Vision',
  'description': 'Small GPU vision/screen understanding',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': 'qwen2.5vl:3b',
  'onnx_repo': None,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'qwen2.5vl-7b',
  'name': 'Qwen2.5-VL 7B',
  'vendor': 'Qwen',
  'category': 'Vision',
  'description': 'Charts, screenshots, OCR, image Q&A',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 16,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': 'qwen2.5vl:7b',
  'onnx_repo': None,
  'benchmark_num_predict': 2048,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'qwen3-30b-a3b',
  'name': 'Qwen3 30B-A3B',
  'vendor': 'Qwen',
  'category': 'Extra Large',
  'description': 'MoE efficiency benchmark',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 32,
  'min_vram_gb': 12,
  'context_length': 0,
  'ollama_tag': 'qwen3:30b-a3b',
  'onnx_repo': None,
  'benchmark_num_predict': 4096,
  'tags': ['local-no-api', 'ollama', 'phase1-shortlist']},
 {'id': 'whisper-v3-turbo-gpu',
  'name': 'Whisper v3 Turbo GPU',
  'vendor': 'OpenAI',
  'category': 'Speech',
  'description': 'GPU ASR benchmark Phase 1 benchmark adapter entry; requires the matching '
                 'app/runtime surface before exposing as an end-user workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': 'openai/whisper-large-v3-turbo',
  'tags': ['local-no-api', 'onnx-faster-whisper', 'phase1-shortlist'],
  'backend': 'onnx/faster-whisper',
  'hf_repo': 'openai/whisper-large-v3-turbo',
  'phase1_adapter': True},
 {'id': 'all-minilm',
  'name': 'All-MiniLM Sentence Embeddings',
  'vendor': 'Sentence Transformers',
  'category': 'Embeddings',
  'description': 'Fast CPU embedding baseline Phase 1 benchmark adapter entry; requires the '
                 'matching app/runtime surface before exposing as an end-user workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 4,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': 'sentence-transformers/all-MiniLM-L6-v2',
  'tags': ['local-no-api', 'phase1-shortlist', 'sentence-transformers-onnx'],
  'backend': 'sentence-transformers/onnx',
  'hf_repo': 'sentence-transformers/all-MiniLM-L6-v2',
  'phase1_adapter': True},
 {'id': 'florence-2-base',
  'name': 'Florence-2 Base',
  'vendor': 'Florence',
  'category': 'Vision',
  'description': 'Caption/OCR/object detection from a single multimodal model. Phase 1 benchmark adapter '
                 'entry; requires the matching app/runtime surface before exposing as an end-user '
                 'workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'tags': ['local-no-api', 'phase1-shortlist', 'transformers'],
  'backend': 'transformers',
  'hf_repo': 'microsoft/Florence-2-base',
  'hf_revision': '5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac',
  'phase1_adapter': True},
 {'id': 'speecht5-tts',
  'name': 'SpeechT5 TTS',
  'vendor': 'SpeechT5',
  'category': 'Speech',
  'description': 'Local speech output from any model response Phase 1 benchmark adapter entry; '
                 'requires the matching app/runtime surface before exposing as an end-user '
                 'workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'tags': ['local-no-api', 'phase1-shortlist', 'transformers'],
  'backend': 'transformers',
  'hf_repo': 'microsoft/speecht5_tts',
  'phase1_adapter': True},
 {'id': 'piper-tts',
  'name': 'Piper TTS',
  'vendor': 'Rhasspy',
  'category': 'Speech',
  'description': 'Fast, fully offline text-to-speech with a per-voice ONNX model. CPU-friendly '
                 '(~150 MB per voice, no GPU required). Used by the Toolbox Speak workflow with a '
                 'multi-language voice picker; not part of the benchmark harness.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 4,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'tags': ['local-no-api', 'transformers', 'speech', 'toolbox-only'],
  'backend': 'onnx',
  'hf_repo': 'rhasspy/piper-voices',
  'phase1_adapter': True},
 {'id': 'table-transformer',
  'name': 'Table Transformer',
  'vendor': 'Table Transformer',
  'category': 'Document AI',
  'description': 'Detect/structure tables into JSON Phase 1 benchmark adapter entry; requires the '
                 'matching app/runtime surface before exposing as an end-user workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'tags': ['local-no-api', 'phase1-shortlist', 'transformers'],
  'backend': 'transformers',
  'hf_repo': 'microsoft/table-transformer-detection',
  'phase1_adapter': True},
 {'id': 'got-ocr2',
  'name': 'GOT-OCR 2.0',
  'vendor': 'StepFun',
  'category': 'Document AI',
  'description': 'High-fidelity end-to-end OCR that returns LaTeX for tables and structured layouts. '
                 'Replaces the older Table Transformer + region-crop flow in the Toolbox Extract '
                 'Table (best) workflow. Loads in float32 unless dtype is set explicitly.',
  'parameters': '580M',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'tags': ['local-no-api', 'transformers', 'document-ai', 'toolbox-only'],
  'backend': 'transformers',
  'hf_repo': 'stepfun-ai/GOT-OCR-2.0-hf',
  'hf_revision': 'd3017ef2c2c1395888c8d635c5e0508bcb0ac78d',
  'phase1_adapter': True},
 {'id': 'trocr-base-printed',
  'name': 'TrOCR Base Printed',
  'vendor': 'TrOCR',
  'category': 'Document AI',
  'description': 'Offline printed OCR Phase 1 benchmark adapter entry; requires the matching '
                 'app/runtime surface before exposing as an end-user workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'tags': ['local-no-api', 'phase1-shortlist', 'transformers'],
  'backend': 'transformers',
  'hf_repo': 'microsoft/trocr-base-printed',
  'phase1_adapter': True},
 {'id': 'trocr-large-printed',
  'name': 'TrOCR Large Printed',
  'vendor': 'TrOCR',
  'category': 'Document AI',
  'description': 'High-accuracy printed OCR Phase 1 benchmark adapter entry; requires the matching '
                 'app/runtime surface before exposing as an end-user workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': None,
  'tags': ['local-no-api', 'phase1-shortlist', 'transformers'],
  'backend': 'transformers',
  'hf_repo': 'microsoft/trocr-large-printed',
  'phase1_adapter': True},
 {'id': 'phi-4-multimodal',
  'name': 'Phi-4 Multimodal Instruct',
  'vendor': 'Phi',
  'category': 'Vision',
  'description': 'Unified multimodal model. Phase 1 benchmark adapter entry; requires the '
                 'matching app/runtime surface before exposing as an end-user workflow.',
  'parameters': '',
  'size_gb': 0,
  'min_ram_gb': 16,
  'min_vram_gb': 0,
  'context_length': 0,
  'ollama_tag': '',
  'onnx_repo': 'microsoft/Phi-4-multimodal-instruct',
  'tags': ['local-no-api', 'phase1-shortlist', 'transformers-onnx'],
  'backend': 'transformers/onnx',
  'hf_repo': 'microsoft/Phi-4-multimodal-instruct',
  'hf_revision': '93f923e1a7727d1c4f446756212d9d3e8fcc5d81',
  'phase1_adapter': True},
 {'id': 'aya-expanse:8b',
  'name': 'Aya Expanse 8B',
  'vendor': 'Cohere For AI',
  'category': 'Medium',
  'description': 'Multilingual-focused chat model that adds translation and globalization value '
                 'for LocalAI Studio users.',
  'parameters': '8B',
  'size_gb': 4.9,
  'min_ram_gb': 10,
  'min_vram_gb': 6,
  'context_length': 8192,
  'ollama_tag': 'aya-expanse:8b',
  'onnx_repo': None,
  'tags': ['medium', 'chat', 'multilingual', 'translation', 'top-20', 'p1'],
  'gpu_super_merge': True,
  'gpu_super_source_id': 'aya-expanse:8b',
  'gpu_super_decision': 'accept-add'},
 {'id': 'dolphin3:latest',
  'name': 'Dolphin 3',
  'vendor': 'Dolphin / Eric Hartford',
  'category': 'Medium',
  'description': 'Community-tuned general chat model with agentic and coding flavor. Include as a '
                 'clearly labeled sandbox / wow-factor option rather than the enterprise default.',
  'parameters': '8B',
  'size_gb': 4.9,
  'min_ram_gb': 10,
  'min_vram_gb': 6,
  'context_length': 32768,
  'ollama_tag': 'dolphin3:latest',
  'learn_more_url': 'https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B-GGUF',
  'onnx_repo': None,
  'tags': ['medium', 'chat', 'agentic', 'coding', 'community', 'sandbox', 'top-20', 'p1'],
  'gpu_super_merge': True,
  'gpu_super_source_id': 'dolphin3:latest',
  'gpu_super_decision': 'accept-add'},
 {'id': 'falcon3:7b',
  'name': 'Falcon 3 7B',
  'vendor': 'TII',
  'category': 'Medium',
  'description': 'Newer Falcon family chat model that adds model-family diversity with strong '
                 'general responses at practical 7B speed.',
  'parameters': '7B',
  'size_gb': 4.7,
  'min_ram_gb': 10,
  'min_vram_gb': 6,
  'context_length': 32768,
  'ollama_tag': 'falcon3:7b',
  'onnx_repo': None,
  'tags': ['medium', 'chat', 'general', 'top-20', 'p0'],
  'gpu_super_merge': True,
  'gpu_super_source_id': 'falcon3:7b',
  'gpu_super_decision': 'accept-add'},
 {'id': 'granite3.3:8b',
  'name': 'Granite 3.3 8B',
  'vendor': 'IBM',
  'category': 'Medium',
  'description': 'IBM Granite enterprise-family chat model that adds governance-friendly '
                 'positioning and strong practical instruction following.',
  'parameters': '8B',
  'size_gb': 4.9,
  'min_ram_gb': 10,
  'min_vram_gb': 6,
  'context_length': 32768,
  'ollama_tag': 'granite3.3:8b',
  'onnx_repo': None,
  'tags': ['medium', 'chat', 'enterprise', 'general', 'top-20', 'p2'],
  'gpu_super_merge': True,
  'gpu_super_source_id': 'granite3.3:8b',
  'gpu_super_decision': 'accept-add'},
 {'id': 'granite3.1-moe:latest',
  'name': 'Granite 3.1 MoE',
  'vendor': 'IBM',
  'category': 'Small',
  'description': 'Efficient IBM Granite mixture-of-experts model with enterprise-friendly '
                 'positioning and very good speed in local testing.',
  'parameters': '3B MoE',
  'size_gb': 2.0,
  'min_ram_gb': 6,
  'min_vram_gb': 3,
  'context_length': 32768,
  'ollama_tag': 'granite3.1-moe:latest',
  'onnx_repo': None,
  'tags': ['small', 'chat', 'moe', 'enterprise', 'fast', 'top-20', 'p1'],
  'gpu_super_merge': True,
  'gpu_super_source_id': 'granite3.1-moe:latest',
  'gpu_super_decision': 'accept-add'},
 {'id': 'nemotron-3-nano:4b',
  'name': 'NVIDIA Nemotron 3 Nano 4B',
  'vendor': 'NVIDIA',
  'category': 'Small',
  'description': 'Official NVIDIA Nemotron 3 Nano 4B Ollama model. A compact 2026 '
                 'reasoning-capable chat model with strong instruction following, coding, '
                 'and planning behavior at practical local speed.',
  'parameters': '4B',
  'size_gb': 2.8,
  'min_ram_gb': 8,
  'min_vram_gb': 4,
  'context_length': 262144,
  'ollama_tag': 'nemotron-3-nano:4b',
  'ollama_num_ctx': 8192,
  'ollama_keep_alive': '30m',
  'onnx_repo': None,
  'benchmark_num_predict': 768,
  'tags': ['small',
           'chat',
           'reasoning',
           'coding',
           'agentic',
           'official-ollama',
           'nvidia',
           '2026',
           'top-20',
           'p0']},
 {'id': 'nemotron-3-nano:4b-q8_0',
  'name': 'NVIDIA Nemotron 3 Nano 4B Q8',
  'vendor': 'NVIDIA',
  'category': 'Small',
  'description': 'Higher-precision Q8 variant of NVIDIA Nemotron 3 Nano 4B for users '
                 'who want a little more answer quality while keeping the model far '
                 'smaller than 30B-class Nemotron checkpoints.',
  'parameters': '4B',
  'size_gb': 4.2,
  'min_ram_gb': 8,
  'min_vram_gb': 8,
  'context_length': 262144,
  'ollama_tag': 'nemotron-3-nano:4b-q8_0',
  'ollama_num_ctx': 8192,
  'ollama_keep_alive': '30m',
  'onnx_repo': None,
  'benchmark_num_predict': 768,
  'tags': ['small',
           'chat',
           'reasoning',
           'coding',
           'agentic',
           'quality',
           'official-ollama',
           'nvidia',
           '2026',
           'top-20',
           'p1']},
 {'id': 'gemma3:1b',
  'name': 'Gemma 3 1B',
  'vendor': 'Google',
  'category': 'Ultra Small',
  'description': 'Tiny Gemma 3 chat model with the best compact-suite result in the May 2026 '
                 'LocalAI benchmark. Excellent CPU/AI PC fallback for fast everyday chat.',
  'parameters': '1B',
  'size_gb': 0.8,
  'min_ram_gb': 4,
  'min_vram_gb': 1,
  'context_length': 32768,
  'ollama_tag': 'gemma3:1b',
  'onnx_repo': None,
  'tags': ['tiny', 'fast', 'chat', 'quality', 'top-20', 'p0'],
  'gpu_super_merge': True,
  'gpu_super_source_id': 'gemma3:1b',
  'gpu_super_decision': 'accept-add'}]

# ── Optional SKU helpers ──────────────────────────────────────────────────────

# Populated at runtime only when the optional private SKU file exists.
OPTIONAL_SKU_ORDER: list[str] = []
OPTIONAL_SKU_VRAM: dict = {}
OPTIONAL_SKU_RAM: dict = {}

# Minimum system RAM (GB) required for CPU-only image generation.
# SD 1.5-family CPU-viable models can run on a small CPU SKU (4 cores / 16 GB);
# larger SDXL/Flux-class models remain GPU-only unless explicitly flagged.
IMAGE_GEN_MIN_CPU_RAM_GB = 12


def is_cpu_viable_image_model(model: dict) -> bool:
    """True when an image gen model is fast enough to be worth running on CPU.

    Only models explicitly flagged ``cpu_viable: True`` qualify. SDXL/Flux/etc
    are technically runnable on CPU but take 30-60+ min/image, so we hide them
    from CPU SKUs and from the CPU-mode toggle to keep the UX honest.
    """
    return model.get("backend") == "comfyui" and bool(model.get("cpu_viable", False))


def is_igpu_viable_image_model(model: dict) -> bool:
    """True when an image gen model can complete inside DXGI TDR on a
    Windows-integrated GPU (Intel Arc Graphics, AMD Radeon Graphics,
    Snapdragon Adreno).

    Windows enforces a per-kernel timeout (TDR — Timeout Detection and Recovery,
    default 2 seconds) that DOES NOT depend on available memory. Heavy SDXL /
    Flux UNets routinely take 6-15 seconds per step on iGPUs even when there's
    plenty of shared GPU memory, so the GPU is killed mid-generation with
    ``The GPU device instance has been suspended. Use GetDeviceRemovedReason``
    even though the model "fits". This is independent from ``cpu_viable``
    (Apple Silicon doesn't have TDR; CPU mode has no per-kernel timeout) and
    from ``min_vram_gb`` (an iGPU with 13 GB shared memory still TDRs on a
    heavy kernel).

    Currently true for:
      - All ``cpu_viable: True`` models (SD 1.5 family — Realistic Vision,
        Counterfeit, etc. — small UNets that fit a single iGPU step inside TDR).
      - Models explicitly flagged ``igpu_viable: True`` in the catalog.
        Today: SDXL Lightning (step-reduced SDXL: 4-8 steps × ~1.5s per step
        at 512×512 fits inside TDR; native 1024×1024 30-step SDXL would not).

    v5.5.12 (Ron, 2026-05-27): added to gate the Image Gen dropdown on
    Windows-iGPU SKUs after Ron's Intel Core Ultra 5 325 (Lunar Lake / Arc
    Graphics) TDR'd on Realistic Vision V6 at 768×512 / 30 steps. See
    ``ModelDropdownSpeedSortContractTests`` and ``IntegratedGpuImageDropdownGatingContractTests``.
    """
    if model.get("backend") != "comfyui":
        return False
    return bool(model.get("cpu_viable", False) or model.get("igpu_viable", False))


def get_optional_min_sku(model: dict) -> str | None:
    """
    Return the minimum SKU that can run *model*, or None if no SKU fits.

    For GPU SKUs: checks min_vram_gb.
    For CPU-only SKUs (vram_gb=0): text models check min_ram_gb. Image gen
    models are allowed only on SKUs with >= IMAGE_GEN_MIN_CPU_RAM_GB and only
    when the model is flagged cpu_viable (currently SD 1.5 only).
    """
    min_vram = model.get("min_vram_gb", 0)
    min_ram = model.get("min_ram_gb", 0)
    is_image_gen = model.get("backend") == "comfyui"
    cpu_viable = is_cpu_viable_image_model(model)
    for sku in OPTIONAL_SKU_ORDER:
        sku_vram = OPTIONAL_SKU_VRAM.get(sku, 0)
        sku_ram = OPTIONAL_SKU_RAM.get(sku, 0)
        if sku_vram == 0:
            if is_image_gen:
                if cpu_viable and sku_ram >= IMAGE_GEN_MIN_CPU_RAM_GB:
                    return sku
                continue
            if sku_ram >= min_ram:
                return sku
        else:
            if sku_vram >= min_vram:
                return sku
    return None


def get_models_for_optional_sku(sku_name: str) -> list[dict]:
    """Return all models whose VRAM requirement fits within *sku_name*."""
    vram = OPTIONAL_SKU_VRAM.get(sku_name, 0)
    return [m for m in MODELS if m.get("min_vram_gb", 0) <= vram]


def sku_model_count(vram_gb: float, ram_gb: float,
                    models: list[dict] | None = None) -> int:
    """Count how many catalog models a SKU with given VRAM/RAM can run.

    Used to determine where to insert 'This Device' in the capability-ordered
    SKU list.  GPU SKUs match by VRAM; CPU-only SKUs match LLMs by RAM and
    also match image gen models when ram_gb >= IMAGE_GEN_MIN_CPU_RAM_GB and
    the model fits in 50% of system RAM.
    """
    count = 0
    cpu_image_gen_ok = vram_gb == 0 and ram_gb >= IMAGE_GEN_MIN_CPU_RAM_GB
    for m in (models or MODELS):
        min_vram = m.get("min_vram_gb", 0)
        min_ram = m.get("min_ram_gb", 0)
        is_image_gen = m.get("backend") == "comfyui"
        if vram_gb > 0 and vram_gb >= min_vram:
            count += 1
        elif is_image_gen:
            if cpu_image_gen_ok and is_cpu_viable_image_model(m):
                count += 1
        elif ram_gb >= min_ram:
            count += 1
    return count


# ── Look-up helpers ───────────────────────────────────────────────────────────

def get_model_by_id(model_id: str, models: list[dict] | None = None) -> dict | None:
    for m in (models or MODELS):
        if m["id"] == model_id:
            return m
    return None

def get_models_by_category(models: list[dict] | None = None) -> dict:
    result: dict = {}
    for m in (models or MODELS):
        result.setdefault(m["category"], []).append(m)
    return result


# ── Catalog file I/O ──────────────────────────────────────────────────────────

def _validate_model(entry: dict) -> bool:
    """Return True if *entry* has all required fields with sane types."""
    missing = _REQUIRED_FIELDS - entry.keys()
    if missing:
        model_id = entry.get("id", "?")
        fields = ", ".join(sorted(missing))
        _log.warning(f"Catalog: skipping {model_id!r} — missing fields: {fields}")
        return False
    # Text models need ollama_tag; image models need comfyui_model. Other local
    # adapters need an HF/ONNX repo so they can still appear in the catalog
    # without pretending to be chat models.
    is_image = entry.get("backend") == "comfyui"
    if is_image and not entry.get("comfyui_model"):
        model_id = entry.get("id", "?")
        _log.warning(f"Catalog: skipping {model_id!r} — image model missing comfyui_model.")
        return False
    if not is_image and not entry.get("ollama_tag") and not (
        entry.get("hf_repo") or entry.get("onnx_repo") or entry.get("ov_repo")
    ):
        model_id = entry.get("id", "?")
        _log.warning(
            f"Catalog: skipping {model_id!r} — local adapter model missing ollama_tag, hf_repo, onnx_repo, or ov_repo."
        )
        return False
    if is_image and entry.get("supports_img2img"):
        wf = entry.get("img2img_workflows")
        if not isinstance(wf, dict):
            model_id = entry.get("id", "?")
            _log.warning(f"Catalog: {model_id!r} supports img2img but has no img2img_workflows block.")
        else:
            denoise_min = float(wf.get("denoise_min", 0.1))
            denoise_default = float(wf.get("denoise_default", 0.75))
            denoise_max = float(wf.get("denoise_max", 1.0))
            if not (0.05 <= denoise_min <= denoise_default <= denoise_max <= 1.0):
                model_id = entry.get("id", "?")
                _log.warning(
                    f"Catalog: {model_id!r} img2img denoise range should satisfy "
                    "0.05 <= min <= default <= max <= 1.0."
                )
    return True


def _with_runtime_defaults(entry: dict) -> dict:
    """Return a shallow copy with optional runtime catalog fields backfilled."""
    model = dict(entry)
    is_image = model.get("backend") == "comfyui" or bool(model.get("comfyui_model"))
    if is_image:
        model.setdefault("supports_img2img", False)
        model.setdefault(
            "img2img_workflows",
            {
                "denoise_default": 0.75,
                "denoise_min": 0.1,
                "denoise_max": 1.0,
                "supports_denoise_override": True,
            },
        )
    return model


def load_catalog(path: Path | None = None) -> list[dict]:
    """
    Load the model list from *path* (default: CATALOG_FILE).

    Falls back to the built-in MODELS list if the file is absent, unreadable,
    or structurally invalid.  Individual entries that fail validation are
    skipped; the rest are still used.
    """
    target = path or CATALOG_FILE
    if not target.exists():
        return [_with_runtime_defaults(m) for m in MODELS]
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        _log.error(f"Catalog: could not read {target} ({exc}) — using built-in defaults.")
        return [_with_runtime_defaults(m) for m in MODELS]

    if not isinstance(data, dict) or "models" not in data:
        _log.error(f"Catalog: {target} missing 'models' key — using built-in defaults.")
        return [_with_runtime_defaults(m) for m in MODELS]

    raw: list = data["models"]
    if not isinstance(raw, list):
        _log.error(f"Catalog: 'models' is not a list in {target} — using built-in defaults.")
        return [_with_runtime_defaults(m) for m in MODELS]

    valid = [_with_runtime_defaults(m) for m in raw if isinstance(m, dict) and _validate_model(m)]
    if not valid:
        _log.warning(f"Catalog: no valid models found in {target} — using built-in defaults.")
        return [_with_runtime_defaults(m) for m in MODELS]

    merge_builtins = bool(data.get("merge_builtins", True))
    disabled_builtin_ids = {
        str(mid) for mid in data.get("disabled_builtin_ids", [])
        if isinstance(mid, str)
    }
    if disabled_builtin_ids:
        valid = [m for m in valid if m.get("id") not in disabled_builtin_ids]

    if merge_builtins:
        # Merge built-in entries missing from JSON unless explicitly disabled.
        existing_ids = {m["id"] for m in valid}
        added = 0
        for builtin in MODELS:
            if builtin["id"] not in existing_ids and builtin["id"] not in disabled_builtin_ids:
                valid.append(_with_runtime_defaults(builtin))
                added += 1
        if added:
            _log.info(
                f"Catalog: merged {added} new built-in model(s) into {target}. "
                "Set merge_builtins=false or disabled_builtin_ids to opt out."
            )
    else:
        _log.info(f"Catalog: merge_builtins=false in {target}; using JSON models only.")

    _log.info(f"Catalog: loaded {len(valid)} models from {target}.")
    return valid


def is_chat_selectable_model(model: dict) -> bool:
    """Return whether *model* belongs in the Chat page model selector."""
    if not isinstance(model, dict):
        return False
    if model.get("chat_selectable") is False:
        return False
    if model.get("backend") == "comfyui":
        return False
    if str(model.get("category") or "") in _NON_CHAT_CATEGORIES:
        return False
    # Phase 1 adapter entries are benchmark/runtime placeholders unless a
    # catalog author explicitly opts them into the Chat surface.
    if model.get("phase1_adapter") and not model.get("chat_selectable"):
        return False

    backend = str(model.get("backend") or "ollama").strip().lower()
    if backend not in _CHAT_BACKENDS and not model.get("chat_selectable"):
        return False

    return bool(model.get("ollama_tag") or model.get("onnx_repo") or model.get("ov_repo"))


def save_catalog(models: list[dict], path: Path | None = None) -> bool:
    """Write *models* to *path* as JSON (default: CATALOG_FILE)."""
    target = path or CATALOG_FILE
    payload = {
        "_readme": _CATALOG_README,
        "version": 1,
        "merge_builtins": True,
        "disabled_builtin_ids": [],
        "models": models,
    }
    try:
        atomic_write_json(target, payload, indent=2, ensure_ascii=False)
        _log.info(f"Catalog: saved {len(models)} models to {target}.")
        return True
    except OSError as exc:
        _log.error(f"Catalog: could not write {target}: {exc}")
        return False


def ensure_catalog_file(path: Path | None = None) -> bool:
    """
    Create the catalog file from built-in MODELS if it does not already exist.
    Returns True if the file was newly created, False if it already existed.
    """
    target = path or CATALOG_FILE
    if target.exists():
        return False
    save_catalog(list(MODELS), target)
    _log.info(f"Catalog: created default catalog at {target}.")
    return True


_HF_SHA_RE = __import__("re").compile(r"^[0-9a-f]{40}$")


def _normalize_ollama_tag_for_match(tag: object) -> str:
    """Normalize Ollama tag names so bare tags match ``:latest`` variants."""
    raw = str(tag or "").strip().lower()
    if not raw:
        return ""
    base, sep, suffix = raw.partition(":")
    base = base.strip()
    suffix = suffix.strip()
    if not base:
        return ""
    if not sep or not suffix:
        suffix = "latest"
    return f"{base}:{suffix}"


def _num_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe_text_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _is_sparse_ollama_user_entry(entry: dict) -> bool:
    """Heuristic for low-fidelity Add-from-HF Ollama placeholder entries."""
    if not isinstance(entry, dict):
        return False
    backend = str(entry.get("backend") or "ollama").strip().lower()
    if backend != "ollama":
        return False
    return (
        _num_or_zero(entry.get("size_gb")) <= 0
        and _num_or_zero(entry.get("context_length")) <= 0
        and not str(entry.get("parameters") or "").strip()
        and len(_dedupe_text_list(entry.get("recommended_for"))) == 0
    )


def _merge_duplicate_ollama_entry(
    existing: dict,
    proposed: dict,
    *,
    source_url: str,
    requires_review: bool,
) -> dict:
    """Merge duplicate user-added Ollama rows for the same tag into one entry."""
    existing_id = str(existing.get("id") or "").strip()
    existing_sparse = _is_sparse_ollama_user_entry(existing)
    proposed_sparse = _is_sparse_ollama_user_entry(proposed)

    if existing_sparse and not proposed_sparse:
        merged = dict(proposed)
        merged["id"] = existing_id
        if existing.get("added_at"):
            merged["added_at"] = existing["added_at"]
    else:
        merged = dict(existing)
        numeric_sparse_keys = {"size_gb", "min_ram_gb", "min_vram_gb", "context_length"}
        for key, value in proposed.items():
            if key == "id":
                continue
            if key in {"tags", "recommended_for"}:
                merged[key] = _dedupe_text_list(
                    _dedupe_text_list(merged.get(key)) + _dedupe_text_list(value)
                )
                continue
            if key in numeric_sparse_keys:
                if _num_or_zero(merged.get(key)) <= 0 < _num_or_zero(value):
                    merged[key] = value
                continue
            current = merged.get(key)
            if current is None:
                merged[key] = value
            elif isinstance(current, str) and not current.strip() and str(value or "").strip():
                merged[key] = value
            elif isinstance(current, (list, dict)) and not current:
                merged[key] = value

    if source_url:
        merged["source_url"] = source_url
    merged["user_added"] = True
    if requires_review or bool(existing.get("requires_review")) or bool(proposed.get("requires_review")):
        merged["requires_review"] = True
    merged.setdefault("added_at", _utcnow_iso())
    return merged


def append_user_model(
    entry: dict,
    *,
    source_url: str,
    requires_review: bool = False,
    catalog_path: Path | None = None,
    existing_models: list[dict] | None = None,
) -> tuple[bool, str, list[dict]]:
    """Stamp *entry* as user-added, append it to the catalog, and save.

    Single source of truth for the Add-from-Hugging-Face commit path so the
    UI dialog (Models > + Add from Hugging Face) and any future CLI / test
    harness all agree on the schema rules:

    - ``user_added`` is set to True (cannot be overridden via *entry*).
    - ``source_url`` is recorded verbatim so the user can find where they
      got the model.
    - For any HF-backed backend (anything except plain Ollama), an
      ``hf_revision`` field MUST already be present and MUST be a 40-char
      hexadecimal commit SHA — never ``main``/a branch.  This extends the
      regression-critical rule (today scoped to ``trust_remote_code``
      utility loaders) to every user-added catalog entry, so the user's
      catalog can't silently change behaviour when the upstream HF repo's
      ``main`` moves.
    - ``added_at`` is stamped (UTC, ISO 8601, second precision) if absent.
    - For user-added Ollama entries, if another user-added row already uses
      the same ``ollama_tag`` (treating bare tags and ``:latest`` as the
      same), the helper updates/merges that existing row instead of creating
      ``-2``/``-3`` duplicates.
    - If the proposed ``id`` collides with an existing entry, the function
      appends ``-2``, ``-3``, … to the slug until it is unique within the
      passed ``existing_models`` list.

    Returns ``(ok, final_id, updated_models)``.  On any validation failure
    the catalog is *not* modified and ``ok`` is False; the second tuple
    element carries the human-readable reason instead of the assigned ID.

    Pass ``existing_models`` (already loaded from ``load_catalog``) to
    avoid double-reading the catalog file; otherwise the helper calls
    ``load_catalog`` itself.  ``catalog_path`` lets tests redirect the
    write target.
    """
    if not isinstance(entry, dict):
        return False, "Proposed entry is not a dict.", existing_models or []

    proposed = dict(entry)
    proposed["user_added"] = True
    proposed["source_url"] = source_url
    if requires_review:
        proposed["requires_review"] = True
    proposed.setdefault("added_at", _utcnow_iso())

    backend = str(proposed.get("backend") or "").strip().lower()
    if backend != "ollama":
        sha = str(proposed.get("hf_revision") or "").strip().lower()
        if not _HF_SHA_RE.match(sha):
            return (
                False,
                "User-added entries must pin hf_revision to a resolved 40-char commit SHA, never a branch name.",
                existing_models or [],
            )
        proposed["hf_revision"] = sha

    if not _validate_model(proposed):
        return (
            False,
            f"Proposed entry failed catalog validation: {proposed.get('id', '?')!r}.",
            existing_models or [],
        )

    models = list(existing_models) if existing_models is not None else load_catalog(catalog_path)

    base_id = str(proposed.get("id") or "").strip()
    if not base_id:
        return False, "Proposed entry has no id.", models

    # Avoid duplicate user-added Ollama rows for the same pull tag.
    if backend == "ollama":
        proposed_tag = _normalize_ollama_tag_for_match(proposed.get("ollama_tag"))
        if proposed_tag:
            for idx, existing in enumerate(models):
                if not isinstance(existing, dict):
                    continue
                if not existing.get("user_added"):
                    continue
                existing_backend = str(existing.get("backend") or "ollama").strip().lower()
                if existing_backend != "ollama":
                    continue
                existing_tag = _normalize_ollama_tag_for_match(existing.get("ollama_tag"))
                if existing_tag != proposed_tag:
                    continue
                existing_id = str(existing.get("id") or "").strip()
                if not existing_id:
                    continue
                merged = _merge_duplicate_ollama_entry(
                    existing,
                    proposed,
                    source_url=source_url,
                    requires_review=requires_review,
                )
                if not _validate_model(merged):
                    return (
                        False,
                        f"Merged Ollama entry failed validation: {existing_id!r}.",
                        models,
                    )
                models[idx] = merged
                if not save_catalog(models, catalog_path):
                    return False, "Catalog write failed; duplicate entry was not merged.", models
                return True, existing_id, models

    existing_ids = {str(m.get("id") or "") for m in models}
    final_id = base_id
    suffix = 2
    while final_id in existing_ids:
        final_id = f"{base_id}-{suffix}"
        suffix += 1
        if suffix > 999:
            return False, f"Could not find a unique id for {base_id!r}.", models
    proposed["id"] = final_id

    models.append(proposed)
    if not save_catalog(models, catalog_path):
        return False, "Catalog write failed; entry was not added.", models

    return True, final_id, models


def _utcnow_iso() -> str:
    """ISO-8601 UTC, second precision — for catalog ``added_at`` stamps."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



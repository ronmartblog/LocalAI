# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Tests for src.hf_compat — the family-detection cascade behind the
Add-from-Hugging-Face dialog.

The compat layer talks to Hugging Face via the thin :class:`_HFClient`
adapter; these tests substitute a hand-rolled fake so we can drive every
branch of the cascade (GGUF / ONNX / OpenVINO / Diffusers / single-file /
Phase 1 / text-gen / gated / not-found / network / Ollama) without hitting
the network.

Key invariants asserted here (mirror the regression-critical contract):

- Every image-gen result MUST populate ``recommended_settings`` AND
  ``perf_profile`` blocks (v5.1/v5.3 validators).
- Every HF-backed result MUST pin ``hf_revision`` to a 40-char SHA — never
  ``main`` / a branch.
- Cascade ordering: GGUF wins over Diffusers; Chroma wins over Flux when
  the filename contains both substrings.
- Gated → ``verdict == "needs_access"`` with ``is_install_blocked == True``.
"""

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import hf_compat
from src.hf_compat import CompatResult, _HFClient, inspect
from src.hf_model_resolver import ParsedTarget


SHA = "a" * 40  # canonical fake 40-char hex SHA used across the suite


class FakeHFClient:
    """Stub _HFClient that returns canned dicts.  Tests inject one per case
    so we can drive every branch of the cascade deterministically.

    Pass ``raises`` to make the call raise (the cascade catches and
    translates into the verdict).
    """

    def __init__(self, info=None, model_index_text=None, raises=None):
        self._info = info
        self._model_index_text = model_index_text
        self._raises = raises
        self.calls = 0

    def model_info(self, repo_id, revision=None):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return dict(self._info)

    def fetch_text_file(self, repo_id, file_path, revision):
        if file_path == "model_index.json":
            return self._model_index_text
        return None


def _siblings(*pairs):
    """Build a fake siblings list from ``(filename, size_bytes)`` tuples."""
    return [{"rfilename": fn, "size": sz, "lfs": True} for fn, sz in pairs]


def _hf(repo_id="someorg/somemodel"):
    return ParsedTarget(route="hf", repo_id=repo_id)


def _assert_image_schema(test, entry):
    """Every image-gen entry must populate both blocks the validators check."""
    rec = entry.get("recommended_settings")
    test.assertIsInstance(rec, dict, "missing recommended_settings block")
    for required in ("width", "height", "sampler", "scheduler", "steps", "cfg", "family_label"):
        test.assertIn(required, rec, f"recommended_settings missing {required!r}")
    perf = entry.get("perf_profile")
    test.assertIsInstance(perf, dict, "missing perf_profile block")
    for required in ("speed_tier", "quality_tier", "category_bucket", "recommendation", "speed_label"):
        test.assertIn(required, perf, f"perf_profile missing {required!r}")


def _assert_sha_pinned(test, entry):
    sha = str(entry.get("hf_revision") or "")
    test.assertTrue(
        re.match(r"^[0-9a-f]{40}$", sha),
        f"hf_revision must be 40-char hex SHA, got {sha!r}",
    )


# ── GGUF branch ──────────────────────────────────────────────────────────────


class GgufCascade(unittest.TestCase):

    def test_flux_gguf_supported(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("flux1-schnell-Q4_K_S.gguf", 4_000_000_000),
                ("flux1-schnell-Q8_0.gguf", 8_000_000_000),
            ),
            "pipeline_tag": None,
            "author": "city96",
        })
        result = inspect(_hf("city96/FLUX.1-schnell-gguf"), client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.backend, "comfyui")
        self.assertEqual(result.family, "flux")
        _assert_image_schema(self, result.proposed_entry)
        _assert_sha_pinned(self, result.proposed_entry)

    def test_chroma_wins_over_flux_in_filename(self):
        """Ordering rule: Chroma checkpoint filenames often contain "flux";
        the cascade must check Chroma BEFORE Flux."""
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(("chroma-flux-base-q4.gguf", 6_000_000_000)),
            "pipeline_tag": None, "author": "lodestones",
        })
        result = inspect(_hf("lodestones/Chroma"), client=client)
        self.assertEqual(result.family, "chroma")

    def test_z_image_gguf_detected(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(("z_image_turbo-q4.gguf", 5_000_000_000)),
            "pipeline_tag": None, "author": "tongyi",
        })
        result = inspect(_hf("tongyi/Z-Image-Turbo"), client=client)
        self.assertEqual(result.family, "z-image")

    def test_z_image_wins_over_chroma_in_hybrid_filename(self):
        """Ordering rule: hybrid checkpoints whose filename contains BOTH
        "chroma" and "z_image" must route to Z-Image. Mirrors the
        comfyui_client.py cascade order (Z-Image → Chroma → Flux)."""
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("chroma_z_image_hybrid-q4.gguf", 5_500_000_000),
            ),
            "pipeline_tag": None, "author": "experimental",
        })
        result = inspect(_hf("experimental/chroma-z-image"), client=client)
        self.assertEqual(result.family, "z-image")

    def test_generic_llm_gguf_routes_to_ollama(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(("llama-3.2-1b-q4.gguf", 800_000_000)),
            "pipeline_tag": "text-generation", "author": "bartowski",
        })
        result = inspect(_hf("bartowski/Llama-3.2-1B-Instruct-GGUF"), client=client)
        self.assertEqual(result.verdict, "unsupported")
        # And the reason should mention Ollama so the user knows what to do.
        joined = " ".join(result.reasons).lower()
        self.assertIn("ollama", joined)

    def test_blob_targeted_file_is_preferred(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("flux1-schnell-Q4_K_S.gguf", 4_000_000_000),
                ("flux1-schnell-Q8_0.gguf", 8_000_000_000),
            ),
            "pipeline_tag": None, "author": "city96",
        })
        target = ParsedTarget(
            route="hf", repo_id="city96/FLUX.1-schnell-gguf",
            file_path="flux1-schnell-Q8_0.gguf",
        )
        result = inspect(target, client=client)
        self.assertEqual(
            result.proposed_entry.get("comfyui_model"),
            "flux1-schnell-Q8_0.gguf",
        )


# ── ONNX branches ────────────────────────────────────────────────────────────


class OnnxCascade(unittest.TestCase):

    def test_onnx_genai_detected_by_genai_config(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("genai_config.json", 1_000),
                ("model.onnx", 3_000_000_000),
                ("config.json", 1_000),
            ),
            "pipeline_tag": "text-generation",
            "author": "microsoft",
        })
        result = inspect(_hf("microsoft/Phi-4-mini-onnx"), client=client)
        self.assertEqual(result.backend, "onnx-genai")
        self.assertIn(result.verdict, {"supported", "warn"})
        _assert_sha_pinned(self, result.proposed_entry)
        # Should not be misclassified into the plain-onnx branch.
        self.assertNotEqual(result.backend, "onnx")

    def test_plain_onnx_when_no_genai_config(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("model.onnx", 500_000_000),
                ("config.json", 1_000),
            ),
            "pipeline_tag": "feature-extraction",
            "author": "intfloat",
        })
        result = inspect(_hf("intfloat/multilingual-e5-small-onnx"), client=client)
        self.assertEqual(result.backend, "onnx")
        self.assertEqual(result.verdict, "supported")
        _assert_sha_pinned(self, result.proposed_entry)


# ── OpenVINO branch ──────────────────────────────────────────────────────────


class OpenvinoCascade(unittest.TestCase):

    def test_openvino_supported_on_windows(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("openvino_model.xml", 100_000),
                ("openvino_model.bin", 600_000_000),
            ),
            "pipeline_tag": "text-generation", "author": "OpenVINO",
        })
        result = inspect(_hf("OpenVINO/llama-3.2-1b-int4-ov"),
                         client=client, platform="win32")
        self.assertEqual(result.backend, "openvino")
        self.assertEqual(result.verdict, "supported")
        # ov_repo set so the catalog validator accepts it.
        self.assertEqual(result.proposed_entry.get("ov_repo"),
                         "OpenVINO/llama-3.2-1b-int4-ov")

    def test_openvino_unsupported_off_windows(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("openvino_model.xml", 100_000),
                ("openvino_model.bin", 600_000_000),
            ),
            "pipeline_tag": "text-generation", "author": "OpenVINO",
        })
        result = inspect(_hf("OpenVINO/llama-3.2-1b-int4-ov"),
                         client=client, platform="linux")
        self.assertEqual(result.verdict, "unsupported")


# ── Diffusers branch ────────────────────────────────────────────────────────


class DiffusersCascade(unittest.TestCase):

    def _make(self, class_name, **extra):
        return FakeHFClient(
            info={
                "sha": SHA,
                "siblings": _siblings(
                    ("model_index.json", 500),
                    ("unet/diffusion_pytorch_model.safetensors", 6_000_000_000),
                ),
                "pipeline_tag": "text-to-image",
                "author": "stabilityai",
                **extra,
            },
            model_index_text=json.dumps({"_class_name": class_name}),
        )

    def test_sd15_diffusers_warn(self):
        client = self._make("StableDiffusionPipeline")
        result = inspect(_hf("runwayml/stable-diffusion-v1-5"), client=client)
        # Diffusers folders warn (we can't run them as single-file checkpoints
        # without conversion) but the schema should still be complete.
        self.assertEqual(result.verdict, "warn")
        self.assertEqual(result.family, "sd15")
        _assert_image_schema(self, result.proposed_entry)

    def test_sdxl_diffusers_warn(self):
        client = self._make("StableDiffusionXLPipeline")
        result = inspect(_hf("stabilityai/stable-diffusion-xl-base-1.0"), client=client)
        self.assertEqual(result.family, "sdxl")
        _assert_image_schema(self, result.proposed_entry)

    def test_flux_diffusers_warn(self):
        client = self._make("FluxPipeline")
        result = inspect(_hf("black-forest-labs/FLUX.1-schnell"), client=client)
        self.assertEqual(result.family, "flux")
        _assert_image_schema(self, result.proposed_entry)

    def test_sd3_unsupported(self):
        client = self._make("StableDiffusion3Pipeline")
        result = inspect(_hf("stabilityai/stable-diffusion-3-medium"), client=client)
        self.assertEqual(result.verdict, "unsupported")
        joined = " ".join(result.reasons).lower()
        # Per P0-4 (designer review): unsupported-family reasons MUST point
        # the user at a supported alternative they can paste next, not just
        # tell them what doesn't work.
        self.assertTrue(
            "flux" in joined or "sdxl" in joined or "z-image" in joined,
            f"unsupported reason should suggest an alternative family: {joined!r}",
        )

    def test_aurraflow_unsupported(self):
        client = self._make("AuraFlowPipeline")
        result = inspect(_hf("fal/AuraFlow"), client=client)
        self.assertEqual(result.verdict, "unsupported")

    def test_missing_model_index_falls_through(self):
        # No model_index.json text returned → can't read the _class_name.
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(("model_index.json", 500)),
            "pipeline_tag": "text-to-image", "author": "x",
        }, model_index_text=None)
        result = inspect(_hf("x/y"), client=client)
        self.assertEqual(result.verdict, "unsupported")


# ── Single-file safetensors branch ──────────────────────────────────────────


class SingleFileSafetensors(unittest.TestCase):

    def test_sdxl_shape_detected(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(("sd_xl_base_1.0.safetensors", 7_000_000_000)),
            "pipeline_tag": None, "author": "stabilityai",
        })
        result = inspect(_hf("stabilityai/stable-diffusion-xl-base-1.0"),
                         client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.family, "sdxl")
        _assert_image_schema(self, result.proposed_entry)

    def test_sd15_shape_detected(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(("v1-5-pruned-emaonly.safetensors", 4_000_000_000)),
            "pipeline_tag": None, "author": "runwayml",
        })
        result = inspect(_hf("runwayml/stable-diffusion-v1-5-single"), client=client)
        self.assertEqual(result.family, "sd15")

    def test_unknown_shape_unsupported(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(("weird-tiny.safetensors", 100_000_000)),
            "pipeline_tag": None, "author": "x",
        })
        result = inspect(_hf("x/y"), client=client)
        self.assertEqual(result.verdict, "unsupported")


# ── Masked single-file (Diffusers folder + root single-file checkpoint) ──────


class MaskedSingleFileCheckpoint(unittest.TestCase):
    """Repos that ship BOTH a Diffusers folder and the single-file checkpoint
    ComfyUI actually loads must resolve to the single file (supported), not the
    Diffusers warn. This is the dominant bare-repo-URL false negative on popular
    image models (DreamShaper, Realistic Vision, SD/SDXL-Turbo, SSD-1B, …)."""

    def _client(self, class_name, *extra_files, **info_extra):
        siblings = [("model_index.json", 500),
                    ("unet/diffusion_pytorch_model.safetensors", 3_438_000_000),
                    ("vae/diffusion_pytorch_model.safetensors", 334_000_000)]
        siblings.extend(extra_files)
        return FakeHFClient(
            info={
                "sha": SHA,
                "siblings": _siblings(*siblings),
                "pipeline_tag": "text-to-image",
                "author": "someorg",
                **info_extra,
            },
            model_index_text=json.dumps({"_class_name": class_name}),
        )

    def test_sd15_single_file_preferred_over_diffusers(self):
        # DreamShaper-8-LCM shape: SD1.5 Diffusers folder + 2 GB merged ckpt.
        client = self._client(
            "StableDiffusionPipeline",
            ("DreamShaper8_LCM.safetensors", 2_133_804_992),
        )
        result = inspect(_hf("Lykon/dreamshaper-8-lcm"), client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.family, "sd15")
        self.assertEqual(
            result.proposed_entry.get("comfyui_model"),
            "DreamShaper8_LCM.safetensors",
        )
        _assert_image_schema(self, result.proposed_entry)
        _assert_sha_pinned(self, result.proposed_entry)

    def test_family_comes_from_model_index_not_size(self):
        # SSD-1B is a distilled SDXL at ~4.5 GB — the size heuristic alone would
        # call it sd15, but the authoritative model_index says SDXL.
        client = self._client(
            "StableDiffusionXLPipeline",
            ("SSD-1B-A1111.safetensors", 4_470_000_000),
            ("SSD-1B.safetensors", 4_470_000_000),
        )
        result = inspect(_hf("segmind/SSD-1B"), client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.family, "sdxl")

    def test_fp16_variant_preferred(self):
        # SDXL-Turbo ships fp32 (13.9 GB, out of band) + fp16 (6.9 GB).
        client = self._client(
            "StableDiffusionXLPipeline",
            ("sd_xl_turbo_1.0.safetensors", 13_900_000_000),
            ("sd_xl_turbo_1.0_fp16.safetensors", 6_940_000_000),
        )
        result = inspect(_hf("stabilityai/sdxl-turbo"), client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(
            result.proposed_entry.get("comfyui_model"),
            "sd_xl_turbo_1.0_fp16.safetensors",
        )

    def test_base_preferred_over_inpainting(self):
        client = self._client(
            "StableDiffusionPipeline",
            ("Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors", 2_140_000_000),
            ("Realistic_Vision_V6.0_NV_B1_fp16.safetensors", 2_130_000_000),
        )
        result = inspect(_hf("SG161222/Realistic_Vision_V6.0_B1_noVAE"), client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(
            result.proposed_entry.get("comfyui_model"),
            "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
        )

    def test_explicit_pasted_file_wins(self):
        client = self._client(
            "StableDiffusionXLPipeline",
            ("sd_xl_turbo_1.0.safetensors", 13_900_000_000),
            ("sd_xl_turbo_1.0_fp16.safetensors", 6_940_000_000),
        )
        target = ParsedTarget(
            route="hf", repo_id="stabilityai/sdxl-turbo",
            file_path="sd_xl_turbo_1.0_fp16.safetensors",
        )
        result = inspect(target, client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(
            result.proposed_entry.get("comfyui_model"),
            "sd_xl_turbo_1.0_fp16.safetensors",
        )

    def test_unet_only_root_file_is_not_treated_as_checkpoint(self):
        # LCM_Dreamshaper_v7 ships a bare UNet at the root whose size equals the
        # unet/ subfolder weight — NOT a loadable single-file checkpoint. Must
        # fall through to the (unsupported LatentConsistency) Diffusers branch.
        client = self._client(
            "LatentConsistencyModelPipeline",
            ("LCM_Dreamshaper_v7_4k.safetensors", 3_438_000_000),
        )
        result = inspect(_hf("SimianLuo/LCM_Dreamshaper_v7"), client=client)
        self.assertEqual(result.verdict, "unsupported")
        self.assertIsNone(result.proposed_entry.get("comfyui_model"))

    def test_component_only_root_files_fall_through_to_warn(self):
        # A genuine Diffusers repo whose only root .safetensors are components
        # (vae, text encoder) must still warn — there is no real single file.
        client = self._client(
            "StableDiffusionPipeline",
            ("vae.safetensors", 334_000_000),
        )
        result = inspect(_hf("some/diffusers-only"), client=client)
        self.assertEqual(result.verdict, "warn")
        self.assertEqual(result.proposed_entry.get("comfyui_model"), "")

    def test_unsupported_family_with_single_file_does_not_promote(self):
        # Even with a credible-sized root file, a known-unsupported pipeline
        # family (SD3) must not be promoted to supported.
        client = self._client(
            "StableDiffusion3Pipeline",
            ("sd3_medium.safetensors", 4_500_000_000),
        )
        result = inspect(_hf("stabilityai/stable-diffusion-3-medium"), client=client)
        self.assertEqual(result.verdict, "unsupported")

    def test_openvino_image_repo_prefers_single_file(self):
        # SDXL Base ships an OpenVINO export alongside the real checkpoint;
        # it must resolve to the single file, not be mistaken for an OpenVINO LLM.
        client = self._client(
            "StableDiffusionXLPipeline",
            ("sd_xl_base_1.0.safetensors", 6_940_000_000),
            ("openvino_model.xml", 50_000),
            ("openvino_model.bin", 6_900_000_000),
        )
        result = inspect(_hf("stabilityai/stable-diffusion-xl-base-1.0"), client=client)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.family, "sdxl")
        self.assertEqual(
            result.proposed_entry.get("comfyui_model"), "sd_xl_base_1.0.safetensors")

    def test_gated_diffusers_unreadable_reports_needs_access(self):
        # Gated repo: model_info succeeds but model_index.json can't be read.
        # Must surface needs_access, not a misleading "couldn't read" unsupported.
        client = FakeHFClient(
            info={
                "sha": SHA,
                "siblings": _siblings(
                    ("model_index.json", 500),
                    ("flux1-dev.safetensors", 23_800_000_000),  # out of band
                ),
                "pipeline_tag": "text-to-image", "author": "black-forest-labs",
                "gated": "manual",
            },
            model_index_text=None,  # gated content read blocked
        )
        result = inspect(_hf("black-forest-labs/FLUX.1-dev"), client=client)
        self.assertEqual(result.verdict, "needs_access")
        self.assertTrue(result.is_install_blocked)


# ── Phase 1 / text-gen fallbacks ────────────────────────────────────────────


class Phase1AndTextgen(unittest.TestCase):

    def test_whisper_phase1_warn(self):
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("pytorch_model.bin", 800_000_000),
                ("config.json", 500),
            ),
            "pipeline_tag": "automatic-speech-recognition",
            "author": "openai",
        })
        result = inspect(_hf("openai/whisper-small"), client=client)
        self.assertEqual(result.backend, "phase1")
        self.assertEqual(result.verdict, "warn")
        # Should land in Speech category.
        self.assertEqual(result.proposed_entry.get("category"), "Speech")

    def test_text_generation_warns_not_unsupported(self):
        """Designer review #6: pytorch text-gen models should WARN (suggest
        Ollama) not be reported as flat-out unsupported."""
        client = FakeHFClient(info={
            "sha": SHA,
            "siblings": _siblings(
                ("pytorch_model-00001-of-00002.bin", 5_000_000_000),
                ("pytorch_model-00002-of-00002.bin", 5_000_000_000),
                ("config.json", 500),
            ),
            "pipeline_tag": "text-generation",
            "author": "Qwen",
        })
        result = inspect(_hf("Qwen/Qwen2.5-7B-Instruct"), client=client)
        self.assertEqual(result.verdict, "warn")
        # And requires_review should be set on the entry so the detail
        # pane shows the banner.
        self.assertTrue(result.proposed_entry.get("requires_review"))


# ── Failure modes ───────────────────────────────────────────────────────────


class FailureModes(unittest.TestCase):

    def test_gated_repo(self):
        client = FakeHFClient(raises=hf_compat.HfGatedError("Access to model is restricted"))
        result = inspect(_hf("meta-llama/Llama-3-70B"), client=client)
        self.assertEqual(result.verdict, "needs_access")
        self.assertTrue(result.needs_hf_token_gated)
        # The install gate MUST refuse this even if the user clicks past.
        self.assertTrue(result.is_install_blocked)

    def test_not_found(self):
        client = FakeHFClient(raises=hf_compat.HfNotFoundError("Repo does not exist"))
        result = inspect(_hf("ghost/missing"), client=client)
        self.assertEqual(result.verdict, "unsupported")

    def test_network_error(self):
        client = FakeHFClient(raises=hf_compat.HfNetworkError("read timed out"))
        result = inspect(_hf("x/y"), client=client)
        self.assertEqual(result.verdict, "unsupported")

    def test_missing_sha_rejected(self):
        client = FakeHFClient(info={
            "sha": None,
            "siblings": _siblings(("flux.gguf", 1_000_000)),
            "pipeline_tag": None, "author": "x",
        })
        result = inspect(_hf("x/y"), client=client)
        self.assertEqual(result.verdict, "unsupported")

    def test_non_40char_sha_rejected(self):
        client = FakeHFClient(info={
            "sha": "main",
            "siblings": _siblings(("flux.gguf", 1_000_000)),
            "pipeline_tag": None, "author": "x",
        })
        result = inspect(_hf("x/y"), client=client)
        self.assertEqual(result.verdict, "unsupported")


# ── Ollama branch (no network call at all) ──────────────────────────────────


class OllamaBranch(unittest.TestCase):

    def test_ollama_target_supported(self):
        target = ParsedTarget(route="ollama", ollama_tag="llama3.2:1b")
        result = inspect(target)  # no client needed
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.backend, "ollama")
        self.assertEqual(result.proposed_entry.get("ollama_tag"), "llama3.2:1b")
        # User-added catalog flag is set.
        self.assertTrue(result.proposed_entry.get("user_added"))

    def test_ollama_no_hf_client_call(self):
        """The Ollama branch must NEVER touch the HF client (perf rule)."""
        seen = {"called": False}

        class TripwireClient:
            def model_info(self, *a, **k):
                seen["called"] = True
                raise RuntimeError("should not be called for ollama route")

            def fetch_text_file(self, *a, **k):
                seen["called"] = True
                raise RuntimeError("should not be called for ollama route")

        # Use a sized tag so the new bare-tag resolver doesn't hit the
        # network — that path is covered by OllamaBareTagResolver below.
        target = ParsedTarget(route="ollama", ollama_tag="phi3:latest")
        inspect(target, client=TripwireClient())
        self.assertFalse(seen["called"])


class OllamaBareTagResolver(unittest.TestCase):
    """Ron 2026-05-19 regression: pasting https://ollama.com/library/nemotron3
    (no `:size`) used to land a bogus `nemotron3` tag in the catalog that
    Ollama then refused to pull with "manifest does not exist".  The fix
    auto-resolves a bare base name against /library/<base>/tags so the
    canonical default size suffix is pinned at inspect time.

    Tag-resolver tests inject a fake fetcher so we don't hit the network."""

    _NEMOTRON_PAGE = '''
        <a href="/library/nemotron3:33b">card</a>
        <span>28GB</span><span>128K context</span>
        <a href="/library/nemotron3:33b-q8">card</a>
        <span>36GB</span><span>128K context</span>
        <a href="/library/nemotron3:33b-q4_K_M">card</a>
        <span>28GB</span><span>128K context</span>
        <a href="/library/nemotron3:33b-bf16">card</a>
        <span>66GB</span><span>128K context</span>
    '''

    def setUp(self):
        # Monkeypatch the module-level tag resolver so the inspector under
        # test calls our fake instead of urllib.urlopen.
        self._real_resolver = hf_compat._resolve_ollama_tags

    def tearDown(self):
        hf_compat._resolve_ollama_tags = self._real_resolver

    def test_resolver_parses_tag_list_in_order(self):
        tags = hf_compat._resolve_ollama_tags(
            "nemotron3", fetcher=lambda _url: self._NEMOTRON_PAGE
        )
        self.assertEqual([t["tag"] for t in tags], [
            "nemotron3:33b",
            "nemotron3:33b-q8",
            "nemotron3:33b-q4_K_M",
            "nemotron3:33b-bf16",
        ])

    def test_resolver_returns_empty_on_network_failure(self):
        def boom(_url):
            raise OSError("network unreachable")
        self.assertEqual(hf_compat._resolve_ollama_tags("nemotron3", fetcher=boom), [])

    def test_resolver_returns_empty_on_unrelated_html(self):
        self.assertEqual(
            hf_compat._resolve_ollama_tags(
                "nemotron3", fetcher=lambda _url: "<html><body>404</body></html>"
            ),
            [],
        )

    def test_resolver_rejects_dangerous_base_names(self):
        # Defensive: never URL-encode-bypass into a different ollama page.
        for bad in ("", "  ", "foo/bar", "foo:bar", "../etc/passwd"):
            self.assertEqual(
                hf_compat._resolve_ollama_tags(
                    bad, fetcher=lambda _url: self._NEMOTRON_PAGE
                ),
                [],
                f"base name {bad!r} must short-circuit before fetch",
            )

    def test_inspect_bare_tag_pins_canonical_default(self):
        # Replace the network call with a fake that returns the nemotron3 page.
        hf_compat._resolve_ollama_tags = lambda base, **_: [
            {"tag": "nemotron3:33b", "size_label": "28GB", "context_label": "128K"},
            {"tag": "nemotron3:33b-q8", "size_label": "36GB", "context_label": "128K"},
            {"tag": "nemotron3:33b-q4_K_M", "size_label": "28GB", "context_label": "128K"},
        ]
        target = ParsedTarget(route="ollama", ollama_tag="nemotron3")
        result = inspect(target)
        self.assertEqual(result.verdict, "supported")
        # Tag was rewritten from the bare "nemotron3" to the canonical
        # "nemotron3:33b" so the Ollama pull won't 404.
        self.assertEqual(result.proposed_entry["ollama_tag"], "nemotron3:33b")
        self.assertEqual(result.proposed_entry["name"], "nemotron3:33b")
        self.assertIn("nemotron3:33b", result.proposed_entry["source_url"])
        # Reason text surfaces the other size options so the user can
        # re-add with a different size if they want a smaller / bigger build.
        reasons_text = " ".join(result.reasons)
        self.assertIn("nemotron3:33b", reasons_text)
        self.assertIn("33b-q8", reasons_text)

    def test_inspect_bare_tag_with_no_resolvable_sizes_warns(self):
        # /tags page didn't load OR didn't list anything pullable.  Inspector
        # MUST still return a usable entry but surface a clear heads-up so the
        # user knows the pull may fail.
        hf_compat._resolve_ollama_tags = lambda base, **_: []
        target = ParsedTarget(route="ollama", ollama_tag="madeupmodel")
        result = inspect(target)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.proposed_entry["ollama_tag"], "madeupmodel")
        reasons_text = " ".join(result.reasons).lower()
        self.assertIn("manifest does not exist", reasons_text)
        self.assertIn(":size", reasons_text)

    def test_inspect_sized_tag_skips_tag_resolver(self):
        # When the user pasted https://ollama.com/library/llama3.2:1b the
        # tag is already canonical — never hit /tags.
        calls = []
        hf_compat._resolve_ollama_tags = lambda base, **_: calls.append(base) or []
        target = ParsedTarget(route="ollama", ollama_tag="llama3.2:1b")
        result = inspect(target)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.proposed_entry["ollama_tag"], "llama3.2:1b")
        self.assertEqual(calls, [], "sized tags must skip the /tags fetch")


# ── Result-shape invariants ─────────────────────────────────────────────────


class CompatResultShape(unittest.TestCase):

    def test_install_blocked_states(self):
        self.assertTrue(CompatResult(verdict="unsupported").is_install_blocked)
        self.assertTrue(CompatResult(verdict="needs_access").is_install_blocked)
        self.assertFalse(CompatResult(verdict="supported").is_install_blocked)
        self.assertFalse(CompatResult(verdict="warn").is_install_blocked)


class HfClientTranslate(unittest.TestCase):
    """Direct coverage of _HFClient._raise_translated — the boundary that
    converts huggingface_hub exception names into LocalAI's taxonomy.
    Driven by exception type *name* so it works without importing the hub."""

    def _raise(self, exc_name, msg="boom"):
        cls = type(exc_name, (Exception,), {})
        return cls(msg)

    def test_gated_translates(self):
        with self.assertRaises(hf_compat.HfGatedError):
            _HFClient._raise_translated(self._raise("GatedRepoError"))

    def test_repo_not_found_translates(self):
        with self.assertRaises(hf_compat.HfNotFoundError):
            _HFClient._raise_translated(self._raise("RepositoryNotFoundError"))

    def test_revision_not_found_translates(self):
        with self.assertRaises(hf_compat.HfNotFoundError):
            _HFClient._raise_translated(self._raise("RevisionNotFoundError"))

    def test_entry_not_found_translates(self):
        with self.assertRaises(hf_compat.HfNotFoundError):
            _HFClient._raise_translated(self._raise("EntryNotFoundError"))

    def test_timeout_translates(self):
        with self.assertRaises(hf_compat.HfNetworkError):
            _HFClient._raise_translated(self._raise("ConnectTimeout"))

    def test_connection_translates(self):
        with self.assertRaises(hf_compat.HfNetworkError):
            _HFClient._raise_translated(self._raise("ConnectionError"))

    def test_unknown_translates_to_network(self):
        with self.assertRaises(hf_compat.HfNetworkError):
            _HFClient._raise_translated(self._raise("SomethingWeird"))


if __name__ == "__main__":
    unittest.main()

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import catalog
from tools import validate_doc_samples as validator


class DocSampleValidatorTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        with patch.object(sys, "argv", ["validate_doc_samples.py"]):
            return validator.parse_args()

    def test_chat_token_default_matches_reasoning_model_budget(self):
        args = self._args()

        payload = validator._chat_payload(
            "demo:latest",
            [{"role": "user", "content": "hi"}],
            args,
        )

        self.assertEqual(args.chat_max_tokens, 1024)
        self.assertEqual(payload["options"]["num_predict"], 1024)
        self.assertFalse(args.unload_chat)
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertNotIn("think", payload)

        qwen_payload = validator._chat_payload(
            "qwen3:30b-a3b",
            [{"role": "user", "content": "hi"}],
            args,
        )
        self.assertIs(qwen_payload["think"], False)

    def test_model_guide_inventory_includes_every_active_chat_model(self):
        # Post-v5.3.4 doc consolidation: catalog-derived samples all carry
        # source_doc=Model-Guide.html (the four legacy prompt docs were retired
        # — see Archive/doc-consolidation-2026-05/).
        root = Path(__file__).resolve().parents[1]
        models = catalog.load_catalog(root / "models_catalog.json")

        samples = validator.collect_samples(models, self._args(), Path("reference.png"))

        active_chat_ids = {
            model["id"] for model in models if catalog.is_chat_selectable_model(model)
        }
        model_guide_chat_ids = {
            sample.model_id
            for sample in samples
            if sample.source_doc == validator.MODEL_GUIDE.name and sample.surface == "chat"
        }
        self.assertEqual(sorted(active_chat_ids - model_guide_chat_ids), [])
        self.assertIn("qwen3-30b-a3b", model_guide_chat_ids)

    def test_inventory_includes_all_doc_surfaces(self):
        root = Path(__file__).resolve().parents[1]
        models = catalog.load_catalog(root / "models_catalog.json")

        samples = validator.collect_samples(models, self._args(), Path("reference.png"))

        # The three legacy prompt docs (ChatPromptIdeas, ImageGenPrompts,
        # ModelDemoPrompts) and model-value-props.html collapsed into a single
        # Model-Guide.html in v5.3.4. image-gen-guide.html is unchanged.
        self.assertTrue({
            "Model-Guide.html",
            "image-gen-guide.html",
        } <= {sample.source_doc for sample in samples})
        self.assertFalse({
            "ModelDemoPrompts.html",
            "ImageGenPrompts.html",
            "ChatPromptIdeas.html",
            "model-value-props.html",
        } & {sample.source_doc for sample in samples})
        self.assertTrue({
            "chat",
            "image",
            "reference_image",
            "image_to_prompt",
            "toolbox",
        } <= {sample.surface for sample in samples})

    def test_blocked_replacement_prompt_is_not_sent_to_comfyui(self):
        class FakeComfyUI:
            def __init__(self):
                self.calls = 0

            def generate_image(self, **_kwargs):
                self.calls += 1
                return b"image"

            def is_running(self):
                return True

        class FakeApp:
            def __init__(self):
                self.comfyui = FakeComfyUI()

            def _default_negative_prompt_for_entry(self, _entry):
                return ""

            def _ensure_image_model_runtime_support(self, _filename, *, prompt=True):
                return True

            def _image_model_runtime_needs_restart(self, _filename):
                return False

        entry = {
            "id": "demo-image",
            "name": "Demo Image",
            "comfyui_model": "demo.safetensors",
            "min_ram_gb": 1,
            "min_vram_gb": 1,
        }
        sample = validator.DocSample(
            id="demo-sample",
            source_doc="doc.html",
            surface="image",
            model_id="demo-image",
            model_name="Demo Image",
            title="Demo",
            prompt="original safe prompt",
            entry=entry,
            settings={
                "width": 64,
                "height": 64,
                "steps": 1,
                "cfg": 1.0,
                "seed": 1,
                "sampler": "euler",
                "scheduler": "normal",
            },
        )

        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.validate_doc_samples._save_image") as save_image, \
             patch("tools.validate_doc_samples._image_quality", return_value={"bad": True}), \
             patch("tools.validate_doc_samples.get_model_demo", return_value={"samples": ["blocked replacement prompt"]}), \
             patch("tools.validate_doc_samples.content_filter.check_prompt", side_effect=[None, "blocked"]):
            path = Path(tmp) / "original.png"
            path.write_bytes(b"bad")
            save_image.return_value = path
            app = FakeApp()
            result = validator._run_image_sample(app, ["demo.safetensors"], sample, Path(tmp))

        self.assertEqual(app.comfyui.calls, 1)
        self.assertEqual(sample.prompt, "original safe prompt")
        self.assertEqual(result.prompt, "original safe prompt")
        self.assertEqual(result.status, "failed")
        self.assertIn("Replacement prompt blocked", result.error)
        self.assertEqual(result.replacement_prompt, "blocked replacement prompt")

    def test_image_sample_restarts_comfyui_when_runtime_support_missing_from_running_process(self):
        order = []

        class FakeComfyUI:
            def is_running(self):
                return True

            def generate_image(self, **_kwargs):
                order.append("generate")
                return b"image"

        class FakeApp:
            def __init__(self):
                self.comfyui = FakeComfyUI()

            def _default_negative_prompt_for_entry(self, _entry):
                return ""

            def _ensure_image_model_runtime_support(self, _filename, *, prompt=True):
                order.append("ensure")
                return True

            def _image_model_runtime_needs_restart(self, _filename):
                return True

        entry = {
            "id": "demo-image",
            "name": "Demo Image",
            "comfyui_model": "demo.gguf",
            "min_ram_gb": 1,
            "min_vram_gb": 1,
        }
        sample = validator.DocSample(
            id="demo-restart",
            source_doc="doc.html",
            surface="image",
            model_id="demo-image",
            model_name="Demo Image",
            title="Demo",
            prompt="safe prompt",
            entry=entry,
            settings={
                "width": 64,
                "height": 64,
                "steps": 1,
                "cfg": 1.0,
                "seed": 1,
                "sampler": "euler",
                "scheduler": "normal",
            },
        )

        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.validate_doc_samples._save_image") as save_image, \
             patch("tools.validate_doc_samples._image_quality", return_value={"bad": False}), \
             patch("tools.validate_doc_samples._restart_comfyui_for_model_support", side_effect=lambda *_args: order.append("restart")), \
             patch("tools.validate_doc_samples.content_filter.check_prompt", return_value=None):
            path = Path(tmp) / "output.png"
            path.write_bytes(b"ok")
            save_image.return_value = path
            result = validator._run_image_sample(FakeApp(), ["demo.gguf"], sample, Path(tmp))

        self.assertEqual(result.status, "passed")
        self.assertLess(order.index("restart"), order.index("generate"))

    def test_phi_multimodal_toolbox_samples_use_utility_adapter_not_describe_workflow(self):
        sample = validator.DocSample(
            id="phi-toolbox",
            source_doc="Model-Guide.html",
            surface="toolbox",
            model_id="phi-4-multimodal",
            model_name="Phi-4 Multimodal",
            title="Phi",
            prompt="Line one\n\"quoted\" path C:\\demo",
            entry={"id": "phi-4-multimodal", "name": "Phi-4 Multimodal", "min_ram_gb": 16, "min_vram_gb": 0},
        )

        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.validate_doc_samples._toolbox_fixture_paths", return_value={"image": Path(tmp) / "image.png"}), \
             patch("tools.validate_doc_samples.phase1_adapters.run_phi_text", return_value={"status": "ok", "output_text": "done", "metric_label": "Utility", "metric_value": "1 response"}) as run_phi, \
             patch("tools.validate_doc_samples.workflows.describe", side_effect=AssertionError("describe should not be called")):
            result = validator._run_toolbox_sample(sample, Path(tmp))

        self.assertEqual(result.status, "passed")
        run_phi.assert_called_once()
        self.assertEqual(run_phi.call_args.kwargs["prompt"], sample.prompt)


if __name__ == "__main__":
    unittest.main()

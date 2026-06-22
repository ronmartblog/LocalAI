import tempfile
import os
import sys
import types
import unittest
import ast
from pathlib import Path

from src import workflows

ROOT = Path(__file__).resolve().parents[1]
APP_TEXT = (ROOT / "src" / "app.py").read_text(encoding="utf-8")


class ToolboxWorkflowTests(unittest.TestCase):
    def _app_function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    def test_workflow_result_shape(self):
        result = workflows.WorkflowResult("ok", metadata={"elapsed_s": 0.1})
        self.assertEqual(result.output_text, "ok")
        self.assertEqual(result.metadata["elapsed_s"], 0.1)

    def test_embed_and_rank_returns_dataclass(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is an optional dependency and is not installed")
        class FakeVector:
            def __init__(self, score):
                self.score = score

            def __matmul__(self, other):
                return other.score

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                return [FakeVector(1.0), FakeVector(0.9), FakeVector(0.1)]

        fake_module = types.SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: FakeModel())
        old_module = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = fake_module
        try:
            result = workflows.embed_and_rank(
                "local ai",
                ["local private model", "weather report"],
                {"hf_repo": "fake/repo"},
            )
            self.assertIn("Ranked snippets", result.output_text)
        finally:
            if old_module is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = old_module

    def test_write_test_wav_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = workflows.write_test_wav(Path(td) / "sample.wav")
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

    def test_florence_ocr_result_formats_text_without_raw_dict(self):
        text = workflows._format_florence_result({"<OCR>": "LocalAI Studio"}, "<OCR>")

        self.assertEqual(text, "LocalAI Studio")
        self.assertEqual(
            workflows._format_florence_result({"<OCR>": ""}, "<OCR>"),
            "No readable text was returned.",
        )

    def test_table_detection_output_dedupes_overlapping_boxes_and_explains_limits(self):
        detections = [
            {"label": "table", "score": 0.91, "box": [10.0, 10.0, 110.0, 110.0]},
            {"label": "table", "score": 0.72, "box": [12.0, 12.0, 108.0, 108.0]},
            {"label": "table", "score": 0.80, "box": [150.0, 20.0, 250.0, 120.0], "crop": "crop.png"},
        ]

        deduped = workflows._dedupe_table_detections(detections)
        output = workflows._format_table_detections(deduped, (300, 150))

        self.assertEqual(len(deduped), 2)
        self.assertIn("Found 2 likely table regions", output)
        self.assertIn("does not extract cell text yet", output)
        self.assertIn("Crop saved: crop.png", output)

    def test_table_crop_saving_writes_crop_paths(self):
        class FakeCrop:
            def save(self, path):
                Path(path).write_bytes(b"crop")

        class FakeImage:
            width = 100
            height = 80

            def crop(self, box):
                self.last_box = box
                return FakeCrop()

        detections = [{"label": "table", "score": 0.9, "box": [1.2, 2.6, 50.4, 60.8]}]
        with tempfile.TemporaryDirectory() as td:
            image = FakeImage()
            workflows._save_table_crops(image, detections, Path(td))

            crop_path = Path(detections[0]["crop"])
            self.assertTrue(crop_path.exists())
            self.assertEqual(image.last_box, (1, 3, 50, 61))

    def test_toolbox_uses_local_hf_cache_and_improves_first_run_ux(self):
        """HF_HOME defaults to <app>/.cache/huggingface and respects ambient overrides.

        Pre-v5.3.10-hotfix: phase1_adapters unconditionally clobbered HF_HOME
        to <app>/models/phase1 at module import, silently overriding the
        setup.bat / run.bat env-var redirection. Post-hotfix: when no ambient
        HF_HOME is set, the default is <app>/.cache/huggingface (sibling to
        every other v5.3.10 cache redirect), and when an ambient value IS set,
        phase1_adapters preserves it.
        """
        ambient = os.environ.get("HF_HOME", "")
        if ambient:
            self.assertEqual(Path(os.environ["HF_HOME"]), Path(ambient))
        else:
            self.assertEqual(Path(os.environ["HF_HOME"]), ROOT / ".cache" / "huggingface")
        self.assertNotIn("models" + os.sep + "phase1", os.environ["HF_HOME"])
        self.assertNotIn("TRANSFORMERS_CACHE", os.environ)
        self.assertEqual(os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"], "1")
        self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"], "1")
        self.assertIn('"models": ["florence-2-base", "trocr-large-printed", "trocr-base-printed"]', APP_TEXT)
        self.assertIn("Downloading/loading", APP_TEXT)
        self.assertIn("status updates", APP_TEXT)
        self.assertIn('"hf_xet"', APP_TEXT)

    def test_toolbox_install_done_refreshes_import_cache_before_card_state(self):
        source = self._app_function_source("_toolbox_install_done")

        self.assertIn("importlib.invalidate_caches()", source)
        self.assertIn("self._refresh_toolbox_cards()", source)

    def test_toolbox_dependency_install_is_single_flight(self):
        install_source = self._app_function_source("_install_toolbox_requirements")
        refresh_source = self._app_function_source("_refresh_toolbox_cards")
        done_source = self._app_function_source("_toolbox_install_done")
        failed_source = self._app_function_source("_toolbox_install_failed")

        self.assertIn("self._toolbox_install_thread", APP_TEXT)
        self.assertIn("self._toolbox_install_workflow_id", APP_TEXT)
        self.assertIn("self._toolbox_busy()", install_source)
        self.assertIn("Toolbox is already running an install or workflow", install_source)
        self.assertIn("self._toolbox_install_thread = thread", install_source)
        self.assertIn("self._toolbox_install_workflow_id = workflow_id", install_source)
        self.assertIn('parts["action_state"] = "disabled"', refresh_source)
        self.assertIn('parts["action_text"] = "Toolbox busy"', refresh_source)
        self.assertIn("self._clear_toolbox_install_state(workflow_id)", done_source)
        self.assertIn("self._clear_toolbox_install_state(workflow_id)", failed_source)

    def test_toolbox_uses_workflow_browser_and_shared_detail_panel(self):
        build_source = self._app_function_source("_build_toolbox_page")

        self.assertIn("_build_toolbox_workflow_row", APP_TEXT)
        self.assertIn("_build_toolbox_detail_panel", APP_TEXT)
        self.assertIn("_apply_toolbox_browser_density()", build_source)
        self.assertIn("_selected_toolbox_workflow_id", APP_TEXT)
        self.assertIn("_toolbox_detail", APP_TEXT)
        self.assertIn('["transcribe", "speak"]', build_source)
        self.assertIn('["read", "tables", "extract_table", "describe"]', build_source)
        self.assertNotIn("def _build_toolbox_card", APP_TEXT)
        self.assertIn("self._toolbox_cards[workflow_id]", APP_TEXT)
        self.assertIn("_save_toolbox_detail_state", APP_TEXT)

    def test_toolbox_settings_no_longer_exposes_layout_selector(self):
        settings_source = self._app_function_source("_build_settings_page")

        self.assertNotIn('"Toolbox list layout:"', settings_source)
        self.assertNotIn("self._toolbox_layout_var", settings_source)
        self.assertNotIn("command=self._on_toolbox_layout_preview_changed", settings_source)

    def test_save_settings_forces_model_right_toolbox_layout(self):
        save_source = self._app_function_source("_save_settings")

        self.assertIn('self.cfg["toolbox_left_column_mode"] = "normal"', save_source)
        self.assertNotIn("_toolbox_layout_mode_for_label", save_source)

    def test_toolbox_browser_density_forces_model_right_compact_layout(self):
        source = self._app_function_source("_apply_toolbox_browser_density")

        self.assertNotIn("mode = self._toolbox_layout_mode()", source)
        self.assertNotIn('model_right = mode == "normal"', source)
        self.assertNotIn("if model_right:", source)
        self.assertIn("row=1,", source)
        self.assertIn("column=1,", source)
        self.assertIn("wraplength=156", source)
        self.assertNotIn("wraplength=240", source)

    def test_toolbox_row_model_text_uses_fixed_compact_truncation(self):
        from src.app import App

        app = object.__new__(App)
        spec = {"models": ["model-a", "model-b", "model-c"]}
        entry = {"name": "This Model Name Is Way Too Long For Right Layout Comparison"}

        text = App._toolbox_row_model_text(app, spec, entry)
        self.assertIn(" +2", text)
        self.assertIn("...", text)
        self.assertLessEqual(len(text), 40)

    def test_toolbox_unique_workflow_contracts_survive_redesign(self):
        run_source = self._app_function_source("_run_toolbox_workflow")

        self.assertIn('"filetypes": [("Audio", "*.wav *.mp3 *.m4a *.flac")', APP_TEXT)
        self.assertIn('"filetypes": [("Images", "*.png *.jpg *.jpeg *.webp")', APP_TEXT)
        self.assertIn('"models": ["florence-2-base", "trocr-large-printed", "trocr-base-printed"]', APP_TEXT)
        # Table workflows: fast (Ollama minicpm-v) + best (GOT-OCR 2.0).
        self.assertIn('"models": ["minicpm-v-vision"]', APP_TEXT)
        self.assertIn('"models": ["got-ocr2"]', APP_TEXT)
        # Speak now uses Piper, not SpeechT5; sample text is generic
        # (no product naming).
        self.assertIn('"models": ["piper-tts"]', APP_TEXT)
        self.assertIn('"sample": "Welcome. This voice was generated locally', APP_TEXT)
        self.assertIn('"sample": "Query: how do I get my money back', APP_TEXT)
        self.assertIn('"models": ["florence-2-base"]', APP_TEXT)
        self.assertNotIn('"models": ["florence-2-base", "phi-4-multimodal"]', APP_TEXT)
        self.assertIn("workflows.transcribe(input_path, entry, progress_cb=_progress, language=language)", run_source)
        self.assertIn("workflows.read_image(input_path, entry, progress_cb=_progress)", run_source)
        self.assertIn("workflows.extract_table_ollama(input_path, entry, output_dir=out_dir, progress_cb=_progress)", run_source)
        self.assertIn("workflows.extract_table_got(input_path, entry, output_dir=out_dir, progress_cb=_progress)", run_source)
        self.assertIn("workflows.synthesize(input_text, entry, output_dir=out_dir, progress_cb=_progress, language=voice_label)", run_source)
        self.assertIn("workflows.embed_and_rank(query, corpus or lines, entry, progress_cb=_progress)", run_source)
        self.assertIn("workflows.describe(input_path, entry, progress_cb=_progress)", run_source)

    def test_toolbox_workflows_are_single_flight_and_keep_result_actions(self):
        run_source = self._app_function_source("_run_toolbox_workflow")
        done_source = self._app_function_source("_toolbox_workflow_done")

        self.assertIn("self._toolbox_active_workflow_id", APP_TEXT)
        self.assertIn("self._toolbox_workflow_thread", APP_TEXT)
        self.assertIn("self._toolbox_workflow_token", APP_TEXT)
        self.assertIn("if self._toolbox_busy():", run_source)
        self.assertIn("self._toolbox_active_workflow_id = workflow_id", run_source)
        self.assertIn("self._toolbox_workflow_thread = thread", run_source)
        self.assertIn("_set_toolbox_activity", run_source)
        self.assertIn("_set_toolbox_result", run_source)
        self.assertIn("_set_toolbox_output_actions(workflow_id, result.output_path)", done_source)
        self.assertIn("Open output", APP_TEXT)
        self.assertIn("Open audio", APP_TEXT)
        self.assertIn("Open folder", APP_TEXT)
        self.assertIn("Output file was reported but was not found", APP_TEXT)


if __name__ == "__main__":
    unittest.main()

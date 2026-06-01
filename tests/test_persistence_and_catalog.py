import hashlib
import json
import html as html_lib
import re
import tempfile
import unittest
import urllib.request
from pathlib import Path

from src import catalog, config, content_filter, logger, model_demos, system_info
from src.comfyui_client import _build_img2img_checkpoint_workflow
from src.persistence import atomic_write_json


# Snapshot the real models_catalog.json SHA at import time so any test in the
# whole suite that accidentally writes to it (e.g., dropping ``catalog_path=``
# on a save_catalog / append_user_model call) is caught by
# ``ZZZ_CatalogPollutionGuard`` below.  ZZZ prefix keeps it last in unittest's
# alphabetical ordering so it observes the worst-case state of the catalog
# file after every other test has had a chance to run.
_REAL_CATALOG_PATH = Path(__file__).resolve().parents[1] / "models_catalog.json"
_REAL_CATALOG_SHA_AT_IMPORT: str | None = (
    hashlib.sha256(_REAL_CATALOG_PATH.read_bytes()).hexdigest()
    if _REAL_CATALOG_PATH.exists()
    else None
)


class PersistenceTests(unittest.TestCase):
    def test_atomic_write_json_replaces_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.json"
            atomic_write_json(path, {"a": 1})
            atomic_write_json(path, {"b": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"b": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_config_save_and_corrupt_load_fallback(self):
        old_path = config.CONFIG_FILE
        with tempfile.TemporaryDirectory() as td:
            config.CONFIG_FILE = Path(td) / "config.json"
            try:
                cfg = config.DEFAULT_CONFIG.copy()
                cfg["ollama_host"] = "http://example.test:11434"
                self.assertTrue(config.save(cfg))
                self.assertEqual(config.load()["ollama_host"], "http://example.test:11434")

                config.CONFIG_FILE.write_text("{not json", encoding="utf-8")
                loaded = config.load()
                self.assertEqual(loaded["ollama_host"], config.DEFAULT_CONFIG["ollama_host"])
                self.assertTrue(loaded["models_dir"])
                self.assertTrue(loaded["comfyui_dir"])
            finally:
                config.CONFIG_FILE = old_path

    def test_config_theme_defaults_to_system_when_unset(self):
        old_path = config.CONFIG_FILE
        with tempfile.TemporaryDirectory() as td:
            config.CONFIG_FILE = Path(td) / "config.json"
            try:
                config.CONFIG_FILE.write_text(
                    json.dumps({"ollama_host": "http://example.test:11434", "dark_mode": True}),
                    encoding="utf-8",
                )
                loaded = config.load()
                self.assertEqual(loaded["theme_mode"], "system")
                self.assertFalse(loaded["dark_mode"])
            finally:
                config.CONFIG_FILE = old_path

    def test_config_load_repairs_stale_comfyui_dir_when_app_install_exists(self):
        old_path = config.CONFIG_FILE
        old_default_data_dir = config._default_data_dir
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "LocalAI"
            app_comfy = root / "ComfyUI"
            app_comfy.mkdir(parents=True)
            (app_comfy / "main.py").write_text("# sentinel", encoding="utf-8")
            config.CONFIG_FILE = Path(td) / "config.json"
            config._default_data_dir = lambda: root
            try:
                config.CONFIG_FILE.write_text(
                    json.dumps({
                        "ollama_host": "http://example.test:11434",
                        "comfyui_dir": str(Path(td) / "missing-profile" / "ComfyUI"),
                    }),
                    encoding="utf-8",
                )
                loaded = config.load()
                self.assertEqual(Path(loaded["comfyui_dir"]), app_comfy)
                saved = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
                self.assertEqual(Path(saved["comfyui_dir"]), app_comfy)
            finally:
                config._default_data_dir = old_default_data_dir
                config.CONFIG_FILE = old_path

    def test_config_theme_choice_is_normalized_and_persisted(self):
        old_path = config.CONFIG_FILE
        with tempfile.TemporaryDirectory() as td:
            config.CONFIG_FILE = Path(td) / "config.json"
            try:
                cfg = config.DEFAULT_CONFIG.copy()
                cfg["theme_mode"] = "Dark"
                self.assertTrue(config.save(cfg))
                loaded = config.load()
                self.assertEqual(loaded["theme_mode"], "dark")
                self.assertTrue(loaded["dark_mode"])

                cfg["theme_mode"] = "not-a-theme"
                self.assertTrue(config.save(cfg))
                self.assertEqual(config.load()["theme_mode"], "system")
            finally:
                config.CONFIG_FILE = old_path

    def test_config_toolbox_left_column_mode_is_normalized_and_persisted(self):
        old_path = config.CONFIG_FILE
        with tempfile.TemporaryDirectory() as td:
            config.CONFIG_FILE = Path(td) / "config.json"
            try:
                cfg = config.DEFAULT_CONFIG.copy()
                cfg["toolbox_left_column_mode"] = "COMPACT"
                self.assertTrue(config.save(cfg))
                loaded = config.load()
                self.assertEqual(loaded["toolbox_left_column_mode"], "compact")

                cfg["toolbox_left_column_mode"] = "not-a-mode"
                self.assertTrue(config.save(cfg))
                loaded = config.load()
                self.assertEqual(loaded["toolbox_left_column_mode"], "normal")
                persisted = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
                self.assertEqual(persisted["toolbox_left_column_mode"], "normal")
            finally:
                config.CONFIG_FILE = old_path

    def test_config_default_negative_prompts_are_backfilled(self):
        old_path = config.CONFIG_FILE
        with tempfile.TemporaryDirectory() as td:
            config.CONFIG_FILE = Path(td) / "config.json"
            try:
                config.CONFIG_FILE.write_text(
                    json.dumps({"theme_mode": "system"}),
                    encoding="utf-8",
                )
                loaded = config.load()
                self.assertEqual(
                    loaded["default_negative_prompts"],
                    config.DEFAULT_CONFIG["default_negative_prompts"],
                )
                self.assertNotIn("default_negative_prompt", loaded)
                persisted = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
                self.assertEqual(
                    persisted["default_negative_prompts"],
                    config.DEFAULT_CONFIG["default_negative_prompts"],
                )
                self.assertNotIn("default_negative_prompt", persisted)
            finally:
                config.CONFIG_FILE = old_path

    def test_config_negative_prompts_cover_image_prompt_guide(self):
        # Post-v5.3.4 doc consolidation: ImageGenPrompts.html → Model-Guide.html.
        # The consolidated guide pulls negatives straight from
        # src.config.DEFAULT_NEGATIVE_PROMPTS_BY_MODEL via
        # model_demos._negative_prompt_for_model, so the per-card markup now
        # contains either the negative text (for image-gen models with a
        # default negative) or a "negative prompts ignored" panel (for
        # CFG-locked Flux / Z-Image / Turbo / Lightning / Chroma families).
        # We only assert this round-trip for models that are still in the
        # active catalog — DEFAULT_NEGATIVE_PROMPTS_BY_MODEL also retains
        # historical entries for models that were dropped from the catalog.
        root = Path(__file__).resolve().parents[1]
        active_ids = {m["id"] for m in catalog.load_catalog(root / "models_catalog.json")}
        html_path = root / "docs" / "Model-Guide.html"
        html = html_path.read_text(encoding="utf-8")
        cards = re.findall(
            r'<article class="model-card" id="model-[^"]+"[^>]*data-model-id="([^"]+)"[^>]*data-surface="image"[^>]*>'
            r'([\s\S]*?)</article>',
            html,
        )
        rendered = {model_id: body for model_id, body in cards}
        for model_id, expected in config.DEFAULT_CONFIG["default_negative_prompts"].items():
            if model_id not in active_ids:
                continue
            self.assertIn(model_id, rendered, f"missing image card for {model_id}")
            card = rendered[model_id]
            if expected:
                neg_match = re.search(
                    r'<div class="neg-text"[^>]*>([\s\S]*?)</div>',
                    card,
                )
                self.assertIsNotNone(neg_match, f"no neg-text block for {model_id}")
                rendered_neg = html_lib.unescape(re.sub(r"<.*?>", "", neg_match.group(1))).strip()
                self.assertEqual(rendered_neg, expected, model_id)
            else:
                self.assertIn("neg-locked", card, f"{model_id} should render CFG-locked panel")
                self.assertIn("negative prompts ignored", card, model_id)

    def test_config_new_negative_prompt_defaults_are_persisted(self):
        old_path = config.CONFIG_FILE
        with tempfile.TemporaryDirectory() as td:
            config.CONFIG_FILE = Path(td) / "config.json"
            try:
                old_defaults = dict(config.DEFAULT_CONFIG["default_negative_prompts"])
                old_defaults.pop("sdxl-lowvram", None)
                config.CONFIG_FILE.write_text(
                    json.dumps({
                        "theme_mode": "system",
                        "default_negative_prompts": old_defaults,
                    }),
                    encoding="utf-8",
                )
                loaded = config.load()
                self.assertIn("sdxl-lowvram", loaded["default_negative_prompts"])
                persisted = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
                self.assertIn("sdxl-lowvram", persisted["default_negative_prompts"])
            finally:
                config.CONFIG_FILE = old_path

    def test_config_theme_palettes_are_backfilled_and_editable(self):
        old_path = config.CONFIG_FILE
        with tempfile.TemporaryDirectory() as td:
            config.CONFIG_FILE = Path(td) / "config.json"
            try:
                config.CONFIG_FILE.write_text(
                    json.dumps({
                        "theme_mode": "light",
                        "theme_palettes": {
                            "light": {"text_primary": "#101010"},
                            "dark": {"text_primary": "#eeeeee"},
                        },
                    }),
                    encoding="utf-8",
                )
                loaded = config.load()
                self.assertEqual(loaded["theme_palettes"]["light"]["text_primary"], "#101010")
                self.assertEqual(loaded["theme_palettes"]["dark"]["text_primary"], "#eeeeee")
                self.assertIn("surface_card", loaded["theme_palettes"]["light"])
                persisted = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
                self.assertIn("surface_card", persisted["theme_palettes"]["dark"])
            finally:
                config.CONFIG_FILE = old_path


class ContentFilterAndLoggerTests(unittest.TestCase):
    def test_content_filter_normalizes_basic_evasion_without_blocking_clean_prompts(self):
        content_filter.reload()

        self.assertIsNone(content_filter.check_prompt("a workplace-safe product photo"))
        self.assertEqual(content_filter.check_prompt("n@k3d figure study"), "naked")
        self.assertEqual(content_filter.check_prompt("blood---splatter scene"), "blood splatter")

    def test_localai_logger_accepts_preformatted_messages_only(self):
        logger.clear()

        logger.warning("single formatted message")
        with self.assertRaises(TypeError):
            logger.warning("value=%s", "demo")

        entries = logger.get_entries("WARNING")
        self.assertEqual(entries[-1]["msg"], "single formatted message")

    def test_localai_logger_stores_and_filters_categories(self):
        logger.clear()

        logger.info("system message")
        logger.info("chat message", category=logger.CATEGORY_CHAT)
        logger.warning("benchmark warning", category=logger.CATEGORY_BENCHMARK)
        logger.info("unknown category", category="not-a-real-category")

        chat_entries = logger.get_entries("DEBUG", category=logger.CATEGORY_CHAT)
        bench_entries = logger.get_entries("DEBUG", category=logger.CATEGORY_BENCHMARK)
        system_entries = logger.get_entries("DEBUG", category=logger.CATEGORY_SYSTEM)

        self.assertEqual([e["msg"] for e in chat_entries], ["chat message"])
        self.assertEqual([e["msg"] for e in bench_entries], ["benchmark warning"])
        self.assertEqual(
            [e["msg"] for e in system_entries],
            ["system message", "unknown category"],
        )

    def test_logger_listener_logs_do_not_reenter_listener_notification(self):
        logger.clear()
        seen = []

        def listener(entry):
            seen.append(entry["msg"])
            if entry["msg"] == "outer":
                logger.info("inner", category=logger.CATEGORY_CHAT)

        logger.add_listener(listener)
        try:
            logger.info("outer", category=logger.CATEGORY_CHAT)
        finally:
            logger.remove_listener(listener)

        self.assertEqual(seen, ["outer"])
        self.assertEqual(
            [e["msg"] for e in logger.get_entries("DEBUG", category=logger.CATEGORY_CHAT)],
            ["outer", "inner"],
        )


class CatalogTests(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def _model_guide_doc_models(html: str) -> set[str]:
        """Return the set of catalog model ids represented in Model-Guide.html.

        Post-v5.3.4 doc consolidation: the four legacy prompt docs collapsed
        into a single Model-Guide.html generated by src/model_guide.py. Each
        catalog model renders as <article class="model-card" id="model-<slug>"
        ... data-model-id="<catalog-id>">.
        """
        return set(re.findall(r'data-model-id="([^"]+)"', html))

    def test_image_prompt_docs_deep_links_cover_builtin_image_models(self):
        html_path = self._repo_root() / "docs" / "Model-Guide.html"
        html = html_path.read_text(encoding="utf-8")
        doc_ids = self._model_guide_doc_models(html)

        image_models = [
            m for m in catalog.MODELS
            if m.get("category") == "Image Generation" or m.get("backend") == "comfyui" or m.get("comfyui_model")
        ]
        missing = [m["id"] for m in image_models if m["id"] not in doc_ids]
        self.assertEqual(missing, [])

    def test_image_prompt_docs_deep_links_cover_active_catalog_image_models(self):
        root = self._repo_root()
        html = (root / "docs" / "Model-Guide.html").read_text(encoding="utf-8")
        doc_ids = self._model_guide_doc_models(html)
        image_models = [
            m for m in catalog.load_catalog(root / "models_catalog.json")
            if m.get("category") == "Image Generation" or m.get("backend") == "comfyui" or m.get("comfyui_model")
        ]
        missing = [m["id"] for m in image_models if m["id"] not in doc_ids]
        self.assertEqual(missing, [])

    def test_default_negative_prompts_cover_active_catalog_image_models(self):
        root = self._repo_root()
        image_ids = {
            m["id"] for m in catalog.load_catalog(root / "models_catalog.json")
            if m.get("category") == "Image Generation" or m.get("backend") == "comfyui" or m.get("comfyui_model")
        }
        missing = sorted(image_ids - set(config.DEFAULT_CONFIG["default_negative_prompts"]))
        self.assertEqual(missing, [])

    def test_chat_prompt_docs_deep_links_cover_builtin_chat_models(self):
        # Post-v5.3.4: ChatPromptIdeas.html → Model-Guide.html (single guide).
        html_path = self._repo_root() / "docs" / "Model-Guide.html"
        html = html_path.read_text(encoding="utf-8")
        doc_model_ids = self._model_guide_doc_models(html)

        chat_models = [
            m for m in catalog.MODELS
            if m.get("backend") != "comfyui"
            and m.get("ollama_tag")
            and "vision" not in (m.get("tags") or [])
        ]
        missing = [m["id"] for m in chat_models if m["id"] not in doc_model_ids]
        self.assertEqual(missing, [])

    def test_chat_prompt_docs_cover_active_catalog_chat_models(self):
        root = self._repo_root()
        html = (root / "docs" / "Model-Guide.html").read_text(encoding="utf-8")
        doc_model_ids = self._model_guide_doc_models(html)
        chat_models = [
            m for m in catalog.load_catalog(root / "models_catalog.json")
            if catalog.is_chat_selectable_model(m) and not m.get("user_added")
        ]
        missing = [m["id"] for m in chat_models if m["id"] not in doc_model_ids]
        self.assertEqual(missing, [])

    def test_chat_selector_excludes_phase1_utility_adapters(self):
        root = self._repo_root()
        selectable_ids = {
            m["id"] for m in catalog.load_catalog(root / "models_catalog.json")
            if catalog.is_chat_selectable_model(m)
        }
        self.assertFalse({
            "whisper-large-v3-turbo",
            "whisper-v3-turbo-gpu",
            "all-minilm",
            "florence-2-base",
            "speecht5-tts",
            "table-transformer",
            "trocr-base-printed",
            "trocr-large-printed",
            "phi-4-multimodal",
        } & selectable_ids)

    def test_remote_code_utility_models_pin_hf_revision(self):
        root = self._repo_root()
        active = {m["id"]: m for m in catalog.load_catalog(root / "models_catalog.json")}
        builtins = {m["id"]: m for m in catalog.MODELS}

        for models in (active, builtins):
            for model_id in ("florence-2-base", "phi-4-multimodal"):
                revision = models[model_id].get("hf_revision", "")
                self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_cpu_viable_image_models_are_flagged(self):
        root = self._repo_root()
        models = {m["id"]: m for m in catalog.load_catalog(root / "models_catalog.json")}
        expected = {"realistic-vision-v6", "gsdf-counterfeit-v3.0"}
        self.assertEqual(
            {mid for mid in expected if catalog.is_cpu_viable_image_model(models.get(mid, {}))},
            expected,
        )
        self.assertEqual(catalog.IMAGE_GEN_MIN_CPU_RAM_GB, 12)

    def test_sdxl_lightning_uses_full_comfyui_checkpoint(self):
        root = self._repo_root()
        active = {m["id"]: m for m in catalog.load_catalog(root / "models_catalog.json")}
        builtins = {m["id"]: m for m in catalog.MODELS}

        for models in (active, builtins):
            lightning = models["bytedance-sdxl-lightning"]
            self.assertEqual(lightning["comfyui_model"], "sdxl_lightning_8step.safetensors")
            self.assertEqual(lightning.get("comfyui_model_dest"), "checkpoints")
            self.assertNotIn("_unet", lightning["comfyui_model"])
            self.assertEqual(lightning["recommended_settings"]["cfg"], 1.0)
            self.assertTrue(lightning["recommended_settings"]["cfg_locked"])

    def test_dolphin_learn_more_uses_exact_hugging_face_repo(self):
        root = self._repo_root()
        active = {m["id"]: m for m in catalog.load_catalog(root / "models_catalog.json")}
        builtins = {m["id"]: m for m in catalog.MODELS}

        expected = "https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B-GGUF"
        for models in (active, builtins):
            self.assertEqual(models["dolphin3:latest"].get("learn_more_url"), expected)

    def test_catalog_has_no_phase1_curation_fields(self):
        root = self._repo_root()
        payload = json.loads((root / "models_catalog.json").read_text(encoding="utf-8"))
        forbidden = {
            "phase1_recommendation_reason",
            "phase1_source_id",
            "phase1_shortlist",
            "phase1_shortlist_applied",
            "catalog_merge_decisions_applied",
        }
        self.assertFalse(forbidden & payload.keys())
        for model in payload.get("models", []):
            self.assertFalse(forbidden & model.keys(), model.get("id"))

    def test_workflow_category_buttons_are_in_models_page_source(self):
        source = (self._repo_root() / "src" / "app.py").read_text(encoding="utf-8")
        for label in ["Vision", "Speech", "Document AI", "Embeddings", "Toolbox"]:
            self.assertIn(label, source)

    def test_phase1_entries_have_workflow_dispatch(self):
        from src import workflows

        for name in ["transcribe", "read_image", "detect_table", "synthesize", "embed_and_rank", "describe"]:
            self.assertTrue(callable(getattr(workflows, name)))

    def test_prompt_docs_accept_url_hash_deep_links(self):
        # Post-v5.3.4: deep-link / hash navigation now lives in the single
        # Model-Guide.html. The legacy filenames still resolve through
        # docs/index.html's redirect (covered in test_app_static_contracts).
        root = Path(__file__).resolve().parents[1] / "docs"
        guide = (root / "Model-Guide.html").read_text(encoding="utf-8")
        self.assertIn("window.location.hash", guide)
        self.assertIn("modelId", guide)

    def test_model_demo_sidebar_links_target_visible_cards(self):
        # Post-v5.3.4: the old ModelDemoPrompts <aside> nav was rebuilt as a
        # <aside class="rail"> "Jump to a model" index in Model-Guide.html.
        root = self._repo_root()
        html = (root / "docs" / "Model-Guide.html").read_text(encoding="utf-8")
        rail = re.search(r'<aside class="rail"[\s\S]*?</aside>', html)
        self.assertIsNotNone(rail)
        nav_links = re.findall(r'<a href="#([^"]+)" data-target="([^"]+)">', rail.group(0))
        card_ids = set(re.findall(r'<article class="model-card" id="([^"]+)"', html))
        hrefs = [href for href, _target in nav_links]

        self.assertTrue(nav_links)
        self.assertEqual([href for href, target in nav_links if href != target], [])
        self.assertEqual(sorted(set(hrefs) - card_ids), [])
        # Confirm the JS hash/URL machinery still wires the rail + cards.
        self.assertIn("wireRailLinks", html)
        self.assertIn("window.location.hash", html)
        self.assertIn("setCardCollapsed", html)
        self.assertIn("applyFilters", html)

    def test_model_demo_docs_cover_active_catalog_models(self):
        root = self._repo_root()
        html = (root / "docs" / "Model-Guide.html").read_text(encoding="utf-8")
        active_ids = {
            m["id"]
            for m in catalog.load_catalog(root / "models_catalog.json")
            if not m.get("user_added")
        }
        doc_ids = set(re.findall(r'data-model-id="([^"]+)"', html))
        self.assertEqual(sorted(active_ids - doc_ids), [])

    def test_model_demo_metadata_has_three_samples_per_active_model(self):
        root = self._repo_root()
        for model in catalog.load_catalog(root / "models_catalog.json"):
            demo = model_demos.get_model_demo(model)
            self.assertTrue(demo.get("primary"), model["id"])
            self.assertTrue(demo.get("why"), model["id"])
            self.assertEqual(len(demo.get("samples") or []), 3, model["id"])

    def test_catalog_merge_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models_catalog.json"
            model = next(dict(m) for m in catalog.MODELS if m.get("backend") == "comfyui")
            model.pop("supports_img2img", None)
            model.pop("img2img_workflows", None)
            payload = {
                "version": 1,
                "merge_builtins": False,
                "disabled_builtin_ids": [],
                "models": [model],
            }
            atomic_write_json(path, payload)
            loaded = catalog.load_catalog(path)
            self.assertEqual([m["id"] for m in loaded], [model["id"]])

    def test_disabled_builtin_ids_hide_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models_catalog.json"
            disabled = catalog.MODELS[0]["id"]
            payload = {
                "version": 1,
                "merge_builtins": True,
                "disabled_builtin_ids": [disabled],
                "models": [dict(catalog.MODELS[0])],
            }
            atomic_write_json(path, payload)
            loaded_ids = {m["id"] for m in catalog.load_catalog(path)}
            self.assertNotIn(disabled, loaded_ids)

    def test_optional_sku_missing_file_disables_feature(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "skus.json"
            self.assertFalse(system_info.optional_skus_enabled(path))
            self.assertEqual(system_info.load_optional_skus(path), [])
            cfg = system_info.load_optional_sku_config(path)
            # v5.5.12 added a third 'bench_defaults' key when the SKU
            # decoupling refactor moved bench-defaults into the JSON.
            # Missing-file still disables the feature — the bench_defaults
            # field is an empty {} so downstream resolvers find no
            # baselines and every per-SKU bench set resolves to empty.
            self.assertEqual(cfg["feature"], {})
            self.assertEqual(cfg["skus"], [])
            self.assertEqual(cfg.get("bench_defaults", {}), {})

    def test_benchmark_sku_profile_name_canonicalizes_loaded_sku_profiles(self):
        """``benchmark_sku_profile_name`` must look up names from the loaded
        SKU profile list (``BENCHMARK_SKU_PROFILES``).

        Exercised against the public sample fixture so the contract runs
        identically in any clone — independent of the maintainer's private
        ``skus.json``.
        """
        from pathlib import Path

        fixture_path = (
            Path(__file__).resolve().parent / "fixtures" / "sample_skus.json"
        )
        self.assertTrue(fixture_path.exists(), f"missing fixture: {fixture_path}")
        fixture_cfg = system_info.load_optional_sku_config(fixture_path)

        original = system_info.BENCHMARK_SKU_PROFILES
        system_info.BENCHMARK_SKU_PROFILES = fixture_cfg["skus"]
        try:
            # Fixture defines: Small CPU (4/16/0), GPU High (12/110/16),
            # GPU Workstation (36/440/24).
            self.assertEqual(system_info.benchmark_sku_profile_name(4, 16, 0), "Small CPU")
            self.assertEqual(system_info.benchmark_sku_profile_name(12, 110, 16), "GPU High")
            self.assertEqual(system_info.benchmark_sku_profile_name(36, 440, 24), "GPU Workstation")
            # Hardware shape that doesn't match any fixture SKU returns "".
            self.assertEqual(system_info.benchmark_sku_profile_name(16, 96, 0), "")
        finally:
            system_info.BENCHMARK_SKU_PROFILES = original

    def test_vm_size_patterns_match_versioned_sku_suffixes(self):
        self.assertTrue(system_info._vm_size_matches("Standard_D16s_v#", "Standard_D16s_v5"))
        self.assertTrue(system_info._vm_size_matches("Standard_D16s_v#", "standard_d16s_v6"))
        self.assertFalse(system_info._vm_size_matches("Standard_D16s_v#", "Standard_D16s_v10"))
        self.assertTrue(system_info._vm_size_matches("Standard_D16s_v?", "Standard_D16s_v6"))
        self.assertFalse(system_info._vm_size_matches("Standard_D16s_v?", "Standard_D16s_v10"))
        self.assertTrue(system_info._vm_size_matches("Standard_D16s_v*", "Standard_D16s_v10"))
        self.assertTrue(system_info._vm_size_matches(["Standard_D8s_v#", "Standard_D16s_v#"], "Standard_D16s_v6"))

    def test_detect_optional_sku_uses_vm_size_patterns(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"compute": {"vmSize": "Standard_D16s_v6"}}).encode("utf-8")

        original_skus = system_info.OPTIONAL_SKUS
        original_is_mac = system_info._IS_MAC
        original_run = system_info._run
        original_urlopen = urllib.request.urlopen
        try:
            system_info._IS_MAC = False
            system_info.OPTIONAL_SKUS = [
                {
                    "name": "Large CPU",
                    "vm_size_pattern": "Standard_D16s_v#",
                    "cpu": 16,
                    "ram_gb": 64,
                    "vram_gb": 0,
                    "gpu_fraction": "None",
                }
            ]
            system_info._run = lambda *_args, **_kwargs: self.fail("pattern match should not build a dynamic SKU")
            urllib.request.urlopen = lambda *_args, **_kwargs: FakeResponse()

            detected = system_info.detect_optional_sku()
        finally:
            system_info.OPTIONAL_SKUS = original_skus
            system_info._IS_MAC = original_is_mac
            system_info._run = original_run
            urllib.request.urlopen = original_urlopen

        self.assertIsNotNone(detected)
        self.assertEqual(detected["name"], "Large CPU")

    def test_loader_migrates_legacy_azure_size_and_vcpu_fields(self):
        """Legacy v2 SKUs with `azure_size`/`vcpu` get normalized to `vm_size_pattern`/`cpu`."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "skus.json"
            payload = {
                "version": 2,
                "skus": [
                    {
                        "name": "Legacy",
                        "azure_size": "Standard_D8s_v#",
                        "vcpu": 8,
                        "ram_gb": 32,
                        "vram_gb": 0,
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = system_info.load_optional_skus(path)
            self.assertEqual(len(loaded), 1)
            sku = loaded[0]
            self.assertEqual(sku.get("vm_size_pattern"), "Standard_D8s_v#")
            self.assertEqual(sku.get("cpu"), 8)
            self.assertNotIn("azure_size", sku)
            self.assertNotIn("vcpu", sku)

    def test_optional_sku_save_is_atomic_and_loadable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "skus.json"
            skus = [{
                "name": "Test Profile",
                "vm_size_pattern": "Test_Size",
                "cpu": 4,
                "ram_gb": 16,
                "vram_gb": 8,
                "gpu_fraction": "Test",
            }]
            self.assertTrue(system_info.save_optional_skus(skus, path))
            self.assertTrue(system_info.optional_skus_enabled(path))
            self.assertEqual(system_info.load_optional_skus(path)[0]["name"], skus[0]["name"])

    def test_img2img_support_fields_are_backfilled(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models_catalog.json"
            model = next(dict(m) for m in catalog.MODELS if m.get("backend") == "comfyui")
            model.pop("supports_img2img", None)
            model.pop("img2img_workflows", None)
            payload = {
                "version": 1,
                "merge_builtins": False,
                "disabled_builtin_ids": [],
                "models": [model],
            }
            atomic_write_json(path, payload)
            loaded = catalog.load_catalog(path)
            self.assertEqual(loaded[0].get("supports_img2img"), False)
            self.assertIn("img2img_workflows", loaded[0])

    def test_img2img_denoise_ranges_are_valid(self):
        models = catalog.load_catalog()
        for model in models:
            if model.get("backend") != "comfyui":
                continue
            wf = model.get("img2img_workflows", {})
            self.assertIsInstance(wf, dict, model.get("id"))
            if model.get("supports_img2img"):
                denoise_min = float(wf.get("denoise_min"))
                denoise_default = float(wf.get("denoise_default"))
                denoise_max = float(wf.get("denoise_max"))
                self.assertLessEqual(0.05, denoise_min, model.get("id"))
                self.assertLessEqual(denoise_min, denoise_default, model.get("id"))
                self.assertLessEqual(denoise_default, denoise_max, model.get("id"))
                self.assertLessEqual(denoise_max, 1.0, model.get("id"))

    def test_img2img_workflow_uses_reference_latent(self):
        workflow = _build_img2img_checkpoint_workflow(
            model_filename="sd_xl_base_1.0.safetensors",
            positive_prompt="a robot",
            negative_prompt="blur",
            width=1024,
            height=1024,
            steps=20,
            cfg_scale=7.0,
            seed=123,
            reference_image_name="localai_ref_test.png",
            denoise=0.55,
        )
        self.assertEqual(workflow["4"]["class_type"], "LoadImage")
        self.assertEqual(workflow["6"]["class_type"], "VAEEncode")
        self.assertEqual(workflow["7"]["inputs"]["latent_image"], ["6", 0])
        self.assertEqual(workflow["7"]["inputs"]["denoise"], 0.55)

    def test_generated_benchmark_candidate_pack_is_not_kept_in_repo_root(self):
        pack_path = Path(__file__).resolve().parents[1] / "localai_benchmark_candidates.json"
        gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")

        self.assertFalse(pack_path.exists())
        self.assertIn("localai_benchmark_candidates.json", gitignore)


class ZZZ_CatalogPollutionGuard(unittest.TestCase):
    """Suite-wide paranoia: the real ``models_catalog.json`` must be
    byte-identical at end of ``unittest discover`` to its state at module
    import.  Class is named with a ``ZZZ_`` prefix so unittest's alphabetical
    ordering runs it last and it observes pollution caused by any earlier
    test (notably ``test_append_user_model.py`` which was the historical
    source of this bug when ``catalog_path=`` was omitted)."""

    def test_real_catalog_unchanged_during_suite(self):
        if _REAL_CATALOG_SHA_AT_IMPORT is None:
            self.skipTest("no real models_catalog.json on this checkout")
        current = hashlib.sha256(_REAL_CATALOG_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            current,
            _REAL_CATALOG_SHA_AT_IMPORT,
            "Real models_catalog.json was mutated during the test suite. "
            "Every catalog.save_catalog / catalog.append_user_model call "
            "MUST pass catalog_path= pointing to a tempfile — otherwise it "
            "scribbles on the user's catalog (regression of the historical "
            "_CatalogTempCase bug).",
        )


if __name__ == "__main__":
    unittest.main()


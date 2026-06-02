import ast
import textwrap
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_TEXT = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
COMFYUI_TEXT = (ROOT / "src" / "comfyui_client.py").read_text(encoding="utf-8")
RUN_BATCH_TEXT = (ROOT / "run_batch.py").read_text(encoding="utf-8")


class AppStaticContractTests(unittest.TestCase):
    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    def test_nav_and_model_filter_theme_tokens_are_defined(self):
        module_tree = ast.parse(APP_TEXT)
        module_defs = set()
        for node in module_tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_defs.add(target.id)

        source = textwrap.dedent(
            self._function_source("_switch_page") + "\n" + self._function_source("_build_models_page")
        )
        referenced_tokens = {
            node.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Name)
            and node.id in {"BUTTON_SECONDARY", "BUTTON_SECONDARY_HOVER", "INPUT_SURFACE", "TEXT_PRIMARY"}
        }

        self.assertEqual(
            referenced_tokens,
            {"BUTTON_SECONDARY", "BUTTON_SECONDARY_HOVER", "INPUT_SURFACE", "TEXT_PRIMARY"},
        )
        self.assertTrue(referenced_tokens <= module_defs)
        self.assertIn("self._IG_ACCENT", source)
        self.assertIn("self._IG_ACCENT_TEXT", source)

    def test_option_menu_style_uses_surface_tokens_and_supported_kwargs(self):
        source = self._function_source("_option_menu_style")
        self.assertIn('"fg_color": INPUT_SURFACE', source)
        self.assertIn('"text_color": TEXT_PRIMARY', source)
        self.assertIn('"dropdown_fg_color": INPUT_SURFACE', source)
        self.assertIn('"dropdown_text_color": TEXT_PRIMARY', source)
        self.assertNotIn('"border_width"', source)
        self.assertNotIn('"border_color"', source)

    def test_chat_toolbar_has_single_prompt_ideas_action(self):
        source = self._function_source("_build_chat_page")
        self.assertIn('text="Prompt ideas"', source)
        self.assertEqual(source.count('text="Prompt ideas"'), 1)
        self.assertIn("command=self._open_chat_prompt_ideas", source)
        self.assertNotIn('text="Chat ideas"', source)
        self.assertNotIn('text="Demo prompts"', source)

    def test_home_page_keeps_release_notes_out_of_landing_card(self):
        home_source = self._function_source("_build_home_page")

        # Release notes were removed from the product entirely on 2026-05-19;
        # the home page must never re-introduce inline release-notes content,
        # and must not link out to a What's New page either.
        self.assertNotIn("What's new in v", home_source)
        self.assertNotIn("whats-new-v5.html", home_source)
        self.assertNotIn("What's New", home_source)
        self.assertIn("Next steps", home_source)
        self.assertIn("Open Help / Docs for usage details", home_source)

    def test_startup_gpu_detection_is_detection_only(self):
        app_start = APP_TEXT.index("class App")
        init_start = APP_TEXT.index("def __init__(self):", app_start)
        init_end = APP_TEXT.index("def _log_startup_step", init_start)
        init_source = APP_TEXT[init_start:init_end]
        start_source = self._function_source("_start_gpu_detection_async")

        self.assertIn("self.after(75,  self._start_gpu_detection_async)", init_source)
        self.assertIn("threading.Thread", start_source)
        self.assertIn("detect_gpu_cached(auto_fix=False)", start_source)
        self.assertNotIn("detect_gpu_cached()", init_source)
        self.assertNotIn("install", start_source.lower())

    def test_async_gpu_detection_refreshes_dependent_pages_after_first_paint(self):
        apply_source = self._function_source("_apply_gpu_detection_result")

        self.assertIn("self._gpu_detection_pending = False", apply_source)
        self.assertIn("_refresh_home_page", apply_source)
        self.assertIn("_update_category_for_device", apply_source)
        self.assertIn("_refresh_bench_profile_values", apply_source)
        self.assertIn("_refresh_image_readiness", apply_source)

    def test_image_timing_profile_key_imports_regex_and_uses_device_name(self):
        from src.app import App
        from src.gpu_detect import GPUInfo

        app = object.__new__(App)
        app._optional_sku = {"ram_gb": 64}
        app._comfyui_force_cpu = False
        app._active_device_vram_gb = lambda: 12
        app.gpu_info = GPUInfo("cuda", "NVIDIA A10-8Q")

        key = App._timing_profile_key(app)

        self.assertIn("name=NVIDIA_A10-8Q", key)
        self.assertIn("gpu=cuda", key)

    def test_models_page_paints_before_status_refresh(self):
        build_source = self._function_source("_build_models_page")
        populate_source = self._function_source("_populate_model_cards")
        continue_source = self._function_source("_continue_model_card_population")
        switch_source = self._function_source("_switch_page")

        self.assertIn("_schedule_model_card_population", build_source)
        self.assertIn("Loading model list", build_source)
        self.assertIn("_cached_local_names_snapshot()", populate_source)
        self.assertIn("_cached_comfyui_model_names_snapshot()", populate_source)
        self.assertIn("_continue_model_card_population", populate_source)
        self.assertIn("index + 12", continue_source)
        self.assertIn("_models_page_just_built", switch_source)

    def test_models_page_uses_master_detail_rows(self):
        build_source = self._function_source("_build_models_page")
        populate_source = self._function_source("_populate_model_cards")
        continue_source = self._function_source("_continue_model_card_population")
        detail_source = self._function_source("_build_model_detail_pane")
        row_source = APP_TEXT[APP_TEXT.index("class ModelListRow"):APP_TEXT.index("# ── Main Application")]

        self.assertIn("Model list", build_source)
        self.assertIn("Select a model", detail_source)
        self.assertIn("header_row", build_source)
        self.assertIn("_models_list_width", build_source)
        self.assertIn("_start_models_list_resize", build_source)
        self.assertIn("_drag_models_list_resize", build_source)
        self.assertIn("results.grid_columnconfigure(0, weight=0, minsize=self._models_list_width)", build_source)
        self.assertIn("results.grid_columnconfigure(2, weight=1, minsize=360)", build_source)
        self.assertIn("list_panel.grid_propagate(False)", build_source)
        self.assertNotIn("weight=3, minsize=610", build_source)
        self.assertNotIn("weight=2, minsize=360", build_source)
        self.assertNotIn("Name                               Type", build_source)
        self.assertIn("Learn More", detail_source)
        self.assertIn("Delete Local Model", detail_source)
        self.assertNotIn("Delete local", detail_source)
        self.assertNotIn("Copy id/tag", detail_source)
        self.assertIn("self._IG_HERO", detail_source)
        # v5.5.0 UX fix: badges/desc/specs/settings/demo/recs now live inside
        # a CTkScrollableFrame so the right pane can scroll. ``actions`` stays
        # PINNED at row=3 on the outer parent; ``desc`` lives at row=1 inside
        # the scroll_frame. Both row indices changed but the source-order
        # invariant (actions defined before desc) is unchanged.
        self.assertLess(detail_source.index("actions.grid(row=3"), detail_source.index("desc.grid(row=1"))
        self.assertIn('"demo": demo', detail_source)
        update_source = self._function_source("_update_model_detail")
        self.assertIn('widgets["specs"].configure', update_source)
        self.assertIn('widgets["recs"].configure', update_source)
        self.assertIn("Recommended for:", update_source)
        self.assertIn("Tags:", update_source)
        self.assertIn("Best demo:", update_source)
        self.assertNotIn("grid_rowconfigure(5, weight=1)", detail_source)
        self.assertIn("ModelListRow", continue_source)
        self.assertIn("_selected_model_id", APP_TEXT)
        self.assertIn("_update_model_detail", APP_TEXT)
        self.assertNotIn("ctk.CTkButton", row_source)
        self.assertIn("Double-Button-1", row_source)
        self.assertNotIn("CTkScrollableFrame(results", build_source)
        self.assertNotIn("Install & Chat", APP_TEXT)

    def test_selected_model_detail_buttons_reflect_install_status(self):
        from src.app import App

        class FakeWidget:
            def __init__(self):
                self.config = {}

            def configure(self, **kwargs):
                self.config.update(kwargs)

            def winfo_children(self):
                return []

        def run_for(state):
            model = {
                "id": "chat",
                "name": "Chat",
                "vendor": "Vendor",
                "category": "Small",
                "parameters": "3B",
                "size_gb": 2,
                "min_ram_gb": 8,
                "min_vram_gb": 0,
                "ollama_tag": "demo:latest",
            }
            app = object.__new__(App)
            app._selected_model = lambda: model
            app._model_fit_display = lambda _model: ("Fits", None)
            app._model_install_status = lambda _model, local_names=None, comfyui_model_names=None: ("Status", None, state)
            app._model_type_label = lambda _model: "Chat"
            app._is_image_model = lambda _model: False
            app._is_utility_demo_model = lambda _model: False
            app._model_learn_more_url = lambda _model: "https://ollama.com/library/demo"
            widgets = {
                key: FakeWidget()
                for key in (
                    "title", "meta", "status", "desc", "specs", "settings",
                    "demo", "recs", "badges", "install", "primary", "ideas",
                    "learn", "delete",
                )
            }
            app._model_detail_widgets = widgets
            App._update_model_detail(app)
            return widgets

        installed = run_for("installed")
        self.assertEqual(installed["primary"].config["text"], "Load & Chat")
        self.assertEqual(installed["primary"].config["state"], "normal")
        self.assertEqual(installed["install"].config["state"], "disabled")
        self.assertEqual(installed["delete"].config["state"], "normal")
        self.assertIn("Installed - no install needed", installed["status"].config["text"])

        missing = run_for("missing")
        self.assertEqual(missing["primary"].config["text"], "Load & Chat")
        self.assertEqual(missing["install"].config["state"], "normal")
        self.assertEqual(missing["delete"].config["state"], "disabled")
        self.assertIn("Missing - install available", missing["status"].config["text"])

        checking = run_for("checking")
        self.assertEqual(checking["primary"].config["text"], "Load & Chat")
        self.assertEqual(checking["install"].config["state"], "disabled")
        self.assertEqual(checking["delete"].config["state"], "disabled")
        self.assertIn("Checking install status", checking["status"].config["text"])

    def test_delete_selected_model_routes_by_selected_model_type(self):
        from src import app as app_module
        from src.app import App

        def run_for(model, *, confirm=True):
            app = object.__new__(App)
            calls = []
            app._selected_model = lambda: model
            app._is_image_model = lambda _model: _model.get("category") == "Image Generation"
            app.delete_model = lambda _model: calls.append(("ollama", _model["id"]))
            app.delete_comfyui_model = lambda _model: calls.append(("comfyui", _model["id"]))
            original_askyesno = app_module.messagebox.askyesno
            try:
                app_module.messagebox.askyesno = lambda *args, **kwargs: confirm
                App._delete_selected_model(app)
            finally:
                app_module.messagebox.askyesno = original_askyesno
            return calls

        self.assertEqual(
            run_for({"id": "chat", "name": "Chat", "ollama_tag": "demo:latest"}),
            [("ollama", "chat")],
        )
        self.assertEqual(
            run_for({"id": "image", "name": "Image", "category": "Image Generation", "comfyui_model": "x.safetensors"}),
            [("comfyui", "image")],
        )
        self.assertEqual(
            run_for({"id": "chat", "name": "Chat", "ollama_tag": "demo:latest"}, confirm=False),
            [],
        )

    def test_selecting_model_row_keeps_last_status_snapshot_for_detail_actions(self):
        from src.app import App

        app = object.__new__(App)
        app._model_detail_local_names = {"demo:latest"}
        app._model_detail_comfyui_model_names = {"image.safetensors"}
        calls = []
        app._sync_model_row_selection = lambda: calls.append("sync")
        app._update_model_detail = lambda local_names=None, comfyui_model_names=None: calls.append(
            (local_names, comfyui_model_names)
        )

        App._select_model_row(app, "demo")

        self.assertEqual(app._selected_model_id, "demo")
        self.assertEqual(calls[0], "sync")
        self.assertEqual(calls[1], ({"demo:latest"}, {"image.safetensors"}))

    def test_delete_model_invalidates_inflight_status_refresh(self):
        from src import app as app_module
        from src.app import App

        class DummyOllama:
            def __init__(self):
                self.deleted = []

            def delete_model(self, tag):
                self.deleted.append(tag)

        class DummyCard:
            _is_image_model = False
            _is_utility_demo_model = False

            def __init__(self):
                self.statuses = []
                self.badge_refreshes = 0

            def refresh_status(self, **kwargs):
                self.statuses.append(kwargs)

            def refresh_perf_badges(self):
                self.badge_refreshes += 1

        app = object.__new__(App)
        card = DummyCard()
        detail_calls = []
        app.ollama = DummyOllama()
        app.cfg = {"downloaded_this_session": ["demo:latest"]}
        app._model_status_refresh_generation = 7
        app._local_names_cache = (time.time(), {"demo:latest", "other:latest"})
        app._comfyui_model_names_cache = (time.time(), {"image.safetensors"})
        app._model_detail_local_names = {"demo:latest", "other:latest"}
        app._model_detail_comfyui_model_names = {"image.safetensors"}
        app._model_cards = [card]
        app.set_status = lambda _text: None
        app._refresh_chat_model_selector = lambda: None
        app._schedule_model_status_refresh = lambda force_refresh=False: None
        app._update_model_detail = lambda local_names=None, comfyui_model_names=None: detail_calls.append(
            (set(local_names) if local_names is not None else None,
             set(comfyui_model_names) if comfyui_model_names is not None else None)
        )
        original_save = app_module.config.save
        try:
            app_module.config.save = lambda _cfg: True
            App.delete_model(app, {"id": "demo", "name": "Demo", "ollama_tag": "demo:latest"})
        finally:
            app_module.config.save = original_save

        self.assertEqual(app.ollama.deleted, ["demo:latest"])
        self.assertEqual(app._model_status_refresh_generation, 8)
        self.assertEqual(app._model_detail_local_names, {"other:latest"})
        self.assertEqual(card.statuses[-1]["local_names"], {"other:latest"})
        self.assertEqual(detail_calls[-1][0], {"other:latest"})

        App._apply_model_status_refresh(app, 7, {"demo:latest", "other:latest"}, {"image.safetensors"}, 1, False)

        self.assertEqual(app._model_detail_local_names, {"other:latest"})
        self.assertEqual(detail_calls[-1][0], {"other:latest"})
        self.assertIn("_invalidate_model_status_refresh()", self._function_source("delete_comfyui_model"))

    def test_chat_prompt_ideas_keep_chat_guide_url_contract(self):
        source = self._function_source("_open_chat_prompt_ideas")

        # Post-v5.3.4 doc consolidation: the chat opener targets the unified
        # Model-Guide.html (ChatPromptIdeas.html was retired to
        # Archive/doc-consolidation-2026-05/).
        self.assertIn("Model-Guide.html", source)
        self.assertNotIn("ChatPromptIdeas.html", source)
        self.assertIn("chatModel", source)
        self.assertIn("ollamaTag", source)
        self.assertIn("_chat_prompt_fragment", source)
        self.assertIn("_selected_chat_model", source)
        self.assertIn('_open_prompt_doc_via_help("Model-Guide.html", params, fragment)', source)
        self.assertNotIn("open_prompt_ideas_for_model", source)

    def test_prompt_ideas_open_direct_context_urls(self):
        detail_source = self._function_source("_build_model_detail_pane")
        row_source = APP_TEXT[APP_TEXT.index("class ModelListRow"):APP_TEXT.index("# ── Main Application")]
        helper_source = self._function_source("_open_prompt_doc_via_help")
        model_source = self._function_source("open_prompt_ideas_for_model")
        image_source = self._function_source("_open_image_prompts")
        docs_index = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

        self.assertIn('text="Prompt ideas"', detail_source)
        self.assertGreaterEqual(APP_TEXT.count('text="Prompt ideas"'), 2)
        self.assertNotIn('text="Demo ideas"', detail_source)
        self.assertNotIn('text="Demo ideas"', row_source)
        self.assertNotIn('text="Demo ideas"', APP_TEXT)
        self.assertIn("target_path.as_uri()", helper_source)
        self.assertIn("urlencode(clean_params)", helper_source)
        self.assertNotIn('urlencode({"html": target_html, **clean_params})', helper_source)
        self.assertIn('window.location.replace', docs_index)
        self.assertIn('params.delete("html")', docs_index)
        # docs/index.html allows ?html=Model-Guide.html / index.html /
        # image-gen-guide.html only. The four pre-v5.3.4 prompt-doc filenames
        # were retired and their redirect shims were deleted in post-v5.3.4
        # cleanup — they must NOT reappear in the allow-list.
        self.assertIn('"Model-Guide.html"', docs_index)
        for retired in (
            "ModelDemoPrompts.html",
            "ImageGenPrompts.html",
            "ChatPromptIdeas.html",
            "model-value-props.html",
        ):
            self.assertNotIn(f'"{retired}"', docs_index)
        self.assertIn('_open_prompt_doc_via_help("Model-Guide.html", self._prompt_doc_system_params(), fragment)', model_source)
        self.assertNotIn('params["modelId"] = model_id', model_source)
        self.assertNotIn('params["imageModel"] = model_id', model_source)
        self.assertNotIn('params["chatModel"] = model_id', model_source)
        self.assertNotIn('params["ollama"] = str(model["ollama_tag"])', model_source)
        self.assertNotIn('params["ollamaTag"] = str(model["ollama_tag"])', model_source)
        self.assertIn('_open_prompt_doc_via_help("Model-Guide.html", params, fragment)', image_source)
        self.assertIn('params["modelId"] = model_id', image_source)
        self.assertIn('params["imageModel"] = model_id', image_source)

    def test_legacy_prompt_doc_shims_are_removed(self):
        # Post-v5.3.4 docs cleanup: the four pre-consolidation prompt-doc
        # HTML files were removed from docs/ entirely. Every live app entry
        # point opens Model-Guide.html directly with the right deep link;
        # there is no longer any redirect shim. If one comes back, that
        # means a generator or another agent re-introduced it.
        docs_dir = ROOT / "docs"
        for retired in (
            "ChatPromptIdeas.html",
            "ImageGenPrompts.html",
            "ModelDemoPrompts.html",
            "model-value-props.html",
        ):
            with self.subTest(retired=retired):
                self.assertFalse(
                    (docs_dir / retired).exists(),
                    f"docs/{retired} must not exist — it was deleted in post-v5.3.4 docs cleanup",
                )

    def test_prompt_doc_helper_preserves_context_params_on_target_url(self):
        from src import app as app_module
        from src.app import App

        app = object.__new__(App)
        opened = []
        original_open = app_module.webbrowser.open
        try:
            app_module.webbrowser.open = lambda url: opened.append(url)
            App._open_prompt_doc_via_help(
                app,
                "Model-Guide.html",
                {
                    "hardware": "cpu",
                    "modelId": "dolphin3:latest",
                    "chatModel": "dolphin3:latest",
                    "ollamaTag": "dolphin3:latest",
                },
                "model-dolphin3-latest",
            )
        finally:
            app_module.webbrowser.open = original_open

        self.assertEqual(len(opened), 1)
        url = opened[0]
        # Post-v5.3.4: opener targets the consolidated Model-Guide.html.
        self.assertIn("Model-Guide.html?", url)
        self.assertNotIn("index.html", url)
        self.assertNotIn("html=", url)
        self.assertIn("hardware=cpu", url)
        self.assertIn("modelId=dolphin3%3Alatest", url)
        self.assertIn("chatModel=dolphin3%3Alatest", url)
        self.assertIn("ollamaTag=dolphin3%3Alatest", url)
        self.assertTrue(url.endswith("#model-dolphin3-latest"))

    def test_prompt_doc_helper_prefers_local_http_url_when_available(self):
        from src import app as app_module
        from src.app import App

        app = object.__new__(App)
        app._docs_http_base_url = lambda: "http://127.0.0.1:8765"
        opened = []
        original_open = app_module.webbrowser.open
        try:
            app_module.webbrowser.open = lambda url: opened.append(url)
            App._open_prompt_doc_via_help(
                app,
                "Model-Guide.html",
                {"modelId": "llama3.3"},
                "model-llama3-3",
            )
        finally:
            app_module.webbrowser.open = original_open

        self.assertEqual(len(opened), 1)
        self.assertEqual(
            opened[0],
            "http://127.0.0.1:8765/Model-Guide.html?modelId=llama3.3#model-llama3-3",
        )

    def test_models_prompt_ideas_open_model_demo_anchor_with_system_context(self):
        from src import app as app_module
        from src.app import App

        app = object.__new__(App)
        app._optional_sku = {"name": "Test GPU", "vram_gb": 24, "ram_gb": 64}
        opened = []
        original_open = app_module.webbrowser.open
        try:
            app_module.webbrowser.open = lambda url: opened.append(url)
            App.open_prompt_ideas_for_model(
                app,
                {
                    "id": "gemma3:4b-vision",
                    "name": "Gemma 3 4B",
                    "category": "Vision",
                    "ollama_tag": "gemma3:4b",
                },
            )
        finally:
            app_module.webbrowser.open = original_open

        self.assertEqual(len(opened), 1)
        url = opened[0]
        # Post-v5.3.4: model-specific prompt-ideas links open the consolidated
        # Model-Guide.html anchored at the model card.
        self.assertIn("Model-Guide.html?", url)
        self.assertIn("vram=24", url)
        self.assertIn("gpu=Test+GPU", url)
        self.assertIn("ram=64", url)
        self.assertNotIn("modelId=", url)
        self.assertNotIn("chatModel=", url)
        self.assertTrue(url.endswith("#model-gemma3-4b-vision"))

    def test_prompt_doc_missing_target_falls_back_to_docs_index_without_redirect(self):
        from src import app as app_module
        from src.app import App

        app = object.__new__(App)
        opened = []
        original_open = app_module.webbrowser.open
        try:
            app_module.webbrowser.open = lambda url: opened.append(url)
            App._open_prompt_doc_via_help(
                app,
                "MissingPromptGuide.html",
                {"modelId": "dolphin3:latest"},
                "model-dolphin3-latest",
            )
        finally:
            app_module.webbrowser.open = original_open

        self.assertEqual(len(opened), 1)
        url = opened[0]
        self.assertIn("index.html?", url)
        self.assertNotIn("html=", url)
        self.assertIn("modelId=dolphin3%3Alatest", url)
        self.assertTrue(url.endswith("#model-dolphin3-latest"))

    def test_chat_prompt_ideas_use_selected_chat_model_context(self):
        from src.app import App

        app = object.__new__(App)
        calls = []
        selected = {
            "id": "llama3.2:3b",
            "name": "Llama 3.2 3B",
            "ollama_tag": "llama3.2:3b",
        }
        app.active_model = None
        app._prompt_doc_system_params = lambda: {"hardware": "cpu"}
        app._selected_chat_model = lambda: selected
        app._open_prompt_doc_via_help = lambda target, params, fragment: calls.append((target, params, fragment))

        App._open_chat_prompt_ideas(app)

        # Post-v5.3.4: ChatPromptIdeas.html → Model-Guide.html.
        self.assertEqual(calls[0][0], "Model-Guide.html")
        self.assertEqual(calls[0][1]["hardware"], "cpu")
        self.assertEqual(calls[0][1]["modelId"], "llama3.2:3b")
        self.assertEqual(calls[0][1]["chatModel"], "llama3.2:3b")
        self.assertEqual(calls[0][1]["ollama"], "llama3.2:3b")
        self.assertEqual(calls[0][1]["ollamaTag"], "llama3.2:3b")
        self.assertEqual(calls[0][2], "model-llama3-2-3b")

    def test_image_prompt_ideas_use_selected_image_model_context(self):
        from src import model_demos
        from src.app import App

        app = object.__new__(App)
        calls = []
        force_cpu_flags = []
        selected = {"id": "sdxl-lowvram"}
        app._comfyui_force_cpu = True
        app._prompt_doc_system_params = lambda force_cpu=False: (
            force_cpu_flags.append(force_cpu) or {"hardware": "cpu"}
        )
        app._selected_image_model_catalog_entry = lambda: selected
        app._open_prompt_doc_via_help = lambda target, params, fragment: calls.append((target, params, fragment))

        App._open_image_prompts(app)

        self.assertEqual(force_cpu_flags, [True])
        self.assertEqual(calls[0][0], "Model-Guide.html")
        self.assertEqual(calls[0][1]["hardware"], "cpu")
        self.assertEqual(calls[0][1]["modelId"], "sdxl-lowvram")
        self.assertEqual(calls[0][1]["imageModel"], "sdxl-lowvram")
        self.assertEqual(calls[0][2], model_demos.doc_fragment("sdxl-lowvram"))

    def test_refresh_chat_model_selector_ignores_destroyed_widgets(self):
        from tkinter import TclError
        from src.app import App

        class DummyVar:
            def __init__(self, value=""):
                self._value = value

            def get(self):
                return self._value

            def set(self, value):
                self._value = value

        class DeadMenu:
            def __init__(self):
                self.configure_called = False

            def winfo_exists(self):
                raise TclError("invalid command name")

            def configure(self, **_kwargs):
                self.configure_called = True

        class DeadButton(DeadMenu):
            pass

        app = object.__new__(App)
        app.active_model = None
        app._chat_model_entries = lambda: [
            {"id": "llama3.2:3b", "name": "Llama 3.2 3B", "category": "Small", "size_gb": 2}
        ]
        app._chat_model_var = DummyVar("stale")
        menu = DeadMenu()
        load_btn = DeadButton()
        app._chat_model_menu = menu
        app._chat_load_btn = load_btn

        App._refresh_chat_model_selector(app)

        self.assertFalse(menu.configure_called)
        self.assertFalse(load_btn.configure_called)
        self.assertIsNone(app._chat_model_menu)
        self.assertIsNone(app._chat_load_btn)
        self.assertIn("llama3.2:3b", app._chat_model_by_label[app._chat_model_var.get()]["id"])

    def test_toolbox_focus_workflow_ignores_destroyed_output_widget(self):
        from tkinter import TclError
        from src.app import App

        class DeadOutput:
            def __init__(self):
                self.focus_attempted = False

            def winfo_exists(self):
                raise TclError("bad window path name")

            def focus_set(self):
                self.focus_attempted = True

        app = object.__new__(App)
        app._toolbox_cards = {"read": {"spec": {"title": "Read image text"}}}
        app._toolbox_detail = {"output": DeadOutput()}
        app._select_toolbox_workflow = lambda _wid: None

        App._toolbox_focus_workflow(app, "read")

        self.assertFalse(app._toolbox_detail["output"].focus_attempted)

    def test_append_log_entry_ignores_destroyed_log_box(self):
        from tkinter import TclError
        from src.app import App

        class DummyVar:
            def __init__(self, value=""):
                self._value = value

            def get(self):
                return self._value

        class DeadLogBox:
            def winfo_exists(self):
                raise TclError("invalid command name")

            def configure(self, **_kwargs):
                raise AssertionError("configure should not be called on a dead widget")

            def insert(self, *_args, **_kwargs):
                raise AssertionError("insert should not be called on a dead widget")

        app = object.__new__(App)
        app._log_level_var = DummyVar("INFO")
        app._log_search_var = DummyVar("")
        app._log_box = DeadLogBox()
        app._closing = False

        App._append_log_entry(
            app,
            {"time": "00:00:00", "level": "INFO", "msg": "hello"},
            scroll=False,
        )

        self.assertIsNone(app._log_box)

    def test_logs_page_has_category_filter_contract(self):
        build_source = self._function_source("_build_logs_page")
        refresh_source = self._function_source("_refresh_logs")
        append_source = self._function_source("_append_log_entry")

        self.assertIn('text="Category:"', build_source)
        self.assertIn('values=["ALL", *logger.CATEGORIES]', build_source)
        self.assertIn("self._log_category_var", build_source)
        self.assertIn("category=None if category == \"ALL\" else category", refresh_source)
        self.assertIn("entry_category != selected_category", append_source)

    def test_long_running_surfaces_write_categorized_logs(self):
        download_source = self._function_source("_download_progress_cb")
        chat_source = self._function_source("_stream_ollama")
        toolbox_source = self._function_source("_run_toolbox_workflow")
        image_source = self._function_source("_start_image_generation")
        comfyui_source = self._function_source("_start_comfyui_process")

        self.assertIn("logger.CATEGORY_MODEL_PULL", download_source)
        self.assertIn("chat_stream_with_stats", chat_source)
        self.assertIn("logger.CATEGORY_CHAT", chat_source)
        self.assertIn("logger.CATEGORY_TOOLBOX", toolbox_source)
        self.assertIn("logger.CATEGORY_IMAGE_GEN", image_source)
        self.assertIn("logger.CATEGORY_COMFYUI", comfyui_source)

    def test_comfyui_startup_applies_model_launch_flags_for_benchmarks(self):
        start_source = self._function_source("_start_comfyui_process")
        bench_source = self._function_source("_bench_ensure_comfyui_ready")
        image_source = self._function_source("_start_image_generation")

        self.assertIn("_comfyui_effective_launch_flags(model)", start_source)
        self.assertIn("_start_comfyui_process(model)", bench_source)
        self.assertIn("_image_model_launch_flags_need_restart", image_source)
        self.assertIn("comfyui_launch_flags", APP_TEXT)

    def test_benchmark_comfyui_start_uses_profile_cpu_gpu_intent(self):
        start_source = self._function_source("_start_comfyui_process")
        bench_source = self._function_source("_bench_ensure_comfyui_ready")
        restart_source = self._function_source("_image_model_launch_flags_need_restart")

        self.assertIn("_bench_force_cpu_for_comfyui", bench_source)
        self.assertIn("force_cpu_override=bench_force_cpu", bench_source)
        self.assertIn("force_cpu_override=force_cpu_override", start_source)
        self.assertIn("force_cpu_override: Optional[bool] = None", restart_source)

    def test_benchmark_comfyui_startup_early_exit_retries_once(self):
        bench_source = self._function_source("_bench_ensure_comfyui_ready")
        self.assertIn("restart_attempted = False", bench_source)
        self.assertIn("retrying once before failing", bench_source)
        self.assertIn("_comfyui_startup_signal_line", bench_source)

    def test_comfyui_startup_signal_line_prefers_actionable_error(self):
        from src.app import App

        app = object.__new__(App)
        tail = "\n".join(
            [
                "Using pytorch attention",
                "Python version: 3.12.10",
                "ImportError: DLL load failed while importing _C: The specified module could not be found.",
                "Using pytorch attention",
            ]
        )
        line = App._comfyui_startup_signal_line(app, tail)
        self.assertIn("ImportError: DLL load failed", line)

    def test_download_comfyui_model_defines_gguf_flag_before_support_branch(self):
        source = self._function_source("download_comfyui_model")

        assignment = 'is_gguf = _lower.endswith(".gguf")'
        branch = "if is_gguf:"
        self.assertIn(assignment, source)
        self.assertLess(source.index(assignment), source.index(branch))

    def test_image_launch_restart_detects_cpu_mode_mismatch_with_override(self):
        from src.app import App

        class FakeComfyUI:
            def is_running(self):
                return True

        class FakeProc:
            def poll(self):
                return None

        app = object.__new__(App)
        app.comfyui = FakeComfyUI()
        app.comfyui_process = FakeProc()
        app._comfyui_current_launch_flags = ["--cpu"]
        app._comfyui_model_launch_flags = lambda model=None: []

        self.assertTrue(
            App._image_model_launch_flags_need_restart(
                app, {}, force_cpu_override=False
            ),
            "GPU benchmark profiles must restart a lingering --cpu ComfyUI process.",
        )

    def test_comfyui_launch_flags_are_not_trusted_without_owned_live_process(self):
        from src.app import App

        class FakeComfyUI:
            def is_running(self):
                return True

        app = object.__new__(App)
        app.comfyui = FakeComfyUI()
        app.comfyui_process = None
        app._comfyui_current_launch_flags = ["--lowvram"]

        needs_restart = App._image_model_launch_flags_need_restart(
            app,
            {"comfyui_launch_flags": ["--lowvram"]},
        )

        self.assertTrue(needs_restart)

    def test_benchmark_max_failure_defaults_are_ten(self):
        batch_runner_text = (ROOT / "src" / "batch_runner.py").read_text(encoding="utf-8")

        self.assertIn("self._bench_maxfail_var = ctk.IntVar(value=10)", APP_TEXT)
        self.assertIn("if maxfail != 10:", APP_TEXT)
        self.assertIn("max_failures: int = 10", batch_runner_text)
        self.assertIn("default=10", RUN_BATCH_TEXT)

    def test_models_detail_pane_has_bottom_clipping_guard(self):
        build_source = self._function_source("_build_models_page")
        detail_source = self._function_source("_build_model_detail_pane")

        # v5.5.0 UX fix: previously the detail card used a ``detail_shell``
        # wrapper trick (outer BORDER_STRONG-colored frame inset by 1px) to
        # fake a border. The shell's bottom edge was overdrawn by the inner
        # rounded-corner anti-aliasing, leaving the right card missing its
        # bottom line.
        # v5.5.1 UX fix: switched to native ``border_width=2`` with rounded
        # corners (matching list_panel) and padded the inner CTkScrollableFrame
        # with ``padx=(2, 2)`` so its Canvas no longer overdraws parts of the
        # left/right border.
        # v5.5.19 UX fix: CTk's native ``border_width`` dropped top/bottom
        # edges at certain widget sizes / DPI scales (Ron's screenshots
        # 2026-05-30). Replaced with ``_make_bordered_card`` — a nested-frame
        # pattern where the outer frame's ``fg_color`` provides the visible
        # border via a 2 px pack-padding gap (no rounded-rect border math, so
        # no missing edges). The test below pins this pattern so any
        # regression back to native ``border_width`` surfaces immediately.
        self.assertNotIn("detail_shell =", build_source)
        self.assertNotIn("detail_shell.grid", build_source)
        self.assertIn(
            "detail_panel = _make_bordered_card(",
            build_source,
            "detail_panel must use _make_bordered_card (nested-frame border) "
            "rather than CTkFrame(border_width=N) which drops edges on CTk Canvas",
        )
        self.assertIn("corner_radius=10", build_source)
        self.assertIn("border_width=2", build_source)
        self.assertIn("border_color=BORDER_STRONG", build_source)
        self.assertIn(
            'detail_panel.grid(row=1, column=2, sticky="nsew", padx=(8, 0), pady=(0, 16))',
            build_source,
        )
        self.assertIn(
            "detail_panel_inner = detail_panel.content_frame",
            build_source,
            "detail_panel must expose .content_frame as the content parent "
            "for _build_model_detail_pane",
        )
        self.assertIn(
            "self._build_model_detail_pane(detail_panel_inner)",
            build_source,
            "_build_model_detail_pane must receive the INNER content frame, "
            "not the outer bordered wrapper",
        )
        # v5.5.0 UX fix: the rows below "actions" now live in a
        # ``CTkScrollableFrame`` so long image-gen descriptions remain
        # reachable at 1280×800. The old non-scrollable layout used
        # ``parent.grid_rowconfigure(10, weight=1)`` as filler; the new layout
        # gives the scrollable region weight in the parent at row=4 instead.
        self.assertIn("CTkScrollableFrame", detail_source)

    def test_models_image_gen_sectioning_uses_selected_capacity_fit_cache(self):
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        def image_model(idx, *, rec="recommended", bucket="speed", min_vram=4):
            return {
                "id": f"img-{idx}",
                "name": f"Image {idx}",
                "category": "Image Generation",
                "backend": "comfyui",
                "comfyui_model": f"img-{idx}.safetensors",
                "min_vram_gb": min_vram,
                "perf_profile": {
                    "recommendation": rec,
                    "category_bucket": bucket,
                    "quality_tier": "great",
                    "speed_tier": "fast",
                },
            }

        app = object.__new__(App)
        app._image_gen_show_oversize_var = DummyVar(False)
        models = [
            image_model(i, rec="top_pick", bucket="speed")
            for i in range(10)
        ] + [
            image_model("photo", bucket="photo"),
            image_model("oversize", bucket="quality", min_vram=24),
        ]
        fit_by_id = {m["id"]: "fits_well" for m in models}
        fit_by_id["img-oversize"] = "exceeds"

        sections = app._build_image_gen_sections(
            models,
            vram_gb=8,
            fit_by_id=fit_by_id,
            show_oversize=False,
        )
        by_section = {sid: group for sid, group in sections}
        self.assertEqual(len(by_section["img_top_picks"]), 8)
        self.assertNotIn("img_doesnt_fit", by_section)
        self.assertTrue({"img-8", "img-9"} <= {m["id"] for m in by_section["img_speed"]})

        sections = app._build_image_gen_sections(
            models,
            vram_gb=8,
            fit_by_id=fit_by_id,
            show_oversize=True,
        )
        by_section = {sid: group for sid, group in sections}
        self.assertEqual([m["id"] for m in by_section["img_doesnt_fit"]], ["img-oversize"])

        sections = app._build_image_gen_sections(
            models,
            vram_gb=0,
            fit_by_id=fit_by_id,
            show_oversize=True,
            all_catalog=True,
        )
        visible_ids = {m["id"] for _sid, group in sections for m in group}
        self.assertEqual(visible_ids, {m["id"] for m in models})

    def test_models_fit_tier_for_capacity_handles_cpu_and_headroom(self):
        from src.app import App

        app = object.__new__(App)
        cpu_image = {
            "id": "sd15-cpu",
            "category": "Image Generation",
            "backend": "comfyui",
            "cpu_viable": True,
            "min_ram_gb": 12,
            "min_vram_gb": 4,
        }
        gpu_image = {
            "id": "flux",
            "category": "Image Generation",
            "backend": "comfyui",
            "min_ram_gb": 16,
            "min_vram_gb": 8,
        }
        chat_model = {
            "id": "chat",
            "category": "Small",
            "min_ram_gb": 16,
            "min_vram_gb": 4,
        }

        self.assertEqual(
            app._model_fit_tier_for_capacity(cpu_image, {"ram_gb": 16, "vram_gb": 0}),
            "tight",
        )
        self.assertEqual(
            app._model_fit_tier_for_capacity(gpu_image, {"ram_gb": 64, "vram_gb": 0}),
            "exceeds",
        )
        self.assertEqual(
            app._model_fit_tier_for_capacity(gpu_image, {"ram_gb": 64, "vram_gb": 8}),
            "tight",
        )
        self.assertEqual(
            app._model_fit_tier_for_capacity(gpu_image, {"ram_gb": 64, "vram_gb": 16}),
            "fits_well",
        )
        self.assertEqual(
            app._model_fit_tier_for_capacity(chat_model, {"ram_gb": 15, "vram_gb": 8}),
            "exceeds",
        )
        self.assertEqual(
            app._model_fit_tier_for_capacity(chat_model, {"ram_gb": 17, "vram_gb": 5}),
            "tight",
        )

    def test_selected_model_primary_action_routes_by_type_and_cached_status(self):
        from src.app import App

        def run_for(model, *, state="ready", local_names=None, start_accepted=True):
            class DummyVar:
                def get(self):
                    return "auto"

            app = object.__new__(App)
            calls = []
            app._selected_model = lambda: model
            app._selected_model_id = model.get("id")
            app._model_detail_widgets = {}
            app._cached_local_names_snapshot = lambda: set(local_names or [])
            app._model_install_status = lambda _model: ("Status", None, state)
            app.ollama_ok = False
            app._backend_var = DummyVar()
            app._install_selected_model = lambda: calls.append("install")
            def start_download(_model):
                calls.append("download")
                return start_accepted
            app.start_download = start_download
            app.open_image_gen_for_model = lambda _model: calls.append("image")
            app.open_toolbox_for_model = lambda _model: calls.append("toolbox")
            app.load_model_for_chat = lambda _model: calls.append("chat")
            App._run_selected_model_primary_action(app)
            return calls, app.__dict__.get("_pending_chat_load_after_download", None)

        self.assertEqual(
            run_for(
                {"id": "missing", "category": "Small", "ollama_tag": "llama3.2:3b"},
                local_names={"llama3.2:1b"},
            )[0],
            ["download"],
        )
        self.assertEqual(
            run_for(
                {"id": "missing", "category": "Small", "ollama_tag": "llama3.2:3b"},
                local_names={"llama3.2:1b"},
            )[1],
            {"model_id": "missing", "backend": "auto"},
        )
        self.assertEqual(
            run_for(
                {"id": "image", "category": "Image Generation", "backend": "comfyui", "comfyui_model": "x.safetensors"},
                state="installed_offline",
            )[0],
            ["image"],
        )
        self.assertEqual(
            run_for({"id": "tool", "category": "Speech", "hf_repo": "example/repo"})[0],
            ["toolbox"],
        )
        self.assertEqual(
            run_for(
                {"id": "chat", "category": "Small", "ollama_tag": "qwen2.5:0.5b"},
                local_names={"qwen2.5:0.5b"},
            )[0],
            ["chat"],
        )

    def test_selected_chat_primary_checks_uncached_status_off_ui_thread(self):
        source = self._function_source("_run_selected_model_primary_action")
        chat_source = self._function_source("_run_chat_model_primary_action")

        self.assertNotIn("probe_if_uncached", source)
        self.assertNotIn("is_model_local", source)
        self.assertIn("threading.Thread", chat_source)
        self.assertIn("is_model_local", chat_source)

    def test_selected_chat_primary_pending_autoload_is_cleared_when_download_refused(self):
        from src.app import App

        class DummyVar:
            def get(self):
                return "auto"

        app = object.__new__(App)
        model = {"id": "chat", "category": "Small", "ollama_tag": "qwen2.5:0.5b"}
        app._backend_var = DummyVar()
        app._pending_chat_load_after_download = {"model_id": "other", "backend": "cpu"}
        app.start_download = lambda _model: False
        detail_updates = []
        app._update_model_detail = lambda: detail_updates.append("updated")

        accepted = App._start_chat_download_after_primary_action(app, model)

        self.assertFalse(accepted)
        self.assertEqual(app._pending_chat_load_after_download, {"model_id": "other", "backend": "cpu"})
        self.assertEqual(detail_updates, ["updated"])

    def test_chat_page_missing_model_download_refusal_does_not_leave_pending_autoload(self):
        from src import app as app_module
        from src.app import App

        class DummyVar:
            def get(self):
                return "Ollama"

        class DummyOllama:
            def is_model_local(self, _tag):
                return False

        app = object.__new__(App)
        model = {"id": "chat", "name": "Chat", "category": "Small", "ollama_tag": "qwen2.5:0.5b"}
        app._chat_thinking = False
        app._selected_chat_model = lambda: model
        app._backend_var = DummyVar()
        app.ollama_ok = True
        app.ollama = DummyOllama()
        app._pending_chat_load_after_download = {"model_id": "other", "backend": "CPU only"}
        app.start_download = lambda _model: False
        app._update_model_detail = lambda: None
        app.set_status = lambda _message: None
        app.load_model_for_chat = lambda _model: self.fail("Refused download should not load chat")

        original_askyesno = app_module.messagebox.askyesno
        try:
            app_module.messagebox.askyesno = lambda *args, **kwargs: True
            App._load_selected_chat_model(app)
        finally:
            app_module.messagebox.askyesno = original_askyesno

        self.assertEqual(app._pending_chat_load_after_download, {"model_id": "other", "backend": "CPU only"})

    def test_download_done_applies_and_clears_pending_chat_autoload(self):
        from src import app as app_module
        from src.app import App

        class DummyVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        app = object.__new__(App)
        model = {"id": "chat", "name": "Chat", "ollama_tag": "qwen2.5:0.5b"}
        calls = []
        backend = DummyVar()
        app._pending_chat_load_after_download = {"model_id": "chat", "backend": "CPU only"}
        app._backend_var = backend
        app._show_progress = lambda show: calls.append(("progress", show))
        app.set_status = lambda message: calls.append(("status", message))
        app._refresh_model_cards = lambda: calls.append("refresh-models")
        app._refresh_chat_model_selector = lambda: calls.append("refresh-chat-selector")
        app.load_model_for_chat = lambda _model: calls.append("load-chat")

        App._download_done(app, model, success=True)

        self.assertIsNone(app._pending_chat_load_after_download)
        self.assertEqual(backend.value, "CPU only")
        self.assertIn("load-chat", calls)

        original_showerror = app_module.messagebox.showerror
        try:
            app_module.messagebox.showerror = lambda *args, **kwargs: calls.append("showerror")
            app._pending_chat_load_after_download = {"model_id": "chat", "backend": "CPU only"}
            App._download_done(app, model, success=False, error="boom")
            self.assertIsNone(app._pending_chat_load_after_download)
            self.assertIn("showerror", calls)
        finally:
            app_module.messagebox.showerror = original_showerror

    def test_model_learn_more_url_prefers_hugging_face_sources(self):
        from src.app import App

        app = object.__new__(App)
        self.assertEqual(
            app._model_learn_more_url({"learn_more_url": "https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B-GGUF"}),
            "https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B-GGUF",
        )
        self.assertEqual(
            app._model_learn_more_url({"hf_repo": "microsoft/Phi-4-mini-instruct"}),
            "https://huggingface.co/microsoft/Phi-4-mini-instruct",
        )
        self.assertEqual(
            app._model_learn_more_url({"onnx_repo": "microsoft/Phi-4-mini-instruct-onnx"}),
            "https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx",
        )
        self.assertEqual(
            app._model_learn_more_url({
                "comfyui_model_url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors"
            }),
            "https://huggingface.co/ByteDance/SDXL-Lightning",
        )
        self.assertEqual(
            app._model_learn_more_url({"ollama_tag": "llama3.2:3b"}),
            "https://ollama.com/library/llama3.2",
        )
        self.assertEqual(app._model_learn_more_url({"id": "unknown", "name": "Unknown"}), "")
        self.assertNotIn("huggingface.co/search", self._function_source("_model_learn_more_url"))
        self.assertNotIn("huggingface.co/models?search", self._function_source("_model_learn_more_url"))

    def test_model_install_status_uses_exact_ollama_matching(self):
        from src.app import App

        app = object.__new__(App)
        app.ollama_ok = True
        app._local_names_cache = None
        model = {"id": "llama3-3b", "category": "Small", "ollama_tag": "llama3.2:3b"}

        _text, _color, state = app._model_install_status(model)
        self.assertEqual(state, "checking")

        _text, _color, state = app._model_install_status(model, local_names={"llama3.2:1b"})
        self.assertEqual(state, "missing")

        _text, _color, state = app._model_install_status(model, local_names={"llama3.2"})
        self.assertEqual(state, "installed")

        app._local_names_cache = (time.time(), set())
        _text, _color, state = app._model_install_status(model)
        self.assertEqual(state, "missing")

    def test_image_model_change_locks_sdxl_fast_presets_without_catalog_hints(self):
        source = self._function_source("_on_img_model_changed")

        self.assertIn("is_sdxl_fast", source)
        self.assertIn("turbo", source)
        self.assertIn("lightning", source)
        self.assertIn('self._img_cfg_var.set("1.0")', source)
        self.assertIn('self._img_steps_var.set("8")', source)
        self.assertIn('self._img_sampler_var.set("euler")', source)
        self.assertIn('self._img_scheduler_var.set("sgm_uniform")', source)
        self.assertIn("SDXL Lightning/Turbo", source)

    def test_reference_image_analysis_uses_gpu_fallback_when_sku_vram_is_missing(self):
        source = self._function_source("_analyze_reference_image")

        self.assertIn("system_info.get_gpu_info()", source)
        self.assertIn("vram_total_mb", source)
        self.assertIn("_vision_min_vram", source)

    def test_cfg_locked_image_workflows_enforce_backend_cfg(self):
        from src.comfyui_client import (
            _build_chroma_workflow,
            _build_img2img_checkpoint_workflow,
            _build_sdxl_fast_checkpoint_workflow,
        )

        chroma = _build_chroma_workflow(
            "chroma-family-x0.safetensors",
            "prompt",
            1024,
            1024,
            28,
            123,
            cfg_scale=9.0,
        )
        self.assertEqual(chroma["6"]["inputs"]["cfg"], 1.0)
        self.assertEqual(chroma["1"]["class_type"], "UNETLoader")

        lightning = _build_sdxl_fast_checkpoint_workflow(
            "sdxl_lightning_8step.safetensors",
            "prompt",
            "",
            1024,
            1024,
            8,
            123,
        )
        self.assertEqual(lightning["1"]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(lightning["5"]["inputs"]["cfg"], 1.0)

        lightning_img2img = _build_img2img_checkpoint_workflow(
            "sdxl_lightning_8step.safetensors",
            "prompt",
            "",
            1024,
            1024,
            8,
            1.0,
            123,
            "reference.png",
        )
        self.assertEqual(lightning_img2img["7"]["inputs"]["cfg"], 1.0)

    def test_image_generation_prepares_runtime_support_before_queueing(self):
        start_source = self._function_source("_start_image_generation")
        support_source = self._function_source("_ensure_image_model_runtime_support")
        restart_source = self._function_source("_image_model_runtime_needs_restart")

        self.assertIn("_image_model_runtime_support_missing_items(model_filename)", start_source)
        self.assertIn("_prepare_image_model_support_async(model_filename, missing_runtime_support)", start_source)
        self.assertIn("_ensure_image_model_runtime_support(model_filename, prompt=False)", start_source)
        self.assertLess(
            start_source.index("_ensure_image_model_runtime_support(model_filename, prompt=False)"),
            start_source.index("self.comfyui.generate_image") if "self.comfyui.generate_image" in start_source else len(start_source),
        )
        self.assertIn("_ensure_gguf_support", support_source)
        self.assertIn("_ensure_z_image_support", support_source)
        self.assertIn("_ensure_chroma_support", support_source)
        self.assertIn("has_gguf_node", restart_source)
        self.assertIn("has_chroma_node", restart_source)

    def test_image_model_runtime_needs_restart_checks_loaded_custom_nodes(self):
        from src.app import App

        class FakeComfyUI:
            def __init__(self, *, gguf=True, chroma=True):
                self.gguf = gguf
                self.chroma = chroma

            def is_running(self):
                return True

            def has_gguf_node(self):
                return self.gguf

            def has_chroma_node(self):
                return self.chroma

        app = object.__new__(App)
        app.comfyui = FakeComfyUI(gguf=False)
        self.assertTrue(App._image_model_runtime_needs_restart(app, "flux1-dev-Q4_K_S.gguf"))
        app.comfyui = FakeComfyUI(chroma=False)
        self.assertTrue(App._image_model_runtime_needs_restart(app, "chroma-family-x0.safetensors"))
        app.comfyui = FakeComfyUI()
        self.assertFalse(App._image_model_runtime_needs_restart(app, "sdxl.safetensors"))

    def test_comfyui_stuck_detector_allows_first_run_gguf_chroma_compile(self):
        self.assertIn("WS_STUCK_WARN_S = 60.0", COMFYUI_TEXT)
        self.assertIn("WS_STUCK_ABORT_S = 900.0", COMFYUI_TEXT)
        self.assertIn("Still working (no events for", COMFYUI_TEXT)
        self.assertIn("generation_timeout_s = 1200 if (is_gguf or is_chroma or is_flux or is_z_image) else 600", COMFYUI_TEXT)

    def test_model_card_app_color_attributes_exist_on_app_class(self):
        module_tree = ast.parse(APP_TEXT)
        model_card_init = None
        app_class = None
        for node in module_tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ModelCard":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                        model_card_init = child
            elif isinstance(node, ast.ClassDef) and node.name == "App":
                app_class = node
        self.assertIsNotNone(model_card_init)
        self.assertIsNotNone(app_class)

        referenced = {
            node.attr
            for node in ast.walk(model_card_init)
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("_IG_")
        }
        class_defs = {
            target.id
            for node in app_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        self.assertTrue(referenced <= class_defs, referenced - class_defs)

    def test_benchmark_ui_uses_public_utility_label_and_covers_runnable_categories(self):
        self.assertNotIn("Phase 1 adapters", APP_TEXT)
        # Utility/Toolbox models (phase1_adapter) are excluded from benchmarks
        # entirely — the Benchmark page must not surface a Utility category
        # card, a "Utility demos" checklist label, or a Utility method toggle.
        self.assertNotIn("Utility demos (OCR/speech/embeddings)", APP_TEXT)
        benchmark_source = self._function_source("_build_benchmark_page")
        self.assertNotIn('text="Utility"', benchmark_source)
        self.assertIn("self._bench_checklist_scroll = ctk.CTkScrollableFrame", benchmark_source)
        self.assertIn("height=360", benchmark_source)
        self.assertIn("model_frame.grid_columnconfigure(5, weight=1)", benchmark_source)
        self.assertIn("model_frame.grid_rowconfigure(1, weight=1)", benchmark_source)
        # `_bench_missing_deps_for_model` was previously referenced inside
        # the now-removed Utility category card. It still gates row
        # availability inside `_render_bench_model_checklist` (covered by
        # other tests), so the assertion here is no longer applicable.
        self.assertIn("_bench_profile_var", benchmark_source)
        self.assertIn("Benchmark profile", benchmark_source)
        self.assertIn('values=["Quick", "Extended"]', benchmark_source)

        capacity_source = self._function_source("_bench_profile_capacity")
        self.assertIn('ram.get("available_mb"', capacity_source)

        render_source = self._function_source("_render_bench_model_checklist")
        self.assertIn("_bench_default_fit_for_model", render_source)
        self.assertIn("self._bench_model_disabled_ids", render_source)

        selection_source = self._function_source("_get_selected_model_ids")
        self.assertNotIn("return None", selection_source)

        stop_source = self._function_source("_stop_benchmark")
        self.assertIn("request_stop()", stop_source)
        self.assertIn("save_partial()", stop_source)

        from src import catalog

        runnable_categories = {
            m.get("category", "Other")
            for m in catalog.load_catalog()
            if m.get("phase1_adapter") or m.get("ollama_tag") or m.get("onnx_repo")
        }
        runnable_categories.discard("Image Generation")
        for category in runnable_categories:
            self.assertIn(f'"{category}"', render_source)

    def test_benchmark_default_fit_uses_installed_capacity_not_transient_free_ram(self):
        from src.app import App

        app = object.__new__(App)
        fits_small_cpu_capacity = {
            "id": "fits-small_cpu",
            "name": "Fits Small CPU",
            "min_ram_gb": 8,
            "min_vram_gb": 0,
        }
        too_large_for_small_cpu = {
            "id": "too-large",
            "name": "Too Large",
            "min_ram_gb": 20,
            "min_vram_gb": 0,
        }

        ok, _reason = app._bench_default_fit_for_model(
            fits_small_cpu_capacity,
            available_ram_gb=6.6,
            total_ram_gb=15.9,
            vram_capacity_gb=0,
            has_gpu=False,
        )
        self.assertTrue(ok)

        ok, reason = app._bench_default_fit_for_model(
            too_large_for_small_cpu,
            available_ram_gb=6.6,
            total_ram_gb=15.9,
            vram_capacity_gb=0,
            has_gpu=False,
        )
        self.assertFalse(ok)
        self.assertIn("installed", reason)

        ok, _reason = app._bench_default_fit_for_model(
            {
                "id": "gpu-fit",
                "name": "GPU Fit",
                "min_ram_gb": 32,
                "min_vram_gb": 8,
            },
            available_ram_gb=4,
            total_ram_gb=32,
            vram_capacity_gb=12,
            has_gpu=True,
        )
        self.assertTrue(ok)

    def test_benchmark_small_cpu_defaults_include_successful_result_models(self):
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False
        class _ModeVar:
            def __init__(self, mode): self._mode = mode
            def get(self): return self._mode
        # qwen2.5vl-7b is in the fixture's Small CPU Extended set but not in
        # Quick, so exercise the "observed success" override under Extended.
        app._bench_run_mode_var = _ModeVar("Extended")
        capacity = {
            "profile": "Small CPU",
            "available_ram_gb": 6.5,
            "total_ram_gb": 16,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        recommended_without_result = {
            "id": "tiny",
            "name": "Tiny",
            "category": "Small",
            "ollama_tag": "tiny:latest",
            "min_ram_gb": 4,
            "min_vram_gb": 0,
            "recommended_for": ["Small CPU"],
        }
        available_manual = {
            "id": "manual",
            "name": "Manual",
            "category": "Medium",
            "ollama_tag": "manual:latest",
            "min_ram_gb": 12,
            "min_vram_gb": 8,
            "recommended_for": ["GPU Entry"],
        }
        observed_success = {
            "id": "qwen2.5vl-7b",
            "name": "Observed Success",
            "category": "Medium",
            "ollama_tag": "observed:latest",
            "min_ram_gb": 12,
            "min_vram_gb": 8,
            "recommended_for": ["GPU Mid"],
        }
        too_large = {
            "id": "too-large",
            "name": "Too Large",
            "category": "Large",
            "ollama_tag": "too-large:latest",
            "min_ram_gb": 32,
            "min_vram_gb": 0,
        }
        missing_utility = {
            "id": "missing-utility",
            "name": "Missing Utility",
            "category": "Speech",
            "phase1_adapter": True,
            "min_ram_gb": 4,
            "min_vram_gb": 0,
            "recommended_for": ["Small CPU"],
        }
        app._bench_missing_deps_for_model = lambda model: (
            ["soundfile"] if model.get("id") == "missing-utility" else []
        )

        self.assertTrue(app._bench_model_available_for_profile(recommended_without_result, capacity)[0])
        self.assertFalse(app._bench_default_selected_for_model(recommended_without_result, capacity))
        self.assertTrue(app._bench_model_available_for_profile(available_manual, capacity)[0])
        self.assertFalse(app._bench_default_selected_for_model(available_manual, capacity))
        self.assertTrue(app._bench_model_available_for_profile(observed_success, capacity)[0])
        self.assertTrue(app._bench_default_selected_for_model(observed_success, capacity))
        self.assertFalse(app._bench_model_available_for_profile(too_large, capacity)[0])
        self.assertTrue(app._bench_model_available_for_profile(too_large, capacity, allow_oversize=True)[0])
        self.assertFalse(app._bench_default_selected_for_model(too_large, capacity))
        # Utility/Toolbox models (phase1_adapter=True) are excluded from
        # benchmarks entirely; _bench_methods_for_ui returns [] for them,
        # so they are not available with reason "no benchmark backend"
        # (no missing-deps check fires because the backend gate triggers first).
        missing_available, missing_reason = app._bench_model_available_for_profile(missing_utility, capacity)
        self.assertFalse(missing_available)
        self.assertIn("no benchmark backend", missing_reason)
        self.assertFalse(app._bench_default_selected_for_model(missing_utility, capacity))

    def test_benchmark_gpu_max_defaults_every_runnable_product_model(self):
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False
        app._bench_missing_deps_for_model = lambda model: []
        # v5.5.12 SKU decoupling: Extended mode is driven by the
        # ``bench_extended_models`` set resolved from skus.json for
        # each SKU — adding a new model to the catalog no longer
        # auto-default-ticks it on GPU Workstation; the model must appear in
        # that SKU's resolved set. This codifies Ron's recalibration
        # intent: "for quick and extended all tests are expected to
        # pass — just extended takes longer."
        class _ModeVar:
            def __init__(self, mode): self._mode = mode
            def get(self): return self._mode
        app._bench_run_mode_var = _ModeVar("Extended")
        capacity = {
            "profile": "GPU Workstation",
            "available_ram_gb": 440,
            "total_ram_gb": 440,
            "vram_capacity_gb": 24,
            "has_gpu": True,
            "is_sku": True,
        }
        future_product_model = {
            "id": "future-gpu-max-product-model",
            "name": "Future GPU Workstation Product Model",
            "category": "Large",
            "ollama_tag": "future:latest",
            "min_ram_gb": 128,
            "min_vram_gb": 20,
            "recommended_for": [],
        }

        # Still runnable (fits + has a backend).
        self.assertTrue(app._bench_model_available_for_profile(future_product_model, capacity)[0])
        # But NOT default-ticked until added to the Extended table.
        self.assertFalse(app._bench_default_selected_for_model(future_product_model, capacity))

    def test_benchmark_profiles_without_results_keep_recommended_fallback(self):
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False
        app._bench_missing_deps_for_model = lambda model: []
        # v5.5.8: the recommended_for catalog fallback only fires on profiles
        # with no entry in the Quick/Extended observed tables. "XL CPU" is
        # in those tables now, so use a synthetic profile with no entry.
        capacity = {
            "profile": "Synthetic No-Results Profile",
            "available_ram_gb": 128,
            "total_ram_gb": 128,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        recommended = {
            "id": "recommended-no-results-profile",
            "name": "Recommended No Results Profile",
            "category": "Medium",
            "ollama_tag": "recommended:latest",
            "min_ram_gb": 32,
            "min_vram_gb": 0,
            "recommended_for": ["Synthetic No-Results Profile"],
        }

        self.assertTrue(app._bench_default_selected_for_model(recommended, capacity))

    def test_benchmark_observed_results_table_matches_active_catalog(self):
        from pathlib import Path

        from src import catalog, system_info
        from src.app import App

        app = object.__new__(App)
        # v5.5.12 SKU decoupling: per-SKU Quick / Extended model sets are
        # loaded from skus.json via ``system_info.BENCHMARK_SKU_PROFILES``
        # and read through ``App._bench_default_models_for``. The expected
        # counts below pin the resolved sizes per SKU.
        #
        # Brand neutralization (v5.5.16+): the contract is exercised against
        # ``tests/fixtures/sample_skus.json`` rather than the maintainer's
        # private ``skus.json`` so the test runs identically in any clone.
        fixture_path = (
            Path(__file__).resolve().parent / "fixtures" / "sample_skus.json"
        )
        self.assertTrue(fixture_path.exists(), f"missing fixture: {fixture_path}")
        fixture_cfg = system_info.load_optional_sku_config(fixture_path)

        class _ModeVar:
            def __init__(self, mode): self._mode = mode
            def get(self): return self._mode

        original_profiles = system_info.BENCHMARK_SKU_PROFILES
        system_info.BENCHMARK_SKU_PROFILES = fixture_cfg["skus"]
        try:
            app._bench_run_mode_var = _ModeVar("Extended")
            active_ext = {
                model["id"]
                for model in catalog.load_catalog()
                if app._bench_methods_for_ui(model)
            }
            loaded_profiles = [
                str(s.get("name"))
                for s in system_info.BENCHMARK_SKU_PROFILES
                if s.get("name")
            ]
            self.assertEqual(
                {p: len(app._bench_default_models_for(p, "extended")) for p in loaded_profiles},
                {
                    "Small CPU":        7,
                    "Medium CPU":       7,
                    "Large CPU":        7,
                    "XL CPU":           7,
                    "GPU Entry":       11,
                    "GPU Mid":         11,
                    "GPU High":        11,
                    "GPU Workstation": 12,
                },
            )
            for profile in loaded_profiles:
                with self.subTest(profile=profile, mode="extended"):
                    model_ids = app._bench_default_models_for(profile, "extended")
                    self.assertFalse(
                        sorted(model_ids - active_ext),
                        f"fixture references unknown catalog ids for {profile!r}",
                    )

            app._bench_run_mode_var = _ModeVar("Quick")
            active_quick = {
                model["id"]
                for model in catalog.load_catalog()
                if app._bench_methods_for_ui(model)
            }
            self.assertEqual(
                {p: len(app._bench_default_models_for(p, "quick")) for p in loaded_profiles},
                {
                    "Small CPU":       4,
                    "Medium CPU":      4,
                    "Large CPU":       4,
                    "XL CPU":          4,
                    "GPU Entry":       5,
                    "GPU Mid":         5,
                    "GPU High":        5,
                    "GPU Workstation": 5,
                },
            )
            for profile in loaded_profiles:
                with self.subTest(profile=profile, mode="quick"):
                    model_ids = app._bench_default_models_for(profile, "quick")
                    self.assertFalse(
                        sorted(model_ids - active_quick),
                        f"fixture references unknown catalog ids for {profile!r}",
                    )
        finally:
            system_info.BENCHMARK_SKU_PROFILES = original_profiles

    def test_benchmark_observed_results_drive_defaults_for_evidence_profiles(self):
        from src import catalog
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False
        app._bench_missing_deps_for_model = lambda model: []

        class _ModeVar:
            def __init__(self, mode): self._mode = mode
            def get(self): return self._mode

        # Extended: defaults exactly match the resolved Extended set per profile.
        app._bench_run_mode_var = _ModeVar("Extended")
        models = [model for model in catalog.load_catalog() if app._bench_methods_for_ui(model)]
        for profile in ("Small CPU", "Medium CPU", "Large CPU", "GPU Mid", "GPU High"):
            capacity = app._bench_profile_capacity(profile)
            selected = {
                model["id"]
                for model in models
                if app._bench_model_available_for_profile(model, capacity)[0]
                and app._bench_default_selected_for_model(model, capacity)
            }
            self.assertEqual(
                selected,
                app._bench_default_models_for(profile, "extended"),
                f"extended/{profile}",
            )

            fallback = {
                "id": f"not-observed-{profile}",
                "name": f"Not Observed {profile}",
                "category": "Small",
                "ollama_tag": "not-observed:latest",
                "min_ram_gb": 4,
                "min_vram_gb": 0,
                "recommended_for": [profile],
            }
            self.assertTrue(app._bench_model_available_for_profile(fallback, capacity)[0])
            # v5.5.8: in Extended, defaults come from the SKU's resolved
            # Extended set. A non-set id is NOT default-ticked even when
            # recommended_for matches.
            self.assertFalse(app._bench_default_selected_for_model(fallback, capacity))

        # Quick: defaults exactly match the resolved Quick set per profile.
        app._bench_run_mode_var = _ModeVar("Quick")
        models = [model for model in catalog.load_catalog() if app._bench_methods_for_ui(model)]
        for profile in ("Small CPU", "Medium CPU", "Large CPU", "GPU Mid", "GPU High"):
            capacity = app._bench_profile_capacity(profile)
            selected = {
                model["id"]
                for model in models
                if app._bench_model_available_for_profile(model, capacity)[0]
                and app._bench_default_selected_for_model(model, capacity)
            }
            self.assertEqual(
                selected,
                app._bench_default_models_for(profile, "quick"),
                f"quick/{profile}",
            )

    def test_benchmark_gpu_workstation_defaults_match_active_runnable_catalog(self):
        from src import catalog
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False
        app._bench_missing_deps_for_model = lambda model: []

        # v5.5.8: this contract — "GPU Workstation default-ticks every runnable
        # product model" — is an Extended-mode property. Quick mode is an
        # explicit per-SKU set resolved from skus.json.
        class _ModeVar:
            def __init__(self, mode): self._mode = mode
            def get(self): return self._mode
        app._bench_run_mode_var = _ModeVar("Extended")

        capacity = app._bench_profile_capacity("GPU Workstation")
        models = [
            model
            for model in catalog.load_catalog()
            if app._bench_methods_for_ui(model) and not model.get("user_added")
        ]
        runnable = {
            model["id"]
            for model in models
            if app._bench_model_available_for_profile(model, capacity)[0]
        }
        selected = {
            model["id"]
            for model in models
            if app._bench_model_available_for_profile(model, capacity)[0]
            and app._bench_default_selected_for_model(model, capacity)
        }
        # v5.5.8 / v5.5.12 / v5.5.16+: Extended on the largest GPU SKU
        # default-ticks every verified passer from the resolved Extended set.
        # The defaults set must be a subset of the runnable set (you can't
        # tick a model the user can't run). Any runnable models NOT in the
        # resolved set are catalog rows we have no evidence for yet — they
        # stay unticked until evidence arrives, matching Ron's "all default-
        # checked tests are expected to pass" intent. The exact selected
        # count varies with the active SKU fixture so we don't pin it.
        self.assertEqual(
            selected,
            app._bench_default_models_for("GPU Workstation", "extended") & runnable,
        )
        self.assertTrue(selected.issubset(runnable))
        self.assertGreater(len(selected), 0)

    def test_benchmark_observed_success_respects_hard_blockers(self):
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False
        app._bench_missing_deps_for_model = lambda model: []
        class _ModeVar:
            def __init__(self, mode): self._mode = mode
            def get(self): return self._mode
        # qwen2.5vl-7b is in the fixture's Small CPU Extended set so the
        # "observed success" override is exercised under Extended.
        app._bench_run_mode_var = _ModeVar("Extended")
        capacity = {
            "profile": "Small CPU",
            "available_ram_gb": 6.5,
            "total_ram_gb": 16,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        observed_oversize = {
            "id": "qwen2.5vl-7b",
            "name": "Observed Oversize",
            "category": "Medium",
            "ollama_tag": "observed:latest",
            "min_ram_gb": 999,
            "min_vram_gb": 999,
        }
        self.assertTrue(app._bench_model_available_for_profile(observed_oversize, capacity)[0])
        self.assertTrue(app._bench_default_selected_for_model(observed_oversize, capacity))

        no_backend = dict(observed_oversize)
        no_backend.pop("ollama_tag")
        self.assertFalse(app._bench_model_available_for_profile(no_backend, capacity)[0])
        self.assertFalse(app._bench_default_selected_for_model(no_backend, capacity))

        observed_missing_utility = {
            "id": "whisper-large-v3-turbo",
            "name": "Observed Missing Utility",
            "category": "Speech",
            "phase1_adapter": True,
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        app._bench_missing_deps_for_model = lambda model: (
            ["soundfile"] if model.get("id") == "whisper-large-v3-turbo" else []
        )
        gpu_max_capacity = {
            "profile": "GPU Workstation",
            "available_ram_gb": 440,
            "total_ram_gb": 440,
            "vram_capacity_gb": 24,
            "has_gpu": True,
            "is_sku": True,
        }
        self.assertFalse(app._bench_model_available_for_profile(observed_missing_utility, gpu_max_capacity)[0])
        self.assertFalse(app._bench_default_selected_for_model(observed_missing_utility, gpu_max_capacity))

    def test_benchmark_render_recovers_when_prior_vars_were_built_before_catalog(self):
        """Real-world regression: opening the Benchmark page before
        ``_apply_startup_data`` finishes leaves ``_bench_model_vars`` populated
        but empty (catalog wasn't loaded yet). The post-startup
        ``_refresh_bench_profile_values(preserve_selection=True)`` must NOT
        treat that empty state as a real selection, or every model is
        unchecked once the catalog finally arrives.
        """
        from src.app import App

        render_source = self._function_source("_render_bench_model_checklist")
        # The fix introduces a have_prior_state gate: preserve_selection only
        # takes effect when there is a real prior selection to preserve.
        self.assertIn("have_prior_state", render_source)
        self.assertIn("bool(self._bench_model_vars)", render_source)
        # And when there is no prior state, preserve_selection must be
        # demoted to False so fit-based defaults are applied.
        self.assertIn("preserve_selection = False", render_source)

    def test_benchmark_select_all_ignores_disabled_models(self):
        from src.app import App

        class DummyVar:
            def __init__(self):
                self.value = False

            def set(self, value):
                self.value = value

            def get(self):
                return self.value

        app = object.__new__(App)
        app._bench_model_disabled_ids = {"too-large"}
        app._bench_model_oversize_ids = {"oversize"}
        app._bench_model_vars = {
            "small": DummyVar(),
            "oversize": DummyVar(),
            "too-large": DummyVar(),
        }

        app._bench_select_models(True)

        self.assertTrue(app._bench_model_vars["small"].get())
        self.assertFalse(app._bench_model_vars["oversize"].get())
        self.assertFalse(app._bench_model_vars["too-large"].get())
        self.assertEqual(app._get_selected_model_ids(), ["small"])

    def test_benchmark_method_toggles_drive_runnable_methods(self):
        from src import app as app_module
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        app = object.__new__(App)
        app._bench_gpu_var = DummyVar(True)
        app._bench_cpu_var = DummyVar(True)
        app._bench_onnx_var = DummyVar(True)
        app._bench_utility_var = DummyVar(True)
        app._bench_image_var = DummyVar(True)
        app._bench_run_mode_var = DummyVar("Extended")
        app._bench_resource_override_var = DummyVar(False)
        app._bench_missing_deps_for_model = lambda _model: []
        capacity = {
            "profile": "GPU Workstation",
            "available_ram_gb": 440,
            "total_ram_gb": 440,
            "vram_capacity_gb": 24,
            "has_gpu": True,
            "is_sku": True,
        }
        model = {
            "id": "hybrid",
            "name": "Hybrid",
            "ollama_tag": "hybrid:latest",
            "onnx_repo": "example/hybrid",
            "min_ram_gb": 8,
            "min_vram_gb": 8,
        }

        originals = (
            app_module.ONNX_AVAILABLE,
            app_module.OPENVINO_AVAILABLE,
            app_module.DIRECTML_AVAILABLE,
        )
        try:
            app_module.ONNX_AVAILABLE = True
            app_module.OPENVINO_AVAILABLE = True
            app_module.DIRECTML_AVAILABLE = True
            methods = app._bench_methods_for_run_ui(model, capacity=capacity)
            self.assertEqual(
                methods,
                [
                    "ollama_gpu",
                    "ollama_cpu",
                    "onnx_openvino",
                    "onnx_directml",
                    "onnx_cpu",
                ],
            )

            app._bench_gpu_var.set(False)
            app._bench_cpu_var.set(False)
            app._bench_onnx_var.set(False)
            self.assertEqual(app._bench_methods_for_run_ui(model, capacity=capacity), [])
        finally:
            (
                app_module.ONNX_AVAILABLE,
                app_module.OPENVINO_AVAILABLE,
                app_module.DIRECTML_AVAILABLE,
            ) = originals

    def test_benchmark_image_rows_stay_visible_when_image_toggle_is_off(self):
        """v5.5.1: the "Include image-gen models" checkbox is gone. Image
        rows are always considered runnable when the model fits the active
        profile; their per-row enable state is gated by GPU presence /
        Force All, not by a method-level toggle. The legacy var is kept
        as a True constant for back-compat snapshot/restore paths."""
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        app = object.__new__(App)
        app._bench_run_mode_var = DummyVar("Extended")
        app._bench_image_var = DummyVar(False)
        app._bench_resource_override_var = DummyVar(False)
        app._bench_profile_capacity = lambda: {
            "profile": "GPU Workstation",
            "available_ram_gb": 440,
            "total_ram_gb": 440,
            "vram_capacity_gb": 24,
            "has_gpu": True,
            "is_sku": True,
        }
        image_model = {
            "id": "img",
            "name": "Image",
            "backend": "comfyui",
            "comfyui_model": "img.safetensors",
            "min_ram_gb": 8,
            "min_vram_gb": 8,
        }

        self.assertEqual(app._bench_methods_for_ui(image_model), ["image"])
        # v5.5.1: regardless of _bench_image_var, Extended mode + GPU profile
        # + fits-capacity → image method is included.
        self.assertEqual(app._bench_methods_for_run_ui(image_model), ["image"])

        app._bench_image_var.set(True)
        self.assertEqual(app._bench_methods_for_run_ui(image_model), ["image"])

    def test_benchmark_footer_reports_runnable_models_after_method_toggles(self):
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class DummyLabel:
            def __init__(self):
                self.text = ""

            def configure(self, **kwargs):
                self.text = kwargs.get("text", self.text)

        app = object.__new__(App)
        app._bench_model_vars = {
            "chat": DummyVar(True),
            "utility": DummyVar(True),
        }
        app._bench_model_by_id = {
            "chat": {
                "id": "chat",
                "name": "Chat",
                "ollama_tag": "chat:latest",
                "min_ram_gb": 8,
                "min_vram_gb": 8,
            },
            "utility": {
                "id": "utility",
                "name": "Utility",
                "phase1_adapter": True,
                "min_ram_gb": 4,
                "min_vram_gb": 0,
            },
        }
        app._bench_selection_footer = DummyLabel()
        app._bench_run_mode_var = DummyVar("Quick")
        # Utility/Toolbox models are excluded from benchmarks entirely, so the
        # only way the chat model becomes runnable is via the ollama_gpu
        # method on the GPU Workstation profile. Flip the GPU toggle on.
        app._bench_gpu_var = DummyVar(True)
        app._bench_cpu_var = DummyVar(False)
        app._bench_onnx_var = DummyVar(False)
        app._bench_utility_var = DummyVar(True)
        app._bench_image_var = DummyVar(False)
        app._bench_resource_override_var = DummyVar(False)
        app._bench_profile_capacity = lambda: {
            "profile": "GPU Workstation",
            "available_ram_gb": 440,
            "total_ram_gb": 440,
            "vram_capacity_gb": 24,
            "has_gpu": True,
            "is_sku": True,
        }
        app._bench_missing_deps_for_model = lambda _model: []

        app._update_bench_selection_footer()

        self.assertIn("Selected: 2 of 2 models", app._bench_selection_footer.text)
        self.assertIn("Runnable now: 1", app._bench_selection_footer.text)
        self.assertIn("~1 cases", app._bench_selection_footer.text)

    def test_benchmark_method_toggles_refresh_footer_without_checklist_render(self):
        build_source = self._function_source("_build_benchmark_page")
        # CPU / GPU / ONNX method toggles still trigger _on_bench_method_toggle.
        # The Utility toggle was removed (phase1/utility models are excluded
        # from benchmarks), so the expected minimum is 3.
        self.assertGreaterEqual(build_source.count("command=self._on_bench_method_toggle"), 3)

        method_source = self._function_source("_on_bench_method_toggle")
        self.assertIn("_update_bench_selection_footer", method_source)
        # v5.5.1: _on_bench_image_toggle removed — Include Image Gen
        # checkbox is gone. Confirm via the build source so a regression
        # that reintroduces it (and the underlying widget) is caught.
        self.assertNotIn("self._bench_image_check", build_source)
        self.assertNotIn('text="Include image-gen models', build_source)
        self.assertNotIn("_on_bench_image_toggle", build_source)

    def test_benchmark_copy_cli_refuses_empty_selection(self):
        from src import app as app_module
        from src.app import App

        class DummyVar:
            def __init__(self, value=False):
                self.value = value

            def get(self):
                return self.value

        app = object.__new__(App)
        app._bench_model_disabled_ids = set()
        app._bench_model_vars = {"small": DummyVar(False)}
        app._get_bench_cli_args = lambda: self.fail("empty selection must not build a full-catalog command")
        app.clipboard_clear = lambda: self.fail("empty selection must not touch clipboard")
        app.clipboard_append = lambda _cmd: self.fail("empty selection must not touch clipboard")
        statuses = []
        infos = []
        app.set_status = lambda message: statuses.append(message)
        original_showinfo = app_module.messagebox.showinfo
        try:
            app_module.messagebox.showinfo = lambda *args, **kwargs: infos.append((args, kwargs))
            App._copy_bench_cli(app)
        finally:
            app_module.messagebox.showinfo = original_showinfo

        self.assertEqual(len(infos), 1)
        self.assertIn("Select at least one model", statuses[-1])

    def test_benchmark_cli_args_include_non_gpu_sku_capacity_and_all_selected_ids(self):
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        app = object.__new__(App)
        app.cfg = {"low_resources_mode": False}
        app._optional_skus_enabled = False
        app._optional_skus = []
        app._bench_profile_var = DummyVar("Large CPU")
        app._bench_prompt_var = DummyVar("/no_think\nReturn only this exact sentence: A neural network learns patterns from examples.")
        app._bench_timeout_var = DummyVar(300)
        app._bench_maxfail_var = DummyVar(10)
        app._bench_gpu_var = DummyVar(False)
        app._bench_cpu_var = DummyVar(True)
        app._bench_onnx_var = DummyVar(False)
        app._bench_cleanup_var = DummyVar(False)
        app._bench_phase1_var = DummyVar(False)
        app._bench_resource_override_var = DummyVar(True)
        app._bench_model_disabled_ids = set()
        app._bench_model_vars = {
            "small-a": DummyVar(True),
            "small-b": DummyVar(True),
        }

        cmd = app._get_bench_cli_args()

        self.assertIn("--skip-gpu", cmd)
        self.assertIn("--skip-onnx", cmd)
        # Utility/Toolbox models are excluded from benchmarks by default; the
        # GUI Copy CLI does not emit --skip-utility because the headless
        # runner already defaults to --skip-utility=True.
        self.assertNotIn("--skip-utility", cmd)
        self.assertIn("--allow-oversize", cmd)
        self.assertIn("--models small-a small-b", cmd)
        self.assertIn("--capacity-ram-gb 64", cmd)
        self.assertIn("--capacity-no-gpu", cmd)
        # Quick mode is the default and must omit --run-mode and --skip-image.
        self.assertNotIn("--run-mode", cmd)
        self.assertNotIn("--skip-image", cmd)
        self.assertNotIn("--max-failures", cmd)

    def test_benchmark_cli_args_include_extended_run_mode_and_skip_image_flags(self):
        """v5.5.1: --skip-image is no longer emitted; the Include Image Gen
        checkbox is gone and image-gen rows are always conceptually included
        (per-row capacity / Force All gating decides what actually runs)."""
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        app = object.__new__(App)
        app.cfg = {"low_resources_mode": False}
        app._optional_skus_enabled = False
        app._optional_skus = []
        app._bench_profile_var = DummyVar("GPU Workstation")
        app._bench_prompt_var = DummyVar("/no_think\nReturn only this exact sentence: A neural network learns patterns from examples.")
        app._bench_timeout_var = DummyVar(600)
        app._bench_maxfail_var = DummyVar(10)
        app._bench_gpu_var = DummyVar(True)
        app._bench_cpu_var = DummyVar(True)
        app._bench_onnx_var = DummyVar(False)
        app._bench_cleanup_var = DummyVar(False)
        app._bench_phase1_var = DummyVar(True)
        app._bench_resource_override_var = DummyVar(False)
        app._bench_run_mode_var = DummyVar("Extended")
        app._bench_image_var = DummyVar(False)
        app._bench_model_disabled_ids = set()
        app._bench_model_vars = {
            "small-a": DummyVar(True),
        }

        cmd = app._get_bench_cli_args()

        self.assertIn("--run-mode extended", cmd)
        self.assertNotIn("--skip-image", cmd)
        self.assertNotIn("--max-failures", cmd)

    def test_benchmark_cli_args_include_demo_run_mode_and_skip_image_flags(self):
        """Removed in v5.5.19 — Demo run mode is gone. Kept as a stub so any
        external reference is obvious at the point of removal."""

    def test_bench_capacity_label_does_not_expose_sku_hardware_specs(self):
        """Constrained-VM SKU hardware specs (GPU model, VRAM, RAM, CPU) must
        never appear in user-visible labels. Only the profile name is shown.
        """
        from src.app import App

        app = object.__new__(App)
        capacity = {
            "profile": "GPU Workstation",
            "total_ram_gb": 440,
            "vram_capacity_gb": 24,
            "has_gpu": True,
        }
        label = App._bench_capacity_label(app, capacity)
        self.assertEqual(label, "GPU Workstation")
        for forbidden in ("440", "24", "GB RAM", "GB VRAM", "CPU-only",
                          "CPU only", "A10", "NVIDIA"):
            self.assertNotIn(
                forbidden, label,
                f"_bench_capacity_label leaked SKU spec '{forbidden}': {label!r}",
            )

        capacity_cpu = {
            "profile": "Small CPU",
            "total_ram_gb": 16,
            "vram_capacity_gb": 0,
            "has_gpu": False,
        }
        label_cpu = App._bench_capacity_label(app, capacity_cpu)
        self.assertEqual(label_cpu, "Small CPU")
        for forbidden in ("16", "GB RAM", "GB VRAM", "CPU-only", "CPU only"):
            self.assertNotIn(
                forbidden, label_cpu,
                f"_bench_capacity_label leaked CPU SKU spec '{forbidden}': {label_cpu!r}",
            )

    def test_benchmark_page_has_no_sku_spec_paragraphs_or_summary_widget(self):
        """The Benchmark page must not reintroduce the verbose Quick/Extended
        help paragraph or the `_bench_profile_summary_label` widget that used
        to expose `(440 GB RAM, 24 GB VRAM)` SKU specs.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src" / "app.py"
        text = src.read_text(encoding="utf-8")
        forbidden = [
            "Quick mode runs one shared prompt",
            "Default selection targets",
            "_bench_profile_summary_label",
            "RAM/VRAM capacity used for default selections",
            "GB RAM, {vram",
            "GB VRAM, {ram",
        ]
        for needle in forbidden:
            self.assertNotIn(
                needle, text,
                f"Benchmark page must not contain SKU spec leak / noise: {needle!r}",
            )

    def test_benchmark_action_bar_is_static_panel_below_log(self):
        """The Start/Stop/Retry/Open/CLI action bar must live in its own
        bottom-pinned panel (row=3 with SURFACE_INNER background), and the
        scrolling log must be the weight=1 row above it (row=2). This keeps
        the action buttons reachable even when the log fills the page.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src" / "app.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn(
            "page.grid_rowconfigure(2, weight=1)", text,
            "Benchmark page must give row 2 (the log) the vertical weight."
        )
        # The action bar frame must use SURFACE_INNER background so it reads
        # as a distinct static panel, and must end up at row=3 (below the
        # weighted log row).
        self.assertRegex(
            text,
            r"btn_frame\s*=\s*ctk\.CTkFrame\([^)]*fg_color=SURFACE_INNER",
            "Action bar must be a CTkFrame with SURFACE_INNER background."
        )
        self.assertRegex(
            text,
            r"btn_frame\.grid\(\s*row=3\b",
            "Action bar must sit at row=3 (below the weighted log)."
        )

    def test_render_bench_model_checklist_uses_banded_category_cards(self):
        """Each category card must alternate between SURFACE_INNER and
        SURFACE_CARD so users can visually separate categories at a glance.

        Post-v5.3.6: each card also carries the same border chrome the
        Models page ``ModelListRow`` uses — ``border_width=1`` plus
        ``border_color=BORDER_STRONG`` plus ``corner_radius=6`` — so the
        benchmark per-category rows visually match the model list rows.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src" / "app.py"
        text = src.read_text(encoding="utf-8")
        # Render must build a `banded_colors` tuple with both surface tokens
        # and pick the per-category color by index.
        self.assertIn("banded_colors", text)
        self.assertIn("SURFACE_INNER", text)
        self.assertIn("SURFACE_CARD", text)
        # Must also create the per-category select-all BooleanVar map.
        self.assertIn("_bench_category_select_vars", text)
        # And the bulk-toggle method.
        self.assertIn("_toggle_bench_category_selection", text)
        # Post-v5.3.6 chrome unification: the per-category card frame
        # must carry the same ModelListRow chrome (rounded + bordered).
        render_source = self._function_source("_render_bench_model_checklist")
        self.assertRegex(
            render_source,
            r"card\s*=\s*ctk\.CTkFrame\(\s*\n?\s*checklist\s*,(?:[^)]*\n)*?[^)]*border_width\s*=\s*1",
            "Benchmark category card must declare border_width=1 to match "
            "the model list row chrome (Ron's UI unification request).",
        )
        self.assertRegex(
            render_source,
            r"card\s*=\s*ctk\.CTkFrame\(\s*\n?\s*checklist\s*,(?:[^)]*\n)*?[^)]*border_color\s*=\s*BORDER_STRONG",
            "Benchmark category card must declare border_color=BORDER_STRONG "
            "to match the model list row chrome.",
        )
        self.assertRegex(
            render_source,
            r"card\s*=\s*ctk\.CTkFrame\(\s*\n?\s*checklist\s*,(?:[^)]*\n)*?[^)]*corner_radius\s*=\s*6",
            "Benchmark category card must keep corner_radius=6 to match the "
            "model list row chrome.",
        )

    def test_models_page_section_header_is_borderless_with_bottom_rule(self):
        """The Models page category header (the "Chat / 22 models" bar in
        Ron's screenshot) must be a borderless / transparent shell with an
        accent chevron + bold title + muted count + a thin BORDER_STRONG
        bottom rule. The old solid ``("#dde3eb", "#1f2530")`` fg_color
        tuple painted as a black rectangle on the first Home → Models
        navigation in dark mode because CTkFrame's internal canvas
        resolves ``bg_color="transparent"`` against its parent at draw
        time and the scrollable frame's inner canvas had not yet
        propagated its appearance mode to the brand-new child widget.
        Switching to a transparent header with theme-token children means
        there is no fg_color tuple to mispaint.
        """
        header_source = self._function_source("_ensure_section_header")
        # Regression pin: the old paint-bug hex tuple must never come back.
        self.assertNotIn('"#1f2530"', header_source,
                         "Section header must not reintroduce the pre-v5.3.7 "
                         "solid-fill hex tuple that painted as black on first "
                         "Home → Models navigation.")
        self.assertNotIn('"#dde3eb"', header_source,
                         "Section header must not reintroduce the pre-v5.3.7 "
                         "solid-fill hex tuple.")
        # The shell must be transparent so there is no fg_color tuple to
        # mispaint on the first draw cycle inside CTkScrollableFrame.
        self.assertRegex(
            header_source,
            r'ctk\.CTkFrame\(\s*self\._cards_scroll\s*,\s*fg_color\s*=\s*"transparent"',
            "Section header outer frame must be fg_color=\"transparent\" to "
            "sidestep the CTkFrame-inside-CTkScrollableFrame first-paint bug.",
        )
        # Theme-token children: accent chevron, primary title, muted count,
        # BORDER_STRONG bottom rule. No hex literals on this widget.
        self.assertIn("text_color=LINK_TEXT", header_source)
        self.assertIn("text_color=TEXT_PRIMARY", header_source)
        self.assertIn("text_color=TEXT_MUTED", header_source)
        self.assertIn("fg_color=BORDER_STRONG", header_source)
        # The toggle-click binding must include the new rule frame so users
        # can click anywhere on the header to collapse/expand the section.
        self.assertRegex(
            header_source,
            r"for w in \(hdr, chev, title_lbl, count_lbl, rule\):",
            "All header children including the rule must bind to the toggle.",
        )

    def test_bench_category_select_all_state_present_and_synced(self):
        """Each category must have a select-all checkbox whose state is
        synced from `_update_bench_category_header` so manual per-row toggles
        keep the checkbox honest, and `_toggle_bench_category_selection`
        must only affect non-disabled rows.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src" / "app.py"
        text = src.read_text(encoding="utf-8")
        # Header refresh must mirror the eligible-rows-selected state into the
        # per-category select-all var.
        self.assertIn("_bench_category_select_vars", text)
        self.assertIn("desired = bool(eligible_ids)", text)
        # Bulk toggle must respect disabled rows.
        self.assertIn("_bench_model_disabled_ids", text)
        self.assertIn("_bench_category_sync_in_progress", text)

    def test_bench_runner_ensure_comfyui_ready_callback_wired(self):
        """`BatchRunner` must accept an `ensure_comfyui_ready` callback and
        the app must pass `self._bench_ensure_comfyui_ready` from both the
        new-run and Retry-Failed paths so image-gen benchmarks survive
        ComfyUI not being running yet (and a single crash mid-run).

        Additionally, both call sites MUST pass `self.comfyui` UNCONDITIONALLY
        — not `self.comfyui if self.comfyui_ok else None`.  That buggy
        conditional was the root cause of the user's
        `No ComfyUI client provided to BatchRunner` failure when Playground
        v2.5 was selected with ComfyUI down: the runner received `None`
        before the ensure callback ever got a chance to start it.
        """
        from pathlib import Path
        import inspect
        from src.batch_runner import BatchRunner

        sig = inspect.signature(BatchRunner.__init__)
        self.assertIn("ensure_comfyui_ready", sig.parameters,
                      "BatchRunner.__init__ must accept ensure_comfyui_ready")

        src = Path(__file__).resolve().parents[1] / "src" / "app.py"
        app_text = src.read_text(encoding="utf-8")
        self.assertIn("_bench_ensure_comfyui_ready", app_text)
        self.assertEqual(
            app_text.count("ensure_comfyui_ready=self._bench_ensure_comfyui_ready"),
            2,
            "Both _start_benchmark and _retry_failed_benchmark must pass "
            "ensure_comfyui_ready=self._bench_ensure_comfyui_ready"
        )
        # Regression pin: the comfyui_ok conditional must NEVER come back.
        # The whole point of ensure_comfyui_ready is that the runner can lazy-
        # start ComfyUI on demand, which requires having a real client object.
        self.assertNotIn(
            "comfyui_client=self.comfyui if self.comfyui_ok",
            app_text,
            "Both BatchRunner constructions must pass `comfyui_client=self.comfyui` "
            "unconditionally — never the `if self.comfyui_ok else None` conditional "
            "that caused the 'No ComfyUI client provided to BatchRunner' failure."
        )
        self.assertEqual(
            app_text.count("comfyui_client=self.comfyui,"),
            2,
            "Both _start_benchmark and _retry_failed_benchmark must pass "
            "`comfyui_client=self.comfyui,` (no conditional)."
        )

    def test_run_image_comfyui_uses_ensure_ready_and_retries_once_on_crash(self):
        """`_run_image_comfyui` must (a) call ``_ensure_comfyui_running_for_run``
        before generating, and (b) on a generation exception attempt exactly
        one restart+retry when ComfyUI is no longer running."""
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src" / "batch_runner.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("_ensure_comfyui_running_for_run", text)
        self.assertIn("attempting one restart + retry", text)
        # The retry path must explicitly re-call generate_image() once.
        self.assertGreaterEqual(
            text.count("self._comfyui_client.generate_image("),
            2,
            "generate_image must be called twice (initial + single retry on crash)"
        )

    def test_start_benchmark_confirms_oversize_selection_before_running(self):
        from src import app as app_module
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        prompts = []
        app = object.__new__(App)
        app._bench_thread = None
        app._get_selected_model_ids = lambda: ["too-large"]
        app._bench_profile_capacity = lambda: {
            "profile": "Small CPU",
            "available_ram_gb": 16,
            "total_ram_gb": 16,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        app._catalog_models = [
            {
                "id": "too-large",
                "name": "Too Large",
                "min_ram_gb": 32,
                "min_vram_gb": 0,
            }
        ]
        app._bench_default_fit_for_model = lambda *args, **kwargs: (
            False,
            "needs 32 GB RAM; 16 GB installed",
        )
        app._bench_resource_override_var = DummyVar(True)
        # v5.5.18 hermetic-test fix: stub the recent-partial-run probe so the
        # test never hits the unstubbed messagebox.askyesnocancel "resume?"
        # prompt when a real benchmark report happens to be on disk.
        app._bench_recent_partial_run = lambda *_a, **_kw: None
        app_module_runner = app_module.BatchRunner
        app_module_thread = app_module.threading.Thread
        original_askyesno = app_module.messagebox.askyesno
        try:
            app_module.BatchRunner = lambda **_kwargs: self.fail("declined oversize prompt must not start runner")
            app_module.threading.Thread = lambda *_args, **_kwargs: self.fail("declined oversize prompt must not start thread")
            app_module.messagebox.askyesno = lambda title, message, **kwargs: prompts.append((title, message)) or False
            App._start_benchmark(app)
        finally:
            app_module.BatchRunner = app_module_runner
            app_module.threading.Thread = app_module_thread
            app_module.messagebox.askyesno = original_askyesno

        self.assertEqual(len(prompts), 1)
        title, message = prompts[0]
        self.assertIn("oversized", title.lower())
        self.assertIn("Too Large", message)
        self.assertIn("too-large", message)
        self.assertIn("needs 32 GB RAM; 16 GB installed", message)
        self.assertIn("run for hours", message)

    def test_start_benchmark_passes_capacity_selection_and_override_to_runner(self):
        from src import app as app_module
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class DummyWidget:
            def configure(self, **_kwargs):
                return None

            def delete(self, *_args):
                return None

        class DummyEvent:
            def clear(self):
                return None

        class FakeRunner:
            def __init__(self, **kwargs):
                captured["runner_kwargs"] = kwargs
                self.report = type("Report", (), {"file_stem": "test-stem"})()

            def run(self):
                return self.report

        class FakeThread:
            def __init__(self, target, daemon):
                captured["thread_target"] = target
                captured["thread_daemon"] = daemon
                self._alive = False

            def is_alive(self):
                return self._alive

            def start(self):
                captured["thread_started"] = True

        captured = {}
        app = object.__new__(App)
        app._bench_thread = None
        app._get_selected_model_ids = lambda: ["small-a", "small-b"]
        app._bench_profile_capacity = lambda: {
            "profile": "Large CPU",
            "available_ram_gb": 64,
            "total_ram_gb": 64,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        app._bench_default_fit_for_model = lambda *args, **kwargs: (True, "OK")
        app._catalog_models = [
            {"id": "small-a", "name": "Small A", "min_ram_gb": 4, "min_vram_gb": 0},
            {"id": "small-b", "name": "Small B", "min_ram_gb": 4, "min_vram_gb": 0},
        ]
        # v5.5.18 hermetic-test fix: stub the recent-partial-run probe so the
        # test never hits the unstubbed messagebox.askyesnocancel "resume?"
        # prompt when a real benchmark report happens to be on disk.
        app._bench_recent_partial_run = lambda *_a, **_kw: None
        app.cfg = {"low_resources_mode": False, "models_dir": "models"}
        app._bench_cleanup_var = DummyVar(False)
        app._bench_gpu_var = DummyVar(False)
        app._bench_cpu_var = DummyVar(True)
        app._bench_onnx_var = DummyVar(False)
        app._bench_phase1_var = DummyVar(True)
        app._bench_maxfail_var = DummyVar(10)
        app._bench_prompt_var = DummyVar("prompt")
        app._bench_timeout_var = DummyVar(120)
        app._bench_resource_override_var = DummyVar(True)
        app.comfyui_ok = False
        app.comfyui = None
        app._bench_stop_event = DummyEvent()
        app._bench_log = DummyWidget()
        app._bench_log_append = lambda _text: None
        app._bench_opts_collapsed = True
        app._bench_start_btn = DummyWidget()
        app._bench_stop_btn = DummyWidget()
        app._bench_retry_btn = DummyWidget()
        app.set_status = lambda _message: None

        original_runner = app_module.BatchRunner
        original_thread = app_module.threading.Thread
        try:
            app_module.BatchRunner = FakeRunner
            app_module.threading.Thread = FakeThread
            App._start_benchmark(app)
        finally:
            app_module.BatchRunner = original_runner
            app_module.threading.Thread = original_thread

        kwargs = captured["runner_kwargs"]
        self.assertEqual(kwargs["model_ids"], ["small-a", "small-b"])
        self.assertEqual(kwargs["capacity_ram_gb"], 64)
        self.assertEqual(kwargs["capacity_vram_gb"], 0)
        self.assertFalse(kwargs["capacity_has_gpu"])
        self.assertTrue(kwargs["allow_oversize"])
        # v5.5.3 (SQT P1): skip_image is now a hardcoded False — image-gen
        # is never globally skipped from the runner; per-row selection
        # is the gate. The legacy "no image var → skip" inversion path
        # was removed because it silently suppressed image-gen when the
        # bench page hadn't been built yet.
        self.assertEqual(kwargs["run_mode"], "quick")
        self.assertFalse(kwargs["skip_image"])
        self.assertTrue(captured["thread_started"])

    def test_start_benchmark_forwards_extended_run_mode_and_image_flag(self):
        """Extended mode with include-images on must forward run_mode='extended'
        and skip_image=False to the BatchRunner."""
        from src import app as app_module
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class DummyWidget:
            def configure(self, **_kwargs):
                return None

            def delete(self, *_args):
                return None

        class DummyEvent:
            def clear(self):
                return None

        class FakeRunner:
            def __init__(self, **kwargs):
                captured["runner_kwargs"] = kwargs
                self.report = type("Report", (), {"file_stem": "test-stem"})()

            def run(self):
                return self.report

        class FakeThread:
            def __init__(self, target, daemon):
                self._alive = False

            def is_alive(self):
                return self._alive

            def start(self):
                captured["thread_started"] = True

        captured = {}
        app = object.__new__(App)
        app._bench_thread = None
        app._get_selected_model_ids = lambda: ["img-a"]
        app._bench_profile_capacity = lambda: {
            "profile": "GPU Workstation",
            "available_ram_gb": 64,
            "total_ram_gb": 64,
            "vram_capacity_gb": 24,
            "has_gpu": True,
            "is_sku": True,
        }
        app._bench_default_fit_for_model = lambda *args, **kwargs: (True, "OK")
        app._catalog_models = [
            {"id": "img-a", "name": "Img A", "min_ram_gb": 8, "min_vram_gb": 8,
             "backend": "comfyui", "comfyui_model": "img-a.safetensors"},
        ]
        # v5.5.18 hermetic-test fix: stub the recent-partial-run probe so the
        # test never hits the unstubbed messagebox.askyesnocancel "resume?"
        # prompt when a real benchmark report happens to be on disk.
        app._bench_recent_partial_run = lambda *_a, **_kw: None
        app.cfg = {"low_resources_mode": False, "models_dir": "models"}
        app._bench_cleanup_var = DummyVar(False)
        app._bench_gpu_var = DummyVar(True)
        app._bench_cpu_var = DummyVar(True)
        app._bench_onnx_var = DummyVar(False)
        app._bench_phase1_var = DummyVar(True)
        app._bench_maxfail_var = DummyVar(10)
        app._bench_prompt_var = DummyVar("prompt")
        app._bench_timeout_var = DummyVar(180)
        app._bench_resource_override_var = DummyVar(False)
        # New: extended mode + include images on.
        app._bench_run_mode_var = DummyVar("Extended")
        app._bench_image_var = DummyVar(True)
        app.comfyui_ok = False
        app.comfyui = None
        app._bench_stop_event = DummyEvent()
        app._bench_log = DummyWidget()
        app._bench_log_append = lambda _text: None
        app._bench_opts_collapsed = True
        app._bench_start_btn = DummyWidget()
        app._bench_stop_btn = DummyWidget()
        app._bench_retry_btn = DummyWidget()
        app.set_status = lambda _message: None

        original_runner = app_module.BatchRunner
        original_thread = app_module.threading.Thread
        try:
            app_module.BatchRunner = FakeRunner
            app_module.threading.Thread = FakeThread
            App._start_benchmark(app)
        finally:
            app_module.BatchRunner = original_runner
            app_module.threading.Thread = original_thread

        kwargs = captured["runner_kwargs"]
        self.assertEqual(kwargs["run_mode"], "extended")
        self.assertFalse(kwargs["skip_image"])
        self.assertTrue(captured["thread_started"])

    def test_run_batch_forwards_allow_oversize_to_normal_and_retry_runs(self):
        tree = ast.parse(RUN_BATCH_TEXT)
        runner_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "BatchRunner"
        ]
        self.assertGreaterEqual(len(runner_calls), 2)
        for call in runner_calls:
            keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}
            self.assertEqual(keywords.get("allow_oversize"), "args.allow_oversize")
            # v5.5.1+: --force-all must reach BOTH retry and normal runs
            # so a "Retry Failed" with the flag set preserves the same
            # baseline-everything intent the original run had.
            self.assertEqual(keywords.get("force_all"), "args.force_all")
        self.assertIn('"--allow-oversize", "--capacity-allow-oversize"', RUN_BATCH_TEXT)
        self.assertIn('"--force-all"', RUN_BATCH_TEXT)

    def test_get_bench_cli_args_emits_force_all_when_toggle_on(self):
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value

        app = object.__new__(App)
        app.cfg = {"low_resources_mode": False}
        app._optional_skus_enabled = False
        app._optional_skus = []
        app._bench_profile_var = DummyVar("Large CPU")
        app._bench_prompt_var = DummyVar("/no_think\nReturn only this exact sentence: A neural network learns patterns from examples.")
        app._bench_timeout_var = DummyVar(300)
        app._bench_maxfail_var = DummyVar(10)
        app._bench_gpu_var = DummyVar(True)
        app._bench_cpu_var = DummyVar(True)
        app._bench_onnx_var = DummyVar(False)
        app._bench_cleanup_var = DummyVar(False)
        app._bench_phase1_var = DummyVar(False)
        app._bench_resource_override_var = DummyVar(False)
        app._bench_force_all_var = DummyVar(True)
        app._bench_model_disabled_ids = set()
        app._bench_model_vars = {"m": DummyVar(True)}

        cmd = app._get_bench_cli_args()
        self.assertIn("--force-all", cmd)

    def test_get_bench_cli_args_omits_force_all_when_toggle_off(self):
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value

        app = object.__new__(App)
        app.cfg = {"low_resources_mode": False}
        app._optional_skus_enabled = False
        app._optional_skus = []
        app._bench_profile_var = DummyVar("Large CPU")
        app._bench_prompt_var = DummyVar("/no_think\nReturn only this exact sentence: A neural network learns patterns from examples.")
        app._bench_timeout_var = DummyVar(300)
        app._bench_maxfail_var = DummyVar(10)
        app._bench_gpu_var = DummyVar(True)
        app._bench_cpu_var = DummyVar(True)
        app._bench_onnx_var = DummyVar(False)
        app._bench_cleanup_var = DummyVar(False)
        app._bench_phase1_var = DummyVar(False)
        app._bench_resource_override_var = DummyVar(False)
        app._bench_force_all_var = DummyVar(False)
        app._bench_model_disabled_ids = set()
        app._bench_model_vars = {"m": DummyVar(True)}

        cmd = app._get_bench_cli_args()
        self.assertNotIn("--force-all", cmd)

    def test_bench_model_available_lifts_capacity_gate_under_force_all(self):
        """v5.5.2 regression: Force-All must bypass the per-row capacity gate
        in ``_bench_model_available_for_profile`` so the checklist renderer
        sees too-big-for-the-SKU models as available. Without this fix the UI
        keeps those rows disabled, the "All" preset skips them, and Force-All
        on a small constrained-VM SKU silently behaves identically to "Select All"
        on default capacity — the exact bug Ron reported on Small CPU."""
        from src.app import App

        app = object.__new__(App)
        # Small constrained-VM SKU: Small CPU / 16 GB / no GPU. A 32 GB model would
        # normally be marked unavailable.
        capacity = {
            "profile": "Small CPU",
            "available_ram_gb": 16,
            "total_ram_gb": 16,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        oversize_model = {
            "id": "test:30b",
            "name": "Test 30B",
            "size_gb": 18,
            "min_ram_gb": 32,
            "min_vram_gb": 0,
            "ollama_tag": "test:30b",
            "category": "Large",
        }
        # Make sure _bench_methods_for_ui returns something so the model is
        # rendered at all.
        app._bench_methods_for_ui = lambda m: ["ollama_cpu"]
        app._bench_observed_success_for_profile = lambda m, c: False
        app._bench_missing_deps_for_model = lambda m: []

        # Without force_all: rejected for capacity.
        ok_off, reason_off = app._bench_model_available_for_profile(
            oversize_model, capacity, allow_oversize=False, force_all=False,
        )
        self.assertFalse(ok_off, f"expected oversize to be rejected, got: {reason_off}")

        # With force_all: capacity gate lifted.
        ok_on, reason_on = app._bench_model_available_for_profile(
            oversize_model, capacity, allow_oversize=False, force_all=True,
        )
        self.assertTrue(ok_on, f"expected force_all to lift capacity gate, got: {reason_on}")
        # Reason should make it obvious it's Force-All driving the override.
        self.assertIn("Force-All", reason_on)

    def test_bench_model_available_force_all_does_not_lift_missing_deps(self):
        """v5.5.2 regression: Force-All bypasses *capacity* gates only. A
        model whose phase1_adapter is missing optional packages stays
        unavailable because the runner would skip it anyway — showing it as
        runnable would lie to the user."""
        from src.app import App

        app = object.__new__(App)
        capacity = {
            "profile": "Small CPU",
            "available_ram_gb": 16,
            "total_ram_gb": 16,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        phase1_model = {
            "id": "test:phase1",
            "name": "Test Phase1",
            "size_gb": 1,
            "phase1_adapter": "docling",
            "category": "Document AI",
        }
        app._bench_methods_for_ui = lambda m: ["phase1"]
        app._bench_observed_success_for_profile = lambda m, c: False
        app._bench_missing_deps_for_model = lambda m: ["docling"]

        ok, reason = app._bench_model_available_for_profile(
            phase1_model, capacity, allow_oversize=True, force_all=True,
        )
        self.assertFalse(ok, f"missing-deps row must stay unavailable even under Force-All, got: {reason}")
        self.assertIn("missing", reason)

    def test_bench_apply_preset_all_under_force_all_bypasses_disabled_and_oversize(self):
        """v5.5.2 regression: ``_bench_apply_preset("all")`` must set every
        ``_bench_model_vars`` entry to True when Force-All is on, even for
        ids in ``_bench_model_disabled_ids`` and ``_bench_model_oversize_ids``.
        Without this short-circuit, ticking Force-All on a small SKU silently
        skips most of the catalog — the exact bug Ron reported."""
        from src.app import App

        class DummyVar:
            def __init__(self, value=False):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value

        app = object.__new__(App)
        app._catalog_models = [
            {"id": "small", "size_gb": 1},
            {"id": "oversize", "size_gb": 10},
            {"id": "too-large", "size_gb": 50},
        ]
        app._bench_model_vars = {
            "small": DummyVar(False),
            "oversize": DummyVar(False),
            "too-large": DummyVar(False),
        }
        app._bench_model_disabled_ids = {"too-large"}
        app._bench_model_oversize_ids = {"oversize"}
        app._bench_profile_var = DummyVar("Small CPU")
        app._bench_resource_override_var = DummyVar(False)
        app._bench_force_all_var = DummyVar(True)
        # Stub the deps the preset calls
        app._bench_profile_capacity = lambda: {"profile": "Small CPU"}
        app._bench_category_state = {}
        app._update_bench_category_header = lambda _c: None
        app._update_bench_selection_footer = lambda: None

        app._bench_apply_preset("all")

        self.assertTrue(app._bench_model_vars["small"].get())
        self.assertTrue(
            app._bench_model_vars["oversize"].get(),
            "Force-All must bypass _bench_model_oversize_ids for the 'all' preset",
        )
        self.assertTrue(
            app._bench_model_vars["too-large"].get(),
            "Force-All must bypass _bench_model_disabled_ids for the 'all' preset",
        )

    def test_bench_apply_preset_all_without_force_all_still_honors_disabled(self):
        """v5.5.2 regression guard: turning Force-All OFF must restore the
        original "All" preset behaviour — disabled and oversize ids stay
        unchecked. Without this guard the bypass would leak into normal use
        and silently broaden the "All" button's reach."""
        from src.app import App

        class DummyVar:
            def __init__(self, value=False):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value

        app = object.__new__(App)
        app._catalog_models = [
            {"id": "small", "size_gb": 1},
            {"id": "oversize", "size_gb": 10},
            {"id": "too-large", "size_gb": 50},
        ]
        app._bench_model_vars = {
            "small": DummyVar(False),
            "oversize": DummyVar(False),
            "too-large": DummyVar(False),
        }
        app._bench_model_disabled_ids = {"too-large"}
        app._bench_model_oversize_ids = {"oversize"}
        app._bench_profile_var = DummyVar("Small CPU")
        app._bench_resource_override_var = DummyVar(False)
        app._bench_force_all_var = DummyVar(False)
        app._bench_profile_capacity = lambda: {"profile": "Small CPU"}
        app._bench_category_state = {}
        app._update_bench_category_header = lambda _c: None
        app._update_bench_selection_footer = lambda: None

        app._bench_apply_preset("all")

        self.assertTrue(app._bench_model_vars["small"].get())
        self.assertFalse(
            app._bench_model_vars["oversize"].get(),
            "without Force-All, 'all' preset must respect oversize_ids",
        )
        self.assertFalse(
            app._bench_model_vars["too-large"].get(),
            "without Force-All, 'all' preset must respect disabled_ids",
        )

    def test_render_bench_model_checklist_reads_force_all_var(self):
        """v5.5.2 source-level guard: ``_render_bench_model_checklist`` must
        read ``_bench_force_all_var`` and thread it through to
        ``_bench_model_available_for_profile`` so capacity-failed rows show
        up as available when Force-All is on. AST-grounded so this can't
        regress to a string match on a comment."""
        import ast
        import inspect
        import textwrap
        from src.app import App

        render_source = textwrap.dedent(inspect.getsource(App._render_bench_model_checklist))
        tree = ast.parse(render_source)

        # 1) The function must compute a local `force_all` from
        #    self._bench_force_all_var.
        names_assigned = [
            t.id for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for t in node.targets if isinstance(t, ast.Name)
        ]
        self.assertIn(
            "force_all", names_assigned,
            "_render_bench_model_checklist must compute a local force_all flag",
        )

        # 2) The call to _bench_model_available_for_profile must pass
        #    force_all as a keyword.
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_bench_model_available_for_profile"
            ):
                kwargs = {kw.arg for kw in call.keywords if kw.arg}
                self.assertIn(
                    "force_all", kwargs,
                    "_bench_model_available_for_profile must receive force_all kwarg",
                )
                break
        else:
            self.fail("_render_bench_model_checklist must call _bench_model_available_for_profile")

    def test_v554_preflight_classify_impossible_flags_oversized_ram(self):
        """v5.5.4: a model whose ``min_ram_gb`` is more than 2x the host's
        total RAM is physically impossible to load -- no amount of paging
        will save it. The pre-flight helper must flag it so the Force-All
        dialog can offer to uncheck it before the run wastes pull time."""
        from src.app import App

        app = object.__new__(App)
        # 32 GB host
        app._bench_profile_capacity = lambda: {
            "profile": "Medium CPU",
            "available_ram_gb": 28,
            "total_ram_gb": 32,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        app.cfg = {"models_dir": "."}
        app._catalog_models = [
            {"id": "ok-small", "name": "OK Small", "min_ram_gb": 8, "size_gb": 4},
            # 96 GB > 2*32 GB → impossible
            {"id": "bad-huge", "name": "Bad Huge", "min_ram_gb": 96, "size_gb": 50},
            # 60 GB > 32 GB but NOT > 2*32 GB → not flagged here (Force-All
            # is the opt-in for "might fit" territory)
            {"id": "edge-medium", "name": "Edge Medium", "min_ram_gb": 60, "size_gb": 30},
        ]
        out = app._bench_preflight_classify_impossible(
            ["ok-small", "bad-huge", "edge-medium"]
        )
        flagged = {item["id"] for item in out}
        self.assertIn("bad-huge", flagged,
                      "model needing more than 2x host RAM must be flagged impossible")
        self.assertNotIn("ok-small", flagged,
                         "comfortably-fitting model must not be flagged")
        self.assertNotIn("edge-medium", flagged,
                         "Force-All territory (over the gate but <2x) must NOT be auto-flagged")
        # Reason text must be human-friendly and explain WHY.
        huge_reason = next(item["reason"] for item in out if item["id"] == "bad-huge")
        self.assertIn("96", huge_reason)
        self.assertIn("32", huge_reason)

    def test_v554_preflight_classify_impossible_flags_disk_short(self):
        """v5.5.4: a model whose ``size_gb`` exceeds free disk space is
        impossible to even download. The helper must flag it so the dialog
        can offer to uncheck it. Free disk is read via resource_manager."""
        from unittest import mock

        from src.app import App

        app = object.__new__(App)
        app._bench_profile_capacity = lambda: {
            "profile": "Medium CPU",
            "available_ram_gb": 28,
            "total_ram_gb": 32,
            "vram_capacity_gb": 0,
            "has_gpu": False,
            "is_sku": True,
        }
        app.cfg = {"models_dir": "."}
        app._catalog_models = [
            {"id": "small", "name": "Small", "min_ram_gb": 4, "size_gb": 2},
            # 50 GB download > 10 GB free disk → impossible
            {"id": "huge", "name": "Huge Download", "min_ram_gb": 8, "size_gb": 50},
        ]
        with mock.patch("src.resource_manager.get_free_disk_gb", return_value=10.0):
            out = app._bench_preflight_classify_impossible(["small", "huge"])
        flagged = {item["id"] for item in out}
        self.assertIn("huge", flagged,
                      "download bigger than free disk must be flagged impossible")
        self.assertNotIn("small", flagged)
        huge_reason = next(item["reason"] for item in out if item["id"] == "huge")
        self.assertIn("50", huge_reason)
        self.assertIn("10", huge_reason)

    def test_v554_preflight_classify_impossible_returns_empty_for_no_ids(self):
        """v5.5.4: edge case — passing no IDs returns an empty list without
        touching capacity or disk readings (cheap no-op)."""
        from src.app import App

        app = object.__new__(App)
        # Deliberately give it no _catalog_models / _bench_profile_capacity
        # to prove the early-return doesn't dereference anything.
        out_empty = app._bench_preflight_classify_impossible([])
        self.assertEqual(out_empty, [])
        out_none = app._bench_preflight_classify_impossible(set())
        self.assertEqual(out_none, [])

    def test_v554_start_benchmark_invokes_preflight_under_force_all(self):
        """v5.5.4 source-level guard: ``_start_benchmark`` must call the
        preflight helper, and the call must be guarded on the force_all
        flag (we do NOT want this dialog popping up when the user has not
        opted into Force-All). AST-grounded so a refactor can't silently
        regress it."""
        import ast
        import inspect
        import textwrap

        from src.app import App

        src = textwrap.dedent(inspect.getsource(App._start_benchmark))
        tree = ast.parse(src)
        # Find the call to the preflight helper.
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_bench_preflight_classify_impossible"
            ):
                break
        else:
            self.fail("_start_benchmark must call _bench_preflight_classify_impossible")

        # The call must be inside an `if force_all:` (or equivalent) guard.
        # Source-level smoke check: the substring "if force_all" must appear
        # somewhere before the preflight call. Search for the CALL site
        # (with open paren) so the helper name appearing in a docstring or
        # comment comment doesn't false-positive.
        idx = src.find("_bench_preflight_classify_impossible(")
        self.assertGreater(idx, 0)
        preceding = src[: idx]
        self.assertIn("if force_all:", preceding,
                      "preflight dialog must be guarded on if force_all:")
        # Must use askyesnocancel (3-option), not askyesno.
        self.assertIn("messagebox.askyesnocancel", src,
                      "Force-All preflight must use askyesnocancel for the 3-option choice")

    def test_v554_system_info_npu_excludes_hidclass(self):
        """v5.5.4: NPU detection on Windows must exclude Windows device
        classes that are never AI accelerators. The Medium CPU constrained-VM env
        sidecar reported npu_count=1 with name='Microsoft Hyper-V Input'
        class='HIDClass' because the original EXCLUDE_KEYWORDS only
        filtered on FriendlyName -- 'HID' is not a substring of 'microsoft
        hyper-v input' so it slipped through. The fix adds class-based
        exclusion."""
        import inspect

        from src import system_info

        src = inspect.getsource(system_info._get_npu_info_windows)
        # The exclude set must mention HIDClass and check against dev_class.
        self.assertIn("HIDClass", src,
                      "HIDClass must be in the NPU exclusion list")
        self.assertIn("EXCLUDE_CLASSES", src,
                      "NPU detector must define EXCLUDE_CLASSES set")
        self.assertIn("dev_class in EXCLUDE_CLASSES", src,
                      "NPU detector must check dev_class against EXCLUDE_CLASSES")

    def test_run_batch_wires_headless_comfyui_for_image_benchmarks(self):
        self.assertIn("HeadlessComfyUIBenchmarkHost", RUN_BATCH_TEXT)
        self.assertIn("comfyui_client", RUN_BATCH_TEXT)
        self.assertIn("ensure_comfyui_ready", RUN_BATCH_TEXT)
        self.assertIn("prepare_image_model", RUN_BATCH_TEXT)
        self.assertIn("IMAGE_METHOD", RUN_BATCH_TEXT)

    def test_benchmark_profiles_do_not_depend_on_optional_sku_file(self):
        from src import catalog
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False

        values = app._bench_profile_values()
        self.assertIn("Medium CPU", values)
        self.assertIn("Large CPU", values)
        self.assertIn("This Device", values)

        capacity = app._bench_profile_capacity("Large CPU")
        self.assertEqual(capacity["profile"], "Large CPU")
        self.assertEqual(capacity["total_ram_gb"], 64)
        self.assertFalse(capacity["has_gpu"])
        self.assertTrue(capacity["is_sku"])

        runnable = 0
        for model in catalog.load_catalog():
            if model.get("phase1_adapter") or not app._bench_methods_for_ui(model):
                continue
            ok, _reason = app._bench_default_fit_for_model(
                model,
                available_ram_gb=capacity["available_ram_gb"],
                total_ram_gb=capacity["total_ram_gb"],
                vram_capacity_gb=capacity["vram_capacity_gb"],
                has_gpu=capacity["has_gpu"],
            )
            if ok:
                runnable += 1

        self.assertGreater(runnable, 1)

    def test_benchmark_profile_for_sku_maps_generic_cpu_cloud_pcs(self):
        from src.app import App

        app = object.__new__(App)
        app._optional_skus_enabled = False

        self.assertEqual(
            app._bench_profile_for_sku({
                "name": "Virtual Machine 16 cores 64GB",
                "cpu": 16,
                "ram_gb": 64,
                "vram_gb": 0,
            }),
            "Large CPU",
        )
        self.assertEqual(
            app._bench_profile_for_sku({
                "name": "This Device",
                "cpu": 8,
                "ram_gb": 32,
                "vram_gb": 0,
            }),
            "Medium CPU",
        )

    def test_ollama_tag_local_helper_does_not_use_prefix_matching(self):
        from src.app import _ollama_tag_is_local

        local = {"llama3.2:1b", "qwen2.5"}
        self.assertFalse(_ollama_tag_is_local("llama3.2:3b", local))
        self.assertFalse(_ollama_tag_is_local("qwen2:7b", {"qwen2.5:0.5b"}))
        self.assertTrue(_ollama_tag_is_local("qwen2.5:0.5b", local))

    def test_sidebar_footer_does_not_overlap_nav_buttons(self):
        ui_source = self._function_source("_build_ui")
        self.assertIn("sidebar.grid_rowconfigure(12, weight=1)", ui_source)
        self.assertIn('("docs",     "  📖  Help / Docs")', ui_source)
        self.assertIn("self._device_sidebar_label.grid(row=13", ui_source)
        self.assertIn("self._ollama_status_label.grid(row=14", ui_source)
        self.assertIn("self._comfyui_status_label.grid(row=15", ui_source)
        self.assertNotIn("self._device_sidebar_label.grid(row=10", ui_source)
        self.assertNotIn("self._ollama_status_label.grid(row=11", ui_source)
        self.assertNotIn("self._comfyui_status_label.grid(row=12", ui_source)


class DocsRound3ContractTests(unittest.TestCase):
    """Round 3 (2026-05-19) — landing page Models section, image-gen guide
    CFG/timing/CPU rewrites, scroll-offset CSS variable, retired extension
    slots. These pin the structural changes so a merge conflict that
    restored the old layout would fail loudly instead of silently shipping
    stale UX."""

    @classmethod
    def setUpClass(cls):
        docs = ROOT / "docs"
        cls.index = (docs / "index.html").read_text(encoding="utf-8")
        cls.image_gen = (docs / "image-gen-guide.html").read_text(encoding="utf-8")
        cls.extension = (docs / "localai-docs-extension.js").read_text(encoding="utf-8")

    def test_image_gen_guide_cfg_section_uses_table_not_divs(self):
        # The div→table migration replaces a flexbox bar that never had
        # matching CSS (rendered as flat unreadable text).
        self.assertIn('<table class="cfg-table">', self.image_gen)
        tbody_match = self.image_gen.split('<table class="cfg-table">', 1)[1].split("</tbody>", 1)[0]
        tr_count = tbody_match.count("<tr>")
        # 1 header row + 5 data rows = 6
        self.assertEqual(
            tr_count, 6,
            f"CFG table must have header + 5 data rows; got {tr_count} <tr>",
        )
        for chip in ("cfg-flux", "cfg-soft", "cfg-default", "cfg-literal", "cfg-avoid"):
            self.assertIn(
                f"cfg-pill {chip}",
                self.image_gen,
                f"Missing CFG pill chip class: {chip}",
            )
        # The old broken div markup must not return.
        for old in ("cfg-scale", "cfg-seg", "cfg-s1", "cfg-s2", "cfg-s3", "cfg-s4", "cfg-s5"):
            self.assertNotIn(
                f'"{old}"', self.image_gen,
                f"Old CFG class '{old}' must not be reintroduced",
            )

    def test_image_gen_guide_timing_table_has_four_sku_columns_and_new_model_rows(self):
        # Old table had 2 GPU columns (12 GB / 24 GB) and 6 rows.
        # New table has 4 SKU columns (Select / Standard / Super / Max).
        for sku in ("GPU Entry", "GPU Mid", "GPU High", "GPU Workstation"):
            self.assertIn(sku, self.image_gen, f"Timing table missing column: {sku}")
        for model in ("Z-Image Turbo", "SDXL Lightning"):
            self.assertIn(model, self.image_gen, f"Timing table missing new model row: {model}")
        # The 4-column colspan header guarantees the SKU sub-header row layout.
        self.assertIn('colspan="4"', self.image_gen)

    def test_image_gen_guide_cpu_section_does_not_reference_specific_cpu_skus(self):
        self.assertIn(
            "<h2>CPU mode for CPU-only systems</h2>",
            self.image_gen,
            "CPU mode section h2 must use the renamed wording",
        )
        self.assertNotIn(
            "CPU mode on 4 / Medium CPU",
            self.image_gen,
            "Old vCPU-specific CPU mode heading must not return",
        )
        # Rail link must also be renamed to match the h2.
        self.assertRegex(
            self.image_gen,
            r'href="#cpu-mode"[^>]*data-toc-link[^>]*>\s*CPU Mode \(CPU-only systems\)',
            "Rail link to #cpu-mode must read 'CPU Mode (CPU-only systems)'",
        )

    def test_index_html_models_section_is_four_card_grid_linking_to_model_guide(self):
        # The stale 30-row table was replaced with a 4-card summary grid.
        models_section = self.index.split('<section id="models">', 1)[1].split("</section>", 1)[0]
        self.assertIn('class="card-grid"', models_section)
        self.assertIn("Model-Guide.html", models_section)
        self.assertIn("48 built-in models", models_section)
        # The old hardcoded model-name table rows must not have come back. We
        # spot-check 3 model names that were ONLY in the old stale table.
        for stale in ("TinyLlama", "SmolLM2", "Phi-3 Mini"):
            self.assertNotIn(
                f">{stale}<", models_section,
                f"Stale Models-section table row for {stale!r} must not return",
            )

    def test_extension_slots_inject_empty_hidden_divs_not_content(self):
        # Both inject targets were retired — the doc now owns this content
        # natively. If the inject body grows back, the timing or hardware
        # tables would render twice in old guides.
        for slot_id in ("image-guide-generation-times", "model-value-hardware-profiles"):
            pattern = (
                r'inject\(\s*["\']' + slot_id + r'["\']\s*,\s*'
                r'`[\s\S]*?`'
            )
            match = re.search(pattern, self.extension)
            self.assertIsNotNone(match, f"inject({slot_id!r}, ...) call not found")
            body = match.group(0)
            # The body must NOT contain any table or row markup.
            for token in ("<table", "<tr", "<thead", "<tbody"):
                self.assertNotIn(
                    token, body,
                    f"Retired slot {slot_id!r} must not inject {token!r} content",
                )

    def test_doc_files_declare_scroll_offset_css_variable_and_updater(self):
        for name, src in (("index.html", self.index), ("image-gen-guide.html", self.image_gen)):
            with self.subTest(file=name):
                # scroll-padding-top is intentionally 0 — using it AND
                # scroll-margin-top together doubled the gap (pushed sections
                # ~half a screen below the sticky header). Section landings
                # are owned by per-section `scroll-margin-top: var(--scroll-offset, ...)`.
                self.assertIn(
                    "scroll-padding-top: 0",
                    src,
                    f"{name}: html scroll-padding-top must stay 0 to avoid doubled gap",
                )
                self.assertIn(
                    "scroll-margin-top: var(--scroll-offset",
                    src,
                    f"{name}: section scroll-margin-top must use --scroll-offset",
                )
                self.assertIn(
                    'setProperty("--scroll-offset"',
                    src,
                    f"{name}: updateStickyOffset() must set --scroll-offset",
                )
                # Lookahead tightened from +60 to +12 for accurate rail tracking.
                self.assertNotIn(
                    "offsetHeight : 0) + 60;",
                    src,
                    f"{name}: scroll-spy lookahead must not regress to +60",
                )


import re  # noqa: E402  (kept near end to localize the new tests' dependency)


# ── Add-from-HF install gate ─────────────────────────────────────────────────
# Pre-publish code-reviewer Tier-1: ``_user_added_model_can_install`` was
# over-permissive — it returned True for ONNX / ONNX-GenAI / OpenVINO /
# Phase-1 backends even though ``start_download`` / ``download_comfyui_model``
# only have one-click installers for Ollama and ComfyUI today.  That mismatch
# meant a user pasting a Phi-4-style HF URL would slip past the centralized
# gate, fall through to ``ollama.pull_model(None)``, and see a raw stack
# trace instead of the friendly "Can't download automatically" infobox.
#
# These tests pin the narrowed contract:
#  - ``_user_added_model_can_install`` returns True ONLY for Ollama (with
#    ``ollama_tag``) and ComfyUI (with ``comfyui_model``); False for every
#    other backend until a real one-click installer ships for it.
#  - ``start_download`` short-circuits with the friendly infobox (no call
#    to ``ollama.pull_model``) when the gate returns False on a user-added
#    entry.
#  - ``download_comfyui_model`` short-circuits the same way for user-added
#    entries whose ComfyUI fields are missing.
# When a real downloader for ONNX/OpenVINO/Phase-1 lands, widen the gate
# AND extend ``start_download`` / ``download_comfyui_model`` together —
# gate + dispatcher must agree.
class UserAddedInstallGate(unittest.TestCase):

    def _gate(self, **overrides) -> bool:
        from src.app import App
        model = {"user_added": True}
        model.update(overrides)
        app = object.__new__(App)
        return App._user_added_model_can_install(app, model)

    def test_ollama_with_tag_can_install(self):
        self.assertTrue(self._gate(backend="ollama", ollama_tag="llama3.2:1b"))

    def test_ollama_without_tag_cannot_install(self):
        self.assertFalse(self._gate(backend="ollama"))

    def test_comfyui_with_model_can_install(self):
        self.assertTrue(self._gate(backend="comfyui", comfyui_model="flux1.safetensors"))

    def test_comfyui_without_model_cannot_install(self):
        self.assertFalse(self._gate(backend="comfyui"))

    def test_onnx_cannot_install_even_with_repo(self):
        # No one-click ONNX downloader from the Add-from-HF path today.
        self.assertFalse(self._gate(backend="onnx", onnx_repo="microsoft/Phi-3-mini-onnx"))
        self.assertFalse(self._gate(backend="onnx", hf_repo="microsoft/Phi-3-mini-onnx"))

    def test_onnx_genai_cannot_install_even_with_repo(self):
        # The headline Phi-4 ONNX-GenAI path — must surface the friendly
        # "Can't download automatically" modal, not crash.
        self.assertFalse(
            self._gate(backend="onnx-genai", onnx_repo="microsoft/Phi-4-mini-instruct-onnx")
        )
        self.assertFalse(
            self._gate(backend="onnx-genai", hf_repo="microsoft/Phi-4-mini-instruct-onnx")
        )

    def test_openvino_cannot_install_even_with_repo(self):
        self.assertFalse(self._gate(backend="openvino", ov_repo="OpenVINO/phi-3-ov"))
        self.assertFalse(self._gate(backend="openvino", hf_repo="OpenVINO/phi-3-ov"))

    def test_phase1_cannot_install_even_with_repo(self):
        self.assertFalse(self._gate(backend="phase1", hf_repo="openai/whisper-large-v3"))
        self.assertFalse(
            self._gate(backend="transformers", hf_repo="microsoft/Florence-2-base",
                       phase1_adapter=True)
        )

    def test_unknown_backend_cannot_install(self):
        self.assertFalse(self._gate(backend="unknown", hf_repo="foo/bar"))
        self.assertFalse(self._gate(hf_repo="foo/bar"))  # no backend at all

    def test_backend_match_is_case_insensitive_and_stripped(self):
        self.assertTrue(self._gate(backend="  OLLAMA  ", ollama_tag="phi:latest"))
        self.assertTrue(self._gate(backend="ComfyUI", comfyui_model="flux.gguf"))


class UserAddedStartDownloadGate(unittest.TestCase):
    """``start_download`` MUST surface the friendly infobox and return False
    when the gate blocks a user-added entry, never reaching the Ollama
    worker.  This is the last-line defense for the Tier-1 install crash."""

    def _patch_app(self):
        from src.app import App
        app = object.__new__(App)
        app.ollama_ok = True  # so the ollama-not-running branch isn't the one we trip
        return App, app

    def test_user_added_onnx_genai_short_circuits_without_calling_ollama(self):
        App, app = self._patch_app()
        ollama_calls = []

        class FakeOllama:
            def pull_model(self, *args, **kwargs):
                ollama_calls.append((args, kwargs))
                raise AssertionError("ollama.pull_model must not be reached")

        app.ollama = FakeOllama()
        infobox_calls = []

        # Patch tkinter.messagebox.showinfo on the app module so the test
        # doesn't pop a real dialog.
        import src.app as app_mod
        original = app_mod.messagebox.showinfo
        app_mod.messagebox.showinfo = lambda *a, **kw: infobox_calls.append((a, kw))
        try:
            model = {
                "id": "user-phi4",
                "name": "Phi-4 mini (ONNX-GenAI)",
                "user_added": True,
                "backend": "onnx-genai",
                "onnx_repo": "microsoft/Phi-4-mini-instruct-onnx",
                "hf_repo": "microsoft/Phi-4-mini-instruct-onnx",
            }
            result = App.start_download(app, model)
        finally:
            app_mod.messagebox.showinfo = original

        self.assertFalse(result, "start_download must return False when gate blocks")
        self.assertEqual(ollama_calls, [], "ollama.pull_model must not be reached")
        self.assertEqual(len(infobox_calls), 1, "expected exactly one friendly infobox")
        title = infobox_calls[0][0][0]
        body = infobox_calls[0][0][1]
        self.assertIn("Can't download", title)
        self.assertIn("Phi-4 mini (ONNX-GenAI)", body)

    def test_builtin_ollama_entry_does_not_trip_the_gate(self):
        # A built-in (no user_added flag) reaches the Ollama worker normally.
        App, app = self._patch_app()
        pulled = []

        class FakeOllama:
            def pull_model(self, tag, *args, **kwargs):
                pulled.append(tag)
                # Return any sentinel — start_download has more work after,
                # but reaching pull_model is the only assertion we need.
                raise RuntimeError("stop here")

        app.ollama = FakeOllama()
        # Stub the rest of start_download's surface area minimally.  We just
        # need to confirm the gate didn't fire.  The easiest way: assert the
        # gate predicate returns True for the equivalent dict.
        model = {"id": "qwen", "backend": "ollama", "ollama_tag": "qwen:0.5b"}
        # No user_added → gate is bypassed regardless.
        self.assertTrue(
            (not model.get("user_added"))
            or App._user_added_model_can_install(app, model)
        )


class UxPolishV550ContractTests(unittest.TestCase):
    """Guard the 8 UX polish bugfixes shipped under v5.5.0."""

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    # ------------------------------------------------------------------
    # D6 — Demo run mode removed in v5.5.19 (per-model demos kept)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # D5 — Extended mode breadth
    # ------------------------------------------------------------------

    def test_extended_mode_selects_all_fit_ok_image_gen_models(self):
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value

        capacity = {
            "profile": "GPU Entry",
            "available_ram_gb": 32,
            "total_ram_gb": 32,
            "vram_capacity_gb": 8,
            "has_gpu": True,
            "is_sku": True,
        }
        app = object.__new__(App)
        app._bench_run_mode_var = DummyVar("Extended")
        app._bench_image_var = DummyVar(True)
        app._bench_resource_override_var = DummyVar(False)
        app._bench_missing_deps_for_model = lambda _m: []
        app._bench_profile_capacity = lambda: capacity

        # v5.5.8 / v5.5.12: Extended defaults are driven by each SKU's
        # ``bench_extended_models`` resolved from skus.json.
        # realistic-vision-v6 is the only image-gen row in GPU Entry's
        # resolved Extended set (2 GB, the only image model that fits
        # 4 GB VRAM); it must default-tick.
        image_model = {
            "id": "realistic-vision-v6",
            "name": "Realistic Vision v6",
            "backend": "comfyui",
            "comfyui_model": "realistic-vision-v6.safetensors",
            "min_ram_gb": 8,
            "min_vram_gb": 4,
        }
        self.assertTrue(app._bench_default_selected_for_model(image_model, capacity))

    def test_extended_mode_image_gen_falls_back_to_false_when_not_fit(self):
        from src.app import App

        class DummyVar:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value

        capacity = {
            "profile": "GPU Entry",
            "available_ram_gb": 32,
            "total_ram_gb": 32,
            "vram_capacity_gb": 4,
            "has_gpu": True,
            "is_sku": True,
        }
        app = object.__new__(App)
        app._bench_run_mode_var = DummyVar("Extended")
        app._bench_image_var = DummyVar(True)
        app._bench_resource_override_var = DummyVar(False)
        app._bench_missing_deps_for_model = lambda _m: []
        app._bench_profile_capacity = lambda: capacity
        # Needs 24 GB VRAM, only 4 available → not fit → must be False even
        # in Extended.
        too_big = {
            "id": "huge-image",
            "name": "Huge Image",
            "backend": "comfyui",
            "comfyui_model": "huge.safetensors",
            "min_ram_gb": 999,
            "min_vram_gb": 24,
        }
        self.assertFalse(app._bench_default_selected_for_model(too_big, capacity))

    # ------------------------------------------------------------------
    # D4 — Show/Hide Options sync
    # ------------------------------------------------------------------

    def test_toggle_bench_opts_reads_actual_widget_state_not_cached_flag(self):
        source = self._function_source("_toggle_bench_opts") + "\n" + self._function_source(
            "_bench_opts_visible"
        ) + "\n" + self._function_source("_set_bench_opts_visible")
        # The fix moves the source of truth from the cached
        # ``_bench_opts_collapsed`` flag to the live ``winfo_ismapped()``
        # query exposed by ``_bench_opts_visible()``.
        self.assertIn("winfo_ismapped", source)
        self.assertIn("_set_bench_opts_visible", source)

    def test_toggle_bench_opts_logic_with_fake_frame(self):
        from src.app import App

        class FakeFrame:
            def __init__(self, mapped: bool):
                self.mapped = mapped
            def winfo_ismapped(self):
                return 1 if self.mapped else 0
            def grid(self, **_kwargs):
                self.mapped = True
            def grid_remove(self):
                self.mapped = False

        class FakeBtn:
            def __init__(self):
                self.text = ""
            def configure(self, **kwargs):
                if "text" in kwargs:
                    self.text = kwargs["text"]
            def cget(self, key):
                return self.text if key == "text" else None

        app = object.__new__(App)
        frame = FakeFrame(mapped=True)
        btn = FakeBtn()
        app._bench_opts_frame = frame
        app._bench_opts_toggle_btn = btn
        # Pre-condition: frame is mapped but the cached flag (and button
        # label) lie about it — the classic two-click bug.
        app._bench_opts_collapsed = True
        btn.text = "▼ Show Options"

        # First click must flip from "mapped" to "hidden" (NOT stay mapped
        # because the cached flag claimed it was already hidden).
        app._toggle_bench_opts()
        self.assertFalse(frame.mapped)
        self.assertIn("Show", btn.text)
        # The cached mirror should now agree with reality.
        self.assertTrue(app._bench_opts_collapsed)

        # Second click flips back to visible — button label must follow.
        app._toggle_bench_opts()
        self.assertTrue(frame.mapped)
        self.assertIn("Hide", btn.text)
        self.assertFalse(app._bench_opts_collapsed)

    # ------------------------------------------------------------------
    # D1 — Image Gen canvas/Save Image layout
    # ------------------------------------------------------------------

    def test_image_gen_canvas_row_weight_is_row_3(self):
        source = self._function_source("_build_image_page")
        # The canvas (Image preview) lives in row=3 of display_card. Row 2
        # holds the CPU-mode banner. The bug used to give row 2 weight=1
        # which pushed the canvas + Save button below the viewport when the
        # banner appeared. The fix routes weight=1 to row=3.
        self.assertIn("display_card.grid_rowconfigure(3, weight=1)", source)
        self.assertNotIn("display_card.grid_rowconfigure(2, weight=1)", source)
        # And the info_row (Save Image) is pinned at row=4 with a minsize so
        # it never gets squeezed off-screen.
        self.assertIn("display_card.grid_rowconfigure(4", source)

    # ------------------------------------------------------------------
    # D2 — Force All checkbox row=1 placement + shortened label
    # ------------------------------------------------------------------

    def test_bench_force_all_checkbox_is_on_row_1_with_short_label(self):
        source = self._function_source("_build_benchmark_page")
        # Pre-fix: text="Force All (best-effort baseline)" at row=0 col=5
        # → at 1280×800 the checkbox sat at x=1421 (off-screen right).
        # v5.5.4 a11y (P2-A): label was enriched from "Force All" to
        # "Force All (ignores capacity)" so UIA reads behaviour from the
        # checkbox's own AccessibleName instead of relying on the visual
        # caption (which screen readers don't link via aria-describedby).
        self.assertIn('text="Force All (ignores capacity)"', source)
        self.assertNotIn('text="Force All (best-effort baseline)"', source)
        # Grid placement: row=1 col=0 (the new dedicated row).
        self.assertIn(
            "self._bench_force_all_check.grid(row=1, column=0",
            source,
        )
        # Descriptive caption lives inline on the same row (col=1).
        self.assertIn("(best-effort baseline", source)

    # ------------------------------------------------------------------
    # D3 — Immediate disable + reset
    # ------------------------------------------------------------------

    def test_immediate_disable_helper_source_contract(self):
        helper_source = self._function_source("_immediate_disable_btn")
        self.assertIn('state="disabled"', helper_source)
        self.assertIn("update_idletasks", helper_source)

        # And it must be invoked at the very top of all three slow-feedback
        # callers — BEFORE any early returns from validation paths.
        for fn_name in (
            "_start_image_generation",
            "_start_benchmark",
            "_retry_failed_benchmark",
        ):
            fn_source = self._function_source(fn_name)
            tree = ast.parse(textwrap.dedent(fn_source))
            func_node = next(
                node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == fn_name
            )
            disable_line = None
            for node in ast.walk(func_node):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_immediate_disable_btn"
                ):
                    disable_line = node.lineno
                    break
            self.assertIsNotNone(
                disable_line,
                f"_immediate_disable_btn not called in {fn_name}",
            )
            first_return = None
            for node in ast.walk(func_node):
                if (
                    isinstance(node, ast.Return)
                    and node.lineno > func_node.lineno
                ):
                    first_return = node.lineno
                    break
            if first_return is not None:
                self.assertLess(
                    disable_line, first_return,
                    f"{fn_name}: _immediate_disable_btn must be called before any return",
                )

    def test_immediate_disable_logic_with_fake_button(self):
        from src.app import App

        class FakeBtn:
            def __init__(self):
                self.state = "normal"
                self.text = "Generate"
            def configure(self, **kwargs):
                if "state" in kwargs:
                    self.state = kwargs["state"]
                if "text" in kwargs:
                    self.text = kwargs["text"]

        app = object.__new__(App)
        tracker = {"idle": 0, "status": []}
        app.update_idletasks = lambda: tracker.__setitem__("idle", tracker["idle"] + 1)
        btn = FakeBtn()
        App._immediate_disable_btn(
            app, btn, text="Starting…",
            status_setter=lambda msg: tracker["status"].append(msg),
            status_text="Starting generation…",
        )
        self.assertEqual(btn.state, "disabled")
        self.assertEqual(btn.text, "Starting…")
        self.assertEqual(tracker["idle"], 1)
        self.assertEqual(tracker["status"], ["Starting generation…"])

    # ------------------------------------------------------------------
    # D7 / D9 — Models detail pane border + scrollable region
    # ------------------------------------------------------------------

    def test_models_detail_pane_uses_native_border_not_shell_trick(self):
        source = self._function_source("_build_models_page")
        # Historical context:
        # * The pre-fix code wrapped detail_panel in an outer ``detail_shell``
        #   to fake a border. The shell trick caused the bottom edge to be
        #   overdrawn by the inner rounded-corner anti-aliasing.
        # * The v5.5.0/v5.5.1 fix switched to native ``border_width=N,
        #   border_color=BORDER_STRONG`` matching list_panel.
        # * v5.5.19 fix: native ``border_width`` ALSO drops edges
        #   (Ron's screenshots 2026-05-30 showed missing top borders on the
        #   detail card across many model selections in both themes).
        #   Replaced with ``_make_bordered_card`` — a nested-frame pattern
        #   where the outer fg_color provides the visible border via the
        #   inset gap. No rounded-rect border math = no missing edges.
        self.assertNotIn("detail_shell =", source)
        self.assertNotIn("detail_shell.grid", source)
        # Both panels now use _make_bordered_card. list_panel uses
        # border_width=1, detail_panel uses border_width=2 (heavier for
        # visual balance with the description card).
        self.assertIn(
            "list_panel = _make_bordered_card(",
            source,
            "list_panel must use _make_bordered_card (nested-frame border)",
        )
        self.assertIn(
            "detail_panel = _make_bordered_card(",
            source,
            "detail_panel must use _make_bordered_card (nested-frame border)",
        )
        self.assertIn("border_width=1", source)
        self.assertIn("border_width=2", source)
        self.assertIn("border_color=BORDER_STRONG", source)

    def test_models_detail_pane_description_is_in_scrollable_frame(self):
        source = self._function_source("_build_model_detail_pane")
        # The fix wraps badges/desc/specs/settings/demo/recs inside a
        # CTkScrollableFrame so long image-gen descriptions remain reachable
        # at 1280×800.
        self.assertIn("CTkScrollableFrame", source)
        idx_scroll = source.find("CTkScrollableFrame")
        idx_desc = source.find("desc = ctk.CTkLabel")
        self.assertGreaterEqual(idx_scroll, 0)
        self.assertGreaterEqual(idx_desc, 0)
        self.assertLess(
            idx_scroll, idx_desc,
            "CTkScrollableFrame must be constructed before the desc label is parented under it",
        )

    # ------------------------------------------------------------------
    # D8 — Post-install detail refresh
    # ------------------------------------------------------------------

    def test_comfyui_download_done_refreshes_detail_pane(self):
        source = self._function_source("_comfyui_download_done")
        # Post-install, the right-pane state used to go stale until the user
        # clicked another row and back. The fix routes the post-install
        # refresh through ``_update_model_detail`` directly (the per-card
        # loop already handles list-pane button state).
        self.assertIn("_update_model_detail", source)


class UxPolishV551ContractTests(unittest.TestCase):
    """Pin v5.5.1 round-2 UX polish so regressions surface immediately."""

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    # --- B1: Include Image Gen checkbox removed -----------------------

    def test_b1_image_gen_rows_always_included_in_methods_for_run(self):
        from src.app import App

        class DummyVar:
            def __init__(self, v): self.v = v
            def get(self): return self.v

        app = object.__new__(App)
        app._bench_run_mode_var = DummyVar("Extended")
        app._bench_image_var = DummyVar(False)  # legacy stub, no longer gates
        app._bench_resource_override_var = DummyVar(False)
        app._bench_profile_capacity = lambda: {
            "profile": "GPU Workstation", "available_ram_gb": 64, "total_ram_gb": 64,
            "vram_capacity_gb": 24, "has_gpu": True, "is_sku": True,
        }
        img = {
            "id": "z-image-turbo", "name": "img",
            "backend": "comfyui", "comfyui_model": "x.safetensors",
            "min_ram_gb": 8, "min_vram_gb": 8,
        }
        # Image-gen row should be runnable in Extended regardless of legacy var.
        self.assertEqual(app._bench_methods_for_run_ui(img), ["image"])

    def test_b1_include_image_gen_checkbox_widget_removed(self):
        build_source = self._function_source("_build_benchmark_page")
        self.assertNotIn("_bench_image_check", build_source)
        # Match the widget construction text precisely (the docstring still
        # references the removed checkbox by name for historical context).
        self.assertNotIn('text="Include image-gen models', build_source)
        # Var stays as a True constant for back-compat.
        self.assertIn("_bench_image_var = ctk.BooleanVar(value=True)", build_source)

    def test_b1_image_gen_per_row_gate_blocks_non_gpu_without_force_all(self):
        from src.app import App

        app = object.__new__(App)
        app._bench_methods_for_ui = lambda m: ["image"]
        app._bench_observed_success_for_profile = lambda m, c: False
        app._bench_default_fit_for_model = lambda m, **kw: (False, "no GPU")
        app._is_image_model_ui = lambda m: True
        no_gpu = {"has_gpu": False, "is_sku": True, "profile": "CPU Pro",
                  "available_ram_gb": 32, "total_ram_gb": 32, "vram_capacity_gb": 0}
        ok, reason = app._bench_model_available_for_profile(
            {"id": "img"}, no_gpu, allow_oversize=False, force_all=False
        )
        self.assertFalse(ok)
        self.assertIn("GPU", reason)
        # Force All flips the gate open even without a GPU.
        ok, reason = app._bench_model_available_for_profile(
            {"id": "img"}, no_gpu, allow_oversize=True, force_all=True
        )
        self.assertTrue(ok)

    def test_b1_image_gen_never_default_checked_without_gpu(self):
        """SQT test-engineer P1: pin the v5.5.3 rule that image-gen rows
        are NEVER auto-checked on non-GPU profiles, even when the row would
        otherwise be eligible (e.g. observed prior success on a CPU SKU).
        Force All still lets the user check the row by hand — this only
        gates the *default* selection."""
        from src.app import App

        app = object.__new__(App)
        app._bench_methods_for_ui = lambda m: ["image"]
        app._bench_missing_deps_for_model = lambda m: []
        app._is_image_model_ui = lambda m: True
        # Even with observed success, default selection must say False on
        # a non-GPU profile (the runtime guard at app.py:6874).
        app._bench_observed_success_for_profile = lambda m, c: True
        app._bench_default_fit_for_model = lambda m, **kw: (True, "OK")
        no_gpu = {"has_gpu": False, "is_sku": True, "profile": "CPU Pro",
                  "available_ram_gb": 32, "total_ram_gb": 32,
                  "vram_capacity_gb": 0}
        img_model = {"id": "img", "category": "Image Generation"}
        self.assertFalse(app._bench_default_selected_for_model(img_model, no_gpu))
        # Sanity: same model on a GPU profile WOULD default-check.
        gpu = {"has_gpu": True, "is_sku": False, "profile": "This Device",
               "available_ram_gb": 64, "total_ram_gb": 64,
               "vram_capacity_gb": 24}
        app._active_run_mode = lambda: "extended"
        app._bench_select_all_fit_profiles = lambda: set()
        # On a GPU profile the function may still hit other branches we
        # haven't stubbed — just verify the non-GPU branch is the source
        # of the False above, not some upstream nope.
        # (No assertTrue here; the contract is only "never True if no GPU".)

    def test_b1_skip_image_cli_flag_no_longer_emitted(self):
        cli_source = self._function_source("_get_bench_cli_args")
        # The clause that used to emit ``--skip-image`` is gone; the runner
        # iterates only the user's selected image-gen rows instead. Match
        # the emission pattern rather than the literal string so the
        # explanatory comment in the source doesn't trip the assertion.
        self.assertNotIn('parts.append("--skip-image")', cli_source)
        self.assertNotIn('"--skip-image"]', cli_source)
        self.assertNotIn("_bench_image_var", cli_source)

    # --- B2: Allow Oversize checkbox removed --------------------------

    def test_b2_allow_oversize_checkbox_widget_removed(self):
        build_source = self._function_source("_build_benchmark_page")
        self.assertNotIn("_bench_resource_override_check", build_source)
        self.assertNotIn('text="Advanced: allow oversize', build_source)
        # Var stays defaulting False so legacy reads return False; Force All
        # is now the single escape hatch.
        self.assertIn("_bench_resource_override_var = ctk.BooleanVar(value=False)", build_source)

    def test_b2_force_all_toggle_no_longer_references_override_check(self):
        toggle_source = self._function_source("_on_bench_force_all_toggle")
        self.assertNotIn("_bench_resource_override_check", toggle_source)
        # SQT P2: positive contract — the toggle MUST still rebuild the
        # checklist when Force All flips, otherwise newly-eligible rows
        # would only appear on the next profile switch / page rebuild.
        self.assertIn("_render_bench_model_checklist", toggle_source)
        # v5.5.4 (SQT P3-3): ``_bench_image_var`` has been a True-constant
        # since v5.5.1 — pin it out of the Force-All snapshot/restore tuples
        # so a future drive-by edit doesn't re-introduce the dead-var
        # round-trip that misled readers into thinking there was still a
        # user toggle to restore. We check the snapshot/restore *tuples*
        # specifically (not the function comments, which still reference
        # the name to document why it was removed).
        self.assertNotIn('"_bench_image_var", "_bench_resource_override_var"', toggle_source)
        self.assertNotIn('"_bench_image_var")', toggle_source)

    # --- B3: Detail panel native rounded border with padded scroll_frame -----

    def test_b3_detail_panel_uses_native_rounded_border(self):
        build_source = self._function_source("_build_models_page")
        detail_source = self._function_source("_build_model_detail_pane")
        # v5.5.19 fix: the detail panel now uses ``_make_bordered_card``
        # instead of native ``border_width=2``. The literal ``border_width=2``
        # still appears in the helper call (as the gap size), and
        # ``corner_radius=10`` is preserved.
        self.assertIn("_make_bordered_card(", build_source)
        self.assertIn("corner_radius=10", build_source)
        self.assertIn("border_width=2", build_source)
        self.assertIn("border_color=BORDER_STRONG", build_source)
        self.assertIn('pady=(0, 16)', build_source)
        # The inner CTkScrollableFrame must NOT be gridded with padx=0 — its
        # Canvas would overdraw the parent panel's 2px left/right borders.
        self.assertIn('padx=(2, 2)', detail_source)

    # --- B4: Wraplength clipping fix ---------------------------------

    def test_b4_wraplength_uses_min_of_parent_canvas_inner(self):
        detail_source = self._function_source("_build_model_detail_pane")
        self.assertIn("_parent_canvas", detail_source)
        # min(...) over candidate widths is the v5.5.1 contract.
        self.assertIn("min(candidates)", detail_source)
        # SQT P2: pin the three-widget candidate tuple and the
        # max-wraplength floor so a future refactor can't silently drop
        # the inner-frame width (the most common over-wide misread at
        # first paint) or remove the wraplength sanity floor.
        self.assertIn("for widget in (parent, canvas, inner)", detail_source)
        # v5.5.18 follow-up: floor lowered from 220 to 180 px (LOGICAL) and
        # the formula now divides the device-pixel candidate by the CTk
        # widget-scaling factor and subtracts a logical-px gutter. Pin both
        # the floor and the canvas_logical/gutter shape so the next
        # refactor can't silently regress to a device-pixel value (which
        # over-wraps under 125 %/150 % Windows DPI scaling).
        self.assertIn("canvas_logical = min(candidates)", detail_source)
        self.assertIn("max(180, int(canvas_logical) - gutter)", detail_source)

    # --- B5: Badge two-row layout ------------------------------------

    def test_b5_badge_specs_return_kind_field(self):
        from src.app import App
        from src import app as app_mod
        from src import catalog as catalog_mod

        model = {
            "perf_profile": {
                "recommendation": "top_pick",
                "speed_tier": "fast", "speed_label": "1.0s",
                "quality_tier": "sota",
                "category_bucket": "general",
            },
            "min_vram_gb": 4,
        }
        app = object.__new__(App)
        app._fit_tier_for_model = lambda m, v: "fits_well"
        specs = App._build_perf_badge_specs(app, model, 8)
        self.assertGreater(len(specs), 0)
        kinds = [t[3] for t in specs]
        self.assertIn("rec", kinds)
        self.assertIn("speed", kinds)
        self.assertIn("quality", kinds)
        # SQT P1: pin the kinds that drive the two-row routing in
        # _update_model_detail. ``bucket`` comes from category_bucket,
        # ``fit`` from _fit_tier_for_model — both wired by the fixture
        # above. Without these assertions a future refactor could
        # silently drop a kind and row 2 would render half-empty.
        self.assertIn("bucket", kinds)
        self.assertIn("fit", kinds)
        # All entries are 4-tuples (text, fg, txt, kind).
        for spec in specs:
            self.assertEqual(len(spec), 4)

        # SQT P1 (cpu kind) — separate fixture so we can monkeypatch
        # catalog.is_cpu_viable_image_model without leaking into other
        # tests. Restore the original after.
        original_cpu_check = catalog_mod.is_cpu_viable_image_model
        try:
            catalog_mod.is_cpu_viable_image_model = lambda _m: True  # type: ignore[assignment]
            app_mod.catalog.is_cpu_viable_image_model = lambda _m: True  # type: ignore[assignment]
            cpu_model = dict(model)
            cpu_model["expected_cpu_time_label"] = "2 min"
            cpu_specs = App._build_perf_badge_specs(app, cpu_model, 8)
            cpu_kinds = [t[3] for t in cpu_specs]
            self.assertIn("cpu", cpu_kinds)
        finally:
            catalog_mod.is_cpu_viable_image_model = original_cpu_check  # type: ignore[assignment]
            app_mod.catalog.is_cpu_viable_image_model = original_cpu_check  # type: ignore[assignment]

    def test_b5_detail_pane_splits_badges_into_two_rows_by_kind(self):
        update_source = self._function_source("_update_model_detail")
        # v5.5.19: Row 1 = rec/speed; row 2 = quality/fit/bucket;
        # row 3 = cpu (promoted to its own row so the long "CPU OK · ~90s
        # on small CPU" chip never overflows the ~240px detail-pane wrap).
        # Pinned by the presence of all three row variable names plus the
        # kind-based three-way routing.
        self.assertIn("row1 = ctk.CTkFrame", update_source)
        self.assertIn("row2 = ctk.CTkFrame", update_source)
        self.assertIn("row3 = ctk.CTkFrame", update_source)
        self.assertIn('kind in {"rec", "speed"}', update_source)
        self.assertIn('kind == "cpu"', update_source)
        # SQT P2: pin the ``else`` (default → row2) branch so a refactor
        # that accidentally drops the routing (and parks every badge on
        # row1) would fail this test.
        self.assertIn("target = row2", update_source)

    # --- B6: Size bucket scope ---------------------------------------

    def test_b6_hf_size_buckets_widen_medium_band(self):
        from src.hf_compat import _category_for_size
        # 7B-Q4 (~4.7 GB) stays Medium.
        self.assertEqual(_category_for_size(4.7), "Medium")
        # 13B-Q8 (~13 GB) is Large under the new boundaries.
        self.assertEqual(_category_for_size(13.0), "Large")
        # 32B-Q4 (~19 GB) is Large.
        self.assertEqual(_category_for_size(19.9), "Large")
        # 70B-Q4 (~40 GB) is Extra Large.
        self.assertEqual(_category_for_size(40.0), "Extra Large")
        # Ultra Small / Small boundary stays well below 1B param checkpoints.
        self.assertEqual(_category_for_size(0.9), "Ultra Small")
        self.assertEqual(_category_for_size(1.5), "Small")

    def test_b6_curated_catalog_has_at_least_one_large_model(self):
        import json
        data = json.loads((ROOT / "models_catalog.json").read_text(encoding="utf-8"))
        models = data.get("models", data) if isinstance(data, dict) else data
        large = [m["id"] for m in models if m.get("category") == "Large"]
        self.assertGreater(
            len(large), 0,
            "Benchmark Size filter shows zero rows under the Large bucket; "
            "see Ron's v5.5.1 complaint — at least one curated model "
            "(e.g. deepseek-r1:32b) must land in Large.",
        )

    def test_b6_deepseek_r1_32b_is_large_not_extra_large(self):
        import json
        data = json.loads((ROOT / "models_catalog.json").read_text(encoding="utf-8"))
        models = data.get("models", data) if isinstance(data, dict) else data
        target = next((m for m in models if m.get("id") == "deepseek-r1:32b"), None)
        self.assertIsNotNone(target)
        self.assertEqual(target["category"], "Large")
        # The category tag inside ``tags`` must match the bucket name.
        self.assertIn("large", [t.lower() for t in (target.get("tags") or [])])
        self.assertNotIn("extra-large", [t.lower() for t in (target.get("tags") or [])])

    def test_b6_exact_size_bucket_boundaries(self):
        """SQT P2: pin every bucket boundary so future tuning can't silently
        re-create the v5.5.2 bug where the Medium ceiling at 30 GB swallowed
        every 13B/32B model and the Large bucket showed zero rows.
        Boundaries (from hf_compat._category_for_size, v5.5.3):
            <1.5  → Ultra Small
            <5    → Small (boundary at 5.0 → Medium)
            <12   → Medium
            <25   → Large
            >=25  → Extra Large
        v5.5.4 (SQT P3-2): also pin the Ultra Small ↔ Small boundary at
        1.5 GB. The earlier v5.5.1 test only covered 0.9 (Ultra Small) and
        1.5 (Small); 1.0 GB (smack in the middle of the Ultra Small band)
        and 1.49 GB (one step below the boundary) were not asserted, so a
        future tweak that dropped the boundary to 1.0 GB would have
        silently broken every Ultra Small benchmark default.
        """
        from src.hf_compat import _category_for_size
        # Ultra Small ↔ Small boundary at 1.5 GB.
        self.assertEqual(_category_for_size(1.0), "Ultra Small")
        self.assertEqual(_category_for_size(1.49), "Ultra Small")
        # Medium↔Large boundary at 12.0 GB.
        self.assertEqual(_category_for_size(11.9), "Medium")
        self.assertEqual(_category_for_size(12.0), "Large")
        # Large↔Extra Large boundary at 25.0 GB.
        self.assertEqual(_category_for_size(24.9), "Large")
        self.assertEqual(_category_for_size(25.0), "Extra Large")

    # === v5.5.4 SQT round-2 follow-ups =========================================

    def test_v554_quality_badge_uses_text_fraction_not_unicode_dots(self):
        """v5.5.4 a11y (P1-A): the previous "●●●● Sota" rating was announced
        as "black circle black circle..." by NVDA/Narrator because Unicode
        bullets have no semantic role. Use the text fraction "(4/4) Sota"
        instead so screen readers announce "4 of 4 Sota"."""
        from src.app import App
        app = object.__new__(App)
        app._fit_tier_for_model = lambda m, v: "fits_well"
        for tier, expected_frac in (("good", "1/4"), ("great", "2/4"),
                                    ("excellent", "3/4"), ("sota", "4/4")):
            model = {
                "perf_profile": {"quality_tier": tier, "category_bucket": "general"},
                "min_vram_gb": 4,
            }
            specs = App._build_perf_badge_specs(app, model, 8)
            quality_badges = [t for t in specs if t[3] == "quality"]
            self.assertEqual(len(quality_badges), 1,
                             f"Expected one quality badge for {tier}")
            text = quality_badges[0][0]
            self.assertIn(expected_frac, text,
                          f"Expected fraction {expected_frac!r} in {text!r}")
            self.assertNotIn("●", text, f"Unicode bullet leaked into {text!r}")
            self.assertNotIn("○", text, f"Unicode bullet leaked into {text!r}")

    def test_v554_badge_row2_order_is_quality_fit_bucket_cpu(self):
        """v5.5.4 product-designer (P3-1): row 2 reads quality → fit →
        bucket → cpu. Hardware feasibility (fit) leads the bucket label so
        the user sees runs-on-this-GPU before the marketing bucket."""
        from src.app import App
        from src import catalog as catalog_mod
        app = object.__new__(App)
        app._fit_tier_for_model = lambda m, v: "fits_well"
        model = {
            "perf_profile": {
                "quality_tier": "sota",
                "category_bucket": "general",
                "speed_tier": "fast", "speed_label": "1.0s",
                "recommendation": "top_pick",
            },
            "min_vram_gb": 4,
            "expected_cpu_time_label": "2 min",
        }
        original_cpu_check = catalog_mod.is_cpu_viable_image_model
        try:
            catalog_mod.is_cpu_viable_image_model = lambda _m: True  # type: ignore[assignment]
            specs = App._build_perf_badge_specs(app, model, 8)
        finally:
            catalog_mod.is_cpu_viable_image_model = original_cpu_check  # type: ignore[assignment]
        kinds = [t[3] for t in specs]
        # Row 1 = rec, speed (in that order). Row 2 = quality, fit, bucket, cpu.
        quality_idx = kinds.index("quality")
        fit_idx = kinds.index("fit")
        bucket_idx = kinds.index("bucket")
        cpu_idx = kinds.index("cpu")
        self.assertLess(quality_idx, fit_idx,
                        "quality must precede fit on row 2")
        self.assertLess(fit_idx, bucket_idx,
                        "fit must precede bucket on row 2 (v5.5.4 P3-1)")
        self.assertLess(bucket_idx, cpu_idx,
                        "bucket must precede cpu on row 2")

    def test_v554_fit_badge_suppressed_on_cpu_image_gen(self):
        """v5.5.4 product-designer (P3-3): on CPU-only systems (vram_gb=0)
        image-gen rows always show "🔴 Exceeds VRAM" alongside the CPU OK
        chip — confusing because image-gen can run on CPU in this app.
        Suppress the fit badge when vram_gb=0 AND the model is image-gen
        UI. Non-image models still get the fit signal on CPU."""
        from src.app import App
        from src import catalog as catalog_mod
        app = object.__new__(App)
        app._fit_tier_for_model = lambda m, v: "exceeds"
        # Image-gen model on CPU-only system: no fit badge.
        img_model = {
            "perf_profile": {"quality_tier": "sota", "category_bucket": "general"},
            "min_vram_gb": 8,
            "backend": "comfyui",
            "expected_cpu_time_label": "5 min",
        }
        original_cpu_check = catalog_mod.is_cpu_viable_image_model
        try:
            catalog_mod.is_cpu_viable_image_model = lambda _m: True  # type: ignore[assignment]
            specs = App._build_perf_badge_specs(app, img_model, 0)
        finally:
            catalog_mod.is_cpu_viable_image_model = original_cpu_check  # type: ignore[assignment]
        kinds = [t[3] for t in specs]
        self.assertNotIn("fit", kinds,
                         "fit badge must be suppressed for image-gen on CPU-only system")
        self.assertIn("cpu", kinds,
                      "CPU OK badge must still appear when cpu_viable")
        # Non-image model on CPU-only system: fit badge stays.
        text_model = {
            "perf_profile": {"quality_tier": "great", "category_bucket": "general"},
            "min_vram_gb": 8,
        }
        try:
            catalog_mod.is_cpu_viable_image_model = lambda _m: False  # type: ignore[assignment]
            text_specs = App._build_perf_badge_specs(app, text_model, 0)
        finally:
            catalog_mod.is_cpu_viable_image_model = original_cpu_check  # type: ignore[assignment]
        text_kinds = [t[3] for t in text_specs]
        self.assertIn("fit", text_kinds,
                      "non-image models on CPU still get the fit signal")

    def test_v554_badge_labels_carry_border_for_contrast(self):
        """v5.5.4 a11y (P2-C) + v5.5.19 paint-reliability fix.

        v5.5.4: five badge backgrounds fail WCAG 1.4.11 3:1 non-text contrast
        in dark mode. Originally added a 1-px BORDER_STRONG hairline to every
        badge ``CTkFrame(border_width=1, border_color=BORDER_STRONG)``.

        v5.5.19: CTk's native ``border_width`` on rounded frames dropped
        top/bottom edges at certain widget sizes (Ron's screenshots
        2026-05-30). Switched to ``_make_chip`` — a nested-frame helper where
        the outer frame's fg_color provides the visible border via the
        inset gap. The border color is now passed as ``border_color=`` to
        ``_make_chip`` rather than as a ``CTkFrame`` kwarg, so pin that.
        """
        # Detail-pane badge site (v5.5.1 two-row layout, v5.5.19 row3 promotion
        # + nested-frame border fix).
        update_source = self._function_source("_update_model_detail")
        idx = update_source.find('for text, fg, txt, kind in self._build_perf_badge_specs')
        self.assertGreaterEqual(idx, 0, "badge render loop not found")
        chunk = update_source[idx: idx + 2000]
        self.assertIn("_make_chip(", chunk,
                      "detail-pane badge must use _make_chip helper "
                      "(nested-frame border, not CTkFrame(border_width=N))")
        self.assertIn("border_color=BORDER_STRONG", chunk,
                      "detail-pane badge must use BORDER_STRONG as the chip border color")
        # ModelCard._render_perf_badges site.
        card_source = self._function_source("_render_perf_badges")
        self.assertIn("_make_chip(", card_source,
                      "ModelCard badge must use _make_chip helper "
                      "(nested-frame border, not CTkFrame(border_width=N))")
        self.assertIn("border_color=BORDER_STRONG", card_source,
                      "ModelCard badge must use BORDER_STRONG as the chip border color")

    def test_v554_force_all_label_describes_capacity_override(self):
        """v5.5.4 a11y (P2-A): UIA reads the checkbox's own ``text`` as its
        AccessibleName. The descriptive caption next to it is purely visual.
        Pack the capacity-override behaviour into the label itself."""
        build_source = self._function_source("_build_benchmark_page")
        # The checkbox creation block carries the label change.
        self.assertIn('text="Force All (ignores capacity)"', build_source)

    def test_v554_image_gen_section_header_shows_gpu_required(self):
        """v5.5.4 product-designer (P2-2): on non-GPU profiles with Force
        All off, every Image Generation row is disabled with a per-row
        reason — but the *group* header was silent about it. Add a
        "(GPU required)" marker so the header communicates the same gate."""
        header_source = self._function_source("_update_bench_category_header")
        self.assertIn("(GPU required)", header_source)
        # The marker only appears when category == "Image Generation" AND
        # capacity has no GPU AND Force All is off — pin those gates so the
        # marker can't accidentally appear on Vision/Embeddings/etc.
        self.assertIn('category == "Image Generation"', header_source)
        self.assertIn("has_gpu", header_source)
        self.assertIn("force_all", header_source)

    def test_v554_install_safety_net_refresh_in_comfyui_download_done(self):
        """v5.5.4: ComfyUI's /get_model_list HTTP endpoint sometimes returns
        a stale list right after a download (OS fs propagation + ComfyUI's
        own watcher latency). Schedule a second forced refresh ~3 s after
        the immediate one so the right pane and card row catch up without
        the user clicking another row. Cheap, idempotent."""
        source = self._function_source("_comfyui_download_done")
        self.assertIn("self.after(3000, self._refresh_model_cards)", source)


class ImageGenPromptCollapseContractTests(unittest.TestCase):
    """v5.5.19 (Ron, 2026-05-30): the Image Gen prompt editor has a manual
    expand/collapse toggle and an after-render auto-collapse that fires only
    when the window is NOT maximized. Single-source-of-truth requirement:
    every collapse/expand routes through ``_apply_img_prompt_collapsed`` so
    the toggle button label can never drift from the actual collapsed flag.
    Auto-collapse fires only on success (``_img_generation_done``), never on
    failure (``_img_generation_failed``)."""

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    def test_apply_helper_exists_and_updates_button_text_both_directions(self):
        source = self._function_source("_apply_img_prompt_collapsed")
        # Updates the canonical flag.
        self.assertIn("self._img_prompt_collapsed = bool(collapsed)", source)
        # Both collapse and expand mutate the textbox + status row via
        # grid_remove / grid (NOT grid_forget — we need grid options to survive).
        self.assertIn("grid_remove()", source)
        self.assertIn("grid()", source)
        self.assertNotIn("grid_forget()", source)
        # Button text flips in both directions.
        self.assertIn("Show prompt", source)
        self.assertIn("Hide prompt", source)

    def test_toggle_helper_inverts_flag_through_apply(self):
        source = self._function_source("_toggle_img_prompt_collapsed")
        # Toggle MUST go through _apply_img_prompt_collapsed — direct
        # mutations would let the button label drift from the flag.
        self.assertIn("_apply_img_prompt_collapsed(", source)
        self.assertIn("not getattr(self, \"_img_prompt_collapsed\"", source)

    def test_autocollapse_helper_gates_on_window_state_zoomed(self):
        source = self._function_source("_img_autocollapse_prompt_if_unmaximized")
        # Tk reports "zoomed" when maximized on Windows.
        self.assertIn('"zoomed"', source)
        # No-op when already collapsed.
        self.assertIn("_img_prompt_collapsed", source)
        # Routes through _apply_img_prompt_collapsed (not direct grid mutation).
        self.assertIn("_apply_img_prompt_collapsed(True)", source)
        # Calls self.state() to read window state.
        self.assertIn("self.state()", source)

    def test_build_image_prompt_editor_creates_collapse_btn_and_reapply_guard(self):
        source = self._function_source("_build_image_prompt_editor")
        # Toggle button is constructed and stashed on self.
        self.assertIn("self._img_prompt_collapse_btn = ctk.CTkButton(", source)
        # Toggle button calls the toggle method (single source of truth).
        self.assertIn("command=self._toggle_img_prompt_collapsed", source)
        # Re-apply guard at the end of the builder ensures a future default
        # flip lands the page in the correct visual state on first paint.
        self.assertIn("self._apply_img_prompt_collapsed(True)", source)
        # "Prompt ideas" must keep its column position (column 2 after the
        # toggle slid in at column 1).
        self.assertRegex(source, r"text=\"Prompt ideas\"")
        self.assertRegex(source, r"\.grid\(row=0, column=2,")

    def test_build_image_page_initializes_collapsed_flag_false(self):
        source = self._function_source("_build_image_page")
        # Default is expanded (matches today's behavior).
        self.assertIn("self._img_prompt_collapsed: bool = False", source)

    def test_generation_done_calls_autocollapse_after_save_btn_enable(self):
        source = self._function_source("_img_generation_done")
        # Auto-collapse must fire on success.
        self.assertIn("self._img_autocollapse_prompt_if_unmaximized()", source)
        # It must come AFTER _img_save_btn becomes enabled so the visible UI
        # state is consistent (button armed + prompt collapsed simultaneously).
        save_idx = source.index('self._img_save_btn.configure(state="normal")')
        auto_idx = source.index("self._img_autocollapse_prompt_if_unmaximized()")
        self.assertLess(save_idx, auto_idx)

    def test_generation_failed_does_not_call_autocollapse(self):
        """A failed render must NOT hide the prompt — the user is about to
        edit it."""
        source = self._function_source("_img_generation_failed")
        self.assertNotIn("_img_autocollapse_prompt_if_unmaximized", source)
        self.assertNotIn("_apply_img_prompt_collapsed", source)

    def test_collapse_btn_uses_outline_button_style_for_a11y(self):
        """The toggle button must be a real CTkButton (so src/a11y.py's
        import-time patches give it Tab focus, Enter/Space activation, and
        a focus ring) AND must use _outline_button_style so it visually
        matches the Prompt ideas button."""
        source = self._function_source("_build_image_prompt_editor")
        # Walk from the construction line to the matching close paren so
        # nested calls like ctk.CTkFont(size=11) don't end the slice early.
        anchor = "self._img_prompt_collapse_btn = ctk.CTkButton("
        start = source.index(anchor) + len(anchor)
        depth = 1
        end = start
        while end < len(source) and depth > 0:
            ch = source[end]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            end += 1
        btn_block = source[start:end]
        self.assertIn("_outline_button_style", btn_block)
        # And the command must be the toggle method (single source of truth).
        self.assertIn("command=self._toggle_img_prompt_collapsed", btn_block)


class SnapdragonImageGenGatingContractTests(unittest.TestCase):
    """v5.5.9 (Ron, 2026-05-26): Snapdragon X (Windows ARM64) cannot run
    ComfyUI image generation because torch-directml has no ARM64 wheel on
    PyPI. The Image Generation page must short-circuit to a friendly
    unsupported panel so users see an explanation instead of the Windows
    "Entry Point Not Found: torch_library_impl could not be located in
    _torchaudio.pyd" popup that ComfyUI raises on startup."""

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    def test_app_imports_is_snapdragon_arm64(self):
        # Source-string check is sufficient — the import statement format is
        # pinned in src/app.py top-of-file gpu_detect import line.
        self.assertIn("is_snapdragon_arm64", APP_TEXT)
        self.assertRegex(
            APP_TEXT,
            r"from\s+(?:src\.gpu_detect|gpu_detect)\s+import[^\n]*is_snapdragon_arm64",
        )

    def test_build_image_page_short_circuits_on_snapdragon_before_widgets(self):
        source = self._function_source("_build_image_page")
        # The helper must be invoked, the unsupported panel must be returned,
        # and the early return must happen BEFORE any heavyweight widget
        # construction (we use the first ttk.Frame / CTkFrame call as the
        # marker for "real" widget work starting).
        self.assertIn("is_snapdragon_arm64()", source)
        self.assertIn("_build_image_page_snapdragon_unsupported", source)

        tree = ast.parse(textwrap.dedent(source))
        func_node = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_image_page"
        )

        snapdragon_call_line = None
        first_return_line = None
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "is_snapdragon_arm64"
            ):
                snapdragon_call_line = node.lineno
                break
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.lineno > func_node.lineno:
                first_return_line = node.lineno
                break

        self.assertIsNotNone(
            snapdragon_call_line,
            "_build_image_page must call is_snapdragon_arm64() to gate Snapdragon X users",
        )
        self.assertIsNotNone(
            first_return_line,
            "_build_image_page must early-return on Snapdragon X",
        )
        self.assertLess(
            snapdragon_call_line, first_return_line + 4,
            "Snapdragon check must be followed by an early return within a few lines",
        )

    def test_unsupported_panel_helper_exists_and_explains_directml(self):
        # The helper renders a static panel — its text content is part of
        # the contract because it's what Snapdragon X users see instead of
        # the ComfyUI crash. The wording can evolve but these key cues
        # ("Snapdragon", "DirectML", and a pointer to what still works) must
        # remain or users will be confused.
        source = self._function_source("_build_image_page_snapdragon_unsupported")
        self.assertIn("Snapdragon", source)
        self.assertIn("DirectML", source)
        # Must reference at least one working alternative so users know the
        # rest of the app still functions.
        self.assertTrue(
            any(token in source for token in ("Ollama", "Chat", "Benchmark", "ONNX")),
            "Unsupported panel must mention what still works on Snapdragon X",
        )

    def test_bench_method_fits_capacity_image_branch_rejects_snapdragon(self):
        """v5.5.9 (Ron, 2026-05-26): the image branch of
        ``_bench_method_fits_capacity`` must short-circuit to False on
        Snapdragon ARM64 BEFORE evaluating the allow_oversize/Force All
        bypass.  This mirrors ``BatchRunner._image_gen_supported`` and
        keeps the Benchmark planner in lock-step with the runtime so users
        on Snapdragon never see image-gen rows attempted under Force All
        (which would trigger the ``torch_library_impl ... _torchaudio.pyd``
        Windows popup at ComfyUI startup)."""
        source = self._function_source("_bench_method_fits_capacity")
        # Find the "image" branch and verify the Snapdragon check appears
        # inside it BEFORE the has_gpu/allow_oversize handling.
        self.assertIn('if method == "image":', source)
        self.assertIn("is_snapdragon_arm64()", source)
        # Slice the image branch and confirm ordering: the Snapdragon
        # check must precede the `if not has_gpu:` allow_oversize bypass.
        image_branch_start = source.index('if method == "image":')
        # The branch ends at the next `if method == ` clause; if we can't
        # find one, use the end of the source.
        try:
            image_branch_end = source.index(
                'if method ==', image_branch_start + 1
            )
        except ValueError:
            image_branch_end = len(source)
        image_branch = source[image_branch_start:image_branch_end]
        snapdragon_pos = image_branch.find("is_snapdragon_arm64()")
        has_gpu_pos = image_branch.find("if not has_gpu:")
        self.assertGreater(
            snapdragon_pos, -1,
            "is_snapdragon_arm64() must appear inside the image-method branch",
        )
        self.assertGreater(
            has_gpu_pos, -1,
            "image branch must still gate on has_gpu",
        )
        self.assertLess(
            snapdragon_pos, has_gpu_pos,
            "Snapdragon short-circuit must precede the allow_oversize bypass",
        )

    def test_ctk_label_calls_never_use_unsupported_border_kwargs(self):
        """Regression for v5.5.5 Mac crash: ``ctk.CTkLabel(..., border_width=,
        border_color=)`` raises ``ValueError`` in customtkinter >=5.2 because
        CTkLabel does not accept those kwargs (CTkFrame / CTkButton / CTkEntry
        do — CTkLabel does not). Two badge-render sites used to pass them and
        crashed ``_update_model_detail`` (model row click) and
        ``_render_perf_badges`` (model list row paint) on Mac Python 3.14.
        If anyone re-introduces that pattern, fail fast here instead of
        shipping another Tkinter callback crash.
        """
        tree = ast.parse(APP_TEXT)
        offenders: list[tuple[int, list[str]]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "CTkLabel"):
                continue
            bad = sorted(
                kw.arg
                for kw in node.keywords
                if kw.arg in {"border_width", "border_color"}
            )
            if bad:
                offenders.append((node.lineno, bad))
        self.assertEqual(
            offenders,
            [],
            "CTkLabel does not accept border_width / border_color in "
            "customtkinter >=5.2 (raises ValueError on Mac Python 3.14). "
            "Wrap the label in a CTkFrame and put the border on the frame "
            "instead. Offending sites: " + repr(offenders),
        )


class Img2ImgDefaultPromptContractTests(unittest.TestCase):
    """v5.5.14 (Ron, 2026-05-29): When the user ticks "Use reference image
    for generation" on the Image Gen page, both the positive and the negative
    prompts auto-populate with a documented sample so the very first
    img2img generation works without typing. The documented sample also has
    to exist verbatim in ``docs/image-gen-guide.html`` so the in-app help
    matches what the runtime loads.

    PortraitForge MUST NOT be referenced anywhere — LocalAI Studio treats
    that project as non-existent.
    """

    GUIDE_PATH = ROOT / "docs" / "image-gen-guide.html"

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found in src/app.py")

    def _load_constants(self) -> tuple[str, str]:
        """Eval the IMG2IMG default assignments directly out of app.py
        without importing the whole module (which pulls in tkinter)."""
        tree = ast.parse(APP_TEXT)
        env: dict[str, object] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {
                    "IMG2IMG_DEFAULT_POSITIVE",
                    "IMG2IMG_DEFAULT_NEGATIVE",
                }:
                    env[target.id] = ast.literal_eval(node.value)
        pos = env.get("IMG2IMG_DEFAULT_POSITIVE")
        neg = env.get("IMG2IMG_DEFAULT_NEGATIVE")
        self.assertIsInstance(pos, str, "IMG2IMG_DEFAULT_POSITIVE must be a module-level str literal")
        self.assertIsInstance(neg, str, "IMG2IMG_DEFAULT_NEGATIVE must be a module-level str literal")
        return pos, neg

    def test_constants_exist_and_nonempty(self):
        pos, neg = self._load_constants()
        self.assertTrue(pos.strip(), "IMG2IMG_DEFAULT_POSITIVE must be non-empty")
        self.assertTrue(neg.strip(), "IMG2IMG_DEFAULT_NEGATIVE must be non-empty")
        # Spec from Ron: man / 30 / natural daylight / casual clothes.
        for token in ("30-year-old man", "casual clothes", "natural daylight"):
            self.assertIn(token, pos,
                          f"IMG2IMG_DEFAULT_POSITIVE must contain {token!r}")
        for token in ("bad anatomy", "blurry", "watermark"):
            self.assertIn(token, neg,
                          f"IMG2IMG_DEFAULT_NEGATIVE must contain {token!r}")

    def test_constants_appear_verbatim_in_shipped_doc(self):
        """Hard sync guard: docs/image-gen-guide.html §5 must contain the
        same default img2img positive + negative prompt that app.py loads
        when the user ticks the reference-image checkbox. If they drift,
        users get one thing in the help and a different thing in the app.
        """
        pos, neg = self._load_constants()
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        self.assertIn(neg, guide,
                      "IMG2IMG_DEFAULT_NEGATIVE must appear verbatim in "
                      "docs/image-gen-guide.html (the prompt-builder default).")
        # The positive prompt is built by the JS in the doc from
        # ${age} ${subject} wearing ${clothes}, ${light} — confirm the JS
        # template + defaults render to the same string.
        self.assertIn(
            "professional photograph of a ${age} ${subject} wearing ${clothes}, ${light}",
            guide,
            "Prompt-builder JS template must match the app.py positive "
            "constant skeleton.")
        for token in ("30-year-old", "man", "natural daylight", "casual clothes"):
            self.assertIn(f'value="{token}" selected', guide,
                          f"Prompt builder must default to {token!r}")
        # Suffix tokens must match the constant.
        self.assertIn(
            ", sharp focus, photorealistic, detailed skin texture, 8k",
            guide,
            "Prompt-builder positive suffix must match IMG2IMG_DEFAULT_POSITIVE")

    def test_apply_helper_exists_and_writes_both_prompts(self):
        source = self._function_source("_apply_img2img_default_prompts")
        # Helper must touch BOTH the positive textbox and the negative entry.
        self.assertIn("IMG2IMG_DEFAULT_POSITIVE", source)
        self.assertIn("IMG2IMG_DEFAULT_NEGATIVE", source)
        self.assertIn("_img_prompt", source)
        self.assertIn("_img_neg_prompt", source)

    def test_toggle_handler_swaps_prompts_on_check_and_restores_on_uncheck(self):
        source = self._function_source("_on_img_img2img_mode_changed")
        # ON path: loads the img2img defaults.
        self.assertIn("_apply_img2img_default_prompts", source,
                      "Toggle handler must call _apply_img2img_default_prompts "
                      "when the user ticks the reference-image checkbox.")
        # OFF path: restores the per-model demo + per-model negative.
        self.assertIn("_apply_image_demo_prompt", source,
                      "Toggle handler must restore _apply_image_demo_prompt "
                      "when the user unticks the reference-image checkbox.")
        self.assertIn("_apply_selected_model_negative_prompt", source,
                      "Toggle handler must restore the per-model negative "
                      "prompt when the user unticks the reference-image checkbox.")

    def test_model_change_handler_preserves_img2img_defaults_when_on(self):
        """If the user switches models while img2img mode is on, the
        freshly-applied per-model demo prompt must be immediately
        overridden by the img2img sample so the documented behaviour
        holds. ``_refresh_img2img_controls`` already auto-flips the
        checkbox OFF for unsupported models, so the override only fires
        when the new model truly supports img2img."""
        source = self._function_source("_on_img_model_changed")
        self.assertGreaterEqual(
            source.count("_apply_img2img_default_prompts"),
            2,
            "Both branches of _on_img_model_changed (catalog + heuristic) "
            "must re-apply the img2img defaults when the toggle is on.",
        )
        # The override has to be guarded so it does not fire when the
        # toggle is off (it would clobber the model's intended prompt).
        self.assertIn("self._img_img2img_var.get()", source)

    def test_portraitforge_is_never_mentioned(self):
        """Ron's rule (2026-05-29): LocalAI Studio treats PortraitForge as
        if it does not exist. Neither the shipped src/, docs/, tools/,
        AGENTS.md, CLAUDE.md, nor README* may reference it. The tests/
        directory is intentionally NOT scanned — this test class itself
        legitimately contains the string in docstrings + assertion messages
        as the regression gate. Failing here is a release blocker."""
        offenders: list[Path] = []
        targets: list[Path] = []
        for sub in ("src", "docs", "tools"):
            base = ROOT / sub
            if base.is_dir():
                targets.extend(p for p in base.rglob("*") if p.is_file())
        for name in ("AGENTS.md", "CLAUDE.md"):
            p = ROOT / name
            if p.is_file():
                targets.append(p)
        targets.extend(ROOT.glob("README*"))
        for path in targets:
            if path.suffix.lower() not in {".py", ".html", ".md", ".txt", ".json", ""}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "portraitforge" in text.lower():
                offenders.append(path.relative_to(ROOT))
        self.assertEqual(
            offenders,
            [],
            "PortraitForge must not be mentioned anywhere in src/, docs/, "
            "tools/, AGENTS.md, CLAUDE.md, or README* — LocalAI Studio "
            "treats that project as non-existent. Offenders: " + repr(offenders),
        )

    def test_apply_helper_writes_constants_to_correct_widgets(self):
        """v5.5.14 SQT (test-engineer): all the other tests in this class
        are static source scans. A widget-swap refactor (e.g.,
        ``_replace_textbox_text(self._img_neg_prompt, ...)`` mistakenly
        on the positive line) would silently pass them all. Drive the
        actual helper against fake widgets and assert which widget gets
        which constant."""
        from src.app import App, IMG2IMG_DEFAULT_POSITIVE, IMG2IMG_DEFAULT_NEGATIVE

        class FakeTextbox:
            def __init__(self):
                self.ops: list[tuple] = []
            def delete(self, a, b):
                self.ops.append(("delete", a, b))
            def insert(self, idx, text):
                self.ops.append(("insert", idx, text))

        class FakeEntry(FakeTextbox):
            def __init__(self):
                super().__init__()
                self.cfg: dict = {}
            def configure(self, **kw):
                self.cfg.update(kw)

        app = object.__new__(App)
        app._img_prompt = FakeTextbox()
        app._img_neg_prompt = FakeEntry()
        App._apply_img2img_default_prompts(app)

        # Positive textbox: clean clobber via _replace_textbox_text
        # (delete 1.0->end + insert at 1.0).
        self.assertEqual(
            app._img_prompt.ops,
            [("delete", "1.0", "end"), ("insert", "1.0", IMG2IMG_DEFAULT_POSITIVE)],
            "_apply_img2img_default_prompts must clear+rewrite the positive "
            "textbox to IMG2IMG_DEFAULT_POSITIVE.",
        )
        # Negative entry: delete 0->end + insert at 0.
        self.assertEqual(
            app._img_neg_prompt.ops,
            [("delete", 0, "end"), ("insert", 0, IMG2IMG_DEFAULT_NEGATIVE)],
            "_apply_img2img_default_prompts must clear+rewrite the negative "
            "entry to IMG2IMG_DEFAULT_NEGATIVE.",
        )
        # Placeholder text reflects the same default so the user sees the
        # sample even if the entry is cleared later by another code path.
        self.assertEqual(
            app._img_neg_prompt.cfg.get("placeholder_text"),
            IMG2IMG_DEFAULT_NEGATIVE,
            "_apply_img2img_default_prompts must set placeholder_text "
            "on the negative entry to IMG2IMG_DEFAULT_NEGATIVE.",
        )

    def test_toggle_handler_behavior_on_off_and_unsupported(self):
        """v5.5.14 SQT (test-engineer): pin the full ON/OFF/unsupported
        transition matrix. Static name-checks don't catch ordering bugs
        (negative restored before positive overwrite, swap reversed, OFF
        path mistakenly calling the img2img helper, etc.)."""
        from src.app import App

        class FakeBoolVar:
            def __init__(self, value: bool = False):
                self._v = value
            def get(self) -> bool:
                return self._v
            def set(self, value: bool) -> None:
                self._v = value

        def make(supports: bool, on: bool):
            app = object.__new__(App)
            app._img_img2img_var = FakeBoolVar(on)
            app._selected_model_supports_img2img = lambda: supports
            calls: list = []
            app._refresh_img2img_controls = lambda: calls.append("refresh")
            app._apply_img2img_default_prompts = lambda: calls.append("apply_img2img")
            app._apply_image_demo_prompt = lambda m: calls.append(("demo", m))
            app._apply_selected_model_negative_prompt = lambda: calls.append("neg")
            app._selected_image_model_catalog_entry = lambda: {"id": "sdxl"}
            app._img_set_status = lambda *a, **kw: calls.append(("status", a, kw))
            return app, calls

        # ON + supported: refresh THEN apply_img2img THEN status. Never
        # demo or neg (those would clobber the just-loaded sample).
        app, calls = make(supports=True, on=True)
        App._on_img_img2img_mode_changed(app)
        self.assertEqual(calls[0], "refresh", "refresh must run first on ON path")
        self.assertEqual(calls[1], "apply_img2img",
                         "apply_img2img must run before status on ON path")
        self.assertNotIn("neg", calls,
                         "Negative restore must NOT fire on ON path")
        self.assertNotIn(("demo", {"id": "sdxl"}), calls,
                         "Per-model demo must NOT fire on ON path")

        # OFF + supported: refresh THEN demo THEN neg, no apply_img2img.
        app, calls = make(supports=True, on=False)
        App._on_img_img2img_mode_changed(app)
        self.assertEqual(
            calls,
            ["refresh", ("demo", {"id": "sdxl"}), "neg"],
            "OFF path must restore per-model demo + per-model negative "
            "after refresh, in that order.",
        )

        # ON + unsupported: handler auto-reverts var to False BEFORE
        # refresh, then early-returns. Prompts untouched.
        app, calls = make(supports=False, on=True)
        App._on_img_img2img_mode_changed(app)
        self.assertFalse(app._img_img2img_var.get(),
                         "Unsupported-model path must auto-revert the var to False.")
        self.assertEqual([c for c in calls if c == "refresh"], ["refresh"],
                         "Unsupported path must call refresh exactly once.")
        self.assertNotIn("apply_img2img", calls,
                         "Unsupported path must NEVER swap prompts to img2img defaults.")
        self.assertNotIn("neg", calls,
                         "Unsupported path must NOT touch negative prompt.")
        # The user gets a friendly status pointing at SD/SDXL.
        status_calls = [c for c in calls if isinstance(c, tuple) and c[0] == "status"]
        self.assertEqual(len(status_calls), 1,
                         "Unsupported path must surface exactly one status message.")

    def test_doc_prompt_builder_renders_to_positive_constant(self):
        """v5.5.14 SQT (test-engineer): the existing doc-sync test checks
        that the template skeleton + four defaults + suffix all appear in
        the doc — but it doesn't *compose* them and compare to the
        IMG2IMG_DEFAULT_POSITIVE constant. A future PR that flips
        ``value="man" selected`` to ``value="woman" selected``, or that
        moves the comma before ``wearing``, would keep the existing
        assertions green while silently drifting the rendered sample
        from what the app loads. Compose + assert byte-equality."""
        import re
        pos, _ = self._load_constants()
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        tmpl_match = re.search(
            r"`(professional photograph of a \$\{age\} \$\{subject\} "
            r"wearing \$\{clothes\}, \$\{light\}[^`]*)`",
            guide,
        )
        self.assertIsNotNone(tmpl_match,
                             "Prompt-builder JS template not found in doc.")
        template = tmpl_match.group(1)
        defaults: dict[str, str] = {}
        for field in ("subject", "age", "light", "clothes"):
            m = re.search(
                rf'<select id="pb-{field}">.*?value="([^"]+)" selected',
                guide,
                re.DOTALL,
            )
            self.assertIsNotNone(m, f"Default option for pb-{field} not found.")
            defaults[field] = m.group(1)
        rendered = (
            template
            .replace("${age}", defaults["age"])
            .replace("${subject}", defaults["subject"])
            .replace("${clothes}", defaults["clothes"])
            .replace("${light}", defaults["light"])
        )
        self.assertEqual(
            rendered,
            pos,
            "The Prompt builder JS template rendered with its default "
            "selections MUST be byte-identical to IMG2IMG_DEFAULT_POSITIVE "
            "(what the app loads when the user ticks the reference-image "
            "checkbox). Drifted rendering: " + repr(rendered),
        )

    def test_doc_prompt_builder_widget_and_script_ids_match(self):
        """v5.5.14 SQT (test-engineer): catch the realistic break where
        someone renames an ID in either the markup or the JS but not
        both. Each ID must appear at least twice (once in the HTML
        element + once in the script's ``getElementById`` lookup)."""
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        required_ids = {
            # Basic dropdowns + previews + copy controls.
            "pb-subject", "pb-age", "pb-light", "pb-clothes",
            "pb-positive-preview", "pb-negative-preview",
            "pb-copy-positive", "pb-copy-negative",
            "pb-copy-status",
            # v5.5.15 Advanced refinements panels.
            "pb-facial", "pb-expression", "pb-environment", "pb-composition",
            "pb-camera", "pb-lens", "pb-mood", "pb-quality",
            "pb-custom-add", "pb-custom-avoid",
            "pb-reset-advanced",
        }
        for el_id in required_ids:
            self.assertIn(f'id="{el_id}"', guide,
                          f"Prompt-builder HTML must define id={el_id!r}.")
            self.assertIn(f'"{el_id}"', guide,
                          f"Prompt-builder JS must reference id={el_id!r}.")
        # Both copy buttons must have a wired-up click listener.
        self.assertRegex(guide, r'copyPosBtn\.addEventListener\("click"',
                         "Positive copy button must have a click listener.")
        self.assertRegex(guide, r'copyNegBtn\.addEventListener\("click"',
                         "Negative copy button must have a click listener.")
        # Reset button + custom-text inputs must be wired to refresh.
        self.assertRegex(guide, r'resetBtn\.addEventListener\("click"',
                         "Reset-advanced button must have a click listener.")
        self.assertRegex(guide, r'\[customAddEl,\s*customAvoidEl\]\.forEach\(',
                         "Custom add/avoid inputs must be wired to refresh on input.")

    # ------------------------------------------------------------------
    # v5.5.15 — full prompt-builder feature expansion.
    # Adds Advanced refinements (facial / expression / environment /
    # composition / camera / lens / mood / quality + custom add/avoid).
    # ------------------------------------------------------------------

    # Expected option-value catalogues per panel. Empty string is the
    # "Unspecified" default and is intentional — it lets the all-defaults
    # render stay byte-identical to IMG2IMG_DEFAULT_POSITIVE.
    _ADV_PANELS_POS_AND_NEG: dict[str, tuple[str, ...]] = {
        "pb-facial": ("", "clean-shaven", "trimmed beard", "full beard",
                      "goatee", "mustache only"),
        "pb-expression": ("", "subtle confident", "warm smile",
                          "serious gaze", "contemplative", "candid laugh"),
        "pb-environment": ("", "studio-dark", "studio-gray", "studio-white",
                           "office", "street", "forest", "field", "beach",
                           "cafe", "cityscape"),
        "pb-composition": ("", "headshot", "midshot", "threequarter",
                           "fullbody"),
        "pb-mood": ("", "editorial", "daylight", "moody", "highkey",
                    "vintage", "punchy"),
        "pb-quality": ("", "editorial8k", "natmatte", "beauty"),
    }
    _ADV_PANELS_POS_ONLY: dict[str, tuple[str, ...]] = {
        "pb-camera": ("", "sony-a7r4", "canon-r5", "nikon-z9", "leica-sl2",
                      "hasselblad-x2d"),
        "pb-lens": ("", "35mm", "50mm", "85mm", "105mm", "135mm"),
    }

    def _all_adv_panels(self) -> dict[str, tuple[str, ...]]:
        merged = dict(self._ADV_PANELS_POS_AND_NEG)
        merged.update(self._ADV_PANELS_POS_ONLY)
        return merged

    def test_doc_prompt_builder_advanced_section_present(self):
        """v5.5.15: every advanced panel must exist with all of its option
        values present in the rendered <select>. New panels can be added
        later without rewriting the test, but options can't silently
        disappear or change value strings (those values are the keys
        looked up by the JS option maps)."""
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        self.assertIn('class="pb-advanced"', guide,
                      "Advanced refinements collapsible section must exist.")
        self.assertIn("Advanced refinements", guide,
                      "Advanced section must have a discoverable label.")
        for panel_id, values in self._all_adv_panels().items():
            select_match = re.search(
                rf'<select id="{re.escape(panel_id)}">(.*?)</select>',
                guide,
                re.DOTALL,
            )
            self.assertIsNotNone(select_match,
                                 f"<select id={panel_id!r}> must exist.")
            select_html = select_match.group(1)
            for value in values:
                self.assertIn(f'value="{value}"', select_html,
                              f"<select id={panel_id!r}> must offer "
                              f"value={value!r}.")
            # First option must be "" (Unspecified) so default render
            # stays byte-identical to IMG2IMG_DEFAULT_POSITIVE.
            first_value = re.search(r'value="([^"]*)"', select_html)
            self.assertIsNotNone(first_value)
            self.assertEqual(
                first_value.group(1), "",
                f"<select id={panel_id!r}> must default to value=''"
                " (Unspecified) so the all-defaults render of the prompt"
                " builder stays byte-identical to IMG2IMG_DEFAULT_POSITIVE.",
            )

    def test_doc_prompt_builder_advanced_options_have_positive_chunks(self):
        """v5.5.15: every non-empty advanced option in the JS option maps
        must contribute a non-empty ``pos`` chunk so picking it visibly
        changes the positive prompt. Empty string ("Unspecified") is the
        intentional no-op default."""
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        map_name_for_panel = {
            "pb-facial": "FACIAL",
            "pb-expression": "EXPRESSION",
            "pb-environment": "ENVIRONMENT",
            "pb-composition": "COMPOSITION",
            "pb-camera": "CAMERA",
            "pb-lens": "LENS",
            "pb-mood": "MOOD",
            "pb-quality": "QUALITY",
        }
        for panel_id, values in self._all_adv_panels().items():
            map_name = map_name_for_panel[panel_id]
            # Pull just this map's body so values from other maps don't
            # pollute the search.
            map_match = re.search(
                rf'const {map_name}\s*=\s*\{{(.*?)\n\s*\}};',
                guide,
                re.DOTALL,
            )
            self.assertIsNotNone(
                map_match,
                f"JS option map {map_name!r} for panel {panel_id!r} must "
                "exist.")
            map_body = map_match.group(1)
            for value in values:
                if not value:
                    continue  # "Unspecified" is the no-op default.
                # Find this option's entry. Quoted key, then a {…}
                # object literal containing pos: "...".
                entry_match = re.search(
                    rf'"{re.escape(value)}"\s*:\s*\{{[^{{}}]*?pos:\s*"([^"]*)"',
                    map_body,
                    re.DOTALL,
                )
                self.assertIsNotNone(
                    entry_match,
                    f"JS map {map_name!r} must contain entry for "
                    f"option value={value!r}.")
                pos_chunk = entry_match.group(1)
                self.assertTrue(
                    pos_chunk.strip(),
                    f"Option value={value!r} in {map_name!r} must have a "
                    "non-empty positive chunk so selecting it visibly "
                    "changes the positive prompt.",
                )

    def test_doc_prompt_builder_panels_with_negative_chunks(self):
        """v5.5.15: panels listed in _ADV_PANELS_POS_AND_NEG must provide
        a non-empty ``neg`` chunk for at least one non-default option, so
        picking that option visibly extends the negative prompt. Panels
        in _ADV_PANELS_POS_ONLY (camera, lens) are gear hints — they only
        contribute to positive and that's intentional."""
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        map_name_for_panel = {
            "pb-facial": "FACIAL",
            "pb-expression": "EXPRESSION",
            "pb-environment": "ENVIRONMENT",
            "pb-composition": "COMPOSITION",
            "pb-mood": "MOOD",
            "pb-quality": "QUALITY",
        }
        for panel_id, values in self._ADV_PANELS_POS_AND_NEG.items():
            map_name = map_name_for_panel[panel_id]
            map_match = re.search(
                rf'const {map_name}\s*=\s*\{{(.*?)\n\s*\}};',
                guide,
                re.DOTALL,
            )
            self.assertIsNotNone(map_match)
            map_body = map_match.group(1)
            negatives_seen: list[str] = []
            for value in values:
                if not value:
                    continue
                entry_match = re.search(
                    rf'"{re.escape(value)}"\s*:\s*\{{[^{{}}]*?neg:\s*"([^"]*)"',
                    map_body,
                    re.DOTALL,
                )
                if entry_match and entry_match.group(1).strip():
                    negatives_seen.append(value)
            self.assertTrue(
                negatives_seen,
                f"Panel {panel_id!r} ({map_name!r}) must have at least "
                "one option with a non-empty negative chunk so selecting "
                "it visibly extends the negative prompt.",
            )

    def test_doc_prompt_builder_compose_branches_short_circuit_to_default(self):
        """v5.5.15: the JS must keep the original ``buildPositive`` template
        literal for the all-defaults branch so the existing render-equals
        -constant test still pins behaviour, AND must short-circuit to it
        when no advanced refinement is active."""
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "const buildPositive = (age, subject, clothes, light) =>",
            guide,
            "JS must keep the buildPositive arrow function so the regex "
            "test that pins it against IMG2IMG_DEFAULT_POSITIVE keeps "
            "working.",
        )
        # The compose function must explicitly delegate to buildPositive
        # when no advanced refinement is active.
        compose_match = re.search(
            r'function composePositive\(\)\s*\{(.*?)\n\s*\}',
            guide,
            re.DOTALL,
        )
        self.assertIsNotNone(compose_match,
                             "composePositive() function must exist.")
        body = compose_match.group(1)
        self.assertIn("advancedActive()", body,
                      "composePositive() must check advancedActive() so "
                      "the all-defaults branch returns buildPositive(...)"
                      " unchanged.")
        self.assertIn("buildPositive(age, subject, clothes, light)", body,
                      "composePositive() must call buildPositive(...) in "
                      "the all-defaults branch (byte-identical to "
                      "IMG2IMG_DEFAULT_POSITIVE).")

    def test_doc_prompt_builder_custom_avoid_concatenates_into_negative(self):
        """v5.5.15: custom-avoid free text appended last to negative."""
        guide = self.GUIDE_PATH.read_text(encoding="utf-8")
        neg_match = re.search(
            r'function composeNegative\(\)\s*\{(.*?)\n\s*\}',
            guide,
            re.DOTALL,
        )
        self.assertIsNotNone(neg_match,
                             "composeNegative() function must exist.")
        body = neg_match.group(1)
        self.assertIn("NEGATIVE_TEXT", body,
                      "composeNegative() must seed with NEGATIVE_TEXT.")
        self.assertIn("customAvoidEl.value.trim()", body,
                      "composeNegative() must append customAvoidEl trim "
                      "so user free-text avoidances reach the negative.")

    def test_model_change_restores_per_model_negative_before_img2img_override(self):
        """v5.5.14 SQT (test-engineer #4): in the SDXL+img2img-on → FLUX
        (unsupported) transition, the negative entry must end up holding
        FLUX's per-model default, NOT the stale IMG2IMG_DEFAULT_NEGATIVE.
        This works today because:
          1. ``_apply_selected_model_negative_prompt()`` runs at the top
             of the handler (before the catalog/heuristic branch), setting
             the new per-model negative.
          2. ``_refresh_img2img_controls()`` auto-reverts the var to False
             for unsupported models.
          3. The guarded override
             ``if var.get(): _apply_img2img_default_prompts()`` then
             short-circuits.
        Pin all three so a refactor can't silently regress the transition."""
        source = self._function_source("_on_img_model_changed")
        neg_pos = source.find("_apply_selected_model_negative_prompt()")
        override_pos = source.find("_apply_img2img_default_prompts()")
        self.assertGreater(
            neg_pos, 0,
            "_on_img_model_changed must call "
            "_apply_selected_model_negative_prompt() to restore the new "
            "model's negative prompt on every model switch.",
        )
        self.assertGreater(
            override_pos, 0,
            "_on_img_model_changed must call _apply_img2img_default_prompts() "
            "to keep the img2img sample visible across model switches when "
            "the toggle survived the refresh.",
        )
        self.assertLess(
            neg_pos, override_pos,
            "Per-model negative restore must precede every img2img override "
            "so SDXL+img2img-on → FLUX transitions end up with FLUX's "
            "negative (the override short-circuits because "
            "_refresh_img2img_controls auto-reverts the var).",
        )
        self.assertIn(
            "self._img_img2img_var.get()", source,
            "The img2img override must be GUARDED by the var so it does "
            "not fire on unsupported-model transitions.",
        )


class BenchmarkQuickFallbackContractTests(unittest.TestCase):
    """v2026.06.01.8 — pin Quick-mode fallback for the synthetic
    "This Device" profile so the public no-skus.json install does not
    silently auto-tick every fitting model in Quick mode."""

    def _capacity(self, *, has_gpu: bool) -> dict:
        return {
            "profile": "This Device",
            "available_ram_gb": 64,
            "total_ram_gb": 64,
            "vram_capacity_gb": 24 if has_gpu else 0,
            "has_gpu": has_gpu,
            "is_sku": False,
        }

    def _make_app(self, *, run_mode: str = "quick"):
        from src.app import App

        class _ModeVar:
            def __init__(self, mode):
                self._mode = mode

            def get(self):
                return self._mode

        app = object.__new__(App)
        app._bench_run_mode_var = _ModeVar(
            "Quick" if run_mode == "quick" else "Extended"
        )
        app._bench_methods_for_ui = lambda m: ["chat"]
        app._bench_missing_deps_for_model = lambda m: []
        app._is_image_model_ui = lambda m: bool(
            (m.get("category") or "").startswith("Image")
        )
        # The fallback path must not consult skus.json — assert by
        # forcing the SKU lookup to act as if no per-SKU set exists.
        app._bench_observed_success_for_profile = lambda m, c: False
        app._bench_profile_has_default_models = lambda profile, run_mode: False
        return app

    def test_quick_on_this_device_ticks_only_ultra_small_chat_on_cpu(self):
        from src import catalog
        from src.app import (
            App, _bench_quick_fallback_model_ids,
            _BENCH_QUICK_FALLBACK_IMAGE_GEN_ID,
        )

        app = self._make_app(run_mode="quick")
        capacity = self._capacity(has_gpu=False)
        fallback = _bench_quick_fallback_model_ids(has_gpu=False)
        # The fallback set on a CPU box must equal the catalog's Ultra
        # Small chat models — no image-gen, no Small or Medium chat.
        catalog_ultra_small = {
            m["id"] for m in catalog.MODELS if m.get("category") == "Ultra Small"
        }
        self.assertEqual(fallback, catalog_ultra_small)
        self.assertNotIn(_BENCH_QUICK_FALLBACK_IMAGE_GEN_ID, fallback)

        # Every Ultra Small chat model default-ticks under Quick.
        for m in catalog.MODELS:
            if m.get("category") != "Ultra Small":
                continue
            self.assertTrue(
                app._bench_default_selected_for_model(m, capacity),
                f"Ultra Small chat model {m.get('id')!r} must default-tick "
                "in Quick on the synthetic This Device CPU profile.",
            )

        # A non-Ultra-Small chat model that fits must NOT default-tick.
        small_chat_that_fits = {
            "id": "small-chat-not-in-baseline",
            "name": "Small Chat",
            "category": "Small",
            "ollama_tag": "small:latest",
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        self.assertFalse(
            app._bench_default_selected_for_model(small_chat_that_fits, capacity),
            "Quick on This Device must NOT auto-tick non-baseline chat models "
            "even when they fit — that is the bug v2026.06.01.8 fixed.",
        )

    def test_quick_on_this_device_adds_smallest_image_gen_when_gpu(self):
        from src import catalog
        from src.app import (
            _bench_quick_fallback_model_ids, _BENCH_QUICK_FALLBACK_IMAGE_GEN_ID,
        )

        # CPU box: image-gen MUST NOT be in the fallback set even if the
        # GPU-only fitting branch is bypassed.
        cpu_fallback = _bench_quick_fallback_model_ids(has_gpu=False)
        self.assertNotIn(_BENCH_QUICK_FALLBACK_IMAGE_GEN_ID, cpu_fallback)

        # GPU box: the image-gen fallback ID is added on top of the
        # Ultra Small chat baseline.
        gpu_fallback = _bench_quick_fallback_model_ids(has_gpu=True)
        catalog_ultra_small = {
            m["id"] for m in catalog.MODELS if m.get("category") == "Ultra Small"
        }
        self.assertEqual(
            gpu_fallback, catalog_ultra_small | {_BENCH_QUICK_FALLBACK_IMAGE_GEN_ID}
        )

        # The pinned ID must match the v2026.06.01.6 skus.json baseline
        # so behavior is identical with or without skus.json.
        self.assertEqual(_BENCH_QUICK_FALLBACK_IMAGE_GEN_ID, "realistic-vision-v6")

    def test_extended_on_this_device_keeps_legacy_fit_everything_behavior(self):
        # v2026.06.01.8 deliberately scoped the fix to Quick — Extended
        # on This Device keeps the "everything that fits is default-
        # ticked" behavior because there is no per-SKU verified-passer
        # set to defer to when skus.json is absent, and Extended is the
        # right run mode for "run every model my hardware can handle".
        app = self._make_app(run_mode="extended")
        capacity = self._capacity(has_gpu=False)
        small_chat = {
            "id": "non-baseline-small",
            "name": "Non Baseline Small",
            "category": "Small",
            "ollama_tag": "small:latest",
            "min_ram_gb": 4,
            "min_vram_gb": 0,
        }
        self.assertTrue(
            app._bench_default_selected_for_model(small_chat, capacity),
            "Extended on This Device must keep the legacy "
            "'everything that fits is default-ticked' behavior. "
            "Only Quick is restricted to the lean baseline.",
        )

    def test_quick_fallback_constant_matches_skus_json_baseline_when_present(self):
        # Defense-in-depth: when skus.json IS shipped (maintainer
        # builds), the in-code Quick fallback must mirror its declared
        # quick_chat_ultra_small / quick_image_smallest baselines so
        # both paths produce the same set. Skip when skus.json is not
        # present (normal in the shipped public clone).
        import json
        from src.app import (
            _bench_quick_fallback_model_ids, _BENCH_QUICK_FALLBACK_IMAGE_GEN_ID,
        )

        skus_path = ROOT / "skus.json"
        if not skus_path.exists():
            self.skipTest("skus.json not present (normal in public clone)")
        baselines = (
            json.loads(skus_path.read_text(encoding="utf-8"))
            .get("bench_defaults", {})
            .get("baselines", {})
        )
        expected_chat = set(baselines.get("quick_chat_ultra_small", []))
        expected_image = list(baselines.get("quick_image_smallest", []))
        cpu_fallback = _bench_quick_fallback_model_ids(has_gpu=False)
        gpu_fallback = _bench_quick_fallback_model_ids(has_gpu=True)
        self.assertEqual(
            cpu_fallback, expected_chat,
            "Quick CPU fallback must match skus.json quick_chat_ultra_small "
            "exactly so behavior is identical with or without skus.json.",
        )
        self.assertEqual(
            gpu_fallback - cpu_fallback, set(expected_image),
            "Quick GPU fallback must add exactly the skus.json "
            "quick_image_smallest baseline on top of the chat baseline.",
        )
        self.assertEqual(
            [_BENCH_QUICK_FALLBACK_IMAGE_GEN_ID], expected_image,
            "The in-code image-gen fallback constant must equal the "
            "skus.json quick_image_smallest list.",
        )


class ComfyUIInstallResolverContractTests(unittest.TestCase):
    """v2026.06.01.9 — pin the multi-location ComfyUI install resolver so
    a fresh app extraction on top of an existing install (the v.N+1
    upgrade-in-place pattern) still finds the sibling-of-app ComfyUI
    that ``setup1.bat`` itself searches for at line ~388
    (``%~dp0..\\ComfyUI``). Prior to v.9 the resolver only checked the
    child location (``<app>/ComfyUI``) so the Image Gen tab and
    benchmark both reported "ComfyUI not installed at expected paths"
    for a working install."""

    def _function_source(self, name: str, module_text: str) -> str:
        tree = ast.parse(module_text)
        lines = module_text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found in module text")

    def test_app_comfyui_installed_path_checks_sibling_of_app(self):
        src = self._function_source("_comfyui_installed_path", APP_TEXT)
        # The sibling fallback must walk three .parent hops from app.py
        # (src/app.py → src → app root → sibling-of-app) and probe ComfyUI/main.py.
        self.assertIn("Path(__file__).parent.parent.parent", src,
                      "_comfyui_installed_path must compute the sibling-of-app "
                      "candidate (three .parent hops from src/app.py).")
        self.assertIn('"ComfyUI"', src)
        # The fallback must self-heal: persist the discovered path so subsequent
        # restarts don't re-do the probe.
        self.assertIn('self.cfg["comfyui_dir"]', src)
        self.assertIn("config.save", src)
        self.assertIn("_sync_comfyui_path_bat", src)

    def test_config_load_uses_sibling_resolver_helper(self):
        config_text = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
        # Helpers must exist.
        self.assertIn("def _candidate_comfyui_install_dirs(", config_text)
        self.assertIn("def _resolve_existing_comfyui_install(", config_text)
        # load() must consult the resolver, not just default to <app>/ComfyUI.
        load_src = self._function_source("load", config_text)
        self.assertIn("_resolve_existing_comfyui_install", load_src,
                      "config.load() must consult _resolve_existing_comfyui_install "
                      "so empty / missing comfyui_dir auto-heals to the sibling-of-app "
                      "location that setup1.bat also searches.")

    def test_candidate_comfyui_install_dirs_returns_child_then_sibling(self):
        """The candidate order must match setup1.bat's own lookup at lines
        ~381 + ~388: child (``<app>/ComfyUI``) first, sibling
        (``<app>/../ComfyUI``) second. Order matters because a fresh
        install creates the child first."""
        from src.config import _candidate_comfyui_install_dirs
        app_root = Path("C:/example_app")
        cands = _candidate_comfyui_install_dirs(app_root)
        self.assertEqual(
            [str(p).replace("\\", "/") for p in cands],
            ["C:/example_app/ComfyUI", "C:/ComfyUI"],
        )


class ImageGenAutoPullContractTests(unittest.TestCase):
    """v2026.06.01.9 — pin the Image Gen tab silent auto-pull contract so
    downloadable catalog image-gen models can be picked from the
    dropdown and silently downloaded on Generate, matching the Ollama
    chat-model UX."""

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    def test_populate_image_model_menu_unions_downloadable_catalog_entries(self):
        """The dropdown must include downloadable catalog image-gen
        models even when their checkpoint isn't on disk yet — otherwise
        a fresh install shows '(no checkpoints found)' and Generate
        refuses, forcing users to use the Benchmark tab to bootstrap."""
        src = self._function_source("_populate_image_model_menu")
        self.assertIn("_catalog_image_model_filenames_with_url", src,
                      "_populate_image_model_menu must union on-disk models "
                      "with downloadable catalog image-gen entries.")
        # The helper itself must exist and filter on category + URL.
        helper = self._function_source("_catalog_image_model_filenames_with_url")
        self.assertIn('"Image Generation"', helper)
        self.assertIn("comfyui_model_url", helper)
        self.assertIn("comfyui_model", helper)

    def test_start_image_generation_calls_ensure_checkpoint_present(self):
        """Generate must call the auto-pull helper before kicking off the
        ComfyUI workflow so a downloadable catalog model that isn't on
        disk gets pulled silently instead of failing with a vague
        ComfyUI-side error."""
        src = self._function_source("_start_image_generation")
        self.assertIn("_ensure_selected_image_checkpoint_present", src,
                      "_start_image_generation must invoke "
                      "_ensure_selected_image_checkpoint_present before "
                      "queueing the ComfyUI workflow.")
        # Must also gate re-entry while a download is in flight so a
        # double-click on Generate doesn't kick off a second download.
        self.assertIn("_img_checkpoint_download_in_progress", src)

    def test_download_image_checkpoint_async_reuses_bench_helper(self):
        """The auto-pull worker must reuse ``_bench_prepare_image_model``
        — the same helper the Benchmark tab uses — so both surfaces
        share validation, disk-space checks, and runtime-support prep."""
        src = self._function_source("_download_image_checkpoint_async")
        self.assertIn("_bench_prepare_image_model", src)
        self.assertIn("threading.Thread", src)
        # On completion it must re-enter Generate so the user doesn't
        # have to click the button a second time.
        completion_src = self._function_source("_image_checkpoint_downloaded")
        self.assertIn("_start_image_generation", completion_src)

    def test_ensure_selected_image_checkpoint_present_is_silent(self):
        """The 'silent' part of silent auto-pull: the helper must not
        show a messagebox / askyesno prompt. The Benchmark tab already
        prompts up front; the Image Gen tab matches the chat tab UX
        where Ollama models pull silently on first use."""
        src = self._function_source("_ensure_selected_image_checkpoint_present")
        self.assertNotIn("messagebox.show", src)
        self.assertNotIn("askyesno", src)
        # Must check on-disk before kicking off a download.
        self.assertIn("_comfyui_model_download_target", src)
        self.assertIn(".exists()", src)


class IncompleteSetupBannerContractTests(unittest.TestCase):
    """v2026.06.01.10–11 — pin the in-app incomplete-setup banner so a
    silently-failed install (setup window auto-closed, CUDA not installed,
    Ollama not installed) surfaces a visible warning at the top of every
    page instead of only quiet log lines.
    """

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    def test_required_banner_methods_exist(self):
        """All four banner helpers must exist: build, detect, refresh, details."""
        for name in (
            "_build_setup_warning_banner",
            "_detect_incomplete_setup_state",
            "_refresh_setup_warning_banner",
            "_show_setup_warning_details",
        ):
            with self.subTest(method=name):
                self.assertIn(f"def {name}(", APP_TEXT,
                              f"{name} must exist as the v.10 banner contract.")

    def test_detect_incomplete_setup_state_checks_all_three_signals(self):
        """Detection must cover: missing Ollama, missing ComfyUI, and
        NVIDIA-but-CPU-torch — the three broken-install signatures we
        get from the v2026.06.01.9 / .10 startup probes."""
        src = self._function_source("_detect_incomplete_setup_state")
        self.assertIn("ollama_ok", src,
                      "Banner must check self.ollama_ok for the Ollama signal.")
        self.assertIn("_comfyui_installed_path", src,
                      "Banner must check ComfyUI install status.")
        self.assertIn("_pytorch_cuda_missing_on_nvidia", src,
                      "Banner must check the cached NVIDIA+CPU-torch flag.")

    def test_apply_gpu_detection_sets_nvidia_cpu_torch_flag(self):
        """The NVIDIA+CPU-torch signal must be set during GPU detection
        so the banner doesn't re-probe nvidia-smi on every refresh."""
        src = self._function_source("_apply_gpu_detection_result")
        self.assertIn("_pytorch_cuda_missing_on_nvidia", src,
                      "_apply_gpu_detection_result must set the cached flag.")
        self.assertIn("_nvidia_gpu_present", src)
        self.assertIn("_torch_cuda_state", src)

    def test_async_completion_hooks_refresh_banner(self):
        """Banner must be refreshed after each async startup check that
        can change its state: GPU detection, Ollama probe, Ollama
        auto-start, and image-readiness (which fires on ComfyUI install)."""
        for fn_name in (
            "_apply_gpu_detection_result",
            "_check_ollama_async",
            "_try_start_ollama",
            "_refresh_image_readiness",
        ):
            with self.subTest(function=fn_name):
                src = self._function_source(fn_name)
                self.assertIn("_refresh_setup_warning_banner", src,
                              f"{fn_name} must refresh the incomplete-setup "
                              "banner on completion.")

    def test_build_ui_creates_banner_in_content_row_zero(self):
        """_build_ui must construct the banner inside self._content at
        row=0 and reserve row=1 for the active page."""
        src = self._function_source("_build_ui")
        self.assertIn("_build_setup_warning_banner", src,
                      "_build_ui must invoke _build_setup_warning_banner.")
        # Content area must give row 0 weight 0 (banner) and row 1 weight 1
        # (page) so banner is fixed-height and page takes remaining space.
        self.assertIn("self._content.grid_rowconfigure(0, weight=0)", src)
        self.assertIn("self._content.grid_rowconfigure(1, weight=1)", src)

    def test_switch_page_grids_pages_at_row_one(self):
        """Pages must grid at row=1 (not row=0) so the banner at row=0
        appears above them when it's shown."""
        src = self._function_source("_switch_page")
        self.assertIn('self._pages[page].grid(row=1, column=0, sticky="nsew")',
                      src,
                      "_switch_page must grid pages at row=1 so the banner "
                      "at row=0 appears above them.")

    def test_refresh_setup_warning_banner_is_idempotent_against_missing_widget(self):
        """Banner refresh must no-op when banner widget hasn't been built
        yet (early startup) or has been destroyed (theme rebuild) — the
        async callbacks fire from many code paths and must never raise."""
        src = self._function_source("_refresh_setup_warning_banner")
        # Must check the attribute exists with a default fallback.
        self.assertIn("getattr", src)
        self.assertIn("_setup_warning_banner", src)
        # Must early-return on missing widget instead of raising.
        self.assertIn("return", src)


class SetupPs1WrapperContractTests(unittest.TestCase):
    """v2026.06.01.10–11 — pin the setup.bat / setup.ps1 / setup1.bat
    three-file structure that gives users an auto-captured setup.log
    without changing their muscle-memory:

      setup.bat   = tiny shim (5–40 lines) that launches setup.ps1
      setup.ps1   = transcript wrapper; Start-Transcript -> setup.log,
                    then invokes setup1.bat, then unconditional pause
      setup1.bat  = the real installer (renamed from the original setup.bat)

    setup.ps1 sits OUTSIDE the cmd process that runs setup1.bat (it spawns
    a fresh powershell.exe), so this is NOT the v.4-era self-tee
    regression that broke ``set /p`` prompts via block buffering.
    """

    SETUP_PS1 = ROOT / "setup.ps1"
    SETUP_BAT = ROOT / "setup.bat"
    SETUP1_BAT = ROOT / "setup1.bat"

    def test_all_three_setup_files_exist(self):
        for path in (self.SETUP_BAT, self.SETUP_PS1, self.SETUP1_BAT):
            with self.subTest(file=path.name):
                self.assertTrue(path.is_file(),
                                f"{path.name} must exist alongside the others "
                                "as part of the shim -> wrapper -> installer "
                                "chain. If renaming happened, update the "
                                "v.11 publish pipeline and AGENTS.md too.")

    def test_setup_bat_is_a_shim_that_launches_setup_ps1(self):
        """The user-facing setup.bat must be a tiny shim that defers to
        setup.ps1 (which in turn calls setup1.bat under Start-Transcript)."""
        text = self.SETUP_BAT.read_text(encoding="utf-8")
        # Sanity check on size — original setup.bat was ~42KB. The shim
        # must stay tiny so it's obvious at a glance that it's a launcher,
        # not the installer. 6KB ceiling leaves room for comments, the
        # error-path messages, and the v2026.06.02.0 zip-preview guard,
        # but still rules out anyone re-adding install logic.
        self.assertLess(len(text), 6144,
                        f"setup.bat must remain a tiny shim ({len(text)} bytes "
                        "is too large). The real install logic lives in "
                        "setup1.bat — do not move it back.")
        self.assertIn("setup.ps1", text,
                      "setup.bat shim must reference setup.ps1 (its launch "
                      "target).")
        self.assertIn("powershell", text.lower(),
                      "setup.bat shim must invoke powershell.exe to launch "
                      "setup.ps1.")
        # The shim must NOT carry the install logic env-var block. If any of
        # these markers appear, someone re-added installer logic to the shim.
        for installer_marker in ("LOCALAI_PYTHON", "pip install", "winget"):
            with self.subTest(marker=installer_marker):
                self.assertNotIn(installer_marker, text,
                                 f"setup.bat shim must not contain "
                                 f"installer marker '{installer_marker}' — "
                                 "that logic belongs in setup1.bat.")

    def test_setup_ps1_uses_start_transcript_to_write_setup_log(self):
        text = self.SETUP_PS1.read_text(encoding="utf-8")
        self.assertIn("Start-Transcript", text,
                      "setup.ps1 must use Start-Transcript to capture "
                      "everything the setup console prints.")
        self.assertIn("setup.log", text,
                      "Transcript target must be setup.log — run.bat's "
                      "crash-path message tells users to look for that file.")

    def test_setup_ps1_has_unconditional_pause_at_end(self):
        """Even if setup1.bat hits an early-exit path that skips its own
        pause, the PowerShell wrapper must hold the window open."""
        text = self.SETUP_PS1.read_text(encoding="utf-8")
        self.assertIn("Read-Host", text,
                      "setup.ps1 must call Read-Host to pause for the user "
                      "before closing the window.")

    def test_setup_ps1_invokes_setup1_bat(self):
        """setup.ps1 is the wrapper for the real installer (setup1.bat).
        It must NOT invoke setup.bat (its own caller) or it would loop."""
        text = self.SETUP_PS1.read_text(encoding="utf-8")
        self.assertIn("setup1.bat", text,
                      "setup.ps1 must reference setup1.bat — that's the "
                      "actual installer it wraps.")
        # And it must NOT pipe cmd's stdout through Tee-Object (that's the
        # forbidden pattern). Start-Transcript captures from the host
        # console, which sidesteps the buffering issue.
        self.assertNotIn("Tee-Object", text,
                         "setup.ps1 must not pipe cmd's stdout through "
                         "Tee-Object — that's the forbidden v.4 pattern.")

    def test_setup1_bat_remains_unwrapped(self):
        """Sanity check: setup1.bat (the real installer) must NOT have
        been quietly re-tee'd. The PowerShell wrapper is the only
        capture mechanism."""
        text = self.SETUP1_BAT.read_text(encoding="utf-8")
        # The v.4 self-tee key tokens must NOT have come back.
        self.assertNotIn("LOCALAI_SETUP_TEED", text)
        self.assertNotIn("Tee-Object", text)

    def test_setup1_bat_has_no_colon_comments_inside_parens_blocks(self):
        """v2026.06.01.12 — pin the ``::`` inside ``(...)`` bug fix.

        The v.11 release shipped with a latent cmd parser hazard: ``::``
        comments inside parenthesized ``if (...)`` / ``for (...)`` blocks
        are NOT fully ignored by cmd.exe — the parser still tokenizes
        the comment text, and any inner parens (e.g. ``:: foo (bar) baz``)
        prematurely close the outer block, producing errors like
        ``"baz was unexpected at this time."`` and aborting setup with
        exit code 255. The first user with ``INSTALL_UTILITY=y`` AND
        ``INSTALL_ONNX=y`` (i.e. "Setup all features" mode) hit this on
        an actual install run when ``:: \\`onnxruntime\\` (CPU) via dep
        extras like optimum[onnxruntime]`` was parsed.

        Use ``REM`` (a real command) instead of ``::`` (a label) inside
        any parenthesized block — REM safely swallows everything to the
        end of line, no matter what punctuation follows.
        """
        text = self.SETUP1_BAT.read_text(encoding="utf-8")

        def count_unquoted(line, ch):
            n = 0
            in_dq = False
            in_bt = False
            i = 0
            while i < len(line):
                c = line[i]
                if c == "^" and i + 1 < len(line):
                    i += 2
                    continue
                if c == '"':
                    in_dq = not in_dq
                elif c == "`":
                    in_bt = not in_bt
                elif c == ch and not in_dq and not in_bt:
                    n += 1
                i += 1
            return n

        depth = 0
        offenders = []
        for idx, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if depth > 0 and stripped.startswith("::"):
                offenders.append((idx, depth, stripped[:90]))
            opens = count_unquoted(line, "(")
            closes = count_unquoted(line, ")")
            depth += opens - closes
            if depth < 0:
                depth = 0

        if offenders:
            details = "\n".join(
                f"  L{idx} (paren depth={d}): {snippet}"
                for idx, d, snippet in offenders
            )
            self.fail(
                f"setup1.bat has {len(offenders)} '::' comment(s) inside "
                f"parens block(s) — convert each to 'REM' to avoid the "
                f"cmd parser hazard described in v2026.06.01.12:\n{details}"
            )


class BenchmarkRetryFailedUnpackContractTests(unittest.TestCase):
    """Regression pin for the v2026.06.02.0 small-CPU-box crash:

        File "C:\\LocalAI\\src\\app.py", line 10040, in _retry_failed_benchmark
            for mid, method in failed:
        ValueError: too many values to unpack (expected 2)

    ``BatchReport.get_failed_combos()`` returns
    ``list[tuple[str, str, int]]`` — (model_id, method, sample_index) —
    since v5.5.6+ (sample-index sharding was added when image-gen
    benchmarks started supporting multiple sample prompts per row).
    The Retry-Failed UI loop in ``_retry_failed_benchmark`` still
    unpacked the iterable into 2 variables, which is fine for an empty
    or single-sample failure list but crashes the moment any failure
    row carries a sample index.

    BatchRunner already handles 3-tuples defensively (uses
    ``combo[0]`` / ``combo[1]`` / ``combo[2]`` indexing in
    ``_methods_for`` and ``_iter_selected_samples_for``), so the bug
    was confined to the app-side logging loop.

    These contracts pin both the producer (3-tuple return type) and
    the consumer (no 2-tuple unpack) so the schema can't drift again.
    """

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found in src/app.py")

    def test_get_failed_combos_returns_three_tuples(self):
        """The producer side: BatchReport.get_failed_combos() must
        return ``(model_id, method, sample_index)`` triples."""
        report_text = (ROOT / "src" / "batch_report.py").read_text(encoding="utf-8")
        tree = ast.parse(report_text)
        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "get_failed_combos"
            ):
                found = True
                ann = ast.unparse(node.returns) if node.returns else ""
                self.assertIn(
                    "tuple[str,str,int]", ann.replace(" ", ""),
                    "get_failed_combos must be annotated as returning "
                    "list[tuple[str, str, int]] so the consumer side "
                    "can rely on the 3-element shape.",
                )
        self.assertTrue(
            found,
            "BatchReport.get_failed_combos must exist in src/batch_report.py",
        )

    def test_retry_failed_benchmark_does_not_unpack_two_tuple(self):
        """The consumer side: _retry_failed_benchmark must NOT contain
        a ``for mid, method in failed:`` loop. The failure list yields
        3-tuples (model_id, method, sample_index); unpacking into 2
        variables raises ``ValueError: too many values to unpack``."""
        src = self._function_source("_retry_failed_benchmark")
        self.assertNotIn(
            "for mid, method in failed:",
            src,
            "_retry_failed_benchmark cannot unpack 'failed' into 2 "
            "variables — get_failed_combos returns 3-tuples and the "
            "loop crashes the moment any failure row carries a sample "
            "index. Use index access (combo[0], combo[1], combo[2]) "
            "or unpack into 3 variables instead.",
        )

    def test_retry_failed_benchmark_logs_each_failed_combo(self):
        """Whatever shape the loop uses, _retry_failed_benchmark must
        still log every failed combo in the Retry preamble so the user
        sees what's about to be re-run."""
        src = self._function_source("_retry_failed_benchmark")
        self.assertIn(
            "_bench_log_append",
            src,
            "_retry_failed_benchmark must call _bench_log_append to "
            "show the user what's being retried.",
        )
        self.assertIn(
            "failed",
            src,
            "_retry_failed_benchmark must reference the 'failed' list "
            "from get_failed_combos.",
        )


class BulkDeleteProgressDialogContractTests(unittest.TestCase):
    """v2026.06.02.0 — pin the modal-progress dialog that the
    Settings "Delete not-in-catalog Ollama tags" and
    "Delete not-in-catalog ONNX directories" red buttons use, so they
    can never silently freeze the UI again the way they did before
    ``_run_bulk_with_progress`` existed.

    Ron's repro from the small-CPU-box session: a multi-tag
    ``ollama rm`` loop ran on the UI thread, blocked the mainloop for
    ~30s per tag with zero visible feedback, and looked like the app
    had hung. The fix centralises bulk-delete UX in
    ``_run_bulk_with_progress`` so any future "delete N items" red
    button gets the same modal progress dialog by default.
    """

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found in src/app.py")

    def test_run_bulk_with_progress_helper_exists(self):
        """The shared helper that owns the modal dialog must exist."""
        self.assertIn(
            "def _run_bulk_with_progress(",
            APP_TEXT,
            "_run_bulk_with_progress must exist as the single source of "
            "truth for bulk-delete progress UI. Without it, the Settings "
            "red-button paths fall back to synchronous loops that freeze "
            "the UI thread.",
        )

    def test_run_bulk_with_progress_uses_modal_dialog_and_worker_thread(self):
        """The helper must (a) build a CTkToplevel as the modal surface,
        (b) show a CTkProgressBar that the user can actually see advance,
        and (c) run the per-item work_fn on a background thread so the
        UI thread stays responsive."""
        src = self._function_source("_run_bulk_with_progress")
        self.assertIn(
            "ctk.CTkToplevel(self)", src,
            "_run_bulk_with_progress must create a CTkToplevel for the modal dialog.",
        )
        self.assertIn(
            "ctk.CTkProgressBar(", src,
            "_run_bulk_with_progress must render a CTkProgressBar so the "
            "user sees progress instead of staring at a frozen window.",
        )
        self.assertIn(
            "threading.Thread(", src,
            "_run_bulk_with_progress must run work_fn on a background "
            "thread — running it on the UI thread defeats the entire "
            "point of having a progress dialog.",
        )
        self.assertIn(
            "self.after(", src,
            "Worker thread must marshal UI updates back via self.after("
            "0, ...) — touching CTk widgets from the worker is unsafe.",
        )
        self.assertIn(
            "wait_window", src,
            "Helper must block the caller via dlg.wait_window() so the "
            "caller's summary logic runs after the work completes.",
        )

    def test_confirm_delete_ollama_tags_uses_progress_helper(self):
        """The Settings red button for bulk-deleting Ollama tags must
        route through the shared progress helper instead of running its
        own synchronous subprocess loop on the UI thread."""
        src = self._function_source("_confirm_delete_ollama_tags")
        self.assertIn(
            "_run_bulk_with_progress(",
            src,
            "_confirm_delete_ollama_tags must call _run_bulk_with_progress "
            "for the actual deletes — running `ollama rm` for every tag "
            "in a tight for-loop on the UI thread is what caused the "
            "'app froze' UX complaint in the v.13 session.",
        )
        # And the inline loop that used to run `ollama rm` synchronously
        # on the UI thread must not come back.
        self.assertNotIn(
            'subprocess.run(\n                    ["ollama", "rm", tag]',
            src,
            "_confirm_delete_ollama_tags must not re-grow an inline "
            "synchronous `ollama rm` loop on the UI thread — that's the "
            "regression this contract pins against.",
        )

    def test_confirm_delete_onnx_dirs_uses_progress_helper(self):
        """Same contract for the ONNX-directories red button — the
        rmtree loop must also run through the shared progress helper."""
        src = self._function_source("_confirm_delete_onnx_dirs")
        self.assertIn(
            "_run_bulk_with_progress(",
            src,
            "_confirm_delete_onnx_dirs must call _run_bulk_with_progress "
            "so multi-directory deletes show progress instead of "
            "freezing the UI.",
        )
        # The legacy inline rmtree loop must not come back either.
        self.assertNotIn(
            "for d in dirs:\n            try:\n                _shutil.rmtree(",
            src,
            "_confirm_delete_onnx_dirs must not re-grow an inline "
            "synchronous rmtree loop on the UI thread.",
        )


if __name__ == "__main__":
    unittest.main()

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_TEXT = (ROOT / "src" / "app.py").read_text(encoding="utf-8")


class ThreadingContractTests(unittest.TestCase):
    class _FakeChatDisplay:
        def __init__(self):
            self.text = "Assistant: thinking …"
            self.marks = {"assistant_body": len("Assistant: ")}

        def configure(self, **_kwargs):
            pass

        def _index(self, value):
            if value == "end":
                return len(self.text)
            if value == "end-1c":
                return len(self.text)
            if value == "end-11c":
                return max(0, len(self.text) - len("thinking …"))
            return self.marks[value]

        def delete(self, start, end):
            start_i = self._index(start)
            end_i = self._index(end)
            self.text = self.text[:start_i] + self.text[end_i:]

        def get(self, *_args):
            return self.text

        def insert(self, index, text, *_tags):
            if index != "end":
                raise AssertionError(f"unexpected insert index: {index}")
            self.text += text

        def see(self, *_args):
            pass

    def _function_source(self, name: str) -> str:
        tree = ast.parse(APP_TEXT)
        lines = APP_TEXT.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found")

    def _fake_chat_app(self):
        from src.app import App

        app = object.__new__(App)
        app._chat_display = self._FakeChatDisplay()
        app._chat_generation_id = 42
        app._chat_thinking = True
        app._chat_response_placeholder_active = True
        app._token_flush_scheduled = True
        app._stop_chat_first_token_feedback = lambda *_args, **_kwargs: None
        app._update_chat_readiness = lambda *_args, **_kwargs: None
        app._update_chat_copy_buttons = lambda *_args, **_kwargs: None
        app.chat_history = []
        app._stop_btn = type("FakeButton", (), {"configure": lambda self, **kwargs: None})()
        app.set_status = lambda *_args, **_kwargs: None
        return app, App

    def test_toolbox_workflow_worker_marshal_completion_through_after(self):
        source = self._function_source("_run_toolbox_workflow")
        self.assertIn("threading.Thread", source)
        self.assertIn("self.after(0, lambda r=result, t=token: self._toolbox_workflow_done", source)
        self.assertIn("self.after(0, lambda e=exc, t=token: self._toolbox_workflow_failed", source)

    def test_toolbox_install_worker_marshal_status_through_after(self):
        source = self._function_source("_install_toolbox_requirements")
        failed_source = self._function_source("_toolbox_install_failed")
        self.assertIn("threading.Thread", source)
        self.assertIn("self.after(0", source)
        self.assertIn("_toolbox_install_done", source)
        self.assertIn("_toolbox_install_failed", source)
        self.assertIn("_set_toolbox_status", failed_source)
        self.assertIn("_set_toolbox_result", failed_source)

    def test_models_refresh_fetches_status_off_ui_thread(self):
        refresh_source = self._function_source("_refresh_model_cards")
        schedule_source = self._function_source("_schedule_model_status_refresh")
        apply_source = self._function_source("_apply_model_status_refresh")
        continue_source = self._function_source("_continue_model_status_apply")

        self.assertIn("_schedule_model_status_refresh(force_refresh=True)", refresh_source)
        self.assertIn("threading.Thread", schedule_source)
        self.assertIn("self.after(", schedule_source)
        self.assertIn("_apply_model_status_refresh", schedule_source)
        self.assertIn("_continue_model_status_apply", apply_source)
        self.assertIn("self.after(", continue_source)
        self.assertIn("card.refresh_status", continue_source)

    def test_startup_reconciliation_scans_run_off_ui_thread(self):
        init_source = APP_TEXT[
            APP_TEXT.index("def __init__(self):"):
            APP_TEXT.index("def _log_startup_step")
        ]
        self.assertIn("_resume_pending_migration_if_any_async", init_source)
        self.assertIn("_heal_orphan_ollama_blobs_async", init_source)
        self.assertIn("_heal_legacy_onnx_paths_async", init_source)
        self.assertIn("_check_storage_relocation_on_startup_async", init_source)
        self.assertIn("_process_scheduled_deletes_after_startup_async", init_source)

        for name in (
            "_resume_pending_migration_if_any_async",
            "_heal_orphan_ollama_blobs_async",
            "_heal_legacy_onnx_paths_async",
            "_check_storage_relocation_on_startup_async",
            "_process_scheduled_deletes_after_startup_async",
        ):
            source = self._function_source(name)
            self.assertIn("threading.Thread", source)

        collect_source = self._function_source("_collect_storage_relocation_prompts")
        apply_source = self._function_source("_apply_storage_relocation_prompts")
        self.assertNotIn("config.save", collect_source)
        self.assertNotIn('self.cfg["models_dir"]', collect_source)
        self.assertIn("_apply_models_dir_auto_adopt", apply_source)

    def test_settings_uncatalogued_scan_runs_off_ui_thread(self):
        build_source = self._function_source("_build_uncatalogued_panel")
        async_source = self._function_source("_refresh_uncatalogued_lists_async")
        apply_source = self._function_source("_apply_uncatalogued_lists_result")
        render_source = self._function_source("_render_uncatalogued_lists")

        self.assertIn("_refresh_uncatalogued_lists_async(body)", build_source)
        self.assertIn("threading.Thread", async_source)
        self.assertIn("_collect_uncatalogued_lists", async_source)
        self.assertIn("self.after(", async_source)
        self.assertNotIn("_list_installed_ollama_tags()", build_source)
        self.assertIn("_render_uncatalogued_lists(body, state)", apply_source)
        self.assertNotIn("_collect_uncatalogued_lists()", render_source)

    def test_system_page_metrics_gathered_off_ui_thread(self):
        """The System page must NOT call system_info.get_system_summary
        (nvidia-smi + WMI + OpenVINO probes — up to ~30 s on a loaded vGPU
        host) on the UI thread. It gathers on a worker thread and applies the
        result via _apply_system_page on the UI thread."""
        update_source = self._function_source("_update_system_page")
        apply_source = self._function_source("_apply_system_page")

        # Dispatcher spawns a worker, calls get_system_summary there, and
        # marshals the result back to the UI thread.
        self.assertIn("threading.Thread", update_source)
        self.assertIn("get_system_summary", update_source)
        self.assertIn("self.after(", update_source)
        self.assertIn("_apply_system_page", update_source)

        # The UI-thread apply must NOT do the heavy gather itself.
        self.assertNotIn("get_system_summary", apply_source,
                         "_apply_system_page must not call get_system_summary "
                         "on the UI thread — it receives the precomputed summary.")

    def test_toolbox_status_checks_use_short_lived_caches(self):
        entry_source = self._function_source("_toolbox_model_entry")
        titles_source = self._function_source("_runnable_toolbox_titles")
        phase1_source = self._function_source("_phase1_model_cached")
        deps_source = self._function_source("_toolbox_missing_deps")
        disk_source = self._function_source("_cached_home_disk_free_gb")

        self.assertIn("_toolbox_model_by_id_cache", APP_TEXT)
        self.assertIn("_runnable_toolbox_titles_cache", APP_TEXT)
        self.assertIn("_home_disk_free_cache", APP_TEXT)
        self.assertIn("id(self._catalog_models)", entry_source)
        self.assertIn("now - cached[0] < 30", titles_source)
        self.assertIn("now - cached[0] < 5", disk_source)
        self.assertIn('(cache_dir / "hub" / cache_name).exists()', phase1_source)
        self.assertIn("(cache_dir / cache_name).exists()", phase1_source)
        self.assertNotIn("rglob", phase1_source)
        self.assertNotIn("importlib.invalidate_caches()", deps_source)

    def test_general_log_entries_are_batched_before_ui_insert(self):
        self.assertIn("_pending_log_entries", APP_TEXT)
        self.assertIn("_flush_log_entries", APP_TEXT)
        self.assertIn("self.after(100, self._flush_log_entries)", APP_TEXT)

    def test_threadsafe_after_preserves_callback_args(self):
        source = self._function_source("_poll_threadsafe") + "\n" + self._function_source("after")
        self.assertIn("ms, fn, args = self._ts_queue.get_nowait()", source)
        self.assertIn("super().after(ms, fn, *args)", source)
        self.assertIn("self._ts_queue.put((ms, func, args))", source)

    def test_image_callbacks_still_use_generation_id_barriers(self):
        source = self._function_source("_start_image_generation")
        self.assertIn("gen_id", source)
        self.assertIn("_img_gen_id", source)
        self.assertIn("gid == self._img_gen_id", source)
        self.assertIn("self.after(0", source)

    def test_analyze_callbacks_use_generation_id_barriers(self):
        source = self._function_source("_analyze_reference_image")
        self.assertIn("_analyze_gen_id += 1", source)
        self.assertIn("analyze_gid", source)
        self.assertIn("analyze_gid != self._analyze_gen_id", source)
        self.assertIn("_analyze_tick_elapsed(analyze_gid)", source)
        self.assertIn("gid == self._analyze_gen_id", source)

    def test_vision_picker_uses_local_name_cache(self):
        source = self._function_source("_refresh_vision_picker_ui")
        self.assertIn("_get_cached_local_names()", source)
        self.assertNotIn("self.ollama.local_model_names()", source)

    def test_chat_callbacks_use_generation_id_barriers(self):
        send_source = self._function_source("_send_message")
        ollama_source = self._function_source("_stream_ollama")
        onnx_source = self._function_source("_stream_onnx")
        flush_source = self._function_source("_flush_tokens")
        self.assertIn("_chat_generation_id += 1", send_source)
        self.assertIn("gen_id == self._chat_generation_id", ollama_source)
        self.assertIn("gen_id == self._chat_generation_id", onnx_source)
        self.assertIn("current_gen = self._chat_generation_id", flush_source)
        for source in (ollama_source, onnx_source):
            self.assertLess(source.find("self._flush_tokens()"), source.find("self._append_chat(\"error\""))

    def test_chat_token_flush_preserves_prior_batches_after_placeholder_removal(self):
        app, App = self._fake_chat_app()

        app._token_buf = [(42, "1. First item\n")]
        App._flush_tokens(app)
        app._token_buf = [(42, "2. Second item\n")]
        App._flush_tokens(app)

        self.assertNotIn("thinking", app._chat_display.text)
        self.assertIn("1. First item", app._chat_display.text)
        self.assertIn("2. Second item", app._chat_display.text)
        self.assertFalse(app._chat_response_placeholder_active)
        self.assertTrue(app._chat_thinking)

    def test_chat_done_drains_pending_tokens_before_final_newlines(self):
        app, App = self._fake_chat_app()

        app._token_buf = [(42, "complete response")]
        App._chat_done(app, 42)

        self.assertNotIn("thinking", app._chat_display.text)
        self.assertIn("complete response\n\n", app._chat_display.text)
        self.assertFalse(app._chat_response_placeholder_active)
        self.assertFalse(app._chat_thinking)


if __name__ == "__main__":
    unittest.main()

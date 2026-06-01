import ast
import json
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class UtilityAdapterStaticTests(unittest.TestCase):
    def _function_source(self, relative_path: Path, name: str) -> str:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(text)
        lines = text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
        self.fail(f"{name} not found in {relative_path}")

    def _assert_whisper_uses_current_generation_contract(self, source: str) -> None:
        """Shared shape checks for both Whisper call sites.

        Both `run_whisper` (benchmark adapter, hard-coded English) and
        `transcribe` (Toolbox, multi-language) must:
        - pass `return_attention_mask=True`
        - thread `attention_mask` into generate_kwargs
        - splat `**generate_kwargs` into model.generate
        - request the transcribe task (kwarg or dict-value form)
        - clear `generation_config.forced_decoder_ids = None`
        - avoid the deprecated `get_decoder_prompt_ids` /
          `forced_decoder_ids=` / `torch_dtype=_torch_dtype()` paths
        """
        self.assertIn("return_attention_mask=True", source)
        self.assertIn('generate_kwargs["attention_mask"]', source)
        self.assertIn("**generate_kwargs", source)
        self.assertTrue(
            'task="transcribe"' in source or '"task": "transcribe"' in source,
            "expected task=transcribe in some form (kwarg or generate_kwargs dict)",
        )
        self.assertIn("generation_config.forced_decoder_ids = None", source)
        self.assertNotIn("get_decoder_prompt_ids", source)
        self.assertNotIn("forced_decoder_ids=", source)
        self.assertNotIn("torch_dtype=_torch_dtype()", source)

    def _assert_benchmark_whisper_pins_english(self, source: str) -> None:
        self._assert_whisper_uses_current_generation_contract(source)
        # Benchmark must pin English for deterministic, reproducible
        # results across runs.
        self.assertIn('language="english"', source)

    def _assert_toolbox_whisper_threads_language(self, source: str) -> None:
        self._assert_whisper_uses_current_generation_contract(source)
        # Toolbox accepts a language= kwarg and only sets the language
        # in generate_kwargs when it's truthy (None means auto-detect).
        self.assertIn("language: str | None = None", source)
        self.assertIn('generate_kwargs["language"] = language', source)

    def _assert_loader_prefers_current_dtype_keyword(self, source: str) -> None:
        dtype_index = source.find("dtype=dtype")
        fallback_index = source.find("torch_dtype=dtype")
        self.assertGreaterEqual(dtype_index, 0)
        self.assertGreater(fallback_index, dtype_index)

    def test_batch_whisper_adapter_avoids_deprecated_generation_warnings(self):
        source = self._function_source(Path("src") / "phase1_adapters.py", "run_whisper")

        self._assert_benchmark_whisper_pins_english(source)

    def test_toolbox_transcribe_avoids_deprecated_generation_warnings(self):
        source = self._function_source(Path("src") / "workflows.py", "transcribe")

        self._assert_toolbox_whisper_threads_language(source)

    def test_transformers_loaders_prefer_dtype_over_deprecated_torch_dtype(self):
        phase1_source = self._function_source(Path("src") / "phase1_adapters.py", "_from_pretrained_with_dtype")
        workflow_source = self._function_source(Path("src") / "workflows.py", "_from_pretrained_with_dtype")

        self._assert_loader_prefers_current_dtype_keyword(phase1_source)
        self._assert_loader_prefers_current_dtype_keyword(workflow_source)

    def test_table_transformer_loaders_avoid_known_warning_paths(self):
        phase1_source = self._function_source(Path("src") / "phase1_adapters.py", "run_table_transformer")
        workflow_source = self._function_source(Path("src") / "workflows.py", "detect_table")

        for source in (phase1_source, workflow_source):
            self.assertIn("use_fast=False", source)
            self.assertIn("low_cpu_mem_usage=False", source)
            self.assertIn("_quiet_known_hf_loader_warnings()", source)

    def test_florence_uses_fast_tokenizer_and_creates_synthetic_asset_dirs(self):
        florence_source = self._function_source(Path("src") / "phase1_adapters.py", "run_florence")
        workflow_source = self._function_source(Path("src") / "workflows.py", "read_image")
        asset_source = self._function_source(Path("src") / "phase1_adapters.py", "_ensure_vision_image")

        for source in (florence_source, workflow_source):
            self.assertIn("use_fast=True", source)
            self.assertNotIn("use_fast=False", source)
            self.assertIn("_quiet_known_hf_loader_warnings()", source)
        self.assertIn("path.parent.mkdir(parents=True, exist_ok=True)", asset_source)

    def test_speech_fixture_creates_parent_before_fallback_wav(self):
        from src import phase1_adapters

        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.phase1_adapters.subprocess.run", side_effect=RuntimeError("speech unavailable")):
            wav_path = phase1_adapters._ensure_speech_wav(Path(tmp) / "missing-root")

            self.assertEqual(wav_path.name, "phase1_reference_speech.wav")
            self.assertTrue(wav_path.exists())
            self.assertGreater(wav_path.stat().st_size, 1000)

    def test_all_minilm_routes_by_model_id_not_only_runtime_metadata(self):
        from src import phase1_adapters

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(phase1_adapters, "run_sentence_transformer", return_value={"status": "ok"}) as run:
            result = phase1_adapters.run_transformers_adapter(
                {"id": "all-minilm", "runtime": "", "hf_repo": "sentence-transformers/all-MiniLM-L6-v2"},
                Path(tmp),
            )

        self.assertEqual(result, {"status": "ok"})
        run.assert_called_once()

    def test_remote_code_loaders_pin_catalog_revision(self):
        phase1_source = self._function_source(Path("src") / "phase1_adapters.py", "run_florence")
        workflow_source = self._function_source(Path("src") / "workflows.py", "read_image")

        for source in (phase1_source, workflow_source):
            self.assertIn("_require_hf_revision", source)
            self.assertIn("revision=revision", source)

    def test_toolbox_workflows_module_exposes_piper_and_whisper_language_tables(self):
        from src import workflows

        self.assertIn("Auto-detect", workflows.WHISPER_LANGUAGES)
        self.assertIsNone(workflows.WHISPER_LANGUAGES["Auto-detect"])
        self.assertEqual(workflows.WHISPER_LANGUAGES["English"], "english")
        self.assertGreaterEqual(len(workflows.WHISPER_LANGUAGES), 25)

        self.assertGreaterEqual(len(workflows.PIPER_VOICES), 5)
        first = next(iter(workflows.PIPER_VOICES.values()))
        self.assertIn("key", first)
        self.assertIn("onnx", first)
        self.assertIn("json", first)
        self.assertTrue(first["onnx"].endswith(".onnx"))
        self.assertTrue(first["json"].endswith(".onnx.json"))

    def test_toolbox_transcribe_accepts_language_kwarg_and_handles_auto_detect(self):
        source = self._function_source(Path("src") / "workflows.py", "transcribe")

        self.assertIn("language: str | None = None", source)
        self.assertIn("if language:", source)
        self.assertIn('generate_kwargs["language"] = language', source)

    def test_toolbox_synthesize_uses_piper_voice_loader(self):
        source = self._function_source(Path("src") / "workflows.py", "synthesize")

        self.assertIn("language: str | None = None", source)
        self.assertIn("from piper.voice import PiperVoice", source)
        self.assertIn("PIPER_VOICES", source)
        self.assertIn("piper_voice.synthesize", source)
        self.assertNotIn("SpeechT5", source)

    def test_toolbox_synthesize_iterates_audio_chunks_and_writes_int16_bytes(self):
        # Regression guard for the v5.5.17 ship bug: piper-tts >=1.2 returns
        # Iterable[AudioChunk]; calling synthesize() without iterating yields
        # an empty (0-second) WAV with just the header. The fix must iterate
        # chunks, write chunk.audio_int16_bytes, and trust the chunk's
        # sample_rate over the static config fallback.
        source = self._function_source(Path("src") / "workflows.py", "synthesize")

        # MUST iterate the synthesize() generator.
        self.assertIn("for chunk in piper_voice.synthesize(text):", source)
        # MUST write int16 PCM bytes from each chunk.
        self.assertIn("chunk.audio_int16_bytes", source)
        self.assertIn("writeframes(audio_bytes)", source)
        # MUST pull rate/width/channels off the chunk, not just hardcode.
        self.assertIn('getattr(chunk, "sample_rate"', source)
        # MUST fail loudly when the synthesizer yielded zero chunks instead
        # of silently shipping a 0-second WAV.
        self.assertIn("Piper produced no audio", source)
        # Must NOT use the old <1.0 wave-file-as-argument signature.
        self.assertNotIn("piper_voice.synthesize(text, wav_file)", source)

    def test_toolbox_table_workflows_use_ollama_and_got_with_dtype_pin(self):
        ollama_source = self._function_source(Path("src") / "workflows.py", "extract_table_ollama")
        got_source = self._function_source(Path("src") / "workflows.py", "extract_table_got")
        run_got_source = self._function_source(Path("src") / "workflows.py", "_run_got")

        self.assertIn("chat_stream", ollama_source)
        self.assertIn('"images": [image_b64]', ollama_source)
        self.assertIn("_table_text_to_tsv", ollama_source)

        self.assertIn("_run_got", got_source)
        self.assertIn("_table_text_to_tsv", got_source)

        # GOT-OCR silently loads in float32 unless dtype is pinned.
        self.assertIn("torch_dtype=dtype", run_got_source)
        self.assertIn("_got_dtype_and_device", run_got_source)
        self.assertIn("format=True", run_got_source)

    def test_table_text_to_tsv_extracts_latex_tabular_block(self):
        from src import workflows

        latex = (
            r"\begin{tabular}{ll}"
            "\nHeader1 & Header2 \\\\\n\\hline\n"
            "RowA1 & RowA2 \\\\\n"
            "RowB1 & RowB2 \\\\\n"
            r"\end{tabular}"
        )
        tsv = workflows._table_text_to_tsv(latex)
        rows = [line.split("\t") for line in tsv.splitlines() if line]

        self.assertEqual(rows[0], ["Header1", "Header2"])
        self.assertIn(["RowA1", "RowA2"], rows)
        self.assertIn(["RowB1", "RowB2"], rows)

    def test_table_text_to_tsv_falls_back_to_tab_and_whitespace_split(self):
        from src import workflows

        tab_input = "Name\tAge\nAlice\t30\nBob\t25"
        ws_input = "Name    Age\nAlice    30\nBob    25"

        for source in (tab_input, ws_input):
            tsv = workflows._table_text_to_tsv(source)
            rows = [line.split("\t") for line in tsv.splitlines() if line]
            self.assertEqual(rows[0], ["Name", "Age"])
            self.assertIn(["Alice", "30"], rows)
            self.assertIn(["Bob", "25"], rows)

    def test_phi_adapter_uses_ollama_client_for_local_status_and_delete(self):
        is_local_source = self._function_source(Path("src") / "phase1_adapters.py", "_ollama_tag_is_local")
        remove_source = self._function_source(Path("src") / "phase1_adapters.py", "_remove_ollama_tag")

        self.assertIn("OllamaClient().is_model_local(tag)", is_local_source)
        self.assertIn("except OllamaError", is_local_source)
        self.assertIn("OllamaClient().delete_model(tag)", remove_source)
        self.assertIn("except OllamaError", remove_source)
        self.assertNotIn('["ollama", "list"]', is_local_source)
        self.assertNotIn('["ollama", "rm", tag]', remove_source)

    def test_phi_adapter_does_not_delete_preexisting_model_on_pull_failure(self):
        from src import phase1_adapters

        def fake_run(cmd, **kwargs):
            if cmd == ["ollama", "pull", "phi4"]:
                raise subprocess.CalledProcessError(1, cmd, stderr="pull failed")
            self.fail(f"unexpected command: {cmd}")

        with patch("src.phase1_adapters.subprocess.run", side_effect=fake_run), \
             patch("src.phase1_adapters._ollama_tag_is_local", return_value=True), \
             patch("src.phase1_adapters._remove_ollama_tag") as remove:
            result = phase1_adapters.run_phi_text({"id": "phi-4-multimodal"}, Path("."))

        self.assertEqual(result["status"], "error")
        self.assertIn("Ollama command failed", result["error"])
        self.assertEqual(result["cleanup_status"], "retained_existing")
        remove.assert_not_called()

    def test_phi_adapter_cleans_downloaded_model_after_api_failure(self):
        from src import phase1_adapters

        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        with patch("src.phase1_adapters.subprocess.run", side_effect=fake_run), \
             patch("src.phase1_adapters._ollama_tag_is_local", return_value=False), \
             patch("src.phase1_adapters._remove_ollama_tag", return_value="deleted") as remove, \
             patch("src.phase1_adapters.urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            result = phase1_adapters.run_phi_text({"id": "phi-4-multimodal"}, Path("."))

        self.assertEqual(result["status"], "error")
        self.assertIn("Ollama API request failed", result["error"])
        self.assertEqual(result["cleanup_status"], "deleted")
        self.assertIn(["ollama", "pull", "phi4"], commands)
        remove.assert_called_once_with("phi4")

    def test_phi_adapter_json_encodes_prompt_text(self):
        from src import phase1_adapters

        captured = {}
        prompt = 'Line one\n"quoted" path C:\\demo'

        def fake_run(cmd, **_kwargs):
            if cmd == ["ollama", "pull", "phi4"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="")
            self.fail(f"unexpected command: {cmd}")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"response":"ok"}'

        def fake_urlopen(req, timeout=0):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResponse()

        with patch("src.phase1_adapters.subprocess.run", side_effect=fake_run), \
             patch("src.phase1_adapters._ollama_tag_is_local", return_value=True), \
             patch("src.phase1_adapters.urllib.request.urlopen", side_effect=fake_urlopen):
            result = phase1_adapters.run_phi_text({"id": "phi-4-multimodal"}, Path("."), prompt=prompt)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(captured["payload"]["prompt"], prompt)
        self.assertEqual(captured["payload"]["model"], "phi4")


if __name__ == "__main__":
    unittest.main()

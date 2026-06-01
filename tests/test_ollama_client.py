import unittest

from src import ollama_client
from src.ollama_client import OllamaClient


class OllamaClientTests(unittest.TestCase):
    def test_is_model_local_requires_exact_tag_or_exact_base(self):
        client = OllamaClient()
        client.local_model_names = lambda: {"llama3.2:1b", "qwen2.5"}

        self.assertFalse(client.is_model_local("llama3.2:3b"))
        self.assertFalse(client.is_model_local("qwen2:7b"))
        self.assertTrue(client.is_model_local("qwen2.5:0.5b"))

    def test_is_model_local_treats_bare_tags_as_latest_only(self):
        client = OllamaClient()
        client.local_model_names = lambda: {
            "llama3.3:latest",
            "mistral-nemo:latest",
            "mistral-small3.2:latest",
            "phi4:latest",
            "phi4-mini:latest",
            "phi4-reasoning:plus",
            "llama3.2:1b",
        }

        for tag in ("llama3.3", "mistral-nemo", "mistral-small3.2", "phi4", "phi4-mini"):
            with self.subTest(tag=tag):
                self.assertTrue(client.is_model_local(tag))
        self.assertTrue(client.is_model_local("phi4-reasoning:plus"))
        self.assertFalse(client.is_model_local("phi4-reasoning"))
        self.assertFalse(client.is_model_local("llama3.2:3b"))

    def test_delete_model_uses_current_ollama_payload_and_legacy_fallback(self):
        client = OllamaClient("http://ollama.test")
        calls = []

        class Response:
            def __init__(self, status_code, text=""):
                self.status_code = status_code
                self.text = text

        def fake_delete(url, json, timeout):
            calls.append((url, json, timeout))
            return Response(400, "missing model key") if json == {"model": "demo:latest"} else Response(200)

        original_delete = ollama_client.requests.delete
        try:
            ollama_client.requests.delete = fake_delete
            client.delete_model("demo:latest")
        finally:
            ollama_client.requests.delete = original_delete

        self.assertEqual(
            calls,
            [
                ("http://ollama.test/api/delete", {"model": "demo:latest"}, 15),
                ("http://ollama.test/api/delete", {"name": "demo:latest"}, 15),
            ],
        )

    def test_delete_model_does_not_send_fallback_after_success(self):
        client = OllamaClient("http://ollama.test")
        calls = []

        class Response:
            status_code = 200
            text = ""

        def fake_delete(url, json, timeout):
            calls.append((url, json, timeout))
            return Response()

        original_delete = ollama_client.requests.delete
        try:
            ollama_client.requests.delete = fake_delete
            client.delete_model("demo:latest")
        finally:
            ollama_client.requests.delete = original_delete

        self.assertEqual(calls, [("http://ollama.test/api/delete", {"model": "demo:latest"}, 15)])

    def test_delete_model_reports_both_payload_failures(self):
        client = OllamaClient("http://ollama.test")

        class Response:
            def __init__(self, status_code, text):
                self.status_code = status_code
                self.text = text

        def fake_delete(_url, json, timeout):
            key = next(iter(json))
            return Response(400, f"{key} failed")

        original_delete = ollama_client.requests.delete
        try:
            ollama_client.requests.delete = fake_delete
            with self.assertRaisesRegex(Exception, "model=.*model failed.*name=.*name failed"):
                client.delete_model("demo:latest")
        finally:
            ollama_client.requests.delete = original_delete

    def test_qwen3_requests_disable_ollama_thinking_for_visible_ui_output(self):
        self.assertIs(ollama_client.think_option_for_model("qwen3:4b"), False)
        self.assertIs(ollama_client.think_option_for_model("qwen3:30b-a3b"), False)
        self.assertIsNone(ollama_client.think_option_for_model("qwen2.5:7b"))

    def test_reasoning_tag_models_disable_ollama_thinking(self):
        """GPU-Super §1.5: every reasoning-tag base name must short-circuit `think`."""
        for tag in (
            "deepseek-r1:1.5b",
            "deepseek-r1:32b",
            "DeepSeek-R1:8b",
            "magistral:24b",
            "phi4-mini-reasoning:latest",
            "phi-4-mini-reasoning",
            "phi4-reasoning",
            "phi4-reasoning:plus",
            "qwen3-coder:30b",
            "nemotron-3-nano:4b",
            "nemotron-3-nano:4b-q8_0",
            "nemotron3:33b",
        ):
            with self.subTest(tag=tag):
                self.assertIs(
                    ollama_client.think_option_for_model(tag),
                    False,
                    f"think_option_for_model({tag!r}) must return False to avoid "
                    "empty Chat bubbles when the visible response is starved by "
                    "the <think>...</think> block.",
                )
        # Plain non-reasoning chat tags must still return None (default behavior).
        for tag in ("llama3.1:8b", "phi3:mini", "gemma2:9b"):
            with self.subTest(tag=tag):
                self.assertIsNone(ollama_client.think_option_for_model(tag))

    def test_strip_think_blocks_keeps_visible_answer(self):
        self.assertEqual(
            ollama_client.strip_think_blocks("<think>hidden</think>Visible answer."),
            "Visible answer.",
        )
        self.assertEqual(
            ollama_client.strip_think_blocks("<think>hidden</think>\n\n  Visible answer.  "),
            "Visible answer.",
        )
        self.assertEqual(
            ollama_client.strip_think_blocks("hidden tokens</think>{\"ok\":true}"),
            "{\"ok\":true}",
        )
        self.assertEqual(
            ollama_client.strip_think_blocks("Visible answer.<think>unfinished hidden tokens"),
            "Visible answer.",
        )

    def _capture_chat_payload(self, *, with_stats: bool, **call_kwargs):
        """Drive a chat_stream(*_with_stats) call and return the Ollama payload.

        Patches ``requests.post`` so the real Ollama service is never
        contacted; the test confirms the JSON body the client builds.
        """
        client = OllamaClient("http://ollama.test")
        captured = {}

        class _DummyResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())  # empty stream — we only care about the request body

        def fake_post(url, json, stream, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["stream"] = stream
            captured["timeout"] = timeout
            return _DummyResp()

        original_post = ollama_client.requests.post
        try:
            ollama_client.requests.post = fake_post
            messages = [{"role": "user", "content": "hi"}]
            if with_stats:
                gen = client.chat_stream_with_stats("demo:latest", messages, **call_kwargs)
            else:
                gen = client.chat_stream("demo:latest", messages, **call_kwargs)
            list(gen)
        finally:
            ollama_client.requests.post = original_post
        return captured

    def test_chat_stream_omits_repeat_options_by_default(self):
        for with_stats in (False, True):
            with self.subTest(with_stats=with_stats):
                captured = self._capture_chat_payload(with_stats=with_stats)
                opts = captured["json"]["options"]
                self.assertNotIn("repeat_penalty", opts)
                self.assertNotIn("repeat_last_n", opts)

    def test_chat_stream_forwards_positive_repeat_options(self):
        for with_stats in (False, True):
            with self.subTest(with_stats=with_stats):
                captured = self._capture_chat_payload(
                    with_stats=with_stats,
                    repeat_penalty=1.15,
                    repeat_last_n=256,
                )
                opts = captured["json"]["options"]
                self.assertEqual(opts["repeat_penalty"], 1.15)
                self.assertEqual(opts["repeat_last_n"], 256)

    def test_chat_stream_rejects_non_positive_repeat_options(self):
        # Zero, negative, bool, and None values must be silently ignored so
        # callers can pass them through from optional catalog fields.
        for value_pair in ((0, 0), (-1.0, -1), (False, True), (None, None)):
            penalty, last_n = value_pair
            for with_stats in (False, True):
                with self.subTest(values=value_pair, with_stats=with_stats):
                    captured = self._capture_chat_payload(
                        with_stats=with_stats,
                        repeat_penalty=penalty,
                        repeat_last_n=last_n,
                    )
                    opts = captured["json"]["options"]
                    self.assertNotIn("repeat_penalty", opts)
                    self.assertNotIn("repeat_last_n", opts)

    def test_chat_stream_rejects_string_repeat_options(self):
        # Defensive: a catalog author might quote the value as a JSON string
        # ("1.15") and the runner's _positive_float coercion would normally
        # filter that, but if it ever leaks through the client must NOT
        # forward a string into the Ollama options payload (Ollama would
        # reject the request and the row would fail with a confusing
        # protocol error instead of just losing the guard).
        for with_stats in (False, True):
            with self.subTest(with_stats=with_stats):
                captured = self._capture_chat_payload(
                    with_stats=with_stats,
                    repeat_penalty="1.15",
                    repeat_last_n="256",
                )
                opts = captured["json"]["options"]
                self.assertNotIn("repeat_penalty", opts)
                self.assertNotIn("repeat_last_n", opts)

    def test_chat_stream_accepts_either_repeat_option_independently(self):
        # Catalog overrides can specify only one of the two values (e.g.
        # tightening just the window without changing the penalty strength);
        # the client must forward whichever side is positive without
        # requiring its sibling.
        for with_stats in (False, True):
            with self.subTest(side="penalty_only", with_stats=with_stats):
                captured = self._capture_chat_payload(
                    with_stats=with_stats,
                    repeat_penalty=1.2,
                )
                opts = captured["json"]["options"]
                self.assertEqual(opts.get("repeat_penalty"), 1.2)
                self.assertNotIn("repeat_last_n", opts)
            with self.subTest(side="window_only", with_stats=with_stats):
                captured = self._capture_chat_payload(
                    with_stats=with_stats,
                    repeat_last_n=128,
                )
                opts = captured["json"]["options"]
                self.assertEqual(opts.get("repeat_last_n"), 128)
                self.assertNotIn("repeat_penalty", opts)


if __name__ == "__main__":
    unittest.main()

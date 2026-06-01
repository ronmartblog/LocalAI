import unittest
from pathlib import Path

from src.onnx_client import _format_genai_chat_prompt


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="ignore")


class OnnxClientTests(unittest.TestCase):
    def test_genai_prompts_are_wrapped_for_chat_models(self):
        wrapped = _format_genai_chat_prompt("Return exactly: hello.")

        self.assertEqual(
            wrapped,
            "<|im_start|>user<|im_sep|>Return exactly: hello.<|im_end|><|im_start|>assistant<|im_sep|>",
        )

    def test_genai_prompt_wrapper_preserves_preformatted_chat(self):
        prompt = "<|im_start|>user<|im_sep|>hello<|im_end|><|im_start|>assistant<|im_sep|>"

        self.assertEqual(_format_genai_chat_prompt(prompt), prompt)


class OnnxClientDefensiveImportContractTests(unittest.TestCase):
    """v2026.06.01.4: import blocks must catch every exception, not just ImportError.

    A broken optional runtime (e.g. an ``onnxruntime`` install whose Python ABI
    no longer matches the installed Python version) can satisfy ``import
    onnxruntime`` but raise ``AttributeError`` or ``OSError`` on first symbol
    access. Catching only ``ImportError`` lets that crash propagate and takes
    the whole app down at module load — exactly the regression observed on a
    high-VRAM workstation on 2026-06-01. The fix is to mirror
    ``src/openvino_client.py``'s ``except Exception`` defensive pattern.
    """

    def test_onnxruntime_import_block_catches_all_exceptions(self):
        src = _read("src/onnx_client.py")
        self.assertIn("import onnxruntime as ort", src)
        snippet_start = src.index("import onnxruntime as ort")
        # Look at the ~25 lines following the import for the matching except.
        window = src[snippet_start : snippet_start + 1200]
        self.assertIn(
            "except Exception",
            window,
            "onnxruntime import block must use `except Exception` so a partial "
            "install cannot crash app startup (v2026.06.01.4 regression).",
        )

    def test_onnxruntime_genai_import_block_catches_all_exceptions(self):
        src = _read("src/onnx_client.py")
        self.assertIn("import onnxruntime_genai as _og", src)
        snippet_start = src.index("import onnxruntime_genai as _og")
        window = src[snippet_start : snippet_start + 1200]
        self.assertIn(
            "except Exception",
            window,
            "onnxruntime_genai import block must use `except Exception` "
            "(same defensive pattern as onnxruntime).",
        )

    def test_broken_runtime_leaves_availability_flags_false(self):
        """If the import block is entered via the except branch, every
        availability flag in that block must be set to False. Otherwise the
        rest of the codebase will gate on a True flag and then crash when it
        tries to use the broken runtime."""
        src = _read("src/onnx_client.py")
        # Find the onnxruntime except block and verify it resets every flag.
        ort_block_start = src.index("import onnxruntime as ort")
        ort_block = src[ort_block_start : ort_block_start + 1500]
        for flag in (
            "ONNX_AVAILABLE = False",
            "DIRECTML_AVAILABLE = False",
            "COREML_AVAILABLE = False",
            "OPENVINO_AVAILABLE = False",
        ):
            self.assertIn(
                flag,
                ort_block,
                f"onnxruntime except branch must set `{flag}` so downstream "
                f"gates fall back to the Ollama path.",
            )
        # Same guarantee for the genai block.
        genai_block_start = src.index("import onnxruntime_genai as _og")
        genai_block = src[genai_block_start : genai_block_start + 1500]
        for flag in (
            "GENAI_AVAILABLE = False",
            "GENAI_DML_AVAILABLE = False",
        ):
            self.assertIn(
                flag,
                genai_block,
                f"onnxruntime_genai except branch must set `{flag}`.",
            )


if __name__ == "__main__":
    unittest.main()

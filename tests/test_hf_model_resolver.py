# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Tests for src.hf_model_resolver — the pure URL/slug parser that feeds
the Add-from-Hugging-Face dialog.

The dialog itself is Tk-bound; this layer is plain strings in / dataclass
out, so the matrix below covers every shape the test-engineer flagged plus
the explicit reject list. Keeping these tests fast and dependency-free is
part of the "pure module" contract documented in the file's docstring.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hf_model_resolver import (
    InvalidHFUrl,
    ParsedTarget,
    parse_url,
    slug_to_model_id,
)


class ParseValidHfUrls(unittest.TestCase):
    def test_bare_repo_url(self):
        t = parse_url("https://huggingface.co/black-forest-labs/FLUX.1-schnell")
        self.assertEqual(t.route, "hf")
        self.assertEqual(t.repo_id, "black-forest-labs/FLUX.1-schnell")
        self.assertIsNone(t.revision)
        self.assertIsNone(t.file_path)

    def test_bare_repo_url_trailing_slash(self):
        t = parse_url("https://huggingface.co/org/name/")
        self.assertEqual(t.repo_id, "org/name")

    def test_tree_url(self):
        t = parse_url("https://huggingface.co/org/name/tree/main")
        self.assertEqual(t.repo_id, "org/name")
        self.assertEqual(t.revision, "main")
        self.assertIsNone(t.file_path)

    def test_blob_url(self):
        t = parse_url(
            "https://huggingface.co/city96/FLUX.1-schnell-gguf/"
            "blob/main/flux1-schnell-Q4_K_S.gguf"
        )
        self.assertEqual(t.repo_id, "city96/FLUX.1-schnell-gguf")
        self.assertEqual(t.revision, "main")
        self.assertEqual(t.file_path, "flux1-schnell-Q4_K_S.gguf")

    def test_resolve_url_with_nested_path(self):
        t = parse_url(
            "https://huggingface.co/org/name/resolve/abc123/sub/folder/file.safetensors"
        )
        self.assertEqual(t.file_path, "sub/folder/file.safetensors")
        self.assertEqual(t.revision, "abc123")

    def test_short_host_hf_co(self):
        t = parse_url("https://hf.co/org/name")
        self.assertEqual(t.route, "hf")
        self.assertEqual(t.repo_id, "org/name")

    def test_url_with_query_params_strips_them(self):
        # urlparse drops the query off path; revision/file_path should still parse cleanly.
        t = parse_url("https://huggingface.co/org/name?utm_source=demo")
        self.assertEqual(t.repo_id, "org/name")

    def test_raw_url_treated_like_blob(self):
        t = parse_url("https://huggingface.co/org/name/raw/main/README.md")
        self.assertEqual(t.repo_id, "org/name")
        self.assertEqual(t.revision, "main")
        self.assertEqual(t.file_path, "README.md")

    def test_pasted_with_whitespace(self):
        t = parse_url("   https://huggingface.co/org/name   ")
        self.assertEqual(t.repo_id, "org/name")


class ParseBareSlug(unittest.TestCase):
    def test_simple_slug(self):
        t = parse_url("Qwen/Qwen2.5-0.5B")
        self.assertEqual(t.route, "hf")
        self.assertEqual(t.repo_id, "Qwen/Qwen2.5-0.5B")
        self.assertIsNone(t.revision)

    def test_slug_with_dots_in_name(self):
        t = parse_url("black-forest-labs/FLUX.1-schnell")
        self.assertEqual(t.repo_id, "black-forest-labs/FLUX.1-schnell")

    def test_slug_with_dash_in_org(self):
        t = parse_url("microsoft/Phi-4-mini-instruct")
        self.assertEqual(t.repo_id, "microsoft/Phi-4-mini-instruct")


class ParseOllamaShapes(unittest.TestCase):
    def test_ollama_scheme_with_tag(self):
        t = parse_url("ollama:llama3.2:1b")
        self.assertEqual(t.route, "ollama")
        self.assertEqual(t.ollama_tag, "llama3.2:1b")
        self.assertEqual(t.repo_id, "")

    def test_ollama_scheme_with_bare_tag(self):
        t = parse_url("ollama:phi3")
        self.assertEqual(t.route, "ollama")
        self.assertEqual(t.ollama_tag, "phi3")

    def test_ollama_library_url(self):
        t = parse_url("https://ollama.com/library/llama3.2")
        self.assertEqual(t.route, "ollama")
        self.assertEqual(t.ollama_tag, "llama3.2")

    def test_ollama_library_url_with_size(self):
        t = parse_url("https://ollama.com/library/llama3.2:1b")
        self.assertEqual(t.ollama_tag, "llama3.2:1b")

    def test_ollama_library_url_www(self):
        t = parse_url("https://www.ollama.com/library/phi3")
        self.assertEqual(t.ollama_tag, "phi3")


class ParseExplicitRejects(unittest.TestCase):
    """Inputs that MUST raise InvalidHFUrl, per the test-engineer reject list."""

    def test_empty_string(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("")

    def test_whitespace_only(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("   ")

    def test_none_input(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url(None)

    def test_http_url_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("http://huggingface.co/org/name")

    def test_unknown_host_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://civitai.com/models/12345")

    def test_hf_spaces_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/spaces/org/cool-demo")

    def test_hf_datasets_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/datasets/org/some-dataset")

    def test_hf_collections_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/collections/abc")

    def test_hf_papers_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/papers/2401.12345")

    def test_hf_org_only_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/black-forest-labs")

    def test_hf_discussions_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/org/name/discussions/42")

    def test_path_traversal_in_repo(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/org/../etc/passwd")

    def test_path_traversal_in_file(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/org/name/blob/main/../../etc/passwd")

    def test_slug_without_slash(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("just-an-org")

    def test_slug_with_too_many_slashes(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("a/b/c")

    def test_ollama_scheme_with_path(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("ollama:org/tag")

    def test_ollama_scheme_empty_tag(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("ollama:")

    def test_tree_url_missing_revision(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/org/name/tree")

    def test_blob_url_missing_file(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://huggingface.co/org/name/blob/main")


class DoubledUrlPasteAccidents(unittest.TestCase):
    """Real-user paste accident: copy/paste lands on top of an existing URL
    selection, producing https://host/path/https://host/path/... .  These
    MUST be rejected loudly with a copy-friendly message — letting them
    flow through to Ollama yields a confusing "manifest does not exist".
    Reported by Ron 2026-05-19 with the nemotron3 paste."""

    def test_doubled_ollama_url(self):
        with self.assertRaises(InvalidHFUrl) as ctx:
            parse_url("https://ollama.com/library/https://ollama.com/library/nemotron3")
        self.assertIn("two urls pasted together", str(ctx.exception).lower())

    def test_doubled_hf_url(self):
        with self.assertRaises(InvalidHFUrl) as ctx:
            parse_url("https://huggingface.co/https://huggingface.co/org/name")
        self.assertIn("two urls pasted together", str(ctx.exception).lower())

    def test_doubled_with_http_inside(self):
        # The inner URL might be http:// not https:// after partial overwrite.
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://ollama.com/library/http://ollama.com/library/foo")

    def test_legitimate_query_string_with_url_inside_is_not_doubled(self):
        # Query strings are stripped by urlparse and live in parsed.query —
        # not parsed.path — so a legit URL with ?ref=https://... must still
        # parse fine.
        t = parse_url("https://huggingface.co/org/name?ref=https://blog.example.com")
        self.assertEqual(t.repo_id, "org/name")

    def test_ollama_extra_path_segments_rejected(self):
        # Not a doubled URL but the same root cause — Ollama tags are a
        # single path segment.  Slashes inside the tag should never reach
        # the Ollama client.
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://ollama.com/library/nemotron3/manifest")

    def test_ollama_tag_with_multiple_colons_rejected(self):
        with self.assertRaises(InvalidHFUrl):
            parse_url("https://ollama.com/library/llama3.2:1b:extra")


class SlugToModelId(unittest.TestCase):
    def test_simple_slug(self):
        self.assertEqual(slug_to_model_id("org/name"), "org-name")

    def test_uppercase_lowered(self):
        self.assertEqual(slug_to_model_id("Org/NAME"), "org-name")

    def test_dots_kept(self):
        # The helper normalises dots to dashes so the resulting id is safe
        # in URLs and filenames.
        self.assertEqual(slug_to_model_id("black-forest-labs/FLUX.1-schnell"),
                         "black-forest-labs-flux-1-schnell")

    def test_suffix_appended(self):
        result = slug_to_model_id("org/name", suffix="img")
        self.assertTrue(result.endswith("-img"))

    def test_returns_string(self):
        self.assertIsInstance(slug_to_model_id("a/b"), str)


class ParsedTargetShape(unittest.TestCase):
    def test_dataclass_is_frozen(self):
        t = ParsedTarget(route="hf", repo_id="org/name")
        with self.assertRaises(Exception):
            t.repo_id = "other/name"  # type: ignore[misc]

    def test_default_fields(self):
        t = ParsedTarget(route="hf")
        self.assertEqual(t.repo_id, "")
        self.assertIsNone(t.revision)
        self.assertIsNone(t.file_path)
        self.assertIsNone(t.ollama_tag)


if __name__ == "__main__":
    unittest.main()

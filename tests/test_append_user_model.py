# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Tests for ``catalog.append_user_model`` — the single source of truth
the Add-from-Hugging-Face dialog uses to persist user-added entries.

Covers the contract documented in the helper's docstring:

- ``user_added`` is force-set True (untrusted input can't override).
- ``source_url`` is recorded verbatim.
- HF-backed entries MUST pin ``hf_revision`` to a 40-char SHA — branch
  names like ``main`` are rejected outright.
- Ollama entries are exempt from the SHA pin.
- ID collisions get ``-2``, ``-3``, … suffixes until unique.
- The full schema round-trips through save/load (validators don't strip
  new fields — fixes the false README claim noted by the test-engineer).
- The README in the saved file documents the new ``user_added`` /
  ``requires_review`` / ``source_url`` / ``added_at`` fields.
"""

import hashlib
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import catalog


SHA = "a" * 40


# ── Catalog pollution guard ──────────────────────────────────────────────────
# Paranoia: an earlier draft of this file called catalog.append_user_model()
# without ``catalog_path=`` and silently scribbled on the real
# ``models_catalog.json``.  These module-level setUp/tearDown hooks snapshot
# the real catalog before the first test in this module runs and assert it is
# byte-identical afterwards.  If any test (or a future regression) drops the
# ``catalog_path=self.path`` keyword, the suite fails loudly instead of
# corrupting the user's catalog.

_REAL_CATALOG = ROOT / "models_catalog.json"
_REAL_CATALOG_SHA_AT_MODULE_LOAD: str | None = None


def setUpModule() -> None:  # noqa: N802 — unittest hook
    global _REAL_CATALOG_SHA_AT_MODULE_LOAD
    if _REAL_CATALOG.exists():
        _REAL_CATALOG_SHA_AT_MODULE_LOAD = hashlib.sha256(
            _REAL_CATALOG.read_bytes()
        ).hexdigest()


def tearDownModule() -> None:  # noqa: N802 — unittest hook
    if _REAL_CATALOG_SHA_AT_MODULE_LOAD is None:
        return
    current = hashlib.sha256(_REAL_CATALOG.read_bytes()).hexdigest()
    if current != _REAL_CATALOG_SHA_AT_MODULE_LOAD:
        raise AssertionError(
            "test_append_user_model.py mutated the real models_catalog.json — "
            "every call to catalog.append_user_model / save_catalog MUST pass "
            "catalog_path=self.path (see _CatalogTempCase). "
            f"expected SHA256={_REAL_CATALOG_SHA_AT_MODULE_LOAD}, "
            f"got {current}."
        )


def _hf_image_entry(id_="user-img-sample", name="Sample SDXL"):
    """Minimal supported user-added SDXL entry — passes _validate_model
    and exercises every schema field the dialog actually writes."""
    return {
        "id": id_,
        "name": name,
        "vendor": "Acme",
        "category": "Image Generation",
        "description": "Imported test entry.",
        "size_gb": 7.0,
        "min_ram_gb": 16,
        "min_vram_gb": 8,
        "backend": "comfyui",
        "comfyui_model": "sample.safetensors",
        "comfyui_model_dest": "checkpoints",
        "hf_repo": "acme/sample-sdxl",
        "hf_revision": SHA,
        "recommended_settings": {
            "width": 1024, "height": 1024, "aspect": "1:1",
            "sampler": "dpmpp_2m", "scheduler": "karras",
            "steps": 30, "cfg": 7.0, "cfg_locked": False,
            "family_label": "SDXL",
        },
        "perf_profile": {
            "speed_tier": "balanced", "quality_tier": "great",
            "category_bucket": "general", "recommendation": "alternative",
            "speed_label": "~25s", "notes": "test",
        },
        "tags": ["user-added", "image-gen", "sdxl"],
    }


def _ollama_entry(id_="user-ollama-llama32", tag="llama3.2:1b"):
    return {
        "id": id_,
        "name": tag,
        "vendor": "Ollama Library",
        "category": "Small",
        "description": "Imported Ollama tag.",
        "size_gb": 0.0,
        "min_ram_gb": 8,
        "min_vram_gb": 0,
        "ollama_tag": tag,
        "backend": "ollama",
        "tags": ["user-added", "ollama"],
    }


class _CatalogTempCase(unittest.TestCase):
    """Base class that gives every test its own temp catalog path so the
    real models_catalog.json is never touched.

    Two layers of defence:

    1.  Every test SHOULD pass ``catalog_path=self.path`` explicitly — it
        documents the intent at the call site.
    2.  This base class ALSO monkeypatches ``catalog.CATALOG_FILE`` to point
        at ``self.path`` for the duration of every test.  So if a future
        test author forgets the kwarg, ``catalog.append_user_model``'s
        fallback path (``load_catalog(None)`` → ``save_catalog(_, None)``)
        still writes to the tempfile, not the shipped catalog.  The
        ``setUpModule`` / ``tearDownModule`` SHA snapshot above is the
        third safety net that asserts the real file is byte-identical
        before and after the module runs.
    """

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8",
        )
        self.tmp.write("{}")
        self.tmp.close()
        self.path = Path(self.tmp.name)
        self._catalog_file_patch = unittest.mock.patch.object(
            catalog, "CATALOG_FILE", self.path,
        )
        self._catalog_file_patch.start()

    def tearDown(self):
        self._catalog_file_patch.stop()
        try:
            self.path.unlink()
        except OSError:
            pass


class AppendUserModelHappyPath(_CatalogTempCase):
    """Successful adds — covers HF + Ollama + the user_added stamp."""

    def setUp(self):
        super().setUp()
        # Seed an empty catalog file so load_catalog returns []
        # (otherwise it falls back to built-in MODELS and tests get noisy).
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({
                "_readme": "test catalog",
                "version": 1,
                "merge_builtins": False,
                "models": [],
            }, fh)

    def test_user_added_flag_force_set(self):
        # Even if the caller passes user_added=False, the helper overrides it.
        entry = _hf_image_entry()
        entry["user_added"] = False
        ok, final_id, _ = catalog.append_user_model(
            entry, source_url="https://huggingface.co/acme/sample-sdxl",
            catalog_path=self.path, existing_models=[],
        )
        self.assertTrue(ok, final_id)
        reloaded = catalog.load_catalog(self.path)
        match = next(m for m in reloaded if m["id"] == final_id)
        self.assertTrue(match["user_added"])

    def test_source_url_recorded(self):
        entry = _hf_image_entry()
        ok, final_id, _ = catalog.append_user_model(
            entry, source_url="https://huggingface.co/acme/sample-sdxl?utm=demo",
            catalog_path=self.path, existing_models=[],
        )
        self.assertTrue(ok)
        match = next(m for m in catalog.load_catalog(self.path) if m["id"] == final_id)
        self.assertEqual(match["source_url"],
                         "https://huggingface.co/acme/sample-sdxl?utm=demo")

    def test_added_at_stamped(self):
        entry = _hf_image_entry()
        # Caller may pre-stamp added_at; helper preserves it.
        entry["added_at"] = "2025-01-01T00:00:00Z"
        ok, final_id, _ = catalog.append_user_model(
            entry, source_url="x", catalog_path=self.path, existing_models=[],
        )
        self.assertTrue(ok)
        match = next(m for m in catalog.load_catalog(self.path) if m["id"] == final_id)
        self.assertEqual(match["added_at"], "2025-01-01T00:00:00Z")

    def test_added_at_auto_stamped_when_missing(self):
        entry = _hf_image_entry()
        ok, final_id, _ = catalog.append_user_model(
            entry, source_url="x", catalog_path=self.path, existing_models=[],
        )
        self.assertTrue(ok)
        match = next(m for m in catalog.load_catalog(self.path) if m["id"] == final_id)
        self.assertTrue(match.get("added_at", "").endswith("Z"))

    def test_requires_review_flag_recorded(self):
        ok, final_id, _ = catalog.append_user_model(
            _hf_image_entry(), source_url="x", requires_review=True,
            catalog_path=self.path, existing_models=[],
        )
        self.assertTrue(ok)
        match = next(m for m in catalog.load_catalog(self.path) if m["id"] == final_id)
        self.assertTrue(match.get("requires_review"))

    def test_ollama_entry_does_not_require_sha(self):
        ok, final_id, _ = catalog.append_user_model(
            _ollama_entry(), source_url="https://ollama.com/library/llama3.2:1b",
            catalog_path=self.path, existing_models=[],
        )
        self.assertTrue(ok)
        match = next(m for m in catalog.load_catalog(self.path) if m["id"] == final_id)
        self.assertEqual(match["ollama_tag"], "llama3.2:1b")
        # Ollama entries don't get hf_revision stamped — they don't have one.
        self.assertNotIn("hf_revision", match)


class AppendUserModelShaPinning(_CatalogTempCase):
    """The 40-char SHA pin rule (DO NOT REGRESS extended to all HF-backed
    user-added entries, not just trust_remote_code utility loaders)."""

    def test_branch_name_rejected(self):
        entry = _hf_image_entry()
        entry["hf_revision"] = "main"
        ok, msg, _ = catalog.append_user_model(
            entry, source_url="x", existing_models=[], catalog_path=self.path,
        )
        self.assertFalse(ok)
        self.assertIn("40-char", msg)

    def test_short_sha_rejected(self):
        entry = _hf_image_entry()
        entry["hf_revision"] = "abc1234"  # 7-char short SHA
        ok, msg, _ = catalog.append_user_model(
            entry, source_url="x", existing_models=[], catalog_path=self.path,
        )
        self.assertFalse(ok)

    def test_uppercase_sha_normalized(self):
        entry = _hf_image_entry()
        entry["hf_revision"] = SHA.upper()
        ok, final_id, models = catalog.append_user_model(
            entry, source_url="x", existing_models=[], catalog_path=self.path,
        )
        self.assertTrue(ok, final_id)
        match = next(m for m in models if m["id"] == final_id)
        self.assertEqual(match["hf_revision"], SHA)

    def test_missing_sha_rejected(self):
        entry = _hf_image_entry()
        del entry["hf_revision"]
        ok, msg, _ = catalog.append_user_model(
            entry, source_url="x", existing_models=[], catalog_path=self.path,
        )
        self.assertFalse(ok)


class AppendUserModelIdCollisions(_CatalogTempCase):
    """Three collision cases the test-engineer flagged."""

    def test_first_collision_appends_dash_two(self):
        existing = [_hf_image_entry("user-img-sample")]
        ok, final_id, models = catalog.append_user_model(
            _hf_image_entry("user-img-sample"),
            source_url="x", existing_models=existing, catalog_path=self.path,
        )
        self.assertTrue(ok)
        self.assertEqual(final_id, "user-img-sample-2")
        # Original still present.
        self.assertEqual(sum(1 for m in models if m["id"].startswith("user-img-sample")), 2)

    def test_second_collision_appends_dash_three(self):
        existing = [
            _hf_image_entry("user-img-sample"),
            _hf_image_entry("user-img-sample-2"),
        ]
        ok, final_id, _ = catalog.append_user_model(
            _hf_image_entry("user-img-sample"),
            source_url="x", existing_models=existing, catalog_path=self.path,
        )
        self.assertTrue(ok)
        self.assertEqual(final_id, "user-img-sample-3")

    def test_collision_with_builtin_id(self):
        # Pretend a built-in catalog entry shares the proposed id.
        existing = [{"id": "qwen2.5:0.5b", "name": "Qwen 0.5B"}]
        ok, final_id, _ = catalog.append_user_model(
            _ollama_entry("qwen2.5:0.5b"),
            source_url="x", existing_models=existing, catalog_path=self.path,
        )
        self.assertTrue(ok)
        self.assertEqual(final_id, "qwen2.5:0.5b-2")


class AppendUserModelOllamaDedup(_CatalogTempCase):
    """Duplicate user-added Ollama tags should merge, not append -2 rows."""

    def test_duplicate_user_added_ollama_tag_reuses_existing_id(self):
        existing = [_ollama_entry("user-ollama-nemotron3", "nemotron3:33b")]
        existing[0].update({
            "user_added": True,
            "source_url": "https://ollama.com/library/nemotron3",
            "size_gb": 27.0,
            "min_ram_gb": 48,
            "min_vram_gb": 24,
            "context_length": 131072,
            "category": "Extra Large",
            "vendor": "NVIDIA",
            "parameters": "33B",
            "recommended_for": ["GPU Workstation"],
        })
        proposed = _ollama_entry("user-ollama-nemotron3", "nemotron3:33b")
        ok, final_id, models = catalog.append_user_model(
            proposed,
            source_url="https://ollama.com/library/nemotron3:33b",
            existing_models=existing,
            catalog_path=self.path,
        )
        self.assertTrue(ok)
        self.assertEqual(final_id, "user-ollama-nemotron3")
        self.assertEqual(len(models), 1)
        merged = models[0]
        self.assertEqual(merged["source_url"], "https://ollama.com/library/nemotron3:33b")
        self.assertEqual(merged["size_gb"], 27.0)
        self.assertEqual(merged["category"], "Extra Large")

    def test_sparse_existing_ollama_entry_is_upgraded_by_richer_duplicate(self):
        existing = [_ollama_entry("user-ollama-nemotron3", "nemotron3:33b")]
        existing[0].update({
            "user_added": True,
            "source_url": "https://ollama.com/library/nemotron3",
            "size_gb": 0,
            "min_ram_gb": 8,
            "min_vram_gb": 0,
            "context_length": 0,
            "category": "Small",
            "vendor": "Ollama Library",
            "parameters": "",
            "recommended_for": [],
        })
        proposed = _ollama_entry("user-ollama-nemotron3", "nemotron3:33b")
        proposed.update({
            "vendor": "NVIDIA",
            "category": "Extra Large",
            "size_gb": 27.0,
            "min_ram_gb": 48,
            "min_vram_gb": 24,
            "context_length": 131072,
            "parameters": "33B",
            "recommended_for": ["GPU Workstation"],
            "tags": ["user-added", "ollama", "extra-large"],
        })
        ok, final_id, models = catalog.append_user_model(
            proposed,
            source_url="https://ollama.com/library/nemotron3:33b",
            existing_models=existing,
            catalog_path=self.path,
        )
        self.assertTrue(ok)
        self.assertEqual(final_id, "user-ollama-nemotron3")
        self.assertEqual(len(models), 1)
        upgraded = models[0]
        self.assertEqual(upgraded["vendor"], "NVIDIA")
        self.assertEqual(upgraded["category"], "Extra Large")
        self.assertEqual(upgraded["size_gb"], 27.0)
        self.assertEqual(upgraded["context_length"], 131072)
        self.assertEqual(upgraded["parameters"], "33B")


class AppendUserModelValidation(_CatalogTempCase):

    def test_invalid_entry_rejected(self):
        # Missing required fields.
        ok, msg, _ = catalog.append_user_model(
            {"id": "x"}, source_url="y", existing_models=[], catalog_path=self.path,
        )
        self.assertFalse(ok)

    def test_non_dict_entry_rejected(self):
        ok, msg, _ = catalog.append_user_model(
            "not a dict",  # type: ignore[arg-type]
            source_url="x", existing_models=[], catalog_path=self.path,
        )
        self.assertFalse(ok)


class CatalogSchemaRoundTrip(_CatalogTempCase):
    """End-to-end save_catalog / load_catalog preserves the new fields.
    Fixes the false README claim that "Reload Catalog strips fields outside
    this documented schema" — that was never the actual behaviour."""

    def test_user_added_fields_preserved(self):
        entry = _hf_image_entry()
        entry.update({
            "user_added": True,
            "requires_review": True,
            "source_url": "https://huggingface.co/acme/sample-sdxl",
            "added_at": "2025-01-15T12:34:56Z",
            "pipeline_tag": "text-to-image",
        })
        catalog.save_catalog([entry], self.path)
        loaded = catalog.load_catalog(self.path)
        match = next(m for m in loaded if m["id"] == entry["id"])
        for k in ("user_added", "requires_review", "source_url", "added_at", "pipeline_tag"):
            self.assertEqual(match.get(k), entry[k], f"field {k!r} not preserved")

    def test_readme_documents_new_fields(self):
        catalog.save_catalog([], self.path)
        with open(self.path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        readme = payload.get("_readme", "")
        for token in ("user_added", "requires_review", "source_url"):
            self.assertIn(token, readme, f"_readme missing {token!r}")

    def test_readme_no_longer_lies_about_stripping(self):
        catalog.save_catalog([], self.path)
        with open(self.path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        readme = payload.get("_readme", "")
        self.assertNotIn(
            "strips fields outside this documented schema",
            readme,
            "README must not claim the strip behaviour exists (it doesn't).",
        )


class OpenvinoValidation(_CatalogTempCase):
    """OpenVINO entries with ov_repo must pass _validate_model — was a
    silent-drop bug before this drop."""

    def test_openvino_entry_round_trips(self):
        entry = {
            "id": "user-ov-llama",
            "name": "Llama OV",
            "vendor": "OpenVINO",
            "category": "Small",
            "size_gb": 0.6,
            "min_ram_gb": 8,
            "min_vram_gb": 0,
            "ov_repo": "OpenVINO/llama-3.2-1b-int4-ov",
            "backend": "openvino",
            "hf_revision": SHA,
            "tags": ["user-added", "openvino"],
            "user_added": True,
            "source_url": "https://huggingface.co/OpenVINO/llama-3.2-1b-int4-ov",
        }
        catalog.save_catalog([entry], self.path)
        loaded = catalog.load_catalog(self.path)
        self.assertTrue(
            any(m["id"] == "user-ov-llama" for m in loaded),
            "OpenVINO entry with ov_repo must survive _validate_model",
        )


if __name__ == "__main__":
    unittest.main()

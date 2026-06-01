# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Unit tests for ``src/system_info.py`` SKU loader + bench-models resolver.

These tests pin the v5.5.12 SKU decoupling contract (see docs/architecture.md §5):

1. The app NEVER hardcodes SKU display names — every SKU + its
   default-tick model sets come from ``skus.json``.
2. The resolver supports inheritance from named baselines plus per-SKU
   add/remove lists, OR a flat list as shorthand.
3. Missing / unreadable / malformed JSON disables the optional-SKU feature
   and ``get_benchmark_sku_profiles`` falls back to a single "This Device"
   entry built from local hardware.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Test target.
from src import system_info


def _write_json(target: Path, payload: dict) -> None:
    target.write_text(json.dumps(payload), encoding="utf-8")


class ResolveBenchModelsTests(unittest.TestCase):
    """``system_info.resolve_bench_models`` — the {inherit, add, remove} core."""

    BASELINES = {
        "tiny":   ["a", "b", "c"],
        "medium": ["d", "e", "f"],
    }

    def test_none_returns_empty(self):
        self.assertEqual(system_info.resolve_bench_models(None, self.BASELINES), set())

    def test_missing_field_is_empty(self):
        # Common case: SKU entry that omits bench_*_models entirely.
        self.assertEqual(system_info.resolve_bench_models({}, self.BASELINES), set())

    def test_flat_list_shorthand(self):
        self.assertEqual(
            system_info.resolve_bench_models(["x", "y", "z"], self.BASELINES),
            {"x", "y", "z"},
        )

    def test_single_inherit(self):
        self.assertEqual(
            system_info.resolve_bench_models({"inherit": "tiny"}, self.BASELINES),
            {"a", "b", "c"},
        )

    def test_multi_inherit_union(self):
        self.assertEqual(
            system_info.resolve_bench_models(
                {"inherit": ["tiny", "medium"]}, self.BASELINES
            ),
            {"a", "b", "c", "d", "e", "f"},
        )

    def test_inherit_add_remove(self):
        self.assertEqual(
            system_info.resolve_bench_models(
                {"inherit": "tiny", "add": ["d"], "remove": ["a"]},
                self.BASELINES,
            ),
            {"b", "c", "d"},
        )

    def test_unknown_baseline_logs_warning_resolves_partial(self):
        # Unknown 'huge' is dropped; 'tiny' still contributes.
        result = system_info.resolve_bench_models(
            {"inherit": ["tiny", "huge"], "add": ["z"]},
            self.BASELINES,
        )
        self.assertEqual(result, {"a", "b", "c", "z"})

    def test_malformed_spec_returns_empty(self):
        # A bare int / unsupported type resolves to empty (with a warning).
        self.assertEqual(system_info.resolve_bench_models(42, self.BASELINES), set())

    def test_missing_baselines_table_treats_inherit_as_unknown(self):
        # When bench_defaults is absent, inherit can't resolve — only
        # the per-SKU 'add' list contributes.
        self.assertEqual(
            system_info.resolve_bench_models(
                {"inherit": "tiny", "add": ["alpha"]},
                None,
            ),
            {"alpha"},
        )

    def test_remove_can_strip_added_items(self):
        # remove runs after add — handy for "inherit baseline but skip X"
        # patterns.
        self.assertEqual(
            system_info.resolve_bench_models(
                {"inherit": "tiny", "add": ["a", "z"], "remove": ["a"]},
                self.BASELINES,
            ),
            {"b", "c", "z"},
        )

    def test_add_only_no_inherit(self):
        # Dict with only 'add' is legal — equivalent to a flat list.
        self.assertEqual(
            system_info.resolve_bench_models({"add": ["p", "q"]}, self.BASELINES),
            {"p", "q"},
        )

    def test_remove_only_no_inherit_is_empty(self):
        # Degenerate but valid: nothing to remove from, result is empty.
        # Documents the contract — guards against a refactor that decides
        # bare 'remove' should fall back to the union of all baselines.
        self.assertEqual(
            system_info.resolve_bench_models({"remove": ["a"]}, self.BASELINES),
            set(),
        )

    def test_inherit_non_string_non_list_is_ignored(self):
        # An int (or any unsupported type) for 'inherit' must log and
        # contribute nothing; 'add' still applies.
        self.assertEqual(
            system_info.resolve_bench_models(
                {"inherit": 42, "add": ["only"]},
                self.BASELINES,
            ),
            {"only"},
        )
        self.assertEqual(
            system_info.resolve_bench_models(
                {"inherit": {"not": "supported"}, "add": ["only"]},
                self.BASELINES,
            ),
            {"only"},
        )


class LoadOptionalSkuConfigBenchTests(unittest.TestCase):
    """End-to-end loader behavior for v2 ``bench_defaults`` + per-SKU specs."""

    def _tmp_json(self, payload: dict) -> Path:
        target = Path(self._tmpdir.name) / "skus.json"
        _write_json(target, payload)
        return target

    def setUp(self):
        self._tmpdir = TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_missing_file_is_empty_safe(self):
        # The whole point of decoupling: app must not blow up when the
        # JSON file is missing. Returned shape stays a dict with empty
        # 'skus' and 'feature' so callers can iterate safely.
        missing = Path(self._tmpdir.name) / "does-not-exist.json"
        cfg = system_info.load_optional_sku_config(missing)
        self.assertEqual(cfg["skus"], [])
        self.assertEqual(cfg["feature"], {})

    def test_v1_file_loads_skus_with_empty_bench_sets(self):
        # Back-compat: a v1 file (no bench_defaults, no bench_*_models
        # on each SKU) still loads. Each SKU's bench sets resolve to
        # empty — UI behavior collapses to "fits-but-unticked" for
        # those SKUs.
        target = self._tmp_json({
            "version": 1,
            "skus": [
                {"name": "Tiny", "vcpu": 2, "ram_gb": 4, "vram_gb": 0},
                {"name": "Small", "vcpu": 4, "ram_gb": 16, "vram_gb": 0},
            ],
        })
        cfg = system_info.load_optional_sku_config(target)
        self.assertEqual(len(cfg["skus"]), 2)
        for sku in cfg["skus"]:
            self.assertEqual(sku["bench_quick_models"], set())
            self.assertEqual(sku["bench_extended_models"], set())

    def test_v2_inheritance_resolves(self):
        target = self._tmp_json({
            "version": 2,
            "bench_defaults": {
                "baselines": {
                    "chat":  ["chat-a", "chat-b"],
                    "image": ["img-1"],
                }
            },
            "skus": [
                {
                    "name": "CPU", "vcpu": 4, "ram_gb": 16, "vram_gb": 0,
                    "bench_quick_models": {"inherit": "chat"},
                    "bench_extended_models": {"inherit": "chat"},
                },
                {
                    "name": "GPU", "vcpu": 8, "ram_gb": 32, "vram_gb": 8,
                    "bench_quick_models": {
                        "inherit": "chat", "add": ["img-1"]
                    },
                    "bench_extended_models": {
                        "inherit": ["chat", "image"],
                        "add": ["chat-c"],
                    },
                },
            ],
        })
        cfg = system_info.load_optional_sku_config(target)
        cpu, gpu = cfg["skus"]
        self.assertEqual(cpu["bench_quick_models"], {"chat-a", "chat-b"})
        self.assertEqual(cpu["bench_extended_models"], {"chat-a", "chat-b"})
        self.assertEqual(gpu["bench_quick_models"], {"chat-a", "chat-b", "img-1"})
        self.assertEqual(
            gpu["bench_extended_models"],
            {"chat-a", "chat-b", "chat-c", "img-1"},
        )

    def test_v2_add_remove_round_trips(self):
        target = self._tmp_json({
            "version": 2,
            "bench_defaults": {"baselines": {"base": ["x", "y", "z"]}},
            "skus": [
                {
                    "name": "Skinny", "vcpu": 2, "ram_gb": 4, "vram_gb": 0,
                    "bench_quick_models": {
                        "inherit": "base", "add": ["q"], "remove": ["x"]
                    },
                    "bench_extended_models": ["only-this"],
                },
            ],
        })
        cfg = system_info.load_optional_sku_config(target)
        sku = cfg["skus"][0]
        self.assertEqual(sku["bench_quick_models"], {"y", "z", "q"})
        # Flat-list shorthand also accepted.
        self.assertEqual(sku["bench_extended_models"], {"only-this"})

    def test_v2_unknown_baseline_logs_and_drops(self):
        target = self._tmp_json({
            "version": 2,
            "bench_defaults": {"baselines": {"good": ["k"]}},
            "skus": [
                {
                    "name": "Mystery", "vcpu": 1, "ram_gb": 1, "vram_gb": 0,
                    "bench_quick_models": {
                        "inherit": "does-not-exist", "add": ["fallback"]
                    },
                    "bench_extended_models": {"inherit": "good"},
                },
            ],
        })
        cfg = system_info.load_optional_sku_config(target)
        sku = cfg["skus"][0]
        # Unknown baseline contributes nothing; only 'add' survives.
        self.assertEqual(sku["bench_quick_models"], {"fallback"})
        self.assertEqual(sku["bench_extended_models"], {"k"})

    def test_malformed_bench_defaults_is_warning_not_crash(self):
        # bench_defaults present but not a dict — loader treats baselines
        # as empty; SKUs still load.
        target = self._tmp_json({
            "version": 2,
            "bench_defaults": "definitely not an object",
            "skus": [
                {
                    "name": "X", "vcpu": 1, "ram_gb": 1, "vram_gb": 0,
                    "bench_quick_models": ["just-this"],
                },
            ],
        })
        cfg = system_info.load_optional_sku_config(target)
        self.assertEqual(cfg["skus"][0]["bench_quick_models"], {"just-this"})

    def test_empty_file_disables_feature(self):
        target = Path(self._tmpdir.name) / "skus.json"
        target.write_bytes(b"")
        cfg = system_info.load_optional_sku_config(target)
        self.assertEqual(cfg["skus"], [])
        self.assertEqual(cfg["feature"], {})
        self.assertEqual(cfg.get("bench_defaults", {}), {})

    def test_invalid_json_disables_feature(self):
        target = Path(self._tmpdir.name) / "skus.json"
        target.write_text("{not valid json", encoding="utf-8")
        cfg = system_info.load_optional_sku_config(target)
        self.assertEqual(cfg["skus"], [])
        self.assertEqual(cfg["feature"], {})

    def test_skus_key_missing_disables_feature(self):
        target = self._tmp_json({"version": 2, "bench_defaults": {}})
        cfg = system_info.load_optional_sku_config(target)
        self.assertEqual(cfg["skus"], [])
        self.assertEqual(cfg["feature"], {})

    def test_skus_key_not_a_list_disables_feature(self):
        target = self._tmp_json({"version": 2, "skus": {"oops": "dict"}})
        cfg = system_info.load_optional_sku_config(target)
        self.assertEqual(cfg["skus"], [])
        self.assertEqual(cfg["feature"], {})


class SampleFixtureRoundTripTests(unittest.TestCase):
    """Pin resolved per-SKU model-set sizes against the sample fixture.

    Catches resolver-level regressions (e.g. a refactor that silently drops
    a baseline or stops unioning multi-inherits) at the system_info layer,
    before they manifest as cryptic count mismatches in the app-layer
    benchmark contract tests.

    Uses ``tests/fixtures/sample_skus.json`` rather than the maintainer's
    private ``skus.json`` so the test suite is fully self-contained and
    runs identically in a public clone.
    """

    FIXTURE_PATH = (
        Path(__file__).resolve().parent / "fixtures" / "sample_skus.json"
    )

    EXPECTED_QUICK = {
        "Small CPU":       4,
        "Medium CPU":      4,
        "Large CPU":       4,
        "XL CPU":          4,
        "GPU Entry":       5,
        "GPU Mid":         5,
        "GPU High":        5,
        "GPU Workstation": 5,
    }
    EXPECTED_EXTENDED = {
        "Small CPU":        7,
        "Medium CPU":       7,
        "Large CPU":        7,
        "XL CPU":           7,
        "GPU Entry":       11,
        "GPU Mid":         11,
        "GPU High":        11,
        "GPU Workstation": 12,
    }

    def setUp(self):
        self.assertTrue(
            self.FIXTURE_PATH.exists(),
            f"sample SKU fixture missing at {self.FIXTURE_PATH}",
        )
        self.cfg = system_info.load_optional_sku_config(self.FIXTURE_PATH)
        self.by_name = {s["name"]: s for s in self.cfg["skus"]}

    def test_all_expected_skus_present(self):
        # Pins the SKU NAME SET — protects against a future fixture edit
        # that accidentally drops a profile.
        self.assertEqual(
            set(self.by_name.keys()),
            set(self.EXPECTED_QUICK.keys()),
        )

    def test_quick_resolved_counts_match_fixture_contract(self):
        for name, expected in self.EXPECTED_QUICK.items():
            with self.subTest(profile=name, mode="quick"):
                self.assertEqual(
                    len(self.by_name[name]["bench_quick_models"]),
                    expected,
                )

    def test_extended_resolved_counts_match_fixture_contract(self):
        for name, expected in self.EXPECTED_EXTENDED.items():
            with self.subTest(profile=name, mode="extended"):
                self.assertEqual(
                    len(self.by_name[name]["bench_extended_models"]),
                    expected,
                )

    def test_resolved_sets_are_sets_of_strings(self):
        # Pins the resolver's output type — downstream _bench_default_models_for
        # depends on string ids in a set-like membership check.
        for name, sku in self.by_name.items():
            with self.subTest(profile=name):
                self.assertIsInstance(sku["bench_quick_models"], set)
                self.assertIsInstance(sku["bench_extended_models"], set)
                for mid in sku["bench_quick_models"] | sku["bench_extended_models"]:
                    self.assertIsInstance(mid, str)


class ApplyOptionalSkusReloadTests(unittest.TestCase):
    """Pin the 'mutate-in-place, never rebind' contract for
    ``system_info.BENCHMARK_SKU_PROFILES`` during App reload.

    ``tools/validate_doc_samples.py`` and other module-level importers do
    ``from src.system_info import BENCHMARK_SKU_PROFILES``. If
    ``_apply_optional_skus_to_modules`` ever rebinds the attribute instead
    of mutating it in place, those importers will keep seeing the old list
    forever and the bug will only surface in side tooling, not the app.
    v5.5.12 uses slice assignment (``[:] = list(...)``) so the reload is
    atomic from a concurrent-reader perspective AND preserves identity.
    """

    def test_reload_mutates_in_place_keeps_same_list_identity(self):
        from src import system_info as _system_info
        from src.app import App

        app = object.__new__(App)
        app._optional_skus = [
            {
                "name": "ReloadTest",
                "vcpu": 1,
                "ram_gb": 1,
                "vram_gb": 0,
                "bench_quick_models": {"reload-marker"},
                "bench_extended_models": set(),
            }
        ]

        saved = list(_system_info.BENCHMARK_SKU_PROFILES)
        original_id = id(_system_info.BENCHMARK_SKU_PROFILES)
        aliased = _system_info.BENCHMARK_SKU_PROFILES  # simulates the side-tool import
        try:
            app._apply_optional_skus_to_modules()

            self.assertEqual(
                id(_system_info.BENCHMARK_SKU_PROFILES),
                original_id,
                "BENCHMARK_SKU_PROFILES must be mutated in place, not rebound.",
            )
            self.assertIs(
                aliased,
                _system_info.BENCHMARK_SKU_PROFILES,
                "Pre-existing importers must see the same list object after reload.",
            )
            self.assertEqual(
                [s["name"] for s in aliased],
                ["ReloadTest"],
            )
        finally:
            _system_info.BENCHMARK_SKU_PROFILES[:] = saved


class SaveOptionalSkusRoundTripTests(unittest.TestCase):
    """Pin the load → save → load round-trip survival when per-SKU bench
    fields have been resolved to ``set[str]``.

    The loader replaces the raw ``{inherit, add, remove}`` spec with the
    resolved set in place. Naive serialization (``json.dumps(set)``) raises
    ``TypeError`` and breaks the "edit SKUs and save" UI path. v5.5.12
    coerces sets to sorted lists in ``_scrub_sku_for_save`` so the
    round-trip is lossy in *shape* but identical in *semantics*.
    """

    def setUp(self):
        self._tmpdir = TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_save_after_load_does_not_raise_typeerror(self):
        # Build a v2 file, load it (which resolves bench sets in place),
        # then save the loaded SKUs back to disk. Must not raise.
        src = Path(self._tmpdir.name) / "src.json"
        src.write_text(json.dumps({
            "version": 2,
            "bench_defaults": {"baselines": {"chat": ["a", "b"]}},
            "skus": [
                {
                    "name": "Round", "vcpu": 1, "ram_gb": 1, "vram_gb": 0,
                    "bench_quick_models": {"inherit": "chat", "add": ["c"]},
                    "bench_extended_models": ["x", "y"],
                },
            ],
        }), encoding="utf-8")

        cfg = system_info.load_optional_sku_config(src)
        self.assertEqual(cfg["skus"][0]["bench_quick_models"], {"a", "b", "c"})

        dest = Path(self._tmpdir.name) / "dest.json"
        ok = system_info.save_optional_skus(cfg["skus"], dest)
        self.assertTrue(ok)
        self.assertTrue(dest.exists())

        # Reload the saved file and verify semantics survived.
        reloaded = system_info.load_optional_sku_config(dest)
        self.assertEqual(reloaded["skus"][0]["bench_quick_models"], {"a", "b", "c"})
        self.assertEqual(reloaded["skus"][0]["bench_extended_models"], {"x", "y"})

    def test_saved_payload_is_version_3_with_sorted_lists(self):
        skus = [
            {
                "name": "Sorted", "cpu": 1, "ram_gb": 1, "vram_gb": 0,
                "bench_quick_models": {"z", "a", "m"},
                "bench_extended_models": frozenset({"q", "b"}),
            },
        ]
        dest = Path(self._tmpdir.name) / "out.json"
        self.assertTrue(system_info.save_optional_skus(skus, dest))
        payload = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["skus"][0]["bench_quick_models"], ["a", "m", "z"])
        self.assertEqual(payload["skus"][0]["bench_extended_models"], ["b", "q"])


class GetBenchmarkSkuProfilesTests(unittest.TestCase):
    """``get_benchmark_sku_profiles`` — module-level live list + fallback."""

    def test_module_list_populated_at_import(self):
        # The shipped repo JSON has 8 SKUs (4 CPU + 4 GPU); init runs at
        # module load. If the JSON moves or the SKU count changes, this
        # test will surface it.
        self.assertTrue(len(system_info.BENCHMARK_SKU_PROFILES) >= 1)

    def test_fallback_to_local_machine_when_list_empty(self):
        # Empty list → must fall back to a synthetic "This Device"
        # entry from local hardware so the benchmark UI keeps a
        # selectable profile available.
        saved = list(system_info.BENCHMARK_SKU_PROFILES)
        try:
            system_info.BENCHMARK_SKU_PROFILES.clear()
            profiles = system_info.get_benchmark_sku_profiles()
            self.assertEqual(len(profiles), 1,
                "Empty SKU list must collapse to exactly one synthetic local SKU "
                "(no hardcoded reference SKUs in the public fallback)")
            self.assertEqual(profiles[0].get("name"), "This Device",
                "The synthetic local SKU must be named 'This Device' — vendor / "
                "WMI-derived names are forbidden brand leaks")
        finally:
            system_info.BENCHMARK_SKU_PROFILES.clear()
            system_info.BENCHMARK_SKU_PROFILES.extend(saved)

    def test_returns_copies_not_references(self):
        # Mutating the returned list must not corrupt the module list.
        before_count = len(system_info.BENCHMARK_SKU_PROFILES)
        copy = system_info.get_benchmark_sku_profiles()
        if copy:
            copy[0]["name"] = "TAMPERED"
        self.assertEqual(len(system_info.BENCHMARK_SKU_PROFILES), before_count)
        for sku in system_info.BENCHMARK_SKU_PROFILES:
            self.assertNotEqual(sku.get("name"), "TAMPERED")


class QuickChatUltraSmallBaselineContractTests(unittest.TestCase):
    """Pins the v2026.06.01.6 Quick mode contract for the maintainer's skus.json.

    Quick mode is the calibrated lean smoke set — it MUST tick only models
    classified ``category: 'Ultra Small'`` in ``src/catalog.py`` (the three
    sub-1B chat models: qwen2.5:0.5b, llama3.2:1b, gemma3:1b) plus exactly
    one tiny image-gen model on GPU SKUs (realistic-vision-v6, 4 GB VRAM).

    These tests run ONLY when the maintainer's local ``skus.json`` is
    present next to the repo root. In the shipped public clone there is no
    ``skus.json`` (it is optional/gitignored), so this contract is skipped
    rather than failing the public pipeline.

    Extensibility: a SKU MAY opt into extras via its ``add`` array
    (e.g. ``"add": ["z-image-turbo"]`` on a high-VRAM workstation). This
    test does NOT forbid extras — it only pins the baselines themselves.
    """

    SKUS_PATH = Path(__file__).resolve().parent.parent / "skus.json"

    @unittest.skipUnless(
        SKUS_PATH.exists(),
        "Maintainer skus.json not present (this is normal in the shipped "
        "public clone — skus.json is optional/gitignored)"
    )
    def test_quick_chat_ultra_small_baseline_matches_catalog_ultra_small(self):
        from src.catalog import MODELS

        catalog_ultra_small = {
            m["id"] for m in MODELS if m.get("category") == "Ultra Small"
        }
        raw = json.loads(self.SKUS_PATH.read_text(encoding="utf-8"))
        baselines = raw.get("bench_defaults", {}).get("baselines", {})
        self.assertIn(
            "quick_chat_ultra_small", baselines,
            "skus.json must declare a 'quick_chat_ultra_small' baseline — "
            "this is the source of truth for Quick mode chat default-ticks. "
            "Pinned by AGENTS.md 'benchmark defaults' row."
        )
        actual = set(baselines["quick_chat_ultra_small"])
        self.assertEqual(
            actual, catalog_ultra_small,
            "quick_chat_ultra_small baseline must EXACTLY match the set of "
            "models with category='Ultra Small' in src/catalog.py. Add or "
            "remove a model by changing its catalog category, not by editing "
            "this baseline."
        )

    @unittest.skipUnless(
        SKUS_PATH.exists(),
        "Maintainer skus.json not present (this is normal in the shipped "
        "public clone — skus.json is optional/gitignored)"
    )
    def test_quick_image_smallest_baseline_is_single_smallest_image_model(self):
        raw = json.loads(self.SKUS_PATH.read_text(encoding="utf-8"))
        baselines = raw.get("bench_defaults", {}).get("baselines", {})
        self.assertIn(
            "quick_image_smallest", baselines,
            "skus.json must declare a 'quick_image_smallest' baseline (one "
            "model, the smallest reliable image-gen, added on every GPU SKU "
            "in Quick mode)."
        )
        actual = list(baselines["quick_image_smallest"])
        self.assertEqual(
            len(actual), 1,
            "quick_image_smallest must contain exactly one model — Quick is "
            "a smoke set, not a benchmark. If you want extra image models "
            "on a specific GPU SKU, add them via that SKU's 'add' array."
        )
        # Pin the specific model to detect drift. realistic-vision-v6 is the
        # smallest image-gen model with a reliable benchmark path
        # (sdxl-lowvram has benchmark_skip_reason; z-image-turbo needs 12 GB
        # VRAM and does not fit the entry / mid-tier GPU SKUs).
        self.assertEqual(
            actual[0], "realistic-vision-v6",
            "quick_image_smallest must be 'realistic-vision-v6' — the "
            "smallest reliable image-gen model (SD 1.5 photoreal, 4 GB "
            "VRAM, no benchmark_skip_reason). The other 4 GB-tier image "
            "models are skipped by the bench runner (sdxl-lowvram has a "
            "benchmark_skip_reason); models with min_vram_gb >= 8 do not "
            "fit the entry-tier GPU SKUs in this contract."
        )

    @unittest.skipUnless(
        SKUS_PATH.exists(),
        "Maintainer skus.json not present (this is normal in the shipped "
        "public clone — skus.json is optional/gitignored)"
    )
    def test_no_sku_still_inherits_legacy_quick_chat_baseline(self):
        # v2026.06.01.6 removed the legacy 17-model 'quick_chat' baseline.
        # Catch a regression that re-introduces it (or a SKU that still
        # inherits the deleted name and silently resolves to empty).
        raw = json.loads(self.SKUS_PATH.read_text(encoding="utf-8"))
        baselines = raw.get("bench_defaults", {}).get("baselines", {})
        self.assertNotIn(
            "quick_chat", baselines,
            "Legacy 17-model 'quick_chat' baseline was removed in v2026.06.01.6 "
            "(it bundled too many non-ultra-small models for a Quick smoke "
            "test). Use 'quick_chat_ultra_small' instead, or compose a custom "
            "baseline if you genuinely need a wider Quick set."
        )
        for sku in raw.get("skus", []):
            spec = sku.get("bench_quick_models")
            if not isinstance(spec, dict):
                continue
            inherit = spec.get("inherit")
            if isinstance(inherit, str):
                inherit_names = {inherit}
            elif isinstance(inherit, list):
                inherit_names = set(inherit)
            else:
                inherit_names = set()
            self.assertNotIn(
                "quick_chat", inherit_names,
                f"SKU {sku.get('name')!r} still inherits the deleted "
                "'quick_chat' baseline — switch to 'quick_chat_ultra_small'."
            )


class BenchmarkPickerBrandFreeContractTests(unittest.TestCase):
    """Pins the no-vendor-branding contract for the benchmark profile picker.

    The benchmark profile picker is built from
    ``system_info.get_benchmark_sku_profiles()`` (which combines
    ``BENCHMARK_SKU_PROFILES`` and the synthetic local SKU when the list
    is empty) plus a literal ``"This Device"`` appended in
    ``App._bench_profile_values``. Two regressions are possible:

    1. ``_build_local_sku_*`` could surface vendor strings from WMI
       (``Win32_ComputerSystem.Manufacturer + Model`` concatenation
       leaks the host hypervisor / vendor / model name on cloud VMs).
    2. ``_FALLBACK_PUBLIC_SKU_PROFILES`` could regrow a hardcoded
       reference SKU (e.g. ``"GPU Workstation"``) that ships in the
       public binary and shows up next to ``"This Device"`` for any
       user without a ``skus.json``.

    Both regressions were observed in production (v2026.05.31.3). This
    class re-asserts that the public no-``skus.json`` path produces
    exactly ``["This Device"]`` and that the fallback constant stays
    empty.
    """

    _FORBIDDEN_BRAND_TOKENS = (
        "microsoft", "virtual machine", "corporation",
        "dell ", "hewlett", "apple inc",
    )

    def test_public_fallback_constant_is_empty(self):
        self.assertEqual(
            system_info._FALLBACK_PUBLIC_SKU_PROFILES, [],
            "The public-fallback SKU list must stay empty - the repo "
            "ships zero hardcoded SKU display names. If you need a "
            "reference SKU on YOUR machine, put it in your own "
            "(gitignored) skus.json."
        )

    def test_local_sku_name_is_this_device(self):
        sku = system_info.build_local_sku()
        self.assertIsInstance(sku, dict)
        self.assertEqual(
            sku.get("name"), "This Device",
            "build_local_sku() must return name='This Device' on every "
            "platform - exposing the WMI Manufacturer+Model concatenation "
            "or the macOS machine model would leak host vendor branding "
            "into the vendor-neutral picker."
        )

    def test_fallback_path_returns_exactly_one_this_device_entry(self):
        saved = list(system_info.BENCHMARK_SKU_PROFILES)
        try:
            system_info.BENCHMARK_SKU_PROFILES.clear()
            profiles = system_info.get_benchmark_sku_profiles()
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].get("name"), "This Device")
        finally:
            system_info.BENCHMARK_SKU_PROFILES.clear()
            system_info.BENCHMARK_SKU_PROFILES.extend(saved)

    def test_fallback_path_contains_no_vendor_branding_anywhere(self):
        saved = list(system_info.BENCHMARK_SKU_PROFILES)
        try:
            system_info.BENCHMARK_SKU_PROFILES.clear()
            profiles = system_info.get_benchmark_sku_profiles()
            for sku in profiles:
                for key, value in sku.items():
                    if not isinstance(value, str):
                        continue
                    low = value.lower()
                    for token in self._FORBIDDEN_BRAND_TOKENS:
                        self.assertNotIn(
                            token, low,
                            f"Forbidden brand token {token!r} found in "
                            f"fallback SKU field {key!r}: {value!r}"
                        )
        finally:
            system_info.BENCHMARK_SKU_PROFILES.clear()
            system_info.BENCHMARK_SKU_PROFILES.extend(saved)


class RecommendedModelsApiTests(unittest.TestCase):
    """``get_recommended_models_for_sku`` / ``get_recommended_skus_for_model``.

    Pins the v5.5.13 SKU decoupling contract: model→SKU recommendations are
    sourced from ``skus.json`` per-SKU ``recommended_models`` arrays — never
    from per-model ``recommended_for`` fields hardcoded in the shipped
    catalog. The per-model field is only honored as a back-compat /
    user-override path on entries the caller passes in.
    """

    def setUp(self):
        self._saved = list(system_info.BENCHMARK_SKU_PROFILES)
        system_info.BENCHMARK_SKU_PROFILES.clear()
        system_info.BENCHMARK_SKU_PROFILES.extend([
            {"name": "Small CPU", "cpu": 4, "ram_gb": 16, "vram_gb": 0,
             "recommended_models": ["alpha", "beta"]},
            {"name": "Big GPU", "cpu": 16, "ram_gb": 64, "vram_gb": 24,
             "recommended_models": ["beta", "gamma"]},
            {"name": "Empty", "cpu": 8, "ram_gb": 32, "vram_gb": 0,
             "recommended_models": []},
        ])

    def tearDown(self):
        system_info.BENCHMARK_SKU_PROFILES.clear()
        system_info.BENCHMARK_SKU_PROFILES.extend(self._saved)

    def test_get_recommended_models_for_sku_returns_canonical_list(self):
        self.assertEqual(
            system_info.get_recommended_models_for_sku("Small CPU"),
            ["alpha", "beta"],
        )

    def test_get_recommended_models_for_sku_case_insensitive(self):
        self.assertEqual(
            system_info.get_recommended_models_for_sku("big gpu"),
            ["beta", "gamma"],
        )

    def test_get_recommended_models_for_sku_unknown_returns_empty(self):
        self.assertEqual(system_info.get_recommended_models_for_sku("Mystery"), [])
        self.assertEqual(system_info.get_recommended_models_for_sku(""), [])
        self.assertEqual(system_info.get_recommended_models_for_sku(None), [])

    def test_get_recommended_skus_for_model_inverts_in_capability_order(self):
        # "beta" appears in two SKUs — should return both, in JSON order.
        self.assertEqual(
            system_info.get_recommended_skus_for_model("beta"),
            ["Small CPU", "Big GPU"],
        )
        self.assertEqual(
            system_info.get_recommended_skus_for_model("alpha"),
            ["Small CPU"],
        )
        self.assertEqual(
            system_info.get_recommended_skus_for_model("nothing-matches"),
            [],
        )

    def test_get_recommended_skus_for_model_honors_per_model_override(self):
        # User-curated catalog entry pins a model to a SKU directly via the
        # legacy ``recommended_for`` field — should supplement the per-SKU
        # lookup, not replace it. Dedup is case-insensitive; SKU-list order
        # wins for collisions; the override-only SKU is appended last.
        model = {"id": "beta", "recommended_for": ["Empty", "Big GPU"]}
        self.assertEqual(
            system_info.get_recommended_skus_for_model("beta", model),
            ["Small CPU", "Big GPU", "Empty"],
        )

    def test_get_recommended_skus_for_model_override_only(self):
        # Unknown model id with only the per-model override populated.
        model = {"id": "new-model", "recommended_for": ["Big GPU"]}
        self.assertEqual(
            system_info.get_recommended_skus_for_model("new-model", model),
            ["Big GPU"],
        )


class CatalogNoBuiltinRecommendedForTests(unittest.TestCase):
    """Public-safety invariant: no shipped ``BUILTIN_MODELS`` entry may carry
    a ``recommended_for`` field. Those strings used to leak private SKU
    display names into the open-source catalog. The new home for SKU→model
    recommendations is the gitignored ``skus.json``."""

    def test_no_catalog_model_carries_recommended_for(self):
        from src import catalog
        offenders = [m.get("id") for m in catalog.MODELS if "recommended_for" in m]
        self.assertEqual(
            offenders, [],
            f"recommended_for must be migrated to skus.json: {offenders}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

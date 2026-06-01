"""Contract tests for the v5.3.4 consolidated Model Guide generator.

The four legacy HTML guides (ChatPromptIdeas.html, ImageGenPrompts.html,
ModelDemoPrompts.html, model-value-props.html) were retired and collapsed
into ``docs/Model-Guide.html``. The new generator lives in
``src/model_guide.py`` and is invoked from ``tools/build_model_demo_docs.py``.

These tests pin the contracts that the four legacy callsites in src/app.py
rely on (Model-Guide.html title, dark default, per-model anchors, surface
tabs, CFG-locked rendering, workflow-surface samples-without-copy-buttons).
"""

import re
import unittest
from pathlib import Path

from src import catalog, model_demos
from src.model_guide import build_model_guide_html


ROOT = Path(__file__).resolve().parents[1]


def _card_for(html: str, model_id: str) -> str:
    """Return the rendered <article class="model-card"> for the given catalog id."""
    fragment = model_demos.doc_fragment(model_id)
    pattern = (
        r'<article class="model-card" id="' + re.escape(fragment) + r'"'
        r'[\s\S]*?</article>'
    )
    match = re.search(pattern, html)
    if not match:
        raise AssertionError(f"No model-card rendered for catalog id {model_id!r}")
    return match.group(0)


class ModelGuideBuilderTests(unittest.TestCase):
    def test_empty_catalog_returns_valid_dark_themed_guide(self):
        html = build_model_guide_html([])

        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("<title>LocalAI Studio — Model Guide</title>", html)
        # Dark is the brand default. The head-script sets data-theme="dark"
        # at first paint when no ?clawpilotTheme override is supplied.
        self.assertIn("setAttribute('data-theme', 'dark')", html)
        # The legacy filenames must not appear in the generated guide — they
        # were retired to Archive/doc-consolidation-2026-05/.
        for legacy in (
            "ChatPromptIdeas.html",
            "ImageGenPrompts.html",
            "ModelDemoPrompts.html",
            "model-value-props.html",
        ):
            self.assertNotIn(legacy, html, f"legacy filename {legacy} leaked into guide")
        # Sanity: no model cards rendered for an empty catalog.
        self.assertNotIn('<article class="model-card"', html)

    def test_archived_model_demo_prompt_images_resolve_to_live_docs_assets(self):
        archive = ROOT / "Archive" / "doc-consolidation-2026-05" / "ModelDemoPrompts.html"
        if not archive.exists():
            # The doc-consolidation-2026-05 archive folder is a one-time
            # historical artifact captured during the v5.4.x docs cleanup.
            # Routine Archive pruning (e.g. v5.5.8 cleanup recycled the
            # 20 oldest backup folders) deletes it, at which point this
            # test has nothing to validate — there's no live doc page that
            # depends on the archive. Skip cleanly rather than fail.
            self.skipTest(
                f"Archived doc not present (likely pruned by routine "
                f"Archive cleanup): {archive}"
            )
        html = archive.read_text(encoding="utf-8")
        self.assertNotIn('src="images/model_demos/', html)
        self.assertNotIn('href="images/model_demos/', html)

        image_refs = re.findall(r'(?:src|href)="([^"]*model_demos/[^"]+\.jpg)"', html)
        self.assertTrue(image_refs)
        missing = []
        for ref in image_refs:
            target = archive.parent
            for part in ref.split("/"):
                target = target / part
            if not target.resolve().exists():
                missing.append(ref)
        self.assertEqual(missing, [])

    def test_every_active_catalog_model_renders_a_card(self):
        models = catalog.load_catalog(ROOT / "models_catalog.json")
        self.assertTrue(models, "active catalog should not be empty")

        html = build_model_guide_html(models)
        rendered_ids = set(re.findall(r'data-model-id="([^"]+)"', html))
        expected_ids = {m["id"] for m in models}
        missing = sorted(expected_ids - rendered_ids)
        self.assertEqual(missing, [], f"missing model cards: {missing}")

        # Each card's id must match doc_fragment(catalog_id) so legacy
        # ``#model-<slug>`` deep-links from the app continue to resolve.
        for model in models:
            fragment = model_demos.doc_fragment(model["id"])
            self.assertIn(
                f'<article class="model-card" id="{fragment}"',
                html,
                f"missing/ill-formed card id for {model['id']}",
            )

    def test_retired_image_models_do_not_render_or_remain_in_catalog(self):
        retired_ids = {"chroma-radiance", "hassanblend-hassanblend1.4"}
        models = catalog.load_catalog(ROOT / "models_catalog.json")
        active_ids = {m["id"] for m in models}
        html = build_model_guide_html(models)

        self.assertFalse(retired_ids & active_ids)
        for retired_id in retired_ids:
            self.assertNotIn(retired_id, html)
            self.assertNotIn(model_demos.doc_fragment(retired_id), html)

    def test_surface_tabs_expose_aria_tablist_contract(self):
        models = catalog.load_catalog(ROOT / "models_catalog.json")
        html = build_model_guide_html(models)

        # The surface tab strip is a wai-aria tablist.
        self.assertIn('role="tablist"', html)
        tabs = re.findall(
            r'<button class="surface-tab[^"]*"[^>]*role="tab"[^>]*aria-selected="(true|false)"',
            html,
        )
        # "All" + one tab per non-empty surface (chat/vision/image/speech/embed/doc).
        self.assertGreaterEqual(len(tabs), 2, f"too few surface tabs: {tabs}")
        # Exactly one tab is initially selected (the "All" tab).
        self.assertEqual(tabs.count("true"), 1, f"expected exactly one aria-selected tab, got {tabs}")

    def test_cfg_locked_flux_card_renders_negative_prompts_ignored_panel(self):
        models = catalog.load_catalog(ROOT / "models_catalog.json")
        flux_models = [
            m for m in models
            if "flux" in (m.get("id") or "").lower()
            and (m.get("recommended_settings") or {}).get("cfg_locked")
        ]
        self.assertTrue(flux_models, "active catalog should include at least one CFG-locked Flux model")

        html = build_model_guide_html(models)
        for model in flux_models:
            card = _card_for(html, model["id"])
            self.assertIn(
                "negative prompts ignored",
                card,
                f"{model['id']}: CFG-locked Flux card should render the 'negative prompts ignored' panel",
            )
            self.assertIn("neg-locked", card, f"{model['id']}: missing neg-locked class")
            # CFG-locked cards must NOT render a copyable negative-prompt block.
            self.assertNotIn('class="neg-text"', card, f"{model['id']}: CFG-locked card leaked a neg-text block")

    def test_workflow_surface_samples_render_without_copy_buttons(self):
        # speech / embed / doc surfaces render their "samples" as workflow
        # instructions (not pasteable prompts), so each sample becomes a
        # `<div class="notes workflow-note">` without a Copy button.
        models = catalog.load_catalog(ROOT / "models_catalog.json")
        html = build_model_guide_html(models)

        workflow_surfaces = ("speech", "embed", "doc")
        for surface in workflow_surfaces:
            section_match = re.search(
                r'<section class="surface-section" data-surface="' + surface + r'"'
                r'[\s\S]*?</section>',
                html,
            )
            self.assertIsNotNone(section_match, f"missing surface-section for {surface}")
            section = section_match.group(0)

            # Split into individual card bodies and inspect each card's prompts block.
            cards = re.findall(
                r'<article class="model-card"[\s\S]*?(?=<article class="model-card"|</section>)',
                section,
            )
            self.assertTrue(cards, f"{surface} section has no cards")
            for card in cards:
                prompts_match = re.search(r'<div class="prompts">([\s\S]*?)</div></div></article>', card)
                # Fall back to a looser match if needed (the prompts div is the
                # last block before card-body close + article close).
                prompts_html = prompts_match.group(1) if prompts_match else card
                self.assertIn(
                    'class="notes workflow-note"',
                    prompts_html,
                    f"{surface} cards must render samples as workflow-note divs",
                )
                self.assertNotIn(
                    'class="copy-btn"',
                    prompts_html,
                    f"{surface} cards must not render Copy buttons inside their samples",
                )

    def test_image_samples_avoid_known_bad_benchmark_prompts(self):
        bytedance = model_demos.get_model_demo({"id": "bytedance-sdxl-lightning", "name": "SDXL Lightning"})
        realistic = model_demos.get_model_demo({"id": "realistic-vision-v6", "name": "Realistic Vision"})
        playground = model_demos.get_model_demo({
            "id": "playgroundai-playground-v2.5-1024px-aesthetic",
            "name": "Playground",
        })

        self.assertIn("Bengal tiger", bytedance["samples"][1])
        self.assertIn("crystal-clear water droplets", bytedance["samples"][1])
        self.assertNotIn("road cyclist", bytedance["samples"][1].lower())
        self.assertIn("black-basalt sea cave", realistic["samples"][0])
        self.assertIn("bioluminescent waves", realistic["samples"][0])
        self.assertIn("no readable text", realistic["samples"][0])
        self.assertIn("no hands", realistic["samples"][0])
        self.assertNotIn("Asian woman", realistic["samples"][0])
        self.assertIn("cyan-blue eyes", playground["samples"][2])
        self.assertIn("face painting", playground["samples"][2])
        self.assertIn("dark teal background", playground["samples"][2])
        self.assertIn("visible pores", playground["samples"][2])
        self.assertIn("individual hair strands", playground["samples"][2])
        self.assertNotIn("coastal cottage", playground["samples"][2].lower())


class ModelGuideRound3Tests(unittest.TestCase):
    """Round 3 (2026-05-19) fixes — SKU chip strip, goal gating, sticky tabs."""

    @classmethod
    def setUpClass(cls):
        models = catalog.load_catalog(ROOT / "models_catalog.json")
        cls.html = build_model_guide_html(models)

    def test_sku_chip_strip_renders_six_chips_with_correct_attributes(self):
        chip_block = re.search(
            r'<div class="sku-strip"[\s\S]*?</div>\s*</header>',
            self.html,
        )
        self.assertIsNotNone(chip_block, "SKU chip strip not found inside header")
        block = chip_block.group(0)

        skus = re.findall(r'data-sku="([^"]+)"', block)
        self.assertEqual(
            skus,
            ["all", "cpu16", "cpu32", "cpu64", "4", "8", "16", "24"],
            f"SKU chips must appear in canonical order (All + 3 CPU RAM tiers + GPU tiers); got {skus}",
        )

        all_chip = re.search(
            r'<button class="sku-chip active"[^>]*data-sku="all"[^>]*aria-pressed="true"',
            block,
        )
        self.assertIsNotNone(all_chip, "'all' chip must render as the default active chip")

        for sku in ("cpu16", "cpu32", "cpu64", "4", "8", "16", "24"):
            self.assertRegex(
                block,
                r'<button class="sku-chip"[^>]*data-sku="' + sku + r'"[^>]*aria-pressed="false"',
                f"'{sku}' chip must start unpressed",
            )

    def test_sku_chip_data_sku_values_are_valid_hardware_select_options(self):
        select_block = re.search(
            r'<select id="hardware"[^>]*>([\s\S]*?)</select>',
            self.html,
        )
        self.assertIsNotNone(select_block, "#hardware <select> not found")
        option_values = set(re.findall(r'<option value="([^"]+)"', select_block.group(1)))

        chip_skus = set(re.findall(r'data-sku="([^"]+)"', self.html))
        self.assertTrue(
            chip_skus.issubset(option_values),
            f"Every chip data-sku must map to a #hardware <option>; missing: {chip_skus - option_values}",
        )

    def test_gate_goal_options_js_constant_and_function_are_present(self):
        # The CPU chip's user-visible effect: hide image-gen goals from #goal.
        # The constant defines which goal values to hide.
        self.assertRegex(
            self.html,
            r"IMAGE_GEN_GOALS\s*=\s*\[[^\]]*['\"]photo['\"][^\]]*['\"]fast['\"][^\]]*['\"]anime['\"][^\]]*['\"]art['\"]",
            "IMAGE_GEN_GOALS constant must list photo/fast/anime/art",
        )
        self.assertIn(
            "function gateGoalOptions",
            self.html,
            "gateGoalOptions() function must exist",
        )
        self.assertIn(
            "hw.indexOf('cpu') === 0",
            self.html,
            "gateGoalOptions/applyFilters must branch on any CPU RAM tier chip (cpu16/cpu32/cpu64)",
        )
        self.assertIn(
            "gateGoalOptions(hw)",
            self.html,
            "gateGoalOptions must be invoked from applyFilters()",
        )

    def test_wire_sticky_offsets_sets_tabs_top_css_variable(self):
        # Surface-tabs sticky-positioning relies on --tabs-top being set by JS
        # at runtime. If the variable name drifts, tabs overlap the filter bar
        # in-browser silently.
        self.assertRegex(
            self.html,
            r"\.surface-tabs\s*\{[^}]*top:\s*var\(--tabs-top",
            ".surface-tabs must position via var(--tabs-top, ...)",
        )
        self.assertRegex(
            self.html,
            r"setProperty\(['\"]--tabs-top['\"]\s*,",
            "wireStickyOffsets() must setProperty('--tabs-top', ...)",
        )

    def test_compact_header_toggle_renders_and_persists(self):
        # The compact-header toggle was added on 2026-05-19 so the tall hero
        # (title + subtitle + SKU strip) can be collapsed to a tiny strip,
        # giving cards more screen real-estate. State persists to
        # localStorage and is restored by HEAD_SCRIPT before first paint to
        # avoid a flash of the tall header.
        self.assertRegex(
            self.html,
            r'<button class="compact-toggle" id="compact-toggle"',
            "compact-toggle button must render in the header controls",
        )
        self.assertIn(
            "function wireCompactToggle",
            self.html,
            "wireCompactToggle() function must exist",
        )
        self.assertIn(
            "wireCompactToggle();",
            self.html,
            "wireCompactToggle() must be called from init()",
        )
        self.assertIn(
            "localai-model-guide-compact",
            self.html,
            "compact-toggle state must persist via localStorage key 'localai-model-guide-compact'",
        )
        # HEAD_SCRIPT must restore the compact state before first paint —
        # otherwise the tall header flashes briefly before collapsing.
        self.assertRegex(
            self.html,
            r"localStorage\.getItem\(['\"]localai-model-guide-compact['\"]\)",
            "HEAD_SCRIPT must read the compact preference before first paint",
        )
        self.assertRegex(
            self.html,
            r"setAttribute\(['\"]data-compact['\"]\s*,\s*['\"]1['\"]\)",
            "HEAD_SCRIPT must set data-compact='1' on <html> when stored",
        )
        # CSS rules must use the [data-compact="1"] attribute selector to
        # condense the header.
        for selector in (
            '[data-compact="1"] header.top',
            '[data-compact="1"] .title-block p',
            '[data-compact="1"] .sku-label',
        ):
            with self.subTest(selector=selector):
                self.assertIn(
                    selector,
                    self.html,
                    f"compact-mode CSS rule for {selector!r} must exist",
                )

    def test_rail_click_and_deep_link_use_explicit_sticky_aware_scroll(self):
        # The previous single-rAF + card.scrollIntoView({block:'start'})
        # implementation could land the card behind the sticky stack because
        # (a) the card-expand reflow from setCardCollapsed and (b) the
        # surface-filter reflow from applyFilters() both shifted the card
        # after the rAF measurement. The replacement uses double-rAF +
        # explicit math against --sticky-top so both the rail-click path
        # and the URL-deep-link path land correctly.
        self.assertIn(
            "function scrollCardIntoView",
            self.html,
            "scrollCardIntoView() helper must exist",
        )
        self.assertRegex(
            self.html,
            r"requestAnimationFrame\(\s*\(\)\s*=>\s*requestAnimationFrame\(",
            "scrollCardIntoView must use double requestAnimationFrame to wait for reflow",
        )
        self.assertRegex(
            self.html,
            r"getPropertyValue\(['\"]--sticky-top['\"]\)",
            "scrollCardIntoView must read --sticky-top from runtime CSS, not hardcode 168px",
        )
        self.assertRegex(
            self.html,
            r"window\.scrollTo\(\{\s*top:\s*Math\.max\(0,\s*targetTop",
            "scrollCardIntoView must use window.scrollTo({top: Math.max(0, targetTop), ...})",
        )
        # Both call sites — rail-link clicks AND deep-link focus — must
        # route through scrollCardIntoView so the two paths can never drift.
        self.assertRegex(
            self.html,
            r"function wireRailLinks\(\)[\s\S]*?scrollCardIntoView\(card\)",
            "wireRailLinks() must call scrollCardIntoView(card)",
        )
        self.assertRegex(
            self.html,
            r"function focusDeepLinkCard\(card\)[\s\S]*?scrollCardIntoView\(card\)",
            "focusDeepLinkCard() must call scrollCardIntoView(card)",
        )
        # The old single-rAF + scrollIntoView pattern must not come back —
        # it's the regression we just fixed.
        self.assertNotRegex(
            self.html,
            r"requestAnimationFrame\(\s*\(\)\s*=>\s*\{\s*card\.scrollIntoView",
            "single-rAF card.scrollIntoView pattern must not be re-introduced",
        )

    def test_deep_links_override_conflicting_filters_before_focusing(self):
        # App links include system filters (hardware/surface/goal) plus model
        # deep-link params. The target model must still be reachable even when
        # those filters would otherwise hide it.
        self.assertIn(
            "function ensureDeepLinkCardVisible",
            self.html,
            "ensureDeepLinkCardVisible() helper must exist",
        )
        self.assertRegex(
            self.html,
            r"function applyUrlParams\(\)[\s\S]*?ensureDeepLinkCardVisible\(card\)[\s\S]*?applyFilters\(\)[\s\S]*?focusDeepLinkCard\(card\)",
            "applyUrlParams() must reveal deep-link cards before focusing them",
        )
        self.assertRegex(
            self.html,
            r"function handleHashDeepLink\(\)[\s\S]*?ensureDeepLinkCardVisible\(card\)[\s\S]*?applyFilters\(\)[\s\S]*?focusDeepLinkCard\(card\)",
            "hash deep links must also reveal cards before focusing",
        )
        self.assertIn(
            "window.addEventListener('hashchange', handleHashDeepLink);",
            self.html,
            "init() must wire hashchange deep-link handling",
        )


if __name__ == "__main__":
    unittest.main()

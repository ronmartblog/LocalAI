# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Static contract tests for src/a11y.py — the keyboard-accessibility patch
layer that makes every CustomTkinter interactive widget reachable via Tab,
activatable via Enter / Space, and adjustable via arrow keys.

These tests pin the contract WITHOUT actually instantiating widgets (no
``Tk()`` root needed) by inspecting the module source. A separate runtime
verification path lives in ``tools/verify_a11y.py``.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
A11Y_PATH = REPO_ROOT / "src" / "a11y.py"
APP_PATH = REPO_ROOT / "src" / "app.py"


class A11yModuleContract(unittest.TestCase):
    """Pin the public surface of src/a11y.py so future refactors can't quietly
    drop a widget category from the keyboard-accessibility patch."""

    @classmethod
    def setUpClass(cls):
        cls.source = A11Y_PATH.read_text(encoding="utf-8")

    def test_module_exists_and_has_install_entry_point(self):
        self.assertIn("def install() -> None:", self.source,
                      "a11y.install() must exist as the public entry point")

    def test_installer_patches_all_seven_interactive_widgets(self):
        # Each widget class must appear in the _PATCH_TABLE so its __init__
        # gets wrapped at module import.
        required = [
            "ctk.CTkButton",
            "ctk.CTkSwitch",
            "ctk.CTkCheckBox",
            "ctk.CTkRadioButton",
            "ctk.CTkOptionMenu",
            "ctk.CTkSegmentedButton",
            "ctk.CTkSlider",
        ]
        # Find the _PATCH_TABLE definition window — match through the closing ")"
        # that's on its own line (the table contains many tuples).
        m = re.search(r"_PATCH_TABLE\s*=\s*\((.+?)^\)", self.source, re.DOTALL | re.MULTILINE)
        self.assertIsNotNone(m, "_PATCH_TABLE constant must exist")
        body = m.group(1)
        for cls in required:
            self.assertIn(cls, body,
                          f"{cls} must be in _PATCH_TABLE so its instances get a11y wiring")

    def test_button_helper_binds_enter_and_space(self):
        # The button helper must bind both Return and space; these are the
        # ARIA-standard activation keys.
        fn_src = self._extract_fn("make_button_keyboard_accessible")
        for seq in ("<Return>", "<KP_Enter>", "<space>"):
            self.assertIn(seq, fn_src,
                          f"button helper must bind {seq}")

    def test_toggle_helper_binds_space_for_switch_and_checkbox(self):
        fn_src = self._extract_fn("make_toggle_keyboard_accessible")
        self.assertIn("<space>", fn_src,
                      "Switch/CheckBox helper must bind <space>")
        self.assertIn("toggle", fn_src,
                      "toggle helper must call widget.toggle() — switches and "
                      "checkboxes expose .toggle() in CTk")

    def test_option_menu_helper_binds_arrows_home_end_and_open(self):
        fn_src = self._extract_fn("make_option_menu_keyboard_accessible")
        for seq in ("<Return>", "<space>", "<F4>", "<Down>", "<Up>", "<Home>", "<End>"):
            self.assertIn(seq, fn_src,
                          f"option menu helper must bind {seq}")

    def test_segmented_helper_binds_left_right_home_end(self):
        fn_src = self._extract_fn("make_segmented_keyboard_accessible")
        for seq in ("<Left>", "<Right>", "<Up>", "<Down>", "<Home>", "<End>"):
            self.assertIn(seq, fn_src,
                          f"segmented helper must bind {seq}")
        # Inner segment buttons must NOT take Tab focus — the segmented
        # control is a single tab stop with arrow-key navigation between
        # segments (ARIA tablist / radio pattern).
        self.assertIn("takefocus=0", fn_src,
                      "segmented helper must strip Tab focus from inner segment buttons")

    def test_slider_helper_binds_arrows_pageup_pagedown_home_end(self):
        fn_src = self._extract_fn("make_slider_keyboard_accessible")
        for seq in ("<Left>", "<Right>", "<Up>", "<Down>", "<Prior>", "<Next>", "<Home>", "<End>"):
            self.assertIn(seq, fn_src,
                          f"slider helper must bind {seq}")

    def test_focus_ring_is_installed_per_helper(self):
        # Every helper must call _install_focus_ring so the widget shows a
        # visible "you are here" indicator when focused via keyboard.
        for helper in ("make_button_keyboard_accessible",
                       "make_toggle_keyboard_accessible",
                       "make_radio_keyboard_accessible",
                       "make_option_menu_keyboard_accessible",
                       "make_segmented_keyboard_accessible",
                       "make_slider_keyboard_accessible"):
            fn_src = self._extract_fn(helper)
            self.assertIn("_install_focus_ring", fn_src,
                          f"{helper} must call _install_focus_ring for a visible focus indicator")

    def test_every_helper_calls_enable_tab_focus(self):
        # Without takefocus=1 on the inner canvas, Tab traversal will skip
        # the widget entirely — defeating the whole point.
        for helper in ("make_button_keyboard_accessible",
                       "make_toggle_keyboard_accessible",
                       "make_radio_keyboard_accessible",
                       "make_option_menu_keyboard_accessible",
                       "make_segmented_keyboard_accessible",
                       "make_slider_keyboard_accessible"):
            fn_src = self._extract_fn(helper)
            self.assertIn("_enable_tab_focus", fn_src,
                          f"{helper} must call _enable_tab_focus(widget)")

    def test_helpers_are_idempotent_via_a11y_tag(self):
        # The _A11Y_TAG guard must prevent double-binding on the same instance
        # (important when the wrapped __init__ is called more than once,
        # e.g. CTk subclass __init__ chaining).
        self.assertIn('_A11Y_TAG = "_localai_a11y_installed"', self.source)
        for helper in ("make_button_keyboard_accessible",
                       "make_toggle_keyboard_accessible",
                       "make_radio_keyboard_accessible",
                       "make_option_menu_keyboard_accessible",
                       "make_segmented_keyboard_accessible",
                       "make_slider_keyboard_accessible"):
            fn_src = self._extract_fn(helper)
            self.assertIn("_A11Y_TAG", fn_src,
                          f"{helper} must guard with _A11Y_TAG to stay idempotent")

    def test_bind_app_shortcuts_exists_for_ctrl_digit_page_switch(self):
        self.assertIn("def bind_app_shortcuts(", self.source,
                      "bind_app_shortcuts must exist for global Ctrl+1..9 page-switch keys")
        # Must use bind_all (toplevel binding) so the accelerator fires even
        # when focus is inside a text field on any page.
        m = re.search(r"def bind_app_shortcuts\(.*?(?=\ndef |\Z)", self.source, re.DOTALL)
        self.assertIsNotNone(m)
        self.assertIn("bind_all", m.group(0),
                      "bind_app_shortcuts must use bind_all so the accelerator is global")

    def _extract_fn(self, name: str) -> str:
        m = re.search(rf"\ndef {re.escape(name)}\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        self.assertIsNotNone(m, f"function {name} not found")
        return m.group(0)


class A11yAppIntegration(unittest.TestCase):
    """Pin that src/app.py wires the a11y module at module import time and
    attaches arrow-key navigation to the left navigation rail."""

    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")

    def test_app_imports_a11y_module(self):
        self.assertIn("from src import a11y", self.source,
                      "src/app.py must import the a11y module")

    def test_app_calls_a11y_install_at_module_top(self):
        # The install() call must happen before any CTk widget is constructed.
        # The simplest contract is that it's in the top-level code BEFORE the
        # App class definition.
        install_pos = self.source.find("a11y.install()")
        class_pos = self.source.find("class App(ctk.CTk):")
        self.assertGreater(install_pos, 0, "a11y.install() must be called")
        self.assertLess(install_pos, class_pos,
                        "a11y.install() must run before the App class is defined "
                        "(so the patch is in place when any widget is built)")

    def test_nav_rail_has_arrow_key_handler(self):
        self.assertIn("_wire_nav_rail_arrow_keys", self.source,
                      "App must wire arrow-key navigation onto the left nav rail")
        # The wiring must be invoked from the nav-rail build path (right
        # after the buttons are created).
        self.assertIn("self._wire_nav_rail_arrow_keys()", self.source,
                      "App must call _wire_nav_rail_arrow_keys() during build")

    def test_nav_rail_handler_binds_up_down_home_end(self):
        m = re.search(r"def _wire_nav_rail_arrow_keys\(.*?(?=\n    def |\nclass )",
                      self.source, re.DOTALL)
        self.assertIsNotNone(m, "_wire_nav_rail_arrow_keys body not found")
        body = m.group(0)
        for seq in ("<Down>", "<Up>", "<Home>", "<End>"):
            self.assertIn(seq, body,
                          f"_wire_nav_rail_arrow_keys must bind {seq} on each nav button")
        self.assertIn("a11y.bind_app_shortcuts", body,
                      "_wire_nav_rail_arrow_keys must call a11y.bind_app_shortcuts "
                      "for global Ctrl+1..9 page-switch keys")

    def test_app_installs_global_focus_traversal(self):
        # WHY: Without a global Tab handler, CTk's Canvas-drawn widgets get
        # no Tab navigation because tk.Canvas has no default Tab class
        # binding. App must call a11y.install_global_focus_traversal so
        # Tab/Shift-Tab actually move focus from anywhere.
        self.assertIn("a11y.install_global_focus_traversal(self)", self.source,
                      "App must call a11y.install_global_focus_traversal(self) so "
                      "Tab and Shift-Tab actually move focus")

    def test_app_parks_initial_focus_on_first_nav_button(self):
        # WHY: Without parking initial focus, the App root holds focus on
        # startup. The root has no key bindings so Tab and arrows are dead
        # until the user clicks something. Keyboard-only users would think
        # the app is broken. The fix: park focus on the first nav button at
        # launch so the focus ring is visible and arrows work immediately.
        self.assertRegex(
            self.source,
            r"first_nav\s*=.*self\._nav_btns",
            "App must locate the first nav button to park initial focus",
        )
        self.assertRegex(
            self.source,
            r"self\.after\(\s*\d+\s*,\s*lambda[^)]*?:\s*\w+\.focus_set\(\)\s*\)",
            "App must call after(...) to focus_set() the first nav button "
            "(direct focus_set during build is ignored before the window is mapped)",
        )


class A11yFocusTraversalContract(unittest.TestCase):
    """Pin that the global Tab/Shift-Tab traversal handlers exist in
    src/a11y.py with the exact behaviors they need to fix the runtime
    bug Ron hit on v5.5.19 (Tab and arrows doing nothing in the live app)."""

    @classmethod
    def setUpClass(cls):
        cls.source = A11Y_PATH.read_text(encoding="utf-8")

    def test_install_global_focus_traversal_exists(self):
        self.assertIn("def install_global_focus_traversal(", self.source,
                      "install_global_focus_traversal must exist")

    def test_global_traversal_uses_bind_all_for_tab_and_shift_tab(self):
        m = re.search(
            r"def install_global_focus_traversal\(.*?(?=\ndef |\nclass |\Z)",
            self.source, re.DOTALL)
        self.assertIsNotNone(m, "install_global_focus_traversal body not found")
        body = m.group(0)
        # Must bind globally so the handler runs no matter who has focus,
        # including Text widgets that would normally eat Tab.
        self.assertIn("bind_all", body,
                      "install_global_focus_traversal must use bind_all so Tab works "
                      "from any focused widget (including Text)")
        for seq in ("<Tab>", "<Shift-Tab>"):
            self.assertIn(seq, body,
                          f"install_global_focus_traversal must bind {seq}")

    def test_global_traversal_preserves_tab_insertion_via_ctrl_tab(self):
        # Plain Tab navigates out of a Text widget (the desktop-app
        # convention on Windows); Ctrl-Tab inserts a real tab character.
        m = re.search(
            r"def install_global_focus_traversal\(.*?(?=\ndef |\nclass |\Z)",
            self.source, re.DOTALL)
        body = m.group(0)
        self.assertIn("<Control-Tab>", body,
                      "Ctrl-Tab must still insert a literal tab in Text widgets")
        self.assertIn('bind_class("Text"', body,
                      "Ctrl-Tab binding must scope to the Text widget class")

    def test_focus_next_handler_returns_break(self):
        m = re.search(r"def _focus_next_handler\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        self.assertIsNotNone(m, "_focus_next_handler must exist")
        body = m.group(0)
        self.assertIn("tk_focusNext", body,
                      "_focus_next_handler must call tk_focusNext")
        self.assertIn("focus_set", body,
                      "_focus_next_handler must focus_set on the resolved widget")
        self.assertIn('return "break"', body,
                      "_focus_next_handler must return 'break' to stop the default "
                      "handling that would otherwise insert a tab character")

    def test_focus_prev_handler_returns_break(self):
        m = re.search(r"def _focus_prev_handler\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        self.assertIsNotNone(m, "_focus_prev_handler must exist")
        body = m.group(0)
        self.assertIn("tk_focusPrev", body,
                      "_focus_prev_handler must call tk_focusPrev")
        self.assertIn('return "break"', body,
                      "_focus_prev_handler must return 'break'")


class A11yArrowNavigationContract(unittest.TestCase):
    """Pin the wire_arrow_navigation helper (Up/Down/Left/Right/Home/End
    across a homogeneous group of widgets, e.g. filter rows or list rows)
    and its wiring in src/app.py for the filters + models list."""

    @classmethod
    def setUpClass(cls):
        cls.a11y_source = A11Y_PATH.read_text(encoding="utf-8")
        cls.app_source = APP_PATH.read_text(encoding="utf-8")

    def test_wire_arrow_navigation_exists(self):
        self.assertIn("def wire_arrow_navigation(", self.a11y_source,
                      "wire_arrow_navigation must exist as a reusable helper")

    def test_wire_arrow_navigation_supports_all_orientations(self):
        m = re.search(
            r"def wire_arrow_navigation\(.*?(?=\ndef |\nclass |\Z)",
            self.a11y_source, re.DOTALL)
        self.assertIsNotNone(m, "wire_arrow_navigation body not found")
        body = m.group(0)
        # Horizontal orientation binds Left/Right.
        for seq in ("<Right>", "<Left>"):
            self.assertIn(seq, body, f"horizontal orientation must bind {seq}")
        # Vertical orientation binds Up/Down.
        for seq in ("<Down>", "<Up>"):
            self.assertIn(seq, body, f"vertical orientation must bind {seq}")
        # Home/End always wire regardless of orientation.
        self.assertIn("<Home>", body, "wire_arrow_navigation must bind <Home>")
        self.assertIn("<End>", body, "wire_arrow_navigation must bind <End>")

    def test_wire_arrow_navigation_handlers_return_break(self):
        m = re.search(
            r"def wire_arrow_navigation\(.*?(?=\ndef |\nclass |\Z)",
            self.a11y_source, re.DOTALL)
        body = m.group(0)
        self.assertGreaterEqual(
            body.count('"break"'), 4,
            "wire_arrow_navigation handlers must return 'break' so the "
            "default class binding (e.g. canvas scroll on arrow keys) "
            "doesn't fire after focus has moved",
        )

    def test_type_filter_row_wired_for_arrow_navigation(self):
        # The type-row buttons (Chat / Vision / Image Gen / ...) must be
        # wired with horizontal arrow navigation.
        self.assertIn("a11y.wire_arrow_navigation(", self.app_source,
                      "App must call a11y.wire_arrow_navigation")
        self.assertIn(
            'self._type_btns.values()',
            self.app_source,
            "App must pass the type-filter buttons to wire_arrow_navigation",
        )

    def test_size_filter_row_wired_for_arrow_navigation(self):
        self.assertIn(
            'self._size_btns.values()',
            self.app_source,
            "App must pass the size-filter buttons to wire_arrow_navigation",
        )

    def test_model_list_rows_wired_for_arrow_navigation(self):
        self.assertRegex(
            self.app_source,
            r'a11y\.wire_arrow_navigation\(\s*\n?\s*list\(self\._model_cards\)',
            "App must wire arrow navigation across the model list rows "
            "after every repopulation (called from "
            "_continue_model_card_population)",
        )
        self.assertRegex(
            self.app_source,
            r"a11y\.wire_arrow_navigation\([\s\S]*?orientation=['\"]vertical['\"]",
            "Model list arrow navigation must use vertical orientation",
        )

    def test_model_row_focus_in_selects_for_detail_pane(self):
        # Focus-follows-selection: arrowing onto a row must update the
        # right-pane detail (Windows Explorer / Outlook / Settings pattern).
        m = re.search(
            r"def _focus_in\(self, _event=None\) -> None:.*?(?=\n    def |\nclass )",
            self.app_source, re.DOTALL)
        self.assertIsNotNone(m, "ModelListRow._focus_in body not found")
        body = m.group(0)
        self.assertIn("self.app._select_model_row(", body,
                      "ModelListRow._focus_in must call _select_model_row "
                      "so arrow-key navigation previews the model in the "
                      "detail pane (focus-follows-selection)")


class A11yFocusRingRobustnessContract(unittest.TestCase):
    """Pin that the focus ring survives CTk's button redraws (pressed /
    hover state changes) — the root cause of Ron's 'looks bad on Enter'
    complaint in dark mode.

    Without these guards, pressing Enter on a button triggers CTk's
    pressed-state redraw of the canvas, which can clobber our ring items.
    The ring then visibly blinks off and back on, plus stale items can
    accumulate."""

    @classmethod
    def setUpClass(cls):
        cls.source = A11Y_PATH.read_text(encoding="utf-8")

    def test_focus_ring_items_carry_a_tag(self):
        m = re.search(r"def _install_focus_ring\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        self.assertIsNotNone(m, "_install_focus_ring body not found")
        body = m.group(0)
        self.assertIn("RING_TAG", body,
                      "Focus-ring items must be tagged so we can re-raise "
                      "them above any later CTk-drawn items")
        self.assertIn('"_la_focus_ring"', body,
                      "Focus-ring tag must be the canonical _la_focus_ring")

    def test_focus_ring_raised_after_draw(self):
        m = re.search(r"def _install_focus_ring\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        body = m.group(0)
        self.assertIn("tag_raise(RING_TAG)", body,
                      "Focus ring must tag_raise after every draw so it "
                      "stays on top of CTk's pressed/hover overlays")

    def test_focus_ring_dark_mode_color_is_high_contrast(self):
        # Earlier versions used sky-cyan #7dd3fc which was blue-on-blue
        # against the dark-mode BUTTON_SECONDARY selected color. Amber
        # contrasts strongly against both gray AND blue.
        m = re.search(r"def _focus_ring_color\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        self.assertIsNotNone(m)
        body = m.group(0)
        # The actual return statement (not docstring mentions) must not
        # use the historical cyan that clashed in dark mode.
        ret_m = re.search(r'return\s+["\']([^"\']+)["\']\s+if\s+mode\s*==\s*["\']Light["\']\s+else\s+["\']([^"\']+)["\']',
                          body)
        self.assertIsNotNone(ret_m,
                             "Focus ring color must be returned via a "
                             "mode-conditional expression")
        light_color, dark_color = ret_m.group(1), ret_m.group(2)
        self.assertNotEqual(dark_color, "#7dd3fc",
                            "Dark-mode focus ring must NOT use #7dd3fc — "
                            "that was blue-on-blue against the selected "
                            "BUTTON_SECONDARY color (Ron's 'looks bad on "
                            "Enter' regression)")
        # Should use a warm color (amber) for dark mode high contrast.
        self.assertEqual(dark_color, "#fbbf24",
                         "Dark-mode focus ring should use amber #fbbf24 "
                         "for high contrast against both gray-unselected "
                         "and blue-selected button backgrounds")

    def test_focus_ring_rebinds_configure_and_press_events(self):
        m = re.search(r"def _install_focus_ring\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        body = m.group(0)
        for seq in ("<Configure>", "<ButtonPress-1>", "<ButtonRelease-1>"):
            self.assertIn(seq, body,
                          f"Focus ring must rebind {seq} to redraw after "
                          f"CTk clobbers the canvas")

    def test_focus_ring_checks_real_focus_not_stale_state(self):
        # The earlier implementation tested whether state['ring_outer']
        # was non-None, which lied after CTk wiped the items. The fix is
        # to query focus_displayof() so we know the truth.
        m = re.search(r"def _install_focus_ring\(.*?(?=\ndef |\nclass |\Z)",
                      self.source, re.DOTALL)
        body = m.group(0)
        self.assertIn("focus_displayof()", body,
                      "Focus ring must query focus_displayof() rather than "
                      "trusting stale state['ring_outer'] — CTk's redraw "
                      "may have deleted our items behind our back")


if __name__ == "__main__":
    unittest.main()

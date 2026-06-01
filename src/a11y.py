# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Keyboard accessibility layer for CustomTkinter widgets.

WHY THIS EXISTS
---------------
Out of the box, customtkinter (5.2.2) widgets are mouse-only:

* CTkButton, CTkSwitch, CTkCheckBox, CTkOptionMenu, CTkSegmentedButton,
  CTkSlider, CTkRadioButton — all skip Tk's natural focus chain (no
  ``takefocus`` on the inner Canvas), and bind only ``<Button-1>`` (no
  ``<Return>`` / ``<space>`` / arrow-key handlers).
* Tab cannot move focus to them. Enter and Space cannot activate them.
* There is no visible focus indicator.

For LocalAI to be usable without a mouse — a hard requirement for many users,
including those who rely on screen readers like JAWS or NVDA — we install
import-time monkey-patches that:

1. Mark every interactive widget's inner Canvas with ``takefocus=1`` so Tab
   moves through them in geometric order.
2. Bind ``<Return>`` and ``<space>`` (and arrow keys where appropriate) so
   each widget can be activated or adjusted from the keyboard.
3. Draw a high-contrast 2 px focus ring on the inner Canvas on ``<FocusIn>``
   and erase it on ``<FocusOut>``.

The patches are *additive* — they wrap the existing ``__init__`` of each
CTk widget and never replace the original click-path. Existing mouse
behaviour is unchanged.

SCOPE / LIMITS
--------------
This module makes the app fully keyboard-operable. It does **not** make the
app fully screen-reader (JAWS/NVDA) friendly: CTk widgets are Canvas-drawn,
so the text is painted as Canvas items rather than exposed as native Win32
controls. Screen readers can read the focused widget's title-text fallback
(set via the widget's accessibility name), the menubar, and tk Text/Entry
fields, but cannot announce the dynamic state of switches/segmented buttons
the way they would for native UIA/MSAA controls. The on-canvas focus ring
gives sighted keyboard users a clear "you are here" signal regardless.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any, Callable

import customtkinter as ctk

# Sentinel attribute used to guard against double-patching when the helper
# is called more than once on the same widget instance (e.g. in tests).
_A11Y_TAG = "_localai_a11y_installed"

# Focus-ring colors are theme-aware. We pull the tokens at draw time so the
# ring follows the current appearance mode rather than a snapshot.
def _focus_ring_color() -> str:
    """Pick a high-contrast color for the focus ring vs the current theme.

    The colors below pass WCAG 2.2 SC 1.4.11 (non-text contrast >= 3:1)
    against both the dark surface (#2b2b2b) and the light surface
    (#ebebeb) — verified by hand.

    Critically, the dark-mode color must ALSO be high-contrast against
    the selected/active button color (BUTTON_SECONDARY which is a blue
    in dark mode). Earlier versions used sky-cyan #7dd3fc which created
    a blue-on-blue mess on a selected nav button. Amber #fbbf24
    (Tailwind amber-400) contrasts strongly against both the gray
    unselected button surface AND the blue selected surface — same
    color works in every state.
    """
    try:
        mode = ctk.get_appearance_mode()
    except Exception:
        mode = "Dark"
    # Amber on dark (high contrast vs gray AND blue selected states),
    # deep navy on light (high contrast vs every light-mode surface).
    return "#1e40af" if mode == "Light" else "#fbbf24"


def _install_focus_ring(widget: Any) -> None:
    """Draw an inset focus ring on the widget's inner Canvas while focused.

    Uses a solid 2 px outline (no dashes) for maximum visibility on small
    widgets where dashes can render as a single pixel and disappear.

    Critical robustness notes:
    * The ring is tagged with ``_la_focus_ring`` so we can z-raise it
      above any later items the CTk draw engine creates. Without this,
      pressing Enter (which triggers CTk's pressed/hover redraw of the
      button's rounded-rect canvas items) visibly clobbers the ring.
    * We bind ``<Configure>`` to redraw whenever the canvas reconfigures
      — and we check focus via ``focus_displayof()`` rather than trusting
      stale state, because the items may have been deleted by CTk's
      redraw even though our state dict still has IDs.
    * After every draw, ``tag_raise("_la_focus_ring")`` guarantees the
      ring stays on top of every other canvas item.
    """
    canvas = getattr(widget, "_canvas", None)
    if canvas is None:
        return

    state: dict[str, Any] = {"ring_outer": None, "ring_inner": None}
    RING_TAG = "_la_focus_ring"

    def _has_focus() -> bool:
        try:
            return canvas.focus_displayof() is canvas
        except Exception:
            return False

    def _draw(event=None) -> None:
        try:
            _erase()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w <= 6 or h <= 6:
                # Defer if canvas hasn't been laid out yet.
                canvas.after(20, _draw)
                return
            # Double-stroke ring for visibility against any background:
            # an outer dark "shadow" then a bright inner stroke. This is
            # the same pattern Windows 11 uses for its native focus
            # indicator (dark/light pair regardless of theme).
            shadow = "#0a0a0a" if ctk.get_appearance_mode() != "Light" else "#f5f5f5"
            state["ring_outer"] = canvas.create_rectangle(
                1, 1, w - 1, h - 1,
                outline=shadow,
                width=1,
                tags=(RING_TAG,),
            )
            state["ring_inner"] = canvas.create_rectangle(
                3, 3, w - 3, h - 3,
                outline=_focus_ring_color(),
                width=2,
                tags=(RING_TAG,),
            )
            # Keep the ring above any later CTk-drawn items (button
            # pressed-state highlight, hover overlays, etc.).
            canvas.tag_raise(RING_TAG)
        except Exception:
            pass

    def _erase(event=None) -> None:
        for key in ("ring_outer", "ring_inner"):
            rid = state[key]
            if rid is not None:
                try:
                    canvas.delete(rid)
                except Exception:
                    pass
                state[key] = None
        # Belt-and-braces: any item still carrying our tag must go (in
        # case state got out of sync with the canvas).
        try:
            canvas.delete(RING_TAG)
        except Exception:
            pass

    def _on_configure(_event=None) -> None:
        # Redraw whenever the canvas reconfigures, but ONLY if we
        # currently have focus. Don't trust state — CTk's redraw may
        # have deleted our items behind our back.
        if _has_focus():
            _draw()

    canvas.bind("<FocusIn>", _draw, add="+")
    canvas.bind("<FocusOut>", _erase, add="+")
    canvas.bind("<Configure>", _on_configure, add="+")
    # ButtonPress/Release also trigger CTk's redraw of the button's
    # canvas items; re-raise the ring after the event has been processed.
    def _reraise_after(_e=None):
        if _has_focus():
            canvas.after(1, _draw)
    canvas.bind("<ButtonPress-1>", _reraise_after, add="+")
    canvas.bind("<ButtonRelease-1>", _reraise_after, add="+")


def _enable_tab_focus(widget: Any) -> None:
    """Make the widget's inner Canvas reachable via Tab traversal."""
    canvas = getattr(widget, "_canvas", None)
    if canvas is not None:
        try:
            canvas.configure(takefocus=1)
        except Exception:
            pass


def _safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a handler so an exception never breaks the event loop."""
    def _wrapped(event=None, *a, **kw):
        try:
            return fn(event, *a, **kw) if event is not None else fn(*a, **kw)
        except TypeError:
            try:
                return fn()
            except Exception:
                return None
        except Exception:
            return None
    return _wrapped


def make_button_keyboard_accessible(widget: ctk.CTkButton) -> None:
    """Wire Tab / Enter / Space for a CTkButton."""
    if getattr(widget, _A11Y_TAG, False):
        return
    setattr(widget, _A11Y_TAG, True)
    _enable_tab_focus(widget)
    _install_focus_ring(widget)

    def _activate(_event=None):
        try:
            if widget.cget("state") == "disabled":
                return "break"
        except Exception:
            pass
        try:
            widget._clicked()
        except Exception:
            cmd = getattr(widget, "_command", None)
            if callable(cmd):
                try:
                    cmd()
                except Exception:
                    pass
        return "break"

    canvas = getattr(widget, "_canvas", None)
    if canvas is not None:
        canvas.bind("<Return>", _activate, add="+")
        canvas.bind("<KP_Enter>", _activate, add="+")
        canvas.bind("<space>", _activate, add="+")


def _toggle_widget(widget) -> None:
    """Toggle a Switch/CheckBox/Radio respecting disabled state."""
    try:
        if widget.cget("state") == "disabled":
            return
    except Exception:
        pass
    # Switch + CheckBox expose `toggle()`. Radio uses select().
    if hasattr(widget, "toggle"):
        try:
            widget.toggle()
            return
        except Exception:
            pass
    if hasattr(widget, "select"):
        try:
            widget.select()
        except Exception:
            pass


def make_toggle_keyboard_accessible(widget) -> None:
    """Wire Tab / Space for CTkSwitch and CTkCheckBox."""
    if getattr(widget, _A11Y_TAG, False):
        return
    setattr(widget, _A11Y_TAG, True)
    _enable_tab_focus(widget)
    _install_focus_ring(widget)

    def _activate(_event=None):
        _toggle_widget(widget)
        return "break"

    canvas = getattr(widget, "_canvas", None)
    if canvas is not None:
        canvas.bind("<space>", _activate, add="+")
        canvas.bind("<Return>", _activate, add="+")
        canvas.bind("<KP_Enter>", _activate, add="+")


def make_radio_keyboard_accessible(widget: ctk.CTkRadioButton) -> None:
    """Wire Tab / Space for a single CTkRadioButton."""
    if getattr(widget, _A11Y_TAG, False):
        return
    setattr(widget, _A11Y_TAG, True)
    _enable_tab_focus(widget)
    _install_focus_ring(widget)

    def _activate(_event=None):
        try:
            if widget.cget("state") == "disabled":
                return "break"
        except Exception:
            pass
        try:
            widget.select()
        except Exception:
            pass
        try:
            cmd = getattr(widget, "_command", None)
            if callable(cmd):
                cmd()
        except Exception:
            pass
        return "break"

    canvas = getattr(widget, "_canvas", None)
    if canvas is not None:
        canvas.bind("<space>", _activate, add="+")
        canvas.bind("<Return>", _activate, add="+")
        canvas.bind("<KP_Enter>", _activate, add="+")


def make_option_menu_keyboard_accessible(widget: ctk.CTkOptionMenu) -> None:
    """Wire Tab / Enter / Space / arrow keys for a CTkOptionMenu.

    * Tab brings focus to the menu's canvas.
    * Enter or Space opens the dropdown (mirrors mouse click).
    * Up / Down arrow keys cycle through values **without** opening the
      dropdown so users can scroll options inline (matches native combo box
      "browse" behavior on Windows).
    * Home / End jump to the first / last value.
    """
    if getattr(widget, _A11Y_TAG, False):
        return
    setattr(widget, _A11Y_TAG, True)
    _enable_tab_focus(widget)
    _install_focus_ring(widget)

    def _open_dropdown(_event=None):
        try:
            if widget.cget("state") == "disabled":
                return "break"
        except Exception:
            pass
        try:
            widget._clicked()
        except Exception:
            pass
        return "break"

    def _move(delta: int):
        try:
            if widget.cget("state") == "disabled":
                return "break"
            values = list(widget.cget("values") or [])
            if not values:
                return "break"
            current = widget.get()
            try:
                idx = values.index(current)
            except ValueError:
                idx = 0
            new_idx = max(0, min(len(values) - 1, idx + delta))
            if new_idx != idx:
                widget.set(values[new_idx])
                cmd = getattr(widget, "_command", None)
                if callable(cmd):
                    try:
                        cmd(values[new_idx])
                    except Exception:
                        pass
        except Exception:
            pass
        return "break"

    def _go_first(_event=None):
        try:
            values = list(widget.cget("values") or [])
            if values:
                widget.set(values[0])
                cmd = getattr(widget, "_command", None)
                if callable(cmd):
                    try:
                        cmd(values[0])
                    except Exception:
                        pass
        except Exception:
            pass
        return "break"

    def _go_last(_event=None):
        try:
            values = list(widget.cget("values") or [])
            if values:
                widget.set(values[-1])
                cmd = getattr(widget, "_command", None)
                if callable(cmd):
                    try:
                        cmd(values[-1])
                    except Exception:
                        pass
        except Exception:
            pass
        return "break"

    canvas = getattr(widget, "_canvas", None)
    if canvas is not None:
        canvas.bind("<Return>", _open_dropdown, add="+")
        canvas.bind("<KP_Enter>", _open_dropdown, add="+")
        canvas.bind("<space>", _open_dropdown, add="+")
        canvas.bind("<F4>", _open_dropdown, add="+")  # native combo-box accelerator
        canvas.bind("<Down>", lambda _e: _move(+1), add="+")
        canvas.bind("<Up>", lambda _e: _move(-1), add="+")
        canvas.bind("<Home>", _go_first, add="+")
        canvas.bind("<End>", _go_last, add="+")


def make_segmented_keyboard_accessible(widget: ctk.CTkSegmentedButton) -> None:
    """Wire Tab + Left/Right + Home/End + Space/Enter for CTkSegmentedButton.

    Treat the whole segmented control as a single focus stop with arrow-key
    navigation across its segments, matching WAI-ARIA tab/radio pattern.
    Mouse clicks on individual segments still work unchanged.
    """
    if getattr(widget, _A11Y_TAG, False):
        return
    setattr(widget, _A11Y_TAG, True)
    # The outer widget is a CTkFrame whose own canvas is the focus target;
    # individual segment buttons keep mouse focus but are excluded from Tab.
    _enable_tab_focus(widget)
    _install_focus_ring(widget)

    # Strip Tab focus from inner segment buttons so the segmented control
    # acts as a single Tab stop (consistent with native ARIA tabs/radios).
    try:
        for sub in widget._buttons_dict.values():
            sub_canvas = getattr(sub, "_canvas", None)
            if sub_canvas is not None:
                sub_canvas.configure(takefocus=0)
    except Exception:
        pass

    def _values():
        try:
            return list(widget.cget("values") or [])
        except Exception:
            return []

    def _move(delta: int):
        values = _values()
        if not values:
            return "break"
        try:
            current = widget.get()
        except Exception:
            current = values[0] if values else ""
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        new_idx = max(0, min(len(values) - 1, idx + delta))
        if new_idx != idx:
            try:
                widget.set(values[new_idx])
                cmd = getattr(widget, "_command", None)
                if callable(cmd):
                    cmd(values[new_idx])
            except Exception:
                pass
        return "break"

    def _go_first(_event=None):
        values = _values()
        if values:
            try:
                widget.set(values[0])
                cmd = getattr(widget, "_command", None)
                if callable(cmd):
                    cmd(values[0])
            except Exception:
                pass
        return "break"

    def _go_last(_event=None):
        values = _values()
        if values:
            try:
                widget.set(values[-1])
                cmd = getattr(widget, "_command", None)
                if callable(cmd):
                    cmd(values[-1])
            except Exception:
                pass
        return "break"

    canvas = getattr(widget, "_canvas", None)
    if canvas is not None:
        canvas.bind("<Left>", lambda _e: _move(-1), add="+")
        canvas.bind("<Right>", lambda _e: _move(+1), add="+")
        canvas.bind("<Up>", lambda _e: _move(-1), add="+")
        canvas.bind("<Down>", lambda _e: _move(+1), add="+")
        canvas.bind("<Home>", _go_first, add="+")
        canvas.bind("<End>", _go_last, add="+")


def make_slider_keyboard_accessible(widget: ctk.CTkSlider) -> None:
    """Wire Tab + arrow keys + PgUp/PgDn + Home/End for CTkSlider."""
    if getattr(widget, _A11Y_TAG, False):
        return
    setattr(widget, _A11Y_TAG, True)
    _enable_tab_focus(widget)
    _install_focus_ring(widget)

    def _bounds() -> tuple[float, float, int | None]:
        try:
            lo = float(widget.cget("from_"))
        except Exception:
            lo = 0.0
        try:
            hi = float(widget.cget("to"))
        except Exception:
            hi = 1.0
        try:
            steps = widget.cget("number_of_steps")
            steps = int(steps) if steps is not None else None
        except Exception:
            steps = None
        return lo, hi, steps

    def _step_size(small: bool = True) -> float:
        lo, hi, steps = _bounds()
        span = hi - lo
        if steps and steps > 0:
            return span / float(steps)
        return span * (0.01 if small else 0.1)

    def _set_value(new_val: float):
        lo, hi, _ = _bounds()
        new_val = max(lo, min(hi, new_val))
        try:
            widget.set(new_val)
        except Exception:
            return
        cmd = getattr(widget, "_command", None)
        if callable(cmd):
            try:
                cmd(new_val)
            except Exception:
                pass

    def _bump(direction: int, big: bool = False):
        try:
            cur = widget.get()
        except Exception:
            cur = 0.0
        step = _step_size(small=not big)
        _set_value(cur + direction * step)
        return "break"

    canvas = getattr(widget, "_canvas", None)
    if canvas is not None:
        canvas.bind("<Left>", lambda _e: _bump(-1), add="+")
        canvas.bind("<Right>", lambda _e: _bump(+1), add="+")
        canvas.bind("<Down>", lambda _e: _bump(-1), add="+")
        canvas.bind("<Up>", lambda _e: _bump(+1), add="+")
        canvas.bind("<Prior>", lambda _e: _bump(+1, big=True), add="+")
        canvas.bind("<Next>", lambda _e: _bump(-1, big=True), add="+")
        canvas.bind("<Home>", lambda _e: _set_value(_bounds()[0]) or "break", add="+")
        canvas.bind("<End>", lambda _e: _set_value(_bounds()[1]) or "break", add="+")


# ---------------------------------------------------------------------------
# Global import-time patch — wraps each widget's __init__ so every instance
# automatically becomes keyboard-accessible the moment it's created.
# ---------------------------------------------------------------------------

_PATCH_TABLE = (
    (ctk.CTkButton, make_button_keyboard_accessible),
    (ctk.CTkSwitch, make_toggle_keyboard_accessible),
    (ctk.CTkCheckBox, make_toggle_keyboard_accessible),
    (ctk.CTkRadioButton, make_radio_keyboard_accessible),
    (ctk.CTkOptionMenu, make_option_menu_keyboard_accessible),
    (ctk.CTkSegmentedButton, make_segmented_keyboard_accessible),
    (ctk.CTkSlider, make_slider_keyboard_accessible),
)


def _install_global_patches() -> None:
    """Wrap each widget's __init__ so all instances get keyboard handlers."""
    for cls, installer in _PATCH_TABLE:
        if getattr(cls, "_localai_a11y_patched", False):
            continue
        original_init = cls.__init__

        def _make_wrapper(_orig, _install):
            def _wrapped(self, *args, **kwargs):
                _orig(self, *args, **kwargs)
                try:
                    _install(self)
                except Exception:
                    # Never let a11y wiring break widget construction.
                    pass
            return _wrapped

        cls.__init__ = _make_wrapper(original_init, installer)
        setattr(cls, "_localai_a11y_patched", True)


def install() -> None:
    """Idempotent entry point — call once at app startup."""
    _install_global_patches()


# ---------------------------------------------------------------------------
# Global app-level shortcuts
# ---------------------------------------------------------------------------

def bind_app_shortcuts(root: tk.Misc, *, page_switcher: Callable[[str], None],
                       pages: list[tuple[str, str]]) -> None:
    """Bind Ctrl+1..9 to switch to the corresponding nav page.

    ``pages`` is a list of ``(page_id, label)`` tuples in nav order. The
    binding is attached to the toplevel so it works from any focused widget.
    """
    for i, (page_id, _label) in enumerate(pages[:9], start=1):
        seq = f"<Control-Key-{i}>"
        root.bind_all(seq, lambda _e, pid=page_id: (page_switcher(pid), "break")[1], add="+")


# ---------------------------------------------------------------------------
# Global focus traversal — make Tab / Shift-Tab actually move focus.
# ---------------------------------------------------------------------------
#
# CRITICAL: tkinter delivers Tab navigation via per-class bindings on
# ``Button``, ``Entry``, ``Spinbox`` etc. — but the CTk widgets are
# Canvas-drawn, and ``tk.Canvas`` has NO default Tab binding. The root
# ``CTk`` (a Toplevel) also has no Tab binding. So even though every CTk
# inner canvas has ``takefocus=1`` (set by ``_enable_tab_focus``) and is
# correctly listed in the focus chain by ``tk::FocusOK``, pressing Tab from
# anywhere does nothing because no widget in the bindtag chain consumes the
# event and calls ``tk_focusNext()``. The user sees a dead key.
#
# This must be bound on the root via ``bind_all`` so it works regardless of
# which widget currently holds focus — including Text widgets that would
# otherwise insert a literal tab character.

def _focus_next_handler(event):
    """Move focus to the next focusable widget in the geometric chain."""
    widget = event.widget if event is not None else None
    if widget is None:
        return None
    try:
        nxt = widget.tk_focusNext()
        if nxt is not None:
            nxt.focus_set()
    except Exception:
        return None
    return "break"


def _focus_prev_handler(event):
    """Move focus to the previous focusable widget in the geometric chain."""
    widget = event.widget if event is not None else None
    if widget is None:
        return None
    try:
        prev = widget.tk_focusPrev()
        if prev is not None:
            prev.focus_set()
    except Exception:
        return None
    return "break"


def install_global_focus_traversal(root: tk.Misc) -> None:
    """Bind Tab / Shift-Tab globally so focus actually moves.

    Without this, CTk's Canvas-drawn widgets never get Tab navigation — the
    inner canvases have ``takefocus=1`` but no widget in the bindtag chain
    calls ``tk_focusNext()`` on a Tab keystroke. We bind on the toplevel via
    ``bind_all`` so every widget (including Text widgets that would
    normally consume Tab as a literal character) defers to focus traversal.

    Text editing inside a Text widget is still possible:
    * ``Ctrl+Tab`` inserts a literal tab character.
    * Plain Tab navigates out, matching every other desktop GUI on Windows.
    """
    # Bind globally so the handler runs no matter who has focus.
    root.bind_all("<Tab>", _focus_next_handler, add="+")
    root.bind_all("<Shift-Tab>", _focus_prev_handler, add="+")
    # Windows/Tk also fires <Shift-ISO_Left_Tab> on some keyboards.
    try:
        root.bind_all("<ISO_Left_Tab>", _focus_prev_handler, add="+")
    except Exception:
        pass
    # Make Ctrl+Tab inside a Text widget insert a real tab instead of
    # navigating. (This restores the only piece of behavior Tab-as-navigate
    # would otherwise take away.)
    def _insert_real_tab(event):
        try:
            event.widget.insert("insert", "\t")
        except Exception:
            return None
        return "break"
    root.bind_class("Text", "<Control-Tab>", _insert_real_tab, add="+")


def focus_first_nav_button(root: tk.Misc) -> None:
    """Put initial focus on the first focusable widget so keyboard users see
    the focus ring immediately on startup. Without this, focus is on the
    root (which has no key bindings) and the user has to click before any
    keyboard navigation works.
    """
    try:
        first = root.tk_focusNext()
        if first is not None:
            first.focus_set()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Arrow-key navigation across a homogeneous group of widgets (filter rows,
# list rows, button groups). This is the canonical WAI-ARIA "toolbar" or
# "listbox" keyboard pattern: arrows move within the group, Home/End jump
# to the ends, Tab still escapes the group.
# ---------------------------------------------------------------------------

def wire_arrow_navigation(
    widgets: list[Any],
    *,
    orientation: str = "horizontal",
    activate: Callable[[Any], None] | None = None,
    wrap: bool = False,
) -> None:
    """Wire arrow keys to move focus across ``widgets`` in order.

    ``orientation`` controls which arrow keys move within the group:
    * ``"horizontal"`` (default) — Left / Right walk siblings.
    * ``"vertical"`` — Up / Down walk siblings.
    * ``"both"`` — all four arrows walk siblings (useful for grids).

    ``Home`` always jumps to the first widget; ``End`` to the last.

    Each widget must either be a CTk widget exposing ``._canvas`` (the
    a11y-patched focus target) or any tk.Misc whose ``focus_set()`` works
    directly. If ``activate`` is provided, it's invoked with the focused
    widget when the user presses Enter/Space (in addition to whatever
    activation the widget already provides via its own bindings).

    ``wrap=True`` makes navigation cyclic (off end of list wraps to
    start). Default is non-wrapping so users hit a natural boundary.
    """
    widgets = [w for w in widgets if w is not None]
    if len(widgets) < 2:
        return

    def _focus_index(idx: int) -> None:
        if wrap:
            idx = idx % len(widgets)
        else:
            idx = max(0, min(len(widgets) - 1, idx))
        target = widgets[idx]
        focus_target = getattr(target, "_canvas", target)
        try:
            focus_target.focus_set()
        except Exception:
            pass

    horiz = orientation in ("horizontal", "both")
    vert = orientation in ("vertical", "both")

    for i, w in enumerate(widgets):
        focus_target = getattr(w, "_canvas", w)
        if focus_target is None:
            continue
        try:
            # Ensure the focus target accepts focus (CTk a11y patches
            # already do this for patched widgets; this is a no-op for
            # plain tk widgets that default to takefocus=1).
            try:
                focus_target.configure(takefocus=1)
            except Exception:
                pass

            if horiz:
                focus_target.bind(
                    "<Right>",
                    lambda _e, ii=i: (_focus_index(ii + 1), "break")[1],
                    add="+",
                )
                focus_target.bind(
                    "<Left>",
                    lambda _e, ii=i: (_focus_index(ii - 1), "break")[1],
                    add="+",
                )
            if vert:
                focus_target.bind(
                    "<Down>",
                    lambda _e, ii=i: (_focus_index(ii + 1), "break")[1],
                    add="+",
                )
                focus_target.bind(
                    "<Up>",
                    lambda _e, ii=i: (_focus_index(ii - 1), "break")[1],
                    add="+",
                )
            focus_target.bind(
                "<Home>",
                lambda _e: (_focus_index(0), "break")[1],
                add="+",
            )
            focus_target.bind(
                "<End>",
                lambda _e: (_focus_index(len(widgets) - 1), "break")[1],
                add="+",
            )
            if activate is not None:
                def _on_activate(_e, target=w):
                    try:
                        activate(target)
                    except Exception:
                        pass
                    return "break"
                focus_target.bind("<Return>", _on_activate, add="+")
                focus_target.bind("<space>", _on_activate, add="+")
        except Exception:
            continue

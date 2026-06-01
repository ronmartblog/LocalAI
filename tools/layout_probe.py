"""Layout probe — walks the running LocalAI Studio UI tree and reports any
widget whose bottom edge falls below the visible viewport at a given window
size. Saves a JSON report and a full-window screenshot per page.

Usage (run from the LocalAI repo root):

    py tools/layout_probe.py

The probe imports src.app, builds the App in-process, programmatically resizes
the window to each target size, visits every top-level page, and captures:

  - per-widget (class, geometry, ismapped, scrollable-ancestor, off-screen?) records
  - a pyautogui screenshot of the window

Outputs land in C:/Users/ronmart/OneDrive - Microsoft/Documents/Clawpilot/Scratchpad/
so they show up in the chat UI inline.

This is a developer-only diagnostic — it does not ship with the app.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Safe defaults so the probe doesn't auto-start ComfyUI / Ollama.
os.environ.setdefault("LOCALAI_DISABLE_AUTOSTART", "1")
os.environ.setdefault("LOCALAI_DISABLE_BACKGROUND_THREADS", "1")

OUTPUT_DIR = Path(
    os.environ.get("LOCALAI_LAYOUT_PROBE_OUTPUT_DIR")
    or (Path.home() / "Documents" / "LocalAI-layout-probes")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZES = [
    ("900x600", 900, 600),
    ("1280x800", 1280, 800),
    ("1440x900", 1440, 900),
]

PAGES = ["chat", "toolbox", "models", "image_gen", "benchmark", "system"]


def _has_scrollable_ancestor(widget) -> bool:
    cur = widget
    for _ in range(40):
        cur = getattr(cur, "master", None) or getattr(cur, "_master", None)
        if cur is None:
            return False
        cls = ""
        try:
            cls = cur.winfo_class()
        except Exception:
            pass
        if "Scrollable" in cls or cls == "CTkScrollableFrame":
            return True
        try:
            modname = type(cur).__name__
        except Exception:
            modname = ""
        if "Scrollable" in modname:
            return True
    return False


def _widget_record(widget, win_left: int, win_top: int, win_w: int, win_h: int) -> dict:
    try:
        cls = widget.winfo_class()
    except Exception:
        cls = "?"
    try:
        ismapped = bool(widget.winfo_ismapped())
    except Exception:
        ismapped = False
    try:
        x = widget.winfo_rootx() - win_left
        y = widget.winfo_rooty() - win_top
        w = widget.winfo_width()
        h = widget.winfo_height()
    except Exception:
        x = y = w = h = -1
    off_bottom = ismapped and (y + h) > win_h
    off_right = ismapped and (x + w) > win_w
    try:
        name = str(widget._name) if hasattr(widget, "_name") else str(widget)
    except Exception:
        name = "?"
    text = ""
    try:
        cfg = widget.cget("text")
        text = str(cfg)[:60]
    except Exception:
        pass
    pyclass = type(widget).__name__
    in_scroll = _has_scrollable_ancestor(widget)
    return {
        "name": name,
        "class": cls,
        "pyclass": pyclass,
        "text": text,
        "x": x, "y": y, "w": w, "h": h,
        "ismapped": ismapped,
        "off_bottom": off_bottom,
        "off_right": off_right,
        "in_scroll": in_scroll,
    }


def _walk(widget, win_left, win_top, win_w, win_h, depth=0, max_depth=18, out=None):
    if out is None:
        out = []
    rec = _widget_record(widget, win_left, win_top, win_w, win_h)
    rec["depth"] = depth
    out.append(rec)
    if depth >= max_depth:
        return out
    try:
        children = widget.winfo_children()
    except Exception:
        children = []
    for child in children:
        _walk(child, win_left, win_top, win_w, win_h, depth + 1, max_depth, out)
    return out


def _grab_screenshot(app, dest: Path):
    try:
        import pyautogui
    except Exception:
        return False
    try:
        try:
            app.lift()
            app.attributes("-topmost", True)
            app.update_idletasks()
            app.update()
            time.sleep(0.15)
        except Exception:
            pass
        x = app.winfo_rootx()
        y = app.winfo_rooty()
        w = app.winfo_width()
        h = app.winfo_height()
        img = pyautogui.screenshot(region=(max(0, x), max(0, y), w, h))
        img.save(dest)
        try:
            app.attributes("-topmost", False)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"  screenshot failed: {e}")
        return False


def _try_navigate(app, page: str) -> bool:
    candidates = (
        ("_switch_page", (page,)),
        ("_show_page", (page,)),
        ("_navigate_to", (page,)),
        ("_select_page", (page,)),
        ("show_page", (page,)),
    )
    for attr, args in candidates:
        fn = getattr(app, attr, None)
        if callable(fn):
            try:
                fn(*args)
                return True
            except Exception:
                continue
    pages = getattr(app, "_pages", None)
    if pages and page in pages:
        try:
            pages[page].tkraise()
            return True
        except Exception:
            return False
    return False


def main():
    print("[layout_probe] importing app …")
    from src import app as appmod

    print("[layout_probe] constructing App …")
    application = appmod.App()
    application.update_idletasks()
    application.update()
    time.sleep(0.6)

    reports = []
    for size_label, w, h in TARGET_SIZES:
        print(f"\n[layout_probe] === size {size_label} ===")
        try:
            application.geometry(f"{w}x{h}+80+80")
        except Exception as e:
            print(f"  could not set geometry: {e}")
        application.update_idletasks()
        application.update()
        time.sleep(0.6)

        for page in PAGES:
            print(f"  page={page} …", end=" ")
            if not _try_navigate(application, page):
                print("NAV-FAIL")
                continue
            application.update_idletasks()
            application.update()
            time.sleep(0.4)

            win_left = application.winfo_rootx()
            win_top = application.winfo_rooty()
            win_w = application.winfo_width()
            win_h = application.winfo_height()
            records = _walk(application, win_left, win_top, win_w, win_h)

            # Off-screen *and* not inside a scrollable frame → real bug.
            off_chrome = [r for r in records
                          if (r["off_bottom"] or r["off_right"]) and not r["in_scroll"]]
            off_scroll = [r for r in records
                          if (r["off_bottom"] or r["off_right"]) and r["in_scroll"]]
            print(f"widgets={len(records)} off_chrome={len(off_chrome)} off_scroll={len(off_scroll)}")

            shot = OUTPUT_DIR / f"layout-probe-{page}-{size_label}.png"
            _grab_screenshot(application, shot)

            reports.append({
                "size": size_label,
                "page": page,
                "win": {"x": win_left, "y": win_top, "w": win_w, "h": win_h},
                "total_widgets": len(records),
                "off_chrome_count": len(off_chrome),
                "off_scroll_count": len(off_scroll),
                # Chrome failures are the bugs — keep full list.
                "off_chrome": off_chrome,
                # Scrollable overflow is expected; sample only.
                "off_scroll_samples": off_scroll[:5],
                "screenshot": str(shot),
            })

    out = OUTPUT_DIR / "layout-probe-report.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\n[layout_probe] wrote report: {out}")
    print(f"[layout_probe] screenshots in: {OUTPUT_DIR}")

    try:
        application.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()

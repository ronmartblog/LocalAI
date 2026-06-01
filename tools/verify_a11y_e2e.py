"""End-to-end runtime test: focus starts on nav button, Tab moves to next,
Shift-Tab moves back, Enter on a nav button switches page.

This tests the ACTUAL keyboard flow Ron would use — no calling _clicked()
directly, no inspecting bindings in isolation.
"""
import os, sys, time
os.environ['LOCALAI_DISABLE_AUTO_REFRESH'] = '1'
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, '.')

from src.app import App

root = App()
root.geometry("1280x800+50+50")
root.update_idletasks()
root.deiconify()
root.focus_force()
root.update()
time.sleep(0.4)
root.update()
# Let the after(50) fire that parks initial focus
root.after(100, root.update)
root.update()
time.sleep(0.2)
root.update()

print("=" * 70)
print("TEST 1: Initial focus is on first nav button (Home)")
print("=" * 70)
focus = root.focus_get()
print(f"focus_get: {focus}")
print(f"type: {type(focus).__name__ if focus else 'None'}")
expected_home = root._nav_btns.get("home")
exp_canvas = getattr(expected_home, "_canvas", expected_home)
print(f"expected (home canvas): {exp_canvas}")
ok1 = focus is not None and str(focus) == str(exp_canvas)
print(f"PASS" if ok1 else "FAIL")

print()
print("=" * 70)
print("TEST 2: Press Tab → focus moves to next widget")
print("=" * 70)
before = root.focus_get()
print(f"before Tab: {before}")
# Generate a real Tab key event on the focused widget
root.event_generate("<Tab>", when="now")
root.update()
after = root.focus_get()
print(f"after Tab:  {after}")
ok2 = before is not None and after is not None and str(before) != str(after)
print(f"PASS" if ok2 else "FAIL")

print()
print("=" * 70)
print("TEST 3: Press Shift-Tab → focus moves back")
print("=" * 70)
before = root.focus_get()
print(f"before Shift-Tab: {before}")
root.event_generate("<Shift-Tab>", when="now")
root.update()
after_st = root.focus_get()
print(f"after Shift-Tab:  {after_st}")
ok3 = before is not None and after_st is not None and str(before) != str(after_st)
print(f"PASS" if ok3 else "FAIL")

print()
print("=" * 70)
print("TEST 4: Focus a nav button, press Down arrow → focus next nav button")
print("=" * 70)
home_canvas = getattr(root._nav_btns["home"], "_canvas", None)
home_canvas.focus_set()
root.update()
before = root.focus_get()
print(f"before Down: {before}")
home_canvas.event_generate("<Down>", when="now")
root.update()
after_dn = root.focus_get()
print(f"after Down:  {after_dn}")
ok4 = before is not None and after_dn is not None and str(before) != str(after_dn)
print(f"PASS" if ok4 else "FAIL")

print()
print("=" * 70)
print("TEST 5: Focus a nav button, press Enter → page switches")
print("=" * 70)
# Try the Models nav button
models_btn = root._nav_btns.get("models")
models_canvas = getattr(models_btn, "_canvas", models_btn)
models_canvas.focus_set()
root.update()
before_page = root._current_page
print(f"before Enter: current_page={before_page!r}")
models_canvas.event_generate("<Return>", when="now")
root.update()
time.sleep(0.1)
root.update()
after_page = root._current_page
print(f"after Enter:  current_page={after_page!r}")
ok5 = after_page == "models"
print(f"PASS" if ok5 else "FAIL")

print()
print("=" * 70)
print("TEST 6: Press Space on a button → activates")
print("=" * 70)
home_canvas2 = getattr(root._nav_btns["home"], "_canvas", None)
home_canvas2.focus_set()
root.update()
before_page = root._current_page
print(f"before Space: current_page={before_page!r}")
home_canvas2.event_generate("<space>", when="now")
root.update()
time.sleep(0.1)
root.update()
after_page = root._current_page
print(f"after Space:  current_page={after_page!r}")
ok6 = after_page == "home"
print(f"PASS" if ok6 else "FAIL")

print()
print("=" * 70)
print("TEST 7: Models page — Right arrow on type-filter walks the row")
print("=" * 70)
# Navigate to Models page first
root._switch_page("models")
root.update()
time.sleep(0.5)
# Let the model rows populate
for _ in range(20):
    root.update()
    time.sleep(0.05)
type_btns = list(root._type_btns.values())
print(f"type filter buttons: {len(type_btns)}")
ok7 = False
if len(type_btns) >= 2:
    first_canvas = getattr(type_btns[0], "_canvas", None)
    first_canvas.focus_set()
    root.update()
    before = root.focus_get()
    print(f"before Right: {before}")
    first_canvas.event_generate("<Right>", when="now")
    root.update()
    after_r = root.focus_get()
    print(f"after Right:  {after_r}")
    expected = getattr(type_btns[1], "_canvas", None)
    ok7 = after_r is not None and str(after_r) == str(expected)
print(f"PASS" if ok7 else "FAIL")

print()
print("=" * 70)
print("TEST 8: Models page — Down arrow on model list row walks rows")
print("=" * 70)
rows = list(getattr(root, "_model_cards", []))
print(f"model list rows: {len(rows)}")
ok8 = False
ok8b = False
if len(rows) >= 2:
    row0_canvas = getattr(rows[0], "_canvas", None)
    row0_canvas.focus_set()
    root.update()
    time.sleep(0.05)
    root.update()
    # Verify focus-follows-selection: focusing the row should have updated
    # the selected model id.
    sel_after_focus = root._selected_model_id
    expected_id_0 = rows[0].model.get("id")
    print(f"selected_model_id after focusing row 0: {sel_after_focus!r}")
    print(f"expected (row 0 id): {expected_id_0!r}")
    ok8b = sel_after_focus == expected_id_0
    # Now arrow down
    before = root.focus_get()
    print(f"before Down on row 0: {before}")
    row0_canvas.event_generate("<Down>", when="now")
    root.update()
    after_dn = root.focus_get()
    print(f"after Down: {after_dn}")
    expected = getattr(rows[1], "_canvas", None)
    ok8 = after_dn is not None and str(after_dn) == str(expected)
print(f"PASS (arrow nav)" if ok8 else "FAIL (arrow nav)")
print(f"PASS (focus-follows-selection)" if ok8b else "FAIL (focus-follows-selection)")

print()
print("=" * 70)
print("TEST 9: Focus ring items carry the _la_focus_ring tag")
print("=" * 70)
# Verify the tag-raise behavior is in effect: when a button is focused,
# the ring items should carry our tag.
nav_canvas = getattr(root._nav_btns["chat"], "_canvas", None)
nav_canvas.focus_set()
root.update()
time.sleep(0.1)
root.update()
tagged = nav_canvas.find_withtag("_la_focus_ring")
print(f"items with _la_focus_ring tag on focused chat button canvas: {len(tagged)}")
ok9 = len(tagged) >= 1
print(f"PASS" if ok9 else "FAIL")

print()
print("=" * 70)
results = [ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok8b, ok9]
labels = ["t1 initial focus", "t2 Tab", "t3 Shift-Tab", "t4 Down on nav",
          "t5 Enter switches page", "t6 Space switches page",
          "t7 Right walks type filter row", "t8 Down walks model list",
          "t8b focus-follows-selection", "t9 focus ring tagged"]
print(f"SUMMARY: {sum(results)} / {len(results)} passed")
for label, r in zip(labels, results):
    print(f"  {label}: {'PASS' if r else 'FAIL'}")
print("=" * 70)

root.after(200, root.destroy)
root.mainloop()
sys.exit(0 if all(results) else 1)

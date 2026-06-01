# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""CTkToplevel dialogs that wrap :mod:`src.migration` for the GUI.

This module is kept thin: every dialog owns nothing more than a
:class:`MigrationEngine` reference + a Tk after-loop pump that drains
``progress_callback`` events onto the main thread.  All "business logic"
(pre-flight, state machine, verification) lives in :mod:`src.migration`
so it stays unit-testable without Tk.

Public dialogs
~~~~~~~~~~~~~~
* :class:`MigrationProgressDialog` — live progress with per-file path,
  percent, bytes, ETA, and an always-enabled Cancel button.
* :class:`PreflightFailureDialog` — actionable "your target drive has X GB
  but needs Y GB" message that names a remediation.
* :class:`ResumeDialog` — Resume / Roll back / Decide later for crash
  recovery.

Why this is separate from ``migration.py``: keeping the engine free of
Tk imports means tests can exercise the state machine on Linux CI / from
a headless Python REPL without standing up Toplevel windows.  The UI
module imports the engine, not the other way round.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import customtkinter as ctk
except Exception:  # pragma: no cover - tk-less smoke tests
    ctk = None  # type: ignore[assignment]

from src.migration import (
    MigrationCancelled,
    MigrationEngine,
    MigrationPhase,
    MigrationPlan,
    MigrationState,
    MigrationVerifyFailed,
    PreflightResult,
    ProgressEvent,
    USER_PROMPT_RESUME_PHASES,
)


def _fmt_bytes(n: int) -> str:
    """Compact human-friendly byte formatter."""
    if n <= 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    i = 0
    f = float(n)
    while f >= 1024.0 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.1f} {units[i]}" if i > 0 else f"{int(f)} {units[i]}"


def _fmt_eta(bytes_done: int, bytes_total: int, started_at: float) -> str:
    """Return an ETA string like ``~2m 14s`` or ``calculating…``."""
    if bytes_done <= 0 or bytes_total <= 0:
        return "calculating…"
    elapsed = max(0.001, time.time() - started_at)
    rate = bytes_done / elapsed
    if rate <= 0:
        return "calculating…"
    remaining = (bytes_total - bytes_done) / rate
    if remaining < 1:
        return "<1s"
    mins, secs = divmod(int(remaining), 60)
    if mins >= 60:
        hrs, mins = divmod(mins, 60)
        return f"~{hrs}h {mins}m"
    if mins >= 1:
        return f"~{mins}m {secs}s"
    return f"~{secs}s"


# ── MigrationProgressDialog ──────────────────────────────────────────────────


class MigrationProgressDialog:
    """Modal-ish progress dialog with always-enabled Cancel.

    Usage::

        dialog = MigrationProgressDialog(parent, engine, on_done=lambda ok: ...)
        dialog.start()
        # dialog drives the engine in a worker thread; on_done is called on
        # the main thread when the engine finishes (success, cancel, or fail).
    """

    def __init__(
        self,
        parent,
        engine: MigrationEngine,
        *,
        on_done: Optional[Callable[[bool, Optional[BaseException]], None]] = None,
        title: Optional[str] = None,
    ):
        if ctk is None:
            raise RuntimeError("customtkinter is not available")
        self.engine = engine
        self.on_done = on_done or (lambda ok, exc: None)
        self._progress_queue: "queue.Queue[ProgressEvent]" = queue.Queue(maxsize=512)
        self._engine_thread: Optional[threading.Thread] = None
        self._final_exc: Optional[BaseException] = None
        self._started_at = time.time()
        self._cancel_clicked = False

        self.window = ctk.CTkToplevel(parent)
        self.window.title(title or f"Migrating {engine.plan.kind}…")
        self.window.geometry("620x320")
        try:
            self.window.transient(parent)
        except Exception:
            pass
        # Block window-close via [X] until cancel completes.
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        ctk.CTkLabel(
            self.window,
            text=f"Moving {engine.plan.kind} → {engine.plan.target}",
            font=ctk.CTkFont(size=14, weight="bold"),
            wraplength=560,
        ).pack(padx=20, pady=(18, 4), anchor="w")
        ctk.CTkLabel(
            self.window,
            text=(
                "You can cancel at any time — your existing files are preserved "
                "until the new copy is fully verified."
            ),
            wraplength=560,
            justify="left",
        ).pack(padx=20, anchor="w")

        self.phase_label = ctk.CTkLabel(self.window, text="Phase: pre-flight…", anchor="w")
        self.phase_label.pack(padx=20, pady=(14, 4), fill="x")

        self.progress = ctk.CTkProgressBar(self.window, height=18)
        self.progress.set(0.0)
        self.progress.pack(padx=20, pady=(4, 4), fill="x")

        self.bytes_label = ctk.CTkLabel(self.window, text="0 B of 0 B   •   ETA calculating…", anchor="w")
        self.bytes_label.pack(padx=20, pady=(2, 6), fill="x")

        self.file_label = ctk.CTkLabel(
            self.window, text="", anchor="w", wraplength=560,
            font=ctk.CTkFont(size=11),
        )
        self.file_label.pack(padx=20, pady=(0, 8), fill="x")

        btn_row = ctk.CTkFrame(self.window, fg_color="transparent")
        btn_row.pack(side="bottom", fill="x", padx=20, pady=(8, 16))
        self.cancel_button = ctk.CTkButton(
            btn_row, text="Cancel migration", width=160,
            command=self._on_cancel,
        )
        self.cancel_button.pack(side="right")

    def start(self) -> None:
        """Begin the worker thread + start the after-loop pump."""
        # Wire engine progress callback into a thread-safe queue.
        self.engine.progress_callback = self._enqueue_progress  # type: ignore[assignment]
        self._engine_thread = threading.Thread(
            target=self._run_engine, name="LocalAI-MigrationEngine", daemon=True,
        )
        self._engine_thread.start()
        self.window.after(120, self._pump)

    # ── Internals ─────────────────────────────────────────────────────────

    def _enqueue_progress(self, evt: ProgressEvent) -> None:
        try:
            self._progress_queue.put_nowait(evt)
        except queue.Full:
            pass  # OK to drop; we'll catch up on the next event.

    def _run_engine(self) -> None:
        try:
            self.engine.run()
        except BaseException as exc:  # noqa: BLE001 — we surface everything
            self._final_exc = exc

    def _pump(self) -> None:
        try:
            latest: Optional[ProgressEvent] = None
            while True:
                latest = self._progress_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._render(latest)
        thread = self._engine_thread
        if thread is not None and thread.is_alive():
            self.window.after(150, self._pump)
            return
        # Engine finished.
        self._finalize()

    def _render(self, evt: ProgressEvent) -> None:
        if evt.bytes_total > 0:
            frac = max(0.0, min(1.0, evt.bytes_done / evt.bytes_total))
        else:
            frac = 0.0
        try:
            self.progress.set(frac)
        except Exception:
            pass
        eta = _fmt_eta(evt.bytes_done, evt.bytes_total, self._started_at)
        self.bytes_label.configure(
            text=f"{_fmt_bytes(evt.bytes_done)} of {_fmt_bytes(evt.bytes_total)}   •   ETA {eta}"
        )
        if evt.files_total > 0:
            self.phase_label.configure(
                text=f"Phase: {self.engine.phase.value}   •   "
                     f"{evt.files_done} of {evt.files_total} files"
            )
        else:
            self.phase_label.configure(text=f"Phase: {self.engine.phase.value}")
        if evt.current_file:
            self.file_label.configure(text=f"…{evt.current_file[-80:]}")

    def _on_cancel(self) -> None:
        if self._cancel_clicked:
            return
        self._cancel_clicked = True
        try:
            self.cancel_button.configure(state="disabled", text="Cancelling…")
        except Exception:
            pass
        try:
            self.engine.cancel()
        except Exception:
            pass

    def _finalize(self) -> None:
        exc = self._final_exc
        ok = exc is None and self.engine.phase == MigrationPhase.DONE
        try:
            self.window.destroy()
        except Exception:
            pass
        try:
            self.on_done(ok, exc)
        except Exception:
            pass


# ── PreflightFailureDialog ───────────────────────────────────────────────────


class PreflightFailureDialog:
    """Modal "we can't start because…" dialog with a single OK button."""

    def __init__(self, parent, result: PreflightResult, *, on_close: Optional[Callable[[], None]] = None):
        if ctk is None:
            raise RuntimeError("customtkinter is not available")
        self.on_close = on_close or (lambda: None)
        self.window = ctk.CTkToplevel(parent)
        self.window.title("Migration can't start")
        self.window.geometry("560x260")
        try:
            self.window.transient(parent)
            self.window.grab_set()
        except Exception:
            pass
        ctk.CTkLabel(
            self.window,
            text="Migration can't start",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).pack(padx=20, pady=(18, 4), fill="x")
        ctk.CTkLabel(
            self.window,
            text=result.reason or "Pre-flight check failed.",
            wraplength=520,
            justify="left",
            anchor="w",
        ).pack(padx=20, pady=(4, 8), fill="x")
        if result.target_required > 0:
            ctk.CTkLabel(
                self.window,
                text=(
                    f"Required: {_fmt_bytes(result.target_required)}   •   "
                    f"Free: {_fmt_bytes(result.target_free)}"
                ),
                anchor="w",
            ).pack(padx=20, pady=(0, 8), fill="x")
        ctk.CTkButton(
            self.window, text="OK", width=120, command=self._close,
        ).pack(side="bottom", pady=(0, 16))

    def _close(self) -> None:
        try:
            self.window.destroy()
        except Exception:
            pass
        self.on_close()


# ── ResumeDialog ─────────────────────────────────────────────────────────────


class ResumeDialog:
    """Three-way crash-recovery dialog.

    Buttons:
        Resume — keep going from the persisted phase.
        Roll back — delete the partial target, restore the user's previous state.
        Decide later — close (we'll prompt again next launch).
    """

    RESUME = "resume"
    ROLL_BACK = "roll_back"
    LATER = "later"

    def __init__(self, parent, state: MigrationState, *, on_choice: Optional[Callable[[str], None]] = None):
        if ctk is None:
            raise RuntimeError("customtkinter is not available")
        self.state = state
        self.on_choice = on_choice or (lambda c: None)
        self.window = ctk.CTkToplevel(parent)
        self.window.title("Unfinished migration found")
        self.window.geometry("600x300")
        try:
            self.window.transient(parent)
            self.window.grab_set()
        except Exception:
            pass

        ctk.CTkLabel(
            self.window,
            text="Unfinished migration found",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).pack(padx=20, pady=(18, 4), fill="x")

        ctk.CTkLabel(
            self.window,
            text=(
                f"A previous attempt to move {state.kind} was interrupted at "
                f"the {state.phase.replace('_', ' ')} stage.\n\n"
                f"From: {state.source}\n"
                f"To:   {state.target}\n\n"
                f"Choose Resume to continue (recommended), Roll back to discard "
                f"the partial copy and keep your existing files, or Decide later "
                f"to dismiss this for now."
            ),
            wraplength=540,
            justify="left",
            anchor="w",
        ).pack(padx=20, pady=(4, 12), fill="x")

        row = ctk.CTkFrame(self.window, fg_color="transparent")
        row.pack(side="bottom", fill="x", padx=20, pady=(8, 16))
        ctk.CTkButton(
            row, text="Decide later", width=120, command=lambda: self._click(self.LATER),
        ).pack(side="left")
        ctk.CTkButton(
            row, text="Roll back", width=120, command=lambda: self._click(self.ROLL_BACK),
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            row, text="Resume", width=120, command=lambda: self._click(self.RESUME),
        ).pack(side="right")

    def _click(self, choice: str) -> None:
        try:
            self.window.destroy()
        except Exception:
            pass
        self.on_choice(choice)


__all__ = [
    "MigrationProgressDialog",
    "PreflightFailureDialog",
    "ResumeDialog",
]

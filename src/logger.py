# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Thread-safe in-memory log store + optional file log.
"""

import threading
import time
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Optional

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
CATEGORY_SYSTEM = "SYSTEM"
CATEGORY_STARTUP = "STARTUP"
CATEGORY_MODEL_PULL = "MODEL_PULL"
CATEGORY_CHAT = "CHAT"
CATEGORY_BENCHMARK = "BENCHMARK"
CATEGORY_IMAGE_GEN = "IMAGE_GEN"
CATEGORY_COMFYUI = "COMFYUI"
CATEGORY_TOOLBOX = "TOOLBOX"
CATEGORIES = (
    CATEGORY_SYSTEM,
    CATEGORY_STARTUP,
    CATEGORY_MODEL_PULL,
    CATEGORY_CHAT,
    CATEGORY_BENCHMARK,
    CATEGORY_IMAGE_GEN,
    CATEGORY_COMFYUI,
    CATEGORY_TOOLBOX,
)

_lock = threading.Lock()
_entries: deque = deque(maxlen=2000)
_listeners: list[Callable] = []
_log_file: Optional[Path] = None
_max_log_bytes = 10 * 1024 * 1024
_backup_count = 3
_listener_state = threading.local()


def set_log_file(path: str | Path) -> None:
    global _log_file
    _log_file = Path(path)
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed()


def _rotate_if_needed() -> None:
    if not _log_file or not _log_file.exists():
        return
    try:
        if _log_file.stat().st_size < _max_log_bytes:
            return
        for idx in range(_backup_count - 1, 0, -1):
            src = _log_file.with_name(f"{_log_file.name}.{idx}")
            dst = _log_file.with_name(f"{_log_file.name}.{idx + 1}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        first = _log_file.with_name(f"{_log_file.name}.1")
        if first.exists():
            first.unlink()
        _log_file.rename(first)
    except OSError as exc:
        print(f"LocalAI logger: could not rotate {_log_file}: {exc}", file=sys.stderr)


def add_listener(cb: Callable[[dict], None]) -> None:
    with _lock:
        _listeners.append(cb)


def remove_listener(cb: Callable) -> None:
    with _lock:
        if cb in _listeners:
            _listeners.remove(cb)


def _normalize_category(category: str | None) -> str:
    category = str(category or CATEGORY_SYSTEM).strip().upper()
    return category if category in CATEGORIES else CATEGORY_SYSTEM


def _log(level: str, msg: str, *, category: str = CATEGORY_SYSTEM) -> None:
    category = _normalize_category(category)
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "category": category,
        "msg": msg,
    }
    with _lock:
        _entries.append(entry)
        listeners = list(_listeners)
        if _log_file:
            try:
                _rotate_if_needed()
                with open(_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{entry['time']}] [{level}] [{category}] {msg}\n")
            except OSError as exc:
                print(f"LocalAI logger: could not write {_log_file}: {exc}", file=sys.stderr)
    if getattr(_listener_state, "notifying", False):
        return
    _listener_state.notifying = True
    try:
        callbacks = list(listeners)
        for cb in callbacks:
            try:
                cb(entry)
            except Exception as exc:
                print(f"LocalAI logger: listener failed: {exc}", file=sys.stderr)
    finally:
        _listener_state.notifying = False


def debug(msg: str, *, category: str = CATEGORY_SYSTEM) -> None:   _log("DEBUG",   msg, category=category)
def info(msg: str, *, category: str = CATEGORY_SYSTEM) -> None:    _log("INFO",    msg, category=category)
def warning(msg: str, *, category: str = CATEGORY_SYSTEM) -> None: _log("WARNING", msg, category=category)
def error(msg: str, *, category: str = CATEGORY_SYSTEM) -> None:   _log("ERROR",   msg, category=category)


def get_entries(min_level: str = "DEBUG", *, category: str | None = None) -> list[dict]:
    min_idx = LEVELS.index(min_level) if min_level in LEVELS else 0
    normalized_category = None if not category else _normalize_category(category)
    with _lock:
        entries = [e for e in _entries if LEVELS.index(e["level"]) >= min_idx]
        if normalized_category:
            entries = [
                e for e in entries
                if _normalize_category(e.get("category")) == normalized_category
            ]
        return entries


def clear() -> None:
    with _lock:
        _entries.clear()

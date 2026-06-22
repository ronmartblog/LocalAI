# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
Small persistence helpers shared by config/catalog-style JSON files.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to *path* via a same-directory temp file and atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(tmp, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        # Durability: fsync the containing directory so the rename itself
        # survives a power loss — the step most atomic-write recipes forget.
        # Without it a crash right after os.replace can leave the directory
        # entry pointing at neither file, which for the GPU cache means a
        # silent re-detect on the next launch. Best-effort: some platforms
        # (notably Windows) don't support directory fds; never fatal.
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError, ValueError):
            pass
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Serialize *payload* as JSON and atomically replace *path*."""
    text = json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii)
    atomic_write_text(path, text + "\n")

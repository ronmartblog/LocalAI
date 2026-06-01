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

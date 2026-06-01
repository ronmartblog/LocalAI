"""Date-based version helpers for LocalAI Studio.

Version format: ``<YYYY>.<MM>.<DD>.<N>`` where ``YYYY``/``MM``/``DD`` are the
publish day (zero-padded month and day, 4-digit year) and ``N`` is a
zero-based increment that resets to ``0`` at the start of each calendar day.

``APP_VERSION`` in ``src/app.py`` is the single source of truth. This module
exposes a stateless ``next_version()`` that derives the next publish version
from the current value plus today's date. The publish flow uses the result
verbatim — no external counter file, nothing to keep in sync.

The same algorithm is mirrored in PowerShell inside ``C:/Plans/Publish.md``
so the cross-platform publish scripts share one rule.
"""

from __future__ import annotations

from datetime import date


def next_version(current: str, today: date | None = None) -> str:
    """Return the next date-based version string.

    Rules:
        * Same calendar day as ``current`` → increment the last segment by 1.
        * New calendar day (or ``current`` doesn't match today's
          ``YYYY.MM.DD`` prefix, e.g. migrating from the legacy ``x.y.z``
          scheme) → reset to ``YYYY.MM.DD.0``.

    Args:
        current: The currently installed ``APP_VERSION`` literal.
        today: Override "today" for deterministic tests. Defaults to
            ``date.today()`` in the local timezone (publish runs on the
            developer's machine, so local-day semantics are correct).

    Returns:
        The next version string, never equal to ``current``.
    """
    today = today or date.today()
    ymd = f"{today.year:04d}.{today.month:02d}.{today.day:02d}"
    parts = (current or "").split(".")
    if len(parts) == 4 and ".".join(parts[:3]) == ymd:
        try:
            n = int(parts[3]) + 1
        except ValueError:
            n = 0
        return f"{ymd}.{n}"
    return f"{ymd}.0"

"""Unit tests for ``src.versioning.next_version``."""

from __future__ import annotations

import unittest
from datetime import date

from src.versioning import next_version


class NextVersionTests(unittest.TestCase):
    def test_migration_from_legacy_semver_starts_at_zero(self):
        self.assertEqual(
            next_version("5.5.18", today=date(2026, 5, 30)),
            "2026.05.30.0",
        )

    def test_same_day_increments(self):
        self.assertEqual(
            next_version("2026.05.30.0", today=date(2026, 5, 30)),
            "2026.05.30.1",
        )
        self.assertEqual(
            next_version("2026.05.30.7", today=date(2026, 5, 30)),
            "2026.05.30.8",
        )

    def test_new_day_resets_to_zero(self):
        self.assertEqual(
            next_version("2026.05.30.7", today=date(2026, 5, 31)),
            "2026.05.31.0",
        )

    def test_month_and_day_are_zero_padded(self):
        self.assertEqual(
            next_version("anything", today=date(2026, 1, 3)),
            "2026.01.03.0",
        )

    def test_non_integer_increment_segment_resets_to_zero(self):
        self.assertEqual(
            next_version("2026.05.30.beta", today=date(2026, 5, 30)),
            "2026.05.30.0",
        )

    def test_empty_or_none_current_starts_at_zero(self):
        self.assertEqual(
            next_version("", today=date(2026, 5, 30)),
            "2026.05.30.0",
        )
        self.assertEqual(
            next_version(None, today=date(2026, 5, 30)),  # type: ignore[arg-type]
            "2026.05.30.0",
        )


if __name__ == "__main__":
    unittest.main()

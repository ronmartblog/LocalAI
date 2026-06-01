"""Pytest session-scope autouse fixture: swap the runtime ``skus.json`` for
the in-repo sample fixture (``tests/fixtures/sample_skus.json``).

Why:
    The app loads SKU profiles from ``<app_root>/skus.json`` at module
    import time (``src.system_info._init_benchmark_sku_profiles``). That
    file is intentionally gitignored and is the maintainer's private SKU
    catalog. In a public clone the file is absent; on the maintainer's
    machine it carries vendor-specific names. Either way, contract tests
    that assert on SKU display names ("Small CPU", "GPU Workstation",
    etc.) cannot rely on it.

What:
    A session-scope autouse fixture loads the in-repo sample fixture
    and mutates ``system_info.BENCHMARK_SKU_PROFILES`` IN PLACE (slice
    assignment) so every importer that captured a reference to the list
    sees the swap atomically. The original list is restored at session
    teardown.

This keeps the test suite hermetic and deterministic regardless of which
machine it runs on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import system_info


_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_skus.json"


@pytest.fixture(scope="session", autouse=True)
def _swap_runtime_skus_for_sample_fixture():
    assert _FIXTURE_PATH.exists(), f"missing sample SKU fixture: {_FIXTURE_PATH}"
    cfg = system_info.load_optional_sku_config(_FIXTURE_PATH)
    sample_skus = list(cfg.get("skus", []))

    saved = list(system_info.BENCHMARK_SKU_PROFILES)
    system_info.BENCHMARK_SKU_PROFILES[:] = sample_skus
    try:
        yield
    finally:
        system_info.BENCHMARK_SKU_PROFILES[:] = saved

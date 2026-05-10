"""Shared pytest fixtures for accrue tests.

Auto-isolates the on-disk cache. ``EnrichmentConfig.cache_dir`` defaults
to the relative path ``.accrue``, so without isolation every test that
runs a pipeline with the default config writes to the project's
``.accrue/cache.db`` and leaks state into subsequent tests. Now that
``enable_caching`` defaults to True, that leak is no longer dormant.

Tests that explicitly want to test caching across runs can still pass
``cache_dir=str(tmp_path / "x")`` to override.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache_dir(tmp_path, monkeypatch):
    """Run every test from a fresh cwd so the relative ``.accrue/`` cache
    is per-test rather than shared across the whole suite. monkeypatch
    restores the cwd after each test."""
    monkeypatch.chdir(tmp_path)

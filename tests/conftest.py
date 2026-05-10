"""Shared pytest fixtures for accrue tests.

Auto-isolates the on-disk cache.

``EnrichmentConfig.cache_dir`` defaults to the relative string ``.accrue``.
With ``enable_caching`` now defaulting to True, every test that
constructs a default config writes to ``./.accrue/cache.db`` — and that
file persists across tests, causing cache hits to leak state.

We can't cleanly patch the default itself: ``@dataclass`` binds field
defaults into ``__init__.__defaults__`` at class creation, so a
``monkeypatch.setattr`` on ``__dataclass_fields__["cache_dir"].default``
is a no-op for fresh ``EnrichmentConfig()`` calls. Instead we change
the cwd per test so the relative path resolves to a fresh location.
This is broader scope than ideal, but it's reliable, doesn't touch
library code, and accrue's tests don't currently read fixtures by
relative path (they use absolute paths or ``tmp_path``).

Tests that need a specific cache location can still pass
``cache_dir=str(tmp_path / "x")`` to override.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

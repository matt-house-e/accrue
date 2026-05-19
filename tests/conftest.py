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


@pytest.fixture(autouse=True)
def _isolate_xdg_dirs(tmp_path, monkeypatch):
    """Scope XDG-backed defaults (cache_dir, checkpoint_dir) to a per-test tmp dir.

    ``EnrichmentConfig`` defaults ``cache_dir`` to ``$XDG_CACHE_HOME/accrue`` and
    ``checkpoint_dir`` to ``$XDG_STATE_HOME/accrue`` via ``default_factory``.
    Without isolation, every test sharing the developer's real XDG dirs would see
    cache + checkpoint state bleed between runs — causing spurious cache hits and
    ``complete``-mock-call-count == 0 failures.

    Tests that explicitly pass ``cache_dir=`` / ``checkpoint_dir=`` are unaffected.
    The ``TestXdg*`` classes in ``test_config.py`` / ``test_checkpoint.py`` set
    these vars themselves inside the test body, which overrides this fixture's
    values for that test (monkeypatch is stack-scoped).
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))


@pytest.fixture(autouse=True)
def _scrub_provider_env_vars(monkeypatch):
    """Remove real provider keys and substitute a dummy so non-empty checks pass.

    Unit tests must never read credentials from the developer's shell.
    This fixture scrubs all known provider env vars and injects a clearly
    fake OpenAI key so code paths that assert ``key != ""`` still reach
    the mock layer rather than blowing up at key-validation time.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-mock-key-not-real")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Raise on any real outbound HTTP in unit tests.

    Patches both the async and sync ``httpx`` send methods so accidental
    real API calls fail loudly with a clear message rather than hanging or
    leaking billing charges.  Integration tests that need real network access
    must be decorated with ``@pytest.mark.integration`` and kept under
    ``tests/integration/`` (excluded from the default pytest run via
    ``norecursedirs``).
    """
    import httpx

    _msg = (
        "Unit test attempted a real network call — "
        "mock the provider client or use pytest.mark.integration"
    )

    def _raise_async(self, *args, **kwargs):
        raise RuntimeError(_msg)

    def _raise_sync(self, *args, **kwargs):
        raise RuntimeError(_msg)

    monkeypatch.setattr(httpx.AsyncClient, "send", _raise_async)
    monkeypatch.setattr(httpx.Client, "send", _raise_sync)

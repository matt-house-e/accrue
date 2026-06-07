"""Unit tests for CacheManager and cache key computation."""

from __future__ import annotations

import concurrent.futures
import os
import sqlite3
import sys
import threading
import time
from unittest.mock import patch

import pytest

from accrue.core.cache import (
    CacheManager,
    _compute_step_cache_key,
    canonical_json,
    compute_cache_key,
)
from accrue.core.config import EnrichmentConfig

# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


class TestCacheManager:
    def test_get_miss_returns_none(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        assert mgr.get("nonexistent") is None
        mgr.close()

    def test_set_get_roundtrip(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        mgr.set("k1", "step_a", {"score": 42})
        assert mgr.get("k1") == {"score": 42}
        mgr.close()

    def test_overwrite_existing_key(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        mgr.set("k1", "step_a", {"v": 1})
        mgr.set("k1", "step_a", {"v": 2})
        assert mgr.get("k1") == {"v": 2}
        mgr.close()

    def test_ttl_expiry(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=1)
        with patch("accrue.core.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            mgr.set("k1", "step_a", {"v": 1})

            # Still valid
            mock_time.time.return_value = 1000.5
            assert mgr.get("k1") == {"v": 1}

            # Expired
            mock_time.time.return_value = 1002.0
            assert mgr.get("k1") is None
        mgr.close()

    def test_no_expiry_when_ttl_zero(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=0)
        with patch("accrue.core.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            mgr.set("k1", "step_a", {"v": 1})

            mock_time.time.return_value = 999999.0
            assert mgr.get("k1") == {"v": 1}
        mgr.close()

    def test_delete_step(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        mgr.set("k1", "step_a", {"v": 1})
        mgr.set("k2", "step_a", {"v": 2})
        mgr.set("k3", "step_b", {"v": 3})

        deleted = mgr.delete_step("step_a")
        assert deleted == 2
        assert mgr.get("k1") is None
        assert mgr.get("k2") is None
        assert mgr.get("k3") == {"v": 3}
        mgr.close()

    def test_delete_all(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        mgr.set("k1", "step_a", {"v": 1})
        mgr.set("k2", "step_b", {"v": 2})

        deleted = mgr.delete_all()
        assert deleted == 2
        assert mgr.get("k1") is None
        assert mgr.get("k2") is None
        mgr.close()

    def test_cleanup_expired(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=1)
        with patch("accrue.core.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            mgr.set("k1", "step_a", {"v": 1})
            mgr.set("k2", "step_a", {"v": 2})

            mock_time.time.return_value = 1002.0
            cleaned = mgr.cleanup_expired()
            assert cleaned == 2
        mgr.close()

    def test_db_persists_across_close_reopen(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        mgr.set("k1", "step_a", {"v": 1})
        mgr.close()

        mgr2 = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        assert mgr2.get("k1") == {"v": 1}
        mgr2.close()

    def test_db_created_in_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "my_cache"
        mgr = CacheManager(cache_dir=str(cache_dir), ttl=3600)
        mgr.set("k1", "step_a", {"v": 1})
        assert (cache_dir / "cache.db").exists()
        mgr.close()

    def test_close_idempotent(self, tmp_path):
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        mgr.set("k1", "step_a", {"v": 1})
        mgr.close()
        mgr.close()  # Should not raise

    # ------------------------------------------------------------------
    # Concurrency / thread-safety tests
    # ------------------------------------------------------------------

    def test_concurrent_set_from_two_threads(self, tmp_path):
        """Two threads sharing a CacheManager can both call .set() safely."""
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        errors: list[Exception] = []

        def worker(key: str) -> None:
            try:
                mgr.set(key, "step_a", {"key": key})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(worker, f"key_{i}") for i in range(2)]
            concurrent.futures.wait(futs)

        assert errors == [], f"Unexpected errors: {errors}"
        assert mgr.get("key_0") == {"key": "key_0"}
        assert mgr.get("key_1") == {"key": "key_1"}
        mgr.close()

    def test_cross_thread_get_no_programming_error(self, tmp_path):
        """Connection opened in thread A, .get() from thread B — no ProgrammingError."""
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        # Force connection open on the main thread
        mgr.set("tk", "step_a", {"v": 99})

        result: list = []
        errors: list[Exception] = []

        def reader() -> None:
            try:
                result.append(mgr.get("tk"))
            except sqlite3.ProgrammingError as exc:
                errors.append(exc)

        t = threading.Thread(target=reader)
        t.start()
        t.join()

        assert errors == [], f"ProgrammingError raised: {errors}"
        assert result == [{"v": 99}]
        mgr.close()

    def test_busy_timeout_pragma_is_set(self, tmp_path):
        """PRAGMA busy_timeout is 5000 ms."""
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        conn = mgr._ensure_connection()
        value = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert value == 5000
        mgr.close()

    def test_ttl_cleanup_on_init_removes_expired_rows(self, tmp_path):
        """Expired rows inserted directly are removed when a fresh CacheManager is opened."""
        # Seed a db with an already-expired row via a raw connection
        db_path = tmp_path / "cache.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key        TEXT PRIMARY KEY,
                step_name  TEXT NOT NULL,
                value      TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )
        """)
        # expires_at in the past
        conn.execute(
            "INSERT INTO cache (key, step_name, value, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("expired_key", "step_x", '{"v":1}', time.time() - 7200, time.time() - 3600),
        )
        conn.commit()
        conn.close()

        # Fresh CacheManager should trigger cleanup on first connection
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        # Trigger _ensure_connection via a benign .get()
        mgr.get("unrelated_key")

        # Verify expired row is gone via direct DB read
        raw = sqlite3.connect(str(db_path))
        row = raw.execute("SELECT key FROM cache WHERE key = 'expired_key'").fetchone()
        raw.close()

        assert row is None, "Expired row should have been cleaned up on init"
        mgr.close()


# ---------------------------------------------------------------------------
# canonical_json + compute_cache_key
# ---------------------------------------------------------------------------


class TestCanonicalJson:
    def test_sorted_keys(self):
        assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    def test_deterministic(self):
        obj = {"z": [3, 2, 1], "a": {"nested": True}}
        assert canonical_json(obj) == canonical_json(obj)

    def test_default_str_fallback(self):
        """Non-serializable objects fall back to str()."""
        result = canonical_json({"s": set()})
        assert "set()" in result


class TestComputeCacheKey:
    def test_deterministic(self):
        k1 = compute_cache_key(step="a", row={"x": 1})
        k2 = compute_cache_key(step="a", row={"x": 1})
        assert k1 == k2

    def test_different_row_different_key(self):
        k1 = compute_cache_key(step="a", row={"x": 1})
        k2 = compute_cache_key(step="a", row={"x": 2})
        assert k1 != k2

    def test_different_step_different_key(self):
        k1 = compute_cache_key(step="a", row={"x": 1})
        k2 = compute_cache_key(step="b", row={"x": 1})
        assert k1 != k2

    def test_key_is_hex_sha256(self):
        k = compute_cache_key(step="a", row={})
        assert len(k) == 64
        int(k, 16)  # Valid hex


# ---------------------------------------------------------------------------
# _compute_step_cache_key — duck typing
# ---------------------------------------------------------------------------


class _FakeLLMStep:
    def __init__(
        self,
        name="llm",
        model="gpt-4.1-mini",
        temperature=0.2,
        system_prompt=None,
        system_prompt_header=None,
        max_tokens=None,
        provider_kwargs=None,
        field_specs=None,
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self._custom_system_prompt = system_prompt
        self._system_prompt_header = system_prompt_header
        self.max_tokens = max_tokens
        self.provider_kwargs = provider_kwargs
        self._field_specs = field_specs or {}


class _FakeFunctionStep:
    def __init__(self, name="fn", cache_version=None):
        self.name = name
        self.model = None  # No model attribute signals FunctionStep
        self.cache_version = cache_version


class TestComputeStepCacheKey:
    def test_llm_step_deterministic(self):
        step = _FakeLLMStep()
        row = {"company": "Acme"}
        prior = {"revenue": 100}
        fields = {"market_size": {"prompt": "Estimate TAM"}}

        k1 = _compute_step_cache_key(step, row, prior, fields)
        k2 = _compute_step_cache_key(step, row, prior, fields)
        assert k1 == k2

    def test_different_model_different_key(self):
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(model="gpt-4.1-mini"), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(model="gpt-4.1-nano"), row, {}, {})
        assert k1 != k2

    def test_different_temperature_different_key(self):
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(temperature=0.2), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(temperature=0.8), row, {}, {})
        assert k1 != k2

    def test_different_field_spec_different_key(self):
        step = _FakeLLMStep()
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(step, row, {}, {"f": {"prompt": "A"}})
        k2 = _compute_step_cache_key(step, row, {}, {"f": {"prompt": "B"}})
        assert k1 != k2

    def test_different_system_prompt_different_key(self):
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(system_prompt="v1"), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(system_prompt="v2"), row, {}, {})
        assert k1 != k2

    def test_different_system_prompt_header_different_key(self):
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(system_prompt_header="header v1"), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(system_prompt_header="header v2"), row, {}, {})
        assert k1 != k2

    def test_system_prompt_header_none_vs_empty_same_key(self):
        """None and empty string both normalize to '' for hashing."""
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(system_prompt_header=None), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(system_prompt_header=""), row, {}, {})
        assert k1 == k2

    def test_function_step_deterministic(self):
        step = _FakeFunctionStep(cache_version="v1")
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(step, row, {}, {})
        k2 = _compute_step_cache_key(step, row, {}, {})
        assert k1 == k2

    def test_function_step_different_cache_version(self):
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeFunctionStep(cache_version="v1"), row, {}, {})
        k2 = _compute_step_cache_key(_FakeFunctionStep(cache_version="v2"), row, {}, {})
        assert k1 != k2

    def test_different_row_data_different_key(self):
        step = _FakeLLMStep()
        k1 = _compute_step_cache_key(step, {"x": 1}, {}, {})
        k2 = _compute_step_cache_key(step, {"x": 2}, {}, {})
        assert k1 != k2

    def test_different_prior_results_different_key(self):
        step = _FakeLLMStep()
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(step, row, {"a": 1}, {})
        k2 = _compute_step_cache_key(step, row, {"a": 2}, {})
        assert k1 != k2

    # ------------------------------------------------------------------
    # New: schema, max_tokens, provider_kwargs
    # ------------------------------------------------------------------

    def test_schema_change_invalidates_cache(self):
        """Changing FieldSpec.type from Number to String produces a different key."""
        from accrue.schemas.field_spec import FieldSpec

        row = {"company": "Acme"}
        step_num = _FakeLLMStep(field_specs={"score": FieldSpec(prompt="Rate 1-10", type="Number")})
        step_str = _FakeLLMStep(field_specs={"score": FieldSpec(prompt="Rate 1-10", type="String")})
        k1 = _compute_step_cache_key(step_num, row, {}, {})
        k2 = _compute_step_cache_key(step_str, row, {}, {})
        assert k1 != k2

    def test_max_tokens_change_invalidates_cache(self):
        """Different max_tokens values produce different cache keys."""
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(max_tokens=100), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(max_tokens=500), row, {}, {})
        assert k1 != k2

    def test_provider_kwargs_change_invalidates_cache(self):
        """Different provider_kwargs produce different cache keys."""
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(provider_kwargs={"effort": "low"}), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(provider_kwargs={"effort": "high"}), row, {}, {})
        assert k1 != k2

    def test_provider_kwargs_none_and_empty_dict_same_key(self):
        """None and empty dict are equivalent for backwards-compat."""
        row = {"company": "Acme"}
        k1 = _compute_step_cache_key(_FakeLLMStep(provider_kwargs=None), row, {}, {})
        k2 = _compute_step_cache_key(_FakeLLMStep(provider_kwargs={}), row, {}, {})
        assert k1 == k2

    def test_function_step_key_unaffected_by_new_inputs(self):
        """FunctionStep key is stable and not influenced by schema/max_tokens/provider_kwargs."""
        row = {"company": "Acme"}
        step = _FakeFunctionStep(cache_version="v1")
        k1 = _compute_step_cache_key(step, row, {}, {})
        k2 = _compute_step_cache_key(step, row, {}, {})
        assert k1 == k2
        # FunctionStep has no model — confirm it takes the function path
        assert not hasattr(step, "max_tokens")
        assert not hasattr(step, "provider_kwargs")


# ---------------------------------------------------------------------------
# XDG defaults + 0o600 permissions
# ---------------------------------------------------------------------------


class TestXdgCacheDirDefault:
    def test_xdg_cache_home_respected(self, monkeypatch, tmp_path):
        """XDG_CACHE_HOME is used as the base for cache_dir."""
        xdg_cache = str(tmp_path / "xdg_cache")
        monkeypatch.setenv("XDG_CACHE_HOME", xdg_cache)
        config = EnrichmentConfig()
        assert config.cache_dir == os.path.join(xdg_cache, "accrue")

    def test_fallback_to_home_cache_when_xdg_unset(self, monkeypatch):
        """Falls back to ~/.cache/accrue when XDG_CACHE_HOME is unset."""
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        config = EnrichmentConfig()
        expected = os.path.join(os.path.expanduser("~"), ".cache", "accrue")
        assert config.cache_dir == expected

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod")
    def test_cache_db_has_0o600_permissions(self, tmp_path):
        """cache.db is created with 0o600 permissions."""
        mgr = CacheManager(cache_dir=str(tmp_path), ttl=3600)
        mgr.set("k", "step", {"v": 1})
        stat = (tmp_path / "cache.db").stat()
        assert stat.st_mode & 0o777 == 0o600
        mgr.close()

    def test_custom_cache_dir_constructor_arg_still_works(self, tmp_path):
        """Regression: explicit cache_dir= overrides the XDG default."""
        custom = str(tmp_path / "custom_cache")
        config = EnrichmentConfig(cache_dir=custom)
        assert config.cache_dir == custom

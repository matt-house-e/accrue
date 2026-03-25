"""Unit tests for CacheManager and cache key computation."""

from __future__ import annotations

from unittest.mock import patch

from accrue.core.cache import (
    CacheManager,
    _compute_step_cache_key,
    canonical_json,
    compute_cache_key,
)

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
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self._custom_system_prompt = system_prompt
        self._system_prompt_header = system_prompt_header


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

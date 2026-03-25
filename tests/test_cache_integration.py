"""Integration tests for caching, list[dict] input, and checkpoint_interval."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pandas as pd

from accrue.core.config import EnrichmentConfig
from accrue.pipeline.pipeline import Pipeline
from accrue.steps.base import StepContext
from accrue.steps.function import FunctionStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counter_fn(fields: list[str], counter: dict[str, int], key: str = "calls"):
    """Return a step function that counts invocations."""

    def fn(ctx: StepContext) -> dict[str, Any]:
        counter[key] = counter.get(key, 0) + 1
        return {f: f"{f}_value" for f in fields}

    return fn


def _config_with_cache(tmp_path, **overrides) -> EnrichmentConfig:
    """Create an EnrichmentConfig with caching enabled pointing at tmp_path."""
    defaults = dict(
        enable_caching=True,
        cache_dir=str(tmp_path),
        cache_ttl=3600,
        enable_progress_bar=False,
        max_workers=3,
    )
    defaults.update(overrides)
    return EnrichmentConfig(**defaults)


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


class TestCacheHitMiss:
    def test_cache_hit_skips_step_execution(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter), fields=["f1"]),
            ]
        )
        config = _config_with_cache(tmp_path)
        df = pd.DataFrame([{"company": "Acme"}, {"company": "Beta"}])

        # First run: all cache misses
        result1 = pipeline.run(df, config)
        assert counter["calls"] == 2
        assert result1.cost.steps["step_a"].cache_misses == 2
        assert result1.cost.steps["step_a"].cache_hits == 0

        # Second run: all cache hits
        counter.clear()
        result2 = pipeline.run(df, config)
        assert counter.get("calls", 0) == 0
        assert result2.cost.steps["step_a"].cache_hits == 2
        assert result2.cost.steps["step_a"].cache_misses == 0

    def test_cache_miss_on_changed_input(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter), fields=["f1"]),
            ]
        )
        config = _config_with_cache(tmp_path)

        pipeline.run(pd.DataFrame([{"company": "Acme"}]), config)
        assert counter["calls"] == 1

        # Different input → cache miss
        counter.clear()
        pipeline.run(pd.DataFrame([{"company": "Beta"}]), config)
        assert counter["calls"] == 1

    def test_cache_disabled_per_step(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=_counter_fn(["f1"], counter),
                    fields=["f1"],
                    cache=False,
                ),
            ]
        )
        config = _config_with_cache(tmp_path)
        df = pd.DataFrame([{"company": "Acme"}])

        pipeline.run(df, config)
        assert counter["calls"] == 1

        # cache=False → always re-executes
        counter.clear()
        pipeline.run(df, config)
        assert counter["calls"] == 1

    def test_cache_disabled_globally(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter), fields=["f1"]),
            ]
        )
        config = EnrichmentConfig(
            enable_caching=False,
            enable_progress_bar=False,
        )
        df = pd.DataFrame([{"company": "Acme"}])

        pipeline.run(df, config)
        assert counter["calls"] == 1

        counter.clear()
        pipeline.run(df, config)
        assert counter["calls"] == 1  # re-executes because caching is off


class TestCacheStats:
    def test_cache_stats_in_result(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter), fields=["f1"]),
            ]
        )
        config = _config_with_cache(tmp_path)
        df = pd.DataFrame([{"company": "Acme"}, {"company": "Beta"}, {"company": "Gamma"}])

        # First run
        result = pipeline.run(df, config)
        usage = result.cost.steps["step_a"]
        assert usage.cache_hits == 0
        assert usage.cache_misses == 3
        assert usage.cache_hit_rate == 0.0
        assert usage.rows_processed == 3

        # Second run — all hits
        result2 = pipeline.run(df, config)
        usage2 = result2.cost.steps["step_a"]
        assert usage2.cache_hits == 3
        assert usage2.cache_misses == 0
        assert usage2.cache_hit_rate == 1.0
        assert usage2.rows_processed == 3


class TestFunctionStepCacheVersion:
    def test_cache_version_invalidation(self, tmp_path):
        counter: dict[str, int] = {}
        df = pd.DataFrame([{"company": "Acme"}])
        config = _config_with_cache(tmp_path)

        pipeline_v1 = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=_counter_fn(["f1"], counter),
                    fields=["f1"],
                    cache_version="v1",
                ),
            ]
        )
        pipeline_v1.run(df, config)
        assert counter["calls"] == 1

        # Same version → cache hit
        counter.clear()
        pipeline_v1.run(df, config)
        assert counter.get("calls", 0) == 0

        # Bump version → cache miss
        counter.clear()
        pipeline_v2 = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=_counter_fn(["f1"], counter),
                    fields=["f1"],
                    cache_version="v2",
                ),
            ]
        )
        pipeline_v2.run(df, config)
        assert counter["calls"] == 1


class TestClearCache:
    def test_clear_cache_all(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter), fields=["f1"]),
            ]
        )
        config = _config_with_cache(tmp_path)
        df = pd.DataFrame([{"company": "Acme"}])

        pipeline.run(df, config)
        assert counter["calls"] == 1

        deleted = pipeline.clear_cache(cache_dir=str(tmp_path))
        assert deleted == 1

        # After clearing, re-executes
        counter.clear()
        pipeline.run(df, config)
        assert counter["calls"] == 1

    def test_clear_cache_by_step(self, tmp_path):
        counter_a: dict[str, int] = {}
        counter_b: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter_a), fields=["f1"]),
                FunctionStep("step_b", fn=_counter_fn(["f2"], counter_b), fields=["f2"]),
            ]
        )
        config = _config_with_cache(tmp_path)
        df = pd.DataFrame([{"company": "Acme"}])

        pipeline.run(df, config)
        assert counter_a["calls"] == 1
        assert counter_b["calls"] == 1

        # Clear only step_a
        deleted = pipeline.clear_cache(step="step_a", cache_dir=str(tmp_path))
        assert deleted == 1

        counter_a.clear()
        counter_b.clear()
        pipeline.run(df, config)
        assert counter_a["calls"] == 1  # re-executed
        assert counter_b.get("calls", 0) == 0  # still cached


class TestCacheTTLExpiry:
    def test_cache_ttl_expiry(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter), fields=["f1"]),
            ]
        )
        config = _config_with_cache(tmp_path, cache_ttl=1)
        df = pd.DataFrame([{"company": "Acme"}])

        with patch("accrue.core.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            pipeline.run(df, config)
            assert counter["calls"] == 1

            # Expired
            mock_time.time.return_value = 1002.0
            counter.clear()
            pipeline.run(df, config)
            assert counter["calls"] == 1


class TestCacheKeyChanges:
    def test_cache_key_changes_on_field_spec_change(self, tmp_path):
        """Different field specs on a FunctionStep (via cache_version) cause miss."""
        counter: dict[str, int] = {}
        config = _config_with_cache(tmp_path)
        df = pd.DataFrame([{"company": "Acme"}])

        # cache_version acts as the cache key differentiator for FunctionStep
        p1 = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=_counter_fn(["f1"], counter),
                    fields=["f1"],
                    cache_version="spec_v1",
                ),
            ]
        )
        p1.run(df, config)
        assert counter["calls"] == 1

        counter.clear()
        p2 = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=_counter_fn(["f1"], counter),
                    fields=["f1"],
                    cache_version="spec_v2",
                ),
            ]
        )
        p2.run(df, config)
        assert counter["calls"] == 1  # different version → miss


# ---------------------------------------------------------------------------
# list[dict] input
# ---------------------------------------------------------------------------


class TestListDictInput:
    def test_list_dict_input_returns_list_dict(self, tmp_path):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=lambda ctx: {"f1": "enriched"},
                    fields=["f1"],
                ),
            ]
        )
        config = EnrichmentConfig(enable_progress_bar=False)
        result = pipeline.run([{"company": "Acme"}], config)
        assert isinstance(result.data, list)
        assert len(result.data) == 1

    def test_list_dict_values_correct(self, tmp_path):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=lambda ctx: {"f1": ctx.row["company"].upper()},
                    fields=["f1"],
                ),
            ]
        )
        config = EnrichmentConfig(enable_progress_bar=False)
        result = pipeline.run([{"company": "acme"}, {"company": "beta"}], config)
        assert result.data[0]["f1"] == "ACME"
        assert result.data[1]["f1"] == "BETA"
        # Original data preserved
        assert result.data[0]["company"] == "acme"

    def test_dataframe_input_still_returns_dataframe(self, tmp_path):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=lambda ctx: {"f1": "enriched"},
                    fields=["f1"],
                ),
            ]
        )
        config = EnrichmentConfig(enable_progress_bar=False)
        result = pipeline.run(pd.DataFrame([{"company": "Acme"}]), config)
        assert isinstance(result.data, pd.DataFrame)

    def test_list_dict_with_cache(self, tmp_path):
        counter: dict[str, int] = {}
        pipeline = Pipeline(
            [
                FunctionStep("step_a", fn=_counter_fn(["f1"], counter), fields=["f1"]),
            ]
        )
        config = _config_with_cache(tmp_path)
        data = [{"company": "Acme"}, {"company": "Beta"}]

        result1 = pipeline.run(data, config)
        assert counter["calls"] == 2

        counter.clear()
        result2 = pipeline.run(data, config)
        assert counter.get("calls", 0) == 0
        assert result2.cost.steps["step_a"].cache_hits == 2

    def test_list_dict_internal_fields_filtered(self, tmp_path):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=lambda ctx: {"f1": "val", "__internal": "hidden"},
                    fields=["f1", "__internal"],
                ),
            ]
        )
        config = EnrichmentConfig(enable_progress_bar=False)
        result = pipeline.run([{"company": "Acme"}], config)
        assert "f1" in result.data[0]
        assert "__internal" not in result.data[0]

    def test_list_dict_success_rate(self, tmp_path):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=lambda ctx: {"f1": "val"},
                    fields=["f1"],
                ),
            ]
        )
        config = EnrichmentConfig(enable_progress_bar=False)
        result = pipeline.run([{"company": "Acme"}, {"company": "Beta"}], config)
        assert result.success_rate == 1.0
        assert not result.has_errors


# ---------------------------------------------------------------------------
# Checkpoint interval
# ---------------------------------------------------------------------------


class TestCheckpointInterval:
    def test_checkpoint_interval_fires_callback(self, tmp_path):
        """Verify on_partial_checkpoint is called every N rows."""
        callbacks: list[tuple[str, int]] = []

        def on_partial(step_name, results, completed_count):
            callbacks.append((step_name, completed_count))

        pipeline = Pipeline(
            [
                FunctionStep(
                    "step_a",
                    fn=lambda ctx: {"f1": "val"},
                    fields=["f1"],
                ),
            ]
        )

        config = EnrichmentConfig(
            enable_progress_bar=False,
            checkpoint_interval=2,
        )

        async def run_with_callback():
            rows = [{"company": f"c{i}"} for i in range(6)]
            all_fields = pipeline._collect_field_specs()
            return await pipeline.execute(
                rows=rows,
                all_fields=all_fields,
                config=config,
                on_partial_checkpoint=on_partial,
            )

        asyncio.run(run_with_callback())

        # 6 rows with interval=2 → callbacks at 2, 4, 6
        assert len(callbacks) == 3
        assert all(name == "step_a" for name, _ in callbacks)
        completed_counts = sorted(count for _, count in callbacks)
        assert completed_counts == [2, 4, 6]

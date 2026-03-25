"""Tests for conditional step execution (run_if / skip_if)."""

from __future__ import annotations

import pytest

from accrue.core.exceptions import PipelineError
from accrue.core.hooks import EnrichmentHooks, RowCompleteEvent
from accrue.pipeline.pipeline import Pipeline, _build_skip_values, _should_skip_row
from accrue.schemas.field_spec import FieldSpec
from accrue.steps.base import StepContext, StepResult
from accrue.steps.function import FunctionStep
from accrue.steps.llm import LLMStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockStep:
    """Minimal step for conditional execution testing."""

    def __init__(
        self,
        name: str,
        fields: list[str],
        depends_on: list[str] | None = None,
        run_if=None,
        skip_if=None,
        _field_specs: dict[str, FieldSpec] | None = None,
    ):
        self.name = name
        self.fields = fields
        self.depends_on = depends_on or []
        self.cache = False
        self.run_if = run_if
        self.skip_if = skip_if
        self._field_specs = _field_specs or {}

    async def run(self, ctx: StepContext) -> StepResult:
        return StepResult(values={f: f"val_{f}" for f in self.fields})


# ---------------------------------------------------------------------------
# Validation: mutual exclusivity
# ---------------------------------------------------------------------------


class TestMutualExclusivity:
    def test_function_step_rejects_both(self):
        with pytest.raises(PipelineError, match="mutually exclusive"):
            FunctionStep(
                "s",
                fn=lambda ctx: {},
                fields=["f"],
                run_if=lambda row, prior: True,
                skip_if=lambda row, prior: False,
            )

    def test_llm_step_rejects_both(self):
        with pytest.raises(PipelineError, match="mutually exclusive"):
            LLMStep(
                "s",
                fields=["f"],
                run_if=lambda row, prior: True,
                skip_if=lambda row, prior: False,
            )

    def test_function_step_allows_run_if_only(self):
        step = FunctionStep("s", fn=lambda ctx: {}, fields=["f"], run_if=lambda row, prior: True)
        assert step.run_if is not None
        assert step.skip_if is None

    def test_function_step_allows_skip_if_only(self):
        step = FunctionStep("s", fn=lambda ctx: {}, fields=["f"], skip_if=lambda row, prior: True)
        assert step.skip_if is not None
        assert step.run_if is None

    def test_llm_step_allows_run_if_only(self):
        step = LLMStep("s", fields=["f"], run_if=lambda row, prior: True)
        assert step.run_if is not None
        assert step.skip_if is None

    def test_llm_step_allows_skip_if_only(self):
        step = LLMStep("s", fields=["f"], skip_if=lambda row, prior: True)
        assert step.skip_if is not None
        assert step.run_if is None

    def test_no_predicates_is_default(self):
        step = FunctionStep("s", fn=lambda ctx: {}, fields=["f"])
        assert step.run_if is None
        assert step.skip_if is None


# ---------------------------------------------------------------------------
# _should_skip_row unit tests
# ---------------------------------------------------------------------------


class TestShouldSkipRow:
    @pytest.mark.asyncio
    async def test_no_predicates_never_skips(self):
        step = _MockStep("s", fields=["f"])
        assert await _should_skip_row(step, {}, {}) is False

    @pytest.mark.asyncio
    async def test_run_if_true_does_not_skip(self):
        step = _MockStep("s", fields=["f"], run_if=lambda row, prior: True)
        assert await _should_skip_row(step, {}, {}) is False

    @pytest.mark.asyncio
    async def test_run_if_false_skips(self):
        step = _MockStep("s", fields=["f"], run_if=lambda row, prior: False)
        assert await _should_skip_row(step, {}, {}) is True

    @pytest.mark.asyncio
    async def test_skip_if_true_skips(self):
        step = _MockStep("s", fields=["f"], skip_if=lambda row, prior: True)
        assert await _should_skip_row(step, {}, {}) is True

    @pytest.mark.asyncio
    async def test_skip_if_false_does_not_skip(self):
        step = _MockStep("s", fields=["f"], skip_if=lambda row, prior: False)
        assert await _should_skip_row(step, {}, {}) is False

    @pytest.mark.asyncio
    async def test_run_if_receives_row_data(self):
        step = _MockStep("s", fields=["f"], run_if=lambda row, prior: row.get("country") == "US")
        assert await _should_skip_row(step, {"country": "US"}, {}) is False
        assert await _should_skip_row(step, {"country": "UK"}, {}) is True

    @pytest.mark.asyncio
    async def test_skip_if_receives_prior_results(self):
        step = _MockStep(
            "s",
            fields=["f"],
            skip_if=lambda row, prior: prior.get("label") == "irrelevant",
        )
        assert await _should_skip_row(step, {}, {"label": "irrelevant"}) is True
        assert await _should_skip_row(step, {}, {"label": "relevant"}) is False

    @pytest.mark.asyncio
    async def test_async_run_if(self):
        async def predicate(row, prior):
            return row.get("ok", False)

        step = _MockStep("s", fields=["f"], run_if=predicate)
        assert await _should_skip_row(step, {"ok": True}, {}) is False
        assert await _should_skip_row(step, {"ok": False}, {}) is True

    @pytest.mark.asyncio
    async def test_async_skip_if(self):
        async def predicate(row, prior):
            return row.get("skip", False)

        step = _MockStep("s", fields=["f"], skip_if=predicate)
        assert await _should_skip_row(step, {"skip": True}, {}) is True
        assert await _should_skip_row(step, {"skip": False}, {}) is False


# ---------------------------------------------------------------------------
# _build_skip_values unit tests
# ---------------------------------------------------------------------------


class TestBuildSkipValues:
    def test_no_field_specs_returns_none(self):
        step = _MockStep("s", fields=["a", "b"])
        assert _build_skip_values(step) == {"a": None, "b": None}

    def test_field_spec_defaults_used(self):
        specs = {
            "a": FieldSpec(prompt="p", default="fallback"),
            "b": FieldSpec(prompt="p"),
        }
        step = _MockStep("s", fields=["a", "b"], _field_specs=specs)
        result = _build_skip_values(step)
        assert result == {"a": "fallback", "b": None}

    def test_mixed_fields_with_and_without_specs(self):
        specs = {"a": FieldSpec(prompt="p", default=42)}
        step = _MockStep("s", fields=["a", "b", "c"], _field_specs=specs)
        result = _build_skip_values(step)
        assert result == {"a": 42, "b": None, "c": None}

    def test_default_none_is_used(self):
        """A FieldSpec with explicit default=None should use None (not fall through)."""
        specs = {"a": FieldSpec(prompt="p", default=None)}
        step = _MockStep("s", fields=["a"], _field_specs=specs)
        result = _build_skip_values(step)
        assert result == {"a": None}


# ---------------------------------------------------------------------------
# Integration: run_if execution via FunctionStep + Pipeline
# ---------------------------------------------------------------------------


class TestRunIfExecution:
    @pytest.mark.asyncio
    async def test_run_if_true_executes(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "upper",
                    fn=lambda ctx: {"name": ctx.row["company"].upper()},
                    fields=["name"],
                    run_if=lambda row, prior: True,
                )
            ]
        )
        result = await pipeline.run_async([{"company": "Acme"}])
        assert result.data[0]["name"] == "ACME"

    @pytest.mark.asyncio
    async def test_run_if_false_skips(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "upper",
                    fn=lambda ctx: {"name": ctx.row["company"].upper()},
                    fields=["name"],
                    run_if=lambda row, prior: False,
                )
            ]
        )
        result = await pipeline.run_async([{"company": "Acme"}])
        assert result.data[0]["name"] is None

    @pytest.mark.asyncio
    async def test_run_if_partial_rows(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "upper",
                    fn=lambda ctx: {"name": ctx.row["company"].upper()},
                    fields=["name"],
                    run_if=lambda row, prior: row["country"] == "US",
                )
            ]
        )
        result = await pipeline.run_async(
            [
                {"company": "Acme", "country": "US"},
                {"company": "Beta", "country": "UK"},
                {"company": "Gamma", "country": "US"},
            ]
        )
        assert result.data[0]["name"] == "ACME"
        assert result.data[1]["name"] is None
        assert result.data[2]["name"] == "GAMMA"

    @pytest.mark.asyncio
    async def test_run_if_using_prior_results(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "classify",
                    fn=lambda ctx: {"label": "relevant" if ctx.row["score"] > 5 else "irrelevant"},
                    fields=["label"],
                ),
                FunctionStep(
                    "detail",
                    fn=lambda ctx: {"info": "detailed"},
                    fields=["info"],
                    depends_on=["classify"],
                    run_if=lambda row, prior: prior.get("label") == "relevant",
                ),
            ]
        )
        result = await pipeline.run_async(
            [
                {"score": 10},
                {"score": 1},
            ]
        )
        assert result.data[0]["info"] == "detailed"
        assert result.data[1]["info"] is None


# ---------------------------------------------------------------------------
# Integration: skip_if execution
# ---------------------------------------------------------------------------


class TestSkipIfExecution:
    @pytest.mark.asyncio
    async def test_skip_if_true_skips(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "upper",
                    fn=lambda ctx: {"name": ctx.row["company"].upper()},
                    fields=["name"],
                    skip_if=lambda row, prior: True,
                )
            ]
        )
        result = await pipeline.run_async([{"company": "Acme"}])
        assert result.data[0]["name"] is None

    @pytest.mark.asyncio
    async def test_skip_if_false_executes(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "upper",
                    fn=lambda ctx: {"name": ctx.row["company"].upper()},
                    fields=["name"],
                    skip_if=lambda row, prior: False,
                )
            ]
        )
        result = await pipeline.run_async([{"company": "Acme"}])
        assert result.data[0]["name"] == "ACME"

    @pytest.mark.asyncio
    async def test_skip_if_partial(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "upper",
                    fn=lambda ctx: {"name": ctx.row["company"].upper()},
                    fields=["name"],
                    skip_if=lambda row, prior: row["country"] != "US",
                )
            ]
        )
        result = await pipeline.run_async(
            [
                {"company": "Acme", "country": "US"},
                {"company": "Beta", "country": "UK"},
            ]
        )
        assert result.data[0]["name"] == "ACME"
        assert result.data[1]["name"] is None


# ---------------------------------------------------------------------------
# Skip values: defaults from field specs
# ---------------------------------------------------------------------------


class TestSkipValues:
    @pytest.mark.asyncio
    async def test_skipped_rows_get_none_without_specs(self):
        """FunctionStep with list fields → None for skipped rows."""
        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"a": 1, "b": 2},
                    fields=["a", "b"],
                    run_if=lambda row, prior: False,
                )
            ]
        )
        result = await pipeline.run_async([{"x": 1}])
        assert result.data[0]["a"] is None
        assert result.data[0]["b"] is None

    @pytest.mark.asyncio
    async def test_skipped_rows_use_field_spec_defaults(self):
        """LLMStep with field spec defaults → defaults for skipped rows."""
        # Use _MockStep to simulate LLMStep field specs without API calls
        specs = {
            "market": FieldSpec(prompt="Estimate market", default="Unknown"),
            "score": FieldSpec(prompt="Rate score"),
        }
        step = _MockStep(
            "s",
            fields=["market", "score"],
            _field_specs=specs,
            run_if=lambda row, prior: False,
        )
        pipeline = Pipeline([step])
        result = await pipeline.run_async([{"x": 1}])
        assert result.data[0]["market"] == "Unknown"
        assert result.data[0]["score"] is None


# ---------------------------------------------------------------------------
# Hooks: skipped flag
# ---------------------------------------------------------------------------


class TestHooksSkipped:
    @pytest.mark.asyncio
    async def test_row_complete_skipped_flag_true(self):
        events: list[RowCompleteEvent] = []
        hooks = EnrichmentHooks(on_row_complete=lambda e: events.append(e))

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": 1},
                    fields=["f"],
                    run_if=lambda row, prior: False,
                )
            ]
        )
        await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert len(events) == 1
        assert events[0].skipped is True
        assert events[0].from_cache is False
        assert events[0].values == {"f": None}

    @pytest.mark.asyncio
    async def test_row_complete_skipped_flag_false_for_executed(self):
        events: list[RowCompleteEvent] = []
        hooks = EnrichmentHooks(on_row_complete=lambda e: events.append(e))

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": 1},
                    fields=["f"],
                    run_if=lambda row, prior: True,
                )
            ]
        )
        await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert len(events) == 1
        assert events[0].skipped is False

    @pytest.mark.asyncio
    async def test_mixed_skipped_and_executed_hooks(self):
        events: list[RowCompleteEvent] = []
        hooks = EnrichmentHooks(on_row_complete=lambda e: events.append(e))

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": ctx.row["x"]},
                    fields=["f"],
                    run_if=lambda row, prior: row["x"] > 1,
                )
            ]
        )
        await pipeline.run_async([{"x": 1}, {"x": 2}, {"x": 3}], hooks=hooks)

        assert len(events) == 3
        by_idx = {e.row_index: e for e in events}
        assert by_idx[0].skipped is True
        assert by_idx[1].skipped is False
        assert by_idx[2].skipped is False

    @pytest.mark.asyncio
    async def test_skipped_default_backward_compatible(self):
        """RowCompleteEvent without explicit skipped defaults to False."""
        event = RowCompleteEvent(
            step_name="s",
            row_index=0,
            values={},
            error=None,
            from_cache=False,
        )
        assert event.skipped is False


# ---------------------------------------------------------------------------
# Usage: rows_skipped counter
# ---------------------------------------------------------------------------


class TestUsageRowsSkipped:
    @pytest.mark.asyncio
    async def test_rows_skipped_counted(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": 1},
                    fields=["f"],
                    run_if=lambda row, prior: row["x"] > 1,
                )
            ]
        )
        result = await pipeline.run_async([{"x": 1}, {"x": 2}, {"x": 3}])

        usage = result.cost.steps.get("s")
        assert usage is not None
        assert usage.rows_skipped == 1

    @pytest.mark.asyncio
    async def test_all_skipped_creates_usage(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": 1},
                    fields=["f"],
                    run_if=lambda row, prior: False,
                )
            ]
        )
        result = await pipeline.run_async([{"x": 1}, {"x": 2}])

        usage = result.cost.steps.get("s")
        assert usage is not None
        assert usage.rows_skipped == 2
        assert usage.rows_processed == 0

    @pytest.mark.asyncio
    async def test_no_skipping_zero_rows_skipped(self):
        """A step with no predicates shouldn't report rows_skipped."""
        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": 1},
                    fields=["f"],
                )
            ]
        )
        result = await pipeline.run_async([{"x": 1}])
        # FunctionStep without caching won't produce usage at all
        # (no usage_list, no cache stats, no skips) — that's fine


# ---------------------------------------------------------------------------
# Predicate errors
# ---------------------------------------------------------------------------


class TestPredicateErrors:
    @pytest.mark.asyncio
    async def test_predicate_error_treated_as_row_error(self):
        def bad_predicate(row, prior):
            raise ValueError("predicate boom")

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": 1},
                    fields=["f"],
                    run_if=bad_predicate,
                )
            ]
        )
        result = await pipeline.run_async([{"x": 1}])

        assert result.has_errors
        assert len(result.errors) == 1
        assert "predicate boom" in str(result.errors[0].error)

    @pytest.mark.asyncio
    async def test_async_predicate_error_treated_as_row_error(self):
        async def bad_predicate(row, prior):
            raise RuntimeError("async predicate boom")

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": 1},
                    fields=["f"],
                    skip_if=bad_predicate,
                )
            ]
        )
        result = await pipeline.run_async([{"x": 1}])

        assert result.has_errors
        assert len(result.errors) == 1
        assert "async predicate boom" in str(result.errors[0].error)


# ---------------------------------------------------------------------------
# Async predicates (integration)
# ---------------------------------------------------------------------------


class TestAsyncPredicates:
    @pytest.mark.asyncio
    async def test_async_run_if(self):
        async def is_us(row, prior):
            return row["country"] == "US"

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"name": ctx.row["company"].upper()},
                    fields=["name"],
                    run_if=is_us,
                )
            ]
        )
        result = await pipeline.run_async(
            [
                {"company": "Acme", "country": "US"},
                {"company": "Beta", "country": "UK"},
            ]
        )
        assert result.data[0]["name"] == "ACME"
        assert result.data[1]["name"] is None

    @pytest.mark.asyncio
    async def test_async_skip_if(self):
        async def is_low_priority(row, prior):
            return row.get("priority") == "low"

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"f": "done"},
                    fields=["f"],
                    skip_if=is_low_priority,
                )
            ]
        )
        result = await pipeline.run_async(
            [
                {"priority": "high"},
                {"priority": "low"},
            ]
        )
        assert result.data[0]["f"] == "done"
        assert result.data[1]["f"] is None


# ---------------------------------------------------------------------------
# Multi-step integration: skipped values flow as prior_results
# ---------------------------------------------------------------------------


class TestMultiStepIntegration:
    @pytest.mark.asyncio
    async def test_skipped_values_visible_to_downstream(self):
        """Downstream steps see None/defaults from skipped rows in prior_results."""
        pipeline = Pipeline(
            [
                FunctionStep(
                    "classify",
                    fn=lambda ctx: {"label": "relevant"},
                    fields=["label"],
                    run_if=lambda row, prior: row["score"] > 5,
                ),
                FunctionStep(
                    "detail",
                    fn=lambda ctx: {"info": f"label was {ctx.prior_results.get('label')}"},
                    fields=["info"],
                    depends_on=["classify"],
                ),
            ]
        )
        result = await pipeline.run_async(
            [
                {"score": 10},
                {"score": 1},
            ]
        )
        # Row 0: classify ran → label="relevant" → detail sees it
        assert result.data[0]["info"] == "label was relevant"
        # Row 1: classify skipped → label=None → detail sees None
        assert result.data[1]["info"] == "label was None"

    @pytest.mark.asyncio
    async def test_both_steps_conditional(self):
        pipeline = Pipeline(
            [
                FunctionStep(
                    "step1",
                    fn=lambda ctx: {"a": ctx.row["x"] * 2},
                    fields=["a"],
                    run_if=lambda row, prior: row["x"] > 0,
                ),
                FunctionStep(
                    "step2",
                    fn=lambda ctx: {"b": (ctx.prior_results.get("a") or 0) + 1},
                    fields=["b"],
                    depends_on=["step1"],
                    skip_if=lambda row, prior: prior.get("a") is None,
                ),
            ]
        )
        result = await pipeline.run_async(
            [
                {"x": 5},
                {"x": -1},
            ]
        )
        assert result.data[0]["a"] == 10
        assert result.data[0]["b"] == 11
        assert result.data[1]["a"] is None
        assert result.data[1]["b"] is None

    @pytest.mark.asyncio
    async def test_unconditional_step_unaffected(self):
        """Steps without predicates still work normally alongside conditional steps."""
        pipeline = Pipeline(
            [
                FunctionStep(
                    "always",
                    fn=lambda ctx: {"a": 1},
                    fields=["a"],
                ),
                FunctionStep(
                    "sometimes",
                    fn=lambda ctx: {"b": 2},
                    fields=["b"],
                    depends_on=["always"],
                    run_if=lambda row, prior: row["x"] > 0,
                ),
            ]
        )
        result = await pipeline.run_async(
            [
                {"x": 1},
                {"x": -1},
            ]
        )
        assert result.data[0]["a"] == 1
        assert result.data[0]["b"] == 2
        assert result.data[1]["a"] == 1
        assert result.data[1]["b"] is None


# ---------------------------------------------------------------------------
# Caching: skipped rows don't create cache entries
# ---------------------------------------------------------------------------


class TestCachingWithConditionals:
    @pytest.mark.asyncio
    async def test_skipped_rows_not_cached(self, tmp_path):
        from accrue.core.config import EnrichmentConfig

        call_count = 0

        def counting_fn(ctx):
            nonlocal call_count
            call_count += 1
            return {"f": ctx.row["x"]}

        config = EnrichmentConfig(
            enable_caching=True,
            cache_dir=str(tmp_path),
        )

        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=counting_fn,
                    fields=["f"],
                    run_if=lambda row, prior: row["x"] > 0,
                )
            ]
        )

        # First run
        result1 = await pipeline.run_async(
            [{"x": 1}, {"x": -1}],
            config=config,
        )
        assert call_count == 1  # Only row 0 executed
        assert result1.data[0]["f"] == 1
        assert result1.data[1]["f"] is None

        # Second run — row 0 should be cached, row 1 still skipped
        result2 = await pipeline.run_async(
            [{"x": 1}, {"x": -1}],
            config=config,
        )
        assert call_count == 1  # Still 1 — row 0 served from cache
        assert result2.data[0]["f"] == 1
        assert result2.data[1]["f"] is None

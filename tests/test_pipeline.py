"""Tests for Pipeline — DAG validation and column-oriented execution."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from accrue.core.exceptions import PipelineError
from accrue.pipeline.pipeline import Pipeline
from accrue.steps.base import StepContext, StepResult
from accrue.steps.function import FunctionStep

# -- helpers -------------------------------------------------------------


def _identity_fn(fields: list[str]):
    """Return a function that echoes field names as values."""

    def fn(ctx: StepContext) -> dict[str, Any]:
        return {f: f"{f}_value" for f in fields}

    return fn


def _fn_from_row(field: str, row_key: str):
    """Return a function that reads a row key and writes a field."""

    def fn(ctx: StepContext) -> dict[str, Any]:
        return {field: ctx.row.get(row_key, "")}

    return fn


def _fn_from_prior(field: str, prior_key: str, transform=None):
    """Return a function that reads from prior_results."""

    def fn(ctx: StepContext) -> dict[str, Any]:
        val = ctx.prior_results.get(prior_key, "")
        if transform:
            val = transform(val)
        return {field: val}

    return fn


# -- Construction & DAG validation ----------------------------------------


class TestPipelineConstruction:
    def test_single_step(self):
        p = Pipeline([FunctionStep("a", fn=lambda ctx: {}, fields=["f1"])])
        assert p.step_names == ["a"]
        assert p.execution_levels == [["a"]]

    def test_two_independent_steps(self):
        p = Pipeline(
            [
                FunctionStep("a", fn=lambda ctx: {}, fields=["f1"]),
                FunctionStep("b", fn=lambda ctx: {}, fields=["f2"]),
            ]
        )
        assert p.execution_levels == [["a", "b"]]

    def test_linear_chain(self):
        p = Pipeline(
            [
                FunctionStep("a", fn=lambda ctx: {}, fields=["f1"]),
                FunctionStep("b", fn=lambda ctx: {}, fields=["f2"], depends_on=["a"]),
                FunctionStep("c", fn=lambda ctx: {}, fields=["f3"], depends_on=["b"]),
            ]
        )
        assert p.execution_levels == [["a"], ["b"], ["c"]]

    def test_diamond_dependency(self):
        p = Pipeline(
            [
                FunctionStep("a", fn=lambda ctx: {}, fields=["f1"]),
                FunctionStep("b", fn=lambda ctx: {}, fields=["f2"], depends_on=["a"]),
                FunctionStep("c", fn=lambda ctx: {}, fields=["f3"], depends_on=["a"]),
                FunctionStep("d", fn=lambda ctx: {}, fields=["f4"], depends_on=["b", "c"]),
            ]
        )
        assert p.execution_levels == [["a"], ["b", "c"], ["d"]]

    def test_duplicate_name_raises(self):
        with pytest.raises(PipelineError, match="Duplicate step names"):
            Pipeline(
                [
                    FunctionStep("dup", fn=lambda ctx: {}, fields=["f1"]),
                    FunctionStep("dup", fn=lambda ctx: {}, fields=["f2"]),
                ]
            )

    def test_missing_dependency_raises(self):
        with pytest.raises(PipelineError, match="unknown step 'missing'"):
            Pipeline(
                [
                    FunctionStep("a", fn=lambda ctx: {}, fields=["f1"], depends_on=["missing"]),
                ]
            )

    def test_cycle_two_steps(self):
        s1 = FunctionStep("a", fn=lambda ctx: {}, fields=["f1"], depends_on=["b"])
        s2 = FunctionStep("b", fn=lambda ctx: {}, fields=["f2"], depends_on=["a"])
        with pytest.raises(PipelineError, match="Cycle detected"):
            Pipeline([s1, s2])

    def test_cycle_three_steps(self):
        s1 = FunctionStep("a", fn=lambda ctx: {}, fields=["f1"], depends_on=["c"])
        s2 = FunctionStep("b", fn=lambda ctx: {}, fields=["f2"], depends_on=["a"])
        s3 = FunctionStep("c", fn=lambda ctx: {}, fields=["f3"], depends_on=["b"])
        with pytest.raises(PipelineError, match="Cycle detected"):
            Pipeline([s1, s2, s3])


# -- Execution -----------------------------------------------------------


class TestPipelineExecution:
    @pytest.mark.asyncio
    async def test_single_step_execution(self):
        p = Pipeline(
            [
                FunctionStep("a", fn=_identity_fn(["f1"]), fields=["f1"]),
            ]
        )
        rows = [{"company": "Acme"}, {"company": "Beta"}]
        results, errors, cost = await p.execute(rows, all_fields={"f1": {"prompt": "test"}})

        assert len(results) == 2
        assert results[0] == {"f1": "f1_value"}
        assert results[1] == {"f1": "f1_value"}
        assert errors == []

    @pytest.mark.asyncio
    async def test_two_independent_steps(self):
        p = Pipeline(
            [
                FunctionStep("a", fn=_identity_fn(["f1"]), fields=["f1"]),
                FunctionStep("b", fn=_identity_fn(["f2"]), fields=["f2"]),
            ]
        )
        results, errors, cost = await p.execute([{"x": 1}], all_fields={"f1": {}, "f2": {}})

        assert results[0] == {"f1": "f1_value", "f2": "f2_value"}
        assert errors == []

    @pytest.mark.asyncio
    async def test_dependency_routing(self):
        """Step B should see outputs from Step A in prior_results."""

        def step_a_fn(ctx):
            return {"intermediate": ctx.row.get("input", "") + "_processed"}

        def step_b_fn(ctx):
            return {"final": ctx.prior_results.get("intermediate", "") + "_done"}

        p = Pipeline(
            [
                FunctionStep("a", fn=step_a_fn, fields=["intermediate"]),
                FunctionStep("b", fn=step_b_fn, fields=["final"], depends_on=["a"]),
            ]
        )

        rows = [{"input": "hello"}, {"input": "world"}]
        results, errors, cost = await p.execute(rows, all_fields={})

        assert results[0]["final"] == "hello_processed_done"
        assert results[1]["final"] == "world_processed_done"
        assert errors == []

    @pytest.mark.asyncio
    async def test_diamond_dependency_routing(self):
        """Diamond: A -> B, A -> C, B+C -> D."""

        p = Pipeline(
            [
                FunctionStep("a", fn=lambda ctx: {"a_out": 1}, fields=["a_out"]),
                FunctionStep(
                    "b",
                    fn=lambda ctx: {"b_out": ctx.prior_results["a_out"] + 10},
                    fields=["b_out"],
                    depends_on=["a"],
                ),
                FunctionStep(
                    "c",
                    fn=lambda ctx: {"c_out": ctx.prior_results["a_out"] + 100},
                    fields=["c_out"],
                    depends_on=["a"],
                ),
                FunctionStep(
                    "d",
                    fn=lambda ctx: {
                        "d_out": ctx.prior_results["b_out"] + ctx.prior_results["c_out"]
                    },
                    fields=["d_out"],
                    depends_on=["b", "c"],
                ),
            ]
        )

        results, errors, cost = await p.execute([{"x": 0}], all_fields={})
        assert results[0] == {"a_out": 1, "b_out": 11, "c_out": 101, "d_out": 112}
        assert errors == []

    @pytest.mark.asyncio
    async def test_empty_rows(self):
        p = Pipeline([FunctionStep("a", fn=lambda ctx: {"f": 1}, fields=["f"])])
        results, errors, cost = await p.execute([], all_fields={})
        assert results == []
        assert errors == []

    @pytest.mark.asyncio
    async def test_config_max_workers(self):
        """Verify semaphore uses config.max_workers."""
        from accrue.core.config import EnrichmentConfig

        call_count = 0

        async def slow_fn(ctx):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return {"f": call_count}

        p = Pipeline([FunctionStep("a", fn=slow_fn, fields=["f"])])
        config = EnrichmentConfig(max_workers=2, on_error="continue")
        results, errors, cost = await p.execute(
            [{"x": i} for i in range(5)],
            all_fields={},
            config=config,
        )
        assert len(results) == 5
        assert errors == []

    @pytest.mark.asyncio
    async def test_internal_fields_passed_between_steps(self):
        """Fields prefixed with __ are internal inter-step fields."""
        p = Pipeline(
            [
                FunctionStep(
                    "search",
                    fn=lambda ctx: {"__web_ctx": "search data"},
                    fields=["__web_ctx"],
                ),
                FunctionStep(
                    "analyze",
                    fn=lambda ctx: {
                        "summary": f"Based on: {ctx.prior_results.get('__web_ctx', '')}"
                    },
                    fields=["summary"],
                    depends_on=["search"],
                ),
            ]
        )

        results, errors, cost = await p.execute([{"q": "test"}], all_fields={})
        assert results[0]["summary"] == "Based on: search data"
        # Internal fields are still in results — Enricher filters them later
        assert results[0]["__web_ctx"] == "search data"
        assert errors == []


# -- cost aggregation ----------------------------------------------------


class TestCostAggregation:
    @pytest.mark.asyncio
    async def test_function_step_has_no_cost(self):
        p = Pipeline(
            [
                FunctionStep("a", fn=_identity_fn(["f1"]), fields=["f1"]),
            ]
        )
        results, errors, cost = await p.execute([{"x": 1}], all_fields={})
        assert cost.total_tokens == 0
        assert cost.steps == {}

    @pytest.mark.asyncio
    async def test_cost_from_step_with_usage(self):
        """Step that returns usage info gets aggregated."""
        from accrue.schemas.base import UsageInfo

        class UsageStep:
            name = "llm_mock"
            fields = ["f1"]
            depends_on = []

            async def run(self, ctx):
                return StepResult(
                    values={"f1": "val"},
                    usage=UsageInfo(
                        prompt_tokens=100,
                        completion_tokens=50,
                        total_tokens=150,
                        model="test-model",
                    ),
                )

        p = Pipeline([UsageStep()])
        results, errors, cost = await p.execute(
            [{"x": 1}, {"x": 2}],
            all_fields={},
        )
        assert cost.total_prompt_tokens == 200
        assert cost.total_completion_tokens == 100
        assert cost.total_tokens == 300
        assert "llm_mock" in cost.steps
        assert cost.steps["llm_mock"].rows_processed == 2
        assert cost.steps["llm_mock"].model == "test-model"


# -- get_step ------------------------------------------------------------


class TestPipelineHelpers:
    def test_get_step(self):
        step = FunctionStep("a", fn=lambda ctx: {}, fields=["f1"])
        p = Pipeline([step])
        assert p.get_step("a") is step

    def test_get_step_missing(self):
        p = Pipeline([FunctionStep("a", fn=lambda ctx: {}, fields=["f1"])])
        with pytest.raises(KeyError):
            p.get_step("missing")

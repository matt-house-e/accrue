"""Tests for Pipeline — DAG validation and column-oriented execution."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest

from accrue.core.exceptions import PipelineError
from accrue.pipeline.pipeline import Pipeline, _merge_results_into_df
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


# -- _merge_results_into_df helper -------------------------------------------


class TestMergeResultsIntoDf:
    def test_untouched_columns_preserve_dtype(self):
        """Columns not in accumulated results keep their original dtype."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        accumulated = [{"score": 0.1}, {"score": 0.2}, {"score": 0.3}]
        df_out = _merge_results_into_df(df, accumulated, overwrite_fields=True)
        assert df_out["id"].dtype == df["id"].dtype
        assert df_out["name"].dtype == df["name"].dtype

    def test_new_columns_aligned_to_original_index(self):
        """Result columns align to the original non-default index."""
        df = pd.DataFrame({"val": ["x", "y", "z"]}, index=[10, 20, 30])
        accumulated = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]
        df_out = _merge_results_into_df(df, accumulated, overwrite_fields=True)
        assert list(df_out.index) == [10, 20, 30]
        assert list(df_out["tag"]) == ["a", "b", "c"]

    def test_overwrite_false_keeps_existing_non_null(self):
        """overwrite_fields=False: existing non-null value wins."""
        df = pd.DataFrame({"f": ["existing", "also_existing"]})
        accumulated = [{"f": "new"}, {"f": "new"}]
        df_out = _merge_results_into_df(df, accumulated, overwrite_fields=False)
        assert list(df_out["f"]) == ["existing", "also_existing"]

    def test_overwrite_false_fills_nan(self):
        """overwrite_fields=False: NaN and empty string are replaced by new value."""
        df = pd.DataFrame({"f": [None, "", "kept"]})
        accumulated = [{"f": "filled_nan"}, {"f": "filled_empty"}, {"f": "new"}]
        df_out = _merge_results_into_df(df, accumulated, overwrite_fields=False)
        assert df_out["f"].iloc[0] == "filled_nan"
        assert df_out["f"].iloc[1] == "filled_empty"
        assert df_out["f"].iloc[2] == "kept"

    def test_overwrite_true_always_overwrites(self):
        """overwrite_fields=True: new value wins even when existing is non-null."""
        df = pd.DataFrame({"f": ["old", "also_old"]})
        accumulated = [{"f": "new1"}, {"f": "new2"}]
        df_out = _merge_results_into_df(df, accumulated, overwrite_fields=True)
        assert list(df_out["f"]) == ["new1", "new2"]

    def test_internal_fields_filtered(self):
        """Keys starting with __ are not written to the output DataFrame."""
        df = pd.DataFrame({"x": [1]})
        accumulated = [{"__web_context": "secret", "summary": "kept"}]
        df_out = _merge_results_into_df(df, accumulated, overwrite_fields=True)
        assert "summary" in df_out.columns
        assert "__web_context" not in df_out.columns

    def test_duplicate_column_names_raises(self):
        """Input DataFrame with duplicate column names raises ValueError."""
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        accumulated = [{"b": "val"}]
        with pytest.raises(ValueError, match="duplicate column names"):
            _merge_results_into_df(df, accumulated, overwrite_fields=True)


# -- NaN → None boundary conversion -----------------------------------------


class TestNanToNoneBoundary:
    """DataFrame → row dict conversion must replace NaN/NaT/pd.NA with None."""

    def _make_pipeline(self, captured: list):
        """Return a pipeline that records ctx.row['x'] into *captured* by input index.

        Uses an ``idx`` sentinel column on the input DataFrame so callers can
        capture values by their original row index rather than completion order
        (the streaming worker pool finishes rows non-deterministically).
        """

        def fn(ctx: StepContext) -> dict[str, Any]:
            captured[ctx.row["idx"]] = ctx.row["x"]
            return {}

        return Pipeline([FunctionStep("capture", fn=fn, fields=[])])

    @pytest.mark.asyncio
    async def test_float_nan_becomes_none(self):
        """float('nan') in a DataFrame column is converted to None before step.run."""
        captured: list = [object()] * 3
        p = self._make_pipeline(captured)
        df = pd.DataFrame({"idx": [0, 1, 2], "x": [1.0, float("nan"), 3.0]})
        await p.run_async(df)
        assert captured[1] is None, f"expected None, got {captured[1]!r}"

    @pytest.mark.asyncio
    async def test_pd_na_becomes_none(self):
        """pd.NA in a DataFrame column is converted to None before step.run."""
        captured: list = [object()] * 3
        p = self._make_pipeline(captured)
        df = pd.DataFrame({"idx": [0, 1, 2], "x": pd.array([1, pd.NA, 3], dtype="Int64")})
        await p.run_async(df)
        assert captured[1] is None, f"expected None, got {captured[1]!r}"

    @pytest.mark.asyncio
    async def test_nat_becomes_none(self):
        """pd.NaT in a datetime column is converted to None before step.run."""
        captured: list = [object()] * 3
        p = self._make_pipeline(captured)
        df = pd.DataFrame(
            {
                "idx": [0, 1, 2],
                "x": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            }
        )
        await p.run_async(df)
        assert captured[1] is None, f"expected None, got {captured[1]!r}"

    @pytest.mark.asyncio
    async def test_none_roundtrips_as_none(self):
        """An explicit Python None in input stays None after conversion."""
        captured: list = [object()] * 3
        p = self._make_pipeline(captured)
        df = pd.DataFrame({"idx": [0, 1, 2], "x": [1, None, 3]})
        await p.run_async(df)
        assert captured[1] is None, f"expected None, got {captured[1]!r}"

    @pytest.mark.asyncio
    async def test_run_if_predicate_is_falsy_for_nan_was_none(self):
        """run_if lambda receives None (not nan) so the predicate evaluates falsy."""
        skipped: list = []

        def fn(ctx: StepContext) -> dict[str, Any]:
            skipped.append(ctx.row["x"])
            return {}

        p = Pipeline(
            [
                FunctionStep(
                    "conditional",
                    fn=fn,
                    fields=[],
                    run_if=lambda row, _prior: row.get("x"),
                )
            ]
        )
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        await p.run_async(df)
        # Middle row should be skipped — None is falsy, nan is truthy.
        # Compare as a set since completion order is non-deterministic under
        # the streaming worker pool.
        assert len(skipped) == 2
        assert set(skipped) == {1.0, 3.0}

    @pytest.mark.asyncio
    async def test_non_null_dataframe_regression(self):
        """Existing behaviour with fully-populated DataFrame is unchanged."""
        captured: list = [None] * 3
        p = self._make_pipeline(captured)
        df = pd.DataFrame({"idx": [0, 1, 2], "x": [10, 20, 30]})
        await p.run_async(df)
        assert captured == [10, 20, 30]


# -- Worker-pool bounded task count ----------------------------------------


class TestWorkerPoolBoundedTaskCount:
    """Streaming worker pool: in-flight asyncio Task count stays bounded.

    Before this refactor, all N row tasks were created up-front, so
    asyncio.all_tasks() would show ~N tasks.  The new design uses a fixed
    pool of max_workers workers, so in-flight tasks must stay bounded
    regardless of row count.
    """

    @pytest.mark.asyncio
    async def test_inflight_tasks_bounded_by_max_workers(self):
        """1000 rows with max_workers=4 never spawns more than ~max_workers+overhead tasks."""
        num_rows = 1000
        max_workers = 4
        # Overhead: workers + producer + pipeline-level tasks + test task itself.
        # We use a generous ceiling (max_workers * 3) to avoid false positives.
        task_ceiling = max_workers * 3

        peak_task_count = 0

        async def counting_fn(ctx: StepContext) -> dict[str, Any]:
            nonlocal peak_task_count
            current = len(asyncio.all_tasks())
            if current > peak_task_count:
                peak_task_count = current
            # yield to the event loop so other tasks can be observed
            await asyncio.sleep(0)
            return {"v": 1}

        from accrue.core.config import EnrichmentConfig

        step = FunctionStep("count", fn=counting_fn, fields=["v"])
        p = Pipeline([step])
        rows = [{"i": i} for i in range(num_rows)]
        config = EnrichmentConfig(max_workers=max_workers)

        results, errors, cost = await p.execute(rows, all_fields={}, config=config)

        assert errors == [], f"Unexpected errors: {errors}"
        assert len(results) == num_rows
        assert all(r.get("v") == 1 for r in results), "Some rows produced wrong output"

        # Core assertion: in-flight task count stayed bounded
        assert peak_task_count <= task_ceiling, (
            f"Peak task count {peak_task_count} exceeded ceiling {task_ceiling}. "
            f"The eager-creation regression may have been reintroduced."
        )

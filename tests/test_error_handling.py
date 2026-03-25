"""Tests for per-row error handling — one row failure doesn't kill the pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from accrue.core.config import EnrichmentConfig
from accrue.core.exceptions import RowError, StepError
from accrue.pipeline.pipeline import Pipeline
from accrue.steps.function import FunctionStep

# -- helpers -----------------------------------------------------------------


def _failing_step(name: str, fields: list[str], fail_indices: set[int], **kwargs) -> FunctionStep:
    """Step that raises StepError for specific row indices."""

    def fn(ctx):
        idx = ctx.row.get("__idx")
        if idx in fail_indices:
            raise StepError(f"Row {idx} failed", step_name=name)
        return {f: f"{f}_value_{idx}" for f in fields}

    return FunctionStep(name=name, fn=fn, fields=fields, **kwargs)


# -- per-row error collection ------------------------------------------------


class TestPerRowErrors:
    @pytest.mark.asyncio
    async def test_single_row_failure_preserves_others(self):
        """One row fails, others succeed — no crash."""
        p = Pipeline([_failing_step("s", ["f"], fail_indices={1})])
        rows = [{"__idx": 0}, {"__idx": 1}, {"__idx": 2}]

        results, errors, cost = await p.execute(rows, all_fields={})

        assert len(results) == 3
        assert results[0]["f"] == "f_value_0"
        assert results[1]["f"] is None  # sentinel for failed row
        assert results[2]["f"] == "f_value_2"
        assert len(errors) == 1
        assert errors[0].row_index == 1
        assert errors[0].step_name == "s"

    @pytest.mark.asyncio
    async def test_multiple_row_failures(self):
        p = Pipeline([_failing_step("s", ["f"], fail_indices={0, 2, 4})])
        rows = [{"__idx": i} for i in range(5)]

        results, errors, cost = await p.execute(rows, all_fields={})

        assert len(errors) == 3
        failed_indices = {e.row_index for e in errors}
        assert failed_indices == {0, 2, 4}

        # Successful rows still have values
        assert results[1]["f"] == "f_value_1"
        assert results[3]["f"] == "f_value_3"

    @pytest.mark.asyncio
    async def test_all_rows_fail(self):
        p = Pipeline([_failing_step("s", ["f"], fail_indices={0, 1, 2})])
        rows = [{"__idx": i} for i in range(3)]

        results, errors, cost = await p.execute(rows, all_fields={})

        assert len(errors) == 3
        assert all(r["f"] is None for r in results)

    @pytest.mark.asyncio
    async def test_no_errors_returns_empty_list(self):
        p = Pipeline([_failing_step("s", ["f"], fail_indices=set())])
        rows = [{"__idx": i} for i in range(3)]

        results, errors, cost = await p.execute(rows, all_fields={})

        assert errors == []
        assert all(r["f"] is not None for r in results)


# -- on_error="raise" mode --------------------------------------------------


class TestOnErrorRaise:
    @pytest.mark.asyncio
    async def test_raise_mode_raises_on_first_failure(self):
        p = Pipeline([_failing_step("s", ["f"], fail_indices={1})])
        rows = [{"__idx": 0}, {"__idx": 1}, {"__idx": 2}]
        config = EnrichmentConfig(max_workers=1, on_error="raise")

        with pytest.raises(StepError, match="Row 1 failed"):
            await p.execute(rows, all_fields={}, config=config)


# -- error in multi-step pipeline --------------------------------------------


class TestMultiStepErrors:
    @pytest.mark.asyncio
    async def test_error_in_first_step_sentinels_propagate(self):
        """Row fails in step 1 → step 2 sees None sentinel in prior_results."""

        def step_b_fn(ctx):
            prior_val = ctx.prior_results.get("f1")
            return {"f2": f"got:{prior_val}"}

        p = Pipeline(
            [
                _failing_step("a", ["f1"], fail_indices={1}),
                FunctionStep("b", fn=step_b_fn, fields=["f2"], depends_on=["a"]),
            ]
        )
        rows = [{"__idx": 0}, {"__idx": 1}]

        results, errors, cost = await p.execute(rows, all_fields={})

        # Row 0: step a succeeded, step b sees the value
        assert results[0]["f1"] == "f1_value_0"
        assert results[0]["f2"] == "got:f1_value_0"

        # Row 1: step a failed (sentinel None), step b still runs and sees None
        assert results[1]["f1"] is None
        assert results[1]["f2"] == "got:None"

        # Only 1 error (from step a, row 1)
        assert len(errors) == 1
        assert errors[0].step_name == "a"


# -- RowError dataclass ------------------------------------------------------


class TestRowError:
    def test_str_representation(self):
        err = RowError(row_index=5, step_name="analyze", error=ValueError("bad value"))
        s = str(err)
        assert "row=5" in s
        assert "analyze" in s
        assert "ValueError" in s

    def test_error_type_auto_set(self):
        err = RowError(row_index=0, step_name="s", error=StepError("fail"))
        assert err.error_type == "StepError"


# -- Enricher integration with errors ----------------------------------------


class TestEnricherWithErrors:
    def test_enricher_returns_partial_results(self):
        """Enricher still returns a DataFrame when some rows fail."""
        from accrue.core.enricher import Enricher

        p = Pipeline([_failing_step("s", ["f"], fail_indices={1})])

        enricher = Enricher(p)
        df = pd.DataFrame({"__idx": [0, 1, 2]})
        result = enricher.run(df)

        assert "f" in result.columns
        assert result.at[0, "f"] == "f_value_0"
        assert result.at[1, "f"] is None  # failed row
        assert result.at[2, "f"] == "f_value_2"

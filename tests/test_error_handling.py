"""Tests for per-row error handling — one row failure doesn't kill the pipeline."""

from __future__ import annotations

import asyncio

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

        results, errors, cost, _ = await p.execute(rows, all_fields={})

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

        results, errors, cost, _ = await p.execute(rows, all_fields={})

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

        results, errors, cost, _ = await p.execute(rows, all_fields={})

        assert len(errors) == 3
        assert all(r["f"] is None for r in results)

    @pytest.mark.asyncio
    async def test_no_errors_returns_empty_list(self):
        p = Pipeline([_failing_step("s", ["f"], fail_indices=set())])
        rows = [{"__idx": i} for i in range(3)]

        results, errors, cost, _ = await p.execute(rows, all_fields={})

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

        results, errors, cost, _ = await p.execute(rows, all_fields={})

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

    # -- secret sanitization --------------------------------------------------

    def test_api_key_sk_redacted(self):
        """sk-... API key in exception message is redacted from __str__."""
        key = "sk-abc123def456ghi789jklmnop"
        err = RowError(row_index=0, step_name="s", error=ValueError(f"auth failed: {key}"))
        s = str(err)
        assert "***REDACTED***" in s
        assert key not in s

    def test_bearer_token_redacted(self):
        """Bearer token in exception message is redacted from __str__."""
        err = RowError(
            row_index=1,
            step_name="s",
            error=ValueError("401: Authorization: Bearer eyJh.abc.def"),
        )
        s = str(err)
        assert "***REDACTED***" in s
        assert "eyJh.abc.def" not in s

    def test_api_key_param_redacted(self):
        """api_key=... in exception message is redacted from __str__."""
        err = RowError(
            row_index=2,
            step_name="s",
            error=ValueError("api_key=sk-xyz failed"),
        )
        s = str(err)
        assert "***REDACTED***" in s
        assert "sk-xyz" not in s

    def test_plain_error_unchanged(self):
        """Plain English error messages pass through unsanitized."""
        msg = "missing required field 'company'"
        err = RowError(row_index=3, step_name="s", error=ValueError(msg))
        assert msg in str(err)


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
        assert pd.isna(result.at[1, "f"])  # failed row
        assert result.at[2, "f"] == "f_value_2"


# -- Cancellation / drain correctness ----------------------------------------


class TestCancellationFlushesPartialState:
    @pytest.mark.asyncio
    async def test_cancellation_flushes_step_values_and_calls_checkpoint(self):
        """CancelledError mid-step persists accumulated results and invokes on_partial_checkpoint.

        Strategy: rows sleep for increasing durations so cancellation lands when
        some are done and others are still in-flight, guaranteeing completed_count > 0.
        """
        partial_calls: list[tuple[str, int]] = []

        def on_partial(step_name: str, results: list, count: int) -> None:
            partial_calls.append((step_name, count))

        # Rows sleep idx * 0.05s — row 0 is instant, row 9 sleeps 0.45s.
        # With max_workers=10 all start at once; rows 0-2 complete by ~0.10s.
        async def staggered_fn(ctx):
            idx = ctx.row.get("__idx", 0)
            await asyncio.sleep(idx * 0.05)
            return {"f": idx}

        step = FunctionStep("slow", fn=staggered_fn, fields=["f"])
        p = Pipeline([step])
        rows = [{"__idx": i} for i in range(10)]
        # checkpoint_interval=2: callback fires after rows 2, 4, … complete
        config = EnrichmentConfig(checkpoint_interval=2, max_workers=10)

        async def run():
            return await p.execute(
                rows,
                all_fields={},
                config=config,
                on_partial_checkpoint=on_partial,
            )

        task = asyncio.create_task(run())
        # Rows 0-2 complete in ~0.0–0.10s; cancel after they're done but before 9 finishes
        await asyncio.sleep(0.12)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # At least one checkpoint call happened (rows 0-2 done → completed_count hit 2)
        assert len(partial_calls) >= 1, "on_partial_checkpoint was never called after cancellation"
        assert all(name == "slow" for name, _ in partial_calls)


class TestLevelGatherSiblingsAwaited:
    @pytest.mark.asyncio
    async def test_on_error_raise_does_not_leave_orphaned_tasks(self):
        """Parallel steps at the same level: one fails, siblings are awaited not dropped."""
        finished_b: list[bool] = []

        async def fast_fail(ctx):
            raise StepError("deliberate", step_name="step_a")

        async def slow_b(ctx):
            await asyncio.sleep(0.05)
            finished_b.append(True)
            return {"b": 1}

        p = Pipeline(
            [
                FunctionStep("step_a", fn=fast_fail, fields=["a"]),
                FunctionStep("step_b", fn=slow_b, fields=["b"]),
            ]
        )
        rows = [{"x": 1}]
        config = EnrichmentConfig(on_error="raise", max_workers=2)

        with pytest.raises(StepError, match="deliberate"):
            await p.execute(rows, all_fields={}, config=config)

        # After the raise, all tasks created inside _execute_step must be done
        remaining = [t for t in asyncio.all_tasks() if not t.done()]
        step_tasks = [t for t in remaining if "tracked_row" in repr(t) or "process_row" in repr(t)]
        assert step_tasks == [], f"Orphaned step tasks found: {step_tasks}"


class TestSiblingExceptionLogging:
    @pytest.mark.asyncio
    async def test_sibling_exceptions_both_logged(self, caplog):
        """Two parallel steps both raise; second exception is logged at WARNING."""
        import logging

        async def fail_a(ctx):
            raise ValueError("step_a blew up")

        async def fail_b(ctx):
            raise RuntimeError("step_b blew up")

        p = Pipeline(
            [
                FunctionStep("step_a", fn=fail_a, fields=["a"]),
                FunctionStep("step_b", fn=fail_b, fields=["b"]),
            ]
        )
        rows = [{"x": 1}]
        config = EnrichmentConfig(on_error="raise", max_workers=2)

        with caplog.at_level(logging.WARNING, logger="accrue.pipeline.pipeline"):
            with pytest.raises((ValueError, RuntimeError)):
                await p.execute(rows, all_fields={}, config=config)

        # One of the two exceptions is logged as a sibling warning
        sibling_warnings = [
            r for r in caplog.records if "Sibling step in same level also raised" in r.message
        ]
        assert len(sibling_warnings) == 1
        # The warning mentions the suppressed exception type
        msg = sibling_warnings[0].message
        assert "ValueError" in msg or "RuntimeError" in msg

    @pytest.mark.asyncio
    async def test_single_failure_no_sibling_warning(self, caplog):
        """Single step failure: no sibling-suppressed warning emitted."""
        import logging

        p = Pipeline([_failing_step("s", ["f"], fail_indices={0})])
        rows = [{"__idx": 0}]
        config = EnrichmentConfig(on_error="raise", max_workers=1)

        with caplog.at_level(logging.WARNING, logger="accrue.pipeline.pipeline"):
            with pytest.raises(Exception):
                await p.execute(rows, all_fields={}, config=config)

        sibling_warnings = [
            r for r in caplog.records if "Sibling step in same level also raised" in r.message
        ]
        assert sibling_warnings == []


class TestOnErrorContinueRegression:
    @pytest.mark.asyncio
    async def test_continue_mode_collects_errors_and_succeeds(self):
        """on_error='continue' keeps going; success_rate reflects failures."""
        p = Pipeline([_failing_step("s", ["f"], fail_indices={1, 3})])
        rows = [{"__idx": i} for i in range(5)]
        config = EnrichmentConfig(on_error="continue")

        results, errors, cost, _ = await p.execute(rows, all_fields={}, config=config)

        assert len(errors) == 2
        failed = {e.row_index for e in errors}
        assert failed == {1, 3}
        # Successful rows still have values
        assert results[0]["f"] == "f_value_0"
        assert results[2]["f"] == "f_value_2"
        assert results[4]["f"] == "f_value_4"
        # Failed rows get None sentinel
        assert results[1]["f"] is None
        assert results[3]["f"] is None

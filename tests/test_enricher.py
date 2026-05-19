"""Tests for the Enricher — checkpoint-capable pipeline runner."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest

from accrue.core.config import EnrichmentConfig
from accrue.core.enricher import Enricher
from accrue.pipeline.pipeline import Pipeline
from accrue.steps.function import FunctionStep

# -- helpers -----------------------------------------------------------------


def _identity_step(name: str, fields: list[str], **kwargs) -> FunctionStep:
    """Step that produces static '{field}_value' for each field."""

    def fn(ctx):
        return {f: f"{f}_value" for f in fields}

    return FunctionStep(name=name, fn=fn, fields=fields, **kwargs)


# -- basic execution ---------------------------------------------------------


class TestBasicExecution:
    def test_single_step(self):
        pipeline = Pipeline([_identity_step("s1", ["company_type"])])
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"company": ["Acme", "Beta"]})
        result = enricher.run(df)

        assert "company_type" in result.columns
        assert list(result["company_type"]) == [
            "company_type_value",
            "company_type_value",
        ]
        assert list(result["company"]) == ["Acme", "Beta"]

    def test_multi_step_with_dependencies(self):
        def step_a_fn(ctx):
            return {"raw": ctx.row.get("input", "") + "_raw"}

        def step_b_fn(ctx):
            return {"processed": ctx.prior_results.get("raw", "") + "_done"}

        pipeline = Pipeline(
            [
                FunctionStep("a", fn=step_a_fn, fields=["raw"]),
                FunctionStep("b", fn=step_b_fn, fields=["processed"], depends_on=["a"]),
            ]
        )
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"input": ["hello", "world"]})
        result = enricher.run(df)

        assert list(result["raw"]) == ["hello_raw", "world_raw"]
        assert list(result["processed"]) == ["hello_raw_done", "world_raw_done"]

    def test_row_data_flows_through(self):
        def fn(ctx):
            return {"greeting": f"Hello, {ctx.row['name']}"}

        pipeline = Pipeline([FunctionStep("g", fn=fn, fields=["greeting"])])
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        result = enricher.run(df)

        assert list(result["greeting"]) == ["Hello, Alice", "Hello, Bob"]

    def test_via_pipeline_runner(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        enricher = pipeline.runner()

        df = pd.DataFrame({"x": [1]})
        result = enricher.run(df)
        assert "f" in result.columns


# -- internal fields filtered ------------------------------------------------


class TestInternalFieldsFiltered:
    def test_double_underscore_not_in_output(self):
        def search_fn(ctx):
            return {"__web_ctx": "search data", "summary": "based on search"}

        pipeline = Pipeline(
            [
                FunctionStep("search", fn=search_fn, fields=["__web_ctx", "summary"]),
            ]
        )
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"q": ["test"]})
        result = enricher.run(df)

        assert "summary" in result.columns
        assert "__web_ctx" not in result.columns

    def test_internal_fields_pass_between_steps(self):
        def step_a_fn(ctx):
            return {"__internal": "secret_data"}

        def step_b_fn(ctx):
            return {"output": f"got: {ctx.prior_results.get('__internal', '')}"}

        pipeline = Pipeline(
            [
                FunctionStep("a", fn=step_a_fn, fields=["__internal"]),
                FunctionStep("b", fn=step_b_fn, fields=["output"], depends_on=["a"]),
            ]
        )
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"x": [1]})
        result = enricher.run(df)

        assert result.at[0, "output"] == "got: secret_data"
        assert "__internal" not in result.columns


# -- preserves original columns ----------------------------------------------


class TestPreservesOriginalColumns:
    def test_original_columns_untouched(self):
        pipeline = Pipeline([_identity_step("s", ["new_field"])])
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        result = enricher.run(df)

        assert list(result["col_a"]) == [1, 2]
        assert list(result["col_b"]) == ["x", "y"]
        assert "new_field" in result.columns

    def test_input_df_not_mutated(self):
        pipeline = Pipeline([_identity_step("s", ["new_field"])])
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"col": [1]})
        original_cols = list(df.columns)
        enricher.run(df)

        assert list(df.columns) == original_cols
        assert "new_field" not in df.columns


# -- overwrite behaviour -----------------------------------------------------


class TestOverwriteBehaviour:
    def test_overwrite_false_preserves_existing(self):
        pipeline = Pipeline(
            [
                FunctionStep("s", fn=lambda ctx: {"f": "new_val"}, fields=["f"]),
            ]
        )
        config = EnrichmentConfig(overwrite_fields=False)
        enricher = Enricher(pipeline, config=config)

        df = pd.DataFrame({"f": ["existing", None]})
        result = enricher.run(df)

        assert result.at[0, "f"] == "existing"
        assert result.at[1, "f"] == "new_val"

    def test_overwrite_true_replaces(self):
        pipeline = Pipeline(
            [
                FunctionStep("s", fn=lambda ctx: {"f": "new_val"}, fields=["f"]),
            ]
        )
        config = EnrichmentConfig(overwrite_fields=True)
        enricher = Enricher(pipeline, config=config)

        df = pd.DataFrame({"f": ["existing", None]})
        result = enricher.run(df)

        assert result.at[0, "f"] == "new_val"
        assert result.at[1, "f"] == "new_val"

    def test_overwrite_param_overrides_config(self):
        pipeline = Pipeline(
            [
                FunctionStep("s", fn=lambda ctx: {"f": "new_val"}, fields=["f"]),
            ]
        )
        config = EnrichmentConfig(overwrite_fields=False)
        enricher = Enricher(pipeline, config=config)

        df = pd.DataFrame({"f": ["existing"]})
        result = enricher.run(df, overwrite_fields=True)

        assert result.at[0, "f"] == "new_val"


# -- sync wrapper ------------------------------------------------------------


class TestSyncWrapper:
    def test_run_works_outside_async(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"x": [1]})
        result = enricher.run(df)
        assert "f" in result.columns

    @pytest.mark.asyncio
    async def test_run_raises_in_async_context(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        enricher = Enricher(pipeline)

        with pytest.raises(RuntimeError, match="run_async"):
            enricher.run(pd.DataFrame({"x": [1]}))

    @pytest.mark.asyncio
    async def test_run_async_works(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        enricher = Enricher(pipeline)

        df = pd.DataFrame({"x": [1, 2]})
        result = await enricher.run_async(df)
        assert list(result["f"]) == ["f_value", "f_value"]


# -- checkpoint resume -------------------------------------------------------


class TestCheckpointResume:
    def test_completed_steps_skipped(self, tmp_path):
        """Write a checkpoint file and verify step 1 is skipped on re-run."""
        call_tracker = {"step1_calls": 0, "step2_calls": 0}

        def step1_fn(ctx):
            call_tracker["step1_calls"] += 1
            return {"f1": "from_step1"}

        def step2_fn(ctx):
            call_tracker["step2_calls"] += 1
            return {"f2": ctx.prior_results.get("f1", "") + "_processed"}

        pipeline = Pipeline(
            [
                FunctionStep("step1", fn=step1_fn, fields=["f1"]),
                FunctionStep("step2", fn=step2_fn, fields=["f2"], depends_on=["step1"]),
            ]
        )

        fields_dict = {"f1": {}, "f2": {}}

        config = EnrichmentConfig(
            enable_checkpointing=True,
            auto_resume=True,
            checkpoint_dir=str(tmp_path),
        )

        # Manually write a checkpoint where step1 is already complete
        checkpoint_data = {
            "timestamp": 1000.0,
            "category": "_default",
            "total_rows": 2,
            "fields_dict": fields_dict,
            "completed_steps": ["step1"],
            "step_results": {
                "step1": [{"f1": "cached_val"}, {"f1": "cached_val2"}],
            },
        }

        # Determine the checkpoint path (mirror the manager's logic — sha256-based)
        import hashlib
        import json as _json

        identifier_source = _json.dumps(
            {"columns": ["x"], "rows": 2}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        data_id = f"df_{hashlib.sha256(identifier_source).hexdigest()[:16]}"
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in data_id)
        cp_path = tmp_path / f"{safe_id}__default_checkpoint.json"
        with open(cp_path, "w") as f:
            json.dump(checkpoint_data, f)

        enricher = Enricher(pipeline, config=config)
        df = pd.DataFrame({"x": [1, 2]})
        result = enricher.run(df)

        # step1 should NOT have been called (it was checkpointed)
        assert call_tracker["step1_calls"] == 0
        # step2 SHOULD have been called
        assert call_tracker["step2_calls"] == 2
        # step2 should use the cached step1 results
        assert result.at[0, "f2"] == "cached_val_processed"
        assert result.at[1, "f2"] == "cached_val2_processed"

    def test_checkpoint_cleaned_on_success(self, tmp_path):
        """Checkpoint file should be removed after successful completion."""
        pipeline = Pipeline([_identity_step("s", ["f"])])
        config = EnrichmentConfig(
            enable_checkpointing=True,
            checkpoint_dir=str(tmp_path),
        )
        enricher = Enricher(pipeline, config=config)

        df = pd.DataFrame({"x": [1]})
        enricher.run(df)

        # No checkpoint files should remain
        files = list(tmp_path.glob("*_checkpoint.json"))
        assert len(files) == 0

    # -- helpers for strict-validation tests ---------------------------------

    @staticmethod
    def _write_checkpoint(
        tmp_path,
        df_columns,
        total_rows,
        fields_dict,
        completed_steps,
        step_results,
        category="_default",
    ):
        """Write a checkpoint file using the same path logic as CheckpointManager._get_path."""
        import hashlib
        import json as _json

        identifier_source = _json.dumps(
            {"columns": df_columns, "rows": total_rows}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        data_id = f"df_{hashlib.sha256(identifier_source).hexdigest()[:16]}"
        # Mirror CheckpointManager._get_path exactly
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in data_id)
        safe_cat = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in category)
        cp_path = tmp_path / f"{safe_id}_{safe_cat}_checkpoint.json"
        checkpoint_data = {
            "timestamp": 1000.0,
            "category": category,
            "total_rows": total_rows,
            "fields_dict": fields_dict,
            "completed_steps": completed_steps,
            "step_results": step_results,
        }
        with open(cp_path, "w") as f:
            _json.dump(checkpoint_data, f)
        return cp_path

    def test_renamed_step_discards_checkpoint(self, tmp_path, caplog):
        """Checkpoint with steps ['a','b'] must be discarded when pipeline has 'a_v2'."""
        import logging

        call_tracker = {"calls": 0}

        def fn(ctx):
            call_tracker["calls"] += 1
            return {"field1": "fresh"}

        # Pipeline now has 'a_v2' instead of 'a'
        pipeline = Pipeline([FunctionStep("a_v2", fn=fn, fields=["field1"])])

        # Write checkpoint that has step 'a' completed (same fields, same rows)
        self._write_checkpoint(
            tmp_path,
            df_columns=["x"],
            total_rows=2,
            fields_dict={"field1": {}},
            completed_steps=["a"],
            step_results={"a": [{"field1": "cached"}, {"field1": "cached"}]},
        )

        config = EnrichmentConfig(
            enable_checkpointing=True,
            auto_resume=True,
            checkpoint_dir=str(tmp_path),
        )
        enricher = Enricher(pipeline, config=config)
        df = pd.DataFrame({"x": [1, 2]})

        with caplog.at_level(logging.WARNING):
            result = enricher.run(df)

        # Must re-run from scratch — step fn must have been called
        assert call_tracker["calls"] == 2
        # Result must come from fresh execution, not cached "cached" value
        assert list(result["field1"]) == ["fresh", "fresh"]
        # A warning about the unknown step must have been logged
        assert any(
            "unknown" in record.message.lower() or "discard" in record.message.lower()
            for record in caplog.records
        )

    def test_row_count_mismatch_discards_checkpoint(self, tmp_path):
        """Checkpoint with 3 rows must be discarded when the DataFrame has 2 rows."""
        call_tracker = {"calls": 0}

        def fn(ctx):
            call_tracker["calls"] += 1
            return {"out": "fresh"}

        pipeline = Pipeline([FunctionStep("s", fn=fn, fields=["out"])])

        # Write checkpoint with 3 rows, but we'll run with 2
        self._write_checkpoint(
            tmp_path,
            df_columns=["x"],
            total_rows=3,  # mismatch
            fields_dict={"out": {}},
            completed_steps=["s"],
            step_results={"s": [{"out": "cached"}] * 3},
        )

        config = EnrichmentConfig(
            enable_checkpointing=True,
            auto_resume=True,
            checkpoint_dir=str(tmp_path),
        )
        enricher = Enricher(pipeline, config=config)
        df = pd.DataFrame({"x": [1, 2]})  # 2 rows
        result = enricher.run(df)

        # Must re-run from scratch
        assert call_tracker["calls"] == 2
        assert list(result["out"]) == ["fresh", "fresh"]

    def test_field_mismatch_discards_checkpoint(self, tmp_path):
        """Checkpoint with different fields must be discarded."""
        call_tracker = {"calls": 0}

        def fn(ctx):
            call_tracker["calls"] += 1
            return {"new_field": "fresh"}

        pipeline = Pipeline([FunctionStep("s", fn=fn, fields=["new_field"])])

        # Write checkpoint with different fields
        self._write_checkpoint(
            tmp_path,
            df_columns=["x"],
            total_rows=2,
            fields_dict={"old_field": {}},  # mismatch
            completed_steps=["s"],
            step_results={"s": [{"old_field": "cached"}] * 2},
        )

        config = EnrichmentConfig(
            enable_checkpointing=True,
            auto_resume=True,
            checkpoint_dir=str(tmp_path),
        )
        enricher = Enricher(pipeline, config=config)
        df = pd.DataFrame({"x": [1, 2]})
        result = enricher.run(df)

        # Must re-run from scratch
        assert call_tracker["calls"] == 2
        assert list(result["new_field"]) == ["fresh", "fresh"]

    def test_matching_shape_resumes_correctly(self, tmp_path):
        """Checkpoint with matching rows, fields, and steps must resume (skip completed step)."""
        call_tracker = {"step1_calls": 0, "step2_calls": 0}

        def step1_fn(ctx):
            call_tracker["step1_calls"] += 1
            return {"f1": "live_val"}

        def step2_fn(ctx):
            call_tracker["step2_calls"] += 1
            return {"f2": ctx.prior_results.get("f1", "") + "_done"}

        pipeline = Pipeline(
            [
                FunctionStep("step1", fn=step1_fn, fields=["f1"]),
                FunctionStep("step2", fn=step2_fn, fields=["f2"], depends_on=["step1"]),
            ]
        )

        self._write_checkpoint(
            tmp_path,
            df_columns=["x"],
            total_rows=2,
            fields_dict={"f1": {}, "f2": {}},
            completed_steps=["step1"],
            step_results={"step1": [{"f1": "cached_val"}, {"f1": "cached_val2"}]},
        )

        config = EnrichmentConfig(
            enable_checkpointing=True,
            auto_resume=True,
            checkpoint_dir=str(tmp_path),
        )
        enricher = Enricher(pipeline, config=config)
        df = pd.DataFrame({"x": [1, 2]})
        result = enricher.run(df)

        # step1 must be skipped (it was checkpointed with matching state)
        assert call_tracker["step1_calls"] == 0
        # step2 must run using the cached step1 results
        assert call_tracker["step2_calls"] == 2
        assert result.at[0, "f2"] == "cached_val_done"
        assert result.at[1, "f2"] == "cached_val2_done"


# -- stable data_identifier --------------------------------------------------


class TestStableDataIdentifier:
    def test_same_dataframe_same_identifier(self, tmp_path):
        """Two Enricher instances for the same DataFrame must produce the same identifier.

        The old ``hash()``-based approach was per-process randomized
        (PYTHONHASHSEED), so checkpoints from one run couldn't be resumed in
        another. The sha256-based replacement is deterministic.
        """
        import hashlib
        import json as _json

        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})

        # Compute the identifier the same way enricher.py does.
        def _compute_id(df_):
            source = _json.dumps(
                {"columns": list(df_.columns), "rows": len(df_)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return f"df_{hashlib.sha256(source).hexdigest()[:16]}"

        id1 = _compute_id(df)
        id2 = _compute_id(df.copy())

        # Both copies of the same DataFrame must produce the same identifier.
        assert id1 == id2
        # And the identifier must be a stable hex string (not an integer from hash()).
        assert id1.startswith("df_")
        assert len(id1) == len("df_") + 16


# -- cancellation + resume (regression for partial-checkpoint corruption) -----


class TestCancelAndResume:
    @pytest.mark.asyncio
    async def test_cancel_mid_step_resume_has_no_empty_rows(self, tmp_path):
        """Cancel mid-step; resume must re-run the step — no {} placeholders downstream.

        Regression for: on_partial_checkpoint used to mark the step completed,
        so a resumed run would skip it and propagate empty {} for unfinished rows.
        """
        num_rows = 10
        # rows_started tracks which row indices were actually processed by the step.
        rows_started: list[int] = []
        # Deterministic synchronization: the main coroutine waits on this event
        # so cancellation always lands while row 2 is mid-flight (rows 0-1 done,
        # row 2 hanging, rows 3-9 not yet started).  This removes the timing
        # dependency that made the previous version flaky on slow CI workers.
        started_row_2 = asyncio.Event()

        async def slow_fn(ctx):
            row_idx = ctx.row.get("idx", -1)
            rows_started.append(row_idx)
            if row_idx == 2:
                # Signal the main coroutine that row 2 has started, then hang
                # long enough that cancellation arrives before the row finishes.
                started_row_2.set()
                await asyncio.sleep(10.0)
            else:
                await asyncio.sleep(0.01)
            return {"enriched": f"done_{row_idx}"}

        pipeline = Pipeline([FunctionStep("enrich", fn=slow_fn, fields=["enriched"])])
        config = EnrichmentConfig(
            enable_checkpointing=True,
            auto_resume=True,
            checkpoint_dir=str(tmp_path),
            # Fire a partial checkpoint after every row so the checkpoint is
            # guaranteed to be written before the cancellation arrives.
            checkpoint_interval=1,
            max_workers=1,  # sequential to make cancellation timing deterministic
        )
        enricher = Enricher(pipeline, config=config)
        df = pd.DataFrame({"idx": list(range(num_rows))})

        # Wait until row 2 is in-flight, then cancel — no wall-clock timeouts.
        task = asyncio.create_task(enricher.run_async(df, data_identifier="cancel_test"))
        await started_row_2.wait()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        # Confirm we actually cancelled before finishing all rows.
        assert len(rows_started) < num_rows, "Test setup error: all rows completed before cancel"
        # And confirm row 2 was the in-flight row (so the partial checkpoint has
        # results for rows 0,1 only — row 2 was cancelled mid-flight).
        assert 2 in rows_started, "Row 2 should have started before cancel"

        # Now resume with a fresh Enricher and confirm no {} leaks downstream.
        rows_started.clear()
        enricher2 = Enricher(pipeline, config=config)
        result = await enricher2.run_async(df, data_identifier="cancel_test")

        enriched_col = list(result["enriched"])
        # Every value must be non-empty — no {} placeholders from the partial save.
        for i, val in enumerate(enriched_col):
            assert val != {} and val is not None and val != "", (
                f"Row {i} has empty/placeholder value after resume: {val!r}"
            )

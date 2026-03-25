"""Tests for the Enricher — checkpoint-capable pipeline runner."""

from __future__ import annotations

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

        # Determine the checkpoint path (mirror the manager's logic)
        data_id = f"df_{hash(str(['x']) + str(2))}"
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

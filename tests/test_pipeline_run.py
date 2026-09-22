"""Tests for Pipeline.run() / run_async() — the primary public API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from accrue.core.config import EnrichmentConfig
from accrue.core.enricher import Enricher
from accrue.pipeline.pipeline import Pipeline, PipelineResult
from accrue.schemas.base import CostSummary
from accrue.steps.function import FunctionStep
from accrue.steps.llm import LLMStep

# -- helpers -----------------------------------------------------------------


def _identity_step(name: str, fields: list[str], **kwargs) -> FunctionStep:
    def fn(ctx):
        return {f: f"{f}_value" for f in fields}

    return FunctionStep(name=name, fn=fn, fields=fields, **kwargs)


# -- Pipeline.run() ----------------------------------------------------------


class TestPipelineRun:
    def test_simple_run(self):
        pipeline = Pipeline([_identity_step("s", ["market_size"])])
        df = pd.DataFrame({"company": ["Acme", "Beta"]})

        result = pipeline.run(df)

        assert isinstance(result, PipelineResult)
        assert "market_size" in result.data.columns
        assert list(result.data["market_size"]) == [
            "market_size_value",
            "market_size_value",
        ]
        assert list(result.data["company"]) == ["Acme", "Beta"]

    def test_run_returns_cost_and_errors(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        df = pd.DataFrame({"x": [1]})

        result = pipeline.run(df)

        assert isinstance(result.cost, CostSummary)
        assert result.errors == []
        assert not result.has_errors
        assert result.success_rate == 1.0

    def test_run_with_config(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        df = pd.DataFrame({"x": [1]})
        config = EnrichmentConfig(max_workers=5)

        result = pipeline.run(df, config=config)
        assert result.data.at[0, "f"] == "f_value"

    def test_run_filters_internal_fields(self):
        """__ prefixed fields are not in the output DataFrame."""
        pipeline = Pipeline(
            [
                FunctionStep(
                    "s",
                    fn=lambda ctx: {"__internal": "secret", "visible": "yes"},
                    fields=["__internal", "visible"],
                ),
            ]
        )
        df = pd.DataFrame({"x": [1]})

        result = pipeline.run(df)
        assert "visible" in result.data.columns
        assert "__internal" not in result.data.columns

    def test_run_multi_step_with_dependencies(self):
        def step_a_fn(ctx):
            return {"intermediate": ctx.row["input"] + "_processed"}

        def step_b_fn(ctx):
            return {"final": ctx.prior_results["intermediate"] + "_done"}

        pipeline = Pipeline(
            [
                FunctionStep("a", fn=step_a_fn, fields=["intermediate"]),
                FunctionStep("b", fn=step_b_fn, fields=["final"], depends_on=["a"]),
            ]
        )
        df = pd.DataFrame({"input": ["hello", "world"]})

        result = pipeline.run(df)
        assert list(result.data["final"]) == [
            "hello_processed_done",
            "world_processed_done",
        ]


# -- Pipeline.run_async() ---------------------------------------------------


class TestPipelineRunAsync:
    @pytest.mark.asyncio
    async def test_run_async(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        df = pd.DataFrame({"x": [1, 2]})

        result = await pipeline.run_async(df)
        assert isinstance(result, PipelineResult)
        assert list(result.data["f"]) == ["f_value", "f_value"]

    @pytest.mark.asyncio
    async def test_run_async_with_errors(self):
        from accrue.core.exceptions import StepError

        def failing_fn(ctx):
            if ctx.row["x"] == 2:
                raise StepError("fail", step_name="s")
            return {"f": "ok"}

        pipeline = Pipeline([FunctionStep("s", fn=failing_fn, fields=["f"])])
        df = pd.DataFrame({"x": [1, 2, 3]})

        result = await pipeline.run_async(df)
        assert result.has_errors
        assert len(result.errors) == 1
        assert result.errors[0].row_index == 1
        assert result.data.at[0, "f"] == "ok"
        assert pd.isna(result.data.at[1, "f"])  # sentinel


# -- Pipeline.runner() ------------------------------------------------------


class TestPipelineRunner:
    def test_runner_returns_enricher(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        runner = pipeline.runner()
        assert isinstance(runner, Enricher)

    def test_runner_with_config(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        config = EnrichmentConfig(max_workers=5)
        runner = pipeline.runner(config=config)
        assert runner.config.max_workers == 5


# -- LLMStep inline field specs ---------------------------------------------


class TestInlineFieldSpecs:
    def test_dict_string_shorthand(self):
        step = LLMStep(
            "analyze",
            fields={
                "market_size": "Estimate TAM in billions USD",
                "competition": "Rate Low/Medium/High",
            },
        )
        assert step.fields == ["market_size", "competition"]
        # _field_specs now stores FieldSpec objects
        assert step._field_specs["market_size"].prompt == "Estimate TAM in billions USD"
        assert step._field_specs["competition"].prompt == "Rate Low/Medium/High"

    def test_dict_full_spec(self):
        step = LLMStep(
            "analyze",
            fields={
                "market_size": {
                    "prompt": "Estimate TAM",
                    "type": "String",
                },
            },
        )
        assert step.fields == ["market_size"]
        assert step._field_specs["market_size"].type == "String"

    def test_list_fields_backward_compat(self):
        step = LLMStep("analyze", fields=["market_size", "competition"])
        assert step.fields == ["market_size", "competition"]
        assert step._field_specs == {}

    def test_inline_specs_used_in_system_message(self):
        from accrue.steps.base import StepContext

        step = LLMStep(
            "analyze",
            fields={
                "market_size": "Estimate TAM in billions USD",
            },
        )
        ctx = StepContext(
            row={"company": "Acme"},
            fields={},
            prior_results={},
        )
        parts = step._build_prompt(ctx)
        assert "Estimate TAM in billions USD" in parts.system

    def test_collect_field_specs_from_pipeline(self):
        pipeline = Pipeline(
            [
                LLMStep(
                    "analyze",
                    fields={
                        "market_size": "Estimate TAM",
                        "competition": {"prompt": "Rate competition", "type": "String"},
                    },
                ),
                FunctionStep("search", fn=lambda ctx: {"__ctx": "data"}, fields=["__ctx"]),
            ]
        )
        specs = pipeline._collect_field_specs()
        assert "market_size" in specs
        assert specs["market_size"]["prompt"] == "Estimate TAM"
        assert "competition" in specs
        assert "__ctx" not in specs  # internal fields excluded

    def test_pipeline_run_with_inline_specs(self):
        """Full integration: Pipeline.run() with LLMStep inline fields (mocked)."""
        import json

        from accrue.schemas.base import UsageInfo
        from accrue.steps.providers.base import LLMResponse

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps({"market_size": "$5B", "competition": "High"}),
                usage=UsageInfo(
                    prompt_tokens=50,
                    completion_tokens=20,
                    total_tokens=70,
                    model="test",
                ),
            )
        )

        pipeline = Pipeline(
            [
                LLMStep(
                    "analyze",
                    fields={
                        "market_size": "Estimate TAM",
                        "competition": "Rate competition level",
                    },
                    client=mock_client,
                ),
            ]
        )

        df = pd.DataFrame({"company": ["Acme"]})
        result = pipeline.run(df)

        assert result.data.at[0, "market_size"] == "$5B"
        assert result.data.at[0, "competition"] == "High"
        assert result.cost.total_tokens == 70
        assert "analyze" in result.cost.steps


# -- PipelineResult ----------------------------------------------------------


class TestPipelineResult:
    def test_success_rate_no_errors(self):
        r = PipelineResult(data=pd.DataFrame({"x": [1, 2, 3]}))
        assert r.success_rate == 1.0
        assert not r.has_errors

    def test_success_rate_with_errors(self):
        from accrue.core.exceptions import RowError

        r = PipelineResult(
            data=pd.DataFrame({"x": [1, 2, 3]}),
            errors=[RowError(row_index=1, step_name="s", error=ValueError("bad"))],
        )
        assert r.has_errors
        assert abs(r.success_rate - 2 / 3) < 0.01

    def test_empty_dataframe(self):
        r = PipelineResult(data=pd.DataFrame())
        assert r.success_rate == 1.0


# -- attachments through the full DataFrame path (#153) ----------------------


class TestPipelineRunAttachments:
    """A DataFrame column of Python lists must survive row loading intact."""

    def test_document_blocks_reach_the_client(self, tmp_path):
        import json

        from accrue.schemas.base import UsageInfo
        from accrue.steps.providers.base import LLMResponse

        a = tmp_path / "a.pdf"
        a.write_bytes(b"%PDF-A")
        b = tmp_path / "b.pdf"
        b.write_bytes(b"%PDF-B")

        captured: list[list] = []

        async def fake_complete(**kwargs):
            captured.append(kwargs["messages"])
            return LLMResponse(
                content=json.dumps({"summary": "ok"}),
                usage=UsageInfo(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2, model="claude"
                ),
            )

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(side_effect=fake_complete)

        step = LLMStep(
            name="read_docs",
            fields={"summary": "Summarise the attached documents"},
            client=mock_client,
            attachments="documents",
            cache=False,
        )
        df = pd.DataFrame(
            {
                "company": ["Acme", "Beta"],
                "documents": [[str(a)], [str(a), str(b)]],
            }
        )

        result = Pipeline([step]).run(df, config=EnrichmentConfig(enable_caching=False))

        assert list(result.data["summary"]) == ["ok", "ok"]
        assert len(captured) == 2

        by_company = {msgs[1]["content"][-1]["text"]: msgs[1]["content"][:-1] for msgs in captured}
        acme_text = next(t for t in by_company if "Acme" in t)
        beta_text = next(t for t in by_company if "Beta" in t)

        # <row_data> never carries the attachments column.
        assert "documents" not in acme_text
        assert "a.pdf" not in acme_text

        assert [blk["data"] for blk in by_company[acme_text]] == [b"%PDF-A"]
        assert [blk["data"] for blk in by_company[beta_text]] == [b"%PDF-A", b"%PDF-B"]

"""Tests for Pipeline.plan() — dry-run preview + cost extrapolation (issue #12)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from accrue.core.config import EnrichmentConfig
from accrue.pipeline.pipeline import Pipeline
from accrue.pipeline.plan import PipelinePlan, extrapolate_cost
from accrue.schemas.base import CostSummary, StepUsage, UsageInfo
from accrue.steps.base import StepContext
from accrue.steps.function import FunctionStep
from accrue.steps.llm import LLMStep
from accrue.steps.providers.base import LLMResponse

# -- helpers -------------------------------------------------------------


def _double_step() -> FunctionStep:
    def fn(ctx: StepContext) -> dict[str, Any]:
        return {"v": ctx.row["i"] * 2}

    return FunctionStep("double", fn=fn, fields=["v"])


def _mock_client(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    resp = LLMResponse(
        content=content,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model="gpt-4.1-mini",
        ),
    )
    client = AsyncMock()
    client.complete = AsyncMock(return_value=resp)
    return client


def _fresh_config(tmp_path) -> EnrichmentConfig:
    """Caching on, but pointed at an empty cache dir so every sample row misses."""
    return EnrichmentConfig(cache_dir=str(tmp_path / "cache"))


# -- cost extrapolation math --------------------------------------------


class TestExtrapolateCost:
    def test_scales_by_executed_rows(self):
        sample = CostSummary(total_prompt_tokens=300, total_completion_tokens=150, total_tokens=450)
        sample.steps["s"] = StepUsage(
            prompt_tokens=300,
            completion_tokens=150,
            total_tokens=450,
            rows_processed=3,
            cache_misses=3,
        )
        est = extrapolate_cost(sample, total_rows=30)
        # 3 executed rows → 100/50 per row → ×30
        assert est.total_prompt_tokens == 3000
        assert est.total_completion_tokens == 1500
        assert est.total_tokens == 4500

    def test_cached_rows_not_double_charged(self):
        # 3 rows processed but only 2 actually called the API (1 cache hit).
        sample = CostSummary(total_prompt_tokens=200, total_tokens=200)
        sample.steps["s"] = StepUsage(
            prompt_tokens=200,
            total_tokens=200,
            rows_processed=3,
            cache_hits=1,
            cache_misses=2,
        )
        est = extrapolate_cost(sample, total_rows=20)
        # Divide by executed=2 (misses), not 3 → 100/row → ×20 = 2000
        assert est.total_prompt_tokens == 2000

    def test_caching_off_falls_back_to_rows_processed(self):
        # Caching off: misses reported as 0, rows_processed is the executed count.
        sample = CostSummary(total_prompt_tokens=100, total_tokens=100)
        sample.steps["s"] = StepUsage(
            prompt_tokens=100, total_tokens=100, rows_processed=2, cache_misses=0
        )
        est = extrapolate_cost(sample, total_rows=10)
        assert est.total_prompt_tokens == 500

    def test_zero_token_step_contributes_nothing(self):
        sample = CostSummary()
        sample.steps["fn"] = StepUsage(rows_processed=3, cache_misses=3)
        est = extrapolate_cost(sample, total_rows=99)
        assert est.total_tokens == 0


# -- sample selection + structure ----------------------------------------


class TestPlanSampling:
    def test_selects_first_n_rows_list(self):
        data = [{"i": i} for i in range(10)]
        plan = Pipeline([_double_step()]).plan(data, sample_size=3)
        assert isinstance(plan, PipelinePlan)
        assert plan.total_rows == 10
        assert plan.sample_size == 3
        assert plan.sample_rows == [{"i": 0}, {"i": 1}, {"i": 2}]

    def test_selects_first_n_rows_dataframe(self):
        df = pd.DataFrame({"i": list(range(8))})
        plan = Pipeline([_double_step()]).plan(df, sample_size=2)
        assert plan.total_rows == 8
        assert [r["i"] for r in plan.sample_rows] == [0, 1]

    def test_sample_size_clamped_to_dataset(self):
        plan = Pipeline([_double_step()]).plan([{"i": 1}], sample_size=5)
        assert plan.sample_size == 1
        assert len(plan.sample_outputs) == 1

    def test_sample_outputs_are_real(self):
        plan = Pipeline([_double_step()]).plan([{"i": 4}, {"i": 5}, {"i": 6}], sample_size=2)
        assert [r["v"] for r in plan.sample_outputs] == [8, 10]


# -- schema + prompt rendering (LLM step, mocked) ------------------------


class TestPlanSchemaRendering:
    def test_captures_schema_fields_and_prompt(self, tmp_path):
        step = LLMStep(
            name="enrich",
            fields={"market_size": "Estimate TAM"},
            structured_outputs=True,
            client=_mock_client('{"market_size": "42"}'),
        )
        plan = Pipeline([step]).plan(
            [{"company": "Acme"}, {"company": "Beta"}],
            sample_size=2,
            config=_fresh_config(tmp_path),
        )
        sp = next(s for s in plan.steps if s.name == "enrich")
        assert sp.kind == "llm"
        assert sp.model == "gpt-4.1-mini"
        assert sp.response_format["type"] == "json_schema"
        assert "market_size" in sp.response_format["json_schema"]["schema"]["properties"]
        assert sp.system_prompt  # resolved, non-empty
        # And it surfaces in the rendered summary
        text = plan.summary()
        assert "market_size" in text

    def test_extrapolates_llm_cost_to_full_dataset(self, tmp_path):
        # 6-row dataset, sample 2 → each mock call is 10/5 tokens, 2 misses.
        data = [{"company": f"c{i}"} for i in range(6)]
        step = LLMStep(
            name="enrich",
            fields={"x": "do"},
            structured_outputs=True,
            client=_mock_client('{"x": "y"}', prompt_tokens=10, completion_tokens=5),
        )
        plan = Pipeline([step]).plan(data, sample_size=2, config=_fresh_config(tmp_path))
        # sample: 2 rows × 10 prompt = 20; extrapolate ×(6/2) = 60
        assert plan.sample_cost.total_prompt_tokens == 20
        assert plan.estimated_cost.total_prompt_tokens == 60
        assert plan.estimated_cost.total_completion_tokens == 30


# -- summary rendering ---------------------------------------------------


class TestPlanSummary:
    def test_summary_includes_all_sections(self):
        plan = Pipeline([_double_step()]).plan([{"i": i} for i in range(5)], sample_size=2)
        text = plan.summary()
        assert "Pipeline Plan" in text
        assert "double" in text  # step name
        assert "Sample outputs" in text
        assert "full run (est.)" in text
        assert "5" in text  # total rows mentioned


# -- run(confirm=True) ---------------------------------------------------


class TestRunConfirm:
    def test_confirm_yes_proceeds(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
        result = Pipeline([_double_step()]).run([{"i": 3}], confirm=True)
        assert result.data[0]["v"] == 6
        # The plan preview was printed
        assert "Pipeline Plan" in capsys.readouterr().out

    def test_confirm_no_aborts(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
        with pytest.raises(RuntimeError, match="aborted"):
            Pipeline([_double_step()]).run([{"i": 3}], confirm=True)

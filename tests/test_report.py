"""Tests for ``PipelineResult.report()`` and the builtin heuristics."""

from __future__ import annotations

import pandas as pd
import pytest

from accrue.core.exceptions import RowError
from accrue.pipeline.pipeline import Pipeline, PipelineResult
from accrue.pipeline.report import (
    BUILTIN_HEURISTICS,
    Finding,
    ReportContext,
    render_html,
    render_markdown,
    run_heuristics,
)
from accrue.schemas.base import CostSummary, StepUsage
from accrue.steps.function import FunctionStep

# -- helpers -----------------------------------------------------------------


def _ctx(
    *,
    data: pd.DataFrame | list[dict] | None = None,
    cost: CostSummary | None = None,
    errors: list[RowError] | None = None,
    field_specs: dict | None = None,
    pipeline_elapsed: float = 1.0,
    step_elapsed: dict[str, float] | None = None,
) -> ReportContext:
    return ReportContext(
        data=data if data is not None else pd.DataFrame(),
        cost=cost or CostSummary(),
        errors=errors or [],
        field_specs=field_specs or {},
        pipeline_elapsed_seconds=pipeline_elapsed,
        step_elapsed_seconds=step_elapsed or {},
    )


# -- enum_collapse -----------------------------------------------------------


class TestEnumCollapse:
    def test_fires_when_top_value_dominates(self):
        df = pd.DataFrame({"category": (["Other"] * 73) + (["A"] * 17) + (["B"] * 10)})
        ctx = _ctx(
            data=df,
            field_specs={"category": {"prompt": "x", "enum": ["Other", "A", "B"]}},
        )
        out = BUILTIN_HEURISTICS["enum_collapse"](ctx)
        assert len(out) == 1
        f = out[0]
        assert f.code == "enum_collapse"
        assert f.subject == "category"
        assert "Other" in f.message
        assert "73%" in f.message
        assert f.evidence["top_value"] == "Other"

    def test_silent_when_well_distributed(self):
        df = pd.DataFrame({"category": (["A"] * 30) + (["B"] * 35) + (["C"] * 35)})
        ctx = _ctx(data=df, field_specs={"category": {"prompt": "x", "enum": ["A", "B", "C"]}})
        assert BUILTIN_HEURISTICS["enum_collapse"](ctx) == []

    def test_only_runs_on_enum_fields(self):
        df = pd.DataFrame({"name": ["Acme"] * 20})
        ctx = _ctx(data=df, field_specs={"name": {"prompt": "x"}})
        assert BUILTIN_HEURISTICS["enum_collapse"](ctx) == []


# -- numeric_clipping --------------------------------------------------------


class TestNumericClipping:
    def test_fires_when_max_dominates(self):
        df = pd.DataFrame({"score": ([100] * 41) + [50, 60, 70, 80, 90] * 11 + [10] * 4})
        ctx = _ctx(data=df, field_specs={"score": {"prompt": "x", "type": "Number"}})
        out = BUILTIN_HEURISTICS["numeric_clipping"](ctx)
        assert len(out) == 1
        assert "max" in out[0].message
        assert "100" in out[0].message

    def test_silent_when_dominant_value_is_in_middle(self):
        # Top value is 50, which is neither the min (10) nor the max (100).
        df = pd.DataFrame({"score": [50] * 50 + [10] * 25 + [100] * 25})
        ctx = _ctx(data=df, field_specs={"score": {"prompt": "x", "type": "Number"}})
        assert BUILTIN_HEURISTICS["numeric_clipping"](ctx) == []

    def test_skips_non_numeric_field(self):
        df = pd.DataFrame({"label": ["hi"] * 10})
        ctx = _ctx(data=df, field_specs={"label": {"prompt": "x", "type": "String"}})
        assert BUILTIN_HEURISTICS["numeric_clipping"](ctx) == []


# -- length_anomaly ----------------------------------------------------------


class TestLengthAnomaly:
    def test_fires_when_outputs_too_short(self):
        # Hint: 80-120 words. Outputs are 12 words on average.
        df = pd.DataFrame(
            {"summary": ["one two three four five six seven eight nine ten eleven twelve"] * 20}
        )
        ctx = _ctx(
            data=df,
            field_specs={"summary": {"prompt": "x", "type": "String", "format": "80-120 words"}},
        )
        out = BUILTIN_HEURISTICS["length_anomaly"](ctx)
        assert len(out) == 1
        assert "12" in out[0].message
        assert "80" in out[0].message and "120" in out[0].message
        assert "truncated" in out[0].message or "lazy" in out[0].message

    def test_silent_when_within_range(self):
        words = " ".join(["w"] * 100)
        df = pd.DataFrame({"summary": [words] * 5})
        ctx = _ctx(
            data=df,
            field_specs={"summary": {"prompt": "x", "type": "String", "format": "80-120 words"}},
        )
        assert BUILTIN_HEURISTICS["length_anomaly"](ctx) == []

    def test_silent_when_format_has_no_range(self):
        df = pd.DataFrame({"summary": ["short"] * 10})
        ctx = _ctx(
            data=df,
            field_specs={"summary": {"prompt": "x", "type": "String", "format": "YYYY-MM-DD"}},
        )
        assert BUILTIN_HEURISTICS["length_anomaly"](ctx) == []


# -- retry_storm -------------------------------------------------------------


class TestRetryStorm:
    def test_fires_above_5_pct_errors(self):
        cost = CostSummary(steps={"analyze": StepUsage(rows_processed=100, rows_skipped=0)})
        errors = [
            RowError(row_index=i, step_name="analyze", error=ValueError("fail")) for i in range(8)
        ]
        ctx = _ctx(data=[{}] * 100, cost=cost, errors=errors)
        out = BUILTIN_HEURISTICS["retry_storm"](ctx)
        assert len(out) == 1
        assert "analyze" in out[0].message

    def test_silent_below_threshold(self):
        cost = CostSummary(steps={"analyze": StepUsage(rows_processed=100, rows_skipped=0)})
        errors = [RowError(row_index=0, step_name="analyze", error=ValueError("fail"))]
        ctx = _ctx(data=[{}] * 100, cost=cost, errors=errors)
        assert BUILTIN_HEURISTICS["retry_storm"](ctx) == []


# -- cache_thrash ------------------------------------------------------------


class TestCacheThrash:
    def test_fires_when_low_hit_rate_with_some_hits(self):
        cost = CostSummary(
            steps={"classify": StepUsage(rows_processed=100, cache_hits=3, cache_misses=97)}
        )
        ctx = _ctx(data=[{}] * 100, cost=cost)
        out = BUILTIN_HEURISTICS["cache_thrash"](ctx)
        assert len(out) == 1
        assert out[0].subject == "classify"

    def test_silent_on_first_run_with_no_hits(self):
        # No cache_hits at all — first run, not thrash.
        cost = CostSummary(
            steps={"classify": StepUsage(rows_processed=100, cache_hits=0, cache_misses=100)}
        )
        ctx = _ctx(data=[{}] * 100, cost=cost)
        assert BUILTIN_HEURISTICS["cache_thrash"](ctx) == []


# -- refusal_pattern ---------------------------------------------------------


class TestRefusalPattern:
    def test_fires_above_5_pct(self):
        df = pd.DataFrame(
            {"summary": (["I cannot determine the answer."] * 8 + ["actual content here"] * 92)}
        )
        ctx = _ctx(data=df, field_specs={"summary": {"prompt": "x", "type": "String"}})
        out = BUILTIN_HEURISTICS["refusal_pattern"](ctx)
        assert len(out) == 1
        assert "summary" in out[0].subject

    def test_silent_when_below_threshold(self):
        df = pd.DataFrame({"summary": ["I cannot determine."] + ["fine"] * 100})
        ctx = _ctx(data=df, field_specs={"summary": {"prompt": "x", "type": "String"}})
        assert BUILTIN_HEURISTICS["refusal_pattern"](ctx) == []


# -- cost_outlier ------------------------------------------------------------


class TestCostOutlier:
    def test_fires_when_step_dominates_cost(self):
        cost = CostSummary(
            total_tokens=10_000,
            total_prompt_tokens=8_000,
            total_completion_tokens=2_000,
            steps={
                "extract": StepUsage(total_tokens=1_000),
                "analyze": StepUsage(total_tokens=9_000),
            },
        )
        ctx = _ctx(cost=cost)
        out = BUILTIN_HEURISTICS["cost_outlier"](ctx)
        assert len(out) == 1
        assert out[0].subject == "analyze"
        assert "90%" in out[0].message

    def test_silent_when_cost_balanced(self):
        cost = CostSummary(
            total_tokens=10_000,
            steps={
                "a": StepUsage(total_tokens=5_000),
                "b": StepUsage(total_tokens=5_000),
            },
        )
        ctx = _ctx(cost=cost)
        assert BUILTIN_HEURISTICS["cost_outlier"](ctx) == []

    def test_silent_on_single_step_pipeline(self):
        # A 1-step pipeline always has 100% of cost in the only step; flagging
        # it adds no information.  Real cost outliers need >=2 steps to compare.
        cost = CostSummary(
            total_tokens=50_000,
            steps={"only": StepUsage(total_tokens=50_000)},
        )
        ctx = _ctx(cost=cost)
        assert BUILTIN_HEURISTICS["cost_outlier"](ctx) == []


# -- run_heuristics ----------------------------------------------------------


class TestRunHeuristics:
    def test_disable_skips_heuristic(self):
        df = pd.DataFrame({"category": ["X"] * 100})
        ctx = _ctx(
            data=df,
            field_specs={"category": {"prompt": "x", "enum": ["X", "Y"]}},
        )
        with_collapse = [f.code for f in run_heuristics(ctx)]
        without_collapse = [f.code for f in run_heuristics(ctx, disable=["enum_collapse"])]
        assert "enum_collapse" in with_collapse
        assert "enum_collapse" not in without_collapse

    def test_findings_sorted_by_severity(self):
        cost = CostSummary(
            total_tokens=10_000,
            steps={
                "a": StepUsage(total_tokens=1_000),
                "b": StepUsage(total_tokens=9_000),
            },
        )
        df = pd.DataFrame({"cat": ["X"] * 100})
        ctx = _ctx(
            data=df,
            cost=cost,
            field_specs={"cat": {"prompt": "x", "enum": ["X", "Y"]}},
        )
        findings = run_heuristics(ctx)
        # warnings come before info
        ranks = ["warning" if f.severity == "warning" else f.severity for f in findings]
        assert ranks == sorted(ranks, key=lambda s: {"warning": 1, "info": 2}.get(s, 99))


# -- renderers ---------------------------------------------------------------


class TestRenderers:
    def test_markdown_includes_findings_and_per_step(self):
        cost = CostSummary(
            total_tokens=1_000,
            total_prompt_tokens=700,
            total_completion_tokens=300,
            steps={"s": StepUsage(rows_processed=10, total_tokens=1_000, model="gpt-x")},
        )
        ctx = _ctx(
            data=pd.DataFrame({"x": range(10)}),
            cost=cost,
            step_elapsed={"s": 1.5},
        )
        findings = [
            Finding(
                code="enum_collapse",
                severity="warning",
                subject="cat",
                message="`cat` collapsed to `X` on 99% of rows. Try a wider enum.",
            )
        ]
        md = render_markdown(ctx, findings)
        assert "Pipeline Run Report" in md
        assert "Flagged patterns" in md
        assert "cat" in md
        assert "(`enum_collapse`)" in md
        assert "Per-step" in md
        assert "gpt-x" in md

    def test_markdown_no_findings_message(self):
        ctx = _ctx(data=pd.DataFrame({"x": [1, 2, 3]}))
        md = render_markdown(ctx, [])
        assert "No flagged patterns" in md

    def test_html_is_well_formed_and_escaped(self):
        ctx = _ctx(data=pd.DataFrame({"x": range(3)}))
        findings = [
            Finding(
                code="x",
                severity="warning",
                subject="evil",
                message="<script>alert(1)</script> bad",
            )
        ]
        html = render_html(ctx, findings)
        assert html.startswith("<!doctype html>")
        assert "</body></html>" in html
        # raw <script> must not appear
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# -- PipelineResult.report ---------------------------------------------------


class TestPipelineResultReport:
    def test_default_markdown(self):
        r = PipelineResult(
            data=pd.DataFrame({"x": [1, 2]}),
            pipeline_elapsed_seconds=0.5,
        )
        out = r.report()
        assert "Pipeline Run Report" in out
        assert out.endswith("\n")

    def test_html_format(self):
        r = PipelineResult(
            data=pd.DataFrame({"x": [1, 2]}),
            pipeline_elapsed_seconds=0.5,
        )
        out = r.report(format="html")
        assert out.startswith("<!doctype html>")

    def test_unknown_format_raises(self):
        r = PipelineResult(data=pd.DataFrame({"x": [1]}))
        with pytest.raises(ValueError, match="Unknown report format"):
            r.report(format="json")

    def test_path_writes_file(self, tmp_path):
        r = PipelineResult(
            data=pd.DataFrame({"x": [1, 2]}),
            pipeline_elapsed_seconds=0.1,
        )
        target = tmp_path / "run.html"
        out = r.report(format="html", path=str(target))
        assert target.exists()
        assert target.read_text() == out

    def test_disable_silences_heuristic(self):
        df = pd.DataFrame({"category": ["X"] * 100})
        r = PipelineResult(
            data=df,
            field_specs={"category": {"prompt": "x", "enum": ["X", "Y"]}},
        )
        with_flag = r.report()
        without_flag = r.report(disable=["enum_collapse"])
        assert "enum_collapse" in with_flag
        assert "enum_collapse" not in without_flag

    def test_end_to_end_through_pipeline(self):
        """Smoke test: report() works on a real pipeline run."""
        pipeline = Pipeline(
            [
                FunctionStep(
                    "categorise",
                    fn=lambda ctx: {"category": "Other"},
                    fields=["category"],
                )
            ]
        )
        df = pd.DataFrame({"company": [f"c{i}" for i in range(10)]})
        result = pipeline.run(df)

        out = result.report()
        assert "Pipeline Run Report" in out
        assert result.pipeline_elapsed_seconds >= 0
        # FunctionStep doesn't register field_specs (no FieldSpec), so no enum
        # heuristic — but the structure should still render.
        assert "categorise" in out or "No flagged patterns" in out

    def test_end_to_end_with_list_of_dicts(self):
        """report() runs on a pipeline whose input was a list[dict]."""
        pipeline = Pipeline([FunctionStep("tag", fn=lambda ctx: {"tag": "x"}, fields=["tag"])])
        rows = [{"company": f"c{i}"} for i in range(5)]
        result = pipeline.run(rows)

        # Sanity: data round-tripped as list[dict], not a DataFrame.
        assert isinstance(result.data, list)
        out = result.report()
        assert "Pipeline Run Report" in out

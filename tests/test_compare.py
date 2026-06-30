"""Tests for ``accrue.compare()`` — diffing two ``PipelineResult`` objects."""

from __future__ import annotations

import dataclasses
import warnings

import pandas as pd
import pytest

from accrue.core.exceptions import RowError
from accrue.pipeline.compare import (
    ComparisonResult,
    _align_rows,
    _cells_equal,
    compare,
)
from accrue.pipeline.pipeline import PipelineResult
from accrue.schemas.base import CostSummary, StepUsage

# -- helpers ------------------------------------------------------------------


def _result(
    data,
    field_specs=None,
    errors=None,
    cost=None,
    pipeline_elapsed_seconds=0.0,
) -> PipelineResult:
    return PipelineResult(
        data=data,
        field_specs=field_specs or {},
        errors=errors or [],
        cost=cost or CostSummary(),
        pipeline_elapsed_seconds=pipeline_elapsed_seconds,
    )


# -- _align_rows ---------------------------------------------------------------


class TestAlignRows:
    def test_both_default_range_index_same_length(self):
        df_a = pd.DataFrame({"x": [1, 2, 3]})
        df_b = pd.DataFrame({"x": [4, 5, 6]})
        pos_a, pos_b, mode = _align_rows(df_a, df_b)
        assert mode == "positional"
        assert pos_a == [0, 1, 2]
        assert pos_b == [0, 1, 2]

    def test_both_default_range_index_different_length_warns(self):
        df_a = pd.DataFrame({"x": [1, 2, 3, 4]})
        df_b = pd.DataFrame({"x": [4, 5]})
        with pytest.warns(UserWarning, match="row counts differ"):
            pos_a, pos_b, mode = _align_rows(df_a, df_b)
        assert mode == "positional"
        assert pos_a == [0, 1]
        assert pos_b == [0, 1]

    def test_both_real_unique_index_aligns_by_label(self):
        df_a = pd.DataFrame({"x": [1, 2, 3]}, index=["r1", "r2", "r3"])
        df_b = pd.DataFrame({"x": [9, 8, 7]}, index=["r3", "r1", "r2"])
        pos_a, pos_b, mode = _align_rows(df_a, df_b)
        assert mode == "label"
        # Order follows df_a's index order: r1, r2, r3.
        assert [df_a.index[i] for i in pos_a] == ["r1", "r2", "r3"]
        assert [df_b.index[i] for i in pos_b] == ["r1", "r2", "r3"]

    def test_real_index_partial_overlap_warns_and_compares_intersection(self):
        df_a = pd.DataFrame({"x": [1, 2, 3]}, index=["r1", "r2", "r3"])
        df_b = pd.DataFrame({"x": [9, 8]}, index=["r2", "r4"])
        with pytest.warns(UserWarning, match="no matching index label"):
            pos_a, pos_b, mode = _align_rows(df_a, df_b)
        assert mode == "label"
        assert [df_a.index[i] for i in pos_a] == ["r2"]
        assert [df_b.index[i] for i in pos_b] == ["r2"]

    def test_duplicate_labels_falls_back_to_positional(self):
        df_a = pd.DataFrame({"x": [1, 2, 3]}, index=["r1", "r1", "r2"])
        df_b = pd.DataFrame({"x": [9, 8, 7]}, index=["r1", "r2", "r3"])
        with pytest.warns(UserWarning, match="non-unique"):
            pos_a, pos_b, mode = _align_rows(df_a, df_b)
        assert mode == "positional"
        assert pos_a == [0, 1, 2]
        assert pos_b == [0, 1, 2]

    def test_mixed_default_and_real_index_falls_back_to_positional(self):
        df_a = pd.DataFrame({"x": [1, 2, 3]})  # default RangeIndex
        df_b = pd.DataFrame({"x": [9, 8, 7]}, index=["r1", "r2", "r3"])
        with pytest.warns(UserWarning, match="custom index"):
            pos_a, pos_b, mode = _align_rows(df_a, df_b)
        assert mode == "positional"
        assert pos_a == [0, 1, 2]
        assert pos_b == [0, 1, 2]


# -- _cells_equal ---------------------------------------------------------------


class TestCellsEqual:
    def test_both_nan_is_equal(self):
        assert _cells_equal(float("nan"), float("nan")) is True

    def test_none_and_nan_is_equal(self):
        assert _cells_equal(None, float("nan")) is True

    def test_null_vs_value_is_not_equal(self):
        assert _cells_equal(None, "x") is False
        assert _cells_equal("x", None) is False

    def test_plain_scalars(self):
        assert _cells_equal("a", "a") is True
        assert _cells_equal("a", "b") is False
        assert _cells_equal(1, 1) is True
        assert _cells_equal(1, 2) is False

    def test_list_cells_do_not_raise(self):
        assert _cells_equal(["a", "b"], ["a", "b"]) is True
        assert _cells_equal(["a", "b"], ["a", "c"]) is False

    def test_dict_cells_do_not_raise(self):
        assert _cells_equal({"a": 1}, {"a": 1}) is True
        assert _cells_equal({"a": 1}, {"a": 2}) is False

    def test_dict_key_order_does_not_matter(self):
        assert _cells_equal({"a": 1, "b": 2}, {"b": 2, "a": 1}) is True


# -- compare() / ComparisonResult -----------------------------------------------


class TestCompare:
    def test_returns_comparison_result_with_overlapping_fields(self):
        df_a = pd.DataFrame({"category": ["A", "B"], "score": [1, 2]})
        df_b = pd.DataFrame({"category": ["A", "C"], "score": [1, 3]})
        a = _result(df_a, {"category": {"enum": ["A", "B", "C"]}, "score": {"type": "Number"}})
        b = _result(df_b, {"category": {"enum": ["A", "B", "C"]}, "score": {"type": "Number"}})

        diff = compare(a, b)
        assert isinstance(diff, ComparisonResult)
        assert diff.fields == ["category", "score"]
        assert diff.label_a == "A"
        assert diff.label_b == "B"
        assert len(diff.aligned_a) == 2
        assert len(diff.aligned_b) == 2

    def test_custom_labels(self):
        df = pd.DataFrame({"x": [1]})
        a, b = _result(df, {"x": {}}), _result(df, {"x": {}})
        diff = compare(a, b, label_a="v1", label_b="v2")
        assert diff.label_a == "v1"
        assert diff.label_b == "v2"

    def test_schema_mismatch_warns_and_compares_overlap_only(self):
        df_a = pd.DataFrame({"category": ["A"], "extra_a": [1]})
        df_b = pd.DataFrame({"category": ["A"], "extra_b": [2]})
        a = _result(df_a, {"category": {}, "extra_a": {}})
        b = _result(df_b, {"category": {}, "extra_b": {}})

        with pytest.warns(UserWarning, match="field-spec schemas differ"):
            diff = compare(a, b)
        # extra_a/extra_b aren't columns on both sides, so they're dropped
        # even though the union-of-spec-keys check is what fired the warning.
        assert diff.fields == ["category"]

    def test_field_only_in_one_field_spec_but_column_in_both_is_included(self):
        # b's pipeline didn't declare a spec for "tag" (e.g. it came from a
        # FunctionStep with no FieldSpec), but the column is present on both
        # sides, so it's still compared, using a's spec for classification.
        df_a = pd.DataFrame({"tag": ["x"]})
        df_b = pd.DataFrame({"tag": ["y"]})
        a = _result(df_a, {"tag": {"type": "String"}})
        b = _result(df_b, {})

        diff = compare(a, b)
        assert diff.fields == ["tag"]
        assert diff.field_specs["tag"] == {"type": "String"}

    def test_no_schema_mismatch_warning_when_specs_match(self):
        df = pd.DataFrame({"x": [1]})
        a, b = _result(df, {"x": {}}), _result(df, {"x": {}})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compare(a, b)

    def test_identical_runs_with_null_cells_report_zero_changes(self):
        df_a = pd.DataFrame({"score": [1.0, None, 3.0], "category": ["A", None, "C"]})
        df_b = df_a.copy()
        a = _result(df_a, {"score": {"type": "Number"}, "category": {"enum": ["A", "C"]}})
        b = _result(df_b, {"score": {"type": "Number"}, "category": {"enum": ["A", "C"]}})

        diff = compare(a, b)
        for f in diff.fields:
            col_a = diff.aligned_a[f].tolist()
            col_b = diff.aligned_b[f].tolist()
            assert all(_cells_equal(x, y) for x, y in zip(col_a, col_b))

    def test_json_and_list_field_cells_do_not_raise(self):
        df_a = pd.DataFrame(
            {
                "tags": [["a", "b"], ["c"]],
                "meta": [{"k": 1}, {"k": 2}],
            }
        )
        df_b = pd.DataFrame(
            {
                "tags": [["a", "b"], ["d"]],
                "meta": [{"k": 1}, {"k": 3}],
            }
        )
        a = _result(df_a, {"tags": {"type": "List[String]"}, "meta": {"type": "JSON"}})
        b = _result(df_b, {"tags": {"type": "List[String]"}, "meta": {"type": "JSON"}})

        diff = compare(a, b)
        # Must not raise comparing these columns cell-by-cell.
        for f in diff.fields:
            col_a = diff.aligned_a[f].tolist()
            col_b = diff.aligned_b[f].tolist()
            results = [_cells_equal(x, y) for x, y in zip(col_a, col_b)]
        assert results == [True, False]

    def test_list_input_normalised_to_dataframe(self):
        rows_a = [{"x": 1}, {"x": 2}]
        rows_b = [{"x": 1}, {"x": 3}]
        a = _result(rows_a, {"x": {"type": "Number"}})
        b = _result(rows_b, {"x": {"type": "Number"}})
        diff = compare(a, b)
        assert isinstance(diff.aligned_a, pd.DataFrame)
        assert diff.aligned_a["x"].tolist() == [1, 2]


# -- changed_rows() -------------------------------------------------------------


class TestChangedRows:
    def test_no_changes_returns_empty_dataframe(self):
        df = pd.DataFrame({"category": ["A", "B"], "score": [1, 2]})
        a = _result(df, {"category": {}, "score": {"type": "Number"}})
        b = _result(df.copy(), {"category": {}, "score": {"type": "Number"}})
        diff = compare(a, b)
        out = diff.changed_rows()
        assert out.empty
        assert list(out.columns) == ["category_A", "category_B", "score_A", "score_B"]

    def test_flags_rows_with_any_field_difference(self):
        df_a = pd.DataFrame({"category": ["A", "B", "C"], "score": [1, 2, 3]})
        df_b = pd.DataFrame({"category": ["A", "X", "C"], "score": [1, 2, 9]})
        a = _result(df_a, {"category": {}, "score": {"type": "Number"}})
        b = _result(df_b, {"category": {}, "score": {"type": "Number"}})
        diff = compare(a, b)
        out = diff.changed_rows()
        assert len(out) == 2
        assert out["category_A"].tolist() == ["B", "C"]
        assert out["category_B"].tolist() == ["X", "C"]
        assert out["score_A"].tolist() == [2, 3]
        assert out["score_B"].tolist() == [2, 9]

    def test_field_filter_only_considers_that_field(self):
        df_a = pd.DataFrame({"category": ["A", "B"], "score": [1, 2]})
        df_b = pd.DataFrame({"category": ["A", "X"], "score": [9, 2]})
        a = _result(df_a, {"category": {}, "score": {"type": "Number"}})
        b = _result(df_b, {"category": {}, "score": {"type": "Number"}})
        diff = compare(a, b)

        by_category = diff.changed_rows("category")
        assert list(by_category.columns) == ["category_A", "category_B"]
        assert len(by_category) == 1
        assert by_category["category_A"].tolist() == ["B"]

        by_score = diff.changed_rows("score")
        assert list(by_score.columns) == ["score_A", "score_B"]
        assert len(by_score) == 1
        assert by_score["score_A"].tolist() == [1]

    def test_unknown_field_raises_key_error(self):
        df = pd.DataFrame({"x": [1]})
        a, b = _result(df, {"x": {}}), _result(df.copy(), {"x": {}})
        diff = compare(a, b)
        with pytest.raises(KeyError):
            diff.changed_rows("nope")

    def test_error_on_one_side_only_counts_as_changed(self):
        # Both sides have the same value for the row, but A errored on
        # row 0 -- that row must still be flagged as a change.
        df_a = pd.DataFrame({"x": [None, "same"]})
        df_b = pd.DataFrame({"x": [None, "same"]})
        err = RowError(row_index=0, step_name="s", error=ValueError("x"))
        a = _result(df_a, {"x": {}}, errors=[err])
        b = _result(df_b, {"x": {}})
        diff = compare(a, b)
        out = diff.changed_rows()
        assert len(out) == 1

    def test_identical_with_errors_on_both_sides_same_row_not_flagged(self):
        df_a = pd.DataFrame({"x": [None, "same"]})
        df_b = pd.DataFrame({"x": [None, "same"]})
        err_a = RowError(row_index=0, step_name="s", error=ValueError("x"))
        err_b = RowError(row_index=0, step_name="s", error=ValueError("y"))
        a = _result(df_a, {"x": {}}, errors=[err_a])
        b = _result(df_b, {"x": {}}, errors=[err_b])
        diff = compare(a, b)
        out = diff.changed_rows()
        assert out.empty


# -- distribution_shift() --------------------------------------------------------


class TestDistributionShift:
    def test_enum_field_frequency_table(self):
        df_a = pd.DataFrame({"category": ["AI", "AI", "Other", "Other", "Other"]})
        df_b = pd.DataFrame({"category": ["AI", "AI", "AI", "Other", "Other"]})
        a = _result(df_a, {"category": {"enum": ["AI", "Other"]}})
        b = _result(df_b, {"category": {"enum": ["AI", "Other"]}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["category"]

        assert shift.kind == "enum"
        assert shift.changed == 1
        assert shift.freq_a == {"AI": 0.4, "Other": 0.6}
        assert shift.freq_b == {"AI": 0.6, "Other": 0.4}

    def test_enum_field_with_container_values_does_not_raise(self):
        # Defensive: enum classification keys off the spec, not the dtype,
        # but value_counts() must still tolerate non-scalar cells.
        df_a = pd.DataFrame({"tags": [["a"], ["a"], ["b"]]})
        df_b = pd.DataFrame({"tags": [["a"], ["b"], ["b"]]})
        a = _result(df_a, {"tags": {"enum": ["a", "b"]}})
        b = _result(df_b, {"tags": {"enum": ["a", "b"]}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["tags"]
        assert shift.kind == "enum"
        assert sum(shift.freq_a.values()) == pytest.approx(1.0)

    def test_numeric_field_mean_delta(self):
        df_a = pd.DataFrame({"score": [10, 20, 30]})
        df_b = pd.DataFrame({"score": [10, 25, 40]})
        a = _result(df_a, {"score": {"type": "Number"}})
        b = _result(df_b, {"score": {"type": "Number"}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["score"]

        assert shift.kind == "numeric"
        assert shift.changed == 2
        assert shift.mean_a == pytest.approx(20.0)
        assert shift.mean_b == pytest.approx(25.0)
        assert shift.mean_delta == pytest.approx(5.0)
        assert shift.std_a is not None and shift.std_b is not None

    def test_numeric_field_non_numeric_values_coerced_to_nan(self):
        df_a = pd.DataFrame({"score": ["10", "oops", "30"]})
        df_b = pd.DataFrame({"score": ["10", "20", "30"]})
        a = _result(df_a, {"score": {"type": "Number"}})
        b = _result(df_b, {"score": {"type": "Number"}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["score"]
        assert shift.mean_a == pytest.approx(20.0)  # (10 + 30) / 2, "oops" dropped
        assert shift.mean_b == pytest.approx(20.0)

    def test_numeric_field_all_non_numeric_yields_none_mean(self):
        df = pd.DataFrame({"score": ["nope", "also-nope"]})
        a = _result(df, {"score": {"type": "Number"}})
        b = _result(df.copy(), {"score": {"type": "Number"}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["score"]
        assert shift.mean_a is None
        assert shift.mean_delta is None

    def test_string_field_differs_count_and_token_length_delta(self):
        df_a = pd.DataFrame({"summary": ["short text", "another one here"]})
        df_b = pd.DataFrame({"summary": ["short text", "a much longer summary here now"]})
        a = _result(df_a, {"summary": {"type": "String"}})
        b = _result(df_b, {"summary": {"type": "String"}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["summary"]

        assert shift.kind == "text"
        assert shift.changed == 1
        assert shift.avg_len_tokens_a is not None
        assert shift.avg_len_tokens_b is not None
        assert shift.len_delta_tokens > 0

    def test_boolean_field_differs_count_only_no_length_delta(self):
        df_a = pd.DataFrame({"flag": [True, False]})
        df_b = pd.DataFrame({"flag": [True, True]})
        a = _result(df_a, {"flag": {"type": "Boolean"}})
        b = _result(df_b, {"flag": {"type": "Boolean"}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["flag"]

        assert shift.kind == "text"
        assert shift.changed == 1
        assert shift.avg_len_tokens_a is None
        assert shift.len_delta_tokens is None

    def test_json_field_differs_count_only_does_not_raise(self):
        df_a = pd.DataFrame({"meta": [{"k": 1}, {"k": 2}]})
        df_b = pd.DataFrame({"meta": [{"k": 1}, {"k": 9}]})
        a = _result(df_a, {"meta": {"type": "JSON"}})
        b = _result(df_b, {"meta": {"type": "JSON"}})
        diff = compare(a, b)
        shift = diff.distribution_shift()["meta"]
        assert shift.kind == "text"
        assert shift.changed == 1


# -- cost_delta() -----------------------------------------------------------------


class TestCostDelta:
    def test_token_deltas(self):
        df = pd.DataFrame({"x": [1]})
        cost_a = CostSummary(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_tokens=150,
            steps={"s": StepUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)},
        )
        cost_b = CostSummary(
            total_prompt_tokens=112,
            total_completion_tokens=59,
            total_tokens=171,
            steps={"s": StepUsage(prompt_tokens=112, completion_tokens=59, total_tokens=171)},
        )
        a = _result(df, {"x": {}}, cost=cost_a)
        b = _result(df.copy(), {"x": {}}, cost=cost_b)
        diff = compare(a, b)
        delta = diff.cost_delta()

        assert delta.prompt_tokens_delta == 12
        assert delta.completion_tokens_delta == 9
        assert delta.total_tokens_delta == 21

    def test_cache_hit_rate_is_aggregated_across_steps(self):
        df = pd.DataFrame({"x": [1]})
        cost_a = CostSummary(
            steps={
                "s1": StepUsage(cache_hits=90, cache_misses=10),
                "s2": StepUsage(cache_hits=0, cache_misses=100),
            }
        )
        cost_b = CostSummary(steps={"s1": StepUsage(cache_hits=0, cache_misses=100)})
        a = _result(df, {"x": {}}, cost=cost_a)
        b = _result(df.copy(), {"x": {}}, cost=cost_b)
        diff = compare(a, b)
        delta = diff.cost_delta()

        # 90 hits / 200 total = 0.45 on A; 0 on B.
        assert delta.cache_hit_rate_a == pytest.approx(0.45)
        assert delta.cache_hit_rate_b == pytest.approx(0.0)
        assert delta.cache_hit_rate_delta == pytest.approx(-0.45)

    def test_cache_hit_rate_guards_divide_by_zero(self):
        df = pd.DataFrame({"x": [1]})
        a = _result(df, {"x": {}}, cost=CostSummary())
        b = _result(df.copy(), {"x": {}}, cost=CostSummary())
        diff = compare(a, b)
        delta = diff.cost_delta()
        assert delta.cache_hit_rate_a == 0.0
        assert delta.cache_hit_rate_b == 0.0
        assert delta.cache_hit_rate_delta == 0.0

    def test_latency_delta(self):
        df = pd.DataFrame({"x": [1]})
        a = _result(df, {"x": {}}, pipeline_elapsed_seconds=10.0)
        b = _result(df.copy(), {"x": {}}, pipeline_elapsed_seconds=14.5)
        diff = compare(a, b)
        delta = diff.cost_delta()
        assert delta.elapsed_seconds_a == 10.0
        assert delta.elapsed_seconds_b == 14.5
        assert delta.elapsed_seconds_delta == pytest.approx(4.5)

    def test_no_pricing_fields(self):
        df = pd.DataFrame({"x": [1]})
        a, b = _result(df, {"x": {}}), _result(df.copy(), {"x": {}})
        diff = compare(a, b)
        delta = diff.cost_delta()
        field_names = {f.name for f in dataclasses.fields(delta)}
        assert not any("dollar" in n or n.startswith("$") or "price" in n for n in field_names)


# -- summary() --------------------------------------------------------------------


class TestSummary:
    def test_identical_runs_short_circuit_to_no_differences(self):
        df = pd.DataFrame({"category": ["A", "B"]})
        a = _result(df, {"category": {"enum": ["A", "B"]}})
        b = _result(df.copy(), {"category": {"enum": ["A", "B"]}})
        diff = compare(a, b, label_a="v1", label_b="v2")
        out = diff.summary()

        assert "No differences detected" in out
        assert "0 rows changed" in out
        assert "2 rows identical" in out
        # No per-field/cost sections on the early-return path.
        assert "Per-field churn" not in out
        assert "Cost delta" not in out

    def test_header_shows_labels_and_models_not_prompt_rev(self):
        df = pd.DataFrame({"score": [1, 2]})
        cost_a = CostSummary(steps={"s": StepUsage(model="gpt-4.1-mini")})
        cost_b = CostSummary(steps={"s": StepUsage(model="gpt-4.1-mini")})
        a = _result(df, {"score": {"type": "Number"}}, cost=cost_a)
        b = _result(pd.DataFrame({"score": [1, 5]}), {"score": {"type": "Number"}}, cost=cost_b)
        diff = compare(a, b, label_a="v1", label_b="v2")
        out = diff.summary()

        assert "v1" in out and "v2" in out
        assert "gpt-4.1-mini" in out
        assert "prompt-rev" not in out

    def test_includes_totals_and_per_field_churn_and_cost_delta(self):
        df_a = pd.DataFrame({"category": ["AI", "Other"], "score": [1, 2]})
        df_b = pd.DataFrame({"category": ["AI", "AI"], "score": [1, 9]})
        a = _result(
            df_a,
            {"category": {"enum": ["AI", "Other"]}, "score": {"type": "Number"}},
            cost=CostSummary(total_tokens=100),
        )
        b = _result(
            df_b,
            {"category": {"enum": ["AI", "Other"]}, "score": {"type": "Number"}},
            cost=CostSummary(total_tokens=150),
        )
        diff = compare(a, b)
        out = diff.summary()

        assert "2 rows compared" in out
        assert "1 rows changed in at least one field" in out
        assert "Per-field churn" in out
        assert "category" in out
        assert "score" in out
        assert "Cost delta" in out

    def test_ends_with_newline(self):
        df = pd.DataFrame({"x": [1]})
        a, b = _result(df, {"x": {}}), _result(df.copy(), {"x": {}})
        diff = compare(a, b)
        assert diff.summary().endswith("\n")

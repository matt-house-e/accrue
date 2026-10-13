"""Tests for ``accrue.merge_runs()`` — N runs merged into one answer per cell."""

from __future__ import annotations

import math
import warnings
from datetime import date

import pandas as pd
import pytest

from accrue import (
    Candidate,
    EnrichmentConfig,
    FunctionStep,
    MergedResult,
    Pipeline,
    merge_runs,
)
from accrue.core.exceptions import RowError
from accrue.pipeline.pipeline import PipelineResult
from accrue.schemas.base import CostSummary, StepUsage

ENUM = {"prompt": "p", "type": "String", "enum": ["low", "high"]}
TEXT = {"prompt": "p", "type": "String"}
NUM = {"prompt": "p", "type": "Number"}


def _run(rows, specs=None, errors=None, cost=None, elapsed=0.0) -> PipelineResult:
    return PipelineResult(
        data=pd.DataFrame(rows),
        field_specs=specs if specs is not None else {"tier": ENUM},
        errors=errors or [],
        cost=cost or CostSummary(),
        pipeline_elapsed_seconds=elapsed,
    )


def _tiers(*values):
    return _run([{"name": f"r{i}", "tier": v} for i, v in enumerate(values)])


def _cell(m: MergedResult, row, field="tier") -> dict:
    a = m.agreement
    return a[(a["row"] == row) & (a["field"] == field)].iloc[0].to_dict()


def _err(row, step="s"):
    return RowError(row_index=row, step_name=step, error=RuntimeError("boom"))


# -- voting --------------------------------------------------------------------


class TestVoting:
    def test_unanimous(self):
        m = merge_runs([_tiers("low", "high"), _tiers("low", "high")])
        assert m.data["tier"].tolist() == ["low", "high"]
        c = _cell(m, 0)
        assert (c["votes"], c["voters"], c["agreed"], c["needs_review"], c["basis"]) == (
            2,
            2,
            True,
            False,
            "unanimous",
        )
        assert m.needs_review.empty

    def test_two_of_three_majority_wins_without_review(self):
        m = merge_runs([_tiers("low"), _tiers("high"), _tiers("high")])
        c = _cell(m, 0)
        assert m.data["tier"].tolist() == ["high"]
        assert (c["votes"], c["voters"], c["agreed"], c["needs_review"], c["basis"]) == (
            2,
            3,
            False,
            False,
            "majority",
        )
        assert c["values"] == ("low", "high", "high")

    def test_two_runs_disagreeing_is_a_tie_flagged_with_run0_value(self):
        m = merge_runs([_tiers("low"), _tiers("high")])
        c = _cell(m, 0)
        assert m.data["tier"].tolist() == ["low"]
        assert c["needs_review"] and c["basis"] == "no_majority" and c["votes"] == 1
        assert len(m.needs_review) == 1

    def test_three_way_split_flagged(self):
        spec = {"tier": {"type": "Number"}}
        runs = [_run([{"tier": v}], specs=spec) for v in (1, 2, 3)]
        c = _cell(merge_runs(runs), 0)
        assert c["needs_review"] and c["value"] == 1

    def test_both_null_is_agreement_null_vs_value_is_not(self):
        runs = [_run([{"tier": None}, {"tier": None}]), _run([{"tier": math.nan}, {"tier": "low"}])]
        m = merge_runs(runs)
        assert _cell(m, 0)["agreed"]
        assert _cell(m, 1)["needs_review"]

    def test_numbers_compare_by_value(self):
        runs = [_run([{"n": 3}], specs={"n": NUM}), _run([{"n": 3.0}], specs={"n": NUM})]
        assert _cell(merge_runs(runs), 0, "n")["agreed"]


class TestDefaultRules:
    def test_free_text_keeps_run0_and_is_not_flagged(self):
        specs = {"summary": TEXT}
        runs = [
            _run([{"summary": "Acme makes rockets."}], specs=specs),
            _run([{"summary": "Acme builds rockets."}], specs=specs),
        ]
        m = merge_runs(runs)
        c = _cell(m, 0, "summary")
        assert c["value"] == "Acme makes rockets."
        assert (c["agreed"], c["needs_review"], c["basis"]) == (False, False, "first")
        assert m.rules == {"summary": "first"}

    def test_field_without_spec_is_voted(self):
        specs = {"score": {}}  # FunctionStep output
        runs = [_run([{"score": 1}], specs=specs), _run([{"score": 2}], specs=specs)]
        m = merge_runs(runs)
        assert _cell(m, 0, "score")["needs_review"]
        assert m.rules == {"score": "majority"}

    def test_rule_override_by_name(self):
        specs = {"summary": TEXT}
        runs = [_run([{"summary": "a"}], specs=specs), _run([{"summary": "b"}], specs=specs)]
        assert _cell(merge_runs(runs, resolve="majority"), 0, "summary")["needs_review"]
        m = merge_runs(runs, resolve={"summary": "majority"})
        assert _cell(m, 0, "summary")["needs_review"]

    def test_unknown_rule_name_raises(self):
        with pytest.raises(ValueError, match="unknown rule"):
            merge_runs([_tiers("low"), _tiers("low")], resolve={"tier": "vote"})


# -- key= and resolve= -----------------------------------------------------------


def _parse(v):
    for fmt in ("%Y-%m-%d", "%d %b %Y"):
        try:
            return pd.to_datetime(v, format=fmt).date()
        except (ValueError, TypeError):
            pass
    return v


class TestKey:
    def test_key_normalises_before_comparing_and_keeps_raw_value(self):
        specs = {"start": {"type": "Date"}}
        runs = [
            _run([{"start": "2026-01-01"}], specs=specs),
            _run([{"start": "1 Jan 2026"}], specs=specs),
        ]
        m = merge_runs(runs, key={"start": _parse})
        c = _cell(m, 0, "start")
        assert c["agreed"] and c["value"] == "2026-01-01"
        assert _parse("1 Jan 2026") == date(2026, 1, 1)

    def test_single_callable_key_applies_to_every_field(self):
        runs = [_tiers("LOW"), _tiers(" low ")]
        assert _cell(merge_runs(runs, key=lambda v: str(v).strip().lower()), 0)["agreed"]

    def test_key_for_unmerged_field_raises(self):
        with pytest.raises(ValueError, match="not merged"):
            merge_runs([_tiers("low"), _tiers("low")], key={"tierr": str})


RANK = {"verified": 2, "unverified": 1}


def better_evidence(field, candidates):
    ranks = [RANK.get(c.row.get(f"{field}_evidence"), 0) for c in candidates]
    top = [c for c, r in zip(candidates, ranks) if r == max(ranks)]
    return top[0] if len(top) == 1 else None


class TestResolver:
    def _runs(self, ev_a, ev_b):
        return [
            _run([{"tier": "low", "tier_evidence": ev_a}]),
            _run([{"tier": "high", "tier_evidence": ev_b}]),
        ]

    def test_resolver_picks_from_companion_column(self):
        m = merge_runs(self._runs("unverified", "verified"), resolve={"tier": better_evidence})
        c = _cell(m, 0)
        assert m.data["tier"].tolist() == ["high"]
        assert m.data["tier_evidence"].tolist() == ["unverified"]  # unmerged: run 0's
        assert (c["basis"], c["needs_review"], c["votes"]) == ("resolver", False, 1)
        assert m.rules == {"tier": "resolver"}

    def test_resolver_declining_falls_back_to_default_rule(self):
        m = merge_runs(self._runs("verified", "verified"), resolve=better_evidence)
        c = _cell(m, 0)
        assert c["basis"] == "no_majority" and c["needs_review"]

    def test_resolver_only_called_on_disagreement(self):
        calls = []

        def spy(field, candidates):
            calls.append(field)
            return None

        merge_runs([_tiers("low", "low"), _tiers("low", "high")], resolve=spy)
        assert calls == ["tier"]

    def test_resolver_must_return_a_given_candidate(self):
        def bad(field, candidates):
            return Candidate(run=0, value="low", row={})

        with pytest.raises(TypeError, match="one of the candidates"):
            merge_runs(self._runs("a", "b"), resolve={"tier": bad})


# -- errors, alignment, cost ----------------------------------------------------


class TestErroredRows:
    def test_row_errored_in_one_of_two_runs_is_a_single_read(self):
        b = _tiers(None, "high")
        b.errors = [_err(0)]
        m = merge_runs([_tiers("low", "high"), b])
        c = _cell(m, 0)
        assert m.errored_rows == {0: [1]}
        assert (c["value"], c["voters"], c["basis"], c["needs_review"]) == (
            "low",
            1,
            "single_read",
            True,
        )
        assert c["values"] == ("low", None)
        assert not _cell(m, 1)["needs_review"]

    def test_errored_run_does_not_vote(self):
        c_run = _tiers("low")
        c_run.errors = [_err(0)]
        c = _cell(merge_runs([_tiers("high"), _tiers("high"), c_run]), 0)
        assert (c["votes"], c["voters"], c["agreed"]) == (2, 2, True)

    def test_row_errored_everywhere(self):
        a, b = _tiers(None), _tiers(None)
        a.errors, b.errors = [_err(0)], [_err(0)]
        c = _cell(merge_runs([a, b]), 0)
        assert (c["basis"], c["value"], c["voters"], c["needs_review"]) == (
            "no_reads",
            None,
            0,
            True,
        )


class TestAlignment:
    def test_aligns_by_label(self):
        a = _run([{"tier": "low"}, {"tier": "high"}])
        b = _run([{"tier": "high"}, {"tier": "low"}])
        a.data.index = ["x", "y"]
        b.data.index = ["y", "x"]
        m = merge_runs([a, b])
        assert m.agreement["agreed"].all()
        assert list(m.data.index) == ["x", "y"]

    def test_rows_missing_from_a_run_are_dropped(self):
        a = _run([{"tier": "low"}, {"tier": "high"}, {"tier": "low"}])
        b = _run([{"tier": "high"}, {"tier": "low"}])
        a.data.index, b.data.index = ["x", "y", "z"], ["y", "x"]
        with pytest.warns(UserWarning, match="merge_runs\\(\\)"):
            m = merge_runs([a, b])
        assert list(m.data.index) == ["x", "y"]

    def test_list_of_dict_data(self):
        a = PipelineResult(data=[{"tier": "low"}], field_specs={"tier": ENUM})
        b = PipelineResult(data=[{"tier": "low"}], field_specs={"tier": ENUM})
        assert merge_runs([a, b]).data["tier"].tolist() == ["low"]

    def test_explicit_fields_must_exist(self):
        with pytest.raises(KeyError, match="run 1"):
            merge_runs([_run([{"tier": "a", "x": 1}]), _tiers("a")], fields=["tier", "x"])


def _cost(prompt, completion, hits=0, misses=1):
    step = StepUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        rows_processed=hits + misses,
        cache_hits=hits,
        cache_misses=misses,
        model="m",
    )
    return CostSummary(
        total_prompt_tokens=prompt,
        total_completion_tokens=completion,
        total_tokens=prompt + completion,
        steps={"extract": step},
    )


class TestCostAndWarnings:
    def test_cost_and_time_summed_across_runs(self):
        a = _run([{"tier": "low"}], cost=_cost(100, 10), elapsed=2.0)
        b = _run([{"tier": "high"}], cost=_cost(120, 20), elapsed=3.0)
        m = merge_runs([a, b])
        assert m.cost.total_tokens == 250
        assert m.cost.steps["extract"].prompt_tokens == 220
        assert m.cost.steps["extract"].rows_processed == 2
        assert m.cost.steps["extract"].model == "m"
        assert m.elapsed_seconds == 5.0
        assert a.cost.steps["extract"].prompt_tokens == 100  # inputs untouched

    def test_batch_ids_from_every_run(self):
        # A realtime run first, then two batch runs (#169).
        costs = [_cost(1, 1), _cost(1, 1), _cost(1, 1)]
        costs[1].steps["extract"].batch_id = "a-1"
        costs[1].steps["extract"].batch_ids = ["a-1", "a-2"]
        costs[2].steps["extract"].batch_id = "b-1"
        costs[2].steps["extract"].batch_ids = ["b-1"]
        m = merge_runs([_run([{"tier": "low"}], cost=c) for c in costs])
        usage = m.cost.steps["extract"]
        assert usage.batch_ids == ["a-1", "a-2", "b-1"]
        assert usage.batch_id == "a-1"  # still the first one
        assert costs[0].steps["extract"].batch_ids == []  # inputs untouched

    def test_warns_when_a_run_looks_like_a_cache_replay(self):
        a = _run([{"tier": "low"}], cost=_cost(100, 10))
        b = _run([{"tier": "low"}], cost=_cost(0, 0, hits=1, misses=0))
        with pytest.warns(UserWarning, match="replay of run 0"):
            merge_runs([a, b])

    def test_no_replay_warning_when_cached_run_differs(self):
        a = _run([{"tier": "low"}], cost=_cost(100, 10))
        b = _run([{"tier": "high"}], cost=_cost(0, 0, hits=1, misses=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            merge_runs([a, b])

    def test_needs_two_runs(self):
        with pytest.raises(ValueError, match="at least two"):
            merge_runs([_tiers("low")])


def test_summary_lists_fields():
    text = merge_runs([_tiers("low", "low"), _tiers("low", "high")]).summary()
    assert "2 runs, 2 rows, 2 cells" in text
    assert "1 cells need review" in text
    assert "| `tier` | majority | 50% | 1 |" in text


def test_derived_fields_recomputed_by_rerunning_function_steps():
    """The documented recipe: merge the read, then rerun the derived step on it."""
    runs = [_tiers("low", "high"), _tiers("high", "high"), _tiers("high", "low")]
    m = merge_runs(runs)
    derive = Pipeline(
        [
            FunctionStep(
                "price",
                fn=lambda ctx: {"price": 10 if ctx.row["tier"] == "high" else 1},
                fields=["price"],
            )
        ]
    )
    out = derive.run(m.data, config=EnrichmentConfig(enable_caching=False))
    assert out.data["price"].tolist() == [10, 10]


def test_per_run_cache_dir_keeps_runs_independent(tmp_path):
    """The documented recipe: one cache_dir per run, else run 2 replays run 1."""
    reads = iter(["low", "high", "low"])
    read = Pipeline([FunctionStep("read", fn=lambda ctx: {"tier": next(reads)}, fields=["tier"])])
    rows = [{"name": "r0"}]

    def config(d):
        return EnrichmentConfig(cache_dir=str(tmp_path / d), checkpoint_dir=str(tmp_path / d))

    shared = [read.run(rows, config=config("shared")) for _ in range(2)]
    with pytest.warns(UserWarning, match="replay"):
        replayed = merge_runs(shared)
    assert replayed.agreement["agreed"].all()  # one read counted twice

    independent = [read.run(rows, config=config(f"run-{i}")) for i in range(2)]
    assert merge_runs(independent).agreement["values"][0] == ("high", "low")
    # Rerunning the merge script replays each run from its own cache, for free.
    again = [read.run(rows, config=config(f"run-{i}")) for i in range(2)]
    assert [r.data[0]["tier"] for r in again] == ["high", "low"]

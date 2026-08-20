"""Failed-only resume — ``Pipeline.retry_failed()`` / ``retry_failed_async()``.  Issue #129.

Deterministic function steps only: no LLM providers, no network, no API keys.
Every step records the row indices it was invoked with, so "pay for exactly
the failed cells" is asserted by counting invocations rather than by trusting
a log line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accrue import EnrichmentConfig, FunctionStep, Pipeline
from accrue.core.checkpoint import (
    DEFAULT_CATEGORY,
    CheckpointManager,
    derive_data_identifier,
)
from accrue.core.exceptions import PipelineError

# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class Tracker:
    """Records every step invocation by row index, and which rows should fail."""

    def __init__(self, failing: set[int] | None = None) -> None:
        self.calls: dict[str, list[int]] = {}
        self.failing: set[int] = set(failing or ())

    def record(self, step: str, row_index: int) -> None:
        self.calls.setdefault(step, []).append(row_index)

    def rows(self, step: str) -> list[int]:
        return sorted(self.calls.get(step, []))

    def count(self, step: str) -> int:
        return len(self.calls.get(step, []))

    def reset(self) -> None:
        self.calls = {}


def _pipeline(tracker: Tracker) -> Pipeline:
    """normalize -> score -> flag.  ``score`` fails for ``tracker.failing`` rows."""

    async def normalize(ctx):
        tracker.record("normalize", ctx.row["idx"])
        return {"name_upper": str(ctx.row["company"]).upper()}

    async def score(ctx):
        idx = ctx.row["idx"]
        tracker.record("score", idx)
        if idx in tracker.failing:
            raise ValueError(f"cannot score row {idx}")
        return {"score": idx * 10}

    async def flag(ctx):
        tracker.record("flag", ctx.row["idx"])
        return {"flagged": (ctx.prior_results.get("score") or 0) > 20}

    return Pipeline(
        [
            FunctionStep("normalize", normalize, fields=["name_upper"]),
            FunctionStep("score", score, fields=["score"], depends_on=["normalize"]),
            FunctionStep("flag", flag, fields=["flagged"], depends_on=["score"]),
        ]
    )


def _rows(n: int = 5) -> list[dict]:
    return [{"company": f"company-{i:02d}", "idx": i} for i in range(n)]


def _config(tmp_path: Path, **overrides) -> EnrichmentConfig:
    defaults = dict(
        enable_checkpointing=True,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        # Caching off so an un-invoked row can only have come from the
        # checkpoint — a cache hit would be indistinguishable otherwise.
        enable_caching=False,
        enable_progress_bar=False,
        max_workers=2,
    )
    defaults.update(overrides)
    return EnrichmentConfig(**defaults)


def _checkpoint_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "checkpoints").glob("*_checkpoint.json"))


def _load_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------
# The acceptance case
# --------------------------------------------------------------------------


class TestRetryReExecutesOnlyFailedCells:
    def test_retry_invokes_exactly_the_failed_cells(self, tmp_path):
        """N of M rows fail on a middle step; the retry costs exactly N calls."""
        tracker = Tracker(failing={1, 3})
        pipeline = _pipeline(tracker)
        data = _rows(5)
        config = _config(tmp_path)

        first = pipeline.run(data, config=config)

        assert len(first.errors) == 2
        assert {e.row_index for e in first.errors} == {1, 3}
        assert tracker.count("normalize") == 5
        assert tracker.count("score") == 5
        assert tracker.count("flag") == 5
        # The checkpoint survives a run that left failures — it is what the
        # retry reads.
        assert len(_checkpoint_files(tmp_path)) == 1

        tracker.reset()
        tracker.failing.clear()  # the transient failure has cleared

        healed = pipeline.retry_failed(data, config=config)

        # Exactly N step executions, and only for the failed rows.
        assert tracker.count("score") == 2
        assert tracker.rows("score") == [1, 3]
        # Rows that succeeded were served from the checkpoint, un-invoked.
        assert tracker.count("normalize") == 0
        assert tracker.count("flag") == 0

        assert healed.errors == []
        assert [row["score"] for row in healed.data] == [0, 10, 20, 30, 40]
        # Untouched cells kept their first-run values.
        assert [row["name_upper"] for row in healed.data] == [f"COMPANY-{i:02d}" for i in range(5)]
        # Only the failed cells are re-run: a downstream step is NOT cascaded,
        # so row 3's `flagged` still reflects the score it saw last time.
        assert healed.data[3]["flagged"] is False
        # Nothing left to heal -> the checkpoint is cleaned up.
        assert _checkpoint_files(tmp_path) == []

    async def test_retry_failed_async_is_the_same_run(self, tmp_path):
        tracker = Tracker(failing={2})
        pipeline = _pipeline(tracker)
        data = _rows(4)
        config = _config(tmp_path)

        first = await pipeline.run_async(data, config=config)
        assert len(first.errors) == 1

        tracker.reset()
        tracker.failing.clear()
        healed = await pipeline.retry_failed_async(data, config=config)

        assert tracker.rows("score") == [2]
        assert tracker.count("normalize") == 0
        assert healed.errors == []
        assert healed.data[2]["score"] == 20

    async def test_sync_entrypoint_refuses_a_running_loop(self, tmp_path):
        pipeline = _pipeline(Tracker())
        with pytest.raises(RuntimeError, match="retry_failed_async"):
            pipeline.retry_failed(_rows(2), config=_config(tmp_path))

    def test_dataframe_input_round_trips(self, tmp_path):
        pd = pytest.importorskip("pandas")
        tracker = Tracker(failing={1})
        pipeline = _pipeline(tracker)
        df = pd.DataFrame(_rows(3))
        config = _config(tmp_path)

        first = pipeline.run(df, config=config)
        assert len(first.errors) == 1

        tracker.reset()
        tracker.failing.clear()
        healed = pipeline.retry_failed(df, config=config)

        assert tracker.rows("score") == [1]
        assert list(healed.data["score"]) == [0, 10, 20]


class TestAutoResumeSemantics:
    """Checkpointed no longer means done: a resumed run re-executes failed cells."""

    def test_a_plain_rerun_reruns_only_the_failed_cells(self, tmp_path):
        tracker = Tracker(failing={2})
        pipeline = _pipeline(tracker)
        data = _rows(5)
        config = _config(tmp_path)

        assert len(pipeline.run(data, config=config).errors) == 1

        tracker.reset()
        tracker.failing.clear()
        again = pipeline.run(data, config=config)

        assert tracker.rows("score") == [2], "only the errored cell re-runs"
        assert tracker.count("normalize") == 0, "succeeded cells are not re-invoked"
        assert tracker.count("flag") == 0
        assert again.errors == []
        assert again.data[2]["score"] == 20

    def test_the_enricher_runner_behaves_the_same(self, tmp_path):
        pd = pytest.importorskip("pandas")
        tracker = Tracker(failing={1})
        pipeline = _pipeline(tracker)
        df = pd.DataFrame(_rows(4))
        runner = pipeline.runner(_config(tmp_path))

        runner.run(df)
        assert tracker.count("score") == 4
        assert len(_checkpoint_files(tmp_path)) == 1

        tracker.reset()
        tracker.failing.clear()
        out = runner.run(df)

        assert tracker.rows("score") == [1]
        assert tracker.count("normalize") == 0
        assert list(out["score"]) == [0, 10, 20, 30]
        assert _checkpoint_files(tmp_path) == []


class TestPermanentFailure:
    def test_a_row_that_fails_again_surfaces_normally(self, tmp_path):
        """The second error is recorded, not swallowed, and stays retryable."""
        tracker = Tracker(failing={2})
        pipeline = _pipeline(tracker)
        data = _rows(4)
        config = _config(tmp_path)
        log = tmp_path / "run.jsonl"

        first = pipeline.run(data, config=config, run_log=log)
        assert len(first.errors) == 1

        tracker.reset()  # row 2 is still failing

        retry = pipeline.retry_failed(data, config=config, run_log=log)

        assert tracker.rows("score") == [2]
        assert len(retry.errors) == 1
        assert retry.errors[0].row_index == 2
        assert retry.errors[0].step_name == "score"
        assert isinstance(retry.errors[0].error, ValueError)
        assert retry.data[2]["score"] is None

        # The second failure is in the log, with its message.
        records = _load_log(log)
        segment = records[next(i for i, r in enumerate(records) if r["type"] == "retry_start") :]
        errored = [r for r in segment if r["type"] == "row_complete" and r["status"] == "error"]
        assert len(errored) == 1
        assert errored[0]["row"] == 2
        assert errored[0]["error"] == {"type": "ValueError", "msg": "cannot score row 2"}
        assert segment[-1]["type"] == "retry_end"
        assert segment[-1]["total_errors"] == 1

        # Still recorded as failed, so a third attempt can heal it.
        assert len(_checkpoint_files(tmp_path)) == 1
        tracker.reset()
        tracker.failing.clear()
        third = pipeline.retry_failed(data, config=config)
        assert tracker.rows("score") == [2]
        assert third.errors == []
        assert third.data[2]["score"] == 20


class TestSelection:
    def test_rows_restriction_keeps_the_others_retryable(self, tmp_path):
        tracker = Tracker(failing={1, 3})
        pipeline = _pipeline(tracker)
        data = _rows(5)
        config = _config(tmp_path)

        pipeline.run(data, config=config)
        tracker.reset()
        tracker.failing.clear()

        partial = pipeline.retry_failed(data, config=config, rows=[1])

        assert tracker.rows("score") == [1]
        assert partial.data[1]["score"] == 10
        assert partial.data[3]["score"] is None
        assert _checkpoint_files(tmp_path), "row 3 is still failing — keep the checkpoint"

        # Row 3's failure survived the partial retry's write-back.
        tracker.reset()
        rest = pipeline.retry_failed(data, config=config)
        assert tracker.rows("score") == [3]
        assert rest.data[3]["score"] == 30
        assert _checkpoint_files(tmp_path) == []

    def test_steps_restriction_selects_nothing_when_the_step_is_clean(self, tmp_path):
        tracker = Tracker(failing={1})
        pipeline = _pipeline(tracker)
        data = _rows(4)
        config = _config(tmp_path)

        pipeline.run(data, config=config)
        tracker.reset()
        tracker.failing.clear()

        result = pipeline.retry_failed(data, config=config, steps=["normalize"])

        # Nothing selected -> nothing invoked; the checkpointed run is returned.
        assert tracker.calls == {}
        assert result.errors == []
        assert result.data[1]["score"] is None
        # `score` is still failing, so the checkpoint stays for a real retry.
        assert len(_checkpoint_files(tmp_path)) == 1

    def test_steps_restriction_targets_one_step(self, tmp_path):
        tracker = Tracker(failing={0, 2})
        pipeline = _pipeline(tracker)
        data = _rows(3)
        config = _config(tmp_path)

        pipeline.run(data, config=config)
        tracker.reset()
        tracker.failing.clear()

        pipeline.retry_failed(data, config=config, steps=["score"], rows=[0, 2])
        assert tracker.rows("score") == [0, 2]
        assert tracker.count("normalize") == 0


class TestRunLogContinuity:
    def test_retry_appends_a_segment_to_the_same_run_log(self, tmp_path):
        tracker = Tracker(failing={1, 3})
        pipeline = _pipeline(tracker)
        data = _rows(5)
        config = _config(tmp_path)
        log = tmp_path / "run.jsonl"

        pipeline.run(data, config=config, run_log=log, display_key="company")
        original = _load_log(log)
        run_id = original[0]["run_id"]
        assert original[0]["type"] == "pipeline_start"
        assert original[-1]["type"] == "pipeline_end"

        tracker.reset()
        tracker.failing.clear()
        pipeline.retry_failed(data, config=config, run_log=log)

        records = _load_log(log)
        assert records[: len(original)] == original, "appended to, not rewritten"

        # One run, one retry segment.
        types = [r["type"] for r in records]
        assert types.count("pipeline_start") == 1
        assert types.count("retry_start") == 1
        assert types.count("retry_end") == 1
        assert types[-1] == "retry_end"

        start = records[types.index("retry_start")]
        assert start["run_id"] == run_id, "the retry keeps the original run_id"
        assert start["num_cells"] == 2
        assert sorted((c["step"], c["row"]) for c in start["cells"]) == [
            ("score", 1),
            ("score", 3),
        ]

        # Schema v1 throughout, `t` non-decreasing across both segments.
        assert {r["v"] for r in records} == {1}
        times = [r["t"] for r in records]
        assert times == sorted(times)

        # Recovered cells arrive as ordinary row_complete records, keyed.
        segment = records[types.index("retry_start") :]
        recovered = [r for r in segment if r["type"] == "row_complete"]
        assert len(recovered) == 2
        assert sorted(r["row"] for r in recovered) == [1, 3]
        assert all(r["status"] == "ok" for r in recovered)
        assert all(r["key"] == f"company-{r['row']:02d}" for r in recovered)
        assert all(r["values"]["score"] == r["row"] * 10 for r in recovered)
        # Only the retried step reports; the checkpointed ones stay silent.
        assert {r["step"] for r in segment if r["type"] in {"step_start", "step_end"}} == {"score"}

    def test_retry_without_an_existing_log_starts_a_normal_one(self, tmp_path):
        tracker = Tracker(failing={0})
        pipeline = _pipeline(tracker)
        data = _rows(2)
        config = _config(tmp_path)

        pipeline.run(data, config=config)
        tracker.failing.clear()
        fresh = tmp_path / "fresh.jsonl"
        pipeline.retry_failed(data, config=config, run_log=fresh)

        records = _load_log(fresh)
        assert records[0]["type"] == "pipeline_start"
        assert records[-1]["type"] == "pipeline_end"


class TestAbortedRun:
    def test_steps_the_run_never_finished_still_execute_in_full(self, tmp_path):
        """A killed run leaves unfinished steps; the retry has to run them."""
        tracker = Tracker()
        pipeline = _pipeline(tracker)
        data = _rows(4)
        config = _config(tmp_path)

        manager = CheckpointManager(config)
        manager.save_step(
            data_identifier=derive_data_identifier(["company", "idx"], 4),
            category=DEFAULT_CATEGORY,
            step_name="normalize",
            step_row_results=[{"name_upper": f"COMPANY-{i:02d}"} for i in range(4)],
            total_rows=4,
            fields_dict=pipeline._collect_field_specs(),
            existing_completed=[],
            existing_results={},
        )

        result = pipeline.retry_failed(data, config=config)

        assert tracker.count("normalize") == 0, "the finished step is reused"
        assert tracker.count("score") == 4, "the unfinished steps run in full"
        assert tracker.count("flag") == 4
        assert result.errors == []
        assert [row["name_upper"] for row in result.data] == [f"COMPANY-{i:02d}" for i in range(4)]


class TestRefusals:
    def test_retry_without_checkpointing_enabled_is_refused(self, tmp_path):
        pipeline = _pipeline(Tracker())
        config = _config(tmp_path, enable_checkpointing=False)
        with pytest.raises(PipelineError, match="enable_checkpointing"):
            pipeline.retry_failed(_rows(2), config=config)

    def test_retry_without_a_checkpoint_is_refused(self, tmp_path):
        pipeline = _pipeline(Tracker())
        with pytest.raises(PipelineError, match="nothing to retry"):
            pipeline.retry_failed(_rows(2), config=_config(tmp_path))

    def test_retry_after_a_clean_run_is_refused(self, tmp_path):
        """A run with no failures removes its checkpoint — there is nothing to retry."""
        tracker = Tracker()
        pipeline = _pipeline(tracker)
        data = _rows(3)
        config = _config(tmp_path)

        assert pipeline.run(data, config=config).errors == []
        assert _checkpoint_files(tmp_path) == []
        with pytest.raises(PipelineError, match="nothing to retry"):
            pipeline.retry_failed(data, config=config)

    def test_changed_dataset_shape_is_refused(self, tmp_path):
        tracker = Tracker(failing={1})
        pipeline = _pipeline(tracker)
        config = _config(tmp_path)

        pipeline.run(_rows(4), config=config)
        with pytest.raises(PipelineError, match="nothing to retry"):
            pipeline.retry_failed(_rows(6), config=config)


# --------------------------------------------------------------------------
# CheckpointManager.resume_failed — selection on top of strict validation
# --------------------------------------------------------------------------


class TestResumeFailedSelection:
    @staticmethod
    def _save(manager: CheckpointManager, *, row_errors=None, total_rows=3):
        manager.save_step(
            data_identifier="ds",
            category=DEFAULT_CATEGORY,
            step_name="score",
            step_row_results=[{"score": None} for _ in range(total_rows)],
            total_rows=total_rows,
            fields_dict={"score": {}},
            existing_completed=["normalize"],
            existing_results={"normalize": [{"name_upper": "X"} for _ in range(total_rows)]},
            step_row_errors=row_errors,
        )

    def test_selects_recorded_failures(self, tmp_path):
        manager = CheckpointManager(_config(tmp_path))
        self._save(manager, row_errors={0: {"type": "ValueError", "msg": "boom"}})

        resumed = manager.resume_failed("ds", DEFAULT_CATEGORY)
        assert resumed is not None
        checkpoint, cells = resumed
        assert cells == {"score": [0]}
        assert checkpoint.row_errors["score"][0]["msg"] == "boom"

    def test_row_and_step_filters(self, tmp_path):
        manager = CheckpointManager(_config(tmp_path))
        self._save(
            manager,
            row_errors={
                0: {"type": "ValueError", "msg": "a"},
                2: {"type": "ValueError", "msg": "b"},
            },
        )

        _, cells = manager.resume_failed("ds", DEFAULT_CATEGORY, rows=[2])
        assert cells == {"score": [2]}
        _, cells = manager.resume_failed("ds", DEFAULT_CATEGORY, steps=["normalize"])
        assert cells == {}

    def test_strict_validation_is_unchanged(self, tmp_path):
        manager = CheckpointManager(_config(tmp_path))
        self._save(manager, row_errors={0: {"type": "ValueError", "msg": "a"}}, total_rows=3)

        assert manager.resume_failed("ds", DEFAULT_CATEGORY, expected_total_rows=9) is None
        assert manager.resume_failed("ds", DEFAULT_CATEGORY, expected_fields={"other": {}}) is None
        assert manager.resume_failed("ds", DEFAULT_CATEGORY, expected_steps=["nope"]) is None
        assert manager.resume_failed("ds", DEFAULT_CATEGORY, expected_total_rows=3) is not None

    def test_ignores_auto_resume_but_honours_the_enable_flag(self, tmp_path):
        manager = CheckpointManager(_config(tmp_path, auto_resume=False))
        self._save(manager, row_errors={1: {"type": "ValueError", "msg": "a"}})

        # load() stays gated on auto_resume; resume_failed() is explicit.
        assert manager.load("ds", DEFAULT_CATEGORY) is None
        assert manager.resume_failed("ds", DEFAULT_CATEGORY) is not None

        disabled = CheckpointManager(_config(tmp_path, enable_checkpointing=False))
        assert disabled.resume_failed("ds", DEFAULT_CATEGORY) is None

    def test_pre_129_checkpoints_carry_no_row_errors(self, tmp_path):
        """A checkpoint written before row-error tracking resumes, selecting nothing."""
        manager = CheckpointManager(_config(tmp_path))
        self._save(manager)  # no step_row_errors

        resumed = manager.resume_failed("ds", DEFAULT_CATEGORY)
        assert resumed is not None
        checkpoint, cells = resumed
        assert checkpoint.row_errors == {}
        assert cells == {}

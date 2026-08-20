"""Tests for the run-log contract v1 (accrue/core/runlog.py).  Issue #128.

Validates the committed golden fixture (tests/fixtures/run_small.jsonl)
line-by-line against the schema, checks record-sequence coherence, and
exercises ``Pipeline.run(..., run_log=...)`` end-to-end with function
steps only — no LLM providers, no network, no API keys.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from accrue import EnrichmentConfig, EnrichmentHooks, FunctionStep, JsonlRunLogger, Pipeline
from accrue.core.hooks import RowCompleteEvent
from accrue.core.runlog import (
    SCHEMA_VERSION,
    default_display_key,
    new_run_id,
    resolve_run_log_path,
)
from accrue.schemas.base import UsageInfo
from accrue.steps.base import StepContext, StepResult
from tests.fixtures.generate_run_fixture import (
    ERROR_ROW,
    NUM_ROWS,
    SKIP_ROW,
    generate,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_small.jsonl"

RECORD_TYPES = {"pipeline_start", "step_start", "row_complete", "step_end", "pipeline_end"}

REQUIRED_KEYS = {
    "pipeline_start": {"run_id", "started_at", "num_rows", "display_key", "steps", "plan"},
    "step_start": {"step", "level", "mode", "num_rows"},
    "row_complete": {
        "step",
        "row",
        "status",
        "from_cache",
        "values",
        "error",
        "usage",
        "elapsed_ms",
    },
    "step_end": {"step", "num_errors", "usage", "elapsed_s", "batch_id"},
    "pipeline_end": {"num_rows", "total_errors", "cost", "elapsed_s"},
}


def _load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, f"{path} is empty"
    return [json.loads(line) for line in lines]


def _quiet_config(**overrides) -> EnrichmentConfig:
    defaults = dict(enable_caching=False, enable_progress_bar=False, max_workers=1)
    defaults.update(overrides)
    return EnrichmentConfig(**defaults)


def _tiny_pipeline() -> Pipeline:
    return Pipeline(
        [FunctionStep("upper", lambda ctx: {"upper": ctx.row["name"].upper()}, fields=["upper"])]
    )


def _assert_coherent(records: list[dict]) -> None:
    """Structural invariants every v1 run log must satisfy."""
    # Envelope on every record
    for rec in records:
        assert rec["v"] == SCHEMA_VERSION
        assert isinstance(rec["t"], (int, float)) and rec["t"] >= 0
        assert rec["type"] in RECORD_TYPES
        assert REQUIRED_KEYS[rec["type"]] <= set(rec.keys())

    # t is monotonically non-decreasing
    ts = [rec["t"] for rec in records]
    assert ts == sorted(ts), "t must be non-decreasing"

    # Exactly one pipeline_start (first) and one pipeline_end (last)
    assert records[0]["type"] == "pipeline_start"
    assert records[-1]["type"] == "pipeline_end"
    assert sum(1 for r in records if r["type"] == "pipeline_start") == 1
    assert sum(1 for r in records if r["type"] == "pipeline_end") == 1

    start = records[0]
    num_rows = start["num_rows"]
    step_names = [s["name"] for s in start["steps"]]

    # Per step: step_start precedes its row_completes, which precede step_end,
    # and every row appears exactly once.
    for name in step_names:
        indices = [i for i, r in enumerate(records) if r.get("step") == name]
        assert records[indices[0]]["type"] == "step_start"
        assert records[indices[-1]]["type"] == "step_end"
        rows = [records[i]["row"] for i in indices[1:-1]]
        assert all(records[i]["type"] == "row_complete" for i in indices[1:-1])
        assert sorted(rows) == list(range(num_rows))

    # Counts add up
    assert sum(1 for r in records if r["type"] == "step_start") == len(step_names)
    assert sum(1 for r in records if r["type"] == "step_end") == len(step_names)
    assert sum(1 for r in records if r["type"] == "row_complete") == num_rows * len(step_names)
    step_end_errors = sum(r["num_errors"] for r in records if r["type"] == "step_end")
    assert records[-1]["total_errors"] == step_end_errors
    assert records[-1]["num_rows"] == num_rows


# ---------------------------------------------------------------------------
# Golden fixture validation
# ---------------------------------------------------------------------------


class TestGoldenFixture:
    def test_every_line_parses_with_envelope(self):
        records = _load(FIXTURE)
        for rec in records:
            assert rec["v"] == 1
            assert isinstance(rec["t"], (int, float))
            assert rec["type"] in RECORD_TYPES

    def test_sequence_is_coherent(self):
        _assert_coherent(_load(FIXTURE))

    def test_pipeline_start_record(self):
        start = _load(FIXTURE)[0]
        assert start["num_rows"] == NUM_ROWS
        assert start["display_key"] == "company"
        assert start["plan"] is None
        assert isinstance(start["run_id"], str) and start["run_id"]
        datetime.fromisoformat(start["started_at"])  # parseable ISO-8601
        assert [s["name"] for s in start["steps"]] == ["normalize", "score", "flag"]
        for i, step in enumerate(start["steps"]):
            assert step["level"] == i
            assert step["mode"] == "realtime"
            assert step["model"] is None  # function steps expose no model

    def test_error_row(self):
        records = _load(FIXTURE)
        errors = [r for r in records if r["type"] == "row_complete" and r["status"] == "error"]
        assert len(errors) == 1
        err = errors[0]
        assert err["step"] == "score"
        assert err["row"] == ERROR_ROW
        assert err["error"]["type"] == "ValueError"
        assert err["error"]["msg"]
        assert err["values"] is None  # errored rows carry no real output
        # The step_end for 'score' counts it; pipeline_end totals it
        score_end = next(r for r in records if r["type"] == "step_end" and r["step"] == "score")
        assert score_end["num_errors"] == 1
        assert records[-1]["total_errors"] == 1

    def test_skipped_row(self):
        records = _load(FIXTURE)
        skipped = [r for r in records if r["type"] == "row_complete" and r["status"] == "skipped"]
        assert len(skipped) == 1
        assert skipped[0]["step"] == "flag"
        assert skipped[0]["row"] == SKIP_ROW
        assert skipped[0]["error"] is None
        # Skipped rows carry their skip-default values (None-filled here)
        assert skipped[0]["values"] == {"flagged": None}

    def test_ok_rows_have_null_error_and_no_cache(self):
        records = _load(FIXTURE)
        for rec in records:
            if rec["type"] != "row_complete":
                continue
            assert rec["from_cache"] is False
            assert rec["usage"] is None  # function steps emit no usage
            if rec["status"] == "ok":
                assert rec["error"] is None
                assert isinstance(rec["elapsed_ms"], (int, float))
                assert isinstance(rec["values"], dict) and rec["values"]

    def test_ok_row_values_content(self):
        records = _load(FIXTURE)
        row0 = next(
            r
            for r in records
            if r["type"] == "row_complete" and r["step"] == "normalize" and r["row"] == 0
        )
        assert row0["values"] == {"name_upper": "COMPANY-00"}

    def test_step_end_and_pipeline_end_usage_shape(self):
        records = _load(FIXTURE)
        for rec in records:
            if rec["type"] == "step_end":
                assert set(rec["usage"].keys()) >= {"in", "out", "cost"}
                assert rec["usage"]["cost"] is None  # no pricing data by design
                assert rec["batch_id"] is None
            if rec["type"] == "pipeline_end":
                assert set(rec["cost"].keys()) >= {"in", "out", "cost"}

    def test_regeneration_is_stable(self, tmp_path):
        """Regenerating produces the same record sequence, timestamps aside."""
        regenerated = _load(generate(tmp_path / "regen.jsonl"))
        committed = _load(FIXTURE)

        volatile = {"t", "run_id", "started_at", "elapsed_s", "elapsed_ms"}

        def normalize(records: list[dict]) -> list[dict]:
            return [{k: v for k, v in rec.items() if k not in volatile} for rec in records]

        assert normalize(regenerated) == normalize(committed)


# ---------------------------------------------------------------------------
# Pipeline.run(..., run_log=...) end-to-end
# ---------------------------------------------------------------------------


class TestRunLogParam:
    def test_run_log_true_writes_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # run_log=True resolves relative to CWD
        result = _tiny_pipeline().run(
            [{"name": "acme"}, {"name": "globex"}],
            config=_quiet_config(),
            run_log=True,
        )
        files = list((tmp_path / ".accrue" / "runs").glob("*.jsonl"))
        assert len(files) == 1
        records = _load(files[0])
        _assert_coherent(records)
        assert records[0]["run_id"] == files[0].stem
        # The returned result is unchanged by run_log
        assert result.data[0]["upper"] == "ACME"
        assert not result.has_errors

    def test_run_log_str_and_path_create_parents(self, tmp_path):
        for run_log in (str(tmp_path / "a" / "b" / "log.jsonl"), tmp_path / "c" / "log.jsonl"):
            _tiny_pipeline().run([{"name": "acme"}], config=_quiet_config(), run_log=run_log)
            records = _load(Path(run_log))
            _assert_coherent(records)

    def test_run_log_appends_to_existing_file(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text('{"existing": true}\n', encoding="utf-8")
        _tiny_pipeline().run([{"name": "acme"}], config=_quiet_config(), run_log=path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0]) == {"existing": True}
        assert json.loads(lines[1])["type"] == "pipeline_start"

    def test_run_log_false_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _tiny_pipeline().run([{"name": "acme"}], config=_quiet_config())
        assert not (tmp_path / ".accrue" / "runs").exists()

    async def test_run_async_supports_run_log(self, tmp_path):
        path = tmp_path / "log.jsonl"
        pipeline = _tiny_pipeline()
        await pipeline.run_async([{"name": "acme"}], config=_quiet_config(), run_log=path)
        _assert_coherent(_load(path))

    def test_error_row_logged_end_to_end(self, tmp_path):
        def boom(ctx):
            raise RuntimeError("kaboom")

        path = tmp_path / "log.jsonl"
        pipeline = Pipeline([FunctionStep("boom", boom, fields=["x"])])
        pipeline.run([{"name": "acme"}], config=_quiet_config(), run_log=path)
        records = _load(path)
        _assert_coherent(records)
        row = next(r for r in records if r["type"] == "row_complete")
        assert row["status"] == "error"
        assert row["error"] == {"type": "RuntimeError", "msg": "kaboom"}

    def test_user_hooks_still_fire_with_run_log(self, tmp_path):
        seen: list[str] = []
        hooks = EnrichmentHooks(
            on_pipeline_start=lambda e: seen.append("pipeline_start"),
            on_step_start=lambda e: seen.append("step_start"),
            on_row_complete=lambda e: seen.append("row_complete"),
            on_step_end=lambda e: seen.append("step_end"),
            on_pipeline_end=lambda e: seen.append("pipeline_end"),
        )
        path = tmp_path / "log.jsonl"
        _tiny_pipeline().run([{"name": "acme"}], config=_quiet_config(), hooks=hooks, run_log=path)
        assert seen == ["pipeline_start", "step_start", "row_complete", "step_end", "pipeline_end"]
        _assert_coherent(_load(path))

    def test_raising_user_hook_does_not_break_log(self, tmp_path):
        def bad_hook(event):
            raise RuntimeError("user hook exploded")

        path = tmp_path / "log.jsonl"
        _tiny_pipeline().run(
            [{"name": "acme"}],
            config=_quiet_config(),
            hooks=EnrichmentHooks(on_row_complete=bad_hook, on_pipeline_end=bad_hook),
            run_log=path,
        )
        _assert_coherent(_load(path))

    def test_row_usage_recorded_when_step_emits_it(self, tmp_path):
        class UsageStep:
            name = "llmish"
            fields = ["out"]
            depends_on: list[str] = []
            cache = False

            async def run(self, ctx: StepContext) -> StepResult:
                return StepResult(
                    values={"out": "x"},
                    usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
                )

        path = tmp_path / "log.jsonl"
        Pipeline([UsageStep()]).run([{"name": "acme"}], config=_quiet_config(), run_log=path)
        row = next(r for r in _load(path) if r["type"] == "row_complete")
        assert row["usage"] == {"in": 5, "out": 2, "cost": None}
        assert isinstance(row["elapsed_ms"], (int, float))

    def test_non_json_values_degrade_to_strings(self, tmp_path):
        """A non-JSON value (datetime) must not crash the logger (#128)."""
        from datetime import datetime as dt

        moment = dt(2026, 8, 19, 12, 0, 0)
        step = FunctionStep("when", lambda ctx: {"when": moment}, fields=["when"])
        path = tmp_path / "log.jsonl"
        Pipeline([step]).run([{"name": "acme"}], config=_quiet_config(), run_log=path)
        records = _load(path)
        _assert_coherent(records)
        row = next(r for r in records if r["type"] == "row_complete")
        assert row["status"] == "ok"
        assert row["values"] == {"when": str(moment)}  # default=str degradation

    def test_internal_fields_kept_in_values(self, tmp_path):
        """__-prefixed keys stay in the log even though output filters them."""
        step = FunctionStep(
            "ctx", lambda ctx: {"out": 1, "__internal": "kept"}, fields=["out", "__internal"]
        )
        path = tmp_path / "log.jsonl"
        result = Pipeline([step]).run([{"name": "acme"}], config=_quiet_config(), run_log=path)
        row = next(r for r in _load(path) if r["type"] == "row_complete")
        assert row["values"] == {"out": 1, "__internal": "kept"}
        assert "__internal" not in result.data[0]  # output stays filtered


# ---------------------------------------------------------------------------
# display_key
# ---------------------------------------------------------------------------


class TestDisplayKey:
    def test_heuristic_dataframe_first_string_column(self):
        df = pd.DataFrame({"n": [1, 2], "name": ["a", "b"], "other": ["x", "y"]})
        assert default_display_key(df) == "name"

    def test_heuristic_dataframe_no_string_column_falls_back_to_first(self):
        df = pd.DataFrame({"n": [1, 2], "m": [3.0, 4.0]})
        assert default_display_key(df) == "n"

    def test_heuristic_dataframe_empty_is_none(self):
        assert default_display_key(pd.DataFrame()) is None

    def test_heuristic_list_first_string_key(self):
        assert default_display_key([{"n": 1, "company": "acme"}]) == "company"

    def test_heuristic_list_no_string_falls_back_to_first_key(self):
        assert default_display_key([{"n": 1, "m": 2.0}]) == "n"

    def test_heuristic_empty_list_is_none(self):
        assert default_display_key([]) is None
        assert default_display_key([{}]) is None

    def test_explicit_display_key_wins(self, tmp_path):
        path = tmp_path / "log.jsonl"
        _tiny_pipeline().run(
            [{"name": "acme", "id": 1}],
            config=_quiet_config(),
            run_log=path,
            display_key="id",
        )
        assert _load(path)[0]["display_key"] == "id"

    def test_heuristic_applied_end_to_end(self, tmp_path):
        path = tmp_path / "log.jsonl"
        pipeline = Pipeline([FunctionStep("noop", lambda ctx: {"x": 1}, fields=["x"])])
        pipeline.run([{"id": 1, "name": "acme"}], config=_quiet_config(), run_log=path)
        assert _load(path)[0]["display_key"] == "name"


# ---------------------------------------------------------------------------
# JsonlRunLogger standalone + helpers
# ---------------------------------------------------------------------------


class TestJsonlRunLoggerStandalone:
    def test_logger_hooks_work_via_run(self, tmp_path):
        path = tmp_path / "log.jsonl"
        pipeline = _tiny_pipeline()
        logger = JsonlRunLogger(path, pipeline=pipeline, display_key="name")
        pipeline.run([{"name": "acme"}], config=_quiet_config(), hooks=logger.hooks)
        records = _load(path)
        _assert_coherent(records)
        assert records[0]["run_id"] == logger.run_id
        assert records[0]["display_key"] == "name"

    def test_logger_without_pipeline_emits_null_step_metadata(self, tmp_path):
        path = tmp_path / "log.jsonl"
        pipeline = _tiny_pipeline()
        logger = JsonlRunLogger(path)
        pipeline.run([{"name": "acme"}], config=_quiet_config(), hooks=logger.hooks)
        start = _load(path)[0]
        assert start["steps"] == [{"name": "upper", "level": None, "mode": None, "model": None}]

    def test_new_run_id_format(self):
        run_id = new_run_id()
        datetime.strptime(run_id, "%Y-%m-%d-%H%M%S")

    def test_resolve_run_log_path(self):
        assert resolve_run_log_path(True, "rid") == Path(".accrue/runs/rid.jsonl")
        assert resolve_run_log_path("x/y.jsonl", "rid") == Path("x/y.jsonl")
        assert resolve_run_log_path(Path("z.jsonl"), "rid") == Path("z.jsonl")

    def test_row_complete_event_additive_defaults(self):
        event = RowCompleteEvent(
            step_name="s", row_index=0, values={}, error=None, from_cache=False
        )
        assert event.usage is None
        assert event.elapsed_ms is None

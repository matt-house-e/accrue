"""Tests for auto-saving pipeline results to disk (issue #10)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from accrue.pipeline.pipeline import Pipeline, PipelineResult
from accrue.steps.base import StepContext
from accrue.steps.function import FunctionStep

# -- helpers -------------------------------------------------------------


def _echo_step() -> FunctionStep:
    """A FunctionStep that writes a constant field, so runs are deterministic."""

    def fn(ctx: StepContext) -> dict[str, Any]:
        return {"v": ctx.row["i"] * 2}

    return FunctionStep("double", fn=fn, fields=["v"])


# -- PipelineResult.save() -----------------------------------------------


class TestSaveFormats:
    def test_csv_from_dataframe(self, tmp_path):
        result = PipelineResult(data=pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
        dest = result.save(tmp_path / "out.csv")

        assert dest.exists()
        back = pd.read_csv(dest)
        assert list(back["a"]) == [1, 2]
        assert list(back["b"]) == ["x", "y"]

    def test_csv_from_list_of_dicts(self, tmp_path):
        result = PipelineResult(data=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        dest = result.save(tmp_path / "out.csv")

        back = pd.read_csv(dest)
        assert list(back["a"]) == [1, 2]

    def test_json_round_trips(self, tmp_path):
        result = PipelineResult(data=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        dest = result.save(tmp_path / "out.json")

        loaded = json.loads(dest.read_text())
        assert loaded == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_parquet_round_trips(self, tmp_path):
        pytest.importorskip("pyarrow")
        result = PipelineResult(data=pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
        dest = result.save(tmp_path / "out.parquet")

        back = pd.read_parquet(dest)
        assert list(back["a"]) == [1, 2]

    def test_extension_is_case_insensitive(self, tmp_path):
        result = PipelineResult(data=[{"a": 1}])
        dest = result.save(tmp_path / "out.CSV")
        assert dest.exists()

    def test_creates_parent_directories(self, tmp_path):
        result = PipelineResult(data=[{"a": 1}])
        dest = result.save(tmp_path / "nested" / "deeper" / "out.csv")
        assert dest.exists()

    def test_returns_written_path(self, tmp_path):
        result = PipelineResult(data=[{"a": 1}])
        target = tmp_path / "out.csv"
        assert result.save(target) == target

    def test_unknown_extension_raises(self, tmp_path):
        result = PipelineResult(data=[{"a": 1}])
        with pytest.raises(ValueError, match="csv.*json.*parquet"):
            result.save(tmp_path / "out.xlsx")

    def test_no_extension_raises(self, tmp_path):
        result = PipelineResult(data=[{"a": 1}])
        with pytest.raises(ValueError):
            result.save(tmp_path / "out")


# -- run(output_file=...) integration ------------------------------------


class TestRunOutputFile:
    def test_run_writes_file_and_keeps_data_in_memory(self, tmp_path):
        dest = tmp_path / "results.csv"
        pipeline = Pipeline([_echo_step()])
        result = pipeline.run([{"i": 1}, {"i": 2}, {"i": 3}], output_file=dest)

        # In-memory result is unchanged
        assert [r["v"] for r in result.data] == [2, 4, 6]
        # File was written with the same enriched values
        back = pd.read_csv(dest)
        assert list(back["v"]) == [2, 4, 6]

    def test_run_without_output_file_writes_nothing(self, tmp_path):
        pipeline = Pipeline([_echo_step()])
        pipeline.run([{"i": 1}])
        written = [
            p for p in tmp_path.rglob("*") if p.suffix.lower() in {".csv", ".json", ".parquet"}
        ]
        assert written == []

    def test_run_async_writes_file(self, tmp_path):
        import asyncio

        dest = tmp_path / "results.json"
        pipeline = Pipeline([_echo_step()])
        result = asyncio.run(pipeline.run_async([{"i": 5}], output_file=dest))

        assert result.data[0]["v"] == 10
        loaded = json.loads(dest.read_text())
        assert loaded[0]["v"] == 10

    def test_run_saves_before_returning(self, tmp_path):
        """The save must happen inside run(), so a later user error can't lose it."""
        dest = tmp_path / "results.csv"
        pipeline = Pipeline([_echo_step()])
        result = pipeline.run([{"i": 2}], output_file=dest)
        # File already exists by the time run() returns — before any user code runs.
        assert dest.exists()
        assert result.data[0]["v"] == 4

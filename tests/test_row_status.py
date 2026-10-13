"""``PipelineResult.row_status`` and None-preservation in string columns.

Deterministic steps only — FunctionSteps plus a mocked batch client — so no
network and no API keys.  The scenario the feature exists for: an upstream
step errors on one row, a dependent FunctionStep returns a string ``flag``
for that row and ``None`` for the good rows, and the caller has to tell from
the result which rows are good and which are flagged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from accrue import EnrichmentConfig, FunctionStep, Pipeline
from accrue.core.exceptions import StepError
from accrue.pipeline.pipeline import PipelineResult
from accrue.schemas.base import UsageInfo
from accrue.steps.llm import LLMStep
from accrue.steps.providers.base import BatchResult, LLMResponse

FAILING_ROW = 1


def _extract(ctx):
    if ctx.row["idx"] == FAILING_ROW:
        raise RuntimeError("provider exploded")
    return {"value": f"v{ctx.row['idx']}"}


def _verify(ctx):
    # Mirrors amana's verify step: a flag only when the upstream value is
    # missing, None for every good row.
    value = ctx.prior_results.get("value")
    return {"flag": None if value else "nothing extracted", "checked": True}


def _pipeline(**verify_kwargs) -> Pipeline:
    return Pipeline(
        [
            FunctionStep("extract", fn=_extract, fields=["value"]),
            FunctionStep(
                "verify",
                fn=_verify,
                fields=["flag", "checked"],
                depends_on=["extract"],
                **verify_kwargs,
            ),
        ]
    )


def _config(**overrides) -> EnrichmentConfig:
    defaults = dict(enable_progress_bar=False, enable_caching=False)
    defaults.update(overrides)
    return EnrichmentConfig(**defaults)


def _df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"idx": list(range(n))})


# --------------------------------------------------------------------------
# row_status shape and values
# --------------------------------------------------------------------------


class TestRowStatus:
    def test_one_column_per_step_aligned_to_data(self):
        df = _df().set_axis(["a", "b", "c", "d"])
        result = _pipeline().run(df, config=_config())

        assert list(result.row_status.columns) == ["extract", "verify"]
        assert list(result.row_status.index) == ["a", "b", "c", "d"]
        assert result.row_status["extract"].tolist() == ["ok", "error", "ok", "ok"]
        # Dependents still run on a failed upstream cell — they did not raise.
        assert result.row_status["verify"].tolist() == ["ok"] * 4

    def test_row_level_ok_answers_which_rows_to_trust(self):
        result = _pipeline().run(_df(), config=_config())
        ok = ~result.row_status.eq("error").any(axis=1)

        good = result.data[ok]
        assert good["idx"].tolist() == [0, 2, 3]
        # The amana check, written the obvious way, now gives the right answer.
        assert not any(bool(flag) for flag in good["flag"])

    def test_list_input_gets_a_positional_frame(self):
        rows = [{"idx": i} for i in range(4)]
        result = _pipeline().run(rows, config=_config())

        assert isinstance(result.data, list)
        assert isinstance(result.row_status, pd.DataFrame)
        assert list(result.row_status.index) == [0, 1, 2, 3]
        assert result.row_status["extract"].tolist() == ["ok", "error", "ok", "ok"]

    def test_run_if_skip_is_skipped_not_ok(self):
        pipeline = _pipeline(run_if=lambda row, prior: row["idx"] != 3)
        result = pipeline.run(_df(), config=_config())

        assert result.row_status["verify"].tolist() == ["ok", "ok", "ok", "skipped"]

    def test_cache_hit_is_ok(self, tmp_path: Path):
        config = _config(enable_caching=True, cache_dir=str(tmp_path / "cache"))
        _pipeline().run(_df(), config=config)
        second = _pipeline().run(_df(), config=config)

        assert second.cost.steps["verify"].cache_hits == 4
        assert second.row_status["verify"].tolist() == ["ok"] * 4

    def test_data_is_unchanged_by_status(self):
        result = _pipeline().run(_df(), config=_config())
        assert list(result.data.columns) == ["idx", "value", "flag", "checked"]

    def test_empty_input(self):
        result = _pipeline().run(_df(0), config=_config())
        assert list(result.row_status.columns) == ["extract", "verify"]
        assert len(result.row_status) == 0

    def test_default_for_hand_built_result(self):
        assert PipelineResult(data=[]).row_status.empty


# --------------------------------------------------------------------------
# None stays None in string columns (pandas 3 would coerce it to NaN)
# --------------------------------------------------------------------------


class TestNonePreserved:
    def test_none_survives_next_to_strings(self):
        result = _pipeline().run(_df(), config=_config())

        flags = result.data["flag"].tolist()
        assert flags[FAILING_ROW] == "nothing extracted"
        assert [flags[i] for i in (0, 2, 3)] == [None, None, None]
        # The errored cell's placeholder is None too, not NaN.
        assert result.data["value"].tolist()[FAILING_ROW] is None

    def test_dataframe_and_list_paths_agree(self):
        df_out = _pipeline().run(_df(), config=_config()).data
        list_out = _pipeline().run([{"idx": i} for i in range(4)], config=_config()).data

        assert df_out.to_dict(orient="records") == list_out

    def test_numeric_columns_keep_nan(self):
        def score(ctx):
            return {"score": None if ctx.row["idx"] == 0 else ctx.row["idx"] * 1.5}

        pipeline = Pipeline([FunctionStep("score", fn=score, fields=["score"])])
        out = pipeline.run(_df(3), config=_config()).data

        assert pd.api.types.is_float_dtype(out["score"])
        assert out["score"].isna().tolist() == [True, False, False]

    def test_all_string_column_is_untouched(self):
        def name(ctx):
            return {"name": f"n{ctx.row['idx']}"}

        pipeline = Pipeline([FunctionStep("name", fn=name, fields=["name"])])
        out = pipeline.run(_df(3), config=_config()).data
        expected = pd.Series(["n0", "n1", "n2"]).dtype
        assert out["name"].dtype == expected


# --------------------------------------------------------------------------
# Checkpoint resume and retry_failed
# --------------------------------------------------------------------------


def _cp_config(tmp_path: Path) -> EnrichmentConfig:
    return _config(enable_checkpointing=True, checkpoint_dir=str(tmp_path / "cp"), max_workers=2)


class TestRowStatusAcrossRetry:
    def test_retry_heals_and_reports_whole_dataset(self, tmp_path: Path, monkeypatch):
        config = _cp_config(tmp_path)
        pipeline = _pipeline(run_if=lambda row, prior: row["idx"] != 3)
        first = pipeline.run(_df(), config=config)
        assert first.row_status["extract"].tolist() == ["ok", "error", "ok", "ok"]

        monkeypatch.setattr(sys.modules[__name__], "FAILING_ROW", -1)
        healed = pipeline.retry_failed(_df(), config=config)

        assert healed.errors == []
        assert healed.row_status["extract"].tolist() == ["ok"] * 4
        # Row 3 was skipped in the *first* run and served from the checkpoint
        # now — it must still read as skipped, not ok.
        assert healed.row_status["verify"].tolist() == ["ok", "ok", "ok", "skipped"]

    def test_failure_left_out_of_retry_still_reads_error(self, tmp_path: Path):
        failing = {1, 2}

        def extract(ctx):
            if ctx.row["idx"] in failing:
                raise RuntimeError("boom")
            return {"value": "v"}

        pipeline = Pipeline(
            [
                FunctionStep("extract", fn=extract, fields=["value"]),
                FunctionStep(
                    "verify", fn=_verify, fields=["flag", "checked"], depends_on=["extract"]
                ),
            ]
        )
        config = _cp_config(tmp_path)
        pipeline.run(_df(), config=config)

        failing.clear()
        partial = pipeline.retry_failed(_df(), config=config, rows=[1])

        # `errors` only covers this pass, so row 2's failure is silent there.
        assert partial.errors == []
        assert partial.row_status["extract"].tolist() == ["ok", "ok", "error", "ok"]

    def test_checkpoint_records_skipped_cells(self, tmp_path: Path):
        config = _cp_config(tmp_path)
        pipeline = _pipeline(run_if=lambda row, prior: row["idx"] != 3)
        pipeline.run(_df(), config=config)

        (cp_file,) = (tmp_path / "cp").glob("*_checkpoint.json")
        payload = json.loads(cp_file.read_text())
        assert payload["row_skipped"] == {"verify": [3]}

    def test_old_checkpoint_without_row_skipped_still_loads(self, tmp_path: Path, monkeypatch):
        config = _cp_config(tmp_path)
        pipeline = _pipeline()
        pipeline.run(_df(), config=config)

        (cp_file,) = (tmp_path / "cp").glob("*_checkpoint.json")
        payload = json.loads(cp_file.read_text())
        del payload["row_skipped"]
        cp_file.write_text(json.dumps(payload))

        monkeypatch.setattr(sys.modules[__name__], "FAILING_ROW", -1)
        healed = pipeline.retry_failed(_df(), config=config)
        assert healed.row_status["extract"].tolist() == ["ok"] * 4


# --------------------------------------------------------------------------
# Batch path
# --------------------------------------------------------------------------


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2, model="m"),
    )


class TestBatchRowStatus:
    @pytest.mark.asyncio
    async def test_batch_failure_and_skip(self):
        client = AsyncMock(spec=["complete", "submit_batch", "poll_batch", "cancel_batch"])
        client.submit_batch = AsyncMock(return_value="b1")
        client.poll_batch = AsyncMock(
            return_value=BatchResult(
                responses={"row-0": _response('{"size": "big"}')},
                failed_ids=["row-1"],
                batch_id="b1",
                errors={"row-1": "server error"},
            )
        )
        # The realtime fallback for the failed row fails too.
        client.complete = AsyncMock(side_effect=StepError("still down", step_name="analyze"))
        client.cancel_batch = AsyncMock()

        step = LLMStep(
            "analyze",
            fields={"size": "How big"},
            client=client,
            batch=True,
            run_if=lambda row, prior: row["company"] != "Skip",
        )
        rows = [{"company": "Acme"}, {"company": "Beta"}, {"company": "Skip"}]
        result = await Pipeline([step]).run_async(
            rows, config=_config(max_retries=0, retry_base_delay=0.0)
        )

        assert result.row_status["analyze"].tolist() == ["ok", "error", "skipped"]

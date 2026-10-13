"""Run-log facts accrue-ui needs for cost and error advice (#169).

Cache tokens on every ``usage`` / ``cost`` object, a ``retryable`` flag on
error payloads, and every batch id on ``step_end``.  The realtime cases drive
the real :class:`AnthropicClient` against a mocked SDK, so the values on disk
are the ones the adapter actually produces — no network, no API key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from accrue import EnrichmentConfig, LLMStep, Pipeline


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _quiet_config(**overrides) -> EnrichmentConfig:
    defaults = dict(
        enable_caching=False,
        enable_progress_bar=False,
        max_workers=1,
        max_retries=1,
        retry_base_delay=0.001,
    )
    defaults.update(overrides)
    return EnrichmentConfig(**defaults)


@pytest.fixture
def mock_anthropic(monkeypatch):
    """A minimal fake ``anthropic`` package, the same shape test_providers uses."""
    mod = MagicMock()
    mod.APIError = type("APIError", (Exception,), {})
    mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
    mod.RateLimitError = type("RateLimitError", (Exception,), {})
    mod.AsyncAnthropic = MagicMock()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


def _anthropic_step(create: AsyncMock) -> LLMStep:
    from accrue.steps.providers.anthropic import AnthropicClient

    client = AnthropicClient(api_key="test")
    inner = MagicMock()
    inner.messages.create = create
    client._client = inner
    return LLMStep(
        name="analyze",
        fields={"f1": "Estimate something"},
        client=client,
        model="claude-sonnet-4-5-20250929",
        system_prompt_header="A long static domain briefing.",
        cache=False,
    )


def _message(cache_write: int, cache_read: int, *, text: str = '{"f1": "v"}', stop="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text, type="text", citations=None)],
        stop_reason=stop,
        usage=SimpleNamespace(
            input_tokens=40,
            output_tokens=10,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        ),
    )


ROWS = [{"company": "Acme"}, {"company": "Globex"}]


class TestCacheTokens:
    def test_realtime_anthropic_cache_tokens_match_result_cost(self, tmp_path, mock_anthropic):
        # Cold call writes the cached system prefix, warm call reads it back.
        step = _anthropic_step(AsyncMock(side_effect=[_message(19000, 0), _message(0, 19000)]))
        path = tmp_path / "log.jsonl"
        result = Pipeline([step]).run(ROWS, config=_quiet_config(), run_log=path)
        records = _load(path)

        rows = [r for r in records if r["type"] == "row_complete"]
        assert [r["usage"]["cache_write_tokens"] for r in rows] == [19000, 0]
        assert [r["usage"]["cache_read_tokens"] for r in rows] == [0, 19000]

        step_end = next(r for r in records if r["type"] == "step_end")
        end = records[-1]
        assert end["type"] == "pipeline_end"
        for usage in (step_end["usage"], end["cost"]):
            assert usage["cache_write_tokens"] == result.cost.total_cache_write_tokens == 19000
            assert usage["cache_read_tokens"] == result.cost.total_cache_read_tokens == 19000

    def test_cache_tokens_are_zero_when_provider_reports_none(self, tmp_path, mock_anthropic):
        step = _anthropic_step(AsyncMock(return_value=_message(0, 0)))
        path = tmp_path / "log.jsonl"
        Pipeline([step]).run(ROWS[:1], config=_quiet_config(), run_log=path)
        for rec in _load(path):
            usage = rec.get("usage") or rec.get("cost")
            if isinstance(usage, dict):
                assert usage["cache_read_tokens"] == 0
                assert usage["cache_write_tokens"] == 0


def _run_one(tmp_path: Path, step: LLMStep) -> tuple[list[dict], dict]:
    """Run *step* over one row; return its row_attempt records and its row_complete."""
    path = tmp_path / "log.jsonl"
    Pipeline([step]).run(ROWS[:1], config=_quiet_config(), run_log=path)
    records = _load(path)
    attempts = [r for r in records if r["type"] == "row_attempt"]
    complete = next(r for r in records if r["type"] == "row_complete")
    assert complete["status"] == "error"
    return attempts, complete


class TestRetryable:
    def test_truncation_is_not_retryable(self, tmp_path, mock_anthropic):
        truncated = _message(0, 0, text='{"f1": "cut o', stop="max_tokens")
        step = _anthropic_step(AsyncMock(return_value=truncated))
        attempts, complete = _run_one(tmp_path, step)
        assert complete["error"]["type"] == "LLMAPIError"
        assert complete["error"]["retryable"] is False
        assert [a["error"]["retryable"] for a in attempts] == [False]

    def test_schema_too_complex_is_not_retryable(self, tmp_path, mock_anthropic):
        rejected = mock_anthropic.APIError("The schema is too complex for compilation.")
        step = _anthropic_step(AsyncMock(side_effect=rejected))
        attempts, complete = _run_one(tmp_path, step)
        assert complete["error"]["retryable"] is False
        assert [a["error"]["retryable"] for a in attempts] == [False]

    def test_oversize_request_is_not_retryable(self, tmp_path, mock_anthropic):
        create = AsyncMock(return_value=_message(0, 0))
        step = _anthropic_step(create)
        step._client.max_request_bytes = 10
        _, complete = _run_one(tmp_path, step)
        assert complete["error"]["type"] == "StepError"
        assert complete["error"]["retryable"] is False
        create.assert_not_called()

    def test_rate_limit_is_retryable_on_every_attempt_and_the_row(self, tmp_path, mock_anthropic):
        step = _anthropic_step(AsyncMock(side_effect=mock_anthropic.RateLimitError("429")))
        attempts, complete = _run_one(tmp_path, step)
        assert len(attempts) == 2  # max_retries=1 → two api attempts
        assert all(a["status"] == "rate_limited" for a in attempts)
        assert all(a["error"]["retryable"] is True for a in attempts)
        # The row's StepError is chained from the last 429, so it reads true too.
        assert complete["error"]["type"] == "StepError"
        assert complete["error"]["retryable"] is True

    def test_exhausted_parse_retries_are_not_retryable(self, tmp_path, mock_anthropic):
        step = _anthropic_step(AsyncMock(return_value=_message(0, 0, text="not json")))
        attempts, complete = _run_one(tmp_path, step)
        assert len(attempts) == step.max_retries + 1
        assert all(a["status"] == "parse_error" for a in attempts)
        assert all(a["error"]["retryable"] is False for a in attempts)
        assert complete["error"]["retryable"] is False


class TestIsRetryable:
    def test_cause_chain_and_cycles(self):
        from accrue.core.exceptions import StepError
        from accrue.core.runlog import _is_retryable
        from accrue.steps.providers.base import LLMAPIError

        assert _is_retryable(LLMAPIError("x", status_code=503)) is True
        assert _is_retryable(LLMAPIError("x", status_code=400)) is False
        assert _is_retryable(ValueError("boom")) is False

        try:
            raise StepError("exhausted") from LLMAPIError("x", is_rate_limit=True)
        except StepError as exc:
            assert _is_retryable(exc) is True

        a, b = ValueError("a"), ValueError("b")
        a.__cause__, b.__cause__ = b, a
        assert _is_retryable(a) is False  # a cause cycle terminates


class TestBatchIds:
    def test_step_split_into_n_batches_logs_n_ids(self, tmp_path):
        from tests.test_size_limits import CONFIG, FakeClient, _rows, _step

        # Each request ~ 40 kB base64 + overhead; 100 kB batch fits two → 3 batches.
        client = FakeClient(max_batch_bytes=100_000)
        path = tmp_path / "log.jsonl"
        result = Pipeline([_step(client, batch=True)]).run(
            _rows(*[30_000] * 5), config=CONFIG, run_log=path
        )
        assert len(client.batches) == 3

        step_end = next(r for r in _load(path) if r["type"] == "step_end")
        assert step_end["batch_ids"] == ["batch-1", "batch-2", "batch-3"]
        assert step_end["batch_id"] == "batch-1"  # unchanged: still the first
        assert result.cost.steps["read_docs"].batch_ids == step_end["batch_ids"]

    def test_batch_oversize_row_is_not_retryable(self, tmp_path):
        from tests.test_size_limits import CONFIG, FakeClient, _rows, _step

        client = FakeClient(max_request_bytes=100_000, max_batch_bytes=10**9)
        path = tmp_path / "log.jsonl"
        Pipeline([_step(client, batch=True)]).run(
            _rows(1_000, 200_000), config=CONFIG, run_log=path
        )
        errored = [r for r in _load(path) if r["type"] == "row_complete" and r["error"]]
        assert [(r["row"], r["error"]["retryable"]) for r in errored] == [(1, False)]

    @pytest.mark.asyncio
    async def test_event_with_only_batch_id_still_logs_a_list(self, tmp_path):
        from accrue import JsonlRunLogger
        from accrue.core.hooks import StepEndEvent

        logger = JsonlRunLogger(tmp_path / "log.jsonl")
        for batch_id, expected in (("b-1", ["b-1"]), (None, []), ("", [])):
            await logger.hooks.on_step_end(
                StepEndEvent("s", 1, 0, None, 0.0, execution_mode="batch", batch_id=batch_id)
            )
            assert _load(logger.path)[-1]["batch_ids"] == expected
        logger.close()


class TestOldLogsStillParse:
    def test_read_run_context_on_a_pre_169_log(self, tmp_path):
        """A log written before #169 (no cache keys, retryable or batch_ids) still reads."""
        from accrue.core.runlog import read_run_context

        old = [
            {"v": 1, "t": 0.0, "type": "pipeline_start", "run_id": "r1", "display_key": "k"},
            {"v": 1, "t": 0.5, "type": "row_complete", "step": "s", "row": 0,
             "status": "error", "error": {"type": "ValueError", "msg": "x"},
             "usage": {"in": 1, "out": 1, "cost": None}},
            {"v": 1, "t": 0.7, "type": "step_end", "step": "s", "num_errors": 1,
             "usage": {"in": 1, "out": 1, "cost": None}, "elapsed_s": 0.1,
             "batch_id": None},
            {"v": 1, "t": 0.9, "type": "pipeline_end", "num_rows": 1, "total_errors": 1,
             "cost": {"in": 1, "out": 1, "cost": None}, "elapsed_s": 0.9},
        ]  # fmt: skip
        path = tmp_path / "old.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8")
        assert read_run_context(path) == ("r1", "k", 0.9)

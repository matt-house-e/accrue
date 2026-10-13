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

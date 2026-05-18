"""Tests for Anthropic Batch API adapter."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from accrue.core.exceptions import StepError
from accrue.steps.providers.base import (
    BatchCapableLLMClient,
    BatchRequest,
    BatchResult,
)

# -- Mock anthropic module for tests ----------------------------------------

# The anthropic package is optional, so we mock it for testing
_mock_anthropic = MagicMock()
_mock_anthropic.AsyncAnthropic = MagicMock
_mock_anthropic.APIError = type("APIError", (Exception,), {})
_mock_anthropic.APITimeoutError = type("APITimeoutError", (Exception,), {})
_mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})


@pytest.fixture(autouse=True)
def patch_anthropic():
    with patch.dict(sys.modules, {"anthropic": _mock_anthropic}):
        yield


# -- helpers -----------------------------------------------------------------


def _make_batch_request(idx: int = 0) -> BatchRequest:
    return BatchRequest(
        custom_id=f"row-{idx}",
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Analyze the data."},
        ],
        model="claude-sonnet-4-20250514",
        temperature=0.2,
        max_tokens=4000,
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    )


def _make_succeeded_entry(custom_id: str, content: str = '{"result": "ok"}'):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=content)],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                model="claude-sonnet-4-20250514",
            ),
        ),
    )


def _make_failed_entry(custom_id: str, error_msg: str = "server error"):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="errored",
            error=SimpleNamespace(message=error_msg),
        ),
    )


# -- Protocol compliance -----------------------------------------------------


class TestAnthropicBatchProtocol:
    def test_satisfies_batch_capable_protocol(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test")
        assert isinstance(client, BatchCapableLLMClient)


# -- submit_batch ------------------------------------------------------------


class TestAnthropicSubmitBatch:
    @pytest.mark.asyncio
    async def test_submits_with_correct_format(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.create = AsyncMock(
            return_value=SimpleNamespace(id="msgbatch_abc123")
        )

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        batch_id = await client.submit_batch([_make_batch_request(0)])

        assert batch_id == "msgbatch_abc123"
        mock_inner.messages.batches.create.assert_called_once()

        # Verify request format
        call_kwargs = mock_inner.messages.batches.create.call_args.kwargs
        requests = call_kwargs["requests"]
        assert len(requests) == 1
        req = requests[0]
        assert req["custom_id"] == "row-0"
        assert "system" in req["params"]
        assert req["params"]["model"] == "claude-sonnet-4-20250514"
        # System should be separated from messages
        assert all(m["role"] != "system" for m in req["params"]["messages"])

    @pytest.mark.asyncio
    async def test_structured_outputs_translated(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.create = AsyncMock(
            return_value=SimpleNamespace(id="msgbatch_abc")
        )

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.submit_batch([_make_batch_request(0)])

        call_kwargs = mock_inner.messages.batches.create.call_args.kwargs
        params = call_kwargs["requests"][0]["params"]
        assert "output_config" in params
        assert params["output_config"]["format"]["type"] == "json_schema"


# -- poll_batch --------------------------------------------------------------


class TestAnthropicPollBatch:
    @pytest.mark.asyncio
    async def test_completed_batch(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(processing_status="ended")
        )

        # Mock the async iterator for results
        async def mock_results_iter():
            yield _make_succeeded_entry("row-0", '{"market_size": "$5B"}')
            yield _make_succeeded_entry("row-1", '{"market_size": "$3B"}')

        # results() returns an awaitable that yields an async iterator
        async def mock_results(batch_id):
            return mock_results_iter()

        # We need to make the result of .results() be an async iterable
        class MockResultsIterator:
            def __init__(self):
                self.entries = [
                    _make_succeeded_entry("row-0", '{"market_size": "$5B"}'),
                    _make_succeeded_entry("row-1", '{"market_size": "$3B"}'),
                ]
                self.idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.idx >= len(self.entries):
                    raise StopAsyncIteration
                entry = self.entries[self.idx]
                self.idx += 1
                return entry

        mock_inner.messages.batches.results = AsyncMock(return_value=MockResultsIterator())

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        result = await client.poll_batch("msgbatch_abc", poll_interval=0.01)

        assert isinstance(result, BatchResult)
        assert result.batch_id == "msgbatch_abc"
        assert len(result.responses) == 2
        assert result.responses["row-0"].content == '{"market_size": "$5B"}'
        assert result.failed_ids == []

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(processing_status="in_progress")
        )

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        with pytest.raises(StepError, match="timed out"):
            await client.poll_batch("msgbatch_abc", poll_interval=0.01, timeout=0.02)

    @pytest.mark.asyncio
    async def test_partial_failures(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(processing_status="ended")
        )

        class MockResultsIterator:
            def __init__(self):
                self.entries = [
                    _make_succeeded_entry("row-0"),
                    _make_failed_entry("row-1", "rate limit"),
                ]
                self.idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.idx >= len(self.entries):
                    raise StopAsyncIteration
                entry = self.entries[self.idx]
                self.idx += 1
                return entry

        mock_inner.messages.batches.results = AsyncMock(return_value=MockResultsIterator())

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        result = await client.poll_batch("msgbatch_abc", poll_interval=0.01)

        assert len(result.responses) == 1
        assert "row-0" in result.responses
        assert result.failed_ids == ["row-1"]
        assert "rate limit" in result.errors["row-1"]


# -- cancel_batch ------------------------------------------------------------


class TestAnthropicCancelBatch:
    @pytest.mark.asyncio
    async def test_cancel_success(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.cancel = AsyncMock()

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.cancel_batch("msgbatch_abc")
        mock_inner.messages.batches.cancel.assert_called_once_with("msgbatch_abc")

    @pytest.mark.asyncio
    async def test_cancel_swallows_errors(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.cancel = AsyncMock(side_effect=Exception("fail"))

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        # Should not raise
        await client.cancel_batch("msgbatch_abc")


# -- custom_id uniqueness ----------------------------------------------------


class TestAnthropicSubmitBatchCustomIdValidation:
    @pytest.mark.asyncio
    async def test_duplicate_custom_id_raises(self):
        from accrue.steps.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test")
        client._client = MagicMock()

        req_a = _make_batch_request(0)
        req_b = _make_batch_request(0)  # same index → same custom_id "row-0"

        with pytest.raises(ValueError, match="Duplicate custom_id"):
            await client.submit_batch([req_a, req_b])


# -- CancelledError cleanup --------------------------------------------------


class TestAnthropicPollBatchCancelledError:
    @pytest.mark.asyncio
    async def test_cancelled_error_triggers_cancel_batch(self):
        """CancelledError during polling must call cancel_batch before re-raising."""
        import asyncio

        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()
        mock_inner.messages.batches.retrieve = AsyncMock(side_effect=asyncio.CancelledError())
        mock_inner.messages.batches.cancel = AsyncMock()

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        with pytest.raises((asyncio.CancelledError, BaseException)):
            await client.poll_batch("msgbatch_abc", poll_interval=0.01)

        mock_inner.messages.batches.cancel.assert_called_once_with("msgbatch_abc")


# -- grounding + json_schema warning -----------------------------------------


class TestAnthropicGroundingSchemaWarning:
    @pytest.mark.asyncio
    async def test_grounding_plus_json_schema_logs_warning(self, caplog):
        """When both grounding tools and json_schema are set, a WARNING is emitted."""
        import logging

        from accrue.steps.providers.anthropic import AnthropicClient

        mock_inner = MagicMock()

        # Build a minimal successful response object
        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"result": "ok"}')],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        tools = [{"type": "web_search"}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}, "name": "result"},
        }

        with caplog.at_level(logging.WARNING, logger="accrue.steps.providers.anthropic"):
            await client.complete(
                messages=[{"role": "user", "content": "search for something"}],
                model="claude-sonnet-4-20250514",
                temperature=0.2,
                max_tokens=1000,
                response_format=response_format,
                tools=tools,
            )

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("grounding" in msg and "structured output" in msg for msg in warning_messages), (
            f"Expected grounding+schema warning, got: {warning_messages}"
        )

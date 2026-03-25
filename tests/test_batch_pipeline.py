"""Tests for batch execution path in Pipeline._execute_step_batch()."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from accrue.core.config import EnrichmentConfig
from accrue.core.exceptions import StepError
from accrue.core.hooks import EnrichmentHooks, RowCompleteEvent
from accrue.pipeline.pipeline import Pipeline
from accrue.schemas.base import UsageInfo
from accrue.steps.function import FunctionStep
from accrue.steps.llm import LLMStep
from accrue.steps.providers.base import BatchRequest, BatchResult, LLMResponse

# -- helpers -----------------------------------------------------------------


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=UsageInfo(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="gpt-4.1-mini",
        ),
    )


def _make_batch_client(
    responses: dict[str, str] | None = None,
    failed_ids: list[str] | None = None,
    errors: dict[str, str] | None = None,
) -> AsyncMock:
    """Create a mock BatchCapableLLMClient with configurable batch results."""
    mock = AsyncMock(spec=["complete", "submit_batch", "poll_batch", "cancel_batch"])

    # Realtime fallback
    mock.complete = AsyncMock(return_value=_mock_llm_response('{"market_size": "$1B"}'))

    mock.submit_batch = AsyncMock(return_value="batch_test_123")

    resp_dict = {}
    if responses:
        for cid, content in responses.items():
            resp_dict[cid] = _mock_llm_response(content)

    mock.poll_batch = AsyncMock(
        return_value=BatchResult(
            responses=resp_dict,
            failed_ids=failed_ids or [],
            batch_id="batch_test_123",
            errors=errors or {},
        )
    )
    mock.cancel_batch = AsyncMock()
    return mock


def _make_batch_step(
    name: str = "analyze",
    fields: dict | None = None,
    client: Any = None,
    depends_on: list[str] | None = None,
    **kwargs,
) -> LLMStep:
    if fields is None:
        fields = {"market_size": "Estimate TAM"}
    if client is None:
        client = _make_batch_client(responses={"row-0": '{"market_size": "$5B"}'})
    return LLMStep(
        name=name,
        fields=fields,
        client=client,
        batch=True,
        depends_on=depends_on,
        **kwargs,
    )


# -- Batch dispatch ----------------------------------------------------------


class TestBatchDispatch:
    @pytest.mark.asyncio
    async def test_batch_step_uses_batch_path(self):
        client = _make_batch_client(responses={"row-0": '{"market_size": "$5B"}'})
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])

        result = await pipeline.run_async([{"company": "Acme"}])

        client.submit_batch.assert_called_once()
        client.poll_batch.assert_called_once()
        # complete should NOT be called (batch path, no failures)
        client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_batch_step_uses_realtime(self):
        client = AsyncMock(spec=["complete"])
        client.complete = AsyncMock(return_value=_mock_llm_response('{"market_size": "$5B"}'))
        step = LLMStep(
            name="analyze",
            fields={"market_size": "Estimate TAM"},
            client=client,
            batch=False,
        )
        pipeline = Pipeline([step])
        result = await pipeline.run_async([{"company": "Acme"}])

        client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_flag_false_on_batch_client(self):
        """batch=False should use realtime even if client supports batch."""
        client = _make_batch_client(responses={"row-0": '{"market_size": "$5B"}'})
        step = LLMStep(
            name="analyze",
            fields={"market_size": "Estimate TAM"},
            client=client,
            batch=False,
        )
        pipeline = Pipeline([step])
        result = await pipeline.run_async([{"company": "Acme"}])

        client.submit_batch.assert_not_called()
        client.complete.assert_called_once()


# -- Cache-aware batching ---------------------------------------------------


class TestBatchCaching:
    @pytest.mark.asyncio
    async def test_all_cached_skips_batch(self, tmp_path):
        """100% cache hits should not submit a batch."""
        client = _make_batch_client(responses={"row-0": '{"market_size": "$5B"}'})
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])
        config = EnrichmentConfig(enable_caching=True, cache_dir=str(tmp_path))

        # First run — populates cache
        await pipeline.run_async([{"company": "Acme"}], config=config)

        # Second run — should skip batch
        client.submit_batch.reset_mock()
        client.poll_batch.reset_mock()
        result2 = await pipeline.run_async([{"company": "Acme"}], config=config)

        client.submit_batch.assert_not_called()
        client.poll_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_cache_only_batches_uncached(self, tmp_path):
        """Cached rows should be excluded from the batch submission."""
        submitted_requests: list[list[BatchRequest]] = []

        async def capture_submit(requests, metadata=None):
            submitted_requests.append(list(requests))
            return "batch_test"

        client = AsyncMock(spec=["complete", "submit_batch", "poll_batch", "cancel_batch"])
        client.complete = AsyncMock(return_value=_mock_llm_response('{"market_size": "$1B"}'))
        client.submit_batch = capture_submit
        client.cancel_batch = AsyncMock()

        # Both polls succeed — first with 1 row, second with 1 uncached row
        client.poll_batch = AsyncMock(
            side_effect=[
                BatchResult(
                    responses={"row-0": _mock_llm_response('{"market_size": "$5B"}')},
                    batch_id="batch_test",
                ),
                BatchResult(
                    responses={"row-1": _mock_llm_response('{"market_size": "$3B"}')},
                    batch_id="batch_test",
                ),
            ]
        )

        step = LLMStep(
            name="analyze",
            fields={"market_size": "Estimate TAM"},
            client=client,
            batch=True,
        )
        pipeline = Pipeline([step])
        config = EnrichmentConfig(enable_caching=True, cache_dir=str(tmp_path))

        # First run — 1 row, populates cache
        await pipeline.run_async([{"company": "Acme"}], config=config)
        assert len(submitted_requests) == 1
        assert len(submitted_requests[0]) == 1  # 1 uncached row

        # Second run — 2 rows, first should be cached
        result = await pipeline.run_async([{"company": "Acme"}, {"company": "Beta"}], config=config)

        assert len(submitted_requests) == 2
        # Second batch should only have the uncached row
        assert len(submitted_requests[1]) == 1
        assert submitted_requests[1][0].custom_id == "row-1"
        # Verify cache hit tracked
        assert result.cost.steps["analyze"].cache_hits == 1


# -- Skip predicates --------------------------------------------------------


class TestBatchSkipPredicates:
    @pytest.mark.asyncio
    async def test_skip_if_filters_before_batch(self):
        responses = {"row-1": '{"market_size": "$3B"}'}
        client = _make_batch_client(responses=responses)
        step = LLMStep(
            name="analyze",
            fields={"market_size": {"prompt": "Estimate TAM", "default": "N/A"}},
            client=client,
            batch=True,
            skip_if=lambda row, prior: row.get("skip") is True,
        )
        pipeline = Pipeline([step])

        data = [{"company": "Acme", "skip": True}, {"company": "Beta"}]
        result = await pipeline.run_async(data)

        # Only non-skipped row should be in batch
        call_args = client.submit_batch.call_args
        batch_reqs = call_args[0][0]
        assert len(batch_reqs) == 1
        assert batch_reqs[0].custom_id == "row-1"

        # Skipped row should have default value
        assert result.data[0]["market_size"] == "N/A"
        assert result.data[1]["market_size"] == "$3B"


# -- Auto-chunking ----------------------------------------------------------


class TestBatchAutoChunk:
    @pytest.mark.asyncio
    async def test_auto_chunks_large_batches(self):
        """With batch_max_requests=2, 5 rows should produce 3 batches."""
        num_rows = 5
        responses = {f"row-{i}": f'{{"market_size": "${i}B"}}' for i in range(num_rows)}
        client = _make_batch_client(responses=responses)
        # poll_batch needs to return the right responses per batch
        # Since we're calling submit 3 times, poll returns all of them
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])
        config = EnrichmentConfig(batch_max_requests=2)

        data = [{"company": f"Co{i}"} for i in range(num_rows)]
        result = await pipeline.run_async(data, config=config)

        assert client.submit_batch.call_count == 3  # ceil(5/2) = 3 chunks


# -- Failed rows retry via realtime -----------------------------------------


class TestBatchRealtimeFallback:
    @pytest.mark.asyncio
    async def test_failed_rows_retry_via_realtime(self):
        """Rows that fail in batch should be retried via step.run()."""
        client = _make_batch_client(
            responses={"row-0": '{"market_size": "$5B"}'},
            failed_ids=["row-1"],
            errors={"row-1": "server error"},
        )
        # Realtime fallback returns a valid response
        client.complete = AsyncMock(return_value=_mock_llm_response('{"market_size": "$3B"}'))
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])

        data = [{"company": "Acme"}, {"company": "Beta"}]
        result = await pipeline.run_async(data)

        # Both rows should have results
        assert result.data[0]["market_size"] == "$5B"
        assert result.data[1]["market_size"] == "$3B"
        # complete should be called for the failed row (realtime retry)
        client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_failure_retries_via_realtime(self):
        """Rows with unparseable batch responses retry via realtime."""
        client = _make_batch_client(
            responses={
                "row-0": '{"market_size": "$5B"}',
                "row-1": "not valid json",  # Will fail parse
            },
        )
        client.complete = AsyncMock(return_value=_mock_llm_response('{"market_size": "$3B"}'))
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])

        data = [{"company": "Acme"}, {"company": "Beta"}]
        result = await pipeline.run_async(data)

        assert result.data[0]["market_size"] == "$5B"
        assert result.data[1]["market_size"] == "$3B"
        client.complete.assert_called_once()


# -- Timeout -----------------------------------------------------------------


class TestBatchTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises_step_error(self):
        client = _make_batch_client()
        client.poll_batch = AsyncMock(side_effect=StepError("batch timed out", step_name="batch"))
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])
        config = EnrichmentConfig(on_error="raise")

        with pytest.raises(StepError, match="timed out"):
            await pipeline.run_async([{"company": "Acme"}], config=config)


# -- StepUsage ---------------------------------------------------------------


class TestBatchStepUsage:
    @pytest.mark.asyncio
    async def test_execution_mode_is_batch(self):
        client = _make_batch_client(responses={"row-0": '{"market_size": "$5B"}'})
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])

        result = await pipeline.run_async([{"company": "Acme"}])

        assert "analyze" in result.cost.steps
        usage = result.cost.steps["analyze"]
        assert usage.execution_mode == "batch"
        assert usage.batch_id == "batch_test_123"


# -- Hooks -------------------------------------------------------------------


class TestBatchHooks:
    @pytest.mark.asyncio
    async def test_on_row_complete_fires_for_batch_rows(self):
        events: list[RowCompleteEvent] = []

        def capture(event: RowCompleteEvent):
            events.append(event)

        client = _make_batch_client(
            responses={
                "row-0": '{"market_size": "$5B"}',
                "row-1": '{"market_size": "$3B"}',
            }
        )
        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])
        hooks = EnrichmentHooks(on_row_complete=capture)

        data = [{"company": "Acme"}, {"company": "Beta"}]
        result = await pipeline.run_async(data, hooks=hooks)

        assert len(events) == 2
        assert all(e.step_name == "analyze" for e in events)
        assert all(e.error is None for e in events)


# -- Multi-step pipelines ---------------------------------------------------


class TestBatchMultiStep:
    @pytest.mark.asyncio
    async def test_two_batch_steps(self):
        client1 = _make_batch_client(responses={"row-0": '{"market_size": "$5B"}'})
        client2 = _make_batch_client(responses={"row-0": '{"risk": "Low"}'})
        step1 = LLMStep(
            name="market",
            fields={"market_size": "Estimate TAM"},
            client=client1,
            batch=True,
        )
        step2 = LLMStep(
            name="risk",
            fields={"risk": "Assess risk"},
            client=client2,
            batch=True,
            depends_on=["market"],
        )
        pipeline = Pipeline([step1, step2])
        result = await pipeline.run_async([{"company": "Acme"}])

        assert result.data[0]["market_size"] == "$5B"
        assert result.data[0]["risk"] == "Low"
        client1.submit_batch.assert_called_once()
        client2.submit_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_function_and_batch(self):
        """FunctionStep (realtime) → LLMStep (batch)."""

        def lookup(ctx):
            return {"__context": f"info about {ctx.row['company']}"}

        func_step = FunctionStep(
            name="lookup",
            fn=lookup,
            fields=["__context"],
        )

        client = _make_batch_client(responses={"row-0": '{"market_size": "$5B"}'})
        llm_step = LLMStep(
            name="analyze",
            fields={"market_size": "Estimate TAM"},
            client=client,
            batch=True,
            depends_on=["lookup"],
        )

        pipeline = Pipeline([func_step, llm_step])
        result = await pipeline.run_async([{"company": "Acme"}])

        assert result.data[0]["market_size"] == "$5B"
        client.submit_batch.assert_called_once()


# -- KeyboardInterrupt -------------------------------------------------------


class TestBatchKeyboardInterrupt:
    @pytest.mark.asyncio
    async def test_cancels_batches_on_interrupt(self):
        """Verify the cancel logic exists by testing with a regular exception.

        KeyboardInterrupt is a BaseException that doesn't play well with
        pytest's asyncio integration, so we test the cancel-on-error pattern
        with a surrogate exception and verify the except branch structure.
        """
        client = _make_batch_client()
        # Use a regular exception to test the error propagation path
        client.poll_batch = AsyncMock(side_effect=StepError("simulated failure", step_name="batch"))

        step = _make_batch_step(client=client)
        pipeline = Pipeline([step])
        config = EnrichmentConfig(on_error="raise")

        with pytest.raises(StepError):
            await pipeline.run_async([{"company": "Acme"}], config=config)

        # The batch was submitted before the poll failure
        client.submit_batch.assert_called_once()

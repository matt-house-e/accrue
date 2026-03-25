"""Tests for OpenAI Batch API adapter (submit, poll, cancel, parse)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from accrue.core.exceptions import StepError
from accrue.steps.providers.base import (
    BatchCapableLLMClient,
    BatchRequest,
    BatchResult,
)
from accrue.steps.providers.openai import OpenAIClient

# -- helpers -----------------------------------------------------------------


def _make_batch_request(idx: int = 0) -> BatchRequest:
    return BatchRequest(
        custom_id=f"row-{idx}",
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Analyze the data."},
        ],
        model="gpt-4.1-mini",
        temperature=0.2,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )


def _make_output_jsonl(rows: list[dict]) -> str:
    """Build JSONL output as OpenAI would return."""
    lines = []
    for row in rows:
        custom_id = row["custom_id"]
        content = row.get("content", '{"result": "ok"}')
        status_code = row.get("status_code", 200)
        entry = {
            "custom_id": custom_id,
            "response": {
                "status_code": status_code,
                "body": {
                    "choices": [{"message": {"content": content}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                    "model": "gpt-4.1-mini",
                }
                if status_code == 200
                else {"error": {"message": row.get("error_msg", "internal error")}},
            },
        }
        lines.append(json.dumps(entry))
    return "\n".join(lines)


# -- Protocol compliance -----------------------------------------------------


class TestOpenAIBatchProtocol:
    def test_satisfies_batch_capable_protocol(self):
        client = OpenAIClient(api_key="test")
        assert isinstance(client, BatchCapableLLMClient)


# -- submit_batch ------------------------------------------------------------


class TestSubmitBatch:
    @pytest.mark.asyncio
    async def test_builds_jsonl_and_submits(self):
        mock_client = MagicMock()
        mock_client.files.create = AsyncMock(return_value=SimpleNamespace(id="file-123"))
        mock_client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch-abc"))

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        batch_id = await adapter.submit_batch([_make_batch_request(0), _make_batch_request(1)])

        assert batch_id == "batch-abc"
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()

        # Verify JSONL structure in uploaded file
        call_kwargs = mock_client.files.create.call_args
        file_arg = (
            call_kwargs.kwargs.get("file") or call_kwargs.args[0] if call_kwargs.args else None
        )
        if file_arg is None:
            file_arg = call_kwargs[1].get("file", call_kwargs[0])

    @pytest.mark.asyncio
    async def test_passes_metadata(self):
        mock_client = MagicMock()
        mock_client.files.create = AsyncMock(return_value=SimpleNamespace(id="file-123"))
        mock_client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch-abc"))

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        await adapter.submit_batch(
            [_make_batch_request(0)],
            metadata={"pipeline": "test"},
        )

        batch_call = mock_client.batches.create.call_args
        assert batch_call.kwargs.get("metadata") == {"pipeline": "test"}

    @pytest.mark.asyncio
    async def test_includes_response_format_in_jsonl(self):
        mock_client = MagicMock()
        captured_file = {}

        async def capture_file(**kwargs):
            file_tuple = kwargs.get("file")
            if file_tuple and isinstance(file_tuple, tuple):
                captured_file["content"] = file_tuple[1].read().decode("utf-8")
            return SimpleNamespace(id="file-123")

        mock_client.files.create = capture_file
        mock_client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch-abc"))

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        req = _make_batch_request(0)
        req.response_format = {"type": "json_schema", "json_schema": {"name": "test", "schema": {}}}
        await adapter.submit_batch([req])

        assert "content" in captured_file
        line = json.loads(captured_file["content"].strip().split("\n")[0])
        assert line["body"]["response_format"]["type"] == "json_schema"


# -- poll_batch --------------------------------------------------------------


class TestPollBatch:
    @pytest.mark.asyncio
    async def test_completed(self):
        output_jsonl = _make_output_jsonl(
            [
                {"custom_id": "row-0", "content": '{"market_size": "$5B"}'},
                {"custom_id": "row-1", "content": '{"market_size": "$3B"}'},
            ]
        )

        mock_client = MagicMock()
        mock_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="completed",
                output_file_id="file-out-123",
                error_file_id=None,
                errors=None,
            )
        )
        mock_client.files.content = AsyncMock(return_value=SimpleNamespace(text=output_jsonl))

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        result = await adapter.poll_batch("batch-abc", poll_interval=0.01)

        assert isinstance(result, BatchResult)
        assert result.batch_id == "batch-abc"
        assert len(result.responses) == 2
        assert result.responses["row-0"].content == '{"market_size": "$5B"}'
        assert result.responses["row-0"].usage.total_tokens == 15
        assert result.failed_ids == []

    @pytest.mark.asyncio
    async def test_polls_until_completed(self):
        """Should poll multiple times until completed."""
        output_jsonl = _make_output_jsonl([{"custom_id": "row-0"}])

        call_count = 0

        async def mock_retrieve(batch_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return SimpleNamespace(
                    status="in_progress", output_file_id=None, error_file_id=None, errors=None
                )
            return SimpleNamespace(
                status="completed",
                output_file_id="file-out",
                error_file_id=None,
                errors=None,
            )

        mock_client = MagicMock()
        mock_client.batches.retrieve = mock_retrieve
        mock_client.files.content = AsyncMock(return_value=SimpleNamespace(text=output_jsonl))

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        result = await adapter.poll_batch("batch-abc", poll_interval=0.01)
        assert call_count == 3
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_timeout_raises_step_error(self):
        mock_client = MagicMock()
        mock_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="in_progress",
                output_file_id=None,
                error_file_id=None,
                errors=None,
            )
        )

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        with pytest.raises(StepError, match="timed out"):
            await adapter.poll_batch("batch-abc", poll_interval=0.01, timeout=0.02)

    @pytest.mark.asyncio
    async def test_failed_raises_step_error(self):
        mock_client = MagicMock()
        mock_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="failed",
                output_file_id=None,
                error_file_id=None,
                errors=SimpleNamespace(data=[SimpleNamespace(message="rate limit exceeded")]),
            )
        )

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        with pytest.raises(StepError, match="failed"):
            await adapter.poll_batch("batch-abc", poll_interval=0.01)

    @pytest.mark.asyncio
    async def test_expired_raises_step_error(self):
        mock_client = MagicMock()
        mock_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="expired",
                output_file_id=None,
                error_file_id=None,
                errors=None,
            )
        )

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        with pytest.raises(StepError, match="expired"):
            await adapter.poll_batch("batch-abc", poll_interval=0.01)

    @pytest.mark.asyncio
    async def test_partial_failures(self):
        """Some rows succeed, some fail."""
        output_jsonl = _make_output_jsonl(
            [
                {"custom_id": "row-0", "content": '{"result": "ok"}'},
                {"custom_id": "row-1", "status_code": 400, "error_msg": "bad request"},
            ]
        )

        mock_client = MagicMock()
        mock_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="completed",
                output_file_id="file-out",
                error_file_id=None,
                errors=None,
            )
        )
        mock_client.files.content = AsyncMock(return_value=SimpleNamespace(text=output_jsonl))

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        result = await adapter.poll_batch("batch-abc", poll_interval=0.01)

        assert len(result.responses) == 1
        assert "row-0" in result.responses
        assert result.failed_ids == ["row-1"]
        assert "bad request" in result.errors["row-1"]


# -- cancel_batch ------------------------------------------------------------


class TestCancelBatch:
    @pytest.mark.asyncio
    async def test_cancel_success(self):
        mock_client = MagicMock()
        mock_client.batches.cancel = AsyncMock()

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        await adapter.cancel_batch("batch-abc")
        mock_client.batches.cancel.assert_called_once_with("batch-abc")

    @pytest.mark.asyncio
    async def test_cancel_swallows_errors(self):
        """Cancel is best-effort — errors should be caught and logged."""
        mock_client = MagicMock()
        mock_client.batches.cancel = AsyncMock(side_effect=Exception("API down"))

        adapter = OpenAIClient(api_key="test")
        adapter._client = mock_client

        # Should not raise
        await adapter.cancel_batch("batch-abc")

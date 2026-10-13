"""Provider size limits: request-size estimate, preflight, byte-aware batch splitting."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from accrue.core.config import EnrichmentConfig
from accrue.core.exceptions import StepError
from accrue.pipeline.pipeline import Pipeline
from accrue.schemas.base import UsageInfo
from accrue.steps.attachments import Document
from accrue.steps.llm import LLMStep
from accrue.steps.providers.anthropic import AnthropicClient, _convert_messages
from accrue.steps.providers.base import BatchRequest, BatchResult, LLMResponse
from accrue.steps.providers.limits import (
    REQUEST_OVERHEAD_BYTES,
    base64_len,
    chunk_by_size,
    client_limit,
    estimate_request_bytes,
)
from accrue.steps.providers.openai import OpenAIClient

OK = '{"summary": "ok"}'


def _response() -> LLMResponse:
    return LLMResponse(
        content=OK,
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2, model="m"),
    )


class FakeClient:
    """Batch-capable client with configurable limits; records what it is sent."""

    def __init__(self, max_request_bytes: int | None = None, max_batch_bytes: int | None = None):
        self.max_request_bytes = max_request_bytes
        self.max_batch_bytes = max_batch_bytes
        self.complete_calls = 0
        self.batches: list[list[BatchRequest]] = []

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.complete_calls += 1
        return _response()

    async def submit_batch(self, requests, metadata=None) -> str:
        self.batches.append(list(requests))
        return f"batch-{len(self.batches)}"

    async def poll_batch(self, batch_id, poll_interval=60.0, timeout=86400.0) -> BatchResult:
        reqs = self.batches[int(batch_id.split("-")[1]) - 1]
        return BatchResult(responses={r.custom_id: _response() for r in reqs}, batch_id=batch_id)

    async def cancel_batch(self, batch_id) -> None:
        pass


def _step(client: Any, batch: bool) -> LLMStep:
    return LLMStep(
        name="read_docs",
        fields={"summary": "Summarise the documents"},
        client=client,
        attachments="documents",
        batch=batch,
        cache=False,
    )


def _rows(*doc_sizes: int) -> list[dict[str, Any]]:
    return [
        {"deal": f"d{i}", "documents": [Document(data=b"x" * n)] if n else None}
        for i, n in enumerate(doc_sizes)
    ]


CONFIG = EnrichmentConfig(enable_caching=False, enable_progress_bar=False)


# -- estimate ---------------------------------------------------------------


class TestEstimate:
    def test_base64_len_matches_encoder(self):
        for n in (0, 1, 2, 3, 4, 1000, 25_700_001):
            if n < 10_000:
                assert base64_len(n) == len(base64.standard_b64encode(b"x" * n))
            else:
                assert base64_len(n) == 4 * -(-n // 3)

    def test_documents_counted_base64_inflated(self):
        doc = 3_000_000
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "document", "media_type": "application/pdf", "data": b"x" * doc},
                    {"type": "text", "text": "hi"},
                ],
            },
        ]
        size = estimate_request_bytes(messages, model="m", max_tokens=10)
        assert size >= 4_000_000  # x4/3, not the raw 3 MB
        assert size < 4_000_000 + 2_000

    def test_estimate_bounds_real_anthropic_body(self):
        """The estimate is never below the body AnthropicClient actually sends."""
        data = bytes(range(256)) * 400
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "media_type": "application/pdf",
                        "data": data,
                        "title": "a.pdf",
                    },
                    {"type": "text", "text": "Extract the fields. ünïcode"},
                ],
            },
        ]
        schema = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
        params = {
            "model": "claude-x",
            "max_tokens": 4000,
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
            "messages": _convert_messages(messages),
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
        }
        real = len(json.dumps({"custom_id": "row-0", "params": params}).encode())
        est = estimate_request_bytes(
            [{"role": "system", "content": "sys"}, *messages],
            model="claude-x",
            max_tokens=4000,
            temperature=0.2,
            response_format=schema,
        )
        assert real <= est <= real + REQUEST_OVERHEAD_BYTES


class TestClientLimit:
    def test_provider_defaults(self):
        assert AnthropicClient.max_request_bytes == 32_000_000
        assert AnthropicClient.max_batch_bytes == 256_000_000
        assert OpenAIClient.max_request_bytes is None
        assert OpenAIClient.max_batch_bytes == 200_000_000

    def test_non_int_attributes_ignored(self):
        from unittest.mock import AsyncMock

        assert client_limit(AsyncMock(), "max_request_bytes") is None
        assert client_limit(object(), "max_request_bytes") is None
        assert client_limit(FakeClient(max_request_bytes=0), "max_request_bytes") is None


class TestChunkBySize:
    def test_splits_on_bytes(self):
        assert chunk_by_size("abcde", [4, 4, 4, 4, 4], max_count=100, max_bytes=10) == [
            ["a", "b"],
            ["c", "d"],
            ["e"],
        ]

    def test_splits_on_count_first_if_reached(self):
        assert chunk_by_size("abcde", [1] * 5, max_count=2, max_bytes=100) == [
            ["a", "b"],
            ["c", "d"],
            ["e"],
        ]

    def test_no_byte_limit(self):
        assert chunk_by_size("abc", [10**9] * 3, max_count=10, max_bytes=None) == [["a", "b", "c"]]

    def test_exact_fit(self):
        assert chunk_by_size("ab", [5, 5], max_count=10, max_bytes=10) == [["a", "b"]]

    def test_config_validates_batch_max_bytes(self):
        with pytest.raises(ValueError, match="batch_max_bytes"):
            EnrichmentConfig(batch_max_bytes=0)


# -- realtime preflight -----------------------------------------------------


class TestRealtimePreflight:
    @pytest.mark.asyncio
    async def test_oversize_row_fails_alone_without_api_call(self):
        client = FakeClient(max_request_bytes=100_000)
        result = await Pipeline([_step(client, batch=False)]).run_async(
            _rows(1_000, 200_000, 0), config=CONFIG
        )
        assert client.complete_calls == 2
        assert [e.row_index for e in result.errors] == [1]
        err = result.errors[0].error
        assert isinstance(err, StepError)
        msg = str(err)
        assert "per-request limit" in msg and "100,000 bytes" in msg and "is 0.27 MB" in msg
        assert [r.get("summary") for r in result.data] == ["ok", None, "ok"]

    @pytest.mark.asyncio
    async def test_no_limit_no_check(self):
        client = FakeClient(max_request_bytes=None)
        result = await Pipeline([_step(client, batch=False)]).run_async(
            _rows(500_000), config=CONFIG
        )
        assert client.complete_calls == 1
        assert not result.errors


# -- batch path -------------------------------------------------------------


class TestBatchLimits:
    @pytest.mark.asyncio
    async def test_splits_batches_by_bytes(self):
        # Each request ~ 40 kB base64 + overhead; 100 kB batch fits two.
        client = FakeClient(max_batch_bytes=100_000)
        result = await Pipeline([_step(client, batch=True)]).run_async(
            _rows(*[30_000] * 5), config=CONFIG
        )
        assert [len(b) for b in client.batches] == [2, 2, 1]
        assert not result.errors
        assert [r.get("summary") for r in result.data] == ["ok"] * 5

    @pytest.mark.asyncio
    async def test_config_overrides_client_default(self):
        client = FakeClient(max_batch_bytes=10**9)
        cfg = EnrichmentConfig(
            enable_caching=False, enable_progress_bar=False, batch_max_bytes=100_000
        )
        await Pipeline([_step(client, batch=True)]).run_async(_rows(*[30_000] * 4), config=cfg)
        assert [len(b) for b in client.batches] == [2, 2]

    @pytest.mark.asyncio
    async def test_request_over_per_request_limit_fails_row_not_batch(self):
        client = FakeClient(max_request_bytes=100_000, max_batch_bytes=10**9)
        result = await Pipeline([_step(client, batch=True)]).run_async(
            _rows(1_000, 200_000, 1_000), config=CONFIG
        )
        assert len(client.batches) == 1
        assert sorted(r.custom_id for r in client.batches[0]) == ["row-0", "row-2"]
        assert client.complete_calls == 0  # no realtime fallback for the oversize row
        assert [e.row_index for e in result.errors] == [1]
        assert "per-request limit" in str(result.errors[0].error)
        assert [r.get("summary") for r in result.data] == ["ok", None, "ok"]

    @pytest.mark.asyncio
    async def test_request_over_batch_limit_fails_row(self):
        client = FakeClient(max_request_bytes=None, max_batch_bytes=100_000)
        result = await Pipeline([_step(client, batch=True)]).run_async(
            _rows(200_000, 1_000), config=CONFIG
        )
        assert [[r.custom_id for r in b] for b in client.batches] == [["row-1"]]
        assert [e.row_index for e in result.errors] == [0]
        assert "batch size limit" in str(result.errors[0].error)

    @pytest.mark.asyncio
    async def test_all_rows_oversize_submits_nothing(self):
        client = FakeClient(max_request_bytes=10_000)
        result = await Pipeline([_step(client, batch=True)]).run_async(
            _rows(50_000, 50_000), config=CONFIG
        )
        assert client.batches == []
        assert [e.row_index for e in result.errors] == [0, 1]

    @pytest.mark.asyncio
    async def test_on_error_raise(self):
        client = FakeClient(max_request_bytes=10_000)
        cfg = EnrichmentConfig(enable_caching=False, enable_progress_bar=False, on_error="raise")
        with pytest.raises(Exception, match="per-request limit"):
            await Pipeline([_step(client, batch=True)]).run_async(_rows(50_000), config=cfg)
        assert client.batches == []

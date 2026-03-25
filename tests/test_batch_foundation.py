"""Tests for batch API foundation: data types, config, LLMStep decomposition."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from accrue.core.config import EnrichmentConfig
from accrue.core.exceptions import PipelineError
from accrue.schemas.base import StepUsage, UsageInfo
from accrue.steps.base import StepContext
from accrue.steps.llm import LLMStep
from accrue.steps.providers.base import (
    BatchCapableLLMClient,
    BatchRequest,
    BatchResult,
    LLMClient,
    LLMResponse,
)

# -- helpers -----------------------------------------------------------------


def _make_ctx(**overrides: Any) -> StepContext:
    defaults: dict = dict(
        row={"company": "Acme", "industry": "Tech"},
        fields={"market_size": {"prompt": "Estimate market size"}},
        prior_results={},
    )
    defaults.update(overrides)
    return StepContext(**defaults)


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


def _make_batch_capable_client() -> AsyncMock:
    """Create a mock that satisfies BatchCapableLLMClient."""
    mock = AsyncMock(spec=["complete", "submit_batch", "poll_batch", "cancel_batch"])
    mock.complete = AsyncMock(return_value=_mock_llm_response('{"market_size": "$5B"}'))
    mock.submit_batch = AsyncMock(return_value="batch_123")
    mock.poll_batch = AsyncMock(
        return_value=BatchResult(
            responses={"row-0": _mock_llm_response('{"market_size": "$5B"}')},
            failed_ids=[],
            batch_id="batch_123",
        )
    )
    mock.cancel_batch = AsyncMock()
    return mock


# -- BatchRequest / BatchResult ----------------------------------------------


class TestBatchRequest:
    def test_construction(self):
        req = BatchRequest(
            custom_id="row-42",
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=4000,
        )
        assert req.custom_id == "row-42"
        assert req.model == "gpt-4.1-mini"
        assert req.response_format is None
        assert req.tools is None

    def test_with_response_format(self):
        req = BatchRequest(
            custom_id="row-0",
            messages=[],
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}


class TestBatchResult:
    def test_empty_result(self):
        result = BatchResult()
        assert result.responses == {}
        assert result.failed_ids == []
        assert result.batch_id == ""
        assert result.errors == {}

    def test_with_responses(self):
        resp = _mock_llm_response('{"x": 1}')
        result = BatchResult(
            responses={"row-0": resp},
            failed_ids=["row-1"],
            batch_id="batch_abc",
            errors={"row-1": "Internal error"},
        )
        assert "row-0" in result.responses
        assert result.failed_ids == ["row-1"]
        assert result.batch_id == "batch_abc"


# -- BatchCapableLLMClient protocol ------------------------------------------


class TestBatchCapableLLMClientProtocol:
    def test_basic_client_is_not_batch_capable(self):
        mock = AsyncMock(spec=["complete"])
        mock.complete = AsyncMock()
        # Basic LLMClient without batch methods
        assert isinstance(mock, LLMClient)
        assert not isinstance(mock, BatchCapableLLMClient)

    def test_batch_client_is_batch_capable(self):
        mock = _make_batch_capable_client()
        assert isinstance(mock, LLMClient)
        assert isinstance(mock, BatchCapableLLMClient)


# -- StepUsage extension -----------------------------------------------------


class TestStepUsageExecutionMode:
    def test_default_is_realtime(self):
        usage = StepUsage()
        assert usage.execution_mode == "realtime"
        assert usage.batch_id is None

    def test_batch_mode(self):
        usage = StepUsage(execution_mode="batch", batch_id="batch_123")
        assert usage.execution_mode == "batch"
        assert usage.batch_id == "batch_123"


# -- EnrichmentConfig batch fields -------------------------------------------


class TestConfigBatchFields:
    def test_defaults(self):
        config = EnrichmentConfig()
        assert config.batch_poll_interval == 60.0
        assert config.batch_timeout == 86400.0
        assert config.batch_max_requests == 50000

    def test_custom_values(self):
        config = EnrichmentConfig(
            batch_poll_interval=30.0,
            batch_timeout=3600.0,
            batch_max_requests=10000,
        )
        assert config.batch_poll_interval == 30.0
        assert config.batch_timeout == 3600.0
        assert config.batch_max_requests == 10000

    def test_invalid_poll_interval(self):
        with pytest.raises(ValueError, match="batch_poll_interval must be positive"):
            EnrichmentConfig(batch_poll_interval=0)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="batch_timeout must be positive"):
            EnrichmentConfig(batch_timeout=-1)

    def test_invalid_max_requests(self):
        with pytest.raises(ValueError, match="batch_max_requests must be positive"):
            EnrichmentConfig(batch_max_requests=0)

    def test_for_batch_preset(self):
        config = EnrichmentConfig.for_batch()
        assert config.enable_caching is True
        assert config.enable_checkpointing is True
        assert config.batch_poll_interval == 60.0
        assert config.batch_timeout == 86400.0
        assert config.max_retries == 5


# -- LLMStep batch param ----------------------------------------------------


class TestLLMStepBatchParam:
    def test_batch_defaults_false(self):
        step = LLMStep(name="s", fields=["f"])
        assert step.batch is False

    def test_batch_true(self):
        step = LLMStep(name="s", fields=["f"], batch=True)
        assert step.batch is True

    def test_batch_and_grounding_raises(self):
        with pytest.raises(PipelineError, match="batch=True and grounding"):
            LLMStep(name="s", fields=["f"], batch=True, grounding=True)

    def test_batch_without_grounding_ok(self):
        step = LLMStep(name="s", fields=["f"], batch=True)
        assert step.batch is True
        assert step._grounding_config is None


# -- LLMStep.is_batch_eligible ----------------------------------------------


class TestIsBatchEligible:
    def test_false_when_batch_false(self):
        mock = _make_batch_capable_client()
        step = LLMStep(name="s", fields=["f"], client=mock, batch=False)
        assert step.is_batch_eligible is False

    def test_true_when_batch_true_and_client_supports(self):
        mock = _make_batch_capable_client()
        step = LLMStep(name="s", fields=["f"], client=mock, batch=True)
        assert step.is_batch_eligible is True

    def test_false_when_client_not_batch_capable(self):
        mock = AsyncMock(spec=["complete"])
        mock.complete = AsyncMock()
        step = LLMStep(name="s", fields=["f"], client=mock, batch=True)
        assert step.is_batch_eligible is False


# -- LLMStep.build_messages -------------------------------------------------


class TestBuildMessages:
    def test_returns_messages_and_kwargs(self):
        mock = AsyncMock(spec=["complete"])
        mock.complete = AsyncMock()
        step = LLMStep(
            name="analyze",
            fields={"market_size": "Estimate TAM"},
            client=mock,
            model="gpt-4.1-mini",
        )
        ctx = _make_ctx()
        messages, kwargs = step.build_messages(ctx)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Analyze the provided data" in messages[1]["content"]

        assert kwargs["model"] == "gpt-4.1-mini"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 4000
        assert kwargs["response_format"] is not None
        assert kwargs["tools"] is None

    def test_uses_config_temperature(self):
        mock = AsyncMock(spec=["complete"])
        mock.complete = AsyncMock()
        step = LLMStep(name="s", fields={"f": "prompt"}, client=mock)
        config = EnrichmentConfig(temperature=0.5, max_tokens=2000)
        ctx = _make_ctx(config=config)
        _, kwargs = step.build_messages(ctx)

        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 2000

    def test_step_temperature_overrides_config(self):
        mock = AsyncMock(spec=["complete"])
        mock.complete = AsyncMock()
        step = LLMStep(name="s", fields={"f": "prompt"}, client=mock, temperature=0.8)
        config = EnrichmentConfig(temperature=0.5)
        ctx = _make_ctx(config=config)
        _, kwargs = step.build_messages(ctx)

        assert kwargs["temperature"] == 0.8


# -- LLMStep.parse_response -------------------------------------------------


class TestParseResponse:
    def test_success(self):
        step = LLMStep(
            name="s",
            fields={"market_size": "Estimate TAM"},
            client=AsyncMock(spec=["complete"]),
        )
        response = _mock_llm_response('{"market_size": "$5B"}')
        result = step.parse_response(response)

        assert result.values == {"market_size": "$5B"}
        assert result.usage is not None
        assert "structured_outputs" in result.metadata

    def test_invalid_json_raises(self):
        step = LLMStep(
            name="s",
            fields={"f": "prompt"},
            client=AsyncMock(spec=["complete"]),
        )
        response = LLMResponse(content="not json")
        with pytest.raises(json.JSONDecodeError):
            step.parse_response(response)

    def test_default_enforcement(self):
        step = LLMStep(
            name="s",
            fields={"risk": {"prompt": "Rate risk", "default": "Unknown"}},
            client=AsyncMock(spec=["complete"]),
        )
        response = _mock_llm_response('{"risk": "N/A"}')
        result = step.parse_response(response)
        assert result.values["risk"] == "Unknown"

    def test_filters_to_declared_fields(self):
        step = LLMStep(
            name="s",
            fields={"market_size": "Estimate TAM"},
            client=AsyncMock(spec=["complete"]),
        )
        response = _mock_llm_response('{"market_size": "$5B"}')
        result = step.parse_response(response)
        assert "market_size" in result.values
        # No extra fields beyond what was declared
        assert set(result.values.keys()) == {"market_size"}

    def test_citations_injected(self):
        from accrue.schemas.grounding import Citation

        step = LLMStep(
            name="s",
            fields={"market_size": "Estimate TAM"},
            client=AsyncMock(spec=["complete"]),
            grounding=False,
            sources_field="sources",
        )
        response = LLMResponse(
            content='{"market_size": "$5B"}',
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            citations=[Citation(url="https://example.com", title="Example")],
        )
        result = step.parse_response(response)
        assert "sources" in result.values
        assert result.values["sources"][0]["url"] == "https://example.com"


# -- run() still works (regression) -----------------------------------------


class TestRunRegression:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        mock_client = AsyncMock(spec=["complete"])
        mock_client.complete = AsyncMock(return_value=_mock_llm_response('{"market_size": "$5B"}'))
        step = LLMStep(
            name="analyze",
            fields={"market_size": "Estimate TAM"},
            client=mock_client,
        )
        ctx = _make_ctx()
        result = await step.run(ctx)

        assert result.values["market_size"] == "$5B"
        assert result.metadata["attempts"] == 1
        assert result.metadata["api_retries"] == 0

"""Tests for LLMStep (LLM calls are mocked via LLMClient protocol)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from accrue.core.config import EnrichmentConfig
from accrue.core.exceptions import StepError
from accrue.schemas.base import UsageInfo
from accrue.steps.base import Step, StepContext, StepResult
from accrue.steps.llm import LLMStep
from accrue.steps.providers.base import LLMAPIError, LLMResponse

# -- helpers -------------------------------------------------------------


def _make_ctx(**overrides: Any) -> StepContext:
    defaults: dict = dict(
        row={"company": "Acme", "industry": "Tech"},
        fields={"market_size": {"prompt": "Estimate market size"}},
        prior_results={},
    )
    defaults.update(overrides)
    return StepContext(**defaults)


def _mock_llm_response(
    content: str, prompt_tokens: int = 10, completion_tokens: int = 5
) -> LLMResponse:
    """Build a fake LLMResponse."""
    return LLMResponse(
        content=content,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model="gpt-4.1-mini",
        ),
    )


def _make_mock_client(responses):
    """Create a mock LLMClient with preset responses."""
    mock = AsyncMock()
    if isinstance(responses, list):
        mock.complete = AsyncMock(side_effect=responses)
    else:
        mock.complete = AsyncMock(return_value=responses)
    return mock


# -- construction --------------------------------------------------------


class TestLLMStepConstruction:
    def test_satisfies_protocol(self):
        step = LLMStep(name="llm", fields=["market_size"])
        assert isinstance(step, Step)

    def test_defaults(self):
        step = LLMStep(name="llm", fields=["f1"])
        assert step.depends_on == []
        assert step.model == "gpt-4.1-mini"
        assert step.temperature is None
        assert step.max_tokens is None
        assert step._custom_system_prompt is None
        assert step.max_retries == 2

    def test_custom_params(self):
        step = LLMStep(
            name="custom",
            fields=["a"],
            depends_on=["dep"],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=2000,
            system_prompt="You are a bot.",
            max_retries=5,
        )
        assert step.model == "gpt-4o"
        assert step.temperature == 0.2
        assert step._custom_system_prompt == "You are a bot."

    def test_system_prompt_header_stored(self):
        step = LLMStep(
            name="llm",
            fields=["f1"],
            system_prompt_header="Analyzing B2B SaaS companies.",
        )
        assert step._system_prompt_header == "Analyzing B2B SaaS companies."

    def test_system_prompt_header_default_none(self):
        step = LLMStep(name="llm", fields=["f1"])
        assert step._system_prompt_header is None

    def test_field_spec_validation_rejects_unknown_keys(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            LLMStep(
                "llm",
                fields={
                    "f1": {"prompt": "test", "unknown_key": "bad"},
                },
            )

    def test_field_spec_validation_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            LLMStep(
                "llm",
                fields={
                    "f1": {"prompt": "test", "type": "InvalidType"},
                },
            )

    def test_field_spec_validation_requires_prompt(self):
        with pytest.raises(ValidationError, match="prompt"):
            LLMStep(
                "llm",
                fields={
                    "f1": {"type": "String"},  # missing prompt
                },
            )


# -- client resolution ---------------------------------------------------


class TestLLMStepClient:
    def test_client_not_created_at_init(self):
        step = LLMStep(name="llm", fields=["f1"], api_key="test-key")
        assert step._client is None

    @patch("openai.AsyncOpenAI")
    def test_default_creates_openai_client(self, mock_cls):
        step = LLMStep(name="llm", fields=["f1"], api_key="test-key")
        client = step._resolve_client()
        # Should have created an OpenAIClient internally
        assert client is not None

    def test_injected_client_used_directly(self):
        mock_client = AsyncMock()
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)
        assert step._resolve_client() is mock_client

    @patch("openai.AsyncOpenAI")
    def test_base_url_creates_openai_client(self, mock_cls):
        step = LLMStep(
            name="llm",
            fields=["f1"],
            base_url="http://localhost:11434/v1",
            api_key="test",
        )
        client = step._resolve_client()
        assert client is not None
        # Trigger lazy creation of the inner OpenAI SDK client
        client._get_client()
        mock_cls.assert_called_once_with(api_key="test", base_url="http://localhost:11434/v1")

    def test_client_reused(self):
        mock_client = AsyncMock()
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)
        c1 = step._resolve_client()
        c2 = step._resolve_client()
        assert c1 is c2


# -- system message building --------------------------------------------


class TestLLMStepMessageBuilding:
    def test_includes_row_and_fields(self):
        step = LLMStep(name="llm", fields={"market_size": "Estimate market size"})
        ctx = _make_ctx()
        msg = step._build_system_message(ctx)

        assert "Acme" in msg
        assert "market_size" in msg
        # Uses XML data boundaries (not old JSON injection)
        assert "<row_data>" in msg
        assert "<field_specifications>" in msg

    def test_includes_prior_results(self):
        step = LLMStep(name="llm", fields={"f1": "test"})
        ctx = _make_ctx(prior_results={"search_ctx": "relevant info"})
        msg = step._build_system_message(ctx)

        assert "<prior_results>" in msg
        assert "relevant info" in msg

    def test_omits_prior_results_when_empty(self):
        step = LLMStep(name="llm", fields={"f1": "test"})
        ctx = _make_ctx(prior_results={})
        msg = step._build_system_message(ctx)

        assert "<prior_results>" not in msg

    def test_custom_system_prompt(self):
        step = LLMStep(name="llm", fields={"f1": "test"}, system_prompt="Custom prompt.")
        ctx = _make_ctx()
        msg = step._build_system_message(ctx)

        assert msg.startswith("Custom prompt.")
        # Custom prompt still gets XML data appended
        assert "<row_data>" in msg

    def test_system_prompt_header_passed_to_builder(self):
        step = LLMStep(
            name="llm",
            fields={"f1": "test"},
            system_prompt_header="Analyzing European markets.",
        )
        ctx = _make_ctx()
        msg = step._build_system_message(ctx)

        assert "# Context" in msg
        assert "Analyzing European markets." in msg

    def test_dynamic_prompt_describes_used_keys_only(self):
        step = LLMStep(
            name="llm",
            fields={
                "f1": {"prompt": "test", "enum": ["A", "B"]},
            },
        )
        ctx = _make_ctx()
        msg = step._build_system_message(ctx)

        # enum is used → described
        assert "enum" in msg.lower()
        # format is NOT used → not described
        assert "**format**" not in msg

    def test_sandwich_pattern_reminder(self):
        step = LLMStep(name="llm", fields={"f1": "test"})
        ctx = _make_ctx()
        msg = step._build_system_message(ctx)

        # Reminder section at end
        assert msg.strip().endswith("No additional text.")


# -- successful run ------------------------------------------------------


class TestLLMStepRun:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        resp = _mock_llm_response(json.dumps({"market_size": "Large"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={"market_size": "Estimate market size"},
            client=mock_client,
        )

        result = await step.run(_make_ctx())

        assert isinstance(result, StepResult)
        assert result.values == {"market_size": "Large"}
        assert result.usage is not None
        assert result.usage.total_tokens == 15
        assert result.usage.model == "gpt-4.1-mini"
        assert result.metadata["attempts"] == 1

    @pytest.mark.asyncio
    async def test_filters_to_declared_fields(self):
        resp = _mock_llm_response(json.dumps({"market_size": "Large", "extra": "junk"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(name="llm", fields={"market_size": "Estimate"}, client=mock_client)

        result = await step.run(_make_ctx())
        assert "extra" not in result.values
        assert result.values == {"market_size": "Large"}

    @pytest.mark.asyncio
    async def test_temperature_from_config(self):
        resp = _mock_llm_response(json.dumps({"f1": "val"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client)

        config = EnrichmentConfig(temperature=0.1, max_tokens=500, max_workers=3)
        ctx = _make_ctx(config=config)
        await step.run(ctx)

        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.1
        assert call_kwargs.kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_temperature_fallback_is_0_2(self):
        """Without config, temperature falls back to 0.2 (not 0.5)."""
        resp = _mock_llm_response(json.dumps({"f1": "val"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client)

        await step.run(_make_ctx())

        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.2


# -- default enforcement -------------------------------------------------


class TestDefaultEnforcement:
    @pytest.mark.asyncio
    async def test_refusal_replaced_with_default(self):
        resp = _mock_llm_response(json.dumps({"f1": "Unable to determine"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={
                "f1": {"prompt": "test", "default": "N/A"},
            },
            client=mock_client,
        )

        result = await step.run(_make_ctx())
        assert result.values["f1"] == "N/A"

    @pytest.mark.asyncio
    async def test_null_replaced_with_default(self):
        resp = _mock_llm_response(json.dumps({"f1": None}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={
                "f1": {"prompt": "test", "default": "fallback"},
            },
            client=mock_client,
        )

        result = await step.run(_make_ctx())
        assert result.values["f1"] == "fallback"

    @pytest.mark.asyncio
    async def test_empty_string_replaced_with_default(self):
        resp = _mock_llm_response(json.dumps({"f1": "  "}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={
                "f1": {"prompt": "test", "default": "fallback"},
            },
            client=mock_client,
        )

        result = await step.run(_make_ctx())
        assert result.values["f1"] == "fallback"

    @pytest.mark.asyncio
    async def test_no_default_leaves_refusal_alone(self):
        """Without default set, refusal text is left as-is."""
        resp = _mock_llm_response(json.dumps({"f1": "Unable to determine"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={
                "f1": {"prompt": "test"},  # no default
            },
            client=mock_client,
        )

        result = await step.run(_make_ctx())
        assert result.values["f1"] == "Unable to determine"

    @pytest.mark.asyncio
    async def test_non_refusal_value_not_replaced(self):
        """Normal values are NOT replaced by default."""
        resp = _mock_llm_response(json.dumps({"f1": "Actual answer"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={
                "f1": {"prompt": "test", "default": "fallback"},
            },
            client=mock_client,
        )

        result = await step.run(_make_ctx())
        assert result.values["f1"] == "Actual answer"

    @pytest.mark.asyncio
    async def test_default_none_replaces_refusal(self):
        """default=None explicitly set should replace refusals with None."""
        resp = _mock_llm_response(json.dumps({"f1": "Unknown"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={
                "f1": {"prompt": "test", "default": None},
            },
            client=mock_client,
        )

        result = await step.run(_make_ctx())
        assert result.values["f1"] is None


# -- retries on validation/parse error -----------------------------------


class TestLLMStepRetries:
    @pytest.mark.asyncio
    async def test_retries_on_json_error_then_succeeds(self):
        bad_resp = _mock_llm_response("not json at all")
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = _make_mock_client([bad_resp, good_resp])
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client, max_retries=2)

        result = await step.run(_make_ctx())
        assert result.values == {"f1": "ok"}
        assert result.metadata["attempts"] == 2

    @pytest.mark.asyncio
    async def test_retries_on_validation_error_then_succeeds(self):
        class Strict(BaseModel):
            f1: int  # must be int, not str

        bad_resp = _mock_llm_response(json.dumps({"f1": "not_an_int"}))
        good_resp = _mock_llm_response(json.dumps({"f1": 42}))
        mock_client = _make_mock_client([bad_resp, good_resp])
        step = LLMStep(name="llm", fields=["f1"], client=mock_client, schema=Strict, max_retries=2)

        result = await step.run(_make_ctx())
        assert result.values == {"f1": 42}

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self):
        bad_resp = _mock_llm_response("bad json")
        mock_client = _make_mock_client([bad_resp, bad_resp])
        step = LLMStep(name="llm", fields=["f1"], client=mock_client, max_retries=1)

        with pytest.raises(StepError, match="failed after 2 parse attempts"):
            await step.run(_make_ctx())

        # 1 initial + 1 retry = 2 calls
        assert mock_client.complete.call_count == 2


# -- API error retries ---------------------------------------------------


class TestLLMStepAPIRetries:
    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self):
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                LLMAPIError("rate limited", status_code=429, is_rate_limit=True),
                good_resp,
            ]
        )
        config = EnrichmentConfig(
            temperature=0.2,
            max_tokens=100,
            max_retries=2,
            retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client)

        result = await step.run(_make_ctx(config=config))
        assert result.values == {"f1": "ok"}
        assert result.metadata["api_retries"] == 1

    @pytest.mark.asyncio
    async def test_retries_on_timeout_then_succeeds(self):
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                LLMAPIError("timeout", status_code=408),
                good_resp,
            ]
        )
        config = EnrichmentConfig(
            temperature=0.2,
            max_tokens=100,
            max_retries=2,
            retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client)

        result = await step.run(_make_ctx(config=config))
        assert result.values == {"f1": "ok"}

    @pytest.mark.asyncio
    async def test_api_retries_exhausted_raises(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=LLMAPIError("rate limited", status_code=429, is_rate_limit=True)
        )
        config = EnrichmentConfig(
            temperature=0.2,
            max_tokens=100,
            max_retries=1,
            retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client)

        with pytest.raises(StepError, match="API error after 2 retries"):
            await step.run(_make_ctx(config=config))

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self):
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                LLMAPIError(
                    "rate limited",
                    status_code=429,
                    retry_after=0.01,
                    is_rate_limit=True,
                ),
                good_resp,
            ]
        )
        config = EnrichmentConfig(
            temperature=0.2,
            max_tokens=100,
            max_retries=2,
            retry_base_delay=0.001,
        )
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client)

        result = await step.run(_make_ctx(config=config))
        assert result.values == {"f1": "ok"}

    @pytest.mark.asyncio
    async def test_parse_error_inside_api_retry(self):
        """Parse error on first API attempt, API error on second, success on third."""
        bad_parse_resp = _mock_llm_response("not json")
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                bad_parse_resp,  # parse attempt 1 fails
                bad_parse_resp,  # parse attempt 2 fails → parse retries exhausted → StepError
            ]
        )
        config = EnrichmentConfig(
            temperature=0.2,
            max_tokens=100,
            max_retries=2,
            retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields=["f1"], client=mock_client, max_retries=1)

        # Parse retries (1) exhausted, raises StepError (not caught by API retry)
        with pytest.raises(StepError, match="parse attempts"):
            await step.run(_make_ctx(config=config))


# -- custom schema -------------------------------------------------------


class TestLLMStepCustomSchema:
    @pytest.mark.asyncio
    async def test_custom_schema_validation(self):
        class MySchema(BaseModel):
            score: float
            label: str

        resp = _mock_llm_response(json.dumps({"score": 0.95, "label": "positive"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields=["score", "label"],
            client=mock_client,
            schema=MySchema,
        )

        result = await step.run(_make_ctx())
        assert result.values == {"score": 0.95, "label": "positive"}


# -- structured outputs --------------------------------------------------


class TestStructuredOutputs:
    def test_auto_on_for_native_openai_with_dict_fields(self):
        """Native OpenAI (no base_url, no custom client) + dict fields → json_schema."""
        step = LLMStep(name="llm", fields={"f1": "Estimate TAM"})
        assert step._use_structured_outputs is True
        assert step._response_format["type"] == "json_schema"

    def test_auto_off_for_list_fields(self):
        """list[str] fields (no specs) → json_object."""
        step = LLMStep(name="llm", fields=["f1"])
        assert step._use_structured_outputs is False
        assert step._response_format == {"type": "json_object"}

    def test_auto_off_for_custom_schema(self):
        """Custom schema → json_object."""

        class MySchema(BaseModel):
            f1: str

        step = LLMStep(name="llm", fields={"f1": "test"}, schema=MySchema)
        assert step._use_structured_outputs is False

    def test_auto_off_for_non_openai_client(self):
        """Unknown custom client → json_object."""
        mock_client = AsyncMock()
        step = LLMStep(name="llm", fields={"f1": "test"}, client=mock_client)
        assert step._use_structured_outputs is False

    def test_auto_on_for_anthropic_client(self):
        """AnthropicClient + dict fields → json_schema (constrained decoding)."""
        from accrue.steps.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test")
        step = LLMStep(name="llm", fields={"f1": "test"}, client=client)
        assert step._use_structured_outputs is True
        assert step._response_format["type"] == "json_schema"

    def test_auto_on_for_google_client(self):
        """GoogleClient + dict fields → json_schema."""
        from accrue.steps.providers.google import GoogleClient

        client = GoogleClient(api_key="test")
        step = LLMStep(name="llm", fields={"f1": "test"}, client=client)
        assert step._use_structured_outputs is True
        assert step._response_format["type"] == "json_schema"

    def test_auto_off_for_base_url(self):
        """base_url set → json_object (third-party provider)."""
        step = LLMStep(name="llm", fields={"f1": "test"}, base_url="http://localhost:11434/v1")
        assert step._use_structured_outputs is False

    def test_force_off(self):
        """structured_outputs=False → json_object even for native OpenAI."""
        step = LLMStep(name="llm", fields={"f1": "test"}, structured_outputs=False)
        assert step._use_structured_outputs is False
        assert step._response_format == {"type": "json_object"}

    def test_force_on_for_base_url(self):
        """structured_outputs=True forces json_schema even with base_url."""
        step = LLMStep(
            name="llm",
            fields={"f1": "test"},
            base_url="http://api.groq.com/v1",
            structured_outputs=True,
        )
        assert step._use_structured_outputs is True
        assert step._response_format["type"] == "json_schema"

    def test_force_on_without_field_specs_falls_back(self):
        """structured_outputs=True but no field specs → json_object fallback."""
        step = LLMStep(name="llm", fields=["f1"], structured_outputs=True)
        assert step._use_structured_outputs is False

    @pytest.mark.asyncio
    async def test_response_format_passed_to_client(self):
        """Verify the correct response_format dict reaches the client."""
        resp = _mock_llm_response(json.dumps({"f1": "val"}))
        mock_client = _make_mock_client(resp)

        # Force json_object
        step = LLMStep(
            name="llm",
            fields={"f1": "test"},
            client=mock_client,
            structured_outputs=False,
        )
        await step.run(_make_ctx())
        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_metadata_includes_structured_outputs_flag(self):
        resp = _mock_llm_response(json.dumps({"f1": "val"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm",
            fields={"f1": "test"},
            client=mock_client,
            structured_outputs=False,
        )
        result = await step.run(_make_ctx())
        assert result.metadata["structured_outputs"] is False

    @pytest.mark.asyncio
    async def test_structured_outputs_uses_dynamic_model(self):
        """When structured outputs active, validation uses the dynamic model."""
        resp = _mock_llm_response(json.dumps({"f1": "val"}))
        mock_client = _make_mock_client(resp)
        # Native OpenAI client → auto-enabled

        step = LLMStep(
            name="llm",
            fields={"f1": "test"},
            client=mock_client,  # not an OpenAIClient → auto OFF
            structured_outputs=True,  # force on
        )
        # structured_outputs=True with field specs → force on
        # But mock_client is not OpenAIClient, however force=True overrides
        result = await step.run(_make_ctx())
        assert result.values == {"f1": "val"}
        assert result.metadata["structured_outputs"] is True

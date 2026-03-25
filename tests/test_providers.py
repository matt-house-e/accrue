"""Tests for LLMClient protocol and provider adapters."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from accrue.schemas.base import UsageInfo
from accrue.steps.providers.base import LLMAPIError, LLMClient, LLMResponse
from accrue.steps.providers.openai import OpenAIClient

# -- LLMResponse ---------------------------------------------------------


class TestLLMResponse:
    def test_creation_minimal(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.usage is None
        assert r.citations == []

    def test_creation_with_usage(self):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15, model="test")
        r = LLMResponse(content="hello", usage=usage)
        assert r.usage.total_tokens == 15

    def test_creation_with_citations(self):
        from accrue.schemas.grounding import Citation

        cites = [Citation(url="https://example.com", title="Example")]
        r = LLMResponse(content="hello", citations=cites)
        assert len(r.citations) == 1
        assert r.citations[0].url == "https://example.com"


# -- LLMAPIError ---------------------------------------------------------


class TestLLMAPIError:
    def test_basic(self):
        e = LLMAPIError("rate limited", status_code=429, is_rate_limit=True)
        assert e.status_code == 429
        assert e.is_rate_limit is True
        assert e.retry_after is None

    def test_with_retry_after(self):
        e = LLMAPIError("rate limited", retry_after=2.5, is_rate_limit=True)
        assert e.retry_after == 2.5


# -- LLMClient protocol check -------------------------------------------


class TestLLMClientProtocol:
    def test_openai_client_satisfies_protocol(self):
        client = OpenAIClient(api_key="test")
        assert isinstance(client, LLMClient)


# -- OpenAIClient (Responses API — native OpenAI) -----------------------


class TestOpenAIClientResponses:
    """Tests for native OpenAI using the Responses API (no base_url)."""

    @patch("openai.AsyncOpenAI")
    def test_lazy_client_creation(self, mock_cls):
        client = OpenAIClient(api_key="key123")
        assert client._client is None
        inner = client._get_client()
        mock_cls.assert_called_once_with(api_key="key123")
        assert inner is mock_cls.return_value

    @pytest.mark.asyncio
    async def test_complete_success(self):
        mock_response = SimpleNamespace(
            output_text='{"f": 1}',
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text='{"f": 1}',
                            annotations=[],
                        )
                    ],
                )
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4.1-nano",
            temperature=0.2,
            max_tokens=100,
        )

        assert isinstance(result, LLMResponse)
        assert result.content == '{"f": 1}'
        assert result.usage.total_tokens == 15
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_complete_with_structured_output(self):
        mock_response = SimpleNamespace(
            output_text="{}",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="{}", annotations=[])],
                )
            ],
            usage=None,
        )
        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            model="test",
            temperature=0.0,
            max_tokens=10,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "strict": True,
                    "schema": {"type": "object", "properties": {}},
                },
            },
        )

        call_kwargs = mock_inner.responses.create.call_args.kwargs
        # System message extracted to instructions
        assert call_kwargs["instructions"] == "You are helpful."
        # Structured output via text.format (flattened)
        assert call_kwargs["text"]["format"]["type"] == "json_schema"
        assert call_kwargs["text"]["format"]["name"] == "test_schema"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        mock_response = SimpleNamespace(
            output_text='{"summary": "test"}',
            output=[
                SimpleNamespace(type="web_search_call", status="completed"),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text='{"summary": "test"}',
                            annotations=[
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://example.com",
                                    title="Example",
                                    start_index=0,
                                    end_index=10,
                                ),
                            ],
                        )
                    ],
                ),
            ],
            usage=SimpleNamespace(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "search for news"}],
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=1000,
            tools=[{"type": "web_search"}],
        )

        call_kwargs = mock_inner.responses.create.call_args.kwargs
        assert call_kwargs["tools"] == [{"type": "web_search"}]
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://example.com"
        assert result.citations[0].title == "Example"

    @pytest.mark.asyncio
    async def test_provider_kwargs_merged_at_top_level(self):
        """provider_kwargs are merged into the tool dict at top level and the key is stripped."""
        mock_response = SimpleNamespace(
            output_text='{"summary": "test"}',
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text='{"summary": "test"}',
                            annotations=[],
                        )
                    ],
                )
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=1000,
            tools=[
                {
                    "type": "web_search",
                    "provider_kwargs": {"search_context_size": "high"},
                }
            ],
        )

        call_kwargs = mock_inner.responses.create.call_args.kwargs
        tool = call_kwargs["tools"][0]
        assert tool["search_context_size"] == "high"
        assert "provider_kwargs" not in tool

    @pytest.mark.asyncio
    async def test_allowed_domains_nested_under_filters(self):
        """allowed_domains is translated to filters.allowed_domains for OpenAI."""
        mock_response = SimpleNamespace(
            output_text='{"f": "v"}',
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text='{"f": "v"}', annotations=[])
                    ],
                )
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=1000,
            tools=[
                {
                    "type": "web_search",
                    "allowed_domains": ["sec.gov", "reuters.com"],
                    "user_location": {"country": "GB", "city": "London"},
                }
            ],
        )

        call_kwargs = mock_inner.responses.create.call_args.kwargs
        tool = call_kwargs["tools"][0]
        # allowed_domains nested under filters
        assert tool["filters"] == {"allowed_domains": ["sec.gov", "reuters.com"]}
        assert "allowed_domains" not in tool
        # user_location wrapped with {"type": "approximate", ...}
        assert tool["user_location"] == {
            "type": "approximate",
            "country": "GB",
            "city": "London",
        }

    @pytest.mark.asyncio
    async def test_unsupported_fields_warn_and_drop(self, caplog):
        """blocked_domains and max_searches log warnings and are dropped."""
        mock_response = SimpleNamespace(
            output_text='{"f": "v"}',
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text='{"f": "v"}', annotations=[])
                    ],
                )
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        with caplog.at_level(logging.WARNING):
            await client.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4.1-mini",
                temperature=0.2,
                max_tokens=1000,
                tools=[
                    {
                        "type": "web_search",
                        "blocked_domains": ["reddit.com"],
                        "max_searches": 5,
                    }
                ],
            )

        call_kwargs = mock_inner.responses.create.call_args.kwargs
        tool = call_kwargs["tools"][0]
        assert "blocked_domains" not in tool
        assert "max_searches" not in tool
        assert "blocked_domains" in caplog.text
        assert "max_searches" in caplog.text

    @pytest.mark.asyncio
    async def test_rate_limit_raises_llm_api_error(self):
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "2.5"}

        exc = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )

        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(side_effect=exc)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        with pytest.raises(LLMAPIError) as exc_info:
            await client.complete(
                messages=[],
                model="test",
                temperature=0.0,
                max_tokens=10,
            )

        assert exc_info.value.is_rate_limit is True
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 2.5

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_api_error(self):
        from openai import APITimeoutError

        exc = APITimeoutError(request=MagicMock())

        mock_inner = MagicMock()
        mock_inner.responses.create = AsyncMock(side_effect=exc)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        with pytest.raises(LLMAPIError) as exc_info:
            await client.complete(
                messages=[],
                model="test",
                temperature=0.0,
                max_tokens=10,
            )

        assert exc_info.value.status_code == 408


# -- OpenAIClient (Chat Completions — base_url providers) ----------------


class TestOpenAIClientChatCompletions:
    """Tests for base_url providers using the Chat Completions API fallback."""

    @patch("openai.AsyncOpenAI")
    def test_base_url_passed_to_sdk(self, mock_cls):
        client = OpenAIClient(api_key="key", base_url="http://localhost:11434/v1")
        client._get_client()
        mock_cls.assert_called_once_with(api_key="key", base_url="http://localhost:11434/v1")

    @pytest.mark.asyncio
    async def test_complete_uses_chat_completions(self):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"f": 1}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_inner = MagicMock()
        mock_inner.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test", base_url="http://localhost:11434/v1")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3",
            temperature=0.2,
            max_tokens=100,
        )

        mock_inner.chat.completions.create.assert_called_once()
        assert result.content == '{"f": 1}'
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_complete_with_response_format(self):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=None,
        )
        mock_inner = MagicMock()
        mock_inner.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test", base_url="http://localhost:11434/v1")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            temperature=0.0,
            max_tokens=10,
            response_format={"type": "json_object"},
        )

        call_kwargs = mock_inner.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


# -- AnthropicClient --------------------------------------------------------


class TestAnthropicClient:
    """Test Anthropic adapter — structured outputs via output_config.format.

    The ``anthropic`` package is an optional extra, so we inject a fake module
    into ``sys.modules`` to satisfy the deferred ``from anthropic import ...``
    inside ``complete()``.
    """

    @staticmethod
    def _install_mock_anthropic():
        """Install a minimal mock ``anthropic`` package into sys.modules."""
        import sys

        mock_mod = MagicMock()
        # Provide the exception classes the adapter imports
        mock_mod.APIError = type("APIError", (Exception,), {})
        mock_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_mod.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_mod.AsyncAnthropic = MagicMock()
        sys.modules.setdefault("anthropic", mock_mod)
        return mock_mod

    @pytest.mark.asyncio
    async def test_json_schema_translated_to_output_config(self):
        """json_schema response_format → Anthropic output_config.format."""
        self._install_mock_anthropic()
        from accrue.steps.providers.anthropic import AnthropicClient

        schema = {
            "type": "object",
            "properties": {"f1": {"type": "string"}},
            "required": ["f1"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "enrichment_result",
                "schema": schema,
                "strict": True,
            },
        }

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"f1": "val"}', type="text", citations=None)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            response_format=response_format,
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        assert "output_config" in call_kwargs
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
        assert call_kwargs["output_config"]["format"]["schema"] == schema
        assert result.content == '{"f1": "val"}'

    @pytest.mark.asyncio
    async def test_json_object_ignored(self):
        """json_object has no Anthropic equivalent — no output_config set."""
        self._install_mock_anthropic()
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"f1": "val"}', type="text", citations=None)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs

    @pytest.mark.asyncio
    async def test_none_response_format_no_output_config(self):
        """No response_format → no output_config."""
        self._install_mock_anthropic()
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text="hello", type="text", citations=None)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs

    @pytest.mark.asyncio
    async def test_tools_disable_structured_outputs(self):
        """When tools are passed, output_config is NOT set (incompatible with citations)."""
        self._install_mock_anthropic()
        from accrue.steps.providers.anthropic import AnthropicClient

        schema = {
            "type": "object",
            "properties": {"f1": {"type": "string"}},
            "required": ["f1"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema, "strict": True},
        }

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"f1": "val"}', type="text", citations=None)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            response_format=response_format,
            tools=[{"type": "web_search"}],
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["type"] == "web_search_20250305"

    @pytest.mark.asyncio
    async def test_web_search_tool_translation(self):
        """web_search tool is translated to web_search_20250305 with all config."""
        self._install_mock_anthropic()
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"f1": "val"}', type="text", citations=None)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            tools=[
                {
                    "type": "web_search",
                    "allowed_domains": ["example.com"],
                    "blocked_domains": ["bad.com"],
                    "user_location": {"country": "US", "city": "NYC"},
                    "max_searches": 3,
                }
            ],
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        tool = call_kwargs["tools"][0]
        assert tool["type"] == "web_search_20250305"
        assert tool["name"] == "web_search"
        assert tool["allowed_domains"] == ["example.com"]
        assert tool["blocked_domains"] == ["bad.com"]
        assert tool["user_location"] == {
            "type": "approximate",
            "country": "US",
            "city": "NYC",
        }
        assert tool["max_uses"] == 3

    @pytest.mark.asyncio
    async def test_provider_kwargs_merged_into_server_tool(self):
        """provider_kwargs are merged into the Anthropic server tool dict."""
        self._install_mock_anthropic()
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"f1": "val"}', type="text", citations=None)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            tools=[
                {
                    "type": "web_search",
                    "provider_kwargs": {"custom_param": "value"},
                }
            ],
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        tool = call_kwargs["tools"][0]
        assert tool["custom_param"] == "value"
        assert "provider_kwargs" not in tool

    @pytest.mark.asyncio
    async def test_citation_extraction(self):
        """Citations are extracted from web_search_result_location blocks."""
        self._install_mock_anthropic()
        from accrue.steps.providers.anthropic import AnthropicClient

        mock_response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="I'll search for that.", citations=None),
                SimpleNamespace(
                    type="server_tool_use",
                    id="srvtoolu_123",
                    name="web_search",
                    input={"query": "test"},
                ),
                SimpleNamespace(
                    type="web_search_tool_result",
                    tool_use_id="srvtoolu_123",
                    content=[],
                ),
                SimpleNamespace(
                    type="text",
                    text="Based on search results, here is the answer.",
                    citations=[
                        SimpleNamespace(
                            type="web_search_result_location",
                            url="https://example.com/article",
                            title="Example Article",
                            cited_text="The answer is 42.",
                            encrypted_index="abc123",
                        ),
                    ],
                ),
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "search"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            tools=[{"type": "web_search"}],
        )

        assert result.content == "I'll search for that.Based on search results, here is the answer."
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://example.com/article"
        assert result.citations[0].title == "Example Article"
        assert result.citations[0].snippet == "The answer is 42."


# -- GoogleClient -----------------------------------------------------------


class TestGoogleClient:
    """Test Google Gemini adapter — structured outputs via response_json_schema.

    The ``google-genai`` package is an optional extra, so we inject a fake module
    into ``sys.modules`` to satisfy the deferred ``from google.genai import types``
    inside ``complete()``.
    """

    @staticmethod
    def _install_mock_google():
        """Install minimal mock ``google.genai`` package into sys.modules."""
        import sys

        mock_types = MagicMock()
        mock_types.GenerateContentConfig = lambda **kw: kw
        mock_types.GoogleSearch = lambda **kw: {"google_search": True, **kw}
        mock_types.Tool = lambda **kw: {"tool": True, **kw}

        mock_genai = MagicMock()
        mock_genai.types = mock_types
        mock_genai.Client = MagicMock()

        # google is a namespace package — install all levels
        sys.modules.setdefault("google", MagicMock())
        sys.modules.setdefault("google.genai", mock_genai)
        sys.modules.setdefault("google.genai.types", mock_types)
        return mock_types

    @pytest.mark.asyncio
    async def test_json_schema_translated_to_response_json_schema(self):
        """json_schema → response_mime_type + response_json_schema."""
        self._install_mock_google()
        from accrue.steps.providers.google import GoogleClient

        schema = {
            "type": "object",
            "properties": {"f1": {"type": "string"}},
            "required": ["f1"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "enrichment_result",
                "schema": schema,
                "strict": True,
            },
        }

        mock_response = SimpleNamespace(
            text='{"f1": "val"}',
            candidates=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=1000,
            response_format=response_format,
        )

        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert config["response_mime_type"] == "application/json"
        assert config["response_json_schema"] == schema
        assert result.content == '{"f1": "val"}'

    @pytest.mark.asyncio
    async def test_json_object_sets_mime_type_only(self):
        """json_object → response_mime_type only, no schema."""
        self._install_mock_google()
        from accrue.steps.providers.google import GoogleClient

        mock_response = SimpleNamespace(
            text='{"f1": "val"}',
            candidates=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert config["response_mime_type"] == "application/json"
        assert "response_json_schema" not in config

    @pytest.mark.asyncio
    async def test_tools_disable_structured_outputs(self):
        """When tools are passed, response_json_schema is NOT set (Gemini 2.x compat)."""
        self._install_mock_google()
        from accrue.steps.providers.google import GoogleClient

        schema = {
            "type": "object",
            "properties": {"f1": {"type": "string"}},
            "required": ["f1"],
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema, "strict": True},
        }

        mock_response = SimpleNamespace(
            text='{"f1": "val"}',
            candidates=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=1000,
            response_format=response_format,
            tools=[{"type": "web_search"}],
        )

        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs["config"]
        # Still sets mime type for JSON output via prompting
        assert config["response_mime_type"] == "application/json"
        # But no schema constraint
        assert "response_json_schema" not in config
        # Tools are set
        assert "tools" in config

    @pytest.mark.asyncio
    async def test_grounding_citation_extraction(self):
        """Citations are extracted from grounding_metadata.grounding_chunks."""
        self._install_mock_google()
        from accrue.steps.providers.google import GoogleClient

        mock_response = SimpleNamespace(
            text='{"summary": "AI is advancing"}',
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    uri="https://example.com/ai",
                                    title="AI News",
                                )
                            ),
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    uri="https://other.com/tech",
                                    title="Tech Report",
                                )
                            ),
                        ],
                        web_search_queries=["AI news 2026"],
                    )
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=20,
                candidates_token_count=10,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        result = await client.complete(
            messages=[{"role": "user", "content": "search"}],
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=1000,
            tools=[{"type": "web_search"}],
        )

        assert len(result.citations) == 2
        assert result.citations[0].url == "https://example.com/ai"
        assert result.citations[0].title == "AI News"
        assert result.citations[1].url == "https://other.com/tech"

    @pytest.mark.asyncio
    async def test_provider_kwargs_merged_into_google_search(self):
        """provider_kwargs are merged into GoogleSearch kwargs."""
        self._install_mock_google()
        from accrue.steps.providers.google import GoogleClient

        mock_response = SimpleNamespace(
            text="{}",
            candidates=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=1000,
            tools=[
                {
                    "type": "web_search",
                    "provider_kwargs": {"dynamic_retrieval_config": {"mode": "dynamic"}},
                }
            ],
        )

        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs["config"]
        # The tool was created with GoogleSearch(**gs_kwargs) where gs_kwargs includes
        # the provider_kwargs merged in. Verify the tool is present.
        assert "tools" in config
        tool = config["tools"][0]
        # GoogleSearch mock captures kwargs: {"google_search": True, ...}
        gs = tool.get("google_search", {})
        assert gs.get("dynamic_retrieval_config") == {"mode": "dynamic"}

    @pytest.mark.asyncio
    async def test_allowed_domains_warns(self, caplog):
        """Passing allowed_domains to Google logs a warning (not supported)."""
        self._install_mock_google()
        import logging

        from accrue.steps.providers.google import GoogleClient

        mock_response = SimpleNamespace(
            text="{}",
            candidates=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        with caplog.at_level(logging.WARNING):
            await client.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="gemini-2.5-flash",
                temperature=0.2,
                max_tokens=1000,
                tools=[{"type": "web_search", "allowed_domains": ["example.com"]}],
            )

        assert "allowed_domains is not supported" in caplog.text

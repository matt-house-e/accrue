"""Tests for the web_search utility factory."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from accrue.steps.base import StepContext
from accrue.utils.web_search import web_search

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides: Any) -> StepContext:
    defaults: dict = dict(
        row={"company": "Acme", "industry": "Tech"},
        fields={},
        prior_results={},
    )
    defaults.update(overrides)
    return StepContext(**defaults)


def _mock_response(text: str = "Search result text", urls: list[str] | None = None):
    """Build a mock OpenAI Responses API response."""
    urls = urls or []

    annotations = []
    for url in urls:
        ann = SimpleNamespace(type="url_citation", url=url)
        annotations.append(ann)

    text_part = SimpleNamespace(type="output_text", text=text, annotations=annotations)
    message_item = SimpleNamespace(type="message", content=[text_part])

    return SimpleNamespace(output=[message_item])


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


class TestWebSearchFactory:
    def test_invalid_search_context_size_raises(self):
        with pytest.raises(ValueError, match="search_context_size"):
            web_search("test query", search_context_size="huge")

    def test_valid_context_sizes(self):
        for size in ("low", "medium", "high"):
            fn = web_search("test", search_context_size=size)
            assert asyncio.iscoroutinefunction(fn)

    def test_returns_async_callable(self):
        fn = web_search("Search {company}")
        assert asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# Query template formatting
# ---------------------------------------------------------------------------


class TestQueryFormatting:
    @pytest.mark.asyncio
    async def test_formats_from_row_data(self):
        fn = web_search("Research {company} in {industry}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["input"] == "Research Acme in Tech"

    @pytest.mark.asyncio
    async def test_formats_from_prior_results(self):
        fn = web_search("Search {company} context: {__web_context}")
        ctx = _make_ctx(prior_results={"__web_context": "prior data"})

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert "prior data" in call_kwargs["input"]

    @pytest.mark.asyncio
    async def test_missing_template_field_raises(self):
        fn = web_search("Research {nonexistent_field}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            with pytest.raises(ValueError, match="missing field"):
                await fn(ctx)


# ---------------------------------------------------------------------------
# API call verification
# ---------------------------------------------------------------------------


class TestAPICall:
    @pytest.mark.asyncio
    async def test_passes_correct_params(self):
        fn = web_search("test query", model="gpt-4.1", search_context_size="high")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4.1"
            assert call_kwargs["tools"] == [{"type": "web_search", "search_context_size": "high"}]

    @pytest.mark.asyncio
    async def test_returns_web_context_and_sources(self):
        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                return_value=_mock_response("Result text", ["https://example.com"])
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result["__web_context"] == "Result text"
        assert result["sources"] == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_include_sources_false(self):
        fn = web_search("Search {company}", include_sources=False)
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                return_value=_mock_response("text", ["https://example.com"])
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result["sources"] == []


# ---------------------------------------------------------------------------
# Error handling (graceful degradation)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self):
        from openai import APIError

        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            # Create a proper APIError
            mock_client.responses.create = AsyncMock(
                side_effect=APIError(message="test error", request=MagicMock(), body=None)
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result == {"__web_context": "", "sources": []}

    @pytest.mark.asyncio
    async def test_rate_limit_returns_empty(self):
        from openai import RateLimitError

        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {}
            mock_client.responses.create = AsyncMock(
                side_effect=RateLimitError(
                    message="rate limited",
                    response=mock_response,
                    body=None,
                )
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result == {"__web_context": "", "sources": []}

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        from openai import APITimeoutError

        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                side_effect=APITimeoutError(request=MagicMock())
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result == {"__web_context": "", "sources": []}


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    @pytest.mark.asyncio
    async def test_multiple_citations(self):
        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                return_value=_mock_response(
                    "text",
                    ["https://a.com", "https://b.com", "https://c.com"],
                )
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert len(result["sources"]) == 3
        assert "https://a.com" in result["sources"]


# ---------------------------------------------------------------------------
# FunctionStep compatibility
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool type
# ---------------------------------------------------------------------------


class TestToolType:
    def test_default_is_ga(self):
        """Default tool_type is 'web_search' (GA, cheaper)."""
        fn = web_search("test")
        assert asyncio.iscoroutinefunction(fn)

    def test_invalid_tool_type_raises(self):
        with pytest.raises(ValueError, match="tool_type"):
            web_search("test", tool_type="bad_type")

    def test_valid_tool_types(self):
        for tt in ("web_search", "web_search_preview"):
            fn = web_search("test", tool_type=tt)
            assert asyncio.iscoroutinefunction(fn)

    @pytest.mark.asyncio
    async def test_tool_type_passed_to_api(self):
        fn = web_search("test query", tool_type="web_search_preview")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["tools"][0]["type"] == "web_search_preview"

    @pytest.mark.asyncio
    async def test_ga_tool_type_in_api_call(self):
        fn = web_search("test query")  # default = web_search
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["tools"][0]["type"] == "web_search"


# ---------------------------------------------------------------------------
# User location
# ---------------------------------------------------------------------------


class TestUserLocation:
    @pytest.mark.asyncio
    async def test_user_location_passed_to_api(self):
        fn = web_search(
            "test query",
            user_location={"country": "GB", "city": "London"},
        )
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            tool = call_kwargs["tools"][0]
            assert tool["user_location"]["country"] == "GB"
            assert tool["user_location"]["city"] == "London"
            assert tool["user_location"]["type"] == "approximate"

    @pytest.mark.asyncio
    async def test_user_location_adds_type_automatically(self):
        fn = web_search("test", user_location={"country": "US"})
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            tool = mock_client.responses.create.call_args.kwargs["tools"][0]
            assert tool["user_location"]["type"] == "approximate"

    @pytest.mark.asyncio
    async def test_no_user_location_omitted(self):
        fn = web_search("test")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            tool = mock_client.responses.create.call_args.kwargs["tools"][0]
            assert "user_location" not in tool


# ---------------------------------------------------------------------------
# Allowed domains
# ---------------------------------------------------------------------------


class TestAllowedDomains:
    @pytest.mark.asyncio
    async def test_allowed_domains_passed_as_filters(self):
        fn = web_search(
            "test query",
            allowed_domains=["crunchbase.com", "linkedin.com"],
        )
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            tool = mock_client.responses.create.call_args.kwargs["tools"][0]
            assert tool["filters"]["allowed_domains"] == [
                "crunchbase.com",
                "linkedin.com",
            ]

    @pytest.mark.asyncio
    async def test_no_domains_omits_filters(self):
        fn = web_search("test")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            tool = mock_client.responses.create.call_args.kwargs["tools"][0]
            assert "filters" not in tool

    def test_allowed_domains_with_preview_raises(self):
        """Domain filtering only works with GA tool type."""
        with pytest.raises(ValueError, match="allowed_domains requires"):
            web_search(
                "test",
                allowed_domains=["example.com"],
                tool_type="web_search_preview",
            )


# ---------------------------------------------------------------------------
# FunctionStep compatibility
# ---------------------------------------------------------------------------


class TestFunctionStepCompat:
    def test_is_async_callable(self):
        fn = web_search("test")
        assert asyncio.iscoroutinefunction(fn)

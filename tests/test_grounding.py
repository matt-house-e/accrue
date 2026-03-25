"""Tests for provider-level grounding on LLMStep (Issue #48)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from accrue import LLMStep
from accrue.core.cache import _compute_step_cache_key
from accrue.core.exceptions import PipelineError, StepError
from accrue.schemas.grounding import Citation, GroundingConfig
from accrue.steps.base import StepContext
from accrue.steps.providers.base import LLMResponse

# ============================================================================
# GroundingConfig
# ============================================================================


class TestGroundingConfig:
    def test_default(self):
        cfg = GroundingConfig()
        assert cfg.allowed_domains is None
        assert cfg.blocked_domains is None
        assert cfg.user_location is None
        assert cfg.max_searches is None

    def test_full_config(self):
        cfg = GroundingConfig(
            allowed_domains=["crunchbase.com", "sec.gov"],
            blocked_domains=["reddit.com"],
            user_location={"country": "US", "city": "NYC"},
            max_searches=3,
        )
        assert cfg.allowed_domains == ["crunchbase.com", "sec.gov"]
        assert cfg.blocked_domains == ["reddit.com"]
        assert cfg.user_location == {"country": "US", "city": "NYC"}
        assert cfg.max_searches == 3

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            GroundingConfig(unknown_field="bad")

    def test_from_dict(self):
        cfg = GroundingConfig.model_validate({"allowed_domains": ["example.com"]})
        assert cfg.allowed_domains == ["example.com"]

    def test_provider_kwargs_accepted(self):
        cfg = GroundingConfig(provider_kwargs={"search_context_size": "high"})
        assert cfg.provider_kwargs == {"search_context_size": "high"}

    def test_provider_kwargs_default_none(self):
        cfg = GroundingConfig()
        assert cfg.provider_kwargs is None


class TestCitation:
    def test_defaults(self):
        c = Citation(url="https://example.com")
        assert c.url == "https://example.com"
        assert c.title == ""
        assert c.snippet == ""

    def test_full(self):
        c = Citation(url="https://example.com", title="Test", snippet="A snippet")
        assert c.title == "Test"
        assert c.snippet == "A snippet"


# ============================================================================
# LLMStep grounding normalization
# ============================================================================


class TestLLMStepGroundingInit:
    def test_grounding_none(self):
        step = LLMStep("s", fields=["f1"])
        assert step._grounding_config is None

    def test_grounding_false(self):
        step = LLMStep("s", fields=["f1"], grounding=False)
        assert step._grounding_config is None

    def test_grounding_true(self):
        step = LLMStep("s", fields=["f1"], grounding=True)
        assert isinstance(step._grounding_config, GroundingConfig)
        assert step._grounding_config.allowed_domains is None

    def test_grounding_dict(self):
        step = LLMStep(
            "s",
            fields=["f1"],
            grounding={
                "allowed_domains": ["example.com"],
                "max_searches": 5,
            },
        )
        assert step._grounding_config.allowed_domains == ["example.com"]
        assert step._grounding_config.max_searches == 5

    def test_grounding_config_instance(self):
        cfg = GroundingConfig(blocked_domains=["bad.com"])
        step = LLMStep("s", fields=["f1"], grounding=cfg)
        assert step._grounding_config is cfg

    def test_grounding_invalid_dict(self):
        with pytest.raises(ValidationError, match="extra"):
            LLMStep("s", fields=["f1"], grounding={"bad_key": True})

    def test_grounding_invalid_type(self):
        with pytest.raises(PipelineError, match="Invalid grounding value"):
            LLMStep("s", fields=["f1"], grounding=42)


# ============================================================================
# LLMStep._build_tools_config()
# ============================================================================


class TestBuildToolsConfig:
    def test_no_grounding_returns_none(self):
        step = LLMStep("s", fields=["f1"])
        assert step._build_tools_config() is None

    def test_grounding_true_returns_web_search(self):
        step = LLMStep("s", fields=["f1"], grounding=True)
        tools = step._build_tools_config()
        assert tools == [{"type": "web_search"}]

    def test_grounding_with_config(self):
        step = LLMStep(
            "s",
            fields=["f1"],
            grounding={
                "allowed_domains": ["a.com"],
                "blocked_domains": ["b.com"],
                "user_location": {"country": "GB"},
                "max_searches": 2,
            },
        )
        tools = step._build_tools_config()
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "web_search"
        assert tool["allowed_domains"] == ["a.com"]
        assert tool["blocked_domains"] == ["b.com"]
        assert tool["user_location"] == {"country": "GB"}
        assert tool["max_searches"] == 2

    def test_provider_kwargs_in_tools(self):
        step = LLMStep(
            "s",
            fields=["f1"],
            grounding={
                "provider_kwargs": {"search_context_size": "high"},
            },
        )
        tools = step._build_tools_config()
        assert tools[0]["provider_kwargs"] == {"search_context_size": "high"}

    def test_provider_kwargs_absent_when_none(self):
        step = LLMStep("s", fields=["f1"], grounding=True)
        tools = step._build_tools_config()
        assert "provider_kwargs" not in tools[0]


# ============================================================================
# LLMStep.run() — grounding integration (mocked)
# ============================================================================


class TestLLMStepRunWithGrounding:
    """Test that run() passes tools to client and injects sources."""

    @pytest.mark.asyncio
    async def test_tools_passed_and_sources_injected(self):
        """When grounding is enabled, tools are passed to client and sources injected."""
        citations = [
            Citation(url="https://example.com", title="Example", snippet="The answer"),
        ]
        mock_response = LLMResponse(
            content='{"summary": "AI is growing"}',
            citations=citations,
        )
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        step = LLMStep(
            "s",
            fields={"summary": "Summarize the topic"},
            grounding=True,
            client=mock_client,
        )

        ctx = StepContext(
            row={"topic": "AI"},
            fields={"summary": {"prompt": "Summarize the topic"}},
            prior_results={},
        )
        result = await step.run(ctx)

        # Tools were passed
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["tools"] == [{"type": "web_search"}]

        # sources injected (visible field, default name "sources")
        assert "sources" in result.values
        assert len(result.values["sources"]) == 1
        assert result.values["sources"][0]["url"] == "https://example.com"

        # Regular field present
        assert result.values["summary"] == "AI is growing"

    @pytest.mark.asyncio
    async def test_no_citations_no_sources(self):
        """When grounding returns no citations, sources is NOT injected."""
        mock_response = LLMResponse(
            content='{"summary": "AI is growing"}',
            citations=[],
        )
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        step = LLMStep(
            "s",
            fields={"summary": "Summarize"},
            grounding=True,
            client=mock_client,
        )

        ctx = StepContext(
            row={"topic": "AI"},
            fields={"summary": {"prompt": "Summarize"}},
            prior_results={},
        )
        result = await step.run(ctx)

        assert "sources" not in result.values

    @pytest.mark.asyncio
    async def test_no_grounding_no_tools(self):
        """When grounding is disabled, tools=None is passed."""
        mock_response = LLMResponse(content='{"f1": "val"}')
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        step = LLMStep("s", fields={"f1": "prompt"}, client=mock_client)

        ctx = StepContext(
            row={"x": 1},
            fields={"f1": {"prompt": "prompt"}},
            prior_results={},
        )
        await step.run(ctx)

        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["tools"] is None

    @pytest.mark.asyncio
    async def test_type_error_caught_for_custom_client(self):
        """TypeError from client.complete() gives a clear error when grounding enabled."""

        class OldClient:
            async def complete(
                self, messages, model, temperature, max_tokens, response_format=None
            ):
                return LLMResponse(content="{}")

        step = LLMStep("s", fields=["f1"], grounding=True, client=OldClient())

        ctx = StepContext(
            row={"x": 1},
            fields={},
            prior_results={},
        )

        with pytest.raises(StepError, match="does not support the 'tools' parameter"):
            await step.run(ctx)


# ============================================================================
# Cache key includes grounding hash
# ============================================================================


class TestCacheKeyGrounding:
    def _make_step(self, grounding=None):
        return LLMStep("test_step", fields=["f1"], model="gpt-4.1-mini", grounding=grounding)

    def test_no_grounding_backward_compat(self):
        """Cache key without grounding includes empty grounding_hash."""
        step = self._make_step()
        key = _compute_step_cache_key(step, {"a": 1}, {}, {"f1": {}})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_grounding_changes_cache_key(self):
        """Different grounding configs produce different cache keys."""
        step_none = self._make_step(grounding=None)
        step_true = self._make_step(grounding=True)
        step_cfg = self._make_step(grounding={"allowed_domains": ["x.com"]})

        row = {"a": 1}
        prior = {}
        fields = {"f1": {}}

        key_none = _compute_step_cache_key(step_none, row, prior, fields)
        key_true = _compute_step_cache_key(step_true, row, prior, fields)
        key_cfg = _compute_step_cache_key(step_cfg, row, prior, fields)

        assert key_none != key_true
        assert key_true != key_cfg
        assert key_none != key_cfg

    def test_same_grounding_same_key(self):
        """Same grounding config produces the same cache key."""
        step1 = self._make_step(grounding={"max_searches": 3})
        step2 = self._make_step(grounding={"max_searches": 3})

        row = {"a": 1}
        prior = {}
        fields = {"f1": {}}

        assert _compute_step_cache_key(step1, row, prior, fields) == _compute_step_cache_key(
            step2, row, prior, fields
        )

    def test_provider_kwargs_changes_cache_key(self):
        """Different provider_kwargs produce different cache keys."""
        step_none = self._make_step(grounding=True)
        step_pk = self._make_step(grounding={"provider_kwargs": {"search_context_size": "high"}})

        row = {"a": 1}
        prior = {}
        fields = {"f1": {}}

        key_none = _compute_step_cache_key(step_none, row, prior, fields)
        key_pk = _compute_step_cache_key(step_pk, row, prior, fields)
        assert key_none != key_pk


# ============================================================================
# sources_field customization
# ============================================================================


class TestSourcesField:
    """Test sources_field param on LLMStep (default, custom name, None, conflict)."""

    def _make_response(self, citations=None):
        return LLMResponse(
            content='{"summary": "result"}',
            citations=citations or [],
        )

    def _make_ctx(self):
        return StepContext(
            row={"topic": "AI"},
            fields={"summary": {"prompt": "Summarize"}},
            prior_results={},
        )

    def test_default_sources_field(self):
        """Default sources_field is 'sources'."""
        step = LLMStep("s", fields={"summary": "Summarize"}, grounding=True)
        assert step.sources_field == "sources"

    def test_custom_sources_field(self):
        step = LLMStep("s", fields={"summary": "Summarize"}, grounding=True, sources_field="refs")
        assert step.sources_field == "refs"

    def test_sources_field_none(self):
        step = LLMStep("s", fields={"summary": "Summarize"}, grounding=True, sources_field=None)
        assert step.sources_field is None

    def test_sources_field_no_grounding(self):
        """sources_field is stored but silently ignored when grounding is disabled."""
        step = LLMStep("s", fields={"summary": "Summarize"}, sources_field="refs")
        assert step.sources_field == "refs"
        assert step._grounding_config is None

    def test_conflict_with_declared_field_raises(self):
        """sources_field matching a declared field name raises PipelineError."""
        with pytest.raises(PipelineError, match="sources_field 'summary' conflicts"):
            LLMStep(
                "s",
                fields={"summary": "Summarize"},
                grounding=True,
                sources_field="summary",
            )

    def test_no_conflict_when_grounding_disabled(self):
        """No conflict validation when grounding is disabled."""
        step = LLMStep("s", fields={"sources": "Find sources"}, sources_field="sources")
        assert step.sources_field == "sources"

    @pytest.mark.asyncio
    async def test_custom_name_injected(self):
        """sources_field='refs' injects citations under 'refs' key."""
        citations = [Citation(url="https://x.com", title="X", snippet="s")]
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=self._make_response(citations))

        step = LLMStep(
            "s",
            fields={"summary": "Summarize"},
            grounding=True,
            sources_field="refs",
            client=mock_client,
        )
        result = await step.run(self._make_ctx())

        assert "refs" in result.values
        assert "sources" not in result.values
        assert result.values["refs"][0]["url"] == "https://x.com"

    @pytest.mark.asyncio
    async def test_none_disables_injection(self):
        """sources_field=None means no citation injection even with citations."""
        citations = [Citation(url="https://x.com", title="X", snippet="s")]
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=self._make_response(citations))

        step = LLMStep(
            "s",
            fields={"summary": "Summarize"},
            grounding=True,
            sources_field=None,
            client=mock_client,
        )
        result = await step.run(self._make_ctx())

        assert "sources" not in result.values
        assert "refs" not in result.values
        # Only the declared field
        assert list(result.values.keys()) == ["summary"]

    @pytest.mark.asyncio
    async def test_default_sources_visible_in_output(self):
        """Default 'sources' field is NOT __ prefixed, so it's visible in output."""
        citations = [Citation(url="https://x.com", title="T", snippet="S")]
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=self._make_response(citations))

        step = LLMStep(
            "s",
            fields={"summary": "Summarize"},
            grounding=True,
            client=mock_client,
        )
        result = await step.run(self._make_ctx())

        # "sources" doesn't start with __ → visible in pipeline output
        assert "sources" in result.values
        assert not result.values["sources"][0]["url"].startswith("__")


# ============================================================================
# Export check
# ============================================================================


class TestExports:
    def test_grounding_config_importable(self):
        from accrue import GroundingConfig

        assert GroundingConfig is not None

    def test_grounding_config_in_all(self):
        import accrue

        assert "GroundingConfig" in accrue.__all__

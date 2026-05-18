"""Web search utility — factory wrapping OpenAI Responses API web search.

Returns an async callable compatible with ``FunctionStep``.  Reduces the
common two-step research-then-analyze pattern from ~30 lines to 3.

Example::

    from accrue import Pipeline, FunctionStep, LLMStep, web_search

    Pipeline([
        FunctionStep("research",
            fn=web_search("Research {company}: market position"),
            fields=["__web_context", "sources"],
        ),
        LLMStep("analyze", fields={...}, depends_on=["research"]),
    ])
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable

from ..core.exceptions import ConfigurationError
from ..steps.base import StepContext
from .logger import get_logger

logger = get_logger(__name__)

_VALID_CONTEXT_SIZES = frozenset({"low", "medium", "high"})
_VALID_TOOL_TYPES = frozenset({"web_search", "web_search_preview"})


def web_search(
    query: str,
    *,
    model: str = "gpt-4.1-mini",
    search_context_size: str = "medium",
    api_key: str | None = None,
    include_sources: bool = True,
    user_location: dict[str, str] | None = None,
    allowed_domains: list[str] | None = None,
    tool_type: str = "web_search",
) -> Callable[[StepContext], Awaitable[dict[str, Any]]]:
    """Factory returning an async callable for ``FunctionStep``.

    Args:
        query: Template string with ``{field}`` placeholders, formatted with
            ``ctx.row`` and ``ctx.prior_results``.
        model: OpenAI model for the search call. Must support web search
            (gpt-4.1-mini or gpt-4.1).
        search_context_size: ``"low"`` | ``"medium"`` | ``"high"`` — amount of
            context from search results.
        api_key: OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
        include_sources: If True, extract URL citations from the response.
        user_location: Geographic bias for search results.  Dict with optional
            keys: ``country`` (ISO 3166-1 two-letter), ``city``, ``region``,
            ``timezone`` (IANA).  Example: ``{"country": "US", "city": "New York"}``.
            A ``"type": "approximate"`` entry is added automatically.
        allowed_domains: Restrict search results to these domains (up to 100).
            Only supported with ``tool_type="web_search"`` (GA).
            Example: ``["crunchbase.com", "linkedin.com", "sec.gov"]``.
        tool_type: ``"web_search"`` (GA, cheaper at $10/1k calls) or
            ``"web_search_preview"`` (legacy, $25/1k calls for non-reasoning
            models).  GA supports domain filtering.  Defaults to ``"web_search"``.

    Returns:
        ``{"__web_context": str, "sources": list[str]}`` — the
        ``__web_context`` value is wrapped in
        ``<untrusted_web_search_results>`` tags to isolate raw search
        content from downstream LLM instructions (OWASP LLM01 mitigation).
    """
    # Eager validation
    if search_context_size not in _VALID_CONTEXT_SIZES:
        raise ValueError(
            f"search_context_size must be one of {sorted(_VALID_CONTEXT_SIZES)}, "
            f"got {search_context_size!r}"
        )
    if tool_type not in _VALID_TOOL_TYPES:
        raise ValueError(f"tool_type must be one of {sorted(_VALID_TOOL_TYPES)}, got {tool_type!r}")
    if allowed_domains and tool_type != "web_search":
        raise ValueError(
            "allowed_domains requires tool_type='web_search' (GA). "
            "Domain filtering is not supported with 'web_search_preview'."
        )

    async def _search(ctx: StepContext) -> dict[str, Any]:
        from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

        # Fix #2: Fail loud if no API key is available — before any try block so the
        # error is never swallowed by the broad except below.
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigurationError(
                "web_search requires an OpenAI API key — pass api_key= or set OPENAI_API_KEY"
            )

        # Format the query template.
        # Fix #3: Pre-escape literal braces in string row values so that row content
        # containing '{' or '}' does not crash str.format; template placeholders like
        # {company} are unaffected because we only escape values, not the template.
        template_vars = {
            k: v.replace("{", "{{").replace("}", "}}") if isinstance(v, str) else v
            for k, v in ctx.row.items()
        }
        if ctx.prior_results:
            template_vars.update(ctx.prior_results)

        try:
            formatted_query = query.format(**template_vars)
        except KeyError as exc:
            raise ValueError(f"web_search query template references missing field: {exc}") from exc

        client = AsyncOpenAI(api_key=key)

        # Build tool config
        tool_config: dict[str, Any] = {
            "type": tool_type,
            "search_context_size": search_context_size,
        }
        if user_location is not None:
            loc = dict(user_location)
            loc.setdefault("type", "approximate")
            tool_config["user_location"] = loc
        if allowed_domains:
            tool_config["filters"] = {"allowed_domains": allowed_domains}

        try:
            response = await client.responses.create(
                model=model,
                tools=[tool_config],
                input=formatted_query,
            )

            # Extract text content
            web_text = ""
            sources: list[str] = []

            for item in response.output:
                if hasattr(item, "content"):
                    # Message output item — extract text
                    for part in item.content:
                        if hasattr(part, "text"):
                            web_text = part.text
                        # Extract citations from annotations
                        if include_sources and hasattr(part, "annotations"):
                            for annotation in part.annotations:
                                if hasattr(annotation, "url"):
                                    sources.append(annotation.url)

            if not include_sources:
                sources = []

            # Fix #1: Wrap raw search output in isolation tags so downstream LLM steps
            # treat it as data rather than instructions (OWASP LLM01 prompt-injection mitigation).
            web_context = (
                f"<untrusted_web_search_results>\n{web_text}\n</untrusted_web_search_results>"
            )
            return {"__web_context": web_context, "sources": sources}

        except (APIError, RateLimitError, APITimeoutError) as exc:
            # Fix #4: Log a warning before degrading to empty context so failures are
            # visible in logs rather than silently dropped.
            logger.warning(
                "web_search degraded for row %s: %s: %s",
                getattr(ctx, "row_index", "?"),
                type(exc).__name__,
                exc,
            )
            return {"__web_context": "", "sources": []}

    return _search

"""Grounding types — config and citation models for provider-level web search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class GroundingConfig(BaseModel):
    """Configuration for provider-level web search grounding.

    Normalised in ``LLMStep.__init__``: ``True`` becomes ``GroundingConfig()``,
    a ``dict`` is validated, ``None`` / ``False`` disables grounding.

    Provider mapping:
      - ``allowed_domains``: OpenAI ``filters.allowed_domains``,
        Anthropic ``allowed_domains``.  Google does not support allow-listing
        (a warning is logged if set).
      - ``blocked_domains``: Anthropic ``blocked_domains``,
        Google ``exclude_domains``.  OpenAI does not support block-listing
        (a warning is logged if set).
      - ``user_location``: All three providers support approximate location.
      - ``max_searches``: Anthropic ``max_uses``.  Other providers ignore.

    Attributes:
        allowed_domains: Only include search results from these domains.
        blocked_domains: Never include search results from these domains.
        user_location: Approximate user location for localised results.
            Keys: ``country`` (ISO 3166-1 alpha-2), ``region``, ``city``,
            ``timezone`` (IANA).
        max_searches: Maximum number of searches per LLM call.
        provider_kwargs: Provider-specific keyword arguments passed through
            to the native tool configuration.  Useful for options that only
            one provider supports (e.g. OpenAI ``search_context_size``,
            Google ``dynamic_retrieval_config``).  These are merged into the
            provider's tool dict after Accrue's own field mappings.
    """

    model_config = ConfigDict(extra="forbid")

    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict[str, str] | None = None
    max_searches: int | None = None
    provider_kwargs: dict[str, Any] | None = None


@dataclass
class Citation:
    """A normalised source citation from a grounded LLM response.

    Provider adapters translate their native citation formats into this
    common representation.  Accrue injects these under the ``sources_field``
    name on the step result (default ``"sources"``, visible in output).
    Set ``sources_field=None`` on LLMStep to disable citation injection.

    Attributes:
        url: Source URL.
        title: Page or document title (may be empty).
        snippet: Short extract from the source (may be empty).
    """

    url: str
    title: str = ""
    snippet: str = ""

"""Google Gemini provider adapter — optional extra: pip install accrue[google]."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from ...schemas.base import UsageInfo
from ...schemas.grounding import Citation
from .base import LLMAPIError, LLMResponse

logger = logging.getLogger(__name__)

# Word-boundary regex for rate-limit detection in Option-B fallback.
# Avoids false positives on words like "iterate", "accelerator", "rated".
_RATE_LIMIT_RE = re.compile(r"\b(rate[\s_-]?limit|resource\s+exhausted)\b", re.IGNORECASE)

# Lazily resolved; populated once on first exception classification.
# google-api-core ships with google-genai so it will normally be present,
# but we don't declare it as a hard dependency.
try:
    import google.api_core.exceptions as _gax_exc
except ImportError:  # pragma: no cover
    _gax_exc = None  # type: ignore[assignment]


class GoogleClient:
    """Adapter for Google's Gemini models.

    Requires: pip install accrue[google]

    Supports the ``google_search`` grounding tool.  When grounding is active,
    structured outputs (``response_json_schema``) are disabled because the
    combination is unsupported on Gemini 2.x models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        http_client: Any | None = None,
    ):
        self._api_key = api_key
        self._http_client = http_client
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai package required: pip install accrue[google]")
            key = self._api_key or os.environ.get("GOOGLE_API_KEY")
            kwargs: dict[str, Any] = {"api_key": key}
            if self._http_client is not None:
                kwargs["http_options"] = {"client": self._http_client}
            self._client = genai.Client(**kwargs)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Convert messages to Gemini format
        system_instruction = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        try:
            from google.genai import types

            config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_instruction:
                config["system_instruction"] = system_instruction

            # Translate tools (e.g. web_search → google_search)
            gemini_tools = _translate_tools(tools, types) if tools else None
            if gemini_tools:
                config["tools"] = gemini_tools

            if response_format:
                fmt_type = response_format.get("type")
                if fmt_type == "json_schema":
                    # Structured outputs: extract schema, pass as response_json_schema
                    # IMPORTANT: incompatible with google_search on Gemini 2.x
                    if not gemini_tools:
                        inner = response_format.get("json_schema", {})
                        schema = inner.get("schema", {})
                        config["response_mime_type"] = "application/json"
                        if schema:
                            config["response_json_schema"] = schema
                    else:
                        config["response_mime_type"] = "application/json"
                elif fmt_type == "json_object":
                    config["response_mime_type"] = "application/json"

            # Merge provider-specific kwargs into config
            if provider_kwargs:
                config.update(provider_kwargs)

            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
        except Exception as exc:
            raise _classify_google_exc(exc, model) from exc

        content = response.text or ""

        # Extract citations from grounding metadata
        citations = _extract_citations(response) if gemini_tools else []

        usage = None
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage, citations=citations)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_google_exc(exc: Exception, model: str) -> LLMAPIError:
    """Classify a Google SDK exception into a typed LLMAPIError.

    Tries typed ``google.api_core.exceptions`` first (Option A); falls back to
    substring heuristics (Option B) for unknown exception types or when the
    ``google-api-core`` package is unavailable.
    """
    # Option A — typed classification via google.api_core.exceptions
    if _gax_exc is not None:
        if isinstance(exc, _gax_exc.ResourceExhausted):
            return LLMAPIError(
                f"Google rate limit for model '{model}': {exc}",
                is_rate_limit=True,
                status_code=429,
            )
        if isinstance(
            exc,
            (
                _gax_exc.ServiceUnavailable,
                getattr(_gax_exc, "InternalServerError", type(None)),
                getattr(_gax_exc, "BadGateway", type(None)),
            ),
        ):
            return LLMAPIError(
                f"Google server error for model '{model}': {exc}",
                status_code=503,
            )
        if isinstance(exc, _gax_exc.DeadlineExceeded):
            return LLMAPIError(
                f"Google deadline exceeded for model '{model}': {exc}",
                retryable=True,
                status_code=408,
            )
        if isinstance(
            exc,
            (
                getattr(_gax_exc, "InvalidArgument", type(None)),
                getattr(_gax_exc, "PermissionDenied", type(None)),
                getattr(_gax_exc, "NotFound", type(None)),
            ),
        ):
            return LLMAPIError(
                f"Google API error for model '{model}': {exc}",
                status_code=400,
            )

    # Option B — substring heuristics (fallback for unknown types)
    exc_str = str(exc)
    if "429" in exc_str or _RATE_LIMIT_RE.search(exc_str):
        return LLMAPIError(
            f"Google rate limit for model '{model}': {exc}",
            is_rate_limit=True,
            status_code=429,
        )
    exc_str_lower = exc_str.lower()
    for code, marker in [
        (500, "internal server error"),
        (502, "bad gateway"),
        (503, "service unavailable"),
        (504, "gateway timeout"),
    ]:
        if marker in exc_str_lower:
            return LLMAPIError(
                f"Google server error for model '{model}': {exc}",
                status_code=code,
            )
    if "deadline exceeded" in exc_str_lower or "timeout" in exc_str_lower:
        return LLMAPIError(
            f"Google timeout for model '{model}': {exc}",
            retryable=True,
            status_code=408,
        )
    return LLMAPIError(f"Google API error for model '{model}': {exc}")


def _translate_tools(tools: list[dict[str, Any]], types: Any) -> list[Any]:
    """Translate generic Accrue tool dicts to Gemini Tool objects."""
    gemini_tools: list[Any] = []
    for tool in tools:
        if tool.get("type") == "web_search":
            gs_kwargs: dict[str, Any] = {}
            # Google supports exclude_domains (mapped from blocked_domains)
            if "blocked_domains" in tool:
                gs_kwargs["exclude_domains"] = tool["blocked_domains"]
            # Google does not support allowed_domains
            if "allowed_domains" in tool:
                logger.warning(
                    "GoogleClient: allowed_domains is not supported by Google Search. "
                    "The parameter will be ignored."
                )
            # Merge provider-specific kwargs (pass-through)
            if "provider_kwargs" in tool:
                gs_kwargs.update(tool["provider_kwargs"])
            gemini_tools.append(types.Tool(google_search=types.GoogleSearch(**gs_kwargs)))
    return gemini_tools


def _extract_citations(response: Any) -> list[Citation]:
    """Extract citations from Gemini grounding metadata."""
    citations: list[Citation] = []
    seen_urls: set[str] = set()

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return citations

    metadata = getattr(candidates[0], "grounding_metadata", None)
    if not metadata:
        return citations

    chunks = getattr(metadata, "grounding_chunks", None)
    if not chunks:
        return citations

    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if not web:
            continue
        url = getattr(web, "uri", "") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            citations.append(
                Citation(
                    url=url,
                    title=getattr(web, "title", "") or "",
                )
            )
    return citations

"""Anthropic provider adapter — optional extra: pip install accrue[anthropic].

Batch API support: Anthropic Message Batches API for 50% cost savings.
Implements ``submit_batch()``, ``poll_batch()``, and ``cancel_batch()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from ...core.exceptions import StepError
from ...schemas.base import UsageInfo
from ...schemas.grounding import Citation
from .base import BatchRequest, BatchResult, LLMAPIError, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Adapter for Anthropic's Claude models.

    Requires: pip install accrue[anthropic]

    Supports the ``web_search_20250305`` server tool for grounded responses.
    When web search tools are active, structured outputs via
    ``output_config.format`` are disabled (incompatible with citations).
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install accrue[anthropic]")
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = AsyncAnthropic(api_key=key)
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

        # Separate system message from conversation messages
        system_content = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system_content:
            # Prompt caching: wrap system content in a content block with
            # cache_control so Anthropic caches the system prompt prefix.
            # Rows 2-N pay 0.1x on the cached system prompt (90% savings).
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Translate tools (e.g. web_search → web_search_20250305 server tool)
        anthropic_tools = _translate_tools(tools) if tools else None
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        # Structured outputs: Anthropic uses output_config.format (GA)
        # json_schema → constrained decoding; json_object → no equivalent, skip
        # IMPORTANT: output_config.format is incompatible with web search citations
        if not anthropic_tools and response_format and response_format.get("type") == "json_schema":
            inner = response_format.get("json_schema", {})
            schema = inner.get("schema", {})
            if schema:
                kwargs["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }

        # Merge provider-specific kwargs (escape hatch for new features
        # like thinking, effort, etc.)
        if provider_kwargs:
            kwargs.update(provider_kwargs)

        try:
            from anthropic import APIError, APITimeoutError, RateLimitError

            response = await client.messages.create(**kwargs)
        except RateLimitError as exc:
            raise LLMAPIError(
                f"Anthropic rate limit for model '{model}': {exc}",
                status_code=429,
                is_rate_limit=True,
            ) from exc
        except APITimeoutError as exc:
            raise LLMAPIError(
                f"Anthropic timeout for model '{model}': {exc}",
                status_code=408,
            ) from exc
        except APIError as exc:
            raise LLMAPIError(
                f"Anthropic API error for model '{model}': {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        # Extract text from potentially multi-block response
        content = _extract_text(response)

        # Extract citations from web_search_result_location blocks
        citations = _extract_citations(response) if anthropic_tools else []

        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage, citations=citations)

    # -- Batch API ---------------------------------------------------------

    async def submit_batch(
        self,
        requests: list[BatchRequest],
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Submit a batch via the Anthropic Message Batches API.

        Args:
            requests: Batch requests to submit.
            metadata: Optional metadata (stored in first request custom_id prefix).

        Returns:
            The Anthropic batch ID.
        """
        client = self._get_client()

        anthropic_requests = []
        for req in requests:
            # Separate system from messages (Anthropic format)
            system_content = ""
            chat_messages = []
            for msg in req.messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    chat_messages.append(msg)

            params: dict[str, Any] = {
                "model": req.model,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "messages": chat_messages,
            }
            if system_content:
                # Prompt caching for batch requests
                params["system"] = [
                    {
                        "type": "text",
                        "text": system_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            # Structured outputs
            if req.response_format and req.response_format.get("type") == "json_schema":
                inner = req.response_format.get("json_schema", {})
                schema = inner.get("schema", {})
                if schema:
                    params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

            # Forward provider_kwargs (e.g. thinking, effort)
            if req.provider_kwargs:
                params.update(req.provider_kwargs)

            anthropic_requests.append(
                {
                    "custom_id": req.custom_id,
                    "params": params,
                }
            )

        try:
            batch = await client.messages.batches.create(requests=anthropic_requests)
            logger.info("Anthropic batch submitted: %s (%d requests)", batch.id, len(requests))
            return batch.id
        except Exception as exc:
            raise LLMAPIError(
                f"Anthropic batch submission failed: {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

    async def poll_batch(
        self,
        batch_id: str,
        poll_interval: float = 60.0,
        timeout: float = 86400.0,
    ) -> BatchResult:
        """Poll an Anthropic batch until completion or timeout.

        Args:
            batch_id: Batch ID from ``submit_batch()``.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            Aggregated batch result.

        Raises:
            StepError: On failure or timeout.
        """
        client = self._get_client()
        start = time.monotonic()

        while True:
            try:
                batch = await client.messages.batches.retrieve(batch_id)
            except Exception as exc:
                raise StepError(
                    f"Anthropic batch status check failed for {batch_id}: {exc}",
                    step_name="batch",
                ) from exc

            status = batch.processing_status
            elapsed = time.monotonic() - start

            if status == "ended":
                logger.info("Anthropic batch %s ended (%.0fs elapsed)", batch_id, elapsed)
                return await self._collect_batch_results(client, batch_id)

            if elapsed > timeout:
                raise StepError(
                    f"Anthropic batch {batch_id} timed out after {elapsed:.0f}s "
                    f"(status={status}). Check the Anthropic dashboard with "
                    f"batch ID {batch_id}.",
                    step_name="batch",
                )

            logger.info(
                "Anthropic batch %s status=%s (%.0fs elapsed), next check in %.0fs",
                batch_id,
                status,
                elapsed,
                poll_interval,
            )
            await asyncio.sleep(poll_interval)

    async def cancel_batch(self, batch_id: str) -> None:
        """Best-effort cancel an Anthropic batch.

        Args:
            batch_id: Batch ID to cancel.
        """
        try:
            client = self._get_client()
            await client.messages.batches.cancel(batch_id)
            logger.info("Anthropic batch %s cancel requested", batch_id)
        except Exception:
            logger.warning("Failed to cancel Anthropic batch %s", batch_id, exc_info=True)

    async def _collect_batch_results(self, client: Any, batch_id: str) -> BatchResult:
        """Stream and parse results from a completed Anthropic batch."""
        responses: dict[str, LLMResponse] = {}
        failed_ids: list[str] = []
        errors: dict[str, str] = {}

        async for entry in await client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            result = entry.result

            if result.type == "succeeded":
                message = result.message
                content = _extract_text(message)
                usage = None
                if message.usage:
                    usage = UsageInfo(
                        prompt_tokens=message.usage.input_tokens,
                        completion_tokens=message.usage.output_tokens,
                        total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                        model=message.model,
                    )
                responses[custom_id] = LLMResponse(content=content, usage=usage)
            else:
                failed_ids.append(custom_id)
                error_msg = getattr(result, "error", {})
                if hasattr(error_msg, "message"):
                    error_msg = error_msg.message
                errors[custom_id] = str(error_msg) if error_msg else f"result type: {result.type}"

        return BatchResult(
            responses=responses,
            failed_ids=failed_ids,
            batch_id=batch_id,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate generic Accrue tool dicts to Anthropic server tool format."""
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "web_search":
            server_tool: dict[str, Any] = {
                "type": "web_search_20250305",
                "name": "web_search",
            }
            # Map config fields
            if "allowed_domains" in tool:
                server_tool["allowed_domains"] = tool["allowed_domains"]
            if "blocked_domains" in tool:
                server_tool["blocked_domains"] = tool["blocked_domains"]
            if "user_location" in tool:
                loc = tool["user_location"]
                server_tool["user_location"] = {"type": "approximate", **loc}
            if "max_searches" in tool:
                server_tool["max_uses"] = tool["max_searches"]
            # Merge provider-specific kwargs (pass-through)
            if "provider_kwargs" in tool:
                server_tool.update(tool["provider_kwargs"])
            anthropic_tools.append(server_tool)
    return anthropic_tools


def _extract_text(response: Any) -> str:
    """Extract all text content from an Anthropic response (may have multiple blocks)."""
    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _extract_citations(response: Any) -> list[Citation]:
    """Extract web_search_result_location citations from an Anthropic response."""
    citations: list[Citation] = []
    seen_urls: set[str] = set()
    for block in response.content:
        if getattr(block, "type", None) != "text":
            continue
        block_citations = getattr(block, "citations", None)
        if not block_citations:
            continue
        for cite in block_citations:
            if getattr(cite, "type", None) == "web_search_result_location":
                url = getattr(cite, "url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    citations.append(
                        Citation(
                            url=url,
                            title=getattr(cite, "title", ""),
                            snippet=getattr(cite, "cited_text", ""),
                        )
                    )
    return citations

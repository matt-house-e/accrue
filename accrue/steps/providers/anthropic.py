"""Anthropic provider adapter — optional extra: pip install accrue[anthropic].

Batch API support: Anthropic Message Batches API for 50% cost savings.
Implements ``submit_batch()``, ``poll_batch()``, and ``cancel_batch()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
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

    def __init__(
        self,
        api_key: str | None = None,
        http_client: Any | None = None,
    ):
        self._api_key = api_key
        self._http_client = http_client
        self._client: Any = None
        # Scope: per-client-instance (not per-process).  LLMStep constructs a
        # fresh AnthropicClient on each Pipeline.run_async() call, so users see
        # this warning once per pipeline run rather than once per process lifetime.
        self._warned_grounding_schema: bool = False
        self._warned_temperature_dropped: bool = False

    def _warn_temperature_dropped(self, model: str, temperature: float | None) -> None:
        """Warn once when a requested temperature is dropped for *model*.

        The adapter cannot tell a user-set value from ``EnrichmentConfig``'s
        default — both arrive as a plain float — so the trigger is whether the
        drop actually changes anything.  These models sample at
        :data:`_IMPLICIT_TEMPERATURE` when the parameter is omitted, so dropping
        that exact value is a no-op and stays silent.

        Args:
            model: Model the request is being made against.
            temperature: The resolved temperature, or ``None`` if none was set.
        """
        if temperature is None or temperature == _IMPLICIT_TEMPERATURE:
            return
        if self._warned_temperature_dropped:
            return
        logger.warning(
            "Model '%s' does not accept an explicit temperature; the requested "
            "value %s was dropped and the model's default of %s applies. Set "
            "temperature=%s to silence this warning, or use a model that supports "
            "temperature (e.g. claude-sonnet-4-5) if you need deterministic sampling.",
            model,
            temperature,
            _IMPLICIT_TEMPERATURE,
            _IMPLICIT_TEMPERATURE,
        )
        self._warned_temperature_dropped = True

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install accrue[anthropic]")
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise StepError(
                    "No Anthropic API key found. Pass api_key= to "
                    "AnthropicClient() or set ANTHROPIC_API_KEY in the "
                    "environment before constructing the client."
                )
            kwargs: dict[str, Any] = {"api_key": key}
            if self._http_client is not None:
                kwargs["http_client"] = self._http_client
            # max_retries=0: disable SDK-level retry so LLMStep's retry loop
            # is the single source of truth and retries are not double-stacked.
            kwargs["max_retries"] = 0
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None,
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
            max_tokens=max_tokens,
        )
        # Omit temperature for models that reject an explicit value (issue #109).
        # The warning is deferred until after provider_kwargs is merged below,
        # so the documented escape hatch does not trigger a "value was dropped"
        # warning about a value it puts straight back.
        if temperature is not None and _supports_temperature(model):
            kwargs["temperature"] = temperature

        if system_content:
            # Prompt caching: wrap system content in a content block with
            # cache_control so Anthropic caches the system prompt prefix.
            # Rows 2-N pay 0.1x on the cached system prompt (90% savings).
            #
            # This only pays off because LLMStep keeps row-specific content out
            # of the system message (see accrue.steps.prompt_builder). Anything
            # per-row in here makes the prefix change every call, so every call
            # writes a fresh entry at 1.25x and none ever reads one.
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
        if anthropic_tools and response_format and response_format.get("type") == "json_schema":
            if not self._warned_grounding_schema:
                logger.warning(
                    "Anthropic grounding (web search tools) is incompatible with strict "
                    "structured outputs; structured output schema will not be enforced for "
                    "this request. Consider running a separate non-grounded LLMStep over "
                    "the grounded context, or disable grounding for this step."
                )
                self._warned_grounding_schema = True
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

        # Warn only when no temperature reaches the API at all (issue #109).
        if "temperature" not in kwargs:
            self._warn_temperature_dropped(model, temperature)

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
            exc_status = getattr(exc, "status_code", None)
            # Only claim a temperature problem when we actually sent one.
            hint = _temperature_hint(model, exc) if "temperature" in kwargs else ""
            # Promote generic 429 to is_rate_limit (covers cases where RateLimitError
            # is not raised but status_code is 429)
            raise LLMAPIError(
                f"Anthropic API error for model '{model}': {exc}{hint}",
                status_code=exc_status,
                is_rate_limit=(exc_status == 429),
            ) from exc

        # Extract text from potentially multi-block response
        content = _extract_text(response)

        # Extract citations from web_search_result_location blocks
        citations = _extract_citations(response) if anthropic_tools else []

        usage = None
        if response.usage:
            u = response.usage
            cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
            usage = UsageInfo(
                prompt_tokens=u.input_tokens,
                completion_tokens=u.output_tokens,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
                total_tokens=u.input_tokens + u.output_tokens + cache_write + cache_read,
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

        # Validate custom_id uniqueness before touching the network
        seen: set[str] = set()
        for req in requests:
            if not req.custom_id:
                raise ValueError("BatchRequest.custom_id must be non-empty")
            if req.custom_id in seen:
                raise ValueError(f"Duplicate custom_id in batch: {req.custom_id!r}")
            seen.add(req.custom_id)

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
                "messages": chat_messages,
            }
            # Same temperature rule as the realtime path (issue #109), including
            # deferring the warning until after provider_kwargs is merged below.
            if req.temperature is not None and _supports_temperature(req.model):
                params["temperature"] = req.temperature
            if system_content:
                # Prompt caching for batch requests — same static-prefix
                # requirement as the realtime path above.
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

            # Warn only when no temperature reaches the API at all (issue #109).
            if "temperature" not in params:
                self._warn_temperature_dropped(req.model, req.temperature)

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
            # Name a model we actually sent a temperature for.  accrue's own
            # batches are single-model, but submit_batch() is public and a
            # caller can mix them, so do not assume requests[0] is the culprit.
            culprit = next(
                (r["params"]["model"] for r in anthropic_requests if "temperature" in r["params"]),
                None,
            )
            hint = _temperature_hint(culprit, exc) if culprit else ""
            raise LLMAPIError(
                f"Anthropic batch submission failed: {exc}{hint}",
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

        try:
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
        except (asyncio.CancelledError, KeyboardInterrupt):
            # User cancelled — best-effort cancel the upstream batch to avoid
            # orphaning a billable job.
            try:
                await self.cancel_batch(batch_id)
            except Exception:
                pass  # cancel_batch is best-effort; don't mask the original interruption
            raise

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
                    u = message.usage
                    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
                    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                    usage = UsageInfo(
                        prompt_tokens=u.input_tokens,
                        completion_tokens=u.output_tokens,
                        cache_write_tokens=cache_write,
                        cache_read_tokens=cache_read,
                        total_tokens=u.input_tokens + u.output_tokens + cache_write + cache_read,
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


# Anthropic models that reject an explicit ``temperature`` (issue #109).
#
# The Claude 5 family and Claude Opus 4.7/4.8 removed the sampling parameters.
# Any explicit value other than the model's own default returns
# ``400 - `temperature` is deprecated for this model.``  Omitting the parameter
# is always accepted, so the fix is to not send it at all.
#
# Matched on model-family prefix rather than an exhaustive list of dated model
# ids, so unreleased Claude 5 variants and vendor-prefixed ids
# (``anthropic.claude-sonnet-5``, ``us.anthropic.claude-sonnet-5-v1:0``) are
# covered without a code change.  The first branch requires an alphabetic tier
# segment between ``claude-`` and the major version, so Claude 3/4 ids that
# merely contain a 5 do not match: ``claude-3-5-sonnet-20241022`` (digit after
# ``claude-``), ``claude-sonnet-4-5`` and ``claude-haiku-4-5`` (major version 4).
_NO_EXPLICIT_TEMPERATURE = re.compile(r"claude-(?:[a-z]+-5|opus-4-[78])(?!\d)")

# The value these models use when ``temperature`` is omitted.  Dropping a
# request for this exact value changes nothing, so it is not worth warning about.
_IMPLICIT_TEMPERATURE = 1.0


def _supports_temperature(model: str) -> bool:
    """Whether *model* accepts an explicit ``temperature`` parameter.

    Args:
        model: Anthropic model identifier, optionally vendor-prefixed.

    Returns:
        ``False`` for models that reject an explicit value (Claude 5 family,
        Claude Opus 4.7/4.8), ``True`` otherwise.
    """
    return _NO_EXPLICIT_TEMPERATURE.search(model.lower()) is None


def _temperature_hint(model: str, exc: Exception) -> str:
    """Actionable suffix for API errors caused by an unsupported ``temperature``.

    Covers models newer than :func:`_supports_temperature` knows about: the raw
    SDK message names a model capability, which reads like an API problem rather
    than an accrue one.

    Args:
        model: Model the request was made against.
        exc: The provider exception.

    Returns:
        A hint string, or ``""`` when the error is unrelated to temperature.
    """
    if "temperature" not in str(exc).lower():
        return ""
    return (
        f" — the model '{model}' rejects an explicit temperature. accrue omits it "
        "automatically for the model families known to have removed it (Claude 5, "
        f"Claude Opus 4.7/4.8); please report '{model}' at "
        "https://github.com/matt-house-e/accrue/issues so the check can cover it."
    )


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

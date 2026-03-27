"""OpenAI provider adapter — default, covers all OpenAI-compatible APIs.

Uses the Responses API (``client.responses.create``).  For OpenAI-compatible
third-party providers that only expose the Chat Completions endpoint (Ollama,
Groq, etc.), the adapter falls back to ``client.chat.completions.create``
when a ``base_url`` is configured.

Batch API support: native OpenAI (no ``base_url``) supports the Batch API
for 50% cost savings.  The adapter implements ``submit_batch()``,
``poll_batch()``, and ``cancel_batch()`` using JSONL file upload with the
``/v1/chat/completions`` endpoint.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from typing import Any

from ...core.exceptions import StepError
from ...schemas.base import UsageInfo
from ...schemas.grounding import Citation
from .base import BatchRequest, BatchResult, LLMAPIError, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Adapter for OpenAI and OpenAI-compatible providers.

    Native OpenAI (no ``base_url``) uses the Responses API which supports
    web search tools, structured output via ``text.format``, and inline
    citations.

    Third-party providers with a ``base_url`` (Ollama, Groq, DeepSeek,
    Together, Fireworks, vLLM, Mistral, LM Studio) use the Chat Completions
    API for maximum compatibility.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._http_client = http_client
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            kwargs: dict[str, Any] = {"api_key": key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._http_client is not None:
                kwargs["http_client"] = self._http_client
            self._client = AsyncOpenAI(**kwargs)
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
        if self._base_url:
            return await self._complete_chat(
                messages,
                model,
                temperature,
                max_tokens,
                response_format,
                provider_kwargs=provider_kwargs,
            )
        return await self._complete_responses(
            messages,
            model,
            temperature,
            max_tokens,
            response_format,
            tools,
            provider_kwargs=provider_kwargs,
        )

    # -- Responses API (native OpenAI) ------------------------------------

    async def _complete_responses(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from openai import APIError, APITimeoutError, RateLimitError

        client = self._get_client()

        # Separate system/instructions from conversation input
        instructions: str | None = None
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            else:
                input_items.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict[str, Any] = dict(
            model=model,
            input=input_items,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if instructions:
            kwargs["instructions"] = instructions

        # Structured outputs: response_format → text.format
        if response_format:
            kwargs["text"] = {"format": _translate_response_format(response_format)}

        # Tools (e.g. web_search)
        if tools:
            kwargs["tools"] = _translate_tools(tools)

        # Merge provider-specific kwargs (escape hatch for new features)
        if provider_kwargs:
            kwargs.update(provider_kwargs)

        try:
            response = await client.responses.create(**kwargs)
        except RateLimitError as exc:
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after_header = exc.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except (ValueError, TypeError):
                        pass
            raise LLMAPIError(
                f"OpenAI rate limit for model '{model}': {exc}",
                status_code=429,
                retry_after=retry_after,
                is_rate_limit=True,
            ) from exc
        except APITimeoutError as exc:
            raise LLMAPIError(
                f"OpenAI timeout for model '{model}': {exc}",
                status_code=408,
            ) from exc
        except APIError as exc:
            raise LLMAPIError(
                f"OpenAI API error for model '{model}': {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        # Extract content
        content = response.output_text or ""

        # Extract citations from url_citation annotations
        citations = _extract_citations(response)

        # Extract usage
        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage, citations=citations)

    # -- Chat Completions API (base_url providers) ------------------------

    async def _complete_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from openai import APIError, APITimeoutError, RateLimitError

        client = self._get_client()
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        # Merge provider-specific kwargs (escape hatch for new features)
        if provider_kwargs:
            kwargs.update(provider_kwargs)

        try:
            response = await client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after_header = exc.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except (ValueError, TypeError):
                        pass
            raise LLMAPIError(
                f"OpenAI rate limit for model '{model}': {exc}",
                status_code=429,
                retry_after=retry_after,
                is_rate_limit=True,
            ) from exc
        except APITimeoutError as exc:
            raise LLMAPIError(
                f"OpenAI timeout for model '{model}': {exc}",
                status_code=408,
            ) from exc
        except APIError as exc:
            raise LLMAPIError(
                f"OpenAI API error for model '{model}': {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage)

    # -- Batch API ---------------------------------------------------------

    async def submit_batch(
        self,
        requests: list[BatchRequest],
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Submit a batch of requests via the OpenAI Batch API.

        Builds a JSONL file with one request per line, uploads it, and
        creates a batch job.  Uses the ``/v1/chat/completions`` endpoint
        (the Batch API does not support Responses API format).

        Args:
            requests: Batch requests to submit.
            metadata: Optional key-value metadata attached to the batch.

        Returns:
            The OpenAI batch job ID.
        """
        from openai import APIError

        client = self._get_client()

        # Build JSONL content
        jsonl_lines: list[str] = []
        for req in requests:
            body: dict[str, Any] = {
                "model": req.model,
                "messages": req.messages,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            }
            if req.response_format:
                body["response_format"] = req.response_format
            # Forward provider_kwargs into batch request body
            if req.provider_kwargs:
                body.update(req.provider_kwargs)
            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            jsonl_lines.append(json.dumps(line, separators=(",", ":")))

        jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")

        try:
            # Upload JSONL file
            uploaded = await client.files.create(
                file=("batch_input.jsonl", io.BytesIO(jsonl_bytes)),
                purpose="batch",
            )

            # Create batch job
            batch_kwargs: dict[str, Any] = {
                "input_file_id": uploaded.id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }
            if metadata:
                batch_kwargs["metadata"] = metadata

            batch = await client.batches.create(**batch_kwargs)
            logger.info("OpenAI batch submitted: %s (%d requests)", batch.id, len(requests))
            return batch.id

        except APIError as exc:
            raise LLMAPIError(
                f"OpenAI batch submission failed: {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

    async def poll_batch(
        self,
        batch_id: str,
        poll_interval: float = 60.0,
        timeout: float = 86400.0,
    ) -> BatchResult:
        """Poll an OpenAI batch job until completion or timeout.

        Args:
            batch_id: Batch job ID from ``submit_batch()``.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            Aggregated batch result.

        Raises:
            StepError: On failure, expiration, or timeout.
        """
        from openai import APIError

        client = self._get_client()
        start = time.monotonic()

        while True:
            try:
                batch = await client.batches.retrieve(batch_id)
            except APIError as exc:
                raise StepError(
                    f"OpenAI batch status check failed for {batch_id}: {exc}",
                    step_name="batch",
                ) from exc

            status = batch.status
            elapsed = time.monotonic() - start

            if status == "completed":
                logger.info("OpenAI batch %s completed (%.0fs elapsed)", batch_id, elapsed)
                return await self._download_batch_results(batch, batch_id)

            if status in ("failed", "expired", "cancelled"):
                error_info = ""
                if hasattr(batch, "errors") and batch.errors:
                    error_data = getattr(batch.errors, "data", [])
                    if error_data:
                        error_info = (
                            f" Errors: {[getattr(e, 'message', str(e)) for e in error_data[:3]]}"
                        )
                raise StepError(
                    f"OpenAI batch {batch_id} {status} after {elapsed:.0f}s.{error_info} "
                    f"Check the OpenAI dashboard for details.",
                    step_name="batch",
                )

            if elapsed > timeout:
                raise StepError(
                    f"OpenAI batch {batch_id} timed out after {elapsed:.0f}s "
                    f"(status={status}). The batch may still be processing — "
                    f"check the OpenAI dashboard with batch ID {batch_id}.",
                    step_name="batch",
                )

            logger.info(
                "OpenAI batch %s status=%s (%.0fs elapsed), next check in %.0fs",
                batch_id,
                status,
                elapsed,
                poll_interval,
            )
            await asyncio.sleep(poll_interval)

    async def cancel_batch(self, batch_id: str) -> None:
        """Best-effort cancel an OpenAI batch job.

        Args:
            batch_id: Batch job ID to cancel.
        """
        try:
            client = self._get_client()
            await client.batches.cancel(batch_id)
            logger.info("OpenAI batch %s cancel requested", batch_id)
        except Exception:
            logger.warning("Failed to cancel OpenAI batch %s", batch_id, exc_info=True)

    async def _download_batch_results(self, batch: Any, batch_id: str) -> BatchResult:
        """Download and parse the output file from a completed batch."""
        client = self._get_client()
        responses: dict[str, LLMResponse] = {}
        failed_ids: list[str] = []
        errors: dict[str, str] = {}

        if not batch.output_file_id:
            return BatchResult(
                responses=responses,
                failed_ids=failed_ids,
                batch_id=batch_id,
                errors=errors,
            )

        file_content = await client.files.content(batch.output_file_id)
        raw_text = file_content.text if hasattr(file_content, "text") else str(file_content)

        for line in raw_text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = entry.get("custom_id", "")
            resp_body = entry.get("response", {})
            status_code = resp_body.get("status_code", 0)

            if status_code == 200:
                body = resp_body.get("body", {})
                choices = body.get("choices", [])
                content = ""
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

                usage_data = body.get("usage", {})
                usage = UsageInfo(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                    model=body.get("model", ""),
                )
                responses[custom_id] = LLMResponse(content=content, usage=usage)
            else:
                failed_ids.append(custom_id)
                error_body = resp_body.get("body", {})
                error_msg = error_body.get("error", {}).get("message", f"status {status_code}")
                errors[custom_id] = error_msg

        # Also check error file for additional failures
        if batch.error_file_id:
            try:
                err_content = await client.files.content(batch.error_file_id)
                err_text = err_content.text if hasattr(err_content, "text") else str(err_content)
                for line in err_text.strip().split("\n"):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    cid = entry.get("custom_id", "")
                    if cid and cid not in errors:
                        failed_ids.append(cid)
                        errors[cid] = entry.get("error", {}).get("message", "unknown error")
            except Exception:
                logger.warning("Failed to read batch error file for %s", batch_id)

        return BatchResult(
            responses=responses,
            failed_ids=failed_ids,
            batch_id=batch_id,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translate_response_format(response_format: dict[str, Any]) -> dict[str, Any]:
    """Translate Chat Completions ``response_format`` to Responses API ``text.format``.

    Chat Completions nests the schema under ``json_schema.schema``;
    the Responses API flattens it into the format dict directly.
    """
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        inner = response_format.get("json_schema", {})
        return {
            "type": "json_schema",
            "name": inner.get("name", "response"),
            "strict": inner.get("strict", True),
            "schema": inner.get("schema", {}),
        }
    if fmt_type == "json_object":
        return {"type": "json_object"}
    return {"type": "text"}


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate generic grounding tool dicts to OpenAI Responses API format.

    Maps cross-provider fields from :class:`GroundingConfig` into the native
    OpenAI ``web_search`` tool structure:

    - ``allowed_domains`` → ``filters.allowed_domains``
    - ``blocked_domains`` → *unsupported* (logged warning, dropped)
    - ``user_location``   → ``user_location`` with ``{"type": "approximate", ...}``
    - ``max_searches``    → *unsupported* (logged warning, dropped)
    - ``provider_kwargs`` → merged at top level (e.g. ``search_context_size``)
    """
    translated: list[dict[str, Any]] = []
    for tool in tools:
        out = dict(tool)

        # -- provider_kwargs: merge at top level -------------------------
        pk = out.pop("provider_kwargs", None)
        if pk:
            out.update(pk)

        # -- allowed_domains → filters.allowed_domains -------------------
        allowed = out.pop("allowed_domains", None)
        if allowed:
            out.setdefault("filters", {})["allowed_domains"] = allowed

        # -- blocked_domains: not supported by OpenAI --------------------
        blocked = out.pop("blocked_domains", None)
        if blocked:
            logger.warning(
                "OpenAI web_search does not support blocked_domains; this setting will be ignored."
            )

        # -- user_location → {"type": "approximate", ...} ---------------
        location = out.pop("user_location", None)
        if location:
            out["user_location"] = {"type": "approximate", **location}

        # -- max_searches: not supported by OpenAI -----------------------
        max_searches = out.pop("max_searches", None)
        if max_searches is not None:
            logger.warning(
                "OpenAI web_search does not support max_searches; this setting will be ignored."
            )

        translated.append(out)
    return translated


def _extract_citations(response: Any) -> list[Citation]:
    """Extract url_citation annotations from a Responses API response."""
    citations: list[Citation] = []
    seen_urls: set[str] = set()
    for item in response.output:
        if not hasattr(item, "content"):
            continue
        for part in item.content:
            if not hasattr(part, "annotations"):
                continue
            for annotation in part.annotations:
                if getattr(annotation, "type", None) == "url_citation":
                    url = getattr(annotation, "url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append(
                            Citation(
                                url=url,
                                title=getattr(annotation, "title", ""),
                            )
                        )
    return citations

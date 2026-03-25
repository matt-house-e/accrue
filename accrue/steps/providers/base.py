"""LLMClient protocol, batch types, and LLMResponse — provider-agnostic LLM interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...schemas.base import UsageInfo
from ...schemas.grounding import Citation


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The text content of the response.
        usage: Token usage information.
        citations: Normalised source citations when grounding tools were used.
    """

    content: str
    usage: UsageInfo | None = None
    citations: list[Citation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Batch API types
# ---------------------------------------------------------------------------


@dataclass
class BatchRequest:
    """A single request within a batch submission.

    Attributes:
        custom_id: Unique identifier to correlate request with response
            (e.g. ``"row-42"``).
        messages: Chat messages in the standard ``[{"role": ..., "content": ...}]``
            format.
        model: Model identifier (e.g. ``"gpt-4.1-mini"``).
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        response_format: Optional structured output format dict.
        tools: Optional tool definitions (e.g. web search).
    """

    custom_id: str
    messages: list[dict[str, Any]]
    model: str
    temperature: float
    max_tokens: int
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    provider_kwargs: dict[str, Any] | None = None


@dataclass
class BatchResult:
    """Aggregated result from a completed batch job.

    Attributes:
        responses: Mapping of ``custom_id`` to the provider's ``LLMResponse``.
        failed_ids: ``custom_id`` values for requests that failed.
        batch_id: Provider batch job identifier (for debugging / dashboard
            correlation).
        errors: Mapping of ``custom_id`` to error message for failed requests.
    """

    responses: dict[str, LLMResponse] = field(default_factory=dict)
    failed_ids: list[str] = field(default_factory=list)
    batch_id: str = ""
    errors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMAPIError(Exception):
    """Provider-agnostic API error for retry logic.

    Wraps provider-specific errors (openai.RateLimitError, anthropic.RateLimitError, etc.)
    so LLMStep retry logic doesn't need to know about specific SDKs.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        is_rate_limit: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.is_rate_limit = is_rate_limit


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Protocol all LLM provider adapters must satisfy."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class BatchCapableLLMClient(LLMClient, Protocol):
    """Extended protocol for providers that support batch API operations.

    Providers that implement this protocol can submit batch jobs (JSONL upload,
    async polling, result download) for 50% cost savings on supported models.
    The pipeline checks ``isinstance(client, BatchCapableLLMClient)`` to decide
    whether a step can use the batch execution path.
    """

    async def submit_batch(
        self,
        requests: list[BatchRequest],
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Submit a batch of requests and return the provider batch ID.

        Args:
            requests: List of batch requests to submit.
            metadata: Optional key-value metadata attached to the batch job.

        Returns:
            The provider-assigned batch job identifier.
        """
        ...

    async def poll_batch(
        self,
        batch_id: str,
        poll_interval: float = 60.0,
        timeout: float = 86400.0,
    ) -> BatchResult:
        """Poll until batch completes or timeout.

        Args:
            batch_id: Provider batch job identifier from ``submit_batch()``.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising an error.

        Returns:
            Aggregated batch result with per-request responses.

        Raises:
            StepError: If the batch fails, expires, or exceeds timeout.
        """
        ...

    async def cancel_batch(self, batch_id: str) -> None:
        """Best-effort cancel a running batch job.

        Args:
            batch_id: Provider batch job identifier to cancel.
        """
        ...

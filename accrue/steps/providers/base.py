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
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.is_rate_limit = is_rate_limit
        self._retryable_override = retryable

    @property
    def retryable(self) -> bool:
        """True when the error is safe to retry.

        Retryable: rate-limit (429), timeout (408), and any 5xx server error.
        Non-retryable: 400 (bad request / context_length_exceeded), 401 (auth),
        403 (permission denied), 404 (not found), 422 (unprocessable), etc.
        An explicit ``retryable=True/False`` passed to the constructor overrides
        the automatic logic.
        """
        if self._retryable_override is not None:
            return self._retryable_override
        if self.is_rate_limit:
            return True
        if self.status_code is not None:
            if self.status_code in {408, 429}:
                return True
            if self.status_code >= 500:
                return True
        return False


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

    Implementing the three methods is necessary but not always sufficient — use
    :func:`is_batch_capable`, not a bare ``isinstance``, to decide whether a
    step can take the batch path.  An adapter whose batch support depends on
    how it was constructed may expose a ``supports_batch`` property to opt out
    at runtime; clients that omit it are treated as always capable.
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


def is_batch_capable(client: Any) -> bool:
    """True when *client* can run batch jobs **in its current configuration**.

    Two gates, because batch support is not purely structural:

    1. The client implements :class:`BatchCapableLLMClient` (structural).
    2. It has not opted out via a falsy ``supports_batch`` attribute (runtime).

    The second gate exists because one adapter class can be batch-capable in
    one configuration and not in another.  :class:`~accrue.providers.OpenAIClient`
    speaks to the OpenAI Batch API natively, but to an OpenAI-*compatible*
    gateway (OpenRouter, Groq, Together, vLLM, Ollama) when ``base_url`` is set
    — and those implement chat completions without the batch endpoints.
    ``isinstance`` cannot see that difference: a ``runtime_checkable`` Protocol
    only checks that the methods exist, and they exist either way.

    Args:
        client: Any LLM client adapter.

    Returns:
        ``True`` when the batch execution path is safe to take.  Clients that
        do not define ``supports_batch`` are treated as capable, so custom
        adapters implementing the three batch methods keep working unchanged.
    """
    if not isinstance(client, BatchCapableLLMClient):
        return False
    return bool(getattr(client, "supports_batch", True))

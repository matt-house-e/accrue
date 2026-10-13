"""Provider payload size limits: request-size estimates, preflight, byte-aware batching.

Providers cap the size of a single request and of a whole batch submission.
With ``attachments=`` (#153) a row can carry tens of megabytes of PDF, so
accrue checks these caps itself instead of finding out from a rejected
request — or worse, a rejected batch that takes every other row down with it.

Limits live on the client as two optional ``int`` attributes, in bytes:

- ``max_request_bytes`` — one request (realtime call, or one batch item).
- ``max_batch_bytes`` — one batch submission, summed over its requests.

``None`` (or absent, for custom clients) means "unknown, don't check".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, TypeVar

from ...core.exceptions import StepError

#: Fixed allowance per request for the envelope accrue does not serialize
#: itself: provider-specific key names, ``custom_id``, the ``params`` /
#: ``body`` wrapper, the system-block ``cache_control`` dict, JSONL newline.
REQUEST_OVERHEAD_BYTES = 1024

T = TypeVar("T")


def base64_len(n: int) -> int:
    """Length of the standard (padded) base64 encoding of *n* bytes."""
    return 4 * ((n + 2) // 3)


def estimate_request_bytes(
    messages: list[dict[str, Any]],
    **call_kwargs: Any,
) -> int:
    """Estimate the serialized size of one provider request, in bytes.

    Document blocks are counted at their base64 length (the x4/3 inflation
    every provider's JSON transport applies) without encoding them, so a
    25 MB PDF is measured, not copied.  Everything else — text, response
    schema, tools, provider kwargs — is measured by JSON-serializing it, the
    same way the SDKs put it on the wire.  The result is within
    :data:`REQUEST_OVERHEAD_BYTES` of the real body for the built-in adapters.

    Args:
        messages: Chat messages in accrue's neutral format.
        **call_kwargs: The remaining call arguments (``model``,
            ``max_tokens``, ``response_format``, ``tools``, ...).

    Returns:
        Estimated request body size in bytes.
    """
    doc_bytes = 0
    stripped: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            blocks: list[Any] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "document":
                    data = block.get("data", b"")
                    size = len(data) if isinstance(data, (bytes, bytearray)) else len(str(data))
                    doc_bytes += base64_len(size)
                    block = {k: v for k, v in block.items() if k != "data"}
                blocks.append(block)
            msg = {**msg, "content": blocks}
        stripped.append(msg)

    body = json.dumps(
        {"messages": stripped, **call_kwargs},
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return len(body) + doc_bytes + REQUEST_OVERHEAD_BYTES


def client_limit(client: Any, name: str) -> int | None:
    """Read a byte limit attribute off *client*, or ``None`` if it has none.

    Only a positive ``int`` counts — a mock or a custom client that returns
    anything else is treated as "no limit known" rather than compared.
    """
    value = getattr(client, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def format_bytes(n: int) -> str:
    """Human-readable size in decimal MB (how providers state their limits)."""
    return f"{n / 1_000_000:.2f} MB ({n:,} bytes)"


def oversize_error(step_name: str, size: int, limit: int, *, scope: str) -> StepError:
    """The row-level error for a request too large to send.

    Args:
        step_name: Step the row belongs to.
        size: Estimated request size in bytes.
        limit: The limit it exceeds, in bytes.
        scope: ``"request"`` (the provider's per-request cap) or ``"batch"``
            (the per-batch cap, which one request alone already exceeds).
    """
    if scope == "request":
        what = "the provider's per-request limit"
    else:
        what = "the batch size limit (EnrichmentConfig.batch_max_bytes or the provider default)"
    return StepError(
        f"Request for this row is {format_bytes(size)} (attachments counted "
        f"base64-encoded), over {what} of {format_bytes(limit)}; not sent. "
        f"Split or shrink the row's attachments.",
        step_name=step_name,
    )


def chunk_by_size(
    items: Sequence[T],
    sizes: Sequence[int],
    *,
    max_count: int,
    max_bytes: int | None,
) -> list[list[T]]:
    """Split *items* into consecutive chunks bounded by count and total size.

    A new chunk starts when adding the next item would exceed *max_count*
    items or *max_bytes* summed size.  Order is preserved.  Callers must
    remove items individually larger than *max_bytes* first; one passed in
    anyway gets a chunk of its own.

    Args:
        items: Items to chunk.
        sizes: Size of each item, parallel to *items*.
        max_count: Maximum items per chunk.
        max_bytes: Maximum summed size per chunk, or ``None`` for no limit.

    Returns:
        List of non-empty chunks.
    """
    chunks: list[list[T]] = []
    current: list[T] = []
    current_bytes = 0
    for item, size in zip(items, sizes):
        over_count = len(current) >= max_count
        over_bytes = max_bytes is not None and current_bytes + size > max_bytes
        if current and (over_count or over_bytes):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(item)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks

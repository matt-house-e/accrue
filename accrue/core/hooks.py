"""Lifecycle hooks for pipeline observability.

Typed event dataclasses + ``EnrichmentHooks`` container.  Hook callables
are optional; ``_fire_hook`` silently catches errors so observability
failures never crash data pipelines.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStartEvent:
    """Fired once at the beginning of ``Pipeline.run_async()``."""

    step_names: list[str]
    num_rows: int
    config: EnrichmentConfig


@dataclass(frozen=True)
class PipelineEndEvent:
    """Fired once at the end of ``Pipeline.run_async()`` (including on error)."""

    num_rows: int
    total_errors: int
    cost: Any
    elapsed_seconds: float


@dataclass(frozen=True)
class StepStartEvent:
    """Fired before a step begins processing rows."""

    step_name: str
    num_rows: int
    level: int
    execution_mode: str = "realtime"


@dataclass(frozen=True)
class StepEndEvent:
    """Fired after a step finishes all rows."""

    step_name: str
    num_rows: int
    num_errors: int
    usage: Any
    elapsed_seconds: float
    execution_mode: str = "realtime"
    batch_id: str | None = None


@dataclass(frozen=True)
class RowCompleteEvent:
    """Fired after each row completes within a step.

    ``usage`` and ``elapsed_ms`` are additive (issue #128): populated on the
    realtime path when available, ``None`` otherwise (batch mode, skipped
    rows, cache hits, steps that emit no usage — e.g. FunctionStep).
    """

    step_name: str
    row_index: int
    values: dict[str, Any]
    error: BaseException | None
    from_cache: bool
    skipped: bool = False
    usage: Any | None = None  # UsageInfo for LLM rows, else None
    elapsed_ms: float | None = None  # wall-clock ms for this row, else None


@dataclass(frozen=True)
class RowAttemptEvent:
    """Fired once per LLM attempt, inside ``LLMStep.run()``'s retry loops (#134).

    An LLM cell that retries emits one of these per attempt — before its single
    :class:`RowCompleteEvent`.  A cell that succeeds on the first try emits one
    (``kind="parse"``, ``status="ok"``); a 3-try cell emits three.

    ``kind`` names which retry loop the attempt belongs to: ``"api"`` for the
    API-error loop (the provider call raised :class:`LLMAPIError`), ``"parse"``
    for the parse/validation loop (the call returned and we tried to parse it,
    whether that parse succeeded or not).

    ``body`` carries the rendered request/response for the attempt when the run
    is at ``capture>=prompts``; it is ``None`` at the default ``metadata`` tier.
    The run logger writes it to the prompt sidecar and the projected
    ``row_attempt`` record points into that file via ``prompt_ref`` — the body
    is never inlined into the main log.
    """

    step_name: str
    row_index: int
    attempt: int  # 1-based, counted across both loops
    kind: str  # "api" | "parse"
    status: str  # "ok" | "rate_limited" | "timeout" | "validation_error" | ...
    latency_ms: float | None  # this attempt's provider-call latency
    backoff_s: float | None  # sleep before the NEXT attempt, else None
    error: BaseException | None = None
    body: dict[str, Any] | None = None  # rendered prompt/response, capture>=prompts


# ---------------------------------------------------------------------------
# EnrichmentHooks container
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentHooks:
    """User-facing hook container — pass to ``Pipeline.run()`` / ``run_async()``.

    All fields are optional callables. Sync and async callables both work.
    Hook errors are caught and logged; they never crash the pipeline.
    """

    on_pipeline_start: Callable[[PipelineStartEvent], Any] | None = None
    on_pipeline_end: Callable[[PipelineEndEvent], Any] | None = None
    on_step_start: Callable[[StepStartEvent], Any] | None = None
    on_step_end: Callable[[StepEndEvent], Any] | None = None
    on_row_complete: Callable[[RowCompleteEvent], Any] | None = None
    on_row_attempt: Callable[[RowAttemptEvent], Any] | None = None


# ---------------------------------------------------------------------------
# Fire helper
# ---------------------------------------------------------------------------


async def _fire_hook(hook: Callable | None, event: Any) -> None:
    """Call *hook* with *event*, awaiting if async.  Silently catches errors.

    Sync hooks are dispatched via ``asyncio.to_thread`` so a blocking hook
    (e.g. one that writes to Slack or Sentry) does not freeze the event loop.
    """
    if hook is None:
        return
    try:
        if inspect.iscoroutinefunction(hook):
            await hook(event)
        else:
            await asyncio.to_thread(hook, event)
    except Exception:
        logger.warning("Hook %s raised an exception", hook, exc_info=True)

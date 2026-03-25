"""Lifecycle hooks for pipeline observability.

Typed event dataclasses + ``EnrichmentHooks`` container.  Hook callables
are optional; ``_fire_hook`` silently catches errors so observability
failures never crash data pipelines.
"""

from __future__ import annotations

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
    """Fired after each row completes within a step."""

    step_name: str
    row_index: int
    values: dict[str, Any]
    error: BaseException | None
    from_cache: bool
    skipped: bool = False


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


# ---------------------------------------------------------------------------
# Fire helper
# ---------------------------------------------------------------------------


async def _fire_hook(hook: Callable | None, event: Any) -> None:
    """Call *hook* with *event*, awaiting if async.  Silently catches errors."""
    if hook is None:
        return
    try:
        result = hook(event)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("Hook %s raised an exception", hook, exc_info=True)

"""Run-log contract v1 — append-only JSONL event stream for pipeline runs.

A run log is a line-per-event JSONL file describing one pipeline run:
``pipeline_start``, ``step_start``, ``row_complete`` (per row per step),
``step_end``, and ``pipeline_end``.  The full contract — envelope, record
types, guarantees, and versioning policy — is specified in
``docs/guides/run-log.md``.  Issue #128.

:class:`JsonlRunLogger` is a plain consumer of the existing
:class:`~accrue.core.hooks.EnrichmentHooks` events; it does not touch
pipeline execution.  ``Pipeline.run(..., run_log=True)`` wires one up
automatically and merges it with any user-supplied hooks.  Standard library
``json`` only — no new dependencies.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..utils.logger import get_logger
from .hooks import (
    EnrichmentHooks,
    PipelineEndEvent,
    PipelineStartEvent,
    RowCompleteEvent,
    StepEndEvent,
    StepStartEvent,
)

if TYPE_CHECKING:
    from ..pipeline.pipeline import Pipeline

logger = get_logger(__name__)

#: Run-log schema version.  Bumped only on breaking changes; additive
#: fields do NOT bump it (consumers must ignore unknown fields).
SCHEMA_VERSION = 1

#: Default directory for ``run_log=True`` logs, relative to the CWD.
DEFAULT_RUN_DIR = Path(".accrue") / "runs"


def new_run_id() -> str:
    """Local-timestamp run id: ``YYYY-MM-DD-HHMMSS``."""
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


def resolve_run_log_path(run_log: bool | str | Path, run_id: str) -> Path:
    """Resolve the ``run_log`` argument to a concrete file path.

    ``True`` → ``.accrue/runs/<run_id>.jsonl`` relative to the CWD;
    a ``str`` / ``Path`` is used as the file path verbatim.
    """
    if run_log is True:
        return DEFAULT_RUN_DIR / f"{run_id}.jsonl"
    return Path(run_log)


def default_display_key(data: pd.DataFrame | list[dict[str, Any]]) -> str | None:
    """Heuristic label column for UI display.

    First column with string/object dtype, else the first column, else
    ``None``.  For ``list[dict]`` input the first row's keys stand in for
    columns and a ``str`` value stands in for string dtype.
    """
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        if not columns:
            return None
        for col in columns:
            dtype = data.dtypes[col]
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                return str(col)
        return str(columns[0])

    if not data:
        return None
    first_row = data[0]
    keys = list(first_row.keys())
    if not keys:
        return None
    for key in keys:
        if isinstance(first_row[key], str):
            return key
    return keys[0]


def _merge_hooks(primary: EnrichmentHooks, extra: EnrichmentHooks | None) -> EnrichmentHooks:
    """Combine two hook containers so both sets of callables fire.

    ``primary`` callables run first (the run logger, so log lines land in
    event order), then ``extra`` (user hooks).  Each callable is guarded
    individually: one raising never suppresses the other.
    """
    if extra is None:
        return primary

    def _chain(first: Any, second: Any) -> Any:
        if first is None:
            return second
        if second is None:
            return first

        async def chained(event: Any) -> None:
            for callback in (first, second):
                try:
                    if inspect.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        # Same dispatch as _fire_hook: don't block the loop.
                        await asyncio.to_thread(callback, event)
                except Exception:
                    logger.warning("Hook %s raised an exception", callback, exc_info=True)

        return chained

    return EnrichmentHooks(
        on_pipeline_start=_chain(primary.on_pipeline_start, extra.on_pipeline_start),
        on_pipeline_end=_chain(primary.on_pipeline_end, extra.on_pipeline_end),
        on_step_start=_chain(primary.on_step_start, extra.on_step_start),
        on_step_end=_chain(primary.on_step_end, extra.on_step_end),
        on_row_complete=_chain(primary.on_row_complete, extra.on_row_complete),
    )


class JsonlRunLogger:
    """Writes the run-log contract v1 as append-only JSONL.

    A plain :class:`EnrichmentHooks` consumer — pass ``logger.hooks`` to
    ``Pipeline.run()`` / ``run_async()``, or let ``run(..., run_log=True)``
    construct and merge one for you.

    Guarantees:

    - **Append-only.**  The file is opened in append mode; an existing file
      is never truncated.
    - **Flush per line.**  Every record is flushed as soon as it is written,
      so ``tail -f`` (or a crashed run) always sees complete lines.
    - **Parents created.**  The log directory is created on first write.

    The ``cost`` fields in ``usage`` records are always ``null``: accrue
    carries no model pricing data by design (see ``compare.py``); consumers
    compute dollar cost themselves.

    Args:
        path: Destination ``.jsonl`` file.
        run_id: Identifier written into ``pipeline_start``; defaults to
            :func:`new_run_id`.
        display_key: Label column recorded in ``pipeline_start`` for UIs.
        pipeline: The :class:`Pipeline` about to run.  Optional, but without
            it the ``pipeline_start`` ``steps`` entries carry ``null``
            level/mode/model (per-step ``step_start`` records still carry
            level and mode).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        display_key: str | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id or new_run_id()
        self.display_key = display_key
        self._pipeline = pipeline
        self._t0: float | None = None
        self._fh: Any = None
        self._lock = threading.Lock()
        #: Pass to ``Pipeline.run(data, hooks=logger.hooks)``.
        self.hooks = EnrichmentHooks(
            on_pipeline_start=self._on_pipeline_start,
            on_pipeline_end=self._on_pipeline_end,
            on_step_start=self._on_step_start,
            on_step_end=self._on_step_end,
            on_row_complete=self._on_row_complete,
        )

    # -- writing ---------------------------------------------------------

    def _emit(self, record_type: str, payload: dict[str, Any]) -> None:
        """Append one record, envelope first, and flush."""
        if self._t0 is None:  # defensive: events before pipeline_start
            self._t0 = time.monotonic()
        record: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "t": round(time.monotonic() - self._t0, 6),
            "type": record_type,
        }
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if self._fh is None or self._fh.closed:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Close the file handle (re-opened automatically if more events arrive)."""
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()

    # -- hook callables --------------------------------------------------
    # All async so the write happens synchronously on the event loop when
    # the hook task runs — log lines land in event order, never reordered
    # by thread scheduling.

    async def _on_pipeline_start(self, event: PipelineStartEvent) -> None:
        self._t0 = time.monotonic()
        self._emit(
            "pipeline_start",
            {
                "run_id": self.run_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "num_rows": event.num_rows,
                "display_key": self.display_key,
                "steps": self._step_metadata(event),
                # Reserved for a Pipeline.plan() snapshot — not captured yet (#128).
                "plan": None,
            },
        )

    async def _on_step_start(self, event: StepStartEvent) -> None:
        self._emit(
            "step_start",
            {
                "step": event.step_name,
                "level": event.level,
                "mode": event.execution_mode,
                "num_rows": event.num_rows,
            },
        )

    async def _on_row_complete(self, event: RowCompleteEvent) -> None:
        if event.skipped:
            status = "skipped"
        elif event.error is not None:
            status = "error"
        else:
            status = "ok"

        error = None
        if event.error is not None:
            error = {"type": type(event.error).__name__, "msg": str(event.error)}

        usage = None
        if event.usage is not None:
            usage = {
                "in": getattr(event.usage, "prompt_tokens", 0),
                "out": getattr(event.usage, "completion_tokens", 0),
                "cost": None,
            }

        # The row's output values, internal ``__`` keys included — consumers
        # filter them; the log stays complete.  Null for errored rows (the
        # event carries an all-None placeholder, not real output) and for
        # rows with no values.  Non-JSON types degrade to strings via the
        # ``default=str`` in _emit rather than crashing the logger.
        values = None
        if event.error is None and event.values:
            values = event.values

        self._emit(
            "row_complete",
            {
                "step": event.step_name,
                "row": event.row_index,
                "status": status,
                "from_cache": event.from_cache,
                "values": values,
                "error": error,
                "usage": usage,
                "elapsed_ms": None if event.elapsed_ms is None else round(event.elapsed_ms, 3),
            },
        )

    async def _on_step_end(self, event: StepEndEvent) -> None:
        usage = event.usage
        self._emit(
            "step_end",
            {
                "step": event.step_name,
                "num_errors": event.num_errors,
                "usage": {
                    "in": usage.prompt_tokens if usage is not None else 0,
                    "out": usage.completion_tokens if usage is not None else 0,
                    "cost": None,
                },
                "elapsed_s": round(event.elapsed_seconds, 6),
                "batch_id": event.batch_id,
            },
        )

    async def _on_pipeline_end(self, event: PipelineEndEvent) -> None:
        cost = event.cost
        self._emit(
            "pipeline_end",
            {
                "num_rows": event.num_rows,
                "total_errors": event.total_errors,
                "cost": {
                    "in": getattr(cost, "total_prompt_tokens", 0),
                    "out": getattr(cost, "total_completion_tokens", 0),
                    "cost": None,
                },
                "elapsed_s": round(event.elapsed_seconds, 6),
            },
        )
        self.close()

    # -- helpers ---------------------------------------------------------

    def _step_metadata(self, event: PipelineStartEvent) -> list[dict[str, Any]]:
        """Per-step {name, level, mode, model} for the pipeline_start record."""
        if self._pipeline is None:
            return [
                {"name": name, "level": None, "mode": None, "model": None}
                for name in event.step_names
            ]
        steps: list[dict[str, Any]] = []
        for level_idx, level in enumerate(self._pipeline.execution_levels):
            for name in level:
                step = self._pipeline.get_step(name)
                model = getattr(step, "model", None)
                steps.append(
                    {
                        "name": name,
                        "level": level_idx,
                        "mode": (
                            "batch" if getattr(step, "is_batch_eligible", False) else "realtime"
                        ),
                        "model": model if isinstance(model, str) and model else None,
                    }
                )
        return steps

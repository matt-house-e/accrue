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
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..utils.logger import get_logger
from .exceptions import _sanitize_secrets
from .hooks import (
    EnrichmentHooks,
    PipelineEndEvent,
    PipelineStartEvent,
    RowAttemptEvent,
    RowCompleteEvent,
    StepEndEvent,
    StepStartEvent,
)
from .manifest import build_manifest

if TYPE_CHECKING:
    from ..pipeline.pipeline import Pipeline
    from .config import EnrichmentConfig

logger = get_logger(__name__)

#: Run-log schema version.  Bumped only on breaking changes; additive
#: fields do NOT bump it (consumers must ignore unknown fields).
SCHEMA_VERSION = 1

#: Default directory for ``run_log=True`` logs, relative to the CWD.
DEFAULT_RUN_DIR = Path(".accrue") / "runs"


#: Nesting depth the NaN/inf scrub walks before giving up.  A self-referential
#: structure hits this, falls through to ``json.dumps``, and is caught there.
_MAX_SCRUB_DEPTH = 20


def new_run_id() -> str:
    """UTC-timestamp run id with a random suffix: ``YYYY-MM-DD-HHMMSS-xxxxxx``.

    The timestamp is UTC so it agrees with the ``started_at`` field beside it
    in ``pipeline_start``.  The suffix is what makes the id an id: whole-second
    local timestamps collided for runs started in the same second, which under
    ``run_log=True`` meant two runs interleaved into one file under one id.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _replace_non_finite(value: Any, depth: int = 0) -> Any:
    """Recursively replace NaN / inf with ``None``.

    ``json.dumps`` writes bare ``NaN`` / ``Infinity`` literals by default,
    which are not JSON — a single such value makes the line unparseable for
    every strict reader downstream.  A null is the honest encoding.
    """
    if depth > _MAX_SCRUB_DEPTH:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _replace_non_finite(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_non_finite(v, depth + 1) for v in value]
    return value


def resolve_run_log_path(run_log: bool | str | Path, run_id: str) -> Path:
    """Resolve the ``run_log`` argument to a concrete file path.

    ``True`` → ``.accrue/runs/<run_id>.jsonl`` relative to the CWD;
    a ``str`` / ``Path`` is used as the file path verbatim.
    """
    if run_log is True:
        return DEFAULT_RUN_DIR / f"{run_id}.jsonl"
    return Path(run_log)


def prompt_sidecar_path(run_log_path: str | Path) -> Path:
    """Path of the prompt sidecar beside a run log: ``<run>.prompts.jsonl``.

    ``.accrue/runs/<id>.jsonl`` → ``.accrue/runs/<id>.prompts.jsonl``;
    ``logs/tonight.jsonl`` → ``logs/tonight.prompts.jsonl``.  Same directory,
    the run log's stem with ``.prompts.jsonl`` swapped in for ``.jsonl`` — see
    the capture-tiers section of ``docs/guides/run-log.md``.
    """
    return Path(run_log_path).with_suffix(".prompts.jsonl")


def read_prompt_ref(sidecar_path: str | Path, off: int, len: int) -> dict[str, Any]:
    """Resolve a ``prompt_ref`` from a ``row_attempt`` record into its body.

    ``off`` / ``len`` are the byte offset and byte length a ``row_attempt``
    record's ``prompt_ref`` carries; this seeks to *off*, reads *len* bytes of
    the sidecar, and parses the one JSON body stored there.  Consumers can splat
    a record straight in: ``read_prompt_ref(sidecar, **rec["prompt_ref"])``.
    """
    with open(sidecar_path, "rb") as fh:
        fh.seek(off)
        raw = fh.read(len)
    return json.loads(raw.decode("utf-8"))


def _sanitize_body(value: Any) -> Any:
    """Recursively redact secret patterns from a captured prompt/response body.

    Captured prompts and responses are user/model text that can quote an API
    key the same way an error message can; they go through the identical
    :func:`_sanitize_secrets` redaction before touching disk (#134).
    """
    if isinstance(value, str):
        return _sanitize_secrets(value)
    if isinstance(value, dict):
        return {k: _sanitize_body(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_body(v) for v in value]
    return value


def read_run_context(path: str | Path) -> tuple[str | None, str | None, float]:
    """Recover ``(run_id, display_key, last_t)`` from an existing run log.

    Used by ``Pipeline.retry_failed()`` to append a retry segment to the log
    of the run it is healing: the retry keeps the original ``run_id`` and
    ``display_key`` (so recovered rows carry the same ``key``), and starts
    its ``t`` clock at the file's last ``t`` so the envelope stays
    non-decreasing across segments.

    Returns ``(None, None, 0.0)`` when the file is missing, empty, or has no
    readable ``pipeline_start``.  Malformed lines are skipped, not raised —
    a partially written log must never block a retry.
    """
    file_path = Path(path)
    if not file_path.exists():
        return None, None, 0.0

    run_id: str | None = None
    display_key: str | None = None
    last_t = 0.0
    try:
        with file_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                t = record.get("t")
                if isinstance(t, (int, float)):
                    last_t = max(last_t, float(t))
                if record.get("type") == "pipeline_start" and run_id is None:
                    raw_id = record.get("run_id")
                    run_id = raw_id if isinstance(raw_id, str) else None
                    raw_key = record.get("display_key")
                    display_key = raw_key if isinstance(raw_key, str) else None
    except OSError as exc:
        logger.warning("Could not read run log %s for retry continuity: %s", file_path, exc)
        return None, None, 0.0

    return run_id, display_key, last_t


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


def resolve_row_keys(
    data: pd.DataFrame | list[dict[str, Any]] | None,
    display_key: str | None,
) -> list[str | None]:
    """Per-row display values: ``str()`` of each row's *display_key* column.

    Returns one entry per input row — ``None`` where the value is missing
    or null — or an empty list when there is no data / no display key to
    resolve (``row_complete.key`` is then ``null`` for every row).
    """
    if data is None or display_key is None:
        return []

    def to_key(value: Any) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass  # non-scalar (list/array) value — keep its str() form
        return str(value)

    if isinstance(data, pd.DataFrame):
        if display_key not in data.columns:
            return []
        return [to_key(value) for value in data[display_key]]
    return [to_key(row.get(display_key)) for row in data]


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
        on_row_attempt=_chain(primary.on_row_attempt, extra.on_row_attempt),
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
        data: The input rows about to run.  Optional, but without it every
            ``row_complete`` carries ``key: null`` — the log holds only step
            *outputs*, so the display-key value can't be recovered from it
            otherwise.
        segment: ``"run"`` (default) frames the events as a whole run,
            ``pipeline_start`` … ``pipeline_end``.  ``"retry"`` frames them as
            a failed-only retry appended to an existing run (#129): the
            wrapper records become ``retry_start`` / ``retry_end`` — additive
            v1 record types — so a consumer reading the file never mistakes
            the appended segment for a second run.
        t_offset: Seconds added to every record's ``t``.  Set to the existing
            file's last ``t`` (see :func:`read_run_context`) when appending a
            retry segment, keeping ``t`` non-decreasing across the file.
        retry_cells: ``{step: [row_index, ...]}`` this retry will re-execute,
            recorded in ``retry_start`` so a UI can mark the cells in flight.
        capture: Capture tier gating the prompt sidecar — ``"metadata"``
            (default) writes no bodies, ``"prompts"`` / ``"full"`` append each
            captured LLM attempt body to ``<run>.prompts.jsonl`` and reference
            it from ``row_attempt`` via ``prompt_ref``.  The logger only ever
            writes a body when an attempt event carries one, so the source (the
            LLM step) is what actually honours the tier; this is a
            belt-and-braces gate on top.
        config: The :class:`~accrue.core.config.EnrichmentConfig` for this run.
            Optional; when given it feeds the ``pipeline_start.manifest.config``
            block (max_workers, caching, checkpointing) and the LLM steps'
            effective temperature / max_tokens fallbacks (#138).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        display_key: str | None = None,
        pipeline: Pipeline | None = None,
        data: pd.DataFrame | list[dict[str, Any]] | None = None,
        segment: str = "run",
        t_offset: float = 0.0,
        retry_cells: dict[str, list[int]] | None = None,
        capture: str = "metadata",
        config: EnrichmentConfig | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id or new_run_id()
        self.display_key = display_key
        self.segment = segment
        self.capture = capture
        self._row_keys = resolve_row_keys(data, display_key)
        self._pipeline = pipeline
        self._config = config
        self._t_offset = t_offset
        self._retry_cells = retry_cells or {}
        self._t0: float | None = None
        self._fh: Any = None
        self._lock = threading.Lock()
        # Prompt sidecar (#134): opened lazily on the first captured body, only
        # when ``capture >= prompts``.  ``_sidecar_bytes`` tracks the running
        # byte offset so each body's ``prompt_ref`` points at the right span.
        self._sidecar_path = prompt_sidecar_path(self.path)
        self._sidecar_fh: Any = None
        self._sidecar_bytes = 0
        #: Pass to ``Pipeline.run(data, hooks=logger.hooks)``.
        self.hooks = EnrichmentHooks(
            on_pipeline_start=self._on_pipeline_start,
            on_pipeline_end=self._on_pipeline_end,
            on_step_start=self._on_step_start,
            on_step_end=self._on_step_end,
            on_row_complete=self._on_row_complete,
            on_row_attempt=self._on_row_attempt,
        )

    # -- writing ---------------------------------------------------------

    def _emit(self, record_type: str, payload: dict[str, Any]) -> None:
        """Append one record, envelope first, and flush.

        A record that cannot be encoded is downgraded, never dropped: every
        row of every executed step gets exactly one ``row_complete``, and that
        guarantee has to survive user values JSON cannot represent (non-string
        dict keys, reference cycles).  Hook exceptions are swallowed upstream,
        so raising here would silently lose the line.
        """
        if self._t0 is None:  # defensive: events before pipeline_start
            self._t0 = time.monotonic()
        record: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "t": round(self._t_offset + time.monotonic() - self._t0, 6),
            "type": record_type,
        }
        record.update(payload)
        line = self._encode(record)
        with self._lock:
            if self._fh is None or self._fh.closed:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Run logs carry row values and error text; keep them
                # owner-only rather than inheriting the 0644 umask default.
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                self._fh = os.fdopen(fd, "a", encoding="utf-8")
                try:
                    os.chmod(self.path, 0o600)
                except (OSError, NotImplementedError):  # pragma: no cover — platform-dependent
                    pass
            self._fh.write(line + "\n")
            self._fh.flush()

    def _write_sidecar(self, body: dict[str, Any]) -> dict[str, int]:
        """Append one captured body to the prompt sidecar; return its ``prompt_ref``.

        The body is redacted (:func:`_sanitize_body`), encoded as one JSON line,
        and appended.  Returns ``{"off": <byte offset>, "len": <byte length>}``
        pointing at the JSON (newline excluded) so :func:`read_prompt_ref`
        resolves it.  The sidecar is append-only, flushed per line, and created
        ``0600`` — the same hardening as the main log (#134).
        """
        sanitized = _sanitize_body(body)
        encoded = json.dumps(sanitized, ensure_ascii=False, default=str, allow_nan=False)
        data = encoded.encode("utf-8")
        with self._lock:
            if self._sidecar_fh is None or self._sidecar_fh.closed:
                self._sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                # Bodies are prompts/responses — often PII; owner-only, like the
                # main log.  Seed the byte offset from any pre-existing file so a
                # retry appended to the same run keeps refs pointing correctly.
                if self._sidecar_path.exists():
                    self._sidecar_bytes = self._sidecar_path.stat().st_size
                fd = os.open(self._sidecar_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                self._sidecar_fh = os.fdopen(fd, "a", encoding="utf-8")
                try:
                    os.chmod(self._sidecar_path, 0o600)
                except (OSError, NotImplementedError):  # pragma: no cover — platform-dependent
                    pass
            off = self._sidecar_bytes
            self._sidecar_fh.write(encoded + "\n")
            self._sidecar_fh.flush()
            self._sidecar_bytes += len(data) + 1  # +1 for the newline
        return {"off": off, "len": len(data)}

    @staticmethod
    def _encode(record: dict[str, Any]) -> str:
        """Serialize one record to a single JSON line that always parses."""
        try:
            return json.dumps(record, ensure_ascii=False, default=str, allow_nan=False)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Run-log %s record could not be serialized (%s: %s); emitting it with "
                "its values replaced by a placeholder.",
                record.get("type"),
                type(exc).__name__,
                exc,
            )
        degraded = dict(record)
        degraded["values"] = {"__unserializable__": True}
        try:
            return json.dumps(degraded, ensure_ascii=False, default=str, allow_nan=False)
        except (TypeError, ValueError):  # pragma: no cover — defensive
            # Last resort: keep the envelope and whatever scalars identify the
            # record, so a consumer still sees the row complete.
            minimal = {
                str(k): (v if isinstance(v, (str, bool, int)) or v is None else None)
                for k, v in degraded.items()
            }
            minimal["values"] = {"__unserializable__": True}
            return json.dumps(minimal, ensure_ascii=False, default=str, allow_nan=False)

    def close(self) -> None:
        """Close the file handles (re-opened automatically if more events arrive)."""
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()
            if self._sidecar_fh is not None and not self._sidecar_fh.closed:
                self._sidecar_fh.close()

    # -- hook callables --------------------------------------------------
    # All async so the write happens synchronously on the event loop when
    # the hook task runs — log lines land in event order, never reordered
    # by thread scheduling.

    async def _on_pipeline_start(self, event: PipelineStartEvent) -> None:
        self._t0 = time.monotonic()
        if self.segment == "retry":
            cells = [
                {"step": step, "row": row}
                for step, indices in self._retry_cells.items()
                for row in indices
            ]
            self._emit(
                "retry_start",
                {
                    "run_id": self.run_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "num_rows": event.num_rows,
                    "num_cells": len(cells),
                    "cells": cells,
                },
            )
            return
        self._emit(
            "pipeline_start",
            {
                "run_id": self.run_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "num_rows": event.num_rows,
                "display_key": self.display_key,
                "steps": self._step_metadata(event),
                # The run's definition for a dashboard Overview — steps + types +
                # models, the enrichment-field schema, and the run config (#138).
                # Built once here, row-independent, additive to schema v1.
                "manifest": self._build_manifest(),
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

        # Error text goes through the same redaction the checkpoint path uses:
        # a provider error can quote the request headers, and the log outlives
        # the process.
        error = None
        if event.error is not None:
            error = {
                "type": type(event.error).__name__,
                "msg": _sanitize_secrets(str(event.error)),
            }

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
        # ``default=str`` in _emit rather than crashing the logger; NaN/inf
        # become null, since a bare ``NaN`` literal is not JSON.
        values = None
        if event.error is None and event.values:
            values = _replace_non_finite(event.values)

        # The row's display value, resolved once from the input data at
        # construction time (additive within v1 — the log otherwise carries
        # only step outputs, so consumers could never label rows).
        key = None
        if 0 <= event.row_index < len(self._row_keys):
            key = self._row_keys[event.row_index]

        self._emit(
            "row_complete",
            {
                "step": event.step_name,
                "row": event.row_index,
                "key": key,
                "status": status,
                "from_cache": event.from_cache,
                "values": values,
                "error": error,
                "usage": usage,
                "elapsed_ms": None if event.elapsed_ms is None else round(event.elapsed_ms, 3),
            },
        )

    async def _on_row_attempt(self, event: RowAttemptEvent) -> None:
        # The main log never inlines bodies: at capture>=prompts a body rides on
        # the event, gets appended to the sidecar, and the record points into it
        # via prompt_ref; at metadata the event carries no body and prompt_ref is
        # null.  The sidecar append + the record emit happen back-to-back with no
        # await between them, so the byte offset a prompt_ref records is exact.
        prompt_ref = None
        if event.body is not None and self.capture != "metadata":
            prompt_ref = self._write_sidecar(event.body)

        # Error text goes through the same redaction as row_complete / the
        # checkpoint path — a provider error can quote request headers.
        error = None
        if event.error is not None:
            error = {
                "type": type(event.error).__name__,
                "msg": _sanitize_secrets(str(event.error)),
            }

        self._emit(
            "row_attempt",
            {
                "step": event.step_name,
                "row": event.row_index,
                "attempt": event.attempt,
                "kind": event.kind,
                "status": event.status,
                "latency_ms": (None if event.latency_ms is None else round(event.latency_ms, 3)),
                "backoff_s": event.backoff_s,
                "error": error,
                "prompt_ref": prompt_ref,
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
        payload: dict[str, Any] = {
            "num_rows": event.num_rows,
            "total_errors": event.total_errors,
            "cost": {
                "in": getattr(cost, "total_prompt_tokens", 0),
                "out": getattr(cost, "total_completion_tokens", 0),
                "cost": None,
            },
            "elapsed_s": round(event.elapsed_seconds, 6),
        }
        if self.segment == "retry":
            payload["num_cells"] = sum(len(v) for v in self._retry_cells.values())
        self._emit("retry_end" if self.segment == "retry" else "pipeline_end", payload)
        self.close()

    # -- helpers ---------------------------------------------------------

    def _build_manifest(self) -> dict[str, Any] | None:
        """The ``pipeline_start.manifest`` object, or ``None`` if it can't be built.

        Introspection is wrapped so a manifest failure can never take the run
        down: the run log's completeness guarantee outranks this additive
        field.  On failure the record still carries ``manifest: null``.
        """
        try:
            return build_manifest(self._pipeline, self._config, self.capture)
        except Exception:  # pragma: no cover — defensive
            logger.warning("Could not build run-log manifest", exc_info=True)
            return None

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

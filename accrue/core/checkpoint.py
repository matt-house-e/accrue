"""Per-step checkpoint manager for column-oriented pipeline execution.

Saves pipeline progress after each step completes across all rows.
Single JSON file per data_identifier + category.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.logger import get_logger
from .exceptions import _sanitize_secrets

if TYPE_CHECKING:
    from .config import EnrichmentConfig
    from .exceptions import RowError

logger = get_logger(__name__)

#: Checkpoint category used by every built-in runner.  Kept as one constant so
#: a run and its later ``Pipeline.retry_failed()`` resolve the same file.
DEFAULT_CATEGORY = "_default"

try:
    import numpy as _numpy
except ImportError:
    _numpy = None  # type: ignore[assignment]


# -- typed encoder / decoder -------------------------------------------------


def _typed_encoder(obj: Any) -> Any:
    """JSON encoder that preserves Python types across checkpoint round-trips.

    Encoded values are written as ``{"__accrue_type__": "<type>", "value": ...}``.
    The namespaced key avoids collision with user data that uses ``__type__``
    as its own discriminator (common in typed-union JSON patterns).
    """
    if isinstance(obj, datetime):
        return {"__accrue_type__": "datetime", "value": obj.isoformat()}
    if isinstance(obj, Decimal):
        return {"__accrue_type__": "Decimal", "value": str(obj)}
    if isinstance(obj, set):
        return {"__accrue_type__": "set", "value": sorted(obj, key=str)}
    if _numpy is not None and hasattr(obj, "tolist"):
        return {"__accrue_type__": "numpy", "value": obj.tolist()}
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable for checkpointing"
    )


def _typed_decoder(d: dict) -> Any:
    """JSON object_hook that restores typed values written by _typed_encoder.

    Only dicts with exactly the keys ``{"__accrue_type__", "value"}`` are
    treated as encoded library values — any extra keys means it's user data
    and must be returned unchanged.  Dicts without ``__accrue_type__`` are
    also returned as-is (covers plain user data and the outer checkpoint
    structure).

    Legacy checkpoint files written with ``__type__`` (pre-1.2.1) would
    silently produce plain dicts with incorrect types on resume.  We detect
    this format and discard the checkpoint so the user gets a clean start
    with a clear warning rather than a silent misparse.
    """
    if d.keys() == {"__type__", "value"} and d["__type__"] in (
        "datetime",
        "Decimal",
        "set",
        "numpy",
    ):
        logger.warning(
            "Legacy checkpoint format detected (pre-1.2.1 __type__ sentinel); "
            "discarding checkpoint and starting fresh."
        )
        raise json.JSONDecodeError("Legacy checkpoint format — discarded", doc="", pos=0)
    if d.keys() != {"__accrue_type__", "value"}:
        return d
    t = d["__accrue_type__"]
    if t == "datetime":
        return datetime.fromisoformat(d["value"])
    if t == "Decimal":
        return Decimal(d["value"])
    if t == "set":
        return set(d["value"])
    if t == "numpy":
        # Return as plain list — resume runs may not have numpy installed.
        return d["value"]
    return d


# -- data model --------------------------------------------------------------


def derive_data_identifier(columns: list[Any], num_rows: int) -> str:
    """Deterministic checkpoint identifier for a dataset shape.

    Derived from the column names and row count (sha256, not ``hash()``,
    so it is stable across processes).  This is the identifier the
    :class:`~accrue.core.enricher.Enricher` and ``Pipeline.retry_failed``
    use when the caller does not pass an explicit ``data_identifier``.
    """
    source = json.dumps(
        {"columns": list(columns), "rows": num_rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"df_{hashlib.sha256(source).hexdigest()[:16]}"


@dataclass
class CheckpointData:
    """Snapshot of pipeline progress loaded from a checkpoint file."""

    timestamp: float
    category: str
    total_rows: int
    fields_dict: dict[str, dict[str, Any]]
    completed_steps: list[str]
    step_results: dict[str, list[dict[str, Any]]]
    row_errors: dict[str, dict[int, dict[str, str]]] = field(default_factory=dict)
    """Failed cells recorded per completed step: ``{step: {row_index: {type, msg}}}``.

    Empty for checkpoints written before row-error tracking (#129) — those
    resume with the old all-or-nothing semantics.
    """


class CheckpointManager:
    """Manages per-step checkpoint files for pipeline execution.

    After each step finishes (across all rows), the full pipeline
    state is written to a single JSON file.  On resume, completed
    steps are skipped and their results are fed into downstream
    dependency routing.
    """

    def __init__(self, config: EnrichmentConfig) -> None:
        self._enabled = config.enable_checkpointing
        self._auto_resume = config.auto_resume
        self._checkpoint_dir: Path | None = None

        raw_dir = config.checkpoint_dir
        if raw_dir is not None:
            self._checkpoint_dir = Path(raw_dir)

    # -- path helpers ----------------------------------------------------

    def _get_path(self, data_identifier: str, category: str) -> Path:
        base_dir = self._checkpoint_dir or Path.cwd()
        base_dir.mkdir(parents=True, exist_ok=True)

        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in data_identifier)
        safe_cat = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in category)
        return base_dir / f"{safe_id}_{safe_cat}_checkpoint.json"

    # -- public API ------------------------------------------------------

    def save_step(
        self,
        data_identifier: str,
        category: str,
        step_name: str,
        step_row_results: list[dict[str, Any]],
        total_rows: int,
        fields_dict: dict[str, dict[str, Any]],
        existing_completed: list[str],
        existing_results: dict[str, list[dict[str, Any]]],
        *,
        partial: bool = False,
        step_row_errors: dict[int, dict[str, str]] | None = None,
        existing_row_errors: dict[str, dict[int, dict[str, str]]] | None = None,
    ) -> bool:
        """Write full pipeline state to disk after a step completes.

        Uses an atomic write (tmp + fsync + os.replace) so a crash mid-write
        never corrupts the existing checkpoint.

        Args:
            partial: When True, save partial row results without marking the
                step as completed.  On resume the step will re-run from
                scratch, ensuring unfinished rows are never silently treated
                as completed.  Defaults to False (normal completion).
            step_row_errors: Failed cells for *this* step, keyed by row index
                (``{row_index: {"type": ..., "msg": ...}}``).  Recorded so a
                later failed-only resume (#129) can re-run exactly these
                cells.  Ignored for partial saves — a partial step re-runs
                from scratch anyway.
            existing_row_errors: Failed cells of previously completed steps,
                carried forward unchanged (``{step: {row_index: {...}}}``).

        Returns True on success or when checkpointing is disabled (no-op).
        """
        if not self._enabled:
            return True

        # For a partial save, do NOT append step_name to completed_steps so
        # that a resumed run re-executes the step from scratch instead of
        # skipping it and propagating empty {} results downstream.
        completed = list(existing_completed) if partial else list(existing_completed) + [step_name]
        results = dict(existing_results)
        results[step_name] = step_row_results

        row_errors = {
            step: dict(cells)
            for step, cells in (existing_row_errors or {}).items()
            if step != step_name and cells
        }
        if not partial and step_row_errors:
            row_errors[step_name] = step_row_errors

        payload = {
            "timestamp": time.time(),
            "category": category,
            "total_rows": total_rows,
            "fields_dict": fields_dict,
            "completed_steps": completed,
            "step_results": results,
            # JSON object keys are strings; row indices are decoded back to
            # int in load().
            "row_errors": {
                step: {str(idx): err for idx, err in cells.items()}
                for step, cells in row_errors.items()
            },
        }

        path = self._get_path(data_identifier, category)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, default=_typed_encoder, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
                if os.name == "posix":
                    try:
                        os.chmod(path, 0o600)
                    except (OSError, NotImplementedError):
                        pass
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
            logger.info(f"Checkpoint saved after step '{step_name}': {path}")
            return True
        except TypeError:
            # Re-raise immediately — caller passed an unserializable type and needs to know.
            raise
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load(
        self,
        data_identifier: str,
        category: str,
        *,
        expected_total_rows: int | None = None,
        expected_fields: dict | None = None,
        expected_steps: list[str] | None = None,
    ) -> CheckpointData | None:
        """Load checkpoint if enabled, auto_resume is on, file exists, and all checks pass.

        Optional keyword arguments validate the saved checkpoint against the
        current pipeline shape. Any mismatch is logged as a warning and the
        checkpoint is discarded (returns None) rather than silently resuming
        against a mismatched dataset.  Pass None to skip a particular check
        (backwards-compatible with callers that pre-date strict validation).
        """
        if not self._enabled or not self._auto_resume:
            return None
        return self._load_validated(
            data_identifier,
            category,
            expected_total_rows=expected_total_rows,
            expected_fields=expected_fields,
            expected_steps=expected_steps,
        )

    def _load_validated(
        self,
        data_identifier: str,
        category: str,
        *,
        expected_total_rows: int | None = None,
        expected_fields: dict | None = None,
        expected_steps: list[str] | None = None,
    ) -> CheckpointData | None:
        """Load + strictly validate a checkpoint file, ignoring the resume gates.

        Shared by :meth:`load` (gated on ``auto_resume``) and
        :meth:`resume_failed` (explicit call — gated on ``enable_checkpointing``
        only).  Validation semantics are identical for both.
        """
        path = self._get_path(data_identifier, category)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f, object_hook=_typed_decoder)

            if raw.get("category") != category:
                logger.warning(
                    f"Checkpoint category mismatch: expected '{category}', "
                    f"got '{raw.get('category')}'"
                )
                return None

            if expected_total_rows is not None and raw.get("total_rows") != expected_total_rows:
                logger.warning(
                    f"Checkpoint row count mismatch: expected {expected_total_rows}, "
                    f"got {raw.get('total_rows')} — discarding checkpoint"
                )
                return None

            if expected_fields is not None:
                saved_keys = set(raw.get("fields_dict", {}).keys())
                expected_keys = set(expected_fields.keys())
                if saved_keys != expected_keys:
                    logger.warning(
                        f"Checkpoint fields mismatch: expected {sorted(expected_keys)}, "
                        f"got {sorted(saved_keys)} — discarding checkpoint"
                    )
                    return None

            if expected_steps is not None:
                saved_steps = raw.get("completed_steps", [])
                # Validate that every saved step is still declared in the pipeline.
                # Extra saved steps (from a shorter pipeline) are safe; missing
                # pipeline steps that appear in the checkpoint are flagged.
                unknown = [s for s in saved_steps if s not in expected_steps]
                if unknown:
                    logger.warning(
                        f"Checkpoint contains steps unknown to the current pipeline: "
                        f"{unknown} — discarding checkpoint"
                    )
                    return None

            data = CheckpointData(
                timestamp=raw["timestamp"],
                category=raw["category"],
                total_rows=raw["total_rows"],
                fields_dict=raw["fields_dict"],
                completed_steps=raw["completed_steps"],
                step_results=raw["step_results"],
                # Absent in pre-#129 checkpoints — default to no recorded
                # failures.  JSON keys are strings; decode row indices to int.
                row_errors={
                    step: {int(idx): err for idx, err in cells.items()}
                    for step, cells in raw.get("row_errors", {}).items()
                },
            )
            logger.info(
                f"Checkpoint loaded: {len(data.completed_steps)} steps completed "
                f"({', '.join(data.completed_steps)})"
            )
            return data
        except json.JSONDecodeError:
            # JSONDecodeError is raised by _typed_decoder for legacy checkpoints
            # (pre-1.2.1 __type__ sentinel) after logging a WARNING there.
            # Return None to give the pipeline a clean start.
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def resume_failed(
        self,
        data_identifier: str,
        category: str,
        *,
        rows: list[int] | None = None,
        steps: list[str] | None = None,
        expected_total_rows: int | None = None,
        expected_fields: dict | None = None,
        expected_steps: list[str] | None = None,
    ) -> tuple[CheckpointData, dict[str, list[int]]] | None:
        """Failed-only resume entrypoint (#129).

        Loads the checkpoint with the same strict validation as :meth:`load`
        and selects the failed ``(step, row)`` cells recorded in it — the
        cells a retry run must re-execute while everything that succeeded is
        served from the checkpoint.

        Unlike :meth:`load`, this is an *explicit* request, so it ignores
        ``auto_resume`` and is gated only on ``enable_checkpointing``.

        Args:
            rows: Restrict selection to these row indices (None = all).
            steps: Restrict selection to these step names (None = all).
            expected_total_rows / expected_fields / expected_steps: Same
                strict-validation kwargs as :meth:`load`.

        Returns:
            ``(checkpoint, failed_cells)`` where ``failed_cells`` maps step
            name to a sorted list of failed row indices (only for steps
            recorded as completed — an uncompleted step re-runs in full
            anyway), or ``None`` when there is no usable checkpoint.
        """
        if not self._enabled:
            return None

        cp = self._load_validated(
            data_identifier,
            category,
            expected_total_rows=expected_total_rows,
            expected_fields=expected_fields,
            expected_steps=expected_steps,
        )
        if cp is None:
            return None

        step_filter = set(steps) if steps is not None else None
        row_filter = set(rows) if rows is not None else None

        failed_cells: dict[str, list[int]] = {}
        for step_name, cells in cp.row_errors.items():
            if step_name not in cp.completed_steps:
                continue
            if step_filter is not None and step_name not in step_filter:
                continue
            indices = sorted(idx for idx in cells if row_filter is None or idx in row_filter)
            if indices:
                failed_cells[step_name] = indices

        total = sum(len(v) for v in failed_cells.values())
        logger.info(
            f"Failed-only resume: {total} failed cell(s) selected across "
            f"{len(failed_cells)} step(s)"
        )
        return cp, failed_cells

    def cleanup(self, data_identifier: str, category: str) -> bool:
        """Remove checkpoint file after successful pipeline completion."""
        if not self._enabled:
            return True

        try:
            path = self._get_path(data_identifier, category)
            if path.exists():
                path.unlink()
                logger.info(f"Checkpoint cleaned up: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoint: {e}")
            return False

    def list_checkpoints(self) -> dict[str, dict[str, Any]]:
        """Scan checkpoint_dir for ``*_checkpoint.json`` files."""
        checkpoints: dict[str, dict[str, Any]] = {}

        if not self._enabled:
            return checkpoints

        scan_dir = self._checkpoint_dir or Path.cwd()
        if not scan_dir.exists():
            return checkpoints

        for path in scan_dir.glob("*_checkpoint.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f, object_hook=_typed_decoder)
                checkpoints[path.stem] = {
                    "path": str(path),
                    "category": raw.get("category"),
                    "total_rows": raw.get("total_rows"),
                    "completed_steps": raw.get("completed_steps", []),
                    "timestamp": raw.get("timestamp"),
                }
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {path}: {e}")

        return checkpoints


# -- per-run session ---------------------------------------------------------


class CheckpointSession:
    """Mutable checkpoint-writing state for one pipeline execution.

    Wraps a :class:`CheckpointManager` with the bookkeeping every
    checkpointed run needs — the completed-step list, per-step results, and
    per-step row errors, seeded from a loaded checkpoint (or empty for a
    fresh run) — and exposes the two callbacks ``Pipeline.execute`` fires.
    Shared by :class:`~accrue.core.enricher.Enricher` and
    ``Pipeline.retry_failed`` so both write identical checkpoint state.

    Error messages are passed through the secret sanitizer before being
    persisted, since checkpoint files outlive the process.

    Args:
        retry_cells: The failed cells this run will actually re-execute
            (``{step: [row_index, ...]}``).  ``None`` means *every* failed
            cell recorded in the checkpoint — the auto-resume default.  A
            narrower selection (``Pipeline.retry_failed(rows=..., steps=...)``)
            matters on write-back: unselected failures are preserved in the
            checkpoint instead of being erased by the retried step's fresh
            error list.
    """

    def __init__(
        self,
        manager: CheckpointManager,
        *,
        data_identifier: str,
        category: str,
        total_rows: int,
        fields_dict: dict[str, dict[str, Any]],
        checkpoint: CheckpointData | None = None,
        retry_cells: dict[str, list[int]] | None = None,
    ) -> None:
        self._manager = manager
        self._data_identifier = data_identifier
        self._category = category
        self._total_rows = total_rows
        self._fields_dict = fields_dict

        if checkpoint is not None:
            self.completed: list[str] = list(checkpoint.completed_steps)
            # Only completed steps' results are trustworthy; partial saves
            # keep their step_results entry but the step re-runs from scratch.
            self.results: dict[str, list[dict[str, Any]]] = {
                k: v for k, v in checkpoint.step_results.items() if k in self.completed
            }
            self.row_errors: dict[str, dict[int, dict[str, str]]] = {
                k: dict(v) for k, v in checkpoint.row_errors.items() if k in self.completed
            }
        else:
            self.completed = []
            self.results = {}
            self.row_errors = {}

        if retry_cells is None:
            selection = {step: set(cells) for step, cells in self.row_errors.items() if cells}
        else:
            selection = {step: set(cells) for step, cells in retry_cells.items() if cells}
        self._selection: dict[str, set[int]] = selection

    @property
    def prior_step_results(self) -> dict[str, list[dict[str, Any]]]:
        """Completed steps' results, for ``Pipeline.execute(prior_step_results=...)``."""
        return dict(self.results)

    @property
    def retry_cells(self) -> dict[str, list[int]]:
        """Selected failed cells, for ``Pipeline.execute(retry_cells=...)``."""
        return {step: sorted(cells) for step, cells in self._selection.items() if cells}

    @property
    def has_failed_cells(self) -> bool:
        """True while any step still has a recorded failed cell."""
        return any(cells for cells in self.row_errors.values())

    def on_step_complete(
        self,
        step_name: str,
        step_row_results: list[dict[str, Any]],
        step_errors: list[RowError],
    ) -> None:
        """Record a completed step (fresh or retried) and persist the checkpoint."""
        errors_by_row = {
            e.row_index: {"type": e.error_type, "msg": _sanitize_secrets(str(e.error))}
            for e in step_errors
        }
        # A retried step only re-executed its selected cells, so its fresh
        # error list says nothing about the ones left out — carry those over
        # rather than dropping them from the checkpoint.
        retried = self._selection.get(step_name)
        if retried is not None:
            carried = {
                idx: err
                for idx, err in self.row_errors.get(step_name, {}).items()
                if idx not in retried
            }
            errors_by_row = {**carried, **errors_by_row}

        prior_completed = [s for s in self.completed if s != step_name]

        self.results[step_name] = step_row_results
        if errors_by_row:
            self.row_errors[step_name] = errors_by_row
        else:
            self.row_errors.pop(step_name, None)
        if step_name not in self.completed:
            self.completed.append(step_name)

        self._manager.save_step(
            data_identifier=self._data_identifier,
            category=self._category,
            step_name=step_name,
            step_row_results=step_row_results,
            total_rows=self._total_rows,
            fields_dict=self._fields_dict,
            existing_completed=prior_completed,
            existing_results={k: v for k, v in self.results.items() if k != step_name},
            step_row_errors=errors_by_row or None,
            existing_row_errors={k: v for k, v in self.row_errors.items() if k != step_name},
        )

    def finish(self) -> None:
        """Settle the checkpoint at the end of a run.

        Removed when nothing is left to heal (the historical
        clean-up-on-success behaviour); **kept** while any cell is still
        failing, because that record is exactly what
        ``Pipeline.retry_failed()`` reads to re-run those cells.
        """
        if self.has_failed_cells:
            total = sum(len(cells) for cells in self.row_errors.values())
            logger.info(
                "Checkpoint kept: %d failed cell(s) recorded — call "
                "Pipeline.retry_failed() to re-run just those.",
                total,
            )
            return
        self._manager.cleanup(self._data_identifier, self._category)

    def on_partial_checkpoint(
        self,
        step_name: str,
        partial_results: list[dict[str, Any]],
        completed_count: int,
    ) -> None:
        """Persist partial progress for an in-flight step (not marked completed).

        The in-flight step is excluded from the completed list even when it
        was previously completed (a retried step): if the retry crashes
        mid-flight, the step must re-run from scratch rather than resume
        from half-updated results.
        """
        self._manager.save_step(
            data_identifier=self._data_identifier,
            category=self._category,
            step_name=step_name,
            step_row_results=partial_results,
            total_rows=self._total_rows,
            fields_dict=self._fields_dict,
            existing_completed=[s for s in self.completed if s != step_name],
            existing_results={k: v for k, v in self.results.items() if k != step_name},
            existing_row_errors={k: v for k, v in self.row_errors.items() if k != step_name},
            partial=True,
        )

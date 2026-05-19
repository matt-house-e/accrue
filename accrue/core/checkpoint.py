"""Per-step checkpoint manager for column-oriented pipeline execution.

Saves pipeline progress after each step completes across all rows.
Single JSON file per data_identifier + category.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from .config import EnrichmentConfig

logger = get_logger(__name__)

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
    structure).  Old checkpoint files written with ``__type__`` (pre-#63)
    pass through as plain dicts; they won't deserialise their typed values
    but they won't crash either.
    """
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


@dataclass
class CheckpointData:
    """Snapshot of pipeline progress loaded from a checkpoint file."""

    timestamp: float
    category: str
    total_rows: int
    fields_dict: dict[str, dict[str, Any]]
    completed_steps: list[str]
    step_results: dict[str, list[dict[str, Any]]]


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
    ) -> bool:
        """Write full pipeline state to disk after a step completes.

        Uses an atomic write (tmp + fsync + os.replace) so a crash mid-write
        never corrupts the existing checkpoint.

        Args:
            partial: When True, save partial row results without marking the
                step as completed.  On resume the step will re-run from
                scratch, ensuring unfinished rows are never silently treated
                as completed.  Defaults to False (normal completion).

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

        payload = {
            "timestamp": time.time(),
            "category": category,
            "total_rows": total_rows,
            "fields_dict": fields_dict,
            "completed_steps": completed,
            "step_results": results,
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
            )
            logger.info(
                f"Checkpoint loaded: {len(data.completed_steps)} steps completed "
                f"({', '.join(data.completed_steps)})"
            )
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

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

"""Per-step checkpoint manager for column-oriented pipeline execution.

Saves pipeline progress after each step completes across all rows.
Single JSON file per data_identifier + category.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from .config import EnrichmentConfig

logger = get_logger(__name__)


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
    ) -> bool:
        """Write full pipeline state to disk after a step completes.

        Returns True on success or when checkpointing is disabled (no-op).
        """
        if not self._enabled:
            return True

        # Merge the newly completed step into existing state
        completed = list(existing_completed) + [step_name]
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

        try:
            path = self._get_path(data_identifier, category)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"Checkpoint saved after step '{step_name}': {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load(self, data_identifier: str, category: str) -> CheckpointData | None:
        """Load checkpoint if enabled, auto_resume is on, file exists, and category matches."""
        if not self._enabled or not self._auto_resume:
            return None

        path = self._get_path(data_identifier, category)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                raw = json.load(f)

            if raw.get("category") != category:
                logger.warning(
                    f"Checkpoint category mismatch: expected '{category}', "
                    f"got '{raw.get('category')}'"
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
                with open(path) as f:
                    raw = json.load(f)
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

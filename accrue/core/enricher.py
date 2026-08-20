"""Enricher — Checkpoint-capable pipeline runner.

Internal runner for Accrue. Created via ``Pipeline.runner(config)``
for repeated execution with checkpointing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from ..pipeline import Pipeline
from ..pipeline.pipeline import _merge_results_into_df
from ..utils.logger import get_logger
from .checkpoint import (
    DEFAULT_CATEGORY,
    CheckpointManager,
    CheckpointSession,
    derive_data_identifier,
)
from .config import EnrichmentConfig

logger = get_logger(__name__)


class Enricher:
    """Orchestrates column-oriented pipeline execution over a DataFrame.

    Created via ``Pipeline.runner(config)`` for repeated execution with
    checkpointing.  Field specs always come from the pipeline's steps.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        config: EnrichmentConfig | None = None,
        hooks: Any = None,
    ) -> None:
        """Create an Enricher (typically via ``Pipeline.runner(config)``).

        Args:
            pipeline: The Pipeline to execute.
            config: Optional :class:`EnrichmentConfig` controlling concurrency,
                retries, caching, checkpointing, and progress display.
            hooks: Optional :class:`EnrichmentHooks` for lifecycle callbacks.
        """
        self.pipeline = pipeline
        self.config = config or EnrichmentConfig()
        self._checkpoint = CheckpointManager(self.config)
        self._hooks = hooks

    # -- sync entry point ------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        overwrite_fields: bool | None = None,
        data_identifier: str | None = None,
    ) -> pd.DataFrame:
        """Synchronous wrapper around :meth:`run_async`.

        Raises ``RuntimeError`` if called from inside a running event loop
        (use ``run_async`` directly in that case).
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Enricher.run() cannot be called from inside an async context. "
                "Use 'await enricher.run_async(...)' instead."
            )
        except RuntimeError as exc:
            # Re-raise our own RuntimeError; swallow the "no running event loop" one
            if "run_async" in str(exc):
                raise
        return asyncio.run(self.run_async(df, overwrite_fields, data_identifier))

    # -- async entry point -----------------------------------------------

    async def run_async(
        self,
        df: pd.DataFrame,
        overwrite_fields: bool | None = None,
        data_identifier: str | None = None,
    ) -> pd.DataFrame:
        """Execute the pipeline across all rows of *df*.

        Args:
            df: Input DataFrame. Rows are converted to dicts internally.
            overwrite_fields: Whether to overwrite non-empty existing columns.
                Defaults to ``config.overwrite_fields``.
            data_identifier: Stable identifier for checkpoint resume. If
                ``None``, derived from the DataFrame's column names and length.

        Returns:
            A copy of *df* with enrichment columns added/updated.
        """
        if overwrite_fields is None:
            overwrite_fields = self.config.overwrite_fields

        # Convert DataFrame to rows, replacing NaN/NaT/pd.NA with None.
        # astype(object) is required first; otherwise float columns keep nan
        # even after where(..., None) because pandas preserves dtype.
        # Done before the identifier is derived so the checkpoint id covers
        # the same normalized values Pipeline.run() would hash.
        rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

        if data_identifier is None:
            data_identifier = derive_data_identifier(list(df.columns), len(df), rows)

        category = DEFAULT_CATEGORY

        # Field specs from steps
        fields_dict = self.pipeline._collect_field_specs()

        # Checkpoint resume
        cp = self._checkpoint.load(
            data_identifier,
            category,
            expected_total_rows=len(df),
            expected_fields=fields_dict,
            expected_steps=self.pipeline.step_names,
        )
        session = CheckpointSession(
            self._checkpoint,
            data_identifier=data_identifier,
            category=category,
            total_rows=len(df),
            fields_dict=fields_dict,
            checkpoint=cp,
        )
        prior_step_results = session.prior_step_results or None
        # Cells that errored in a prior run re-run even though their step is
        # checkpointed — checkpointed no longer means done for failed cells
        # (#129).  Their successful neighbours are served from the checkpoint.
        # Resolving up front adds the downstream cells a healed value
        # invalidates, and tells the session which cells the run really owns.
        retry_cells = (
            self.pipeline._resolve_retry_map(
                session.retry_cells, prior_step_results or {}, len(rows)
            )
            or None
        )
        if retry_cells:
            session.note_retried_cells(retry_cells)
        if cp is not None:
            logger.info(f"Resuming from checkpoint: skipping {session.completed}")
            if retry_cells:
                logger.info(
                    "Re-running %d failed cell(s) from checkpoint: %s",
                    sum(len(v) for v in retry_cells.values()),
                    {k: len(v) for k, v in retry_cells.items()},
                )

        # Set up cache manager
        cache_manager = None
        if self.config.enable_caching:
            from .cache import CacheManager

            cache_manager = CacheManager(
                cache_dir=self.config.cache_dir,
                ttl=self.config.cache_ttl,
            )

        # Set up partial checkpoint callback
        checkpoint_interval = self.config.checkpoint_interval
        on_partial_checkpoint = None
        if checkpoint_interval > 0 and self._checkpoint._enabled:
            on_partial_checkpoint = session.on_partial_checkpoint

        # Execute pipeline
        try:
            accumulated, errors, cost, _step_elapsed = await self.pipeline.execute(
                rows=rows,
                all_fields=fields_dict,
                config=self.config,
                prior_step_results=prior_step_results,
                on_step_complete=session.on_step_complete,
                cache_manager=cache_manager,
                on_partial_checkpoint=on_partial_checkpoint,
                hooks=self._hooks,
                retry_cells=retry_cells,
            )
        finally:
            if cache_manager is not None:
                cache_manager.close()

        # Log error summary if any
        if errors:
            error_summary = {}
            for err in errors:
                error_summary[err.step_name] = error_summary.get(err.step_name, 0) + 1
            logger.warning("Pipeline completed with %d row errors: %s", len(errors), error_summary)

        # Clean the checkpoint up on full success; keep it while cells are
        # still failing so a re-run — or Pipeline.retry_failed() — can
        # re-execute exactly those cells.
        session.finish()

        # Write results back to DataFrame (column-wise, O(n) not O(n²))
        df_out = _merge_results_into_df(df, accumulated, overwrite_fields=overwrite_fields)

        logger.info(f"Enrichment complete: {len(df_out)} rows")
        return df_out

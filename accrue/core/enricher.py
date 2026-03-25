"""Enricher — Checkpoint-capable pipeline runner.

Internal runner for Accrue. Created via ``Pipeline.runner(config)``
for repeated execution with checkpointing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from ..pipeline import Pipeline
from ..utils.logger import get_logger
from .checkpoint import CheckpointManager
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

        if data_identifier is None:
            data_identifier = f"df_{hash(str(df.columns.tolist()) + str(len(df)))}"

        category = "_default"

        # Field specs from steps
        fields_dict = self.pipeline._collect_field_specs()

        # Checkpoint resume
        prior_step_results: dict[str, list[dict[str, Any]]] | None = None
        completed_steps: list[str] = []
        checkpoint_results: dict[str, list[dict[str, Any]]] = {}

        cp = self._checkpoint.load(data_identifier, category)
        if cp is not None:
            # Validate checkpoint compatibility
            if cp.total_rows != len(df):
                logger.warning(
                    f"Checkpoint row count mismatch ({cp.total_rows} vs {len(df)}), starting fresh"
                )
            elif cp.fields_dict != fields_dict:
                logger.warning("Checkpoint fields_dict mismatch, starting fresh")
            else:
                prior_step_results = cp.step_results
                completed_steps = list(cp.completed_steps)
                checkpoint_results = dict(cp.step_results)
                logger.info(f"Resuming from checkpoint: skipping {completed_steps}")

        # Convert DataFrame to rows
        rows = df.to_dict(orient="records")

        # Build on_step_complete callback for checkpointing
        cb_completed = list(completed_steps)
        cb_results = dict(checkpoint_results)

        def on_step_complete(step_name: str, step_row_results: list[dict[str, Any]]) -> None:
            cb_completed.append(step_name)
            cb_results[step_name] = step_row_results
            self._checkpoint.save_step(
                data_identifier=data_identifier,
                category=category,
                step_name=step_name,
                step_row_results=step_row_results,
                total_rows=len(rows),
                fields_dict=fields_dict,
                existing_completed=cb_completed[:-1],
                existing_results={k: v for k, v in cb_results.items() if k != step_name},
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

            def on_partial_checkpoint(step_name, partial_results, completed_count):
                self._checkpoint.save_step(
                    data_identifier=data_identifier,
                    category=category,
                    step_name=step_name,
                    step_row_results=partial_results,
                    total_rows=len(rows),
                    fields_dict=fields_dict,
                    existing_completed=list(cb_completed),
                    existing_results=dict(cb_results),
                )

        # Execute pipeline
        try:
            accumulated, errors, cost = await self.pipeline.execute(
                rows=rows,
                all_fields=fields_dict,
                config=self.config,
                prior_step_results=prior_step_results,
                on_step_complete=on_step_complete,
                cache_manager=cache_manager,
                on_partial_checkpoint=on_partial_checkpoint,
                hooks=self._hooks,
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

        # Write results back to DataFrame
        df_out = df.copy()
        for idx in range(len(df_out)):
            for key, value in accumulated[idx].items():
                # Filter __ internal fields
                if key.startswith("__"):
                    continue
                # Respect overwrite_fields
                if not overwrite_fields and key in df_out.columns:
                    existing = df_out.at[df_out.index[idx], key]
                    if pd.notna(existing) and existing != "":
                        continue
                df_out.at[df_out.index[idx], key] = value

        # Cleanup checkpoint on success
        self._checkpoint.cleanup(data_identifier, category)

        logger.info(f"Enrichment complete: {len(df_out)} rows")
        return df_out

"""Pipeline — DAG-based, column-oriented step execution engine."""

from __future__ import annotations

import asyncio
import inspect
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from tqdm.auto import tqdm

from ..core.config import EnrichmentConfig
from ..core.exceptions import PipelineError, RowError
from ..core.hooks import (
    EnrichmentHooks,
    PipelineEndEvent,
    PipelineStartEvent,
    RowCompleteEvent,
    StepEndEvent,
    StepStartEvent,
    _fire_hook,
)
from ..schemas.base import CostSummary, StepUsage, UsageInfo
from ..steps.base import Step, StepContext, StepResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional step helpers
# ---------------------------------------------------------------------------


async def _evaluate_predicate(predicate: Callable, row: dict, prior_results: dict) -> bool:
    """Call a sync or async predicate, returning its bool result."""
    result = predicate(row, prior_results)
    if inspect.isawaitable(result):
        result = await result
    return bool(result)


async def _should_skip_row(step: Any, row: dict, prior_results: dict) -> bool:
    """Evaluate run_if/skip_if on a step.  Returns True if row should be skipped."""
    run_if = getattr(step, "run_if", None)
    skip_if = getattr(step, "skip_if", None)
    if run_if is not None:
        return not await _evaluate_predicate(run_if, row, prior_results)
    if skip_if is not None:
        return await _evaluate_predicate(skip_if, row, prior_results)
    return False


def _build_skip_values(step: Any) -> dict[str, Any]:
    """Build default values for skipped rows using FieldSpec defaults where available."""
    field_specs = getattr(step, "_field_specs", {})
    values: dict[str, Any] = {}
    for field_name in step.fields:
        spec = field_specs.get(field_name)
        if spec is not None and "default" in spec.model_fields_set:
            values[field_name] = spec.default
        else:
            values[field_name] = None
    return values


@dataclass
class PipelineResult:
    """Result from Pipeline.run() / Pipeline.run_async().

    Attributes:
        data: Enriched DataFrame or list[dict] (matches input type).
        cost: Aggregated token usage across all steps and rows.
        errors: Per-row errors (empty if all rows succeeded).
    """

    data: pd.DataFrame | list[dict[str, Any]]
    cost: CostSummary = field(default_factory=CostSummary)
    errors: list[RowError] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of rows that completed without error."""
        total = len(self.data)
        if total == 0:
            return 1.0
        return 1.0 - len(self.errors) / total

    @property
    def has_errors(self) -> bool:
        """True if any rows produced errors during pipeline execution."""
        return len(self.errors) > 0

    def summary(self) -> str:
        """Return a formatted summary of the pipeline run."""
        DIM = "\033[2m"
        BOLD = "\033[1m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        CYAN = "\033[36m"
        RESET = "\033[0m"

        total = len(self.data)
        errors = len(self.errors)
        ok = total - errors

        lines: list[str] = []
        lines.append("")
        lines.append(f"{DIM}{'─' * 50}{RESET}")
        lines.append(f"{BOLD}  Pipeline Complete{RESET}")
        lines.append(f"{DIM}{'─' * 50}{RESET}")

        # Rows
        status_color = GREEN if errors == 0 else YELLOW
        lines.append(
            f"  {status_color}{BOLD}{ok}{RESET}/{total} rows enriched"
            + (f"  {RED}{errors} errors{RESET}" if errors else "")
        )

        # Per-step breakdown
        if self.cost.steps:
            lines.append("")
            for name, usage in self.cost.steps.items():
                mode = f" {DIM}(batch){RESET}" if usage.execution_mode == "batch" else ""
                cache_info = ""
                if usage.cache_hits > 0:
                    pct = usage.cache_hit_rate * 100
                    cache_info = f"  {DIM}cache: {pct:.0f}%{RESET}"
                skip_info = ""
                if usage.rows_skipped > 0:
                    skip_info = f"  {DIM}skipped: {usage.rows_skipped}{RESET}"
                lines.append(
                    f"  {CYAN}{name}{RESET}  "
                    f"{DIM}{usage.model or 'cached'}{RESET}{mode}"
                    f"  {usage.rows_processed} rows"
                    f"{cache_info}{skip_info}"
                )

        # Tokens
        lines.append("")
        tokens = self.cost.total_tokens
        if tokens >= 1_000_000:
            token_str = f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            token_str = f"{tokens / 1_000:.1f}k"
        else:
            token_str = str(tokens)
        lines.append(
            f"  {DIM}tokens:{RESET} {token_str} total  "
            f"{DIM}({self.cost.total_prompt_tokens:,} in / "
            f"{self.cost.total_completion_tokens:,} out){RESET}"
        )

        # Errors
        if self.errors:
            lines.append("")
            shown = self.errors[:5]
            for err in shown:
                lines.append(f"  {RED}✗{RESET} Row {err.row_index}: {err.error}")
            if errors > 5:
                lines.append(f"  {DIM}... and {errors - 5} more{RESET}")

        lines.append(f"{DIM}{'─' * 50}{RESET}")
        lines.append("")
        return "\n".join(lines)


class Pipeline:
    """Orchestrates a DAG of steps with column-oriented execution.

    Construction validates: no duplicate step names, no missing
    dependencies, no cycles.  Execution runs steps level-by-level
    (topological order).  Steps within a level run in parallel;
    rows within a step run concurrently (bounded by semaphore).
    """

    def __init__(self, steps: list[Step]):
        """Build a pipeline from an ordered list of steps.

        Args:
            steps: Steps (LLMStep, FunctionStep, or any Step-protocol object)
                to execute.  Dependencies between steps are declared via each
                step's ``depends_on`` list; independent steps at the same
                topological level run in parallel.

        Raises:
            PipelineError: If step names are duplicated, a dependency references
                a step that doesn't exist, or the dependency graph contains a
                cycle.
        """
        self._steps = list(steps)
        self._step_map: dict[str, Step] = {}
        self._execution_levels: list[list[str]] = []
        self._validate_and_build()

    # -- public helpers --------------------------------------------------

    @property
    def step_names(self) -> list[str]:
        """All step names in execution order."""
        return [name for level in self._execution_levels for name in level]

    @property
    def execution_levels(self) -> list[list[str]]:
        """Topological execution levels (read-only copy)."""
        return [list(level) for level in self._execution_levels]

    def get_step(self, name: str) -> Step:
        """Look up a step by name.

        Args:
            name: The step name as passed to LLMStep/FunctionStep constructor.

        Returns:
            The Step instance.

        Raises:
            KeyError: If no step with the given name exists in this pipeline.
        """
        return self._step_map[name]

    # -- primary API -----------------------------------------------------

    def run(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        config: EnrichmentConfig | None = None,
        hooks: EnrichmentHooks | None = None,
    ) -> PipelineResult:
        """Synchronous entry point — the ONE way to use Accrue.

        Accepts a DataFrame or ``list[dict]``. Output type matches input type.

        Raises ``RuntimeError`` if called from inside a running event loop
        (use ``await pipeline.run_async(data)`` in that case).
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Pipeline.run() cannot be called from inside an async context. "
                "Use 'await pipeline.run_async(...)' instead."
            )
        except RuntimeError as exc:
            if "run_async" in str(exc):
                raise
        return asyncio.run(self.run_async(data, config, hooks=hooks))

    async def run_async(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        config: EnrichmentConfig | None = None,
        hooks: EnrichmentHooks | None = None,
    ) -> PipelineResult:
        """Async entry point — ``await pipeline.run_async(data)``.

        Args:
            data: Input rows as a DataFrame or ``list[dict]``.  Output type
                matches the input type.
            config: Optional :class:`EnrichmentConfig` controlling concurrency,
                retries, caching, and progress display.  Defaults are sensible
                for most workloads.
            hooks: Optional :class:`EnrichmentHooks` for lifecycle callbacks
                (pipeline start/end, step start/end, row complete).

        Returns:
            A :class:`PipelineResult` containing the enriched data, aggregated
            cost/usage info, and any per-row errors.
        """
        config = config or EnrichmentConfig()
        hooks = hooks or EnrichmentHooks()

        # Collect field specs from steps
        all_fields = self._collect_field_specs()

        # Convert input to rows
        if isinstance(data, list):
            rows = data
            input_is_list = True
        else:
            rows = data.to_dict(orient="records")
            input_is_list = False

        # Set up cache manager
        cache_manager = None
        if config.enable_caching:
            from ..core.cache import CacheManager

            cache_manager = CacheManager(
                cache_dir=config.cache_dir,
                ttl=config.cache_ttl,
            )

        # Fire on_pipeline_start
        await _fire_hook(
            hooks.on_pipeline_start,
            PipelineStartEvent(
                step_names=self.step_names,
                num_rows=len(rows),
                config=config,
            ),
        )

        pipeline_start = _time.monotonic()
        accumulated = None
        errors: list[RowError] = []
        cost = CostSummary()

        try:
            # Execute
            accumulated, errors, cost = await self.execute(
                rows=rows,
                all_fields=all_fields,
                config=config,
                cache_manager=cache_manager,
                hooks=hooks,
            )
        finally:
            if cache_manager is not None:
                cache_manager.close()
            # Fire on_pipeline_end even on error
            elapsed = _time.monotonic() - pipeline_start
            await _fire_hook(
                hooks.on_pipeline_end,
                PipelineEndEvent(
                    num_rows=len(rows),
                    total_errors=len(errors),
                    cost=cost,
                    elapsed_seconds=elapsed,
                ),
            )

        # Build output matching input type
        if input_is_list:
            result_rows: list[dict[str, Any]] = []
            for idx, row in enumerate(rows):
                merged = dict(row)
                for key, value in accumulated[idx].items():
                    if not key.startswith("__"):
                        merged[key] = value
                result_rows.append(merged)
            return PipelineResult(data=result_rows, cost=cost, errors=errors)
        else:
            df_out = self._build_result_df(data, accumulated, config)
            return PipelineResult(data=df_out, cost=cost, errors=errors)

    def runner(self, config: EnrichmentConfig | None = None) -> Any:
        """Create a reusable :class:`Enricher` runner for this pipeline.

        Use when you need repeated execution with checkpointing, or want to
        manage the runner lifecycle yourself (e.g. in a server context).

        Args:
            config: Optional :class:`EnrichmentConfig`.  Passed to the
                Enricher and used for every subsequent ``run()`` call.

        Returns:
            An :class:`Enricher` instance bound to this pipeline and config.
        """
        from ..core.enricher import Enricher

        return Enricher(pipeline=self, config=config)

    def clear_cache(self, step: str | None = None, cache_dir: str = ".accrue") -> int:
        """Delete cached step results from the SQLite cache.

        Args:
            step: If provided, only delete entries for this step name.
                If ``None``, delete all cached entries.
            cache_dir: Directory containing the cache database.

        Returns:
            Number of cache entries deleted.
        """
        from ..core.cache import CacheManager

        mgr = CacheManager(cache_dir=cache_dir, ttl=0)
        try:
            return mgr.delete_step(step) if step else mgr.delete_all()
        finally:
            mgr.close()

    def _collect_field_specs(self) -> dict[str, dict[str, Any]]:
        """Collect field specs from all steps.

        LLMSteps with inline field specs (FieldSpec objects) contribute their
        specs serialised as dicts.  FunctionSteps and LLMSteps with list fields
        contribute empty specs.
        """
        all_fields: dict[str, dict[str, Any]] = {}
        for step in self._steps:
            # Check if step has inline field specs (_field_specs from dict fields)
            field_specs = getattr(step, "_field_specs", {})
            for field_name in step.fields:
                if field_name.startswith("__"):
                    continue
                if field_name in field_specs:
                    spec = field_specs[field_name]
                    # FieldSpec objects → dict; plain dicts pass through
                    if hasattr(spec, "model_dump"):
                        all_fields[field_name] = spec.model_dump(exclude_none=True)
                    else:
                        all_fields[field_name] = spec
                elif field_name not in all_fields:
                    all_fields[field_name] = {}
        return all_fields

    def _build_result_df(
        self,
        df: pd.DataFrame,
        accumulated: list[dict[str, Any]],
        config: EnrichmentConfig | None = None,
    ) -> pd.DataFrame:
        """Build result DataFrame from accumulated step outputs."""
        overwrite_fields = True
        if config is not None:
            overwrite_fields = config.overwrite_fields

        df_out = df.copy()
        for idx in range(len(df_out)):
            for key, value in accumulated[idx].items():
                if key.startswith("__"):
                    continue
                if not overwrite_fields and key in df_out.columns:
                    existing = df_out.at[df_out.index[idx], key]
                    if pd.notna(existing) and existing != "":
                        continue
                df_out.at[df_out.index[idx], key] = value
        return df_out

    # -- validation & DAG build -----------------------------------------

    def _validate_and_build(self) -> None:
        names = [s.name for s in self._steps]

        # Duplicate names
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise PipelineError(
                f"Duplicate step names: {dupes}. Each step must have a unique name."
            )

        self._step_map = {s.name: s for s in self._steps}

        # Missing dependencies
        for step in self._steps:
            for dep in step.depends_on:
                if dep not in self._step_map:
                    raise PipelineError(f"Step '{step.name}' depends on unknown step '{dep}'")

        self._execution_levels = self._topological_sort()

    def _topological_sort(self) -> list[list[str]]:
        """Kahn's algorithm returning grouped execution levels."""
        in_degree: dict[str, int] = {s.name: len(s.depends_on) for s in self._steps}

        # Reverse adjacency: step -> list of steps that depend on it
        dependents: dict[str, list[str]] = {s.name: [] for s in self._steps}
        for step in self._steps:
            for dep in step.depends_on:
                dependents[dep].append(step.name)

        current_level = [name for name, deg in in_degree.items() if deg == 0]
        if not current_level:
            step_names = list(in_degree.keys())
            raise PipelineError(
                f"Cycle detected: no steps without dependencies. "
                f"All steps have unresolved dependencies: {step_names}"
            )

        levels: list[list[str]] = []
        processed: set[str] = set()

        while current_level:
            levels.append(sorted(current_level))
            next_level: list[str] = []
            for name in current_level:
                processed.add(name)
                for dependent in dependents[name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_level.append(dependent)
            current_level = next_level

        if len(processed) != len(self._steps):
            remaining = {s.name for s in self._steps} - processed
            raise PipelineError(f"Cycle detected involving steps: {remaining}")

        return levels

    # -- execution -------------------------------------------------------

    async def execute(
        self,
        rows: list[dict[str, Any]],
        all_fields: dict[str, dict[str, Any]],
        config: EnrichmentConfig | None = None,
        prior_step_results: dict[str, list[dict[str, Any]]] | None = None,
        on_step_complete: Callable[[str, list[dict[str, Any]]], None] | None = None,
        cache_manager: Any = None,
        on_partial_checkpoint: Callable[[str, list[dict], int], None] | None = None,
        hooks: EnrichmentHooks | None = None,
    ) -> tuple[list[dict[str, Any]], list[RowError], CostSummary]:
        """Execute the pipeline across all rows (column-oriented).

        Args:
            rows: Row dicts (converted from DataFrame at the Enricher boundary).
            all_fields: field_name -> field_spec dict.
            config: Optional EnrichmentConfig.
            prior_step_results: Pre-populated results for checkpoint resume.
            on_step_complete: Sync callback fired after each step completes.
            cache_manager: Optional CacheManager for input-hash caching.
            on_partial_checkpoint: Callback(step_name, results, completed_count)
                fired every checkpoint_interval rows.
            hooks: Optional EnrichmentHooks for lifecycle events.

        Returns:
            Tuple of (accumulated results, row errors, cost summary).
        """
        hooks = hooks or EnrichmentHooks()

        max_workers = config.max_workers if config is not None else 3
        on_error = config.on_error if config is not None else "continue"
        show_progress = config.enable_progress_bar if config is not None else True
        checkpoint_interval = config.checkpoint_interval if config is not None else 0

        semaphore = asyncio.Semaphore(max_workers)
        num_rows = len(rows)

        # Pre-populate step_values from checkpoint data
        step_values: dict[str, list[dict[str, Any]]] = {}
        if prior_step_results:
            step_values.update(prior_step_results)

        all_errors: list[RowError] = []
        step_usage_map: dict[str, StepUsage] = {}

        total_steps = sum(len(level) for level in self._execution_levels)
        step_bar = tqdm(
            total=total_steps,
            desc="Pipeline",
            unit="step",
            disable=not show_progress,
        )

        for level_idx, level in enumerate(self._execution_levels):
            # Only execute steps not already in step_values (i.e. not resumed)
            steps_to_run = [name for name in level if name not in step_values]
            skipped = [name for name in level if name in step_values]

            # Advance bar for checkpointed/skipped steps
            if skipped:
                step_bar.update(len(skipped))

            if steps_to_run:
                step_bar.set_postfix(step=", ".join(steps_to_run))

                # Fire on_step_start for each step in this level
                for step_name in steps_to_run:
                    await _fire_hook(
                        hooks.on_step_start,
                        StepStartEvent(
                            step_name=step_name,
                            num_rows=num_rows,
                            level=level_idx,
                        ),
                    )

                level_coros = [
                    self._execute_step(
                        self._step_map[step_name],
                        rows,
                        all_fields,
                        config,
                        step_values,
                        semaphore,
                        num_rows,
                        on_error,
                        cache_manager=cache_manager,
                        checkpoint_interval=checkpoint_interval,
                        on_partial_checkpoint=on_partial_checkpoint,
                        hooks=hooks,
                        show_progress=show_progress,
                    )
                    for step_name in steps_to_run
                ]
                step_results_list = await asyncio.gather(*level_coros)

                for step_name, (step_errors, usage, elapsed_s) in zip(
                    steps_to_run, step_results_list
                ):
                    all_errors.extend(step_errors)
                    if usage:
                        step_usage_map[step_name] = usage

                    # Fire on_step_end
                    await _fire_hook(
                        hooks.on_step_end,
                        StepEndEvent(
                            step_name=step_name,
                            num_rows=num_rows,
                            num_errors=len(step_errors),
                            usage=usage,
                            elapsed_seconds=elapsed_s,
                        ),
                    )

                step_bar.update(len(steps_to_run))

                # Fire callback for each newly-executed step
                if on_step_complete is not None:
                    for step_name in steps_to_run:
                        on_step_complete(step_name, step_values[step_name])

        step_bar.close()

        # Merge all step results in execution order
        accumulated: list[dict[str, Any]] = [{} for _ in range(num_rows)]
        for level in self._execution_levels:
            for step_name in level:
                for idx in range(num_rows):
                    accumulated[idx].update(step_values[step_name][idx])

        # Build cost summary
        cost = CostSummary(
            total_prompt_tokens=sum(s.prompt_tokens for s in step_usage_map.values()),
            total_completion_tokens=sum(s.completion_tokens for s in step_usage_map.values()),
            total_tokens=sum(s.total_tokens for s in step_usage_map.values()),
            steps=step_usage_map,
        )

        return accumulated, all_errors, cost

    async def _execute_step(
        self,
        step: Step,
        rows: list[dict[str, Any]],
        all_fields: dict[str, dict[str, Any]],
        config: EnrichmentConfig | None,
        step_values: dict[str, list[dict[str, Any]]],
        semaphore: asyncio.Semaphore,
        num_rows: int,
        on_error: str = "continue",
        cache_manager: Any = None,
        checkpoint_interval: int = 0,
        on_partial_checkpoint: Callable | None = None,
        hooks: EnrichmentHooks | None = None,
        show_progress: bool = False,
    ) -> tuple[list[RowError], StepUsage | None, float]:
        """Execute a single step across all rows concurrently.

        Args:
            step: The step instance to execute.
            rows: All input rows (original data).
            all_fields: Merged field specs from all pipeline steps.
            config: Optional EnrichmentConfig.
            step_values: Mutable mapping of step_name -> per-row results.
                This method writes its results into ``step_values[step.name]``.
            semaphore: Concurrency limiter (``max_workers``).
            num_rows: Total number of rows.
            on_error: ``"continue"`` (default) or ``"raise"``.
            cache_manager: Optional CacheManager for input-hash caching.
            checkpoint_interval: Save partial progress every N rows (0 = off).
            on_partial_checkpoint: Callback fired at each checkpoint interval.
            hooks: Optional EnrichmentHooks for row-level lifecycle events.

        Returns:
            Tuple of (row errors, aggregated StepUsage or None, elapsed seconds).

        Side effects:
            Populates ``step_values[step.name]`` with per-row result dicts.
        """
        # Dispatch to batch path if eligible
        if getattr(step, "is_batch_eligible", False):
            return await self._execute_step_batch(
                step=step,
                rows=rows,
                all_fields=all_fields,
                config=config,
                step_values=step_values,
                num_rows=num_rows,
                on_error=on_error,
                cache_manager=cache_manager,
                hooks=hooks,
                show_progress=show_progress,
            )

        from ..core.cache import _compute_step_cache_key

        hooks = hooks or EnrichmentHooks()
        step_start = _time.monotonic()

        # Slice fields for this step (internal __ fields won't be in all_fields — that's fine)
        step_fields = {f: all_fields[f] for f in step.fields if f in all_fields}

        results: list[dict[str, Any]] = [{} for _ in range(num_rows)]
        errors: list[RowError] = []
        usage_list: list[UsageInfo] = []

        cache_hits = 0
        cache_misses = 0
        rows_skipped = 0
        step_cache_enabled = cache_manager is not None and getattr(step, "cache", True)

        # Track per-row from_cache and skipped status
        row_from_cache: list[bool] = [False] * num_rows
        row_was_skipped: list[bool] = [False] * num_rows

        row_bar = tqdm(
            total=num_rows,
            desc=f"  {step.name}",
            unit="row",
            disable=not show_progress,
            leave=False,
        )

        async def process_row(idx: int) -> StepResult | BaseException:
            nonlocal cache_hits, cache_misses, rows_skipped
            async with semaphore:
                # Gather prior results from dependency steps
                prior: dict[str, Any] = {}
                for dep_name in step.depends_on:
                    if dep_name in step_values:
                        prior.update(step_values[dep_name][idx])

                # Evaluate run_if/skip_if predicate
                if await _should_skip_row(step, rows[idx], prior):
                    rows_skipped += 1
                    row_was_skipped[idx] = True
                    return StepResult(values=_build_skip_values(step))

                cache_key = None
                if step_cache_enabled:
                    cache_key = _compute_step_cache_key(step, rows[idx], prior, step_fields)
                    cached = cache_manager.get(cache_key)
                    if cached is not None:
                        cache_hits += 1
                        row_from_cache[idx] = True
                        return StepResult(values=cached)

                ctx = StepContext(
                    row=rows[idx],
                    fields=step_fields,
                    prior_results=prior,
                    config=config,
                )

                result = await step.run(ctx)

                if step_cache_enabled:
                    cache_manager.set(cache_key, step.name, result.values)

                if step_cache_enabled:
                    cache_misses += 1
                return result

        if checkpoint_interval > 0 and on_partial_checkpoint is not None:
            # as_completed pattern: fire partial saves as rows complete
            async def tracked_row(idx: int):
                try:
                    return (idx, await process_row(idx), None)
                except BaseException as exc:
                    return (idx, None, exc)

            tasks = [asyncio.create_task(tracked_row(i)) for i in range(num_rows)]
            completed_count = 0
            for coro in asyncio.as_completed(tasks):
                idx, result_or_none, exc = await coro

                if exc is not None:
                    row_error = RowError(
                        row_index=idx,
                        step_name=step.name,
                        error=exc,
                    )
                    errors.append(row_error)
                    results[idx] = {f: None for f in step.fields}
                    logger.warning(
                        "Row %d failed in step '%s': %s",
                        idx,
                        step.name,
                        exc,
                    )

                    # Fire on_row_complete for error
                    await _fire_hook(
                        hooks.on_row_complete,
                        RowCompleteEvent(
                            step_name=step.name,
                            row_index=idx,
                            values={f: None for f in step.fields},
                            error=exc,
                            from_cache=False,
                            skipped=row_was_skipped[idx],
                        ),
                    )

                    if on_error == "raise":
                        step_values[step.name] = results
                        # Cancel remaining tasks
                        for t in tasks:
                            t.cancel()
                        raise exc
                else:
                    results[idx] = result_or_none.values
                    if result_or_none.usage:
                        usage_list.append(result_or_none.usage)

                    # Fire on_row_complete for success
                    await _fire_hook(
                        hooks.on_row_complete,
                        RowCompleteEvent(
                            step_name=step.name,
                            row_index=idx,
                            values=result_or_none.values,
                            error=None,
                            from_cache=row_from_cache[idx],
                            skipped=row_was_skipped[idx],
                        ),
                    )

                completed_count += 1
                row_bar.update(1)
                if completed_count % checkpoint_interval == 0:
                    on_partial_checkpoint(step.name, results, completed_count)
        else:

            async def _tracked_row(idx: int):
                try:
                    return (idx, await process_row(idx), None)
                except BaseException as exc:
                    return (idx, None, exc)

            tasks = [asyncio.create_task(_tracked_row(i)) for i in range(num_rows)]
            for coro in asyncio.as_completed(tasks):
                idx, result_or_none, exc = await coro
                row_bar.update(1)

                if exc is not None:
                    row_error = RowError(
                        row_index=idx,
                        step_name=step.name,
                        error=exc,
                    )
                    errors.append(row_error)
                    results[idx] = {f: None for f in step.fields}
                    logger.warning(
                        "Row %d failed in step '%s': %s",
                        idx,
                        step.name,
                        exc,
                    )

                    # Fire on_row_complete for error
                    await _fire_hook(
                        hooks.on_row_complete,
                        RowCompleteEvent(
                            step_name=step.name,
                            row_index=idx,
                            values={f: None for f in step.fields},
                            error=exc,
                            from_cache=False,
                            skipped=row_was_skipped[idx],
                        ),
                    )

                    if on_error == "raise":
                        step_values[step.name] = results
                        for t in tasks:
                            t.cancel()
                        raise exc
                else:
                    results[idx] = result_or_none.values
                    if result_or_none.usage:
                        usage_list.append(result_or_none.usage)

                    # Fire on_row_complete for success
                    await _fire_hook(
                        hooks.on_row_complete,
                        RowCompleteEvent(
                            step_name=step.name,
                            row_index=idx,
                            values=result_or_none.values,
                            error=None,
                            from_cache=row_from_cache[idx],
                            skipped=row_was_skipped[idx],
                        ),
                    )

        row_bar.close()

        step_values[step.name] = results

        # Aggregate usage for this step
        step_usage: StepUsage | None = None
        if usage_list or cache_hits > 0 or cache_misses > 0 or rows_skipped > 0:
            # rows_processed: when caching is active, count via cache stats
            # (FunctionSteps don't emit usage, so len(usage_list) would be 0)
            if step_cache_enabled:
                rows_processed = cache_hits + cache_misses
            else:
                rows_processed = len(usage_list)

            step_usage = StepUsage(
                prompt_tokens=sum(u.prompt_tokens for u in usage_list),
                completion_tokens=sum(u.completion_tokens for u in usage_list),
                total_tokens=sum(u.total_tokens for u in usage_list),
                rows_processed=rows_processed,
                rows_skipped=rows_skipped,
                model=usage_list[0].model if usage_list else "",
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )

        elapsed = _time.monotonic() - step_start
        return errors, step_usage, elapsed

    async def _execute_step_batch(
        self,
        step: Step,
        rows: list[dict[str, Any]],
        all_fields: dict[str, dict[str, Any]],
        config: EnrichmentConfig | None,
        step_values: dict[str, list[dict[str, Any]]],
        num_rows: int,
        on_error: str = "continue",
        cache_manager: Any = None,
        hooks: EnrichmentHooks | None = None,
        show_progress: bool = False,
    ) -> tuple[list[RowError], StepUsage | None, float]:
        """Execute a batch-eligible LLMStep via the provider Batch API.

        Flow:
          1. Classify rows: skipped / cached / uncached
          2. Build BatchRequests from uncached rows
          3. Auto-chunk and submit batches
          4. Poll until complete (with KeyboardInterrupt cancel)
          5. Parse responses, cache results
          6. Realtime fallback for failed rows
          7. Merge and fire hooks
        """
        import json as _json

        from ..core.cache import _compute_step_cache_key
        from ..steps.providers.base import BatchRequest

        hooks = hooks or EnrichmentHooks()
        step_start = _time.monotonic()

        step_fields = {f: all_fields[f] for f in step.fields if f in all_fields}

        results: list[dict[str, Any]] = [{} for _ in range(num_rows)]
        errors: list[RowError] = []
        usage_list: list[UsageInfo] = []

        cache_hits = 0
        cache_misses = 0
        rows_skipped = 0
        step_cache_enabled = cache_manager is not None and getattr(step, "cache", True)

        row_from_cache: list[bool] = [False] * num_rows
        row_was_skipped: list[bool] = [False] * num_rows

        # Resolve batch config
        poll_interval = 60.0
        batch_timeout = 86400.0
        batch_max_requests = 50000
        if config is not None:
            poll_interval = config.batch_poll_interval
            batch_timeout = config.batch_timeout
            batch_max_requests = config.batch_max_requests

        # Get the batch-capable client
        client = step._resolve_client()

        row_bar = tqdm(
            total=num_rows,
            desc=f"  {step.name} (batch)",
            unit="row",
            disable=not show_progress,
            leave=False,
        )

        # ── Phase 1: Classify rows ─────────────────────────────────────
        uncached_indices: list[int] = []
        # Cache keys for uncached rows (indexed by row idx)
        cache_keys: dict[int, str | None] = {}

        for idx in range(num_rows):
            # Gather prior results
            prior: dict[str, Any] = {}
            for dep_name in step.depends_on:
                if dep_name in step_values:
                    prior.update(step_values[dep_name][idx])

            # Evaluate predicates
            if await _should_skip_row(step, rows[idx], prior):
                rows_skipped += 1
                row_was_skipped[idx] = True
                results[idx] = _build_skip_values(step)
                row_bar.update(1)
                continue

            # Cache check
            cache_key = None
            if step_cache_enabled:
                cache_key = _compute_step_cache_key(step, rows[idx], prior, step_fields)
                cached = cache_manager.get(cache_key)
                if cached is not None:
                    cache_hits += 1
                    row_from_cache[idx] = True
                    results[idx] = cached
                    row_bar.update(1)
                    continue

            cache_keys[idx] = cache_key
            uncached_indices.append(idx)

        # If all rows are cached or skipped, no batch needed
        if not uncached_indices:
            row_bar.close()
            step_values[step.name] = results

            step_usage = None
            if cache_hits > 0 or rows_skipped > 0:
                step_usage = StepUsage(
                    rows_processed=cache_hits,
                    rows_skipped=rows_skipped,
                    cache_hits=cache_hits,
                    cache_misses=0,
                    execution_mode="batch",
                )

            # Fire hooks for cached/skipped rows
            for idx in range(num_rows):
                await _fire_hook(
                    hooks.on_row_complete,
                    RowCompleteEvent(
                        step_name=step.name,
                        row_index=idx,
                        values=results[idx],
                        error=None,
                        from_cache=row_from_cache[idx],
                        skipped=row_was_skipped[idx],
                    ),
                )

            elapsed = _time.monotonic() - step_start
            return errors, step_usage, elapsed

        # ── Phase 2: Build batch requests ──────────────────────────────
        batch_requests: list[BatchRequest] = []
        for idx in uncached_indices:
            prior: dict[str, Any] = {}
            for dep_name in step.depends_on:
                if dep_name in step_values:
                    prior.update(step_values[dep_name][idx])

            ctx = StepContext(
                row=rows[idx],
                fields=step_fields,
                prior_results=prior,
                config=config,
            )
            messages, call_kwargs = step.build_messages(ctx)
            batch_requests.append(
                BatchRequest(
                    custom_id=f"row-{idx}",
                    messages=messages,
                    model=call_kwargs["model"],
                    temperature=call_kwargs["temperature"],
                    max_tokens=call_kwargs["max_tokens"],
                    response_format=call_kwargs["response_format"],
                    tools=call_kwargs.get("tools"),
                    provider_kwargs=call_kwargs.get("provider_kwargs"),
                )
            )

        logger.info(
            "Step '%s': batch mode — %d uncached, %d cached, %d skipped",
            step.name,
            len(uncached_indices),
            cache_hits,
            rows_skipped,
        )

        # ── Phase 3: Auto-chunk and submit ─────────────────────────────
        chunks = [
            batch_requests[i : i + batch_max_requests]
            for i in range(0, len(batch_requests), batch_max_requests)
        ]
        if len(chunks) > 1:
            logger.info(
                "Step '%s': auto-chunking %d requests into %d batches (max %d per batch)",
                step.name,
                len(batch_requests),
                len(chunks),
                batch_max_requests,
            )

        batch_ids: list[str] = []
        for chunk in chunks:
            bid = await client.submit_batch(
                chunk,
                metadata={"step": step.name, "pipeline": "accrue"},
            )
            batch_ids.append(bid)
            logger.info("Step '%s': submitted batch %s (%d requests)", step.name, bid, len(chunk))

        # ── Phase 4: Poll (with KeyboardInterrupt handling) ────────────
        from ..steps.providers.base import BatchResult

        all_responses: dict[str, Any] = {}
        all_failed: list[str] = []
        all_errors_map: dict[str, str] = {}
        primary_batch_id = batch_ids[0] if batch_ids else ""

        row_bar.set_description(f"  {step.name} (batch: polling)")

        try:
            for bid in batch_ids:
                batch_result: BatchResult = await client.poll_batch(
                    bid,
                    poll_interval=poll_interval,
                    timeout=batch_timeout,
                )
                all_responses.update(batch_result.responses)
                all_failed.extend(batch_result.failed_ids)
                all_errors_map.update(batch_result.errors)
        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt — cancelling %d batch(es): %s",
                len(batch_ids),
                batch_ids,
            )
            for bid in batch_ids:
                await client.cancel_batch(bid)
            raise

        # ── Phase 5: Parse results and cache ───────────────────────────
        retry_indices: list[int] = []
        succeeded = 0

        for idx in uncached_indices:
            custom_id = f"row-{idx}"
            if custom_id in all_responses:
                response = all_responses[custom_id]
                try:
                    step_result = step.parse_response(response)
                    results[idx] = step_result.values
                    if step_result.usage:
                        usage_list.append(step_result.usage)

                    # Cache store
                    if step_cache_enabled and cache_keys.get(idx):
                        cache_manager.set(cache_keys[idx], step.name, step_result.values)
                    cache_misses += 1
                    succeeded += 1
                    row_bar.update(1)
                except (_json.JSONDecodeError, Exception) as exc:
                    logger.warning(
                        "Step '%s' batch parse failed for row %d: %s", step.name, idx, exc
                    )
                    retry_indices.append(idx)
            else:
                error_msg = all_errors_map.get(custom_id, "batch request failed")
                logger.warning("Step '%s' batch row %d failed: %s", step.name, idx, error_msg)
                retry_indices.append(idx)

        logger.info(
            "Step '%s': batch complete — %d/%d succeeded, %d to retry",
            step.name,
            succeeded,
            len(uncached_indices),
            len(retry_indices),
        )

        # ── Phase 6: Realtime fallback for failed rows ─────────────────
        for idx in retry_indices:
            prior: dict[str, Any] = {}
            for dep_name in step.depends_on:
                if dep_name in step_values:
                    prior.update(step_values[dep_name][idx])

            ctx = StepContext(
                row=rows[idx],
                fields=step_fields,
                prior_results=prior,
                config=config,
            )

            try:
                step_result = await step.run(ctx)
                results[idx] = step_result.values
                if step_result.usage:
                    usage_list.append(step_result.usage)

                if step_cache_enabled and cache_keys.get(idx):
                    cache_manager.set(cache_keys[idx], step.name, step_result.values)
                cache_misses += 1
                row_bar.update(1)
            except Exception as exc:
                row_error = RowError(
                    row_index=idx,
                    step_name=step.name,
                    error=exc,
                )
                errors.append(row_error)
                results[idx] = {f: None for f in step.fields}
                row_bar.update(1)

                if on_error == "raise":
                    step_values[step.name] = results
                    row_bar.close()
                    raise exc

        row_bar.close()

        # ── Phase 7: Merge and finalize ────────────────────────────────
        step_values[step.name] = results

        # Fire hooks for all rows
        for idx in range(num_rows):
            error_obj = next((e.error for e in errors if e.row_index == idx), None)
            await _fire_hook(
                hooks.on_row_complete,
                RowCompleteEvent(
                    step_name=step.name,
                    row_index=idx,
                    values=results[idx],
                    error=error_obj,
                    from_cache=row_from_cache[idx],
                    skipped=row_was_skipped[idx],
                ),
            )

        # Build step usage
        step_usage: StepUsage | None = None
        if usage_list or cache_hits > 0 or cache_misses > 0 or rows_skipped > 0:
            if step_cache_enabled:
                rows_processed = cache_hits + cache_misses
            else:
                rows_processed = len(usage_list)

            step_usage = StepUsage(
                prompt_tokens=sum(u.prompt_tokens for u in usage_list),
                completion_tokens=sum(u.completion_tokens for u in usage_list),
                total_tokens=sum(u.total_tokens for u in usage_list),
                rows_processed=rows_processed,
                rows_skipped=rows_skipped,
                model=usage_list[0].model if usage_list else "",
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                execution_mode="batch",
                batch_id=primary_batch_id,
            )

        elapsed = _time.monotonic() - step_start
        return errors, step_usage, elapsed

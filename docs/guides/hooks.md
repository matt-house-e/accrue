# Lifecycle Hooks

Hooks let you observe pipeline execution without modifying pipeline logic. Use them for logging, metrics, progress tracking, or integration with external systems.

## Minimal example

```python
from accrue import Pipeline, EnrichmentHooks

hooks = EnrichmentHooks(
    on_step_end=lambda e: print(f"Step '{e.step_name}' done in {e.elapsed_seconds:.1f}s"),
)

result = pipeline.run(data, hooks=hooks)
```

## Full hook registration

```python
from accrue import EnrichmentHooks

hooks = EnrichmentHooks(
    on_pipeline_start=lambda e: print(f"Starting {len(e.step_names)} steps, {e.num_rows} rows"),
    on_step_start=lambda e: print(f"  Step '{e.step_name}' starting ({e.num_rows} rows)"),
    on_step_end=lambda e: print(f"  Step '{e.step_name}' done in {e.elapsed_seconds:.1f}s"),
    on_row_complete=lambda e: log_row(e),
    on_pipeline_end=lambda e: print(f"Done. {e.total_errors} errors in {e.elapsed_seconds:.1f}s"),
)

result = pipeline.run(data, hooks=hooks)
```

## Event types

### PipelineStartEvent

Fired once at the beginning of pipeline execution.

| Field | Type | Description |
|-------|------|-------------|
| `step_names` | `list[str]` | Ordered list of step names |
| `num_rows` | `int` | Total rows in the input data |
| `config` | `EnrichmentConfig` | The config used for this run |

### PipelineEndEvent

Fired once at the end of pipeline execution (including on error).

| Field | Type | Description |
|-------|------|-------------|
| `num_rows` | `int` | Total rows processed |
| `total_errors` | `int` | Number of row-level errors |
| `cost` | `CostSummary` | Aggregated token usage |
| `elapsed_seconds` | `float` | Wall-clock time for the full run |

### StepStartEvent

Fired before each step begins processing.

| Field | Type | Description |
|-------|------|-------------|
| `step_name` | `str` | Name of the step |
| `num_rows` | `int` | Rows to process in this step |
| `level` | `int` | DAG level (0-based) |
| `execution_mode` | `str` | `"realtime"` or `"batch"` |

### StepEndEvent

Fired after each step finishes.

| Field | Type | Description |
|-------|------|-------------|
| `step_name` | `str` | Name of the step |
| `num_rows` | `int` | Rows processed |
| `num_errors` | `int` | Row-level errors in this step |
| `usage` | `StepUsage` | Token usage for this step |
| `elapsed_seconds` | `float` | Wall-clock time for this step |
| `execution_mode` | `str` | `"realtime"` or `"batch"` |
| `batch_id` | `str \| None` | Batch job ID (batch mode only) |

### RowCompleteEvent

Fired after each individual row completes within a step.

| Field | Type | Description |
|-------|------|-------------|
| `step_name` | `str` | Name of the step |
| `row_index` | `int` | Index of the completed row |
| `values` | `dict[str, Any]` | Enrichment values produced |
| `error` | `BaseException \| None` | Error if the row failed |
| `from_cache` | `bool` | Whether the result came from cache |
| `skipped` | `bool` | Whether the row was skipped by `run_if`/`skip_if` |

## Async hooks

Both sync and async callables work. Async hooks are awaited automatically:

```python
async def log_step_end(event):
    await metrics_client.record(
        step=event.step_name,
        duration=event.elapsed_seconds,
        errors=event.num_errors,
    )

hooks = EnrichmentHooks(on_step_end=log_step_end)
```

## Practical patterns

**Tracking batch progress:**

```python
hooks = EnrichmentHooks(
    on_step_start=lambda e: print(
        f"Step '{e.step_name}' [{e.execution_mode}] starting"
    ),
    on_step_end=lambda e: print(
        f"Step '{e.step_name}' [{e.execution_mode}] "
        f"batch_id={e.batch_id} "
        f"done in {e.elapsed_seconds:.1f}s"
    ),
)
```

**Counting cache hits per step:**

```python
def report_caching(event):
    usage = event.usage
    if usage.cache_hits > 0:
        print(
            f"Step '{event.step_name}': "
            f"{usage.cache_hits} cache hits, "
            f"{usage.cache_misses} misses "
            f"({usage.cache_hit_rate:.0%} hit rate)"
        )

hooks = EnrichmentHooks(on_step_end=report_caching)
```

**Error alerting:**

```python
def alert_on_errors(event):
    if event.error is not None:
        send_alert(f"Row {event.row_index} failed in '{event.step_name}': {event.error}")

hooks = EnrichmentHooks(on_row_complete=alert_on_errors)
```

## Importing event types

```python
from accrue import (
    EnrichmentHooks,
    PipelineStartEvent,
    PipelineEndEvent,
    StepStartEvent,
    StepEndEvent,
    RowCompleteEvent,
)
```

## Gotchas

- **All fields are optional.** Only subscribe to events you care about. Unset hooks are simply not called.
- **Hook errors never crash the pipeline.** If a hook raises an exception, it is caught and logged as a warning. Pipeline execution continues normally.
- **Hooks are passed to `run()`, not stored on config.** This keeps `EnrichmentConfig` serializable (plain data) while hooks can be arbitrary callables.
- **`on_row_complete` fires for every row in every step.** For a pipeline with 3 steps and 1,000 rows, it fires 3,000 times. Keep the callback lightweight.
- **`PipelineEndEvent` fires even on error.** Use it for cleanup or final reporting regardless of success or failure.

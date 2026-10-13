# Error Handling

Accrue provides two error modes, a structured exception hierarchy, and two layers of automatic retries. Pipelines are designed to return partial results rather than crash on the first failure.

## Error modes

### continue (default)

Collects errors and returns partial results. Rows that fail get `None` values for the step's fields; other rows are unaffected.

```python
from accrue import EnrichmentConfig

config = EnrichmentConfig(on_error="continue")
result = pipeline.run(data, config=config)
```

### raise

Fails fast on the first row error. Useful during development or when partial results are not acceptable.

```python
config = EnrichmentConfig(on_error="raise")
result = pipeline.run(data, config=config)  # Raises on first failure
```

## Reading errors from the result

```python
result = pipeline.run(data)

if result.has_errors:
    for err in result.errors:
        print(f"Row {err.row_index}, step '{err.step_name}': {err.error}")
        print(f"  Type: {err.error_type}")

print(f"Success rate: {result.success_rate:.1%}")
```

`result.errors` is a list of `RowError` dataclasses. Each contains:

- `row_index` -- the integer index of the failed row.
- `step_name` -- which step failed.
- `error` -- the original exception.
- `error_type` -- the exception class name (e.g. `"StepError"`).

`result.success_rate` is a float between 0.0 and 1.0 representing the fraction of rows without errors.

## Per-cell status: `result.row_status`

`result.errors` lists what failed in *this* pass. It doesn't tell a blank cell that failed apart from one a `run_if` predicate skipped, or from a step that really returned `None`. `result.row_status` does. It is a DataFrame with one column per step, in execution order, aligned to `result.data`'s index (a `RangeIndex` for `list[dict]` input):

```python
result.row_status
#    extract  verify
# 0       ok      ok
# 1    error      ok
# 2       ok skipped
```

| Value | Meaning |
|-------|---------|
| `"ok"` | The step produced values for this row: freshly, from the cache, or from the checkpoint. |
| `"error"` | The step raised after retries, or the checkpoint still records the cell as failed. |
| `"skipped"` | A `run_if` / `skip_if` predicate skipped the row. Its fields hold their FieldSpec defaults (or `None`). |

These are the same values as `row_complete.status` in the [run log](run-log.md). A cache hit is `"ok"`: where a value came from is a separate fact, reported by `RowCompleteEvent.from_cache` and `StepUsage.cache_hits`.

**Filter on the row, not the step.** A dependent step still runs on a row whose upstream cell failed. It receives `None` for the failed step's fields, and it can return a perfectly valid-looking value computed from nothing. That cell reads `"ok"`, because the dependent step itself did not raise. To get the rows you can trust:

```python
ok = ~result.row_status.eq("error").any(axis=1)
good = result.data[ok]
```

To export status next to the data, join it explicitly. `result.data` and `save()` never include it:

```python
result.data.join(result.row_status.add_prefix("status_")).to_csv("out.csv", index=False)
```

After [`retry_failed()`](caching.md#retrying-failed-cells), `row_status` still covers the whole dataset. Cells served from the checkpoint keep the outcome recorded for them. So a failure you left out with `rows=` / `steps=` reads `"error"` even though it is not in that pass's `result.errors`.

### Missing values stay `None`

A step's `None` stays `None` in `result.data`. On pandas 3, a column holding strings and `None` would otherwise be given the `str` dtype, which stores missing values as `NaN`. `NaN` is truthy, so `if row["flag"]:` would fire for every row that returned `None`. accrue builds such a column as `object`. That matches pandas 2 and the `list[dict]` path. Numeric columns still use `NaN` for missing values, so use `pd.isna()` there, never truthiness.

To re-run just those failed cells -- without paying again for the rows that worked -- see [`pipeline.retry_failed()`](caching.md#retrying-failed-cells). It needs `EnrichmentConfig(enable_checkpointing=True)` on the original run.

## Exception hierarchy

```
EnrichmentError (base)
├── FieldValidationError  -- invalid field specs
├── ConfigurationError    -- invalid configuration
├── StepError             -- step failed after retries
└── PipelineError         -- construction or execution failure
```

`RowError` is a dataclass (not an exception). It is collected in `result.errors` rather than raised.

### When each is raised

**FieldValidationError** -- at step construction time when field specs contain unknown keys, missing `prompt`, or invalid `type`/`enum` values.

**ConfigurationError** -- at config construction time when values are out of range.

**StepError** -- during execution when an LLM step exhausts all parse/validation retries or all API retries. Contains a `step_name` attribute.

**PipelineError** -- at construction time for duplicate step names, missing dependencies, or dependency cycles. Also raised at construction when `batch=True` and `grounding` are both set, or when `run_if` and `skip_if` are both set.

## Retry logic

Accrue has two independent retry layers:

### API retries (outer loop)

Handles transient provider errors: HTTP 429, 500, timeouts, and network failures.

- Controlled by `config.max_retries` (default: 3) and `config.retry_base_delay` (default: 1.0s).
- Uses exponential backoff with jitter: `base_delay * 2^attempt + random(0, 25%)`.
- Respects the provider's `Retry-After` header when present.

### Parse retries (inner loop)

Handles JSON parse errors and Pydantic validation failures. The error message is appended to the conversation so the LLM can self-correct.

- Controlled by `LLMStep(max_retries=2)` (step-level, default: 2).
- On each retry, the invalid response and the error are sent back to the model with a request to return valid JSON.

The two layers are nested: each API attempt gets up to `max_retries + 1` parse attempts.

## Importing exceptions

```python
from accrue import (
    EnrichmentError,
    FieldValidationError,
    StepError,
    PipelineError,
    RowError,
)
```

`ConfigurationError` is available from the exceptions module:

```python
from accrue.core.exceptions import ConfigurationError
```

## Gotchas

- In `"continue"` mode, failed rows still appear in `result.data` -- their enrichment fields are `None`. Check `result.has_errors` to detect partial results.
- `StepError` includes a `step_name` attribute. `EnrichmentError` (the base) includes optional `row_index` and `field` attributes for context.
- Hook errors (from `EnrichmentHooks`) are caught and logged separately. They never cause row failures or crash the pipeline.
- Parse retries reuse the same API connection. API retries reset the conversation and start fresh.

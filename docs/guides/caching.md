# Caching and Checkpointing

Accrue provides two complementary persistence features: caching (skip redundant API calls) and checkpointing (crash recovery). They serve different purposes and can be used independently or together.

## Caching

### Enable caching

```python
from accrue import Pipeline, LLMStep, EnrichmentConfig

pipeline = Pipeline([
    LLMStep("analyze", fields={"summary": "Summarize the company"}, model="gpt-4.1-mini"),
])

# Caching is on by default — no config flag needed.
result = pipeline.run(data)
```

Re-running the same pipeline with the same data skips cached rows entirely. No API calls are made for rows that already have results.

### How it works

- Results are stored in a SQLite database at `.accrue/cache.db` (WAL mode for concurrent access).
- Each cached entry is keyed by a SHA-256 hash of: step name, row data, prior results from dependency steps, field specs, model name, and temperature.
- Change a prompt, add a field, switch models, or adjust temperature and the cache auto-invalidates (different hash).
- TTL: default 3600 seconds (1 hour). Expired entries are lazily deleted on next read.

### Configuration

```python
config = EnrichmentConfig(
    enable_caching=True,
    cache_ttl=7200,          # 2 hours (default: 3600). 0 = no expiry.
    cache_dir=".accrue",     # Directory for cache.db (default: ".accrue")
)
```

### What is in the cache key

For LLMStep:
- Step name
- Row data (the full input dict for that row)
- Prior results (merged outputs from dependency steps)
- Field specs (prompts, types, formats, enums, examples, bad_examples, defaults)
- Model name
- Temperature
- System prompt (hashed)
- System prompt header (hashed)
- Grounding config (domains, location, max_searches)

For FunctionStep:
- Step name
- Row data
- Prior results
- `cache_version` (if set)

### What is NOT in the cache key

These can be changed without invalidating cached results:

- `sources_field` -- change the citation output field name freely.
- `provider_kwargs` -- iterate on thinking mode, effort level, etc.
- Hook configuration.
- `max_workers`, `max_retries`, and other runtime config.

### Per-step control

```python
# Default: caching follows the global config setting
LLMStep("analyze", fields={...}, cache=True)

# Disable caching for this step (useful for non-deterministic functions)
FunctionStep("stock_price", fn=get_price, fields=["price"], cache=False)

# Bump version to invalidate cache when function logic changes
FunctionStep("score", fn=score_v2, fields=["score"], cache_version="v2")
```

A step with `cache=False` always makes the API call (or runs the function), even when caching is enabled at the config level. A step with `cache=True` only caches when the global config has `enable_caching=True` (the default).

### Clear cache

```python
# Clear all cached results
pipeline.clear_cache()

# Clear cache for a specific step
pipeline.clear_cache(step="analyze")

# Specify a custom cache directory
pipeline.clear_cache(cache_dir="/path/to/.accrue")
```

## Checkpointing

### Enable checkpointing

```python
config = EnrichmentConfig(
    enable_checkpointing=True,
    auto_resume=True,            # Resume from last checkpoint on re-run (default: True)
)
result = pipeline.run(data, config=config)
```

### How it works

- After each step completes (across all rows), the full pipeline state is written to a JSON file -- per-row results *and* the row indices that errored.
- If the pipeline crashes mid-execution and is re-run, completed steps are skipped. Their results are loaded from the checkpoint and fed to downstream steps.
- Rows that *errored* inside a completed step are the exception: they re-run. Checkpointed means done only for cells that actually succeeded.
- Checkpoint files are cleaned up automatically once nothing is left to heal. A run that ends with row errors **keeps** its checkpoint, because that record is what a retry reads.

`pipeline.run()` and `pipeline.runner(config).run()` both checkpoint; they share the same file for the same dataset, so you can start with one and resume with the other.

### Retrying failed cells

A run that left errors can be healed without paying for the rows that worked:

```python
config = EnrichmentConfig(enable_checkpointing=True)

result = pipeline.run(data, config=config, run_log="logs/tonight.jsonl")
if result.has_errors:
    # Re-runs ONLY the (step, row) cells that errored. Everything that
    # succeeded is served from the checkpoint -- not one extra API call.
    result = pipeline.retry_failed(data, config=config, run_log="logs/tonight.jsonl")
```

`retry_failed_async()` is the `await` form. Both take the same arguments as `run()`, plus:

| Argument | Effect |
|----------|--------|
| `rows` | Restrict the retry to these row indices. Failures left out stay recorded in the checkpoint for a later retry. |
| `steps` | Restrict the retry to these step names. |
| `run_log` | Path of the failed run's log to append to. The retry keeps that run's `run_id` and `display_key` and appends a `retry_start` ... `retry_end` segment, so recovered cells arrive as ordinary `row_complete` records a dashboard can apply directly. |
| `data_identifier` | Explicit checkpoint identifier, if the original run passed one. |

This is the API behind a dashboard's "retry failed rows" button, and the groundwork for CLI resume.

Points worth knowing:

- **Pass the config the run used.** `retry_failed()` needs `enable_checkpointing=True` and the same `checkpoint_dir` to find the file. It raises `PipelineError` rather than silently re-running the whole dataset.
- **A clean run has nothing to retry.** Its checkpoint was removed on success, so `retry_failed()` raises.
- **Steps the run never finished still run in full.** A killed run has no results for them at all.
- **A cell that fails again surfaces normally.** It comes back in `result.errors` and stays recorded for the next attempt.
- **Downstream steps are not cascaded.** Healing `score` for row 7 does not re-run the `flag` step that consumed the old `score`. That is what keeps the retry exactly N calls. When a downstream step *also* errored on that row, it is a failed cell in its own right and gets retried too -- in DAG order, so it sees the healed value. To rebuild a downstream step wholesale, clear its cache and re-run instead.
- **Row indices are positional**, matching the `row` field in the run log.

### Configuration

```python
config = EnrichmentConfig(
    enable_checkpointing=True,
    checkpoint_dir="/tmp/my_checkpoints",  # Default: temp directory
    checkpoint_interval=100,                # Save partial progress every 100 rows (default: 0 = disabled)
    auto_resume=True,                       # Default: True
)
```

`checkpoint_interval` controls intra-step progress saving. When set to 100, the pipeline saves partial results every 100 rows within a single step. This is useful for long-running steps processing thousands of rows.

## Cache vs. checkpoint

| | Caching | Checkpointing |
|---|---------|---------------|
| **Purpose** | Skip redundant API calls | Crash recovery and failed-cell retry |
| **Granularity** | Per row, per step | Per step, plus the failed rows within it |
| **Persistence** | Permanent (until TTL or manual clear) | Temporary (cleaned up once nothing is left to heal) |
| **Storage** | SQLite (`.accrue/cache.db`) | JSON files |
| **When it helps** | Re-running with same/similar data | Pipeline crashes mid-execution |
| **Cost savings** | Yes (avoids duplicate API calls) | Yes (avoids re-running completed steps) |

### Using both together

For production workloads, enable both:

```python
config = EnrichmentConfig(
    enable_caching=True,       # Skip individual row/step combos already computed
    enable_checkpointing=True, # Resume from last completed step on crash
    checkpoint_interval=100,   # Save partial progress within steps
)
```

Or use the production preset:

```python
config = EnrichmentConfig.for_production()
# Sets: max_workers=30, enable_checkpointing=True, enable_caching=True,
#        checkpoint_interval=100, max_retries=5
```

Other presets:

```python
EnrichmentConfig.for_development()  # Low concurrency, caching on, debug logging
EnrichmentConfig.for_server()       # No progress bars, high concurrency
EnrichmentConfig.for_batch()        # Batch API settings with caching and checkpointing
```

## Gotchas

- Caching is on by default (`enable_caching=True`). To run without caching — typically for one-off runs or when you need a guaranteed-fresh result — pass `EnrichmentConfig(enable_caching=False)`. The `cache=True` default on individual steps only means "this step is cacheable"; it does not by itself enable or disable the cache system.
- `cache_version` is a FunctionStep feature. LLMStep cache keys are derived from prompts, model, temperature, and field specs, so they auto-invalidate when those change. There is no `cache_version` on LLMStep.
- Cache TTL is checked lazily on read. Expired entries are not proactively deleted. Call `CacheManager.cleanup_expired()` if you need to reclaim disk space.
- The cache directory (`.accrue/`) should be gitignored. Add `.accrue/` to your `.gitignore`.
- Checkpoint files are identified by a combination of data identifier and category. If you change the input data shape significantly between runs, the checkpoint may not match and will be skipped (with a warning).
- `checkpoint_interval=0` (default) means no intra-step saves. Progress is only saved after each step completes in full. Set it to a positive number for long-running steps.

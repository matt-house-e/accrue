# Caching and Checkpointing

Accrue provides two complementary persistence features: caching (skip redundant API calls) and checkpointing (crash recovery). They serve different purposes and can be used independently or together.

## Caching

### Enable caching

```python
from accrue import Pipeline, LLMStep, EnrichmentConfig

pipeline = Pipeline([
    LLMStep("analyze", fields={"summary": "Summarize the company"}, model="gpt-4.1-mini"),
])

config = EnrichmentConfig(enable_caching=True)
result = pipeline.run(data, config=config)
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

A step with `cache=False` always makes the API call (or runs the function), even when `enable_caching=True` in the config. A step with `cache=True` only caches if the global config also has `enable_caching=True`.

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

- After each step completes (across all rows), the full pipeline state is written to a JSON file.
- If the pipeline crashes mid-execution and is re-run, completed steps are skipped. Their results are loaded from the checkpoint and fed to downstream steps.
- Checkpoint files are cleaned up automatically after successful pipeline completion.

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
| **Purpose** | Skip redundant API calls | Crash recovery |
| **Granularity** | Per row, per step | Per step (all rows) |
| **Persistence** | Permanent (until TTL or manual clear) | Temporary (cleaned up on success) |
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

- Caching is off by default (`enable_caching=False`). You must opt in via config. The `cache=True` default on individual steps only means "this step is cacheable"; it does not enable the cache system.
- `cache_version` is a FunctionStep feature. LLMStep cache keys are derived from prompts, model, temperature, and field specs, so they auto-invalidate when those change. There is no `cache_version` on LLMStep.
- Cache TTL is checked lazily on read. Expired entries are not proactively deleted. Call `CacheManager.cleanup_expired()` if you need to reclaim disk space.
- The cache directory (`.accrue/`) should be gitignored. Add `.accrue/` to your `.gitignore`.
- Checkpoint files are identified by a combination of data identifier and category. If you change the input data shape significantly between runs, the checkpoint may not match and will be skipped (with a warning).
- `checkpoint_interval=0` (default) means no intra-step saves. Progress is only saved after each step completes in full. Set it to a positive number for long-running steps.

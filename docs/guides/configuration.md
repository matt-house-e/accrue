# Configuration

`EnrichmentConfig` controls every aspect of pipeline execution: concurrency, retries, caching, checkpointing, and batch settings. Pass it to `pipeline.run()`:

```python
from accrue import Pipeline, EnrichmentConfig

config = EnrichmentConfig(max_workers=20, enable_caching=True)
result = pipeline.run(data, config=config)
```

## All fields

| Field | Default | Description |
|-------|---------|-------------|
| `max_tokens` | `4000` | Max LLM output tokens |
| `temperature` | `0.2` | LLM temperature (0.1--0.3 best for enrichment) |
| `max_workers` | `10` | Concurrent rows per step |
| `overwrite_fields` | `False` | Overwrite existing field values in the DataFrame |
| `max_retries` | `3` | API error retry attempts (429, 500, timeouts) |
| `retry_base_delay` | `1.0` | Exponential backoff base in seconds |
| `on_error` | `"continue"` | `"continue"` collects errors; `"raise"` fails fast |
| `log_level` | `"INFO"` | Logging level |
| `enable_progress_bar` | `True` | Show tqdm progress bar |
| `enable_caching` | `False` | SQLite input-hash cache |
| `cache_ttl` | `3600` | Cache TTL in seconds |
| `cache_dir` | `".accrue"` | Directory for `cache.db` |
| `enable_checkpointing` | `False` | Step-level crash recovery |
| `checkpoint_dir` | `None` | Checkpoint directory (`None` = temp directory) |
| `auto_resume` | `True` | Auto-resume from checkpoint on re-run |
| `checkpoint_interval` | `0` | Save partial step progress every N rows (0 = disabled) |
| `batch_poll_interval` | `60.0` | Seconds between batch status checks |
| `batch_timeout` | `86400.0` | Max batch wait time (default 24h) |
| `batch_max_requests` | `50000` | Auto-chunk threshold for batch submissions |

## Presets

Four factory methods cover common scenarios:

### for_development()

Low concurrency, verbose logging, caching on. Safe for Tier 1 accounts.

```python
config = EnrichmentConfig.for_development()
# max_workers=5, log_level="DEBUG", enable_caching=True
```

### for_production()

High concurrency with caching and checkpointing. For Tier 2+ accounts.

```python
config = EnrichmentConfig.for_production()
# max_workers=30, enable_checkpointing=True, enable_caching=True,
# checkpoint_interval=100, max_retries=5
```

### for_server()

Async server context (FastAPI, etc.). No progress bars, high concurrency.

```python
config = EnrichmentConfig.for_server()
# max_workers=30, enable_progress_bar=False, max_retries=5, log_level="WARNING"
```

### for_batch()

Cost-optimized for large datasets using the Batch API.

```python
config = EnrichmentConfig.for_batch()
# max_workers=10, enable_caching=True, enable_checkpointing=True,
# batch_poll_interval=60.0, batch_timeout=86400.0, max_retries=5
```

## Temperature and max_tokens resolution

Both `temperature` and `max_tokens` can be set at the step level or the config level. Resolution order:

1. **Step-level** (`LLMStep(temperature=0.5)`) -- highest priority.
2. **Config-level** (`EnrichmentConfig(temperature=0.3)`) -- fallback.
3. **Default** (`0.2` for temperature, `4000` for max_tokens) -- final fallback.

```python
# Step-level takes precedence
LLMStep("creative",
    fields={"tagline": "Write a catchy tagline"},
    temperature=0.8,  # This wins over config.temperature
)
```

## Concurrency tuning

The `max_workers` setting controls how many rows are processed concurrently within each step (via `asyncio.Semaphore`). Set it based on your provider's rate limits.

**OpenAI GPT-4.1 nano/mini guidance:**

| Tier | Spend threshold | Recommended `max_workers` |
|------|----------------|--------------------------|
| Tier 1 | $5 | 5--10 |
| Tier 2 | $50 | 20--50 |
| Tier 3 | $100 | 50--100 |
| Tier 5 | $1,000 | 100--200 |

Setting `max_workers` too high for your tier results in 429 rate-limit errors. The retry logic handles these with exponential backoff, but throughput will be lower than using the right concurrency level.

## Caching

Enable caching to skip redundant API calls. The cache key is derived from the input row data, field specs, model, and temperature. Cached results are stored in a SQLite database at `{cache_dir}/cache.db`.

```python
config = EnrichmentConfig(
    enable_caching=True,
    cache_ttl=7200,       # 2 hours
    cache_dir=".accrue",  # Default
)
```

Note: `provider_kwargs` and `sources_field` are intentionally excluded from the cache key. Changing these between runs does not invalidate cached results.

## Checkpointing

Checkpointing saves completed step results so that a crashed pipeline can resume without re-running earlier steps.

```python
config = EnrichmentConfig(
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints",  # None = temp directory
    auto_resume=True,                # Resume automatically on re-run
    checkpoint_interval=100,         # Save every 100 rows within a step
)
```

When `auto_resume=True`, the pipeline detects an existing checkpoint on re-run and skips already-completed steps. Set `checkpoint_interval` to a positive integer to save partial progress within long-running steps.

## Validation

`EnrichmentConfig` validates all fields on construction:

- `temperature` must be between 0.0 and 2.0.
- `max_tokens`, `max_workers`, `batch_poll_interval`, `batch_timeout`, and `batch_max_requests` must be positive.
- `max_retries`, `cache_ttl`, and `checkpoint_interval` must be non-negative.

Invalid values raise `ValueError` immediately.

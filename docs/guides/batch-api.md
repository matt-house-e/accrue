# Batch API

The Batch API sends requests in bulk at 50% cost compared to realtime API calls. Use it for large enrichment jobs where latency is not critical.

Supported providers: **OpenAI** and **Anthropic**. Google is not yet supported.

## Minimal example

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze",
        fields={"market_size": "Estimate TAM"},
        batch=True,
    ),
])

result = pipeline.run(data)
```

Or use the `for_batch()` config preset, which also enables caching and checkpointing:

```python
from accrue import EnrichmentConfig

config = EnrichmentConfig.for_batch()
result = pipeline.run(data, config=config)
```

## How it works

1. **Cache check** -- only uncached rows are sent to the batch job.
2. **Build request payload** -- `build_messages()` is called for each uncached row.
3. **Auto-chunk** -- if the batch exceeds `batch_max_requests` (default 50,000), it is split into multiple submissions.
4. **Submit batch job** -- the provider's batch API receives the payload.
5. **Poll until complete** -- status is checked every `batch_poll_interval` seconds.
6. **Parse results and cache** -- responses are parsed with `parse_response()` and cached for future runs.
7. **Realtime fallback** -- any rows that fail in the batch are automatically retried via the realtime API.

## Configuration

```python
from accrue import EnrichmentConfig

config = EnrichmentConfig(
    batch_poll_interval=60.0,    # Seconds between status checks (default: 60)
    batch_timeout=86400.0,       # Max wait time in seconds (default: 24h)
    batch_max_requests=50000,    # Auto-chunk threshold (default: 50,000)
)
```

All three values must be positive. The defaults work well for most workloads.

## Custom batch-capable providers

Accrue exposes the batch protocol types from the top-level package so custom
provider adapters can integrate with the same execution path as the built-in
OpenAI and Anthropic adapters:

```python
from accrue import BatchCapableLLMClient, BatchRequest, BatchResult
```

- `BatchRequest` represents one row request in a batch submission. It carries
  the row `custom_id`, chat `messages`, model settings, optional structured
  output format, tool definitions, and `provider_kwargs`.
- `BatchResult` is returned after polling finishes. It maps each successful
  `custom_id` to an `LLMResponse`, records failed request IDs, and keeps the
  provider batch ID for debugging or dashboard lookup.
- `BatchCapableLLMClient` extends the regular `LLMClient` protocol with
  `submit_batch()`, `poll_batch()`, and `cancel_batch()`. A custom client that
  implements these methods can run `LLMStep(batch=True)` through the batch
  path; otherwise Accrue falls back to realtime calls.

## Monitoring batch runs

```python
from accrue import EnrichmentConfig

result = pipeline.run(data, config=EnrichmentConfig.for_batch())

usage = result.cost.steps["analyze"]
print(usage.execution_mode)  # "batch"
print(usage.batch_id)        # "batch_abc123"
print(usage.cache_hits)      # Rows served from cache
print(usage.cache_misses)    # Rows sent to batch
print(usage.cache_hit_rate)  # Float between 0.0 and 1.0
```

## Using with Anthropic

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze",
        fields={"market_size": "Estimate TAM"},
        model="claude-sonnet-4-20250514",
        batch=True,
    ),
])
```

The Anthropic provider is auto-detected from the `claude-` model prefix. Requires `pip install accrue[anthropic]`.

The Anthropic adapter uses the Message Batches API. System messages automatically get `cache_control` annotations for prompt caching savings.

## Constraints and gotchas

- **Incompatible with `grounding`.** Batch APIs do not support tool use (web search). Setting both `batch=True` and `grounding=True` raises a `PipelineError` at construction time.
- **Client must implement `BatchCapableLLMClient`.** The built-in OpenAI and Anthropic adapters do. Custom clients that only implement `LLMClient` will silently fall back to realtime execution.
- **Long-running.** Batches can take minutes to hours depending on provider load and batch size. The `batch_timeout` config (default 24 hours) controls the maximum wait.
- **Enable caching.** Without caching, re-running a pipeline resubmits all rows. The `for_batch()` preset enables caching by default.
- **Enable checkpointing.** If the process crashes mid-poll, checkpointing lets you resume without resubmitting completed steps. The `for_batch()` preset enables this too.
- **`provider_kwargs` are not included in the cache key.** Changing `provider_kwargs` between runs will not invalidate cached results.

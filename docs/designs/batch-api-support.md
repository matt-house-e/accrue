# Design Doc: Batch API Support (Epic #62)

> **Status**: IMPLEMENTED
> **Author**: Matt / Claude
> **Date**: March 2026
> **Related**: #62 (epic), #42 (intra-batch dedup), #43 (chunked execution)

## 1. Problem

Accrue processes rows concurrently via async tasks bounded by a semaphore. Every row is a separate API call. This works, but:

- **Cost**: All three providers (OpenAI, Anthropic, Google) offer Batch APIs at **50% off**.
- **Rate limits**: Batch APIs use separate, higher pools — no 429s.
- **Scale**: At 10K+ rows, even generous rate limits become the bottleneck.

Accrue's column-oriented execution model is a natural fit: each step already runs across ALL rows before the next step starts. That's exactly what a batch job is — "here are N requests, process them all, give me the results."

## 2. Goal

Add an `execution_mode` to Accrue that lets users opt into provider Batch APIs with a single config change, while preserving every existing behavior (caching, checkpointing, structured outputs, hooks, conditional predicates, grounding fallback).

```python
# One-line change for 50% cost savings:
result = pipeline.run(data, config=EnrichmentConfig(execution_mode="batch"))
```

## 3. Non-Goals

- **Streaming batch results** — Batch APIs are fire-and-wait. No partial streaming.
- **Async batch submission** (submit and return immediately) — v1 is synchronous: submit, poll, return. Fire-and-forget is a future enhancement.
- **Custom batch backends** — Only the three built-in providers. Custom `LLMClient` falls back to realtime.
- **Intra-batch deduplication** — Tracked separately in #42. Natural complement, can layer on later.

## 4. Execution Modes

| Mode | Behavior |
|------|----------|
| `"realtime"` | Current behavior. One async API call per row. **(default)** |
| `"batch"` | All eligible LLMSteps use provider Batch API. Ineligible steps fall back to realtime. |
| `"auto"` | Batch for steps that support it (no tools/grounding), realtime otherwise. |

### Per-Step Override

```python
LLMStep("classify", fields={...}, batch=False)  # Always realtime, even in batch mode
LLMStep("analyze", fields={...}, batch=True)    # Always batch, even in realtime mode
```

Per-step `batch` flag takes precedence over `execution_mode`.

### Automatic Fallbacks

These always run realtime regardless of config:

| Condition | Reason |
|-----------|--------|
| `FunctionStep` | Not an LLM call |
| `grounding=True` | Batch APIs don't support tool use |
| `tools` parameter | Same — no tool use in batch |
| Custom `LLMClient` (not built-in) | No `complete_batch()` method |
| `base_url` set on OpenAI | Third-party providers don't have batch APIs |

## 5. `LLMClient` Protocol Extension

### Option A: New method with default fallback (RECOMMENDED)

```python
@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, ...) -> LLMResponse: ...

    # New — default implementation falls back to concurrent complete() calls
    async def complete_batch(
        self,
        requests: list[BatchRequest],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        poll_interval: float = 30.0,
        timeout: float = 86400.0,
    ) -> BatchResult: ...
```

### Why not a separate `BatchClient` protocol?

A separate protocol would require isinstance checks and two code paths in `_execute_step`. Keeping it on `LLMClient` with a sensible default (fan-out to `complete()`) means any existing client "works" — it just doesn't get the batch discount. Built-in adapters override with real batch implementations.

### Data Types

```python
@dataclass
class BatchRequest:
    """One request within a batch submission."""
    custom_id: str              # Unique ID to correlate results (e.g. "row-42")
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None = None

@dataclass
class BatchResult:
    """Aggregated result from a batch job."""
    responses: dict[str, LLMResponse]  # custom_id → response
    failed_ids: list[str]              # custom_ids that failed
    batch_id: str                      # Provider batch job ID (for debugging)
    errors: dict[str, str]             # custom_id → error message (for failed)
```

### Default Fallback Implementation

The base module provides a free function that any client can delegate to:

```python
async def _default_complete_batch(client, requests, model, temperature, max_tokens, response_format, **kwargs):
    """Fan-out to concurrent complete() calls. No batch discount, but functionally correct."""
    results = {}
    failed = []
    errors = {}
    for req in requests:
        try:
            resp = await client.complete(req.messages, model, temperature, max_tokens, response_format)
            results[req.custom_id] = resp
        except Exception as e:
            failed.append(req.custom_id)
            errors[req.custom_id] = str(e)
    return BatchResult(responses=results, failed_ids=failed, batch_id="realtime-fallback", errors=errors)
```

This preserves backward compatibility: custom `LLMClient` implementations that don't have `complete_batch()` still work.

## 6. Provider Adapter Implementations

### 6.1 OpenAI (`OpenAIClient.complete_batch`)

OpenAI Batch API flow:
1. Build JSONL file from `BatchRequest` list (one JSON object per line)
2. Upload JSONL via `client.files.create(file=..., purpose="batch")`
3. Create batch: `client.batches.create(input_file_id=..., endpoint="/v1/chat/completions", completion_window="24h")`
4. Poll `client.batches.retrieve(batch_id)` until status is `completed`, `failed`, or `expired`
5. Download output file: `client.files.content(output_file_id)`
6. Parse JSONL output → map `custom_id` back to `LLMResponse`

**Notes:**
- Uses Chat Completions endpoint (not Responses API) because the Batch API only supports `/v1/chat/completions` and `/v1/responses`
- For native OpenAI (no `base_url`): use `/v1/responses` endpoint
- `response_format` (structured outputs) supported in batch
- Max 50,000 requests per batch; auto-chunk if exceeded
- Not available for `base_url` providers — fallback to realtime

### 6.2 Anthropic (`AnthropicClient.complete_batch`)

Anthropic Message Batches API flow:
1. Build `requests` list: `[{"custom_id": "row-0", "params": {"model": ..., "messages": ..., ...}}]`
2. Submit: `client.messages.batches.create(requests=requests)`
3. Poll `client.messages.batches.retrieve(batch_id)` until `processing_status == "ended"`
4. Stream results: `client.messages.batches.results(batch_id)` — yields per-request results
5. Map `custom_id` → parse message content → `LLMResponse`

**Notes:**
- Max 100K requests or 256MB per batch
- `output_config.format` (structured outputs) supported in batch
- Status polling: check `processing_status` field

### 6.3 Google (`GoogleClient.complete_batch`)

Google Gemini Batch API flow:
1. Build inline requests: list of `GenerateContentRequest` with metadata
2. Submit: `client.batches.create(model=..., requests=..., config=...)`
3. Poll `client.batches.get(name=batch.name)` until `state == "JOB_STATE_SUCCEEDED"`
4. Read results from `batch.dest_uri` (GCS) or inline response
5. Map by index → `LLMResponse`

**Notes:**
- Supports `response_json_schema` in batch
- May require GCS bucket for large batches — investigate inline limit
- Newer API, less battle-tested than OpenAI/Anthropic

## 7. Execution Path in `_execute_step()`

### Current Flow (realtime)

```
for each row (concurrent async tasks):
    evaluate run_if/skip_if → skip?
    cache check → hit? return cached
    build StepContext → step.run(ctx) → validate → cache store
    fire on_row_complete hook
```

### New Flow (batch)

```
Phase 1: Classify rows
    for each row:
        evaluate run_if/skip_if → skip? → add to skipped[]
        cache check → hit? → add to cached[]
        else → add to uncached[]

Phase 2: Build and submit batch
    for each uncached row:
        build messages (same as realtime: system prompt + user message)
        create BatchRequest(custom_id=f"row-{idx}", messages=messages)
    batch_result = await client.complete_batch(requests, ...)

Phase 3: Process results
    for each response in batch_result.responses:
        parse JSON → validate with Pydantic → apply defaults
        cache store
        fire on_row_complete hook
    for each failed_id in batch_result.failed_ids:
        create RowError
        fire on_row_complete hook (with error)

Phase 4: Merge
    merge skipped + cached + fresh results → step_values[step.name]
```

### Key Difference from Realtime

In realtime mode, `step.run(ctx)` handles the full LLM call + retry + parse loop. In batch mode, we need to **decompose** that:

1. **Message building** — extracted from `LLMStep.run()` into a reusable method: `LLMStep.build_messages(ctx) -> list[dict]`
2. **Response parsing** — extracted into: `LLMStep.parse_response(content: str, ctx: StepContext) -> StepResult`
3. **Batch submission** — handled by `_execute_step()` calling `client.complete_batch()`

This decomposition is the main refactoring needed. The `LLMStep.run()` method currently interleaves message building, API calling, and parsing in nested retry loops. We need to pull apart the message-building and result-parsing phases so they can be used independently of the API call.

### Retry Strategy for Failed Batch Rows

Failed rows from a batch get collected into a retry list. Configurable behavior:

```python
EnrichmentConfig(
    execution_mode="batch",
    batch_retry="realtime",  # "realtime" | "batch" | "none"
)
```

| Strategy | Behavior |
|----------|----------|
| `"realtime"` | Failed rows retried as individual async calls (default — fast, small set) |
| `"batch"` | Failed rows submitted as a follow-up batch (cheaper, slower) |
| `"none"` | Failed rows collected as `RowError`, no retry |

## 8. Config Changes

### `EnrichmentConfig` additions

```python
@dataclass
class EnrichmentConfig:
    # ... existing fields ...

    # === Batch Execution ===
    execution_mode: str = "realtime"
    """Execution mode: 'realtime' (default), 'batch', or 'auto'."""

    batch_poll_interval: float = 30.0
    """Seconds between batch status checks."""

    batch_timeout: float = 86400.0
    """Maximum seconds to wait for a batch to complete (default: 24h)."""

    batch_max_requests: int = 0
    """Max requests per batch submission. 0 = use provider default."""

    batch_retry: str = "realtime"
    """Retry strategy for failed batch rows: 'realtime', 'batch', or 'none'."""
```

### Presets

```python
@classmethod
def for_batch(cls) -> "EnrichmentConfig":
    """Batch execution with caching. For cost-optimized large datasets."""
    return cls(
        execution_mode="batch",
        enable_caching=True,
        enable_checkpointing=True,
        checkpoint_interval=0,  # Not needed — batch is atomic per step
        batch_poll_interval=30.0,
        batch_timeout=86400.0,
        batch_retry="realtime",
    )
```

### Per-Step Flag

`LLMStep` and `FunctionStep` gain an optional `batch: bool | None = None`:

- `None` — defer to `execution_mode` (default)
- `True` — force batch execution for this step
- `False` — force realtime execution for this step

`FunctionStep` ignores `batch=True` (always realtime). Logged as a warning.

## 9. Observability

### StepUsage Changes

```python
class StepUsage(BaseModel):
    # ... existing fields ...
    execution_mode: str = "realtime"  # "realtime" | "batch"
    batch_id: str | None = None       # Provider batch job ID
```

### Hook Events

Reuse existing hooks with additional metadata rather than adding new hook types:

- `StepStartEvent` — add `execution_mode: str` field
- `StepEndEvent` — add `execution_mode: str`, `batch_id: str | None` fields
- `RowCompleteEvent` — no change (fires after batch results are parsed, same as realtime)

**Rationale**: Adding `on_batch_submitted` / `on_batch_progress` / `on_batch_complete` hooks would be nice but adds API surface for a niche feature. Users who need progress can check logs. Defer new hooks to v2 if there's demand.

### Logging

```
INFO  Step 'analyze' submitting batch: 150 rows (45 cached, 3 skipped, 102 uncached)
INFO  Step 'analyze' batch submitted: batch_abc123 (OpenAI)
INFO  Step 'analyze' batch progress: 67% complete (45s elapsed)
INFO  Step 'analyze' batch complete: 100/102 succeeded, 2 failed (batch_abc123, 78s)
INFO  Step 'analyze' retrying 2 failed rows via realtime
```

## 10. LLMStep Refactoring

The key refactoring is decomposing `LLMStep.run()` into composable pieces:

### New Methods

```python
class LLMStep:
    def build_messages(self, ctx: StepContext) -> list[dict[str, str]]:
        """Build the messages array for an LLM call. Extracted from run()."""
        system_content = self._build_system_message(ctx)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Analyze the provided data and return the requested fields as JSON."},
        ]

    def parse_response(self, content: str) -> dict[str, Any]:
        """Parse and validate an LLM response string. Extracted from run()."""
        parsed = json.loads(content)
        if self._use_structured_outputs:
            dynamic_model = build_response_model(self._field_specs)
            validated = dynamic_model.model_validate(parsed)
        else:
            validated = self.schema.model_validate(parsed)
        values = {k: v for k, v in validated.model_dump().items() if k in self.fields}
        return self._apply_defaults(values)

    async def run(self, ctx: StepContext) -> StepResult:
        """Unchanged — still the primary per-row entry point for realtime execution."""
        ...
```

`run()` stays as-is for backward compatibility and the realtime path. `build_messages()` and `parse_response()` are used by the batch execution path in `_execute_step()`.

### Parse Retry in Batch Mode

Batch APIs don't support conversational retry (you can't feed an error back to the LLM mid-batch). Two options:

1. **No parse retry in batch** — if JSON parsing fails, treat as a failed row. Rely on structured outputs to prevent this. *(RECOMMENDED for v1)*
2. **Realtime retry for parse failures** — failed parses get retried via the realtime `step.run()` path.

With structured outputs enabled (which it is by default for all three providers), parse failures should be extremely rare. Option 1 keeps things simple.

## 11. Interaction with Existing Features

| Feature | Interaction |
|---------|------------|
| **Cache** | Cache check happens BEFORE batch submission. Only uncached rows enter the batch. Cache store happens AFTER batch results arrive. Same cache keys — switching modes doesn't invalidate. |
| **Checkpoint** | Checkpoint saves after each batch step completes (same as realtime). `checkpoint_interval` is less relevant — batch is atomic per step. |
| **Structured outputs** | Supported by all three providers in batch mode. Same auto-detection logic. |
| **Grounding** | Auto-fallback to realtime. Batch APIs don't support tool use. |
| **run_if/skip_if** | Evaluated BEFORE batch submission. Skipped rows never enter the batch. |
| **depends_on** | Unchanged. Batch step 1 completes → results feed into batch step 2. DAG ordering preserved. |
| **Hooks** | `on_row_complete` fires after batch results are parsed (not during batch processing). `on_step_start/end` fire as normal. |
| **max_retries** | API-level retries apply to the batch submission call itself (e.g., 429 on submit). Parse retries don't apply (no conversational retry in batch). |
| **Error handling** | `on_error="continue"` collects failed batch rows. `on_error="raise"` raises on first failure after batch completes. |

## 12. Suggested Implementation Order

### Phase 1: Foundation (1 PR)
1. Add `BatchRequest`, `BatchResult` data types to `accrue/steps/providers/base.py`
2. Add default `complete_batch()` fallback
3. Add `execution_mode`, `batch_poll_interval`, `batch_timeout`, `batch_retry` to `EnrichmentConfig`
4. Add `batch` parameter to `LLMStep` and `FunctionStep`
5. Extract `build_messages()` and `parse_response()` from `LLMStep.run()`
6. Tests for all config validation and step construction

### Phase 2: OpenAI Adapter (1 PR)
1. Implement `OpenAIClient.complete_batch()` — JSONL upload, create, poll, download, parse
2. Integration tests with mocked OpenAI batch API
3. Auto-chunking for >50K requests

### Phase 3: Batch Execution Path (1 PR)
1. Batch routing logic in `_execute_step()` (classify rows → submit → parse → merge)
2. Wire up cache-aware batching
3. Wire up retry strategy
4. `StepUsage.execution_mode` and `batch_id`
5. End-to-end tests: single-step batch, multi-step batch, mixed pipeline

### Phase 4: Anthropic + Google Adapters (1 PR)
1. `AnthropicClient.complete_batch()`
2. `GoogleClient.complete_batch()`
3. Integration tests for each

### Phase 5: Polish (1 PR)
1. `EnrichmentConfig.for_batch()` preset
2. Hook event extensions (`execution_mode` on StepStartEvent/StepEndEvent)
3. Logging (batch progress, job IDs)
4. Docs: README section, PIPELINE_DESIGN.md update, ADR 014
5. Example: `examples/batch_enrichment.py`

## 13. Open Questions

1. **Google Batch API maturity** — The Gemini Batch API is newer and less documented than OpenAI/Anthropic. Spike needed to validate inline request support and confirm no GCS requirement for small batches. *Consider deferring Google to Phase 5 if unstable.*

2. **Batch size auto-chunking** — OpenAI limits 50K requests/batch. Should we auto-chunk transparently, or surface an error and let users configure `batch_max_requests`? *Recommendation: auto-chunk with a log message. Users don't want to think about provider limits.*

3. **Timeout behavior** — If a batch times out after 24h, what happens? *Recommendation: raise `StepError` with the batch ID so users can check the provider dashboard. Don't silently swallow.*

4. **Cost tracking** — Batch pricing is different (50% off). Should `CostSummary` reflect this? *Recommendation: No. Accrue tracks tokens, not dollars. The pricing is the user's concern. But `execution_mode` on `StepUsage` tells them it was a batch call.*

5. **Cancel on KeyboardInterrupt** — If the user Ctrl-C's during a batch poll, should we cancel the batch job? *Recommendation: Yes, attempt `client.batches.cancel()` in a finally block. Log the batch ID either way so the user can manage it.*

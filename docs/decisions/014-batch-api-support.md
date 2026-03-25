# ADR 014: Batch API Support

**Status**: Accepted
**Date**: March 2026
**Related**: Epic #62, ADR 004 (LLMClient protocol)

## Context

All three major LLM providers offer Batch APIs at 50% cost with higher rate limits and 24-hour turnaround (often under 1 hour). Accrue's column-oriented execution model — each step runs across all rows before the next step starts — maps directly to batch job submission.

## Decision

Add batch execution support via:

1. **`BatchCapableLLMClient` protocol** — extends `LLMClient` with `submit_batch()`, `poll_batch()`, `cancel_batch()`. Separate protocol (not added to `LLMClient`) to avoid breaking existing custom clients.

2. **`LLMStep(batch=True)`** — per-step opt-in. Steps with `batch=True` and a `BatchCapableLLMClient` use the batch execution path. Mutually exclusive with `grounding` (batch APIs don't support tool use).

3. **Batch execution path in `_execute_step_batch()`** — classify rows (skipped/cached/uncached) → build batch requests → auto-chunk → submit → poll → parse → cache → realtime fallback for failures.

4. **OpenAI and Anthropic adapters** — both implement `BatchCapableLLMClient`. Google deferred (API less mature).

## Key Design Choices

- **Separate protocol vs extending LLMClient**: Adding methods to `LLMClient` would break `@runtime_checkable` for existing custom clients. `isinstance(client, BatchCapableLLMClient)` determines eligibility.

- **`build_messages()` / `parse_response()` extraction**: Decomposed from `LLMStep.run()` so the batch path can build all messages upfront and parse responses after batch completes. `run()` refactored to use these methods internally (pure refactor, no behavior change).

- **Cache keys identical for batch and realtime**: Same row + same config = same cache key. Switching modes doesn't invalidate cache.

- **Realtime fallback for failed batch rows**: Rows that fail in batch are retried via the normal `step.run()` path with its full retry logic. Batch mode is strictly better — never worse results.

- **No parse retry in batch**: Batch APIs don't support conversational retry. With structured outputs enabled (default), parse failures are extremely rare. Failed parses fall back to realtime.

- **Auto-chunking**: Batches exceeding `batch_max_requests` (default 50K) are automatically split. Transparent to the user.

- **KeyboardInterrupt cancels batches**: `cancel_batch()` called in exception handler during polling.

## Consequences

- Adds ~300 lines to pipeline.py (`_execute_step_batch`), ~150 lines each to OpenAI/Anthropic adapters
- New config fields: `batch_poll_interval`, `batch_timeout`, `batch_max_requests`
- New preset: `EnrichmentConfig.for_batch()`
- `StepUsage` gains `execution_mode` and `batch_id` fields
- `StepStartEvent` and `StepEndEvent` gain `execution_mode` field
- Google batch support deferred to future work

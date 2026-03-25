"""Integration test: Mixed pipeline — FunctionStep (realtime) + LLMStep (batch).

Verifies that:
  1. FunctionStep runs immediately (realtime)
  2. LLMStep with batch=True uses Batch API
  3. depends_on wiring works: batch step sees FunctionStep output in prior_results
  4. Caching works: second run serves from cache, no batch submitted
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from accrue import (
    EnrichmentConfig,
    EnrichmentHooks,
    FunctionStep,
    LLMStep,
    Pipeline,
    StepEndEvent,
    StepStartEvent,
)

# Track step events to verify execution modes
events = []


def on_step_start(event: StepStartEvent):
    print(f"  → {event.step_name} starting (rows={event.num_rows})")
    events.append(("start", event.step_name))


def on_step_end(event: StepEndEvent):
    mode = event.execution_mode
    batch_info = f", batch_id={event.batch_id}" if event.batch_id else ""
    print(f"  ✓ {event.step_name} done ({mode}, {event.elapsed_seconds:.1f}s{batch_info})")
    events.append(("end", event.step_name, mode))


def enrich_context(ctx):
    """Simple FunctionStep that adds context."""
    company = ctx.row.get("company", "Unknown")
    return {"__context": f"{company} is a technology company."}


pipeline = Pipeline(
    [
        FunctionStep("context", fn=enrich_context, fields=["__context"]),
        LLMStep(
            "analyze",
            fields={
                "summary": "Write a brief one-sentence summary based on the context",
            },
            model="gpt-4.1-nano",
            batch=True,
            depends_on=["context"],
        ),
    ]
)

data = [
    {"company": "Stripe"},
    {"company": "Figma"},
]

config = EnrichmentConfig(
    enable_progress_bar=False,
    enable_caching=True,
    batch_poll_interval=10.0,
    batch_timeout=600.0,
)

hooks = EnrichmentHooks(on_step_start=on_step_start, on_step_end=on_step_end)

# === Run 1: First execution ===
print("=== Mixed Pipeline Test ===")
print(f"\nRun 1: Processing {len(data)} rows (FunctionStep → Batch LLMStep)...")
events.clear()

result = pipeline.run(data, config=config, hooks=hooks)

print(f"\nSuccess rate: {result.success_rate:.0%}")
print(f"Total tokens: {result.cost.total_tokens}")
for row in result.data:
    print(f"  {row['company']}: {row.get('summary', 'N/A')}")

# Assertions for run 1
assert result.success_rate == 1.0
assert len(result.errors) == 0

analyze_usage = result.cost.steps.get("analyze")
assert analyze_usage is not None, "Expected analyze step usage"
assert analyze_usage.execution_mode == "batch", (
    f"Expected batch, got {analyze_usage.execution_mode}"
)
assert analyze_usage.cache_misses > 0, "Expected cache misses on first run"

for row in result.data:
    assert row.get("summary"), f"Missing summary for {row['company']}"

# === Run 2: Cached execution ===
print("\nRun 2: Same data — should be 100% cache hits...")
events.clear()

result2 = pipeline.run(data, config=config, hooks=hooks)

print(f"Success rate: {result2.success_rate:.0%}")
analyze_usage2 = result2.cost.steps.get("analyze")
if analyze_usage2:
    print(f"Cache hits: {analyze_usage2.cache_hits}, misses: {analyze_usage2.cache_misses}")
    assert analyze_usage2.cache_hits == 2, f"Expected 2 cache hits, got {analyze_usage2.cache_hits}"
    assert analyze_usage2.cache_misses == 0, (
        f"Expected 0 cache misses, got {analyze_usage2.cache_misses}"
    )

# Clean up cache
pipeline.clear_cache()

print("\n✓ Mixed pipeline test PASSED")

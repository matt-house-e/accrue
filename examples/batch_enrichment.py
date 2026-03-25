"""Batch enrichment example — 50% cost savings with provider Batch APIs.

This example demonstrates:
  1. Basic batch execution with LLMStep(batch=True)
  2. Cache-aware batching (only uncached rows hit the API)
  3. Mixed pipelines: FunctionStep (realtime) → LLMStep (batch)
  4. Monitoring progress with hooks

Requirements:
  pip install accrue
  export OPENAI_API_KEY=sk-...

Usage:
  python examples/batch_enrichment.py
"""

from accrue import (
    EnrichmentConfig,
    EnrichmentHooks,
    FunctionStep,
    LLMStep,
    Pipeline,
    StepEndEvent,
    StepStartEvent,
)

# ---------------------------------------------------------------------------
# Example 1: Basic batch enrichment
# ---------------------------------------------------------------------------

pipeline = Pipeline(
    [
        LLMStep(
            "analyze",
            fields={
                "market_size": "Estimate the total addressable market in billions USD",
                "competition": {
                    "prompt": "Rate competition level",
                    "enum": ["Low", "Medium", "High"],
                },
            },
            batch=True,  # Use provider Batch API (50% off)
        )
    ]
)

data = [
    {"company": "Stripe", "industry": "Fintech"},
    {"company": "Figma", "industry": "Design Tools"},
    {"company": "Notion", "industry": "Productivity"},
    {"company": "Linear", "industry": "Dev Tools"},
    {"company": "Vercel", "industry": "Cloud Infrastructure"},
]

# Use the batch preset: caching + checkpointing enabled
config = EnrichmentConfig.for_batch()

print("Example 1: Basic batch enrichment")
print(f"  Submitting {len(data)} rows via Batch API...")
result = pipeline.run(data, config=config)
print(f"  Done! {result.success_rate:.0%} success rate")
for row in result.data:
    print(f"  {row['company']}: TAM={row.get('market_size')}, Competition={row.get('competition')}")

# ---------------------------------------------------------------------------
# Example 2: Mixed pipeline with hooks
# ---------------------------------------------------------------------------

print("\nExample 2: Mixed pipeline with progress hooks")


def on_step_start(event: StepStartEvent):
    mode = "BATCH" if event.execution_mode == "batch" else "realtime"
    print(f"  → Step '{event.step_name}' starting ({mode}, {event.num_rows} rows)")


def on_step_end(event: StepEndEvent):
    mode = "BATCH" if event.execution_mode == "batch" else "realtime"
    batch_info = f", batch_id={event.batch_id}" if event.batch_id else ""
    print(f"  ✓ Step '{event.step_name}' done ({mode}, {event.elapsed_seconds:.1f}s{batch_info})")


def enrich_company(ctx):
    """FunctionStep: add context from a hypothetical internal API."""
    return {"__context": f"Internal data for {ctx.row['company']}"}


mixed_pipeline = Pipeline(
    [
        FunctionStep("lookup", fn=enrich_company, fields=["__context"]),
        LLMStep(
            "analyze",
            fields={
                "growth_potential": "Rate growth potential based on context",
                "investment_thesis": "Write a one-sentence investment thesis",
            },
            depends_on=["lookup"],
            batch=True,
        ),
    ]
)

hooks = EnrichmentHooks(on_step_start=on_step_start, on_step_end=on_step_end)
result = mixed_pipeline.run(data, config=config, hooks=hooks)

for row in result.data:
    print(f"  {row['company']}: {row.get('investment_thesis', 'N/A')}")

print(f"\nTotal tokens: {result.cost.total_tokens}")

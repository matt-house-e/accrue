"""Integration test: OpenAI Batch API enrichment.

Submits a small batch via the OpenAI Batch API, polls until complete,
and verifies results are parsed correctly.

NOTE: Batch API can take 1-60 minutes. This test uses a small dataset
and aggressive polling to keep it short.
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from accrue import EnrichmentConfig, LLMStep, Pipeline

pipeline = Pipeline(
    [
        LLMStep(
            "classify",
            fields={
                "category": {
                    "prompt": "Classify this company's primary industry sector",
                    "enum": ["Fintech", "SaaS", "E-commerce", "Healthcare", "Other"],
                },
                "one_liner": {
                    "prompt": "Write a one-sentence description of what this company does",
                    "type": "String",
                },
            },
            model="gpt-4.1-nano",
            batch=True,  # <-- USE BATCH API
        )
    ]
)

data = [
    {"company": "Stripe", "description": "Online payment processing platform"},
    {"company": "Notion", "description": "All-in-one workspace for notes and projects"},
    {"company": "Vercel", "description": "Cloud platform for frontend deployment"},
]

config = EnrichmentConfig(
    enable_progress_bar=False,
    batch_poll_interval=10.0,  # Poll every 10 seconds (batch can be fast)
    batch_timeout=600.0,  # 10 min timeout for this test
)

print("=== OpenAI Batch API Test ===")
print(f"Submitting {len(data)} rows via Batch API...")
print("(This may take 1-5 minutes for the batch to process)\n")

result = pipeline.run(data, config=config)

print(f"Success rate: {result.success_rate:.0%}")
print(f"Errors: {len(result.errors)}")
print(f"Total tokens: {result.cost.total_tokens}")

step_usage = result.cost.steps.get("classify")
if step_usage:
    print(f"Execution mode: {step_usage.execution_mode}")
    print(f"Batch ID: {step_usage.batch_id}")
    print(f"Cache hits: {step_usage.cache_hits}, misses: {step_usage.cache_misses}")

print()
for row in result.data:
    print(f"  {row['company']}: category={row.get('category')}")
    print(f"    {row.get('one_liner', 'N/A')}")

# Assertions
assert result.success_rate == 1.0, f"Expected 100% success, got {result.success_rate}"
assert len(result.errors) == 0, f"Expected 0 errors, got {len(result.errors)}"
assert result.cost.total_tokens > 0, "Expected token usage"

if step_usage:
    assert step_usage.execution_mode == "batch", (
        f"Expected batch execution mode, got {step_usage.execution_mode}"
    )
    assert step_usage.batch_id, "Expected batch_id to be set"

for row in result.data:
    assert row.get("category") in ["Fintech", "SaaS", "E-commerce", "Healthcare", "Other"], (
        f"Unexpected category: {row.get('category')}"
    )
    assert row.get("one_liner"), "Missing one_liner"

print("\n✓ OpenAI Batch API test PASSED")

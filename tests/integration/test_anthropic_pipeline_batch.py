"""Integration test: Full pipeline with Anthropic batch execution.

Tests Pipeline.run() with an AnthropicClient and batch=True.
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from accrue import EnrichmentConfig, LLMStep, Pipeline
from accrue.steps.providers.anthropic import AnthropicClient

client = AnthropicClient()

pipeline = Pipeline(
    [
        LLMStep(
            "classify",
            fields={
                "category": {
                    "prompt": "Classify this company's primary sector",
                    "enum": ["Fintech", "SaaS", "E-commerce", "Healthcare", "Other"],
                },
            },
            model="claude-haiku-4-5-20251001",
            client=client,
            batch=True,
        )
    ]
)

data = [
    {"company": "Stripe", "description": "Online payment processing"},
    {"company": "Notion", "description": "All-in-one workspace app"},
]

config = EnrichmentConfig(
    enable_progress_bar=False,
    batch_poll_interval=10.0,
    batch_timeout=600.0,
)

print("=== Anthropic Pipeline Batch Test ===")
print(f"Submitting {len(data)} rows via Anthropic Batch API...")
print("(Polling every 10s, timeout 10min)\n")

result = pipeline.run(data, config=config)

print(f"Success rate: {result.success_rate:.0%}")
print(f"Errors: {len(result.errors)}")
print(f"Total tokens: {result.cost.total_tokens}")

step_usage = result.cost.steps.get("classify")
if step_usage:
    print(f"Execution mode: {step_usage.execution_mode}")
    print(f"Batch ID: {step_usage.batch_id}")

for row in result.data:
    print(f"  {row['company']}: {row.get('category')}")

assert result.success_rate == 1.0
assert step_usage.execution_mode == "batch"
assert step_usage.batch_id is not None

for row in result.data:
    assert row.get("category") in ["Fintech", "SaaS", "E-commerce", "Healthcare", "Other"]

print("\n✓ Anthropic pipeline batch test PASSED")

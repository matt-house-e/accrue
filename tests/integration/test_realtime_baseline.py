"""Integration test: realtime (non-batch) enrichment baseline.

Verifies the existing realtime path still works end-to-end with a real API call.
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
                "employee_estimate": {
                    "prompt": "Estimate number of employees (rough order of magnitude)",
                    "type": "String",
                },
            },
            model="gpt-4.1-nano",
        )
    ]
)

data = [
    {"company": "Stripe", "description": "Online payment processing platform"},
    {"company": "Shopify", "description": "E-commerce platform for online stores"},
]

print("=== Realtime Baseline Test ===")
print(f"Processing {len(data)} rows via realtime API...")

result = pipeline.run(data, config=EnrichmentConfig(enable_progress_bar=False))

print(f"Success rate: {result.success_rate:.0%}")
print(f"Errors: {len(result.errors)}")
print(f"Total tokens: {result.cost.total_tokens}")

for row in result.data:
    print(
        f"  {row['company']}: category={row.get('category')}, employees={row.get('employee_estimate')}"
    )

# Assertions
assert result.success_rate == 1.0, f"Expected 100% success, got {result.success_rate}"
assert len(result.errors) == 0, f"Expected 0 errors, got {len(result.errors)}"
assert result.cost.total_tokens > 0, "Expected token usage"

for row in result.data:
    assert row.get("category") in ["Fintech", "SaaS", "E-commerce", "Healthcare", "Other"], (
        f"Unexpected category: {row.get('category')}"
    )
    assert row.get("employee_estimate") is not None, "Missing employee_estimate"

print("\n✓ Realtime baseline PASSED")

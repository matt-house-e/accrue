"""End-to-end integration test: Submit, poll, download, parse a real batch.

Uses a single tiny request to minimize wait time. Polls aggressively.
Typically completes in 2-15 minutes.
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from accrue import EnrichmentConfig, LLMStep, Pipeline

pipeline = Pipeline(
    [
        LLMStep(
            "test",
            fields={
                "greeting": "Say hello to this person by name",
            },
            model="gpt-4.1-nano",
            batch=True,
        )
    ]
)

data = [{"person": "Alice"}]

config = EnrichmentConfig(
    enable_progress_bar=False,
    batch_poll_interval=15.0,  # Check every 15 seconds
    batch_timeout=1800.0,  # 30 minute timeout
)

print("=== End-to-End Batch Test ===")
print("Submitting 1 row via Batch API...")
print("(Polling every 15s, timeout 30min)\n")

result = pipeline.run(data, config=config)

print(f"Success rate: {result.success_rate:.0%}")
print(f"Errors: {len(result.errors)}")
print(f"Total tokens: {result.cost.total_tokens}")

step_usage = result.cost.steps.get("test")
if step_usage:
    print(f"Execution mode: {step_usage.execution_mode}")
    print(f"Batch ID: {step_usage.batch_id}")

for row in result.data:
    print(f"  Result: {row.get('greeting')}")

assert result.success_rate == 1.0, f"Expected 100% success, got {result.success_rate}"
assert result.cost.total_tokens > 0, "Expected token usage"
assert step_usage.execution_mode == "batch"
assert step_usage.batch_id is not None

greeting = result.data[0].get("greeting", "")
assert "Alice" in greeting or "alice" in greeting.lower(), (
    f"Expected greeting to mention Alice: {greeting}"
)

print("\n✓ End-to-end batch test PASSED")

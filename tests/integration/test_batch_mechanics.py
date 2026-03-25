"""Integration test: Batch mechanics without waiting for a real batch.

Tests:
  1. build_messages() produces valid messages for a real step
  2. parse_response() correctly handles a real LLM response
  3. is_batch_eligible works correctly
  4. batch=True + non-batch client falls back to realtime
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

import asyncio

from accrue import EnrichmentConfig, LLMStep
from accrue.schemas.base import UsageInfo
from accrue.steps.base import StepContext
from accrue.steps.providers.base import LLMResponse

print("=== Batch Mechanics Test ===\n")

# --- Test 1: build_messages produces valid structure ---
print("Test 1: build_messages()")
step = LLMStep(
    "test",
    fields={"category": {"prompt": "Classify the company", "enum": ["Tech", "Finance", "Other"]}},
    model="gpt-4.1-nano",
    batch=True,
)

ctx = StepContext(
    row={"company": "Stripe", "description": "Payments"},
    fields={"category": {"prompt": "Classify"}},
    prior_results={},
    config=EnrichmentConfig(),
)

messages, kwargs = step.build_messages(ctx)
assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
assert messages[0]["role"] == "system"
assert messages[1]["role"] == "user"
assert kwargs["model"] == "gpt-4.1-nano"
assert kwargs["temperature"] == 0.2
assert kwargs["max_tokens"] == 4000
assert kwargs["response_format"] is not None
print(f"  ✓ Messages: {len(messages)} messages, model={kwargs['model']}")

# --- Test 2: parse_response handles real format ---
print("Test 2: parse_response()")
response = LLMResponse(
    content='{"category": "Tech"}',
    usage=UsageInfo(prompt_tokens=10, completion_tokens=3, total_tokens=13, model="gpt-4.1-nano"),
)
result = step.parse_response(response)
assert result.values == {"category": "Tech"}, f"Unexpected values: {result.values}"
assert result.usage.total_tokens == 13
print(f"  ✓ Parsed: {result.values}")

# Test parse with refusal → default
step_with_default = LLMStep(
    "test2",
    fields={"risk": {"prompt": "Assess risk", "default": "Unknown"}},
    batch=True,
)
response2 = LLMResponse(content='{"risk": "N/A"}', usage=None)
result2 = step_with_default.parse_response(response2)
assert result2.values["risk"] == "Unknown", f"Default not applied: {result2.values}"
print(f"  ✓ Default enforcement: 'N/A' → '{result2.values['risk']}'")

# --- Test 3: is_batch_eligible ---
print("Test 3: is_batch_eligible")
# OpenAIClient satisfies BatchCapableLLMClient
batch_step = LLMStep("s", fields=["f"], batch=True)
assert batch_step.is_batch_eligible is True, "OpenAIClient should be batch eligible"
print("  ✓ OpenAIClient + batch=True → eligible")

non_batch_step = LLMStep("s", fields=["f"], batch=False)
assert non_batch_step.is_batch_eligible is False
print("  ✓ batch=False → not eligible")

# --- Test 4: batch=True with non-batch client falls back ---
print("Test 4: Non-batch client fallback")


class SimpleClient:
    async def complete(
        self, messages, model, temperature, max_tokens, response_format=None, tools=None
    ):
        return LLMResponse(content='{"f": "value"}', usage=UsageInfo(total_tokens=1))


custom_step = LLMStep("s", fields=["f"], client=SimpleClient(), batch=True)
assert custom_step.is_batch_eligible is False, "SimpleClient should not be batch eligible"
print("  ✓ Custom non-batch client + batch=True → not eligible (will use realtime)")

# --- Test 5: Real API call through build_messages → complete → parse_response ---
print("Test 5: build_messages → real API call → parse_response")


async def test_real_call():
    step = LLMStep(
        "real_test",
        fields={"answer": {"prompt": "What is 2+2?", "type": "String"}},
        model="gpt-4.1-nano",
    )
    ctx = StepContext(
        row={"question": "math"},
        fields={},
        prior_results={},
        config=EnrichmentConfig(),
    )
    messages, kwargs = step.build_messages(ctx)
    client = step._resolve_client()
    response = await client.complete(
        messages=messages,
        model=kwargs["model"],
        temperature=kwargs["temperature"],
        max_tokens=kwargs["max_tokens"],
        response_format=kwargs["response_format"],
    )
    result = step.parse_response(response)
    assert "answer" in result.values, f"Missing answer: {result.values}"
    print(
        f"  ✓ Real API: answer={result.values['answer']}, tokens={result.usage.total_tokens if result.usage else 'N/A'}"
    )


asyncio.run(test_real_call())

print("\n✓ Batch mechanics test PASSED")

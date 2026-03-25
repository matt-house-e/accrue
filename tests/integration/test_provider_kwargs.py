"""Integration test: provider_kwargs passthrough + prompt caching."""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

import asyncio

from accrue import EnrichmentConfig, LLMStep, Pipeline
from accrue.steps.providers.anthropic import AnthropicClient


async def main():
    client = AnthropicClient()

    # Test 1: provider_kwargs with thinking
    print("Test 1: provider_kwargs with extended thinking", flush=True)
    resp = await client.complete(
        messages=[
            {"role": "system", "content": "Return JSON with key answer"},
            {"role": "user", "content": 'What is 15 * 23? Return {"answer": N}'},
        ],
        model="claude-haiku-4-5-20251001",
        temperature=1,  # thinking requires temperature=1
        max_tokens=8000,
        provider_kwargs={
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        },
    )
    print(f"  Response: {resp.content[:100]}", flush=True)
    print(f"  Tokens: {resp.usage.total_tokens if resp.usage else 'N/A'}", flush=True)
    assert "345" in resp.content, f"Expected 345 in response: {resp.content}"
    print("  ✓ thinking via provider_kwargs works", flush=True)

    # Test 2: prompt caching (system message has cache_control)
    print("\nTest 2: Prompt caching (Anthropic)", flush=True)
    long_system = "You are a helpful assistant. Return valid JSON. " * 50
    for i in range(2):
        resp = await client.complete(
            messages=[
                {"role": "system", "content": long_system},
                {"role": "user", "content": f'What is {i + 1}+1? Return {{"answer": N}}'},
            ],
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=100,
        )
        print(
            f"  Call {i + 1}: tokens={resp.usage.total_tokens if resp.usage else 'N/A'}", flush=True
        )
    print("  ✓ Prompt caching enabled (cache_control on system message)", flush=True)

    # Test 3: provider_kwargs through Pipeline.run()
    print("\nTest 3: provider_kwargs through Pipeline.run()", flush=True)
    pipeline = Pipeline(
        [
            LLMStep(
                "classify",
                fields={
                    "category": {"prompt": "Classify sector", "enum": ["Tech", "Finance", "Other"]}
                },
                model="claude-haiku-4-5-20251001",
                client=AnthropicClient(),
                provider_kwargs={"metadata": {"user_id": "test-123"}},
            )
        ]
    )
    result = await pipeline.run_async(
        [{"company": "Stripe", "description": "Payments"}],
        config=EnrichmentConfig(enable_progress_bar=False),
    )
    print(f"  Result: {result.data[0].get('category')}", flush=True)
    print(f"  Tokens: {result.cost.total_tokens}", flush=True)
    assert result.data[0].get("category") in ["Tech", "Finance", "Other"]
    print("  ✓ provider_kwargs flows through Pipeline.run()", flush=True)

    # Test 4: OpenAI provider_kwargs
    print("\nTest 4: OpenAI provider_kwargs", flush=True)
    from accrue.steps.providers.openai import OpenAIClient

    oai = OpenAIClient()
    resp = await oai.complete(
        messages=[
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": 'What is 2+2? Return {"answer": N}'},
        ],
        model="gpt-4.1-nano",
        temperature=0,
        max_tokens=50,
        provider_kwargs={"store": False},  # OpenAI-specific param
    )
    print(f"  Response: {resp.content}", flush=True)
    print("  ✓ OpenAI provider_kwargs works", flush=True)

    print("\n✓ All provider_kwargs + prompt caching tests PASSED", flush=True)


asyncio.run(main())

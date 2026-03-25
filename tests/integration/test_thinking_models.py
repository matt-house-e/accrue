"""Integration test: Anthropic thinking models via provider_kwargs.

Tests adaptive thinking (Claude 4.6) and extended thinking (older models)
through the provider_kwargs escape hatch, both realtime and via Pipeline.
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

import asyncio

from accrue import EnrichmentConfig, LLMStep, Pipeline
from accrue.steps.providers.anthropic import AnthropicClient


async def main():
    client = AnthropicClient()

    # ── Test 1: Adaptive thinking (Claude Sonnet 4.6) ─────────────────
    print("Test 1: Adaptive thinking (claude-sonnet-4-6)", flush=True)
    resp = await client.complete(
        messages=[
            {"role": "system", "content": "Return valid JSON with key 'answer'."},
            {
                "role": "user",
                "content": "How many r's are in the word 'strawberry'? Return {\"answer\": N}",
            },
        ],
        model="claude-sonnet-4-6",
        temperature=1,  # adaptive thinking works with temperature=1
        max_tokens=16000,
        provider_kwargs={
            "thinking": {"type": "adaptive"},
        },
    )
    print(f"  Response: {resp.content[:120]}", flush=True)
    print(
        f"  Tokens: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}, total={resp.usage.total_tokens}",
        flush=True,
    )
    assert "3" in resp.content, f"Expected 3 r's in strawberry: {resp.content}"
    print("  ✓ Adaptive thinking works — correct answer", flush=True)

    # ── Test 2: Extended thinking with budget (Haiku 4.5) ─────────────
    print("\nTest 2: Extended thinking with budget (claude-haiku-4-5)", flush=True)
    resp2 = await client.complete(
        messages=[
            {"role": "system", "content": "Return valid JSON with key 'answer'."},
            {"role": "user", "content": 'What is 17 * 29? Return {"answer": N}'},
        ],
        model="claude-haiku-4-5-20251001",
        temperature=1,
        max_tokens=8000,
        provider_kwargs={
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        },
    )
    print(f"  Response: {resp2.content[:120]}", flush=True)
    print(f"  Tokens: total={resp2.usage.total_tokens}", flush=True)
    assert "493" in resp2.content, f"Expected 493: {resp2.content}"
    print("  ✓ Extended thinking with budget works", flush=True)

    # ── Test 3: Adaptive thinking through Pipeline.run() ──────────────
    print("\nTest 3: Pipeline with adaptive thinking (claude-sonnet-4-6)", flush=True)
    pipeline = Pipeline(
        [
            LLMStep(
                "solve",
                fields={
                    "answer": {
                        "prompt": "Solve this math problem step by step and give the final answer",
                        "type": "String",
                    },
                },
                model="claude-sonnet-4-6",
                client=AnthropicClient(),
                temperature=1,
                max_tokens=16000,
                provider_kwargs={
                    "thinking": {"type": "adaptive"},
                },
            )
        ]
    )

    data = [
        {"problem": "If a train travels at 60 mph for 2.5 hours, how far does it go?"},
        {"problem": "What is the sum of the first 10 prime numbers?"},
    ]

    result = await pipeline.run_async(
        data,
        config=EnrichmentConfig(enable_progress_bar=False),
    )

    print(f"  Success rate: {result.success_rate:.0%}", flush=True)
    print(f"  Total tokens: {result.cost.total_tokens}", flush=True)
    for row in result.data:
        print(f"  Problem: {row['problem'][:50]}...", flush=True)
        print(f"    Answer: {row.get('answer', 'N/A')[:80]}", flush=True)

    assert result.success_rate == 1.0, f"Expected 100% success: {result.errors}"
    assert result.data[0].get("answer"), "Missing answer for problem 1"
    assert result.data[1].get("answer"), "Missing answer for problem 2"
    print("  ✓ Pipeline with adaptive thinking works", flush=True)

    # ── Test 4: Thinking + batch mode ─────────────────────────────────
    print("\nTest 4: Batch + thinking (claude-haiku-4-5)", flush=True)
    batch_pipeline = Pipeline(
        [
            LLMStep(
                "solve",
                fields={
                    "result": {
                        "prompt": "Calculate the answer to this math problem",
                        "type": "String",
                    },
                },
                model="claude-haiku-4-5-20251001",
                client=AnthropicClient(),
                temperature=1,
                max_tokens=8000,
                batch=True,
                provider_kwargs={
                    "thinking": {"type": "enabled", "budget_tokens": 2000},
                },
            )
        ]
    )

    batch_data = [
        {"problem": "What is 12 * 13?"},
        {"problem": "What is 99 + 101?"},
    ]

    batch_result = await batch_pipeline.run_async(
        batch_data,
        config=EnrichmentConfig(
            enable_progress_bar=False,
            batch_poll_interval=10.0,
            batch_timeout=600.0,
        ),
    )

    print(f"  Success rate: {batch_result.success_rate:.0%}", flush=True)
    print(f"  Total tokens: {batch_result.cost.total_tokens}", flush=True)
    step_usage = batch_result.cost.steps.get("solve")
    if step_usage:
        print(f"  Execution mode: {step_usage.execution_mode}", flush=True)
        print(f"  Batch ID: {step_usage.batch_id}", flush=True)
    for row in batch_result.data:
        print(f"  {row['problem']}: {row.get('result', 'N/A')}", flush=True)

    assert batch_result.success_rate == 1.0, f"Batch failed: {batch_result.errors}"
    if step_usage:
        assert step_usage.execution_mode == "batch"
    print("  ✓ Batch + thinking works", flush=True)

    print("\n✓ All thinking model tests PASSED", flush=True)


asyncio.run(main())

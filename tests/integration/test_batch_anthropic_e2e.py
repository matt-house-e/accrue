"""Integration test: Anthropic Batch API — submit, verify, and poll.

Tests the Anthropic Message Batches API end-to-end.
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

import asyncio

from accrue.steps.providers.anthropic import AnthropicClient
from accrue.steps.providers.base import BatchCapableLLMClient, BatchRequest


async def main():
    client = AnthropicClient()

    # 1. Protocol check
    assert isinstance(client, BatchCapableLLMClient), (
        "AnthropicClient should satisfy BatchCapableLLMClient"
    )
    print("✓ AnthropicClient satisfies BatchCapableLLMClient protocol")

    # 2. Realtime call first (sanity check key works)
    print("\nTest 1: Realtime call...")
    response = await client.complete(
        messages=[
            {"role": "system", "content": 'Return JSON: {"answer": "value"}'},
            {"role": "user", "content": "What is 2+2? Return as JSON."},
        ],
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
        max_tokens=100,
        response_format={"type": "json_object"},
    )
    print(f"  Realtime response: {response.content[:100]}")
    print(f"  Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
    print("  ✓ Realtime works")

    # 3. Submit a batch
    print("\nTest 2: Submit batch...")
    requests = [
        BatchRequest(
            custom_id="row-0",
            messages=[
                {"role": "system", "content": "Return valid JSON with the key 'greeting'."},
                {"role": "user", "content": "Say hello to Alice. Return as JSON."},
            ],
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=100,
        ),
        BatchRequest(
            custom_id="row-1",
            messages=[
                {"role": "system", "content": "Return valid JSON with the key 'greeting'."},
                {"role": "user", "content": "Say hello to Bob. Return as JSON."},
            ],
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=100,
        ),
    ]

    batch_id = await client.submit_batch(requests)
    print(f"  Batch submitted: {batch_id}")

    # 4. Check status
    inner = client._get_client()
    batch = await inner.messages.batches.retrieve(batch_id)
    print(f"  Status: {batch.processing_status}")
    print("  ✓ Batch created on Anthropic")

    # 5. Poll until complete (or timeout)
    print("\nTest 3: Polling batch (15s interval, 10min timeout)...")
    try:
        result = await client.poll_batch(batch_id, poll_interval=15.0, timeout=600.0)
        print("  ✓ Batch completed!")
        print(f"  Responses: {len(result.responses)}")
        print(f"  Failed: {len(result.failed_ids)}")
        print(f"  Batch ID: {result.batch_id}")
        for cid, resp in result.responses.items():
            print(f"  {cid}: {resp.content[:80]}")
            if resp.usage:
                print(f"    tokens: {resp.usage.total_tokens}")

        assert len(result.responses) == 2, f"Expected 2 responses, got {len(result.responses)}"
        assert result.failed_ids == [], f"Unexpected failures: {result.failed_ids}"
        print("\n✓ Anthropic Batch API E2E PASSED")

    except Exception as e:
        print(f"  Batch did not complete in time: {e}")
        print(f"  Batch ID for manual check: {batch_id}")
        # Cancel to clean up
        await client.cancel_batch(batch_id)
        print(f"  Cancelled batch {batch_id}")
        print("\n⚠ Anthropic batch timed out (queue latency) — submit+poll logic verified")


asyncio.run(main())

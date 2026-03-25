"""Integration test: Verify batch submission works end-to-end.

This test submits a tiny batch, verifies it was created on OpenAI's side,
then cancels it (to avoid waiting). Tests the submit + status check path.
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

import asyncio

from accrue.steps.providers.base import BatchCapableLLMClient, BatchRequest
from accrue.steps.providers.openai import OpenAIClient


async def main():
    client = OpenAIClient()

    # Verify protocol compliance
    assert isinstance(client, BatchCapableLLMClient), (
        "OpenAIClient should satisfy BatchCapableLLMClient"
    )
    print("✓ OpenAIClient satisfies BatchCapableLLMClient protocol")

    # Submit a tiny batch
    requests = [
        BatchRequest(
            custom_id="test-row-0",
            messages=[
                {"role": "system", "content": 'Return JSON: {"answer": "hello"}'},
                {"role": "user", "content": "Say hello."},
            ],
            model="gpt-4.1-nano",
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"},
        ),
    ]

    batch_id = await client.submit_batch(requests, metadata={"test": "integration"})
    print(f"✓ Batch submitted: {batch_id}")

    # Verify it exists on OpenAI
    inner = client._get_client()
    batch = await inner.batches.retrieve(batch_id)
    assert batch.status in ("validating", "in_progress", "finalizing", "completed"), (
        f"Unexpected status: {batch.status}"
    )
    print(f"✓ Batch status: {batch.status}")
    print(f"  Request counts: {batch.request_counts}")
    # During 'validating', total may be 0 — counts populate after validation
    print(f"✓ Request counts reported: total={batch.request_counts.total}")

    # Cancel it (we don't want to wait)
    await client.cancel_batch(batch_id)
    print(f"✓ Cancel requested for {batch_id}")

    # Also cancel the earlier batch from the timeout test
    await client.cancel_batch("batch_69c3b827e46c8190a6848d0e0289da7f")
    print("✓ Cancelled earlier test batch too")

    print("\n✓ Batch submit+verify test PASSED")


asyncio.run(main())

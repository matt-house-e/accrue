"""Poll an existing OpenAI batch to verify result download and parsing works."""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

import asyncio

from accrue.steps.providers.openai import OpenAIClient

BATCH_ID = "batch_69c3b827e46c8190a6848d0e0289da7f"


async def main():
    client = OpenAIClient()

    # First just check the status
    inner = client._get_client()
    batch = await inner.batches.retrieve(BATCH_ID)
    print(f"Batch {BATCH_ID}")
    print(f"  Status: {batch.status}")
    print(f"  Request counts: {batch.request_counts}")

    if batch.status == "completed":
        print("\nBatch is complete! Downloading results...")
        result = await client._download_batch_results(batch, BATCH_ID)
        print(f"  Responses: {len(result.responses)}")
        print(f"  Failed: {len(result.failed_ids)}")
        for cid, resp in result.responses.items():
            print(f"  {cid}: {resp.content[:100]}...")
            print(f"    tokens: {resp.usage.total_tokens if resp.usage else 'N/A'}")
        print("\n✓ Batch result download and parsing PASSED")
    elif batch.status == "in_progress":
        print("\nBatch still processing. Waiting with 30s poll interval...")
        result = await client.poll_batch(BATCH_ID, poll_interval=30.0, timeout=1800.0)
        print(f"  Responses: {len(result.responses)}")
        print(f"  Failed: {len(result.failed_ids)}")
        for cid, resp in result.responses.items():
            print(f"  {cid}: {resp.content[:100]}...")
        print("\n✓ Batch result download and parsing PASSED")
    else:
        print(f"\nBatch is in status '{batch.status}' — cannot download results")


asyncio.run(main())

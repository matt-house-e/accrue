# Batch Processing at Scale

Use the batch API to process large datasets at 50% lower cost. Batch requests are
submitted to the provider (OpenAI or Anthropic) and processed asynchronously, typically
completing within minutes to hours depending on volume.

## Full Working Example

```python
from accrue import Pipeline, LLMStep, EnrichmentConfig

# -- Input data ---------------------------------------------------------------
# In practice, load this from a CSV, database, or API. Batch processing is
# most valuable at 100+ rows where the 50% cost savings add up.

companies = [
    {"company": "Stripe", "industry": "Payments", "employees": 8000},
    {"company": "Notion", "industry": "Productivity", "employees": 1000},
    {"company": "Datadog", "industry": "Monitoring", "employees": 5000},
    {"company": "Figma", "industry": "Design Tools", "employees": 1500},
    {"company": "Plaid", "industry": "Fintech", "employees": 1200},
    {"company": "Vercel", "industry": "Developer Tools", "employees": 400},
    {"company": "Retool", "industry": "Internal Tools", "employees": 500},
    {"company": "Snyk", "industry": "Security", "employees": 1000},
    {"company": "Miro", "industry": "Collaboration", "employees": 1800},
    {"company": "Linear", "industry": "Project Management", "employees": 100},
]

# -- Pipeline with batch=True -------------------------------------------------

pipeline = Pipeline([
    LLMStep(
        "classify",
        fields={
            "category": {
                "prompt": "Classify the primary business sector",
                "type": "String",
                "enum": ["Technology", "Healthcare", "Finance", "Retail", "Other"],
            },
            "summary": {
                "prompt": "Write a one-sentence business summary",
                "type": "String",
            },
            "growth_stage": {
                "prompt": "Classify the company's growth stage based on employee count and industry",
                "type": "String",
                "enum": ["Startup", "Growth", "Scale-up", "Enterprise"],
            },
        },
        batch=True,    # Submit as a batch job instead of realtime API calls
        model="gpt-4.1-mini",
    ),
])

# -- Use the for_batch() config preset ----------------------------------------

config = EnrichmentConfig.for_batch()
# Equivalent to:
#   EnrichmentConfig(
#       enable_caching=True,
#       enable_checkpointing=True,
#       batch_poll_interval=60.0,     # Check status every 60 seconds
#       batch_timeout=86400.0,        # Wait up to 24 hours
#       max_retries=5,
#   )

# -- Run -----------------------------------------------------------------------
# Pipeline.run() blocks until the batch completes. For OpenAI, the pipeline:
#   1. Builds JSONL with one request per uncached row
#   2. Uploads the file and creates a batch job
#   3. Polls for completion at batch_poll_interval
#   4. Downloads results and maps them back to rows
#   5. Falls back to realtime for any failed rows

result = pipeline.run(companies, config=config)

# -- Inspect results -----------------------------------------------------------

for row in result.data:
    print(f"{row['company']:15s} | {row['category']:12s} | {row['growth_stage']:12s} | {row['summary']}")

# -- Monitor batch execution ---------------------------------------------------

usage = result.cost.steps["classify"]
print(f"\nExecution mode: {usage.execution_mode}")   # "batch"
print(f"Batch ID:       {usage.batch_id}")
print(f"Rows processed: {usage.rows_processed}")
print(f"Cache hits:     {usage.cache_hits}")
print(f"Cache misses:   {usage.cache_misses}")
print(f"Total tokens:   {usage.total_tokens:,}")
```

## Expected Output

```
Stripe          | Technology   | Scale-up     | Stripe provides payment processing infrastructure ...
Notion          | Technology   | Growth       | Notion is an all-in-one workspace for notes ...
Datadog         | Technology   | Scale-up     | Datadog offers cloud-scale monitoring and ...
Figma           | Technology   | Growth       | Figma is a collaborative interface design tool ...
Plaid           | Finance      | Growth       | Plaid connects applications to users' bank ...
Vercel          | Technology   | Startup      | Vercel provides a frontend deployment platform ...
Retool          | Technology   | Startup      | Retool enables teams to build internal tools ...
Snyk            | Technology   | Growth       | Snyk is a developer-first security platform ...
Miro            | Technology   | Scale-up     | Miro is an online collaborative whiteboard ...
Linear          | Technology   | Startup      | Linear is a streamlined project management tool ...

Execution mode: batch
Batch ID:       batch_abc123...
Rows processed: 10
Cache hits:     0
Cache misses:   10
Total tokens:   8,500
```

## Cost-Saving Strategy: Sample Then Batch

A practical workflow for large datasets is to test on a small sample with realtime
calls, verify the output quality, then batch the rest:

```python
import pandas as pd

# Load your full dataset
full_data = pd.read_csv("companies.csv")  # e.g., 10,000 rows
print(f"Total rows: {len(full_data)}")

# Step 1: Test on a small sample (realtime, fast feedback)
sample = full_data.sample(10, random_state=42).to_dict("records")

test_pipeline = Pipeline([
    LLMStep(
        "classify",
        fields={
            "category": {
                "prompt": "Classify the primary business sector",
                "enum": ["Technology", "Healthcare", "Finance", "Retail", "Other"],
            },
            "summary": "One-sentence business summary",
        },
        # batch=False is the default -- realtime execution
    ),
])

test_config = EnrichmentConfig(enable_caching=True)
test_result = test_pipeline.run(sample, config=test_config)

# Review output quality
for row in test_result.data:
    print(f"{row['company']}: {row['category']} -- {row['summary']}")

# Step 2: Happy with quality? Batch the full dataset at 50% cost savings
batch_pipeline = Pipeline([
    LLMStep(
        "classify",
        fields={
            "category": {
                "prompt": "Classify the primary business sector",
                "enum": ["Technology", "Healthcare", "Finance", "Retail", "Other"],
            },
            "summary": "One-sentence business summary",
        },
        batch=True,
    ),
])

batch_config = EnrichmentConfig.for_batch()
batch_result = batch_pipeline.run(full_data, config=batch_config)

print(f"Processed {len(batch_result.data)} rows")
print(f"Total tokens: {batch_result.cost.total_tokens:,}")
```

## Batch with Caching

Batch execution is cache-aware. If you have already processed some rows (via realtime
or a previous batch), those rows are served from cache and only the remaining rows
are submitted to the batch API:

```python
# First run: all 1000 rows go to batch
result1 = pipeline.run(data[:1000], config=config)
usage1 = result1.cost.steps["classify"]
print(f"Run 1 -- Cache hits: {usage1.cache_hits}, Misses: {usage1.cache_misses}")
# Run 1 -- Cache hits: 0, Misses: 1000

# Second run: add 200 new rows. The original 1000 come from cache.
result2 = pipeline.run(data[:1200], config=config)
usage2 = result2.cost.steps["classify"]
print(f"Run 2 -- Cache hits: {usage2.cache_hits}, Misses: {usage2.cache_misses}")
# Run 2 -- Cache hits: 1000, Misses: 200
```

## Using Anthropic for Batch

Batch processing works with Anthropic's Message Batches API as well:

```python
from accrue.providers import AnthropicClient

pipeline = Pipeline([
    LLMStep(
        "classify",
        fields={
            "category": {
                "prompt": "Classify the primary business sector",
                "enum": ["Technology", "Healthcare", "Finance", "Retail", "Other"],
            },
            "summary": "One-sentence business summary",
        },
        batch=True,
        client=AnthropicClient(),
        model="claude-sonnet-4-20250514",
    ),
])
```

Both OpenAI and Anthropic batch APIs offer 50% cost reduction. The pipeline handles
the provider-specific batch submission, polling, and result retrieval automatically.

## Constraints and Limitations

**Provider support.** Batch mode is available for OpenAI and Anthropic. Google does not
currently support batch processing.

**No grounding with batch.** Web search grounding (`grounding=True`) is incompatible with
batch execution because grounding requires realtime web access. Use a `FunctionStep` with
`web_search()` before the batch step if you need web context.

**Completion time.** Batch jobs are processed at lower priority by the provider. OpenAI
targets 24-hour completion; most jobs finish within 1-2 hours. Anthropic batch jobs
typically complete faster for smaller volumes.

**Auto-chunking.** Datasets larger than 50,000 rows (configurable via
`config.batch_max_requests`) are automatically split into multiple batch jobs and
reassembled.

**Realtime fallback.** If individual rows fail within a batch, the pipeline automatically
retries those rows using realtime API calls so you always get complete results.

## Configuration Reference

The `EnrichmentConfig` batch settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `batch_poll_interval` | 60.0 | Seconds between status checks |
| `batch_timeout` | 86400.0 | Maximum wait time (24 hours) |
| `batch_max_requests` | 50000 | Auto-chunk threshold |

Override any of these on the config:

```python
config = EnrichmentConfig.for_batch()
config.batch_poll_interval = 30.0   # Check more frequently
config.batch_timeout = 7200.0       # 2-hour timeout
```

# Company Enrichment

The canonical Accrue use case: enrich a list of companies with market intelligence using web research and LLM analysis.

This pipeline demonstrates three steps chained together:

1. **Research** -- web search gathers context about each company
2. **Classify** -- LLM reads the research and produces structured scores
3. **Synthesize** -- LLM writes a human-readable summary from the classification

## Full Working Example

```python
from accrue import (
    Pipeline,
    LLMStep,
    FunctionStep,
    EnrichmentConfig,
    web_search,
)

# -- Input data ---------------------------------------------------------------

companies = [
    {"company": "Stripe", "domain": "stripe.com", "sector": "Fintech"},
    {"company": "Notion", "domain": "notion.so", "sector": "Productivity"},
    {"company": "Datadog", "domain": "datadoghq.com", "sector": "DevOps"},
]

# -- Pipeline definition ------------------------------------------------------

pipeline = Pipeline([
    # Step 1: Web research
    # web_search() returns an async callable that uses OpenAI's Responses API
    # to search the web. The query template uses {field} placeholders filled
    # from the row data.
    #
    # Output fields:
    #   __web_context -- internal field (double-underscore prefix), automatically
    #                    filtered from final output but available to downstream steps
    #   sources       -- list of URLs cited in the research
    FunctionStep(
        "research",
        fn=web_search(
            "Research {company} ({domain}): market position, recent funding, "
            "key products, and competitive landscape in the {sector} sector",
            search_context_size="medium",
        ),
        fields=["__web_context", "sources"],
    ),

    # Step 2: Classify and score
    # depends_on=["research"] makes __web_context and sources available
    # in the LLM's input context. The LLM sees the original row data PLUS
    # all prior_results from dependency steps.
    LLMStep(
        "classify",
        fields={
            "market_position": {
                "prompt": "Classify the company's current market position",
                "type": "String",
                "enum": ["Leader", "Challenger", "Niche", "Emerging"],
                "examples": ["Leader"],
            },
            "growth_score": {
                "prompt": "Rate the company's growth trajectory from 1-10",
                "type": "Number",
                "format": "X/10",
                "examples": ["8"],
            },
            "competitive_moat": {
                "prompt": "Describe the company's primary competitive advantage in one phrase",
                "type": "String",
                "examples": ["Network effects from developer ecosystem"],
            },
            "risk_level": {
                "prompt": "Assess overall investment risk",
                "type": "String",
                "enum": ["Low", "Medium", "High"],
                "default": "Medium",
            },
        },
        depends_on=["research"],
        model="gpt-4.1-mini",
    ),

    # Step 3: Synthesis
    # depends_on=["classify"] gives this step access to classify's outputs
    # AND research's outputs (transitive -- the pipeline merges all prior results).
    LLMStep(
        "synthesize",
        fields={
            "executive_summary": {
                "prompt": (
                    "Write a 2-3 sentence executive summary of this company's "
                    "market position, growth potential, and key risks. "
                    "Reference specific data points from the research."
                ),
                "type": "String",
            },
        },
        depends_on=["classify"],
        model="gpt-4.1-mini",
    ),
])

# -- Configuration ------------------------------------------------------------

config = EnrichmentConfig(
    enable_caching=True,       # Cache results in .accrue/cache.db
    cache_ttl=86400,           # Cache for 24 hours
    max_workers=5,             # Concurrent rows (conservative for Tier 1)
    temperature=0.2,           # Low temperature for consistent structured output
    log_level="INFO",
)

# -- Run the pipeline ----------------------------------------------------------

result = pipeline.run(companies, config=config)

# -- Inspect results -----------------------------------------------------------

# result.data is a list[dict] (matches input type)
for row in result.data:
    print(f"\n{'=' * 60}")
    print(f"Company:          {row['company']}")
    print(f"Market Position:  {row['market_position']}")
    print(f"Growth Score:     {row['growth_score']}/10")
    print(f"Competitive Moat: {row['competitive_moat']}")
    print(f"Risk Level:       {row['risk_level']}")
    print(f"Summary:          {row['executive_summary']}")
    print(f"Sources:          {len(row.get('sources', []))} URLs")

# Cost and performance
print(f"\nTotal tokens used: {result.cost.total_tokens:,}")
print(f"Success rate:      {result.success_rate:.0%}")
print(f"Errors:            {len(result.errors)}")

# Per-step breakdown
for step_name, usage in result.cost.steps.items():
    print(f"\n  Step '{step_name}':")
    print(f"    Rows processed: {usage.rows_processed}")
    print(f"    Cache hits:     {usage.cache_hits}")
    print(f"    Cache misses:   {usage.cache_misses}")
    print(f"    Tokens:         {usage.total_tokens:,}")
```

## Expected Output

```
============================================================
Company:          Stripe
Market Position:  Leader
Growth Score:     9/10
Competitive Moat: Network effects from developer ecosystem
Risk Level:       Low
Summary:          Stripe is the dominant leader in online payments infrastructure,
                  processing hundreds of billions in annual volume. Its developer-first
                  approach and expanding financial services suite drive strong growth,
                  though increasing competition from Adyen poses a measured risk.
Sources:          4 URLs

============================================================
Company:          Notion
Market Position:  Challenger
Growth Score:     7/10
...

Total tokens used: 12,450
Success rate:      100%
Errors:            0

  Step 'research':
    Rows processed: 3
    Cache hits:     0
    Cache misses:   3
    Tokens:         0

  Step 'classify':
    Rows processed: 3
    Cache hits:     0
    Cache misses:   3
    Tokens:         4,200

  Step 'synthesize':
    Rows processed: 3
    Cache hits:     0
    Cache misses:   3
    Tokens:         8,250
```

## Key Concepts Demonstrated

**Web search as a FunctionStep.** The `web_search()` factory returns an async callable
that fits directly into `FunctionStep`. The query string supports `{field}` placeholders
that are interpolated from the row data.

**Internal fields.** The `__web_context` field uses the double-underscore prefix convention.
It is available to downstream steps via `prior_results` but is automatically stripped
from the final output. Use this for intermediate data that the user should not see.

**Field specs with validation.** Each field in the `classify` step has a full spec:
`prompt`, `type`, `enum`, `examples`, and `default`. Accrue validates these at
construction time and generates a structured JSON schema for the LLM call.

**Dependency chains.** `synthesize` depends on `classify`, which depends on `research`.
The pipeline resolves this as a three-level DAG and executes steps in topological order.
Each step receives all upstream outputs merged into `prior_results`.

**Caching.** With `enable_caching=True`, re-running the same pipeline on the same data
skips redundant API calls. Cache keys are based on the input row, step name, model, and
field specs -- so changing a prompt invalidates the cache automatically.

## Running with a DataFrame

Accrue accepts either `list[dict]` or a pandas DataFrame. The output type matches
the input type:

```python
import pandas as pd

df = pd.DataFrame(companies)
result = pipeline.run(df, config=config)

# result.data is now a DataFrame
print(result.data[["company", "market_position", "growth_score"]])
```

## Using a Different Provider

Swap in Anthropic or Google by passing a `client` argument:

```python
from accrue.providers import AnthropicClient

LLMStep(
    "classify",
    fields={...},
    depends_on=["research"],
    client=AnthropicClient(),
    model="claude-sonnet-4-20250514",
)
```

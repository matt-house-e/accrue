# Accrue

**Enrich datasets with LLMs. Compose pipelines, run them across thousands of rows, get structured output back.**

## Install

```bash
pip install accrue
```

Python 3.10+. Set `OPENAI_API_KEY` to get started. Optional extras: `accrue[anthropic]`, `accrue[google]`.

## Example

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate total addressable market in billions USD",
        "risk_level": {
            "prompt": "Assess investment risk",
            "enum": ["Low", "Medium", "High"],
        },
    })
])

result = pipeline.run([
    {"company": "Stripe", "sector": "Fintech"},
    {"company": "Notion", "sector": "Productivity"},
])

print(result.data)  # Enriched data with new fields
print(result.cost)  # Token usage breakdown
```

## Features

- **Multi-step pipelines** -- Chain LLM steps, function steps, and conditional logic into DAGs
- **Provider-agnostic** -- OpenAI, Anthropic, and Google via a clean `LLMClient` protocol
- **7-key field specs** -- Prompt, type, format, enum, examples, bad_examples, and default per output field
- **Structured output** -- Pydantic validation with automatic retries on parse failure
- **Caching** -- SQLite-backed input-hash cache so re-runs skip completed work
- **Batch API** -- `LLMStep(batch=True)` for high-volume jobs at 50% cost via OpenAI and Anthropic batch endpoints
- **Grounding** -- Web search grounding with citations via `LLMStep(grounding=True)`
- **Conditional steps** -- `run_if` / `skip_if` predicates for per-row branching
- **Hooks and observability** -- Progress callbacks, cost tracking, checkpoint recovery

## Quick Links

- [Getting Started](getting-started/quickstart.md) -- Installation, first pipeline, core concepts
- [Guides](guides/field-specifications.md) -- Field specs, caching, batch API, grounding, conditional steps, providers
- [Cookbook](cookbook/company-enrichment.md) -- Real-world examples and patterns
- [API Reference](reference/api.md) -- Full class and method reference

<p align="center">
  <strong>Accrue</strong><br>
  <em>The enrichment pipeline engine.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/accrue/"><img src="https://img.shields.io/pypi/v/accrue?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/accrue/"><img src="https://img.shields.io/pypi/pyversions/accrue" alt="Python"></a>
  <a href="https://github.com/accrue-team/accrue/blob/main/LICENSE"><img src="https://img.shields.io/github/license/accrue-team/accrue" alt="License"></a>
</p>

---

**Define a pipeline. Point it at your data. Get structured results.** Accrue is a Python library for enriching datasets with LLMs. Compose multi-step pipelines, run them across hundreds to tens of thousands of rows, and get validated, structured output back -- with caching, retries, and parallel execution handled for you.

No platform. No markup. Just a pipeline you can version-control, iterate on, and reason about.

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate total addressable market in billions USD",
        "competition": {
            "prompt": "Rate competitive intensity with key competitors",
            "enum": ["Low", "Medium", "High"],
            "examples": ["High - Competes with AWS, Google Cloud"],
        },
        "growth_potential": {
            "prompt": "Assess 5-year growth trajectory",
            "type": "String",
            "format": "X% CAGR - reasoning",
        },
    })
])

result = pipeline.run(df)  # DataFrame in, DataFrame out
print(result.data.head())
print(f"Tokens used: {result.cost.total_tokens:,}")
```

## Install

Requires Python 3.10+.

```bash
pip install accrue
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

That's it. OpenAI is the default provider (zero config, [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) auto-enabled). Anthropic and Google are optional:

```bash
pip install accrue[anthropic]  # Claude
pip install accrue[google]     # Gemini
```

## Why Accrue

You have a spreadsheet of companies, leads, or entities. You need structured fields added to every row -- classifications, summaries, scores, extracted data. You could write a `for` loop and call the OpenAI API, but then you're building retry logic, rate limiting, caching, progress tracking, and crash recovery. You could use Clay, but you'd pay $500/month for something you can't version-control.

Accrue is the pipeline between a single API call and a full platform:

| | Raw API calls | Accrue | Clay |
|---|---|---|---|
| **Scope** | One call at a time | Pipeline of steps across rows | Full SaaS platform |
| **Multi-step** | Manual orchestration | DAG with parallel execution | Sequential drag-and-drop |
| **Caching** | Build it yourself | SQLite, auto-invalidates on prompt change | Platform-managed |
| **Crash recovery** | Start over | Checkpoint + row-level cache resume | Platform-managed |
| **Iterate on prompts** | Re-run everything | Only re-process changed steps/rows | Re-run everything |
| **Cost** | API costs | API costs | $$$$/month + API costs |
| **Version control** | Yes | Yes | No |

## Quick Example

Chain steps together with `depends_on`. Use `web_search()` to ground LLM answers in live data:

```python
from accrue import Pipeline, FunctionStep, LLMStep, web_search

pipeline = Pipeline([
    FunctionStep("research",
        fn=web_search("Research {company}: market position, competitors, recent news"),
        fields=["__web_context", "sources"],
    ),
    LLMStep("analyze",
        fields={
            "market_size": "Estimate TAM in billions USD",
            "competitors": {"prompt": "List top 3 competitors", "type": "List[String]"},
            "investment_thesis": "One-paragraph investment thesis",
        },
        depends_on=["research"],
    ),
])

result = pipeline.run(companies_df)
```

## Features

- **Multi-step pipelines** -- Chain LLM steps and function steps into a DAG with automatic dependency resolution and parallel execution. [Quickstart](docs/getting-started/quickstart.md)

- **Provider-agnostic** -- OpenAI, Anthropic (with automatic prompt caching), and Google ship as adapters. Any OpenAI-compatible API works via `base_url`. Custom providers implement one async method. [Providers guide](docs/guides/providers.md)

- **7-key field specs** -- Control LLM output with `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, and `default`. Drives structured outputs and Pydantic validation automatically. [Field specs guide](docs/guides/field-specifications.md)

- **Caching and checkpointing** -- SQLite input-hash cache auto-invalidates on prompt changes. Checkpointing saves after each step for crash recovery. [Caching guide](docs/guides/caching.md)

- **Batch API** -- `LLMStep(batch=True)` for 50% cost savings via OpenAI and Anthropic batch endpoints. Cache-aware, auto-chunking, realtime fallback on failures. [Batch guide](docs/guides/batch-api.md)

- **Web search and grounding** -- `web_search()` factory for search-then-analyze pipelines, or `grounding=True` for native provider web search with normalized citations. [Web search guide](docs/guides/web-search.md)

- **Conditional steps** -- `run_if` / `skip_if` predicates for per-row branching. Skipped rows get defaults, never hit the API. [Conditional steps guide](docs/guides/conditional-steps.md)

- **Hooks** -- Typed lifecycle events for observability. Sync and async callables, never crash the pipeline. [Hooks guide](docs/guides/hooks.md)

- **`provider_kwargs`** -- Escape hatch for provider-specific features (extended thinking, effort control, etc.) without waiting for first-class support.

## Sweet Spot

Accrue is built for **100 to 50,000 rows** -- too many for manual work or single-call tools, too few to justify big data infrastructure.

| Rows | Time (3 steps, 10 workers) | Cost (gpt-4.1-mini) |
|------|---------------------------|---------------------|
| 100 | ~30s | ~$0.20 |
| 1,000 | ~5 min | ~$2 |
| 10,000 | ~50 min | ~$20 |
| 50,000 | ~50 min (50 workers) | ~$100 |

With `batch=True`, halve the API costs. Cached steps re-run in seconds.

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started/quickstart.md) | Installation, first pipeline, core concepts |
| [Guides](docs/guides/) | Field specs, providers, caching, batch API, grounding, hooks, errors, configuration |
| [Cookbook](docs/cookbook/) | End-to-end examples: [company enrichment](docs/cookbook/company-enrichment.md), [lead scoring](docs/cookbook/lead-scoring.md), [content analysis](docs/cookbook/content-analysis.md), [batch processing](docs/cookbook/batch-processing.md) |
| [API Reference](docs/reference/api.md) | Complete reference for every public export |

## Contributing

```bash
git clone https://github.com/accrue-team/accrue.git
cd accrue
pip install -e ".[dev]"
pytest
```

## License

MIT

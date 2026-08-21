<p align="center">
  <strong>Accrue</strong><br>
  <em>The enrichment pipeline engine.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/accrue/"><img src="https://img.shields.io/pypi/v/accrue?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/accrue/"><img src="https://img.shields.io/pypi/pyversions/accrue" alt="Python"></a>
  <a href="https://github.com/matt-house-e/accrue/blob/main/LICENSE"><img src="https://img.shields.io/github/license/matt-house-e/accrue" alt="License"></a>
</p>

---

**Define a pipeline. Point it at your data. Get structured results.** Accrue is a Python library for enriching datasets with LLMs. Compose multi-step pipelines, run them across hundreds to tens of thousands of rows, and get validated, structured output back -- with caching, retries, and parallel execution handled for you.

No platform. No markup. Just a pipeline you can version-control, iterate on, and reason about.

<p align="center">
  <a href="#watch-it-run"><img src="https://raw.githubusercontent.com/matt-house-e/accrue/main/docs/assets/watch-overview.png" alt="Accrue Watch, Overview tab: a live 5,000-row enrichment run with its pipeline steps, models, produced fields, and cost" width="100%"></a>
  <br><sub>A 5,000-row run in <a href="#watch-it-run"><b>Accrue Watch</b></a> -- every step, model, field, and dollar, read live from the run log.</sub>
</p>

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

## Claude Code Skill

If you use [Claude Code](https://claude.ai/claude-code), Accrue ships with a built-in `/accrue` skill that guides you through building pipelines interactively. It designs fields, picks models, estimates costs, and writes your script -- you just review and run.

```
> /accrue
> I have 500 companies in accounts.csv, I need to qualify them for ICP fit
```

The skill walks you through field design, model selection, pipeline architecture, and configuration before writing a production-ready script. See [Using the Claude Code Skill](docs/getting-started/claude-code-skill.md) for details.

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

## Watch It Run

**Accrue Watch** ([accrue-ui](https://github.com/matt-house-e/accrue-ui)) is a local dashboard for a running pipeline -- rows x steps as a live grid, every cell's values and errors, failures grouped by cause, cost by step. It reads the JSONL run log Accrue writes; there is nothing else to configure. The Overview tab is at the top of this page; this is the run grid:

<p align="center">
  <img src="https://raw.githubusercontent.com/matt-house-e/accrue/main/docs/assets/watch-datagrid.png" alt="Accrue Watch, run grid in data view: one produced field per step, with retrying and failed cells highlighted in place" width="100%">
  <br><sub><b>Run grid, data view</b> -- one field per step, filling in as rows complete; retries and failures show in place.</sub>
</p>

Turn on the run log, then point the dashboard at it:

```python
result = pipeline.run(companies_df, run_log=True)  # writes .accrue/runs/<run_id>.jsonl
```

```bash
pip install git+https://github.com/matt-house-e/accrue-ui  # not on PyPI yet
accrue watch                                                # opens the latest run in your browser
```

`accrue watch` is a thin stub that delegates to accrue-ui. Add `--pipeline module:attr` to enable one-click retry of failed rows from the dashboard. There is no `accrue[ui]` extra until accrue-ui is published. [Run log guide](docs/guides/run-log.md)

## Features

- **Multi-step pipelines** -- Chain LLM steps and function steps into a DAG with automatic dependency resolution and parallel execution. [Quickstart](docs/getting-started/quickstart.md)

- **Provider-agnostic** -- OpenAI, Anthropic (with automatic prompt caching), and Google ship as adapters. Any OpenAI-compatible API works via `base_url`. Custom providers implement one async method. [Providers guide](docs/guides/providers.md)

- **7-key field specs** -- Control LLM output with `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, and `default`. Drives structured outputs and Pydantic validation automatically. [Field specs guide](docs/guides/field-specifications.md)

- **Caching and checkpointing** -- SQLite input-hash cache auto-invalidates on prompt changes. Checkpointing saves after each step for crash recovery. [Caching guide](docs/guides/caching.md)

- **Batch API** -- `LLMStep(batch=True)` for 50% cost savings via OpenAI and Anthropic batch endpoints. Cache-aware, auto-chunking, realtime fallback on failures. [Batch guide](docs/guides/batch-api.md)

- **Web search and grounding** -- `web_search()` factory for search-then-analyze pipelines, or `grounding=True` for native provider web search with normalized citations. [Web search guide](docs/guides/web-search.md)

- **Conditional steps** -- `run_if` / `skip_if` predicates for per-row branching. Skipped rows get defaults, never hit the API. [Conditional steps guide](docs/guides/conditional-steps.md)

- **Hooks** -- Typed lifecycle events for observability. Sync and async callables, never crash the pipeline. [Hooks guide](docs/guides/hooks.md)

- **Run logs** -- `run(..., run_log=True)` streams the run as append-only JSONL (schema v1): per-step and per-row events with status, errors, and token usage. Crash-safe, `tail -f`-able. [Run log guide](docs/guides/run-log.md)

- **Run diffs** -- `accrue.compare(result_a, result_b)` diffs two runs (e.g. before/after a prompt tweak) -- changed rows, per-field churn, distribution shift, and cost delta -- no labels needed. [Compare guide](docs/guides/compare.md)

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
| [Claude Code Skill](docs/getting-started/claude-code-skill.md) | Interactive pipeline builder via `/accrue` |
| [Guides](docs/guides/) | Field specs, providers, caching, batch API, grounding, hooks, run logs, errors, configuration |
| [Cookbook](docs/cookbook/) | End-to-end examples: [company enrichment](docs/cookbook/company-enrichment.md), [lead scoring](docs/cookbook/lead-scoring.md), [content analysis](docs/cookbook/content-analysis.md), [batch processing](docs/cookbook/batch-processing.md) |
| [API Reference](docs/reference/api.md) | Complete reference for every public export |

## Contributing

```bash
git clone https://github.com/matt-house-e/accrue.git
cd accrue
pip install -e ".[dev]"
pytest
```

## License

MIT

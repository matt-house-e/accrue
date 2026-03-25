# Core Concepts

An overview of how Accrue works. Each section links to a detailed guide where applicable.

## Pipelines and Steps

A `Pipeline` orchestrates a directed acyclic graph (DAG) of steps. There are two step types: `LLMStep` calls an LLM to produce structured fields, and `FunctionStep` runs any async callable (web search, API lookups, custom logic). Each step declares a **name**, the **fields** it produces, and optionally **depends_on** to specify which steps must complete first.

## Column-Oriented Execution

Each step processes **all rows** before the next step starts. Steps at the same DAG level with no dependencies between them run in parallel. Within a step, rows are processed concurrently, bounded by `max_workers` in the config.

```
[All rows] --> Step 1 (web search)        --> complete
[All rows] --> Step 2 (classify)          --> complete
[All rows] --> Step 3 (depends on 1 + 2)  --> complete
```

This design keeps API calls batched and makes caching straightforward.

## Field Specifications

Fields are the core abstraction. Each field spec supports up to 7 keys: `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, and `default`. Accrue uses these specs to generate system prompts automatically, enable structured outputs where the provider supports them, and validate responses with Pydantic. See the [field spec guide](../guides/) for details.

## Internal Fields

Fields prefixed with `__` (e.g., `__web_context`) are internal. They pass data between steps but are filtered from the final output automatically. Downstream steps can still access them via `prior_results`.

## Providers

OpenAI is the default provider and requires no extra configuration. Anthropic and Google are available as optional extras (`accrue[anthropic]`, `accrue[google]`). Any OpenAI-compatible API can be used by setting `base_url` on the step. For fully custom providers, implement the `LLMClient` protocol.

## Results

`Pipeline.run()` returns a `PipelineResult` containing enriched data, cost information, and any errors. The output type matches the input type: pass in a DataFrame and get a DataFrame back, pass in a list of dicts and get a list of dicts back. `CostSummary` breaks down token usage and cache hit rates per step.

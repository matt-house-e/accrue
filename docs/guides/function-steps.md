# Function Steps

FunctionStep wraps any Python callable as a pipeline step. Use it for database lookups, API calls, computations, data transformations, or anything that does not need an LLM.

## Basic usage

```python
from accrue import Pipeline, FunctionStep

async def fetch_funding(ctx):
    company = ctx.row["company"]
    # Call any API, database, scraper, etc.
    data = await crunchbase_api.get_funding(company)
    return {"total_funding": data["total"], "last_round": data["round_type"]}

pipeline = Pipeline([
    FunctionStep("funding",
        fn=fetch_funding,
        fields=["total_funding", "last_round"],
    ),
])

result = pipeline.run([{"company": "Stripe"}, {"company": "Snowflake"}])
```

The function must return a `dict[str, Any]` mapping field names to values. Only keys listed in `fields` are kept in the output; extra keys are silently dropped.

## StepContext

Your function receives a `StepContext` with four attributes:

```python
async def my_step(ctx):
    ctx.row            # dict[str, Any] — current row data
    ctx.fields         # dict[str, dict] — field specs for this step
    ctx.prior_results  # dict[str, Any] — merged outputs from dependency steps
    ctx.config         # EnrichmentConfig | None — runtime configuration
    ...
```

Use `ctx.row` for the original input data. Use `ctx.prior_results` to access outputs from upstream steps declared in `depends_on`:

```python
async def enrich(ctx):
    # Access original data
    company = ctx.row["company"]
    # Access output from a dependency step
    web_context = ctx.prior_results.get("__web_context", "")
    return {"enriched_summary": f"{company}: {web_context[:200]}"}

FunctionStep("enrich", fn=enrich, fields=["enriched_summary"], depends_on=["research"])
```

## Sync functions

Sync functions work without any special handling. Accrue runs them via `run_in_executor` so they do not block the async event loop:

```python
import requests

def lookup_status(ctx):
    resp = requests.get(f"https://api.example.com/status/{ctx.row['id']}")
    return {"status": resp.json()["status"]}

FunctionStep("lookup", fn=lookup_status, fields=["status"])
```

## Dependencies

Chain steps with `depends_on`. The pipeline resolves execution order automatically:

```python
pipeline = Pipeline([
    FunctionStep("fetch_data", fn=fetch_data, fields=["raw_data"]),
    FunctionStep("transform", fn=transform, fields=["clean_data"], depends_on=["fetch_data"]),
    LLMStep("analyze", fields={"summary": "Summarize"}, depends_on=["transform"]),
])
```

## Cache control

FunctionStep caching is on by default. Cached results are keyed by step name, row data, and prior results.

```python
# Default: caching enabled
FunctionStep("lookup", fn=lookup, fields=["status"])

# Disable for non-deterministic functions (e.g., current stock price)
FunctionStep("stock_price", fn=get_price, fields=["price"], cache=False)

# Bump version when function logic changes (invalidates existing cache)
FunctionStep("score", fn=score_v2, fields=["score"], cache_version="v2")
```

`cache_version` is included in the cache key. Changing it from `None` to `"v2"` (or from `"v2"` to `"v3"`) invalidates all cached results for that step without affecting other steps.

## Internal fields

Use the `__` prefix for inter-step data that should not appear in the final output. Internal fields are available in `prior_results` for downstream steps but are filtered from the pipeline's output DataFrame:

```python
from accrue import FunctionStep, LLMStep, Pipeline, web_search

pipeline = Pipeline([
    FunctionStep("research",
        fn=web_search("Research {company}: market position, competitors"),
        fields=["__web_context", "sources"],
    ),
    LLMStep("analyze",
        fields={"summary": "Summarize the company based on research"},
        depends_on=["research"],
    ),
])
```

Here, `__web_context` is passed to the "analyze" step via `ctx.prior_results` but does not appear as a column in the final result. The `sources` field (no `__` prefix) does appear in the output.

## Conditional execution

Use `run_if` or `skip_if` to control which rows a step processes:

```python
FunctionStep("premium_lookup",
    fn=premium_api_call,
    fields=["premium_data"],
    run_if=lambda row, prior: row.get("tier") == "enterprise",
)

FunctionStep("fallback_lookup",
    fn=basic_api_call,
    fields=["basic_data"],
    skip_if=lambda row, prior: row.get("tier") == "enterprise",
)
```

`run_if` and `skip_if` are mutually exclusive. Predicates receive `(row, prior_results)` and return a bool. Skipped rows get `None` for all fields (or the field's `default` if one is set on an LLMStep upstream).

## Full parameter reference

```python
FunctionStep(
    name="step_name",          # Unique step name
    fn=my_function,            # Sync or async callable(StepContext) -> dict
    fields=["field1", "f2"],   # Field names this step produces (list[str] only)
    depends_on=["other_step"], # Steps whose output this step needs
    cache=True,                # Enable/disable caching (default: True)
    cache_version=None,        # Bump to invalidate cache on logic changes
    run_if=None,               # Predicate: (row, prior_results) -> bool
    skip_if=None,              # Predicate: (row, prior_results) -> bool
)
```

## Gotchas

- FunctionStep only accepts `fields` as `list[str]`. Dict field specs (with prompts, types, etc.) are an LLMStep feature.
- The returned dict is filtered to declared `fields`. If your function returns `{"a": 1, "b": 2}` but `fields=["a"]`, only `a` appears in the output.
- Sync functions run in a thread pool executor. They are safe to use but cannot access asyncio primitives. If your function is I/O-heavy, prefer an async implementation for better concurrency.
- `cache_version` only affects FunctionStep. LLMStep cache keys are derived from the model, temperature, field specs, and prompt content, so they auto-invalidate when those change.

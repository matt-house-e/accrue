# Quickstart

A 5-minute walkthrough of building an enrichment pipeline with Accrue.

## 1. Define a pipeline

Create a single-step pipeline with inline field specs. Each field tells the LLM what to produce.

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
```

## 2. Run it

Pass a list of dicts (or a DataFrame). Each dict is a row to enrich.

```python
result = pipeline.run([
    {"company": "Stripe", "sector": "Fintech"},
    {"company": "Notion", "sector": "Productivity"},
])
```

## 3. Read the result

`PipelineResult` gives you the enriched data plus diagnostics.

```python
print(result.data)          # list[dict] with enriched fields
print(result.cost)          # Token usage per step
print(result.success_rate)  # 1.0 if no errors
print(result.errors)        # [] if all rows succeeded
```

## 4. Add a second step

Chain steps together with `depends_on`. Use `web_search()` to ground LLM answers in live data. Internal fields prefixed with `__` are filtered from the final output.

```python
from accrue import FunctionStep, web_search

pipeline = Pipeline([
    FunctionStep("research",
        fn=web_search("Research {company}: market position, competitors"),
        fields=["__web_context", "sources"],
    ),
    LLMStep("analyze",
        fields={
            "market_size": "Estimate TAM in billions USD",
            "competitors": {"prompt": "List top 3 competitors", "type": "List[String]"},
        },
        depends_on=["research"],
    ),
])
```

## 5. Enable caching

Avoid redundant API calls on re-runs. Cached results are stored in a local SQLite database.

```python
from accrue import EnrichmentConfig

result = pipeline.run(data, config=EnrichmentConfig(enable_caching=True))
```

## Next steps

- [Core Concepts](core-concepts.md) -- understand the execution model
- [Guides](../guides/) -- field specs, providers, conditional steps, batch API

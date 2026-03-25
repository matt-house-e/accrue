# Conditional Steps

Conditional steps let you run or skip LLM calls on a per-row basis. Use them to avoid wasting API calls on rows that do not need processing.

## run_if -- only run for matching rows

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("classify",
        fields={"tier": "Classify as 'enterprise' or 'smb'"},
    ),
    LLMStep("deep_analysis",
        fields={"report": "Write detailed analysis"},
        depends_on=["classify"],
        run_if=lambda row, prior: prior.get("tier") == "enterprise",
    ),
])
```

Rows where the predicate returns `False` are skipped. Skipped rows receive default values (see below).

## skip_if -- skip matching rows

```python
LLMStep("analyze",
    fields={"summary": "Summarize business"},
    skip_if=lambda row, prior: prior.get("category") == "spam",
)
```

Rows where the predicate returns `True` are skipped. This is the inverse of `run_if`.

## Default values for skipped rows

Skipped rows get `None` for each field unless you specify a `default` in the field spec:

```python
LLMStep("premium_analysis",
    fields={
        "detailed_report": {
            "prompt": "Write a comprehensive market analysis",
            "default": "Analysis not available for this tier",
        },
    },
    depends_on=["classify"],
    run_if=lambda row, prior: prior.get("tier") == "enterprise",
)
```

Rows that fail the `run_if` predicate will have `detailed_report` set to `"Analysis not available for this tier"` instead of `None`.

## Predicate signature

Both `run_if` and `skip_if` accept a callable with this signature:

```python
def predicate(row: dict, prior_results: dict) -> bool:
    ...
```

- `row` is the original input row as a dict.
- `prior_results` is a dict of field values produced by upstream steps (those listed in `depends_on`).
- Return `True` or `False`. Any truthy/falsy value is coerced to bool.

Async callables are also supported:

```python
async def check_eligibility(row, prior):
    return prior.get("score", 0) > 0.8

LLMStep("enrich",
    fields={"details": "Get details"},
    run_if=check_eligibility,
)
```

## Rules

- `run_if` and `skip_if` are **mutually exclusive** on the same step. Setting both raises a `PipelineError`.
- Predicates execute **before cache checks**. Skipped rows never hit the cache or the API.
- Both `LLMStep` and `FunctionStep` support `run_if` and `skip_if`.
- Predicate errors are treated as row errors and follow the `on_error` config setting (`"continue"` or `"raise"`).

## Common patterns

**Two-tier enrichment:**

```python
pipeline = Pipeline([
    LLMStep("classify",
        fields={"tier": "Classify as 'enterprise', 'mid-market', or 'smb'"},
    ),
    LLMStep("basic_summary",
        fields={"summary": "One-sentence summary"},
        depends_on=["classify"],
    ),
    LLMStep("deep_dive",
        fields={
            "competitive_analysis": {
                "prompt": "Detailed competitive landscape",
                "default": "Not available",
            },
            "market_report": {
                "prompt": "Full market report",
                "default": "Not available",
            },
        },
        depends_on=["classify"],
        run_if=lambda row, prior: prior.get("tier") == "enterprise",
    ),
])
```

**Filtering bad data:**

```python
LLMStep("enrich",
    fields={"summary": "Summarize this company"},
    skip_if=lambda row, prior: not row.get("company_name"),
)
```

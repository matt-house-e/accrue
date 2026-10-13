# Field Specifications

Accrue uses a 7-key field specification system to tell the LLM exactly what to extract from each row. Field specs control the prompt, expected type, output format, allowed values, examples, and fallback behavior.

## Two input forms

### Shorthand list

Pass a list of field names when prompts come from an external source (CSV column headers, a separate field spec file, etc.):

```python
LLMStep("classify", fields=["industry", "employee_count"], model="gpt-4.1-mini")
```

### Inline dict

Pass a dict mapping field names to specs. A string value is shorthand for `{"prompt": "..."}`:

```python
LLMStep("analyze", fields={
    "market_size": "Estimate total addressable market",
    "risk_level": {
        "prompt": "Assess investment risk",
        "type": "String",
        "enum": ["Low", "Medium", "High"],
        "default": "Unknown",
    },
})
```

## The 7 keys

Every key except `prompt` is optional. Unknown keys are rejected.

### prompt (required)

The extraction instruction for this field. Be specific about what you want.

```python
"market_size": {"prompt": "Estimate total addressable market in USD"}
```

### type

Expected data type. Default: `"String"`.

Valid values: `"String"`, `"Number"`, `"Boolean"`, `"Date"`, `"List[String]"`, `"JSON"`.

```python
"employee_count": {"prompt": "Number of employees", "type": "Number"}
"is_public": {"prompt": "Is the company publicly traded?", "type": "Boolean"}
"competitors": {"prompt": "List main competitors", "type": "List[String]"}
```

### format

Output format pattern. Guides the LLM on how to structure the value.

```python
"revenue": {"prompt": "Annual revenue", "type": "String", "format": "$X.XB"}
"founded": {"prompt": "Date company was founded", "type": "Date", "format": "YYYY-MM-DD"}
"rating": {"prompt": "Overall quality rating", "type": "String", "format": "X/10"}
```

### enum

Constrained list of allowed values. The LLM must pick from this set.

```python
"risk_level": {
    "prompt": "Assess investment risk",
    "enum": ["Low", "Medium", "High"],
}
```

### examples

Good output examples showing the expected style and level of detail.

```python
"competitive_landscape": {
    "prompt": "Describe the competitive landscape",
    "examples": [
        "High - Strong competition from AWS, Azure, and GCP with similar offerings",
        "Low - Niche market with few direct competitors",
    ],
}
```

### bad_examples

Anti-patterns the LLM should avoid. Useful for steering away from vague or unhelpful outputs.

```python
"risk_assessment": {
    "prompt": "Assess investment risk with reasoning",
    "bad_examples": ["It's risky", "N/A", "Depends on the market"],
}
```

### default

Fallback value when data is insufficient. Replaces LLM refusals like "N/A", "Unable to determine", "Not available", and similar phrases. The replacement happens in Python after the LLM responds, not in the prompt.

```python
"headquarters": {
    "prompt": "Company headquarters city",
    "default": "Unknown",
}
```

Set `default` to `None` explicitly if you want missing data to appear as `None` rather than a refusal string:

```python
"ipo_date": {
    "prompt": "IPO date if publicly traded",
    "type": "Date",
    "format": "YYYY-MM-DD",
    "default": None,
}
```

## Full example

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("company_analysis",
        fields={
            "market_size": {
                "prompt": "Estimate total addressable market",
                "type": "String",
                "format": "$X.XB",
                "examples": ["$4.2B", "$850M"],
            },
            "risk_level": {
                "prompt": "Assess investment risk based on market position and financials",
                "enum": ["Low", "Medium", "High"],
                "examples": ["High - Strong competition from AWS, Azure, and GCP"],
                "bad_examples": ["It's risky", "Depends"],
            },
            "employee_count": {
                "prompt": "Current number of employees",
                "type": "Number",
                "default": None,
            },
            "competitors": {
                "prompt": "List top 3 competitors by market share",
                "type": "List[String]",
            },
            "founded_date": {
                "prompt": "Date the company was founded",
                "type": "Date",
                "format": "YYYY-MM-DD",
                "default": "Unknown",
            },
        },
        model="gpt-4.1-mini",
    ),
])

result = pipeline.run([{"company": "Snowflake", "sector": "Cloud Data"}])
```

## Structured outputs

When using OpenAI with inline dict fields, Accrue auto-enables `json_schema` with `strict: true`. This gives you guaranteed valid JSON matching your field specs.

For other providers or OpenAI-compatible APIs (via `base_url`), Accrue falls back to `json_object` mode.

**Auto-detection rules:**

| Scenario | Mode |
|----------|------|
| OpenAI (native) + dict fields | `json_schema` |
| Anthropic + dict fields | `json_schema` (constrained decoding) |
| Google + dict fields | `json_schema` |
| OpenAI with `base_url` | `json_object` |
| List fields (no specs) | `json_object` |
| Custom `schema=` set | `json_object` |

Override auto-detection explicitly:

```python
LLMStep("analyze", fields={...}, structured_outputs=True)   # Force json_schema
LLMStep("analyze", fields={...}, structured_outputs=False)   # Force json_object
```

## Custom Pydantic schemas

For full control over response validation, pass a Pydantic model with `schema=`:

```python
from pydantic import BaseModel

class CompanyAnalysis(BaseModel):
    market_size: str
    risk_level: str
    confidence: float

LLMStep("analyze",
    fields=["market_size", "risk_level", "confidence"],
    schema=CompanyAnalysis,
)
```

When `schema` is set, structured output auto-detection defaults to `json_object` (since Accrue cannot infer `json_schema` from an arbitrary Pydantic model). Force it with `structured_outputs=True` if your provider supports it.

### Custom schemas and structured outputs: specs are sent twice

Field specs reach the model in two places: the `<field_specifications>` block of the system prompt, and the `description` of each property in the JSON schema. Under `structured_outputs=True` the constrained decoder reads the schema copy right next to the value it is writing, so that copy is the one that steers the value.

A built-in schema (no `schema=`) carries both. A custom `schema=` is used exactly as you wrote it, so it carries the spec text only if you put it there. Without it the model still sees the system-prompt copy, but values can drift, for example picking the wrong clause for a field that has a plausible distractor.

Accrue logs one warning per step listing the top-level fields that match a spec but whose schema description is missing or lacks the spec's prompt. To fix it, wrap the schema:

```python
from accrue import describe_from_specs

schema = describe_from_specs(CompanyAnalysis, fields)
LLMStep("analyze", fields=fields, schema=schema, structured_outputs=True)
```

`describe_from_specs` returns a subclass with each matching field's description set to the prompt (plus format and examples), the same text a built-in schema would carry. A field that already contains the spec's prompt is left alone. Accrue never rewrites your schema on its own, because the schema is the contract you asked it to enforce.

## System prompt tiers

Accrue builds the prompt automatically from field specs and row data. You can inject context or replace the instruction portion entirely.

The system message holds only what is constant for the step — instructions and field specifications — so providers can cache it across rows. Row data and prior-step results are sent in the user message. See [prompt caching](providers.md#anthropic) for why that split matters to your bill.

**Tier 1 (default):** Auto-generated from field specs. No configuration needed. The prompt includes the field names, types, formats, enums, examples, and bad examples in a structured format.

**Tier 2: `system_prompt_header`** -- injected as a context section within the auto-generated prompt. Use this to add domain knowledge or instructions without losing the field spec formatting:

```python
LLMStep("analyze",
    fields={...},
    system_prompt_header="You are analyzing B2B SaaS companies for a venture capital firm. Focus on ARR growth and net revenue retention.",
)
```

**Tier 3: `system_prompt`** -- fully replaces the auto-generated prompt. Use this when the dynamic prompt builder does not fit your needs. You are responsible for instructing the LLM to return JSON matching your field names:

```python
LLMStep("analyze",
    fields=["sentiment", "confidence"],
    system_prompt="You are a sentiment analysis engine. Return JSON with 'sentiment' (positive/negative/neutral) and 'confidence' (0-1 float).",
)
```

Note: `system_prompt_header` is ignored when `system_prompt` is set.

## Gotchas

- Field specs are validated at construction time. A typo like `"typ": "Number"` raises immediately (thanks to `extra="forbid"` on the FieldSpec model).
- The `default` key only triggers on LLM refusals (e.g., "N/A", "Unable to determine"). If the LLM returns an actual value, `default` is not applied.
- Structured outputs (`json_schema`) are disabled automatically when grounding is active on Anthropic and Google, because those providers do not support tool use with structured output simultaneously.
- When using list fields (`["field1", "field2"]`), Accrue cannot auto-generate `json_schema` because there are no specs to derive a schema from. Use dict fields for structured outputs.

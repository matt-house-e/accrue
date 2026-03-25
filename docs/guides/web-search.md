# Web Search

Accrue offers two approaches to web-augmented enrichment: the `web_search()` factory for explicit search-then-analyze pipelines, and the `grounding` parameter for provider-native search integrated into a single LLM call.

## Approach 1: web_search() factory

`web_search()` returns an async callable for use with FunctionStep. It calls the OpenAI Responses API with web search enabled, returning the search context and source URLs.

### Minimal example

```python
from accrue import Pipeline, FunctionStep, LLMStep, web_search

pipeline = Pipeline([
    FunctionStep("research",
        fn=web_search("Research {company}: market position, recent news"),
        fields=["__web_context", "sources"],
    ),
    LLMStep("analyze",
        fields={"summary": "Summarize findings from the research"},
        depends_on=["research"],
    ),
])

result = pipeline.run([{"company": "Snowflake"}])
```

The `__web_context` field (double underscore prefix) is passed to the LLMStep via `prior_results` but filtered from the final output. The `sources` field contains the cited URLs and appears in the output.

### Parameters

```python
web_search(
    query="Research {company}: {topic}",    # Template with {field} placeholders
    model="gpt-4.1-mini",                   # Model for the search call
    search_context_size="medium",            # "low", "medium" (default), "high"
    api_key=None,                            # Falls back to OPENAI_API_KEY
    include_sources=True,                    # Extract URL citations
    user_location={"country": "US", "city": "New York"},  # Geographic bias
    allowed_domains=["crunchbase.com", "sec.gov"],         # Restrict to these domains
    tool_type="web_search",                  # "web_search" (GA) or "web_search_preview"
)
```

**query**: Template string. Placeholders like `{company}` are filled from `ctx.row` and `ctx.prior_results`. Raises `ValueError` if a placeholder references a missing field.

**search_context_size**: Controls how much search result context is included. `"low"` is cheapest, `"high"` gives the most context for complex research queries.

**allowed_domains**: Restricts search results to specific domains (up to 100). Only works with `tool_type="web_search"` (the default GA endpoint).

**tool_type**: `"web_search"` is the GA endpoint at $10/1K calls. `"web_search_preview"` is the legacy endpoint at $25/1K calls for non-reasoning models. GA supports domain filtering.

### Return value

```python
{"__web_context": str, "sources": list[str]}
```

On API errors (rate limits, timeouts), the function degrades gracefully and returns `{"__web_context": "", "sources": []}`. Downstream LLM steps still work using the original row data.

## Approach 2: grounding parameter

The `grounding` parameter on LLMStep enables provider-native web search. The LLM decides when and what to search as part of its response generation.

### Minimal example

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("research",
        fields={"summary": "Summarize recent news about this company"},
        grounding=True,
    ),
])
```

### Configuration options

Pass `True` for defaults, or a dict (or `GroundingConfig`) for fine-grained control:

```python
from accrue import LLMStep

# Simple: provider decides everything
LLMStep("research", fields={...}, grounding=True)

# Configured: restrict domains and location
LLMStep("research",
    fields={"summary": "Summarize recent news"},
    grounding={
        "allowed_domains": ["crunchbase.com", "sec.gov", "bloomberg.com"],
        "blocked_domains": ["reddit.com", "quora.com"],
        "user_location": {"country": "GB", "city": "London"},
        "max_searches": 3,
    },
)

# Using GroundingConfig directly
from accrue import GroundingConfig

LLMStep("research",
    fields={"summary": "Summarize recent news"},
    grounding=GroundingConfig(
        allowed_domains=["crunchbase.com"],
        user_location={"country": "US"},
        provider_kwargs={"search_context_size": "high"},  # OpenAI-specific
    ),
)
```

### Provider support

Not all grounding options are supported by every provider. Unsupported options are silently ignored (with a warning logged).

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| `allowed_domains` | Yes | Yes | No (warns) |
| `blocked_domains` | No (warns) | Yes | Yes |
| `user_location` | Yes | Yes | Yes |
| `max_searches` | Ignored | Yes | Ignored |
| `provider_kwargs` | Merged into tool config | Merged into tool config | Merged into tool config |

### Citations

When grounding is enabled, source citations are automatically injected into the step output under the `sources` field. Each citation includes `url`, `title`, and `snippet`.

```python
# Default: citations in "sources" field
LLMStep("research", fields={...}, grounding=True)
# Output row includes: {"summary": "...", "sources": [{"url": "...", "title": "...", "snippet": "..."}]}

# Custom field name
LLMStep("research", fields={...}, grounding=True, sources_field="refs")

# Disable citation injection entirely
LLMStep("research", fields={...}, grounding=True, sources_field=None)
```

The `sources_field` is not included in the cache key. Changing it does not invalidate cached results.

## When to use which

**Use `web_search()`** when you want:
- Separate search and analysis steps (cache search results independently of LLM analysis)
- Full control over the search query via templates
- To inspect or transform search results before passing to the LLM
- Consistent behavior across providers (always uses OpenAI Responses API for search)

**Use `grounding`** when you want:
- Simpler setup (one step instead of two)
- The LLM to decide when and what to search based on context
- Provider-native search (each provider uses its own search infrastructure)
- Citations automatically attached to the response

## Combining both approaches

You can use `web_search()` for broad research and `grounding` for targeted follow-ups:

```python
pipeline = Pipeline([
    FunctionStep("broad_research",
        fn=web_search("Latest news and developments for {company}"),
        fields=["__web_context", "initial_sources"],
    ),
    LLMStep("deep_analysis",
        fields={
            "market_position": "Analyze current market position",
            "recent_developments": "Key developments in the last 6 months",
        },
        depends_on=["broad_research"],
        grounding=True,  # LLM can search for additional details
    ),
])
```

## Gotchas

- `web_search()` always uses OpenAI (Responses API), regardless of what provider your LLMStep uses. You need a valid `OPENAI_API_KEY` even if your analysis step uses Anthropic or Google.
- `grounding` and `batch=True` are mutually exclusive. Batch APIs do not support tool use (web search). You will get an error at step construction if both are set.
- Structured outputs (`json_schema`) are disabled automatically when grounding is active on Anthropic and Google. Accrue falls back to `json_object` mode.
- The `sources_field` name must not conflict with a declared field name. If your fields dict includes a key called `"sources"`, either rename it or set `sources_field` to a different name (or `None`).
- Grounding config (domains, location, max_searches) is included in the cache key. Changing these settings invalidates cached results. But `sources_field` and `provider_kwargs` are not in the cache key.

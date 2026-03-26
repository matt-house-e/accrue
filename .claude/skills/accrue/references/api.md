# Accrue API Reference

## Public Exports

```python
from accrue import (
    Pipeline, PipelineResult, LLMStep, FunctionStep,
    EnrichmentConfig, EnrichmentHooks, FieldSpec, GroundingConfig,
    Step, StepContext, StepResult, web_search,
    CostSummary, RowError, EnrichmentError,
    BatchCapableLLMClient, BatchRequest, BatchResult,
)
```

## Pipeline

```python
Pipeline(steps: list[Step])
```
- Validates: no duplicate names, no circular deps, no missing deps
- Builds execution DAG via topological sort — steps at same level run in parallel

### Pipeline.run()

```python
def run(
    self,
    data: pd.DataFrame | list[dict[str, Any]],
    config: EnrichmentConfig | None = None,
    hooks: EnrichmentHooks | None = None,
) -> PipelineResult
```

- Sync wrapper around `asyncio.run(self.run_async(...))`
- Input must be a `pd.DataFrame` or `list[dict]` — load CSVs with `pd.read_csv()` first
- Returns `PipelineResult`

### PipelineResult

```python
@dataclass
class PipelineResult:
    data: pd.DataFrame | list[dict[str, Any]]
    cost: CostSummary     # .total_tokens, .input_tokens, .output_tokens
    errors: list[RowError] # row index + error message + step name
```

Access: `result.data`, `result.cost`, `result.errors`
Success rate: `1 - len(result.errors) / len(result.data)`

## LLMStep

```python
LLMStep(
    name: str,
    fields: list[str] | dict[str, str | dict],
    depends_on: list[str] | None = None,
    model: str = "gpt-4.1-mini",
    temperature: float | None = None,        # Overrides config; default from config is 0.2
    max_tokens: int | None = None,           # Overrides config; default from config is 4000
    system_prompt: str | None = None,        # Fully replaces auto-generated prompt
    system_prompt_header: str | None = None, # Injected into auto-generated prompt
    api_key: str | None = None,              # Provider API key
    base_url: str | None = None,             # OpenAI-compatible endpoint
    client: LLMClient | None = None,         # Pre-configured provider client
    schema: Type[BaseModel] = EnrichmentResult,  # Custom Pydantic validation
    max_retries: int = 2,
    cache: bool = True,                      # Per-step cache toggle
    structured_outputs: bool | None = None,  # None = auto-detect
    grounding: bool | dict | GroundingConfig | None = None,
    sources_field: str | None = "sources",   # Where citations go; None to suppress
    run_if: Callable | None = None,          # (row, prior_results) -> bool
    skip_if: Callable | None = None,         # (row, prior_results) -> bool; mutually exclusive with run_if
    batch: bool = False,                     # Use provider batch API
    provider_kwargs: dict[str, Any] | None = None,  # Escape hatch; merged into API call; NOT in cache key
)
```

### System Prompt Tiers

1. **Default** — Auto-generated from field specs + row data. No config needed.
2. **`system_prompt_header`** — Injected as context within auto-generated prompt. Use for domain knowledge.
3. **`system_prompt`** — Fully replaces auto-generated prompt. You must instruct LLM to return JSON matching field names.

`system_prompt_header` is ignored when `system_prompt` is set.

## FunctionStep

```python
FunctionStep(
    name: str,
    fn: Callable,           # async def fn(row: dict, context: StepContext) -> dict
    fields: list[str],      # Field names this step produces
    depends_on: list[str] | None = None,
    cache: bool = True,
    cache_version: str | None = None,  # Bump to invalidate cache
    run_if: Callable | None = None,
    skip_if: Callable | None = None,
)
```

Use for: data cleaning, API calls, computations, validation — anything that's not an LLM call.

## EnrichmentConfig

```python
EnrichmentConfig(
    # LLM
    max_tokens: int = 4000,
    temperature: float = 0.2,
    # Concurrency
    max_workers: int = 10,
    # Fields
    overwrite_fields: bool = False,
    # Reliability
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    on_error: str = "continue",        # "continue" | "raise"
    # Logging
    log_level: str = "INFO",
    enable_progress_bar: bool = True,
    # Checkpointing
    enable_checkpointing: bool = False,
    checkpoint_dir: str | None = None,
    auto_resume: bool = True,
    # Caching
    enable_caching: bool = False,
    cache_ttl: int = 3600,             # Seconds
    cache_dir: str = ".accrue",
    checkpoint_interval: int = 0,
    # Batch API
    batch_poll_interval: float = 60.0,
    batch_timeout: float = 86400.0,    # 24 hours
    batch_max_requests: int = 50000,
)
```

**Important:** `enable_caching` on config is the global toggle. `cache` on LLMStep/FunctionStep is per-step. Both must be True for caching to work. Steps default to `cache=True`, so typically you just set `enable_caching=True` on config.

## FieldSpec (7 keys)

```python
class FieldSpec(BaseModel):
    prompt: str                    # Required. Extraction instruction.
    type: str = "String"           # String | Number | Boolean | Date | List[String] | JSON
    format: str | None = None      # Output format pattern, e.g. "$X.XB", "YYYY-MM-DD"
    enum: list[str] | None = None  # Constrained values. LLM must pick from this set.
    examples: list[str] | None     # Good output examples
    bad_examples: list[str] | None # Anti-patterns to avoid
    default: Any = None            # Fallback for LLM refusals ("N/A", "Unable to determine")
```

**Shorthand forms:**
- String value → `{"prompt": "the string"}`
- Dict value → Full FieldSpec
- List of strings → Field names only (no specs, prompts come from context)

## GroundingConfig

```python
GroundingConfig(
    allowed_domains: list[str] | None = None,   # Restrict web search to these domains
    blocked_domains: list[str] | None = None,
    user_location: dict[str, str] | None = None, # {"country": "US", "city": "San Francisco"}
    max_searches: int | None = None,
    provider_kwargs: dict[str, Any] | None = None,
)
```

Shorthand: `grounding=True` → `GroundingConfig()` with all defaults.

**Provider behavior:**
- OpenAI: Responses API web search tool
- Anthropic: Citations (document-grounded, not web search)
- Google: Google Search grounding
- **Grounding disables structured outputs on Anthropic/Google.** OpenAI is unaffected.

## EnrichmentHooks

```python
EnrichmentHooks(
    on_pipeline_start: Callable[[PipelineStartEvent], Any] | None = None,
    on_pipeline_end: Callable[[PipelineEndEvent], Any] | None = None,
    on_step_start: Callable[[StepStartEvent], Any] | None = None,
    on_step_end: Callable[[StepEndEvent], Any] | None = None,
    on_row_complete: Callable[[RowCompleteEvent], Any] | None = None,
)
```

Sync and async callables both work. Hook errors are caught and logged — never crash the pipeline.

## Conditional Steps

```python
# Predicate signature
def predicate(row: dict, prior_results: dict) -> bool

# Usage
LLMStep("deep_dive",
    fields={...},
    depends_on=["classify"],
    run_if=lambda row, prior: prior.get("tier") == "enterprise",
)
```

- `run_if` and `skip_if` are mutually exclusive
- Predicates run before cache checks — skipped rows never hit cache or API
- Skipped rows get `None` or the field's `default` value

## Internal Fields (__ prefix)

Fields prefixed with `__` are passed between steps but filtered from final output:

```python
LLMStep("research",
    fields={"__company_context": "Detailed company background for later steps"},
)
LLMStep("qualify",
    fields={"icp_fit": "Qualify based on company context"},
    depends_on=["research"],
    # __company_context is available in prior_results but won't appear in output
)
```

## web_search() Factory

```python
from accrue import web_search

step = web_search(
    name="find_info",
    query_template="What does {company_name} do?",
    fields={"answer": "Company description"},
    model="gpt-4.1-mini",
)
```

Convenience factory that returns an LLMStep with grounding pre-configured.

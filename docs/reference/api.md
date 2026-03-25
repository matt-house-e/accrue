# API Reference

Complete reference for every public export in the `accrue` package. All types use Python 3.10+ syntax (`X | None` instead of `Optional[X]`).

---

## Pipeline

```python
from accrue import Pipeline
```

### `Pipeline(steps)`

```python
Pipeline(steps: list[Step])
```

Build a pipeline from an ordered list of steps. Validates on construction:

- No duplicate step names.
- Every `depends_on` reference points to an existing step.
- No cycles in the dependency graph.

Raises `PipelineError` on validation failure.

### `pipeline.run(data, config?, hooks?)`

```python
pipeline.run(
    data: pd.DataFrame | list[dict[str, Any]],
    config: EnrichmentConfig | None = None,
    hooks: EnrichmentHooks | None = None,
) -> PipelineResult
```

Synchronous entry point. Wraps `asyncio.run()` internally. Raises `RuntimeError` if called from inside an existing event loop (use `run_async` in that case). Output type matches input type.

### `pipeline.run_async(data, config?, hooks?)`

```python
await pipeline.run_async(
    data: pd.DataFrame | list[dict[str, Any]],
    config: EnrichmentConfig | None = None,
    hooks: EnrichmentHooks | None = None,
) -> PipelineResult
```

Async variant. Use from FastAPI, Jupyter notebooks, or any async context.

### `pipeline.runner(config?)`

```python
pipeline.runner(config: EnrichmentConfig | None = None) -> Enricher
```

Returns a reusable `Enricher` instance bound to this pipeline and config. Use when you need repeated execution with checkpointing or want to manage the runner lifecycle in a server context.

### `pipeline.clear_cache(step?, cache_dir?)`

```python
pipeline.clear_cache(
    step: str | None = None,
    cache_dir: str = ".accrue",
) -> int
```

Delete cached results from the SQLite cache. Pass `step` to scope deletion to a single step, or `None` to delete all entries. Returns the number of entries deleted.

### `pipeline.step_names`

```python
@property
step_names: list[str]
```

All step names in execution order (topological).

### `pipeline.execution_levels`

```python
@property
execution_levels: list[list[str]]
```

Topological execution levels. Returns a read-only copy. Steps within the same level run in parallel.

### `pipeline.get_step(name)`

```python
pipeline.get_step(name: str) -> Step
```

Look up a step by name. Raises `KeyError` if the name does not exist in this pipeline.

---

## PipelineResult

```python
from accrue import PipelineResult
```

Returned by `Pipeline.run()` and `Pipeline.run_async()`.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `pd.DataFrame \| list[dict[str, Any]]` | Enriched output. Matches the input type. |
| `cost` | `CostSummary` | Aggregated token usage across all steps. |
| `errors` | `list[RowError]` | Per-row errors. Empty if all rows succeeded. |

| Property | Type | Description |
|----------|------|-------------|
| `success_rate` | `float` | Fraction of rows that completed without error (0.0--1.0). |
| `has_errors` | `bool` | `True` if any rows produced errors. |

---

## LLMStep

```python
from accrue import LLMStep
```

Calls an LLM provider to produce structured enrichment values. Provider-agnostic via the `LLMClient` protocol (OpenAI by default).

### Constructor

```python
LLMStep(
    name: str,
    fields: list[str] | dict[str, str | dict],
    depends_on: list[str] | None = None,
    model: str = "gpt-4.1-mini",
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    system_prompt_header: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client: LLMClient | None = None,
    schema: Type[BaseModel] = EnrichmentResult,
    max_retries: int = 2,
    cache: bool = True,
    structured_outputs: bool | None = None,
    grounding: bool | dict | GroundingConfig | None = None,
    sources_field: str | None = "sources",
    run_if: Callable | None = None,
    skip_if: Callable | None = None,
    batch: bool = False,
    provider_kwargs: dict[str, Any] | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `name` | Unique step name used in logs, cache keys, and `depends_on` references. |
| `fields` | Fields this step produces. `list[str]` for names only; `dict[str, str \| dict]` for inline field specs (string values are shorthand for `{"prompt": value}`). |
| `depends_on` | Names of steps whose outputs this step needs. Resolved as DAG edges. |
| `model` | Model identifier passed to the LLM provider. |
| `temperature` | Sampling temperature. Falls back to `config.temperature`, then `0.2`. |
| `max_tokens` | Maximum response tokens. Falls back to `config.max_tokens`, then `4000`. |
| `system_prompt` | Tier 3 -- fully replaces the auto-generated system prompt. |
| `system_prompt_header` | Tier 2 -- injected as a `# Context` section in the dynamic prompt. Ignored when `system_prompt` is set. |
| `api_key` | Provider API key. Falls back to the relevant environment variable (e.g. `OPENAI_API_KEY`). |
| `base_url` | OpenAI-compatible base URL (Ollama, Groq, etc.). Disables structured-output auto-detection. |
| `client` | Pre-configured `LLMClient` instance. Overrides `api_key` and `base_url`. |
| `schema` | Pydantic model for response validation. Default `EnrichmentResult` works with dynamic field specs. |
| `max_retries` | Parse/validation retry attempts per API call (inner retry loop). |
| `cache` | Enable input-hash caching for this step. |
| `structured_outputs` | Override structured-output auto-detection. `True` forces `json_schema`; `False` forces `json_object`; `None` auto-detects. |
| `grounding` | Enable provider-level web search. `True` for defaults, `dict` or `GroundingConfig` for fine-grained control. `None`/`False` disables. |
| `sources_field` | Output field name for grounding citations. Set to `None` to disable citation injection. Not in cache key. |
| `run_if` | Predicate `(row, prior_results) -> bool`. Step only runs for rows where it returns `True`. Mutually exclusive with `skip_if`. |
| `skip_if` | Predicate `(row, prior_results) -> bool`. Step is skipped for rows where it returns `True`. Mutually exclusive with `run_if`. |
| `batch` | Use provider Batch API for this step. Requires `BatchCapableLLMClient`. Mutually exclusive with `grounding`. |
| `provider_kwargs` | Extra kwargs merged into the provider API call. Escape hatch for provider-specific features (e.g. `{"thinking": {"type": "adaptive"}}`). Not in cache key. |

### `step.build_messages(ctx)`

```python
step.build_messages(ctx: StepContext) -> tuple[list[dict[str, str]], dict[str, Any]]
```

Build messages and call kwargs for a single row. Returns `(messages, call_kwargs)` where `call_kwargs` contains `model`, `temperature`, `max_tokens`, `response_format`, `tools`, and `provider_kwargs`. Used by both realtime and batch execution paths.

### `step.parse_response(response)`

```python
step.parse_response(response: LLMResponse) -> StepResult
```

Parse and validate an LLM response. Performs JSON decoding, Pydantic validation, field filtering, default enforcement, and citation injection. Raises `json.JSONDecodeError` or `pydantic.ValidationError` on invalid responses.

### `step.is_batch_eligible`

```python
@property
is_batch_eligible: bool
```

`True` when `batch=True` and the resolved client implements `BatchCapableLLMClient`.

---

## FunctionStep

```python
from accrue import FunctionStep
```

Wraps any sync or async callable as a pipeline step. The callable receives a `StepContext` and must return `dict[str, Any]`.

### Constructor

```python
FunctionStep(
    name: str,
    fn: Callable,
    fields: list[str],
    depends_on: list[str] | None = None,
    cache: bool = True,
    cache_version: str | None = None,
    run_if: Callable | None = None,
    skip_if: Callable | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `name` | Unique step name used in logs, cache keys, and `depends_on` references. |
| `fn` | Sync or async callable. Receives a `StepContext`, returns `dict[str, Any]` mapping field names to values. |
| `fields` | Field names this step produces (`list[str]` only -- no dict field specs). |
| `depends_on` | Names of steps whose outputs this step needs. |
| `cache` | Enable input-hash caching for this step. |
| `cache_version` | Bump to invalidate cached results when function logic changes (e.g. `"v2"`). |
| `run_if` | Predicate `(row, prior_results) -> bool`. Mutually exclusive with `skip_if`. |
| `skip_if` | Predicate `(row, prior_results) -> bool`. Mutually exclusive with `run_if`. |

---

## Step Protocol

```python
from accrue import Step
```

The protocol all steps must satisfy. Implementations can be plain classes -- no inheritance required.

```python
@runtime_checkable
class Step(Protocol):
    name: str
    fields: list[str]
    depends_on: list[str]

    async def run(self, ctx: StepContext) -> StepResult: ...
```

---

## StepContext

```python
from accrue import StepContext
```

Immutable dataclass passed to each step's `run()` method.

```python
@dataclass(frozen=True)
class StepContext:
    row: dict[str, Any]
    fields: dict[str, dict[str, Any]]
    prior_results: dict[str, Any]
    config: EnrichmentConfig | None = None
```

| Field | Description |
|-------|-------------|
| `row` | Original row data as `dict[str, Any]`. |
| `fields` | Resolved field specs for this step only. Always `dict[str, dict]`, even when the constructor received `list[str]`. |
| `prior_results` | Merged outputs from all dependency steps for the current row. |
| `config` | Optional `EnrichmentConfig` for reading runtime settings. |

---

## StepResult

```python
from accrue import StepResult
```

Output from a single step execution.

```python
@dataclass
class StepResult:
    values: dict[str, Any]
    usage: UsageInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

| Field | Description |
|-------|-------------|
| `values` | Field name to produced value. |
| `usage` | Token usage (LLM steps only). |
| `metadata` | Arbitrary metadata for logging/debugging (e.g. `raw_response`, `attempts`). |

---

## EnrichmentConfig

```python
from accrue import EnrichmentConfig
```

Configuration dataclass for pipeline execution. All fields have sensible defaults.

```python
@dataclass
class EnrichmentConfig:
    # LLM
    max_tokens: int = 4000
    temperature: float = 0.2

    # Concurrency
    max_workers: int = 10

    # Fields
    overwrite_fields: bool = False

    # Reliability
    max_retries: int = 3
    retry_base_delay: float = 1.0
    on_error: str = "continue"

    # Logging
    log_level: str = "INFO"
    enable_progress_bar: bool = True

    # Checkpointing
    enable_checkpointing: bool = False
    checkpoint_dir: str | None = None
    auto_resume: bool = True
    checkpoint_interval: int = 0

    # Caching
    enable_caching: bool = False
    cache_ttl: int = 3600
    cache_dir: str = ".accrue"

    # Batch API
    batch_poll_interval: float = 60.0
    batch_timeout: float = 86400.0
    batch_max_requests: int = 50000
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_tokens` | `4000` | Maximum tokens for LLM output. |
| `temperature` | `0.2` | LLM temperature. Low values (0.1--0.3) are best for structured enrichment. |
| `max_workers` | `10` | Concurrent rows per step (semaphore bound). Production: 20--30. |
| `overwrite_fields` | `False` | Whether to overwrite existing field values in the DataFrame. |
| `max_retries` | `3` | Maximum retry attempts for API errors (429, 500, timeouts). |
| `retry_base_delay` | `1.0` | Base delay in seconds for exponential backoff. Actual: `base * 2^attempt + jitter`. |
| `on_error` | `"continue"` | `"continue"` collects errors and returns partial results; `"raise"` fails fast. |
| `log_level` | `"INFO"` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `enable_progress_bar` | `True` | Show tqdm progress bar during execution. |
| `enable_checkpointing` | `False` | Save results after each step for crash recovery. |
| `checkpoint_dir` | `None` | Directory for checkpoint files. `None` uses a temp directory. |
| `auto_resume` | `True` | Automatically resume from checkpoint on re-run. |
| `checkpoint_interval` | `0` | Save partial step progress every N rows. `0` disables. |
| `enable_caching` | `False` | Enable input-hash SQLite cache to skip redundant API calls. |
| `cache_ttl` | `3600` | Cache time-to-live in seconds. |
| `cache_dir` | `".accrue"` | Directory for `cache.db`. |
| `batch_poll_interval` | `60.0` | Seconds between batch job status checks. |
| `batch_timeout` | `86400.0` | Maximum seconds to wait for a batch job (default: 24 hours). |
| `batch_max_requests` | `50000` | Maximum requests per batch submission. Larger sets are auto-chunked. |

### Presets

```python
EnrichmentConfig.for_development()   # max_workers=5, DEBUG, caching on
EnrichmentConfig.for_production()    # max_workers=30, checkpointing + caching
EnrichmentConfig.for_server()        # max_workers=30, no progress bar, WARNING
EnrichmentConfig.for_batch()         # caching + checkpointing, 24h timeout
```

| Preset | Key settings |
|--------|-------------|
| `for_development()` | `max_workers=5`, `log_level="DEBUG"`, `enable_caching=True`. Safe for Tier 1 accounts. |
| `for_production()` | `max_workers=30`, `enable_checkpointing=True`, `enable_caching=True`, `checkpoint_interval=100`, `max_retries=5`. |
| `for_server()` | `max_workers=30`, `enable_progress_bar=False`, `log_level="WARNING"`, `max_retries=5`. For async server contexts. |
| `for_batch()` | `enable_caching=True`, `enable_checkpointing=True`, `batch_poll_interval=60.0`, `batch_timeout=86400.0`, `max_retries=5`. |

---

## EnrichmentHooks

```python
from accrue import EnrichmentHooks
```

Hook container passed to `Pipeline.run()` or `run_async()`. All fields are optional. Sync and async callables both work. Hook errors are caught and logged -- they never crash the pipeline.

```python
@dataclass
class EnrichmentHooks:
    on_pipeline_start: Callable[[PipelineStartEvent], Any] | None = None
    on_pipeline_end: Callable[[PipelineEndEvent], Any] | None = None
    on_step_start: Callable[[StepStartEvent], Any] | None = None
    on_step_end: Callable[[StepEndEvent], Any] | None = None
    on_row_complete: Callable[[RowCompleteEvent], Any] | None = None
```

---

## Event Types

```python
from accrue import (
    PipelineStartEvent,
    PipelineEndEvent,
    StepStartEvent,
    StepEndEvent,
    RowCompleteEvent,
)
```

### PipelineStartEvent

Fired once at the beginning of `run_async()`.

| Field | Type |
|-------|------|
| `step_names` | `list[str]` |
| `num_rows` | `int` |
| `config` | `EnrichmentConfig` |

### PipelineEndEvent

Fired once at the end of `run_async()` (including on error).

| Field | Type |
|-------|------|
| `num_rows` | `int` |
| `total_errors` | `int` |
| `cost` | `Any` |
| `elapsed_seconds` | `float` |

### StepStartEvent

Fired before a step begins processing rows.

| Field | Type | Default |
|-------|------|---------|
| `step_name` | `str` | |
| `num_rows` | `int` | |
| `level` | `int` | |
| `execution_mode` | `str` | `"realtime"` |

### StepEndEvent

Fired after a step finishes all rows.

| Field | Type | Default |
|-------|------|---------|
| `step_name` | `str` | |
| `num_rows` | `int` | |
| `num_errors` | `int` | |
| `usage` | `Any` | |
| `elapsed_seconds` | `float` | |
| `execution_mode` | `str` | `"realtime"` |
| `batch_id` | `str \| None` | `None` |

### RowCompleteEvent

Fired after each row completes within a step.

| Field | Type | Default |
|-------|------|---------|
| `step_name` | `str` | |
| `row_index` | `int` | |
| `values` | `dict[str, Any]` | |
| `error` | `BaseException \| None` | |
| `from_cache` | `bool` | |
| `skipped` | `bool` | `False` |

---

## Schemas

### FieldSpec

```python
from accrue import FieldSpec
```

Pydantic model for the 7-key field specification. Unknown keys are rejected (`extra="forbid"`).

```python
class FieldSpec(BaseModel):
    prompt: str                           # Required. Extraction instruction.
    type: VALID_TYPES = "String"          # "String" | "Number" | "Boolean" | "Date" | "List[String]" | "JSON"
    format: str | None = None             # Output format pattern (e.g. "YYYY-MM-DD").
    enum: list[str] | None = None         # Constrained value list.
    examples: list[str] | None = None     # Good output examples.
    bad_examples: list[str] | None = None # Anti-patterns to avoid.
    default: Any = None                   # Fallback when data is insufficient.
```

Use `"default" in spec.model_fields_set` to detect whether `default` was explicitly provided.

### GroundingConfig

```python
from accrue import GroundingConfig
```

Configuration for provider-level web search grounding. Pass to `LLMStep(grounding=...)`.

```python
class GroundingConfig(BaseModel):
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict[str, str] | None = None
    max_searches: int | None = None
    provider_kwargs: dict[str, Any] | None = None
```

| Field | Description |
|-------|-------------|
| `allowed_domains` | Only include results from these domains. OpenAI and Anthropic only. |
| `blocked_domains` | Never include results from these domains. Anthropic and Google only. |
| `user_location` | Dict with keys: `country` (ISO 3166-1 alpha-2), `region`, `city`, `timezone` (IANA). |
| `max_searches` | Maximum searches per LLM call. Anthropic only (`max_uses`). |
| `provider_kwargs` | Provider-specific kwargs merged into native tool config (e.g. OpenAI `search_context_size`). |

### CostSummary

```python
from accrue import CostSummary
```

Aggregated cost/usage across all pipeline steps. Available as `PipelineResult.cost`.

```python
class CostSummary(BaseModel):
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    steps: dict[str, StepUsage] = {}
```

### StepUsage

Per-step aggregated token usage. Accessed via `CostSummary.steps["step_name"]`.

```python
class StepUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    rows_processed: int = 0
    rows_skipped: int = 0
    model: str = ""
    cache_hits: int = 0
    cache_misses: int = 0
    execution_mode: str = "realtime"
    batch_id: str | None = None
```

| Property | Type | Description |
|----------|------|-------------|
| `cache_hit_rate` | `float` | `cache_hits / (cache_hits + cache_misses)`, or `0.0` if no rows. |

### UsageInfo

```python
from accrue.schemas.base import UsageInfo
```

Token usage from a single LLM call.

```python
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
```

---

## Providers

```python
from accrue.providers import (
    LLMClient,
    LLMResponse,
    LLMAPIError,
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
)
```

### LLMClient Protocol

The protocol all LLM provider adapters must satisfy.

```python
@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
```

### BatchCapableLLMClient Protocol

```python
from accrue import BatchCapableLLMClient
```

Extended protocol for providers that support batch API operations. The pipeline checks `isinstance(client, BatchCapableLLMClient)` to decide the execution path.

```python
@runtime_checkable
class BatchCapableLLMClient(LLMClient, Protocol):
    async def submit_batch(
        self,
        requests: list[BatchRequest],
        metadata: dict[str, str] | None = None,
    ) -> str: ...

    async def poll_batch(
        self,
        batch_id: str,
        poll_interval: float = 60.0,
        timeout: float = 86400.0,
    ) -> BatchResult: ...

    async def cancel_batch(self, batch_id: str) -> None: ...
```

| Method | Description |
|--------|-------------|
| `submit_batch` | Submit a list of `BatchRequest` objects. Returns the provider batch ID. |
| `poll_batch` | Poll until the batch completes or timeout. Returns `BatchResult`. Raises `StepError` on failure/expiry. |
| `cancel_batch` | Best-effort cancel a running batch job. |

### LLMResponse

```python
@dataclass
class LLMResponse:
    content: str
    usage: UsageInfo | None = None
    citations: list[Citation] = field(default_factory=list)
```

### BatchRequest

```python
from accrue import BatchRequest
```

A single request within a batch submission.

```python
@dataclass
class BatchRequest:
    custom_id: str
    messages: list[dict[str, Any]]
    model: str
    temperature: float
    max_tokens: int
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    provider_kwargs: dict[str, Any] | None = None
```

### BatchResult

```python
from accrue import BatchResult
```

Aggregated result from a completed batch job.

```python
@dataclass
class BatchResult:
    responses: dict[str, LLMResponse] = field(default_factory=dict)
    failed_ids: list[str] = field(default_factory=list)
    batch_id: str = ""
    errors: dict[str, str] = field(default_factory=dict)
```

| Field | Description |
|-------|-------------|
| `responses` | Mapping of `custom_id` to the provider `LLMResponse`. |
| `failed_ids` | `custom_id` values for requests that failed. |
| `batch_id` | Provider batch job identifier. |
| `errors` | Mapping of `custom_id` to error message for failed requests. |

### OpenAIClient

```python
from accrue.providers import OpenAIClient

OpenAIClient(
    api_key: str | None = None,
    base_url: str | None = None,
)
```

Default adapter. Native OpenAI (no `base_url`) uses the Responses API with web search, structured outputs, and inline citations. Third-party providers with a `base_url` (Ollama, Groq, DeepSeek, Together, Fireworks, vLLM, Mistral, LM Studio) use Chat Completions for compatibility.

Implements `BatchCapableLLMClient` (native OpenAI only).

### AnthropicClient

```python
from accrue.providers import AnthropicClient

AnthropicClient(api_key: str | None = None)
```

Adapter for Anthropic Claude models. Requires `pip install accrue[anthropic]`. Falls back to `ANTHROPIC_API_KEY` env var. Supports `web_search_20250305` server tool for grounding. Automatic prompt caching via `cache_control` on system messages.

Implements `BatchCapableLLMClient` via the Anthropic Message Batches API.

### GoogleClient

```python
from accrue.providers import GoogleClient

GoogleClient(api_key: str | None = None)
```

Adapter for Google Gemini models. Requires `pip install accrue[google]`. Falls back to `GOOGLE_API_KEY` env var. Supports the `google_search` grounding tool. Structured outputs disabled when grounding is active (Gemini 2.x limitation).

Does not implement `BatchCapableLLMClient`.

### LLMAPIError

```python
from accrue.providers import LLMAPIError

LLMAPIError(
    message: str,
    *,
    status_code: int | None = None,
    retry_after: float | None = None,
    is_rate_limit: bool = False,
)
```

Provider-agnostic API error for retry logic. Wraps provider-specific errors so `LLMStep` retry logic does not need to know about specific SDKs.

| Attribute | Type | Description |
|-----------|------|-------------|
| `status_code` | `int \| None` | HTTP status code (e.g. 429, 500, 408). |
| `retry_after` | `float \| None` | Seconds to wait before retrying (from `Retry-After` header). |
| `is_rate_limit` | `bool` | `True` if this is a rate limit error. |

---

## Utilities

### `web_search()`

```python
from accrue import web_search
```

Factory returning an async callable for use with `FunctionStep`. Wraps the OpenAI Responses API web search tool.

```python
web_search(
    query: str,
    *,
    model: str = "gpt-4.1-mini",
    search_context_size: str = "medium",
    api_key: str | None = None,
    include_sources: bool = True,
    user_location: dict[str, str] | None = None,
    allowed_domains: list[str] | None = None,
    tool_type: str = "web_search",
) -> Callable[[StepContext], Awaitable[dict[str, Any]]]
```

| Parameter | Description |
|-----------|-------------|
| `query` | Template string with `{field}` placeholders, formatted with `ctx.row` and `ctx.prior_results`. |
| `model` | OpenAI model for the search call. Must support web search. |
| `search_context_size` | `"low"` / `"medium"` / `"high"` -- amount of context from search results. |
| `api_key` | OpenAI API key. Falls back to `OPENAI_API_KEY`. |
| `include_sources` | Extract URL citations from the response. |
| `user_location` | Geographic bias dict with keys: `country`, `city`, `region`, `timezone`. |
| `allowed_domains` | Restrict results to these domains. Only with `tool_type="web_search"`. |
| `tool_type` | `"web_search"` (GA, $10/1k) or `"web_search_preview"` (legacy, $25/1k). |

Returns `{"__web_context": str, "sources": list[str]}`. On API error, returns empty strings/lists (graceful degradation).

---

## Errors

### Exception Hierarchy

```
EnrichmentError
  +-- FieldValidationError
  +-- ConfigurationError
  +-- StepError
  +-- PipelineError
```

All exceptions inherit from `EnrichmentError` and are importable from `accrue`.

### EnrichmentError

```python
from accrue import EnrichmentError

EnrichmentError(
    message: str,
    row_index: int | None = None,
    field: str | None = None,
)
```

Base exception for all enrichment-related errors.

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description. |
| `row_index` | `int \| None` | Row that triggered the error. |
| `field` | `str \| None` | Field name involved. |

### FieldValidationError

```python
from accrue import FieldValidationError
```

Raised when field definitions are invalid. Same constructor as `EnrichmentError`. Common causes: unknown keys in a field spec, missing `prompt`, invalid `type` or `enum` values.

### ConfigurationError

```python
from accrue.core.exceptions import ConfigurationError
```

Raised when configuration is invalid. Same constructor as `EnrichmentError`.

### StepError

```python
from accrue import StepError
```

Raised when a pipeline step fails after exhausting retries.

```python
StepError(
    message: str,
    step_name: str | None = None,
    **kwargs,
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `step_name` | `str \| None` | Name of the step that failed. |

### PipelineError

```python
from accrue import PipelineError
```

Raised when pipeline construction or execution fails. Same constructor as `EnrichmentError`. Common causes: duplicate step names, missing dependency references, dependency cycles.

### RowError

```python
from accrue import RowError
```

Per-row failure record for partial result tracking. Dataclass, not an exception.

```python
@dataclass
class RowError:
    row_index: int
    step_name: str
    error: BaseException
    error_type: str = ""  # Auto-set to type(error).__name__
```

---

## Exports

### `from accrue import ...`

```python
from accrue import (
    # Primary API
    Pipeline,
    PipelineResult,
    LLMStep,
    FunctionStep,
    EnrichmentConfig,

    # Hooks
    EnrichmentHooks,
    PipelineStartEvent,
    PipelineEndEvent,
    StepStartEvent,
    StepEndEvent,
    RowCompleteEvent,

    # Step protocol
    Step,
    StepContext,
    StepResult,

    # Schemas
    FieldSpec,
    GroundingConfig,
    CostSummary,

    # Batch API
    BatchCapableLLMClient,
    BatchRequest,
    BatchResult,

    # Utilities
    web_search,

    # Errors
    EnrichmentError,
    FieldValidationError,
    StepError,
    PipelineError,
    RowError,

    # Internal runner (power users)
    Enricher,
)
```

### `from accrue.providers import ...`

```python
from accrue.providers import (
    LLMClient,
    LLMResponse,
    LLMAPIError,
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
)
```

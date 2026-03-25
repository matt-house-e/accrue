# Pipeline Architecture Design

> **Status**: Phases 1-5 COMPLETE. Phase 6B conditional steps (#40) + grounding (#48) COMPLETE. Phase 6A (Ship) next.
> **Version**: v1.0.0
> **Date**: March 2026

## Overview

Accrue v0.3 replaces the monolithic chain model with a composable pipeline of steps. Each step runs across all rows (column-oriented) before the next step starts. Steps declare dependencies; independent steps run in parallel.

## Step Protocol

```python
# accrue/steps/base.py

@dataclass(frozen=True)
class StepContext:
    """Immutable context passed to each step."""
    row: dict[str, Any]                # Original row data (converted from pd.Series at boundary)
    fields: dict[str, dict[str, Any]]  # Field specs for THIS step only (sliced by Pipeline)
    prior_results: dict[str, Any]      # Merged outputs from dependency steps
    config: Any = None                 # EnrichmentConfig

@dataclass
class StepResult:
    """Output from a single step execution."""
    values: dict[str, Any]              # Field name → produced value
    usage: Optional[UsageInfo] = None   # Token usage (LLM steps)
    metadata: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class Step(Protocol):
    """Protocol all steps must satisfy. Async-only to eliminate duplication."""
    name: str
    fields: list[str]          # Field names this step produces
    depends_on: list[str]      # Step names this depends on

    async def run(self, ctx: StepContext) -> StepResult: ...
```

**Why async-only**: The v0.2 codebase had ~400 lines of sync/async duplication. Writing `run()` as async-only and providing a sync wrapper at the Enricher level eliminates this entirely.

**Why Protocol, not ABC**: Allows duck typing. Users can implement steps as plain classes without inheriting anything.

## Pipeline

```python
# accrue/pipeline/pipeline.py

class Pipeline:
    def __init__(self, steps: list[Step]): ...
    def run(self, data, config=None) -> PipelineResult: ...         # DataFrame or list[dict]
    async def run_async(self, data, config=None) -> PipelineResult: ...
    def runner(self, config=None) -> Enricher: ...
    def clear_cache(self, step=None, cache_dir=".accrue") -> int: ...
    async def execute(self, rows, all_fields, config, ...) -> tuple[list[dict], list[RowError], CostSummary]: ...
```

### Construction
- Validates: no duplicate step names, no missing dependencies, no cycles
- Builds `_step_map: dict[str, Step]`
- Computes `_execution_levels: list[list[str]]` via topological sort (Kahn's algorithm)

### Execution (column-oriented)
```
For each level in execution_levels:
    For each step in level (parallel via asyncio.gather):
        For each row (concurrent with semaphore):
            Build StepContext (row data, step's field specs, prior results from deps)
            result = await step.run(ctx)
            Store result
    Merge step results into accumulated results
Return accumulated results for all rows
```

Each level runs sequentially. Steps within a level run in parallel. Rows within a step run concurrently (bounded by semaphore from config.max_workers).

### Internal fields
Fields prefixed with `__` (e.g. `__web_context`) are used for inter-step communication. The Enricher filters them from the final DataFrame output.

## Built-in Steps

### LLMStep (`accrue/steps/llm.py`)

```python
class LLMStep:
    def __init__(
        self,
        name: str,
        fields: list[str] | dict[str, str | dict],  # Phase 2: inline specs
        depends_on: list[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = None,    # Falls back to config
        max_tokens: int = None,       # Falls back to config
        system_prompt: str = None,    # Falls back to built-in enrichment prompt
        api_key: str = None,          # Falls back to OPENAI_API_KEY env
        base_url: str = None,         # Phase 2: OpenAI-compatible endpoints
        client: LLMClient = None,     # Phase 2: any LLMClient protocol adapter
        schema: Type[BaseModel] = EnrichmentResult,
        max_retries: int = 2,
        run_if: Callable = None,      # Phase 6B: (row, prior_results) -> bool
        skip_if: Callable = None,     # Phase 6B: (row, prior_results) -> bool
    ): ...

    async def run(self, ctx: StepContext) -> StepResult: ...
```

**Fields parameter (Phase 2 → redesigned Phase 3):**
- `list[str]` — field names only, specs come from external source (backward compat)
- `dict[str, str]` — shorthand: `{"market_size": "Estimate TAM"}` → prompt only
- `dict[str, dict]` — full spec with 7 supported keys:

| Key | Type | Required | Purpose |
|-----|------|----------|---------|
| `prompt` | `str` | Yes | The extraction instruction |
| `type` | `str` | No | Data type: `String`, `Number`, `Boolean`, `Date`, `List[String]`, `JSON` (default: `String`) |
| `format` | `str` | No | Output format pattern (e.g. `"YYYY-MM-DD"`, `"$X.XB"`, `"X/10"`) |
| `enum` | `list[str]` | No | Constrained value list (e.g. `["Low", "Medium", "High"]`) |
| `examples` | `list[str]` | No | Good output examples showing expected style |
| `bad_examples` | `list[str]` | No | Anti-patterns to avoid |
| `default` | `Any` | No | Fallback value when data is insufficient (enforced in Python) |

Example:
```python
LLMStep("analyze", fields={
    "market_size": "Estimate TAM in billions USD",  # shorthand
    "risk_level": {                                  # full spec
        "prompt": "Assess investment risk based on market and competitive data",
        "type": "String",
        "enum": ["Low", "Medium", "High"],
        "default": "Unknown",
        "examples": ["High", "Medium"],
        "bad_examples": ["Moderately high", "3/5"],
    },
    "revenue": {
        "prompt": "Estimate annual revenue",
        "type": "Number",
        "format": "$X.XB",
    },
})
```

When fields is a dict, LLMStep uses specs directly in `_build_system_message()` — no FieldManager needed.

- Uses `LLMClient` protocol internally (not `openai.AsyncOpenAI` directly)
- `OpenAIClient` is the default adapter (covers OpenAI + all compatible providers via `base_url`)
- `AnthropicClient` and `GoogleClient` ship as optional extras
- `client` parameter accepts any `LLMClient`-compatible adapter
- `base_url` shortcut creates `OpenAIClient(base_url=...)` for OpenAI-compatible providers
- Lazy client initialization (no import-time API key check)
- Builds system message with: system prompt + row data + field specs + prior_results + JSON schema
- Calls with `response_format={"type": "json_object"}`
- Validates response with Pydantic `model_validate()`
- On validation error: appends error to conversation, retries (sends error back to LLM)
- Returns `StepResult` with values filtered to only this step's declared fields

### FunctionStep (`accrue/steps/function.py`)

```python
class FunctionStep:
    def __init__(
        self,
        name: str,
        fn: Callable,          # (StepContext) -> dict[str, Any]
        fields: list[str],
        depends_on: list[str] = None,
        cache: bool = True,             # Phase 4: per-step cache bypass
        cache_version: str = None,      # Phase 4: version-based invalidation
        run_if: Callable = None,        # Phase 6B: (row, prior_results) -> bool
        skip_if: Callable = None,       # Phase 6B: (row, prior_results) -> bool
    ): ...
```

- Wraps any sync or async callable
- Sync functions run via `run_in_executor`
- The escape hatch for any data source: APIs, databases, custom logic
- `cache_version` — user bumps version string when function logic changes; auto-invalidates cache
- `cache=False` — disables caching for non-deterministic functions
- `run_if` / `skip_if` — per-row predicates for conditional execution (Phase 6B)

### LLMClient Protocol (`accrue/steps/providers/base.py`) — Phase 2

```python
class LLMClient(Protocol):
    """What LLMStep needs from any LLM provider."""
    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> LLMResponse: ...

@dataclass
class LLMResponse:
    content: str
    usage: UsageInfo | None = None
```

**Shipped adapters:**
- `OpenAIClient` (~30 lines) — default. Covers OpenAI, DeepSeek, Groq, Together, Fireworks, Ollama, vLLM, Mistral, LM Studio via `base_url`.
- `AnthropicClient` (~30 lines) — optional extra: `pip install accrue[anthropic]`
- `GoogleClient` (~30 lines) — optional extra: `pip install accrue[google]`

**Import:** `from accrue.providers import OpenAIClient, AnthropicClient, GoogleClient`

### Web Search Utility (Phase 5)

**`web_search()` factory** (`accrue/utils/web_search.py`) — Convenience wrapper around OpenAI Responses API. Returns async callable for FunctionStep:

```python
from accrue import Pipeline, FunctionStep, LLMStep, web_search

Pipeline([
    FunctionStep("research",
        fn=web_search("Research {company}: market position and competitors"),
        fields=["__web_context", "sources"],
    ),
    LLMStep("analyze", fields={"market_size": "Estimate TAM"}, depends_on=["research"]),
])
```

Parameters: `query` (template), `model="gpt-4.1-mini"`, `search_context_size="medium"`, `api_key`, `include_sources=True`. Graceful degradation on API errors (returns empty context). Custom FunctionStep still works for full control.

### No Built-in WebSearchStep

**Decision (Feb 2026): WebSearchStep was scrapped.** FunctionStep already handles any data source — web search, APIs, databases. The `web_search()` utility is a convenience factory, not a step type. Users bring their own search provider:

```python
async def web_search(ctx):
    results = await my_search_provider(ctx.row["company_name"])
    return {"__web_context": results}

Pipeline([
    FunctionStep("search", fn=web_search, fields=["__web_context"]),
    LLMStep("analyze", fields=["market_size"], depends_on=["search"]),
])
```

No Tavily dependency. No provider abstraction. No "waterfall resolution." Users write a function; Accrue orchestrates it.

**Web search with citations (recommended pattern):**

OpenAI's web search + structured output in one call is unreliable (truncation bugs). The two-step pattern works:

```python
async def web_research(row):
    """FunctionStep: calls OpenAI Responses API with web search."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    response = await client.responses.create(
        model="gpt-4.1-mini",  # nano doesn't support web search
        tools=[{"type": "web_search", "search_context_size": "medium"}],
        input=f"Research {row['company']}: market position, competitors, recent news",
        include=["web_search_call.action.sources"],
    )
    sources = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                for ann in getattr(block, 'annotations', []):
                    if ann.type == "url_citation":
                        sources.append(ann.url)
    return {"__web_context": response.output_text, "sources": sources}

Pipeline([
    FunctionStep("research", fn=web_research, fields=["__web_context", "sources"]),
    LLMStep("analyze", fields={
        "market_size": {"prompt": "Estimate TAM based on the research context", "format": "$X.XB"},
        "competition": {"prompt": "Rate competition level", "enum": ["Low", "Medium", "High"]},
    }, depends_on=["research"]),
])
```

`__web_context` is internal (filtered from output). `sources` is visible — users get citation URLs per row. The LLMStep sees web context in `prior_results` via `depends_on`.

### Grounding vs `web_search()` — When to Use Each

| | `grounding=True` | `web_search()` + FunctionStep |
|---|---|---|
| **Steps** | 1 | 2 |
| **Query control** | Model decides | You template with `{field}` |
| **Providers** | OpenAI, Anthropic, Google | OpenAI only |
| **Structured outputs** | OpenAI: yes. Anthropic/Google: json_object fallback | Full (search is a separate step) |
| **Citations** | `__sources` (internal, available in `prior_results`) | Explicit `sources` field in output |
| **Provider tuning** | `provider_kwargs` on GroundingConfig | Full control (you write the API call) |
| **Best for** | Simple grounded enrichment, multi-provider | Complex queries, multi-query, need full control |

## Primary API: Pipeline.run()

**Phase 2 redesign:** Pipeline becomes the primary public interface. Enricher becomes an internal runner.

```python
# accrue/pipeline/pipeline.py

class Pipeline:
    def __init__(self, steps: list[Step]): ...

    # Primary entry points
    def run(self, df: pd.DataFrame, config: EnrichmentConfig = None) -> PipelineResult: ...
    async def run_async(self, df: pd.DataFrame, config: EnrichmentConfig = None) -> PipelineResult: ...

    # Power user: reusable runner with config
    def runner(self, config: EnrichmentConfig = None) -> Enricher: ...

    # Core execution (internal)
    async def execute(self, rows, all_fields, config, ...) -> tuple[list[dict], list[RowError], CostSummary]: ...
```

### `run()` implementation
1. Collect field specs from steps (LLMStep provides specs, FunctionStep provides names only)
2. Convert DataFrame rows to `list[dict]`
3. Call `self.execute(rows, fields, config)`
4. Write results back to DataFrame (filtering `__` prefixed internal fields)
5. Return `PipelineResult(data=df, cost=CostSummary, errors=list[RowError])`

### Enricher (internal runner)

```python
# accrue/core/enricher.py — internal, not in public API

class Enricher:
    def __init__(self, pipeline: Pipeline, config: EnrichmentConfig = None): ...
    def run(self, df: pd.DataFrame) -> pd.DataFrame: ...
    async def run_async(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

Enricher handles: DataFrame conversion, checkpointing, `__` field filtering, sync/async wrapping. Created via `pipeline.runner(config=...)` for repeated execution with the same config.

No FieldManager parameter. No category parameter. Field specs come from steps.

## Dependencies

### Remove
- `langchain>=0.3.0`
- `langchain-core>=0.3.0`
- `langchain-openai>=0.3.0`
- `tenacity>=9.0.0`

### Add
- `openai>=1.0.0`
- `pydantic>=2.0.0`

### Keep
- `pandas>=2.0.0`
- `tqdm>=4.65.0`
- `python-dotenv>=1.0.0`

### Optional Extras (Phase 2)
- `anthropic>=1.0.0` — `pip install accrue[anthropic]`
- `google-genai>=1.0.0` — `pip install accrue[google]`
- `pip install accrue[all]` — both

## Checkpoint Model

Per-step checkpointing (implemented in Phase 1):
- Save after each step completes for all rows
- Store: `{completed_steps: ["step1", "step2"], step_results: {...}, metadata: {...}}`
- On resume: skip completed steps, start from next step with saved results as prior_results
- JSON file per pipeline run

## Field Routing Validation

Pipeline validates at construction time:
- No duplicate step names, no missing dependencies, no cycles (already done in Phase 1)
- No two steps produce the same non-`__` field

Optional validation at `run()` time:
- `pipeline.run(df, expected_fields=["market_size", "competition"])` validates coverage
- Missing fields → `FieldValidationError` (fail fast)
- No category concept — just a flat list of expected field names

## System Prompt — Dynamic Prompt Builder (Phase 3)

Implemented in `accrue/steps/prompt_builder.py`. Follows OpenAI GPT-4.1 cookbook: **markdown headers for sections, XML tags for data boundaries**. JSON in prompts performs poorly per OpenAI's long-context testing.

**Structure:**
```markdown
# Role
You are a structured data enrichment engine...

# Field Specification Keys
[DYNAMIC: only describe keys actually present across this step's fields]
- prompt: the extraction instruction
- type: expected output type
[only if any field uses enum:]
- enum: value MUST be one of these options verbatim
[only if any field uses format:]
- format: specific output format pattern
[etc.]

# Output Rules
- Return ONLY valid JSON. No prose, no code fences.
- Keys MUST be exactly the field names below.
[DYNAMIC: enum/default rules only if relevant]

<row_data>
{"company": "Acme Corp", "industry": "Cloud"}
</row_data>

<field_specifications>
<field name="market_size">
  <prompt>Estimate the total addressable market</prompt>
  <type>Number</type>
  <format>$X.XB</format>
</field>
<field name="risk_level">
  <prompt>Assess investment risk</prompt>
  <enum>Low, Medium, High</enum>
  <default>Unknown</default>
</field>
</field_specifications>

<prior_results>
[only if step has dependencies]
</prior_results>
```

**Key principles:**
- Static content at top (enables OpenAI prompt caching)
- Variable content (row data, field specs) at bottom
- Only include field spec key descriptions for keys actually used
- XML per-field blocks only include defined keys (no empty tags)
- Sandwich pattern: key constraints reiterated after data for long-context reliability

### Three-Tier Prompt Customization (Phase 5)

1. **Default** — Dynamic prompt builder handles everything
2. **`system_prompt_header=`** — Inject `# Context` section between Role and Field Specification Keys: `"You are analyzing B2B SaaS companies in the European market."` Ignored when `system_prompt` (Tier 3) is set.
3. **`system_prompt=`** — Full override (existing, power users own the entire prompt)

### Structured Outputs (Phase 5)

Migrated from `response_format={"type": "json_object"}` (legacy) to `{"type": "json_schema", "strict": true}` for OpenAI. Dynamically builds Pydantic model from 7-key field specs via `accrue/steps/schema_builder.py`.

**Auto-detection logic:**
- Native OpenAI + dict fields → `json_schema` + `strict: true` (auto-enabled)
- `base_url` providers → `json_object` (Ollama, Groq may not support it)
- Non-OpenAI client → `json_object`
- Custom `schema=` → `json_object` (user manages validation)
- `list[str]` fields → `json_object` (no specs to build schema from)
- `structured_outputs=True/False` → explicit override

**Schema builder** (`accrue/steps/schema_builder.py`):
- `build_response_model(field_specs)` → `type[BaseModel]` via `pydantic.create_model()`
- Type mapping: `String→str, Number→float, Boolean→bool, Date→str, List[String]→list[str], JSON→dict[str,Any]`
- Enum fields → `Literal["val1", "val2", ...]`
- `extra="forbid"` → `additionalProperties: false` in schema
- `build_json_schema(field_specs)` → `{"type": "json_schema", "json_schema": {"name": "enrichment_result", "schema": ..., "strict": True}}`

Research: `docs/research/prompt-engineering.md`

## Usage Examples

### Simple: One LLM step (primary API)
```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate the total addressable market in billions USD",
        "competition": "Rate as Low/Medium/High with key competitors",
        "growth_potential": "Evaluate growth potential with reasoning",
    })
])
result = pipeline.run(df)
```

### Multi-step with dependencies
```python
from accrue import Pipeline, LLMStep, FunctionStep

async def search_company(ctx):
    results = await my_search(ctx.row["name"])
    return {"__web_context": results}

pipeline = Pipeline([
    FunctionStep("search", fn=search_company, fields=["__web_context"]),
    LLMStep("market", fields={
        "market_size": "Estimate TAM using search results",
        "competition": "Rate competition level",
    }),
    # search + market run in parallel, then synthesis uses both
    LLMStep("synthesis", fields={
        "growth_potential": "Synthesize growth potential from market data and search results",
    }, depends_on=["search", "market"]),
])
result = pipeline.run(df)
```

### Custom function step
```python
def lookup_funding(ctx):
    company = ctx.row.get("company_name")
    return {"funding_amount": my_api_call(company)}

pipeline = Pipeline([
    FunctionStep("crunchbase", fn=lookup_funding, fields=["funding_amount"]),
    LLMStep("analysis", fields={
        "investment_thesis": "Write an investment thesis based on funding data",
    }, depends_on=["crunchbase"]),
])
result = pipeline.run(df)
```

### Power user: reusable runner
```python
from accrue import EnrichmentConfig

runner = pipeline.runner(config=EnrichmentConfig(
    max_workers=10,
    enable_checkpointing=True,
))
result = await runner.run_async(df)
```

### Team with CSV field definitions
```python
from accrue.data import load_fields

fields = load_fields("fields.csv", category="business_analysis")
pipeline = Pipeline([LLMStep("analyze", fields=fields)])
result = pipeline.run(df)
```

---

## Phase 2: Resilience + API Redesign (COMPLETE)

Merged in #32. All issues closed: #19, #20, #23, #24, #25, #26, #27, #28, #29.

### What was built

- **API Redesign (#29):** `Pipeline.run(df)` → `PipelineResult` as primary API. LLMStep accepts `fields={"name": "prompt"}` inline specs. Enricher demoted to internal runner via `pipeline.runner()`. `load_fields()` utility replaces FieldManager in public API. Removed `EnrichmentSpec`, `StructuredResult`, dead FieldManager methods.
- **Per-Row Error Handling (#23):** `asyncio.gather(*coros, return_exceptions=True)`. Failed rows produce `RowError` with sentinel values. Configurable `on_error="continue"|"raise"`.
- **API Retry with Backoff (#24):** Two-layer retry in LLMStep — outer loop for API errors (429/500/timeout) with exponential backoff + jitter + Retry-After header, inner loop for parse/validation errors fed back to LLM.
- **Progress Reporting (#20):** tqdm step-level progress bar wired to `config.enable_progress_bar`.
- **Cost Aggregation (#25):** `CostSummary` with per-step `StepUsage` (prompt/completion/total tokens, rows processed, model). Returned in `PipelineResult.cost`.
- **LLMClient Protocol (#28):** `LLMClient` protocol + `OpenAIClient` (default), `AnthropicClient`, `GoogleClient` adapters. `base_url` shortcut for OpenAI-compatible providers. `LLMAPIError` wrapper for provider-agnostic retry.
- **Dead Code Removal (#26) + Config Cleanup (#19):** Removed `VectorStoreError`, `LLMError`, `EnrichmentSpec`, `StructuredResult`. Config was already clean from Phase 1C.

## Phase 3: Field Spec Redesign + Dynamic Prompt (Epic #33) — COMPLETE

The prompt engineering layer is the core of enrichment quality. This phase rebuilt it from research.

### What was built
1. **7-key field spec validation** — `FieldSpec` Pydantic model (`accrue/schemas/field_spec.py`) with `extra="forbid"`. Enforced on LLMStep construction via `_normalize_field_specs()`. Rejects unknown keys. `prompt` required, all others optional.
2. **Dynamic system prompt builder** — `accrue/steps/prompt_builder.py`. Markdown headers + XML data boundaries (OpenAI GPT-4.1 cookbook). Only describes keys actually present across this step's fields. Static content at top (enables prompt caching), variable data at bottom, sandwich pattern reminder at end.
3. **`default` enforcement in Python** — `LLMStep._apply_defaults()` replaces refusal language ("Unable to determine", "N/A", etc.) with field's `default` value post-extraction. Uses `model_fields_set` to distinguish "default not set" from "default=None".
4. **Default model → `gpt-4.1-mini`** — Changed from `gpt-4.1-nano`. Temperature fallback → 0.2 (was 0.5). Nano stays available per-step.
5. **CSV loader update** — Rewritten for new 7-key format. Backward-compatible: legacy `Instructions`/`Guidance` columns concatenated into `prompt`, `Data_Type` → `Type` fallback. Fixed `get_category_fields()` bug (was stripping examples).

### Out of scope (deferred)
- Structured Outputs migration (`json_schema` + `strict`) — requires dynamic Pydantic model generation
- Regex validation on `format` — future enhancement
- `system_prompt_header` — Phase 5B (#34)

## Phase 4: Caching + Checkpoint Enhancement (Epic #17) — COMPLETE

Input-hash cache for iterative development and large dataset resilience. Without caching, changing one field in a 3-step pipeline re-runs every API call for every row. This is the single biggest DX improvement.

### Design Decisions (Feb 2026)

**Caching and checkpointing are separate concerns:**
- **Checkpoint**: Step-level crash recovery — "pipeline died at step 3, resume from step 2's output." Coarse-grained (per-step), fast resume (one JSON file load). Enhanced in Phase 4 with `checkpoint_interval` for partial step progress on large datasets.
- **Cache**: Input-hash deduplication — "I changed step 3's prompt, skip steps 1 and 2 for unchanged rows." Fine-grained (per-step-per-row), content-addressed. Also provides row-level crash recovery as a side effect.

Both can be enabled independently. For development: enable caching. For production with large datasets: enable both.

**SQLite backend (zero dependencies):**
`sqlite3` is Python stdlib. Benchmarks show 35% faster reads than filesystem JSON (sqlite.org), handles thousands of entries trivially, atomic concurrent writes via WAL mode. Single `.accrue/cache.db` file. Industry precedent: requests-cache, DiskCache (used by DSPy), and Instructor all use SQLite for local caching.

**FunctionStep caching via `cache_version`:**
FunctionSteps are cacheable with an explicit `cache_version` string. Users bump the version when function logic changes. `cache=False` disables caching for non-deterministic functions.

### Scope
1. **Input-hash cache key** — `SHA256(canonical_json(step_name + row_data + prior_results + field_specs + model + temperature + system_prompt_config))`. Canonical JSON (sorted keys) for determinism. Changing a field's prompt, enum, or type auto-invalidates (Instructor pattern).
2. **SQLite backend** — Single `.accrue/cache.db` file. WAL mode + `synchronous=NORMAL`. Zero new dependencies (`sqlite3` is stdlib).
3. **TTL-based expiry** — `cache_ttl` already in `EnrichmentConfig` (unused), wire it up. Lazy expiry on read + periodic cleanup.
4. **Per-step control** — `LLMStep(..., cache=False)` to bypass. `FunctionStep(..., cache_version="v1")` for versioned caching.
5. **Pipeline-level control** — `EnrichmentConfig(enable_caching=True)` (already exists, wire it up). `pipeline.clear_cache()` for manual invalidation.
6. **Cache stats** — Extend `StepUsage` with `cache_hits`, `cache_misses`, `cache_hit_rate`. Flows into `PipelineResult.cost`.
7. **Checkpoint enhancement** — `checkpoint_interval: int = 100` — save partial step progress every N completed rows within a step. For large datasets (1K+ rows), prevents losing an entire step's work on crash.
8. **`list[dict]` input support** — `Pipeline.run()` accepts `pd.DataFrame | list[dict]`. Skip pandas conversion when given dicts. Return type matches input type.

### Cache Key Composition

```python
# LLMStep cache key
cache_key = sha256(canonical_json({
    "step":        step.name,
    "row":         row_data,           # Full row dict from DataFrame
    "prior":       prior_results,      # Outputs from dependency steps
    "fields":      field_specs,        # FieldSpec dicts (prompt, type, enum, etc.)
    "model":       model,              # e.g. "gpt-4.1-mini"
    "temperature": temperature,
    "system":      system_prompt_hash, # Custom system prompt if any
}))

# FunctionStep cache key (function body isn't hashable)
cache_key = sha256(canonical_json({
    "step":          step.name,
    "row":           row_data,
    "prior":         prior_results,
    "cache_version": cache_version,     # User-provided version string
}))
```

**In the key:** Everything that affects the output — step identity, input data, field specs (so changing a prompt auto-invalidates), model config.

**Not in the key:** Execution config (`max_retries`, `on_error`, `max_workers`), credentials (`api_key`, `base_url`), cache meta-config (`cache_ttl`).

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    step_name  TEXT NOT NULL,
    value      TEXT NOT NULL,      -- JSON serialized step output
    created_at REAL NOT NULL,
    expires_at REAL               -- NULL = no expiry
);
CREATE INDEX IF NOT EXISTS idx_step ON cache(step_name);
CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at);
```

Pragmas at connection time:
```sql
PRAGMA journal_mode=WAL;          -- concurrent reads + writes
PRAGMA synchronous=NORMAL;        -- safe for cache (recoverable data)
PRAGMA cache_size=-8000;          -- 8MB in-memory page cache
```

### API Surface

```python
# Config (existing fields wired up + new fields)
EnrichmentConfig(
    enable_caching=True,          # Existing, wire it up
    cache_ttl=3600,               # Existing, wire it up
    cache_dir=".accrue",         # NEW — directory for cache.db
    checkpoint_interval=100,      # NEW — save partial progress every N rows
)

# Per-step control
LLMStep("analyze", fields={...}, cache=False)           # Bypass cache
FunctionStep("search", fn=search, cache_version="v2")   # Version-based invalidation

# Pipeline-level operations
pipeline.clear_cache()              # Delete all cache entries
pipeline.clear_cache(step="search") # Delete entries for one step

# Stats (in existing PipelineResult.cost → StepUsage)
result.cost.steps["analyze"].cache_hits      # 95
result.cost.steps["analyze"].cache_misses    # 5
result.cost.steps["analyze"].cache_hit_rate  # 0.95
```

### Config Preset Updates

```python
EnrichmentConfig.for_development()
    # Adds:
    enable_caching=True          # Avoid re-running during iteration
    cache_dir=".accrue"

EnrichmentConfig.for_production()
    # Adds:
    enable_caching=True
    enable_checkpointing=True    # Existing
    checkpoint_interval=100      # Partial step progress for large datasets
    cache_dir=".accrue"
```

### Integration Points

**Where cache checks happen:** In `Pipeline._execute_step()`, wrapping `step.run(ctx)`:
1. Compute cache key from StepContext inputs (row, prior_results, field_specs, model config)
2. Check SQLite → hit returns cached result (no API call), increment `cache_hits`
3. On miss: call `step.run(ctx)`, store result in SQLite, increment `cache_misses`

**Where checkpoint interval fires:** In `Pipeline._execute_step()`, as rows complete:
1. Track completed row count via atomic counter
2. Every `checkpoint_interval` rows, fire partial checkpoint callback
3. Enricher saves partial step state (completed row indices + their results) to checkpoint file

**Cache + checkpoint interaction:**
- When both enabled: checkpoint provides fast "which steps are fully done" metadata; cache provides row-level recovery within partially-completed steps
- When only cache enabled: row-level recovery is automatic; no step-level metadata
- When only checkpoint enabled: step-level + `checkpoint_interval` row-level recovery; no cross-run deduplication

### `list[dict]` Input Support

`Pipeline.run()` accepts `pd.DataFrame | list[dict]`. When given a list of dicts, skip the pandas conversion and use directly. Return type matches input type: DataFrame in → DataFrame out, list[dict] in → `PipelineResult` with `list[dict]` data.

```python
# DataFrame (existing)
result = pipeline.run(df)
result.data  # pd.DataFrame

# list[dict] (new)
result = pipeline.run([{"company": "Acme"}, {"company": "Beta"}])
result.data  # list[dict]

# Polars users (no native support needed — one-liner)
result = pipeline.run(polars_df.to_dicts())
```

This makes pandas a convenience dependency rather than a hard requirement for the internal path. Server contexts (FastAPI) and test code benefit from skipping DataFrame construction.

### Future (Not Phase 4)

- **Chunked execution** (`chunk_size=5000`): Process N rows at a time through the full pipeline. Memory stays bounded, cache carries across chunks. Changes execution from column-oriented to chunk-oriented — separate issue, depends on Phase 4 caching.
- **Intra-batch deduplication**: If 100 rows have identical inputs, call the LLM once and map result to all 100. Changes the execution loop (group by cache key before dispatch). Significant cost savings for datasets with duplicate entries.
- **Custom cache backends**: `CacheBackend` protocol for Redis, Memcached, etc. Users implement `get()`/`set()`/`delete()`.
- **`cache_key_fn`**: Custom cache key function on FunctionStep for fine-grained control (e.g. cache by company domain only, not full row).
- **Optional Polars support**: Accept `pl.DataFrame` at boundary, convert via `.to_dicts()`. Not needed while `list[dict]` input covers the use case. Revisit post-launch if users request it.

**Note:** `enable_caching` and `cache_ttl` are already on `EnrichmentConfig` from Phase 1 — they're just not wired to anything. This phase makes them real and adds `cache_dir` and `checkpoint_interval`.

### What was built

1. **CacheManager** (`accrue/core/cache.py`) — SQLite-backed cache with WAL mode, lazy connection, TTL expiry on read, cleanup method. Contains `compute_cache_key()` (SHA-256 of canonical JSON) and `_compute_step_cache_key()` (duck-types LLMStep vs FunctionStep).
2. **Cache integration in `_execute_step()`** — Cache check before `step.run()`, cache store after. `step_cache_enabled` flag from `cache_manager is not None and getattr(step, 'cache', True)`.
3. **Cache stats** — `StepUsage.cache_hits`, `cache_misses`, `cache_hit_rate` property. Flows into `PipelineResult.cost.steps["name"]`.
4. **Per-step control** — `LLMStep(..., cache=True)` and `FunctionStep(..., cache=True, cache_version="v1")`. `cache=False` bypasses.
5. **Config wiring** — `cache_dir=".accrue"`, `checkpoint_interval=0` added to `EnrichmentConfig`. Presets updated: `for_development()` enables caching, `for_production()` enables caching + `checkpoint_interval=100`.
6. **`list[dict]` input** — `Pipeline.run()` / `run_async()` accept `pd.DataFrame | list[dict]`. Output type matches input type. Internal fields (`__` prefixed) filtered from list output.
7. **`Pipeline.clear_cache()`** — Full and per-step invalidation.
8. **Checkpoint interval** — `checkpoint_interval=N` saves partial step progress every N rows via `asyncio.as_completed` pattern. Enricher wires callback to `CheckpointManager.save_step()`.
9. **CacheManager wiring** — Created and closed in `run_async()` and `Enricher.run_async()` with `finally` guard.

## Phase 5: Quality + Observability + DX — COMPLETE

Four features closing competitive gaps vs. Instructor, Clay, and LangChain.

### What was built

1. **`system_prompt_header` (#34)** — Tier 2 prompt customization. `LLMStep(..., system_prompt_header="Analyzing B2B SaaS companies.")` injects `# Context` section between Role and Field Specification Keys. Cache key includes header hash. Ignored when `system_prompt` (Tier 3) is set.

2. **Lifecycle hooks (#30)** — `EnrichmentHooks` dataclass with 5 typed event callbacks: `on_pipeline_start`, `on_pipeline_end`, `on_step_start`, `on_step_end`, `on_row_complete`. Passed to `run()`/`run_async()`. Sync + async hooks. Hook errors caught + logged. Events include timing, usage, cache status, errors. Subsumes #11 (streaming) via `on_row_complete`.

3. **Structured outputs migration** — Auto-enabled `json_schema` + `strict: true` for native OpenAI with dict field specs. Dynamic Pydantic model built from FieldSpec definitions (`accrue/steps/schema_builder.py`). Auto-off for base_url, non-OpenAI, custom schema, list fields. `structured_outputs=True/False` override. `StepResult.metadata["structured_outputs"]` flag.

4. **`web_search()` utility (#35)** — Factory wrapping OpenAI Responses API. `web_search("Research {company}: market")` returns async callable for FunctionStep. Template formatting from row + prior_results. Citation extraction. Graceful degradation on API errors.

## Phase 6A: Ship (Epic #18)

Minimum viable distribution. Get Accrue into users' hands.

- **Working examples** — 3-4 runnable scripts with sample data: simple enrichment, multi-step with deps, web search two-step pattern, Anthropic/Google provider usage
- **README rewrite** — Real examples, install instructions, quick start
- **PyPI publish** — `pip install accrue`
- **#21 docs fix** — Update github-standards.md for v0.3 components (quick win)

## Phase 6B: Power User Features — IN PROGRESS

### What was built

1. **Conditional step execution (#40)** — `run_if`/`skip_if` predicates on `LLMStep` and `FunctionStep`. Predicate signature: `(row: dict, prior_results: dict) -> bool`. Mutually exclusive (validated at construction, raises `PipelineError`). Evaluated per-row in `_execute_step()` before cache check — skipped rows never hit cache or call `step.run()`. Skipped rows produce field spec `default` where available, else `None`. `RowCompleteEvent.skipped` flag (backward-compatible `False` default). `StepUsage.rows_skipped` counter. Sync + async predicates via `inspect.isawaitable()`. Predicate exceptions treated as row errors. Step protocol unchanged — attributes read via `getattr()`.

2. **Batch API support (#62)** — `LLMStep(batch=True)` routes eligible steps through provider Batch APIs (50% cost savings). `BatchCapableLLMClient` protocol with `submit_batch()`, `poll_batch()`, `cancel_batch()`. OpenAI adapter (JSONL upload → poll → download → parse) and Anthropic adapter (Message Batches API). Google deferred. Cache-aware batching: only uncached rows enter the batch. Auto-chunking at `batch_max_requests` (default 50K). Realtime fallback for failed batch rows. `StepUsage.execution_mode` and `batch_id` for observability. `StepStartEvent`/`StepEndEvent` gain `execution_mode` field. Config: `batch_poll_interval`, `batch_timeout`, `batch_max_requests`. Preset: `EnrichmentConfig.for_batch()`. `_execute_step_batch()` parallel method alongside `_execute_step()`. `LLMStep.build_messages()` and `parse_response()` extracted for batch/realtime reuse. `batch=True` mutually exclusive with `grounding`. ADR 014.

3. **`provider_kwargs` passthrough** — `LLMStep(provider_kwargs={...})` merges into every provider API call (realtime and batch). Escape hatch for provider-specific features not yet first-class (e.g. `{"thinking": {"type": "adaptive"}}`, `{"output_config": {"effort": "low"}}`). Not included in cache key.

4. **Anthropic prompt caching** — System messages automatically wrapped with `cache_control: {"type": "ephemeral"}`. First row pays 1.25x, rows 2-N pay 0.1x on cached system prompt. ~90% savings on input tokens for column-oriented execution. No user action required. OpenAI Responses API caches automatically.

### Remaining

- **Waterfall enrichment pattern** — Utility for try-source-A, fall-back-to-B
- **Intra-batch deduplication** — Deduplicate identical rows before API call
- **Chunked execution** — `chunk_size=N` for memory-bounded processing
- **CLI** — `accrue run --csv data.csv --fields fields.csv`

## Backlog Triage (Feb 2026)

| # | Issue | Decision | Reasoning |
|---|-------|----------|-----------|
| **#13** | Eval suite | **Closed (won't-fix)** | Conflicts with design principle: "Evals are user-level, not library-level." Accrue exposes data; users run their own evals. |
| **#11** | Streaming output | **Closed** | Subsumed by lifecycle hooks (`EnrichmentHooks.on_row_complete`). |
| **#12** | Sources/provenance | **Kept, deprioritized** | Partially addressed by web search two-step pattern (`sources` column). Full provenance is a post-launch differentiator per TECHNICAL-VISION.md. |

## Design Decisions Log

| Decision | Date | Rationale |
|----------|------|-----------|
| `Pipeline.run(df)` as primary API | Feb 2026 | One concept, one entry point. Enricher is internal detail, not public API. |
| Fields live on steps | Feb 2026 | Single source of truth. No separate field registry. LLMStep declares what AND how. |
| FieldManager → `load_fields()` utility | Feb 2026 | CSV loading is a convenience, not a core dependency. No categories in core. |
| Drop WebSearchStep | Feb 2026 | FunctionStep already handles any data source. No built-in provider steps. |
| Drop waterfall resolution | Feb 2026 | Source priority is user logic (domain-specific), not framework logic. |
| LLMClient protocol + shipped adapters | Feb 2026 | OpenAI default, Anthropic/Google as optional extras (~30 lines each). Covers top 3 providers + all OpenAI-compatible via base_url. |
| No litellm dependency | Feb 2026 | Too heavy (~30 transitive deps). Protocol + thin adapters achieves the same with zero required deps. |
| No langfuse/eval dependency | Feb 2026 | Evals are application-level. Accrue exposes hooks for observability tools to plug in. |
| Phase 2 = Resilience + API redesign | Feb 2026 | Foundation gaps + API simplification must happen before PyPI publish. |
| Default model → gpt-4.1-mini | Feb 2026 | Nano too limited: poor complex retrieval, no web search support, hallucination risk with structured outputs. Mini is "standout star," matches GPT-4o at 83% cheaper. Users can still set nano per-step for simple classification. |
| 7-key field spec (drop `instructions`) | Feb 2026 | Research-backed: `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, `default`. Merged `instructions`/`guidance` into `prompt` — other keys already cover structured concerns. |
| Dynamic system prompt | Feb 2026 | Only describe field spec keys actually used. Saves tokens, avoids confusing gpt-4.1's literal instruction following with irrelevant sections. |
| Markdown headers + XML data boundaries | Feb 2026 | OpenAI GPT-4.1 cookbook recommendation. JSON in prompts "performed particularly poorly" in their long-context testing. |
| Web search = two-step FunctionStep → LLMStep | Feb 2026 | OpenAI web search + structured output in one call is broken (truncation). Two-step avoids this. Citations flow as visible `sources` column. |
| Three-tier prompt customization | Feb 2026 | Default (dynamic), `system_prompt_header=` (domain injection), `system_prompt=` (full override). Covers 99% of use cases without complexity. |
| SQLite cache backend (not filesystem JSON) | Feb 2026 | stdlib `sqlite3`, 35% faster reads than FS, WAL mode for concurrency, single file, TTL via SQL. Industry standard: requests-cache, DiskCache, DSPy all use SQLite. |
| Cache + checkpoint are separate concerns | Feb 2026 | Checkpoint = step-level crash recovery (fast single-file resume). Cache = row-level input-hash deduplication (across runs, auto-invalidates on prompt change). Both can be enabled independently. |
| FunctionStep caching via `cache_version` | Feb 2026 | Function body isn't hashable. User provides explicit version string, bumps it when logic changes. `cache=False` for non-deterministic functions. |
| `checkpoint_interval` for large datasets | Feb 2026 | Save partial step progress every N rows (default 100). Prevents losing entire step of work on crash for datasets with thousands of rows. |
| `list[dict]` input support | Feb 2026 | Internals already work on `list[dict]`. Accept dicts directly — serves server contexts, test code, and Polars users (`.to_dicts()`). |
| No native Polars support (for now) | Feb 2026 | 77% of Python data practitioners use pandas. Zero LLM tools support Polars natively. `list[dict]` input covers Polars users. Revisit post-launch. |
| Chunked execution is separate from Phase 4 | Feb 2026 | Changes execution model from column-oriented to chunk-oriented. Medium effort, depends on caching. Own issue in Phase 5B. |
| Keep pandas as base dependency | Feb 2026 | Accrue's users are pandas users (enrichment workflows, Jupyter, CSV origins). Making pandas optional adds complexity for minimal benefit. |
| Hooks on `run()`, not config or Pipeline | Feb 2026 | Config is serializable data; hooks are callables. Different runs may want different hooks. |
| 5 hooks with typed event dataclasses | Feb 2026 | `on_pipeline_start/end`, `on_step_start/end`, `on_row_complete`. `on_row_error` merged into `on_row_complete` (check `event.error`). Simpler API. |
| Hook errors silently caught + logged | Feb 2026 | Observability failures must not crash data pipelines. `except Exception` catches; `BaseException` propagates. |
| Structured outputs auto-detect | Feb 2026 | On for native OpenAI + dict fields, off for base_url/non-OpenAI/custom schema/list fields. Overridable. |
| `system_prompt_header` as Tier 2 | Feb 2026 | Injects `# Context` between Role and Keys. Ignored when full `system_prompt` override is set. Part of cache key. |
| `web_search()` factory pattern | Feb 2026 | Returns async callable for FunctionStep. Not a step type. Graceful degradation on errors. |
| `run_if`/`skip_if` predicates on steps | Feb 2026 | Per-row conditional execution. `(row, prior_results) -> bool` signature for lambda friendliness. Mutually exclusive. Evaluated before cache check. Skipped rows use field defaults. No cache entries for skips. |
| `BatchCapableLLMClient` separate protocol | Mar 2026 | Adding `submit_batch()` to `LLMClient` would break `@runtime_checkable` for custom clients. Separate protocol + `isinstance` check. |
| `build_messages()` / `parse_response()` on LLMStep | Mar 2026 | Batch path needs message building and response parsing without the API call. Extracted from `run()` as a pure refactor — `run()` delegates to both. |
| Cache keys identical for batch and realtime | Mar 2026 | Same row + same config = same cache key. Switching execution modes never invalidates cache. |
| Realtime fallback for failed batch rows | Mar 2026 | Failed batch rows retry via `step.run()` with full retry logic. Batch mode is strictly better. |
| `provider_kwargs` escape hatch on LLMStep | Mar 2026 | Users shouldn't be blocked when providers ship new features. Merged into API call at last moment. Not in cache key. |
| Anthropic prompt caching (automatic) | Mar 2026 | `cache_control: {"type": "ephemeral"}` on system messages. Column-oriented execution = same prompt × N rows = ideal caching. ~90% savings. |
| Google batch deferred | Mar 2026 | API less mature than OpenAI/Anthropic. Revisit when stable. |

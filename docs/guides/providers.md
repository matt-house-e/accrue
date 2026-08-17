# Providers

Accrue is provider-agnostic. It ships adapters for OpenAI, Anthropic, and Google, supports any OpenAI-compatible API, and exposes a protocol for custom providers.

## OpenAI (default)

No extra install needed. OpenAI is the default provider.

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze",
        fields={"summary": "Summarize the company"},
        model="gpt-4.1-mini",
    ),
])
```

Set `OPENAI_API_KEY` in your environment or pass `api_key=` directly.

OpenAI uses the Responses API natively, which enables web search and structured outputs (`json_schema` with `strict: true`). When `base_url` is set, it falls back to Chat Completions for compatibility with third-party APIs.

## Anthropic

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze",
        fields={"summary": "Summarize the company"},
        model="claude-sonnet-4-5-20250929",
    ),
])
```

The provider is auto-detected from the `claude-` model prefix. No explicit `client=` import needed.

**Install:** `pip install accrue[anthropic]`

**Auth:** Set `ANTHROPIC_API_KEY` (or pass `api_key=` to `LLMStep`).

**Custom client (optional):** For non-default Anthropic SDK options pass `client=AnthropicClient(...)` explicitly — overrides auto-detection.

**Prompt caching:** Accrue automatically adds `cache_control: {"type": "ephemeral"}` to system messages. No configuration required.

Anthropic reads a cached prefix only on an exact match, so Accrue splits every LLM prompt at the static/variable boundary and sends the two halves separately:

| Half | Contents | Varies per row? |
|---|---|---|
| `system` (cached) | Role, your `system_prompt_header` (or `system_prompt`), field spec key descriptions, output rules, and the `<field_specifications>` block | No |
| `user` | `<row_data>`, `<prior_results>`, the task instruction, and the closing reminder | Yes |

The first row of a step writes the cache entry (billed at 1.25x input); every subsequent row reads it at 0.1x. Savings therefore scale with how much of your prompt is static: a step with a large `system_prompt_header` and rich field specs saves most, a step whose prompt is almost entirely row data saves little. Earlier versions built the row's own JSON into the cached block, so the prefix changed on every call and the cache never hit — see [#107](https://github.com/matt-house-e/accrue/issues/107).

Two consequences worth knowing:

- Cache entries are scoped to the exact prefix, so changing `system_prompt_header`, field specs, or the model starts a fresh entry.
- With `max_workers > 1` on a cold cache, the workers in the first wave all start before any entry exists, so each writes its own. The cache settles from the second wave onward.

**Cache token accounting:** Anthropic reports cache-creation and cache-read tokens separately from `input_tokens`. Accrue carries them as `cache_write_tokens` and `cache_read_tokens` on `UsageInfo`, aggregates them into `StepUsage` and `CostSummary`, and counts them in `total_tokens`, so cost reports reflect the real bill. The three input classes are priced differently (1.0x base input, 1.25x cache write, 0.1x cache read), so the cache fields stay separate from `prompt_tokens`. Providers and models that do not report cache tokens simply yield zeros.

**Structured outputs:** Uses constrained decoding (the Anthropic equivalent of `json_schema`). Auto-detected when using dict fields.

## Google

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze",
        fields={"summary": "Summarize the company"},
        model="gemini-2.5-flash",
    ),
])
```

The provider is auto-detected from the `gemini-` model prefix. No explicit `client=` import needed.

**Install:** `pip install accrue[google]`

**Auth:** Set `GOOGLE_API_KEY` (or pass `api_key=` to `LLMStep`).

**Custom client (optional):** Pass `client=GoogleClient(...)` to override auto-detection with non-default options.

**Grounding:** Supports native Google Search grounding via the `grounding` parameter on LLMStep.

## OpenAI-compatible APIs

Any API that implements the OpenAI Chat Completions format works via `base_url`. This includes Ollama, Groq, Together, Fireworks, vLLM, and others.

```python
# Ollama (local)
LLMStep("analyze", fields={...}, model="llama3", base_url="http://localhost:11434/v1")

# Groq
LLMStep("analyze", fields={...}, model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1", api_key="gsk_...")

# Together AI
LLMStep("analyze", fields={...}, model="meta-llama/Llama-3-70b-chat-hf",
        base_url="https://api.together.xyz/v1", api_key="...")
```

When `base_url` is set, structured output auto-detection falls back to `json_object` (since third-party APIs may not support `json_schema`). Override with `structured_outputs=True` if your API supports it.

## Feature comparison

| Feature | OpenAI | Anthropic | Google | OpenAI-compatible |
|---------|--------|-----------|--------|-------------------|
| Structured outputs | `json_schema` | Constrained decoding | `json_schema` | `json_object` |
| Grounding (web search) | Yes | Yes | Yes | No |
| Batch API | Yes | Yes | No | No |
| Prompt caching | No | Automatic | No | No |

## Custom providers

Implement the `LLMClient` protocol -- a single async method:

```python
from accrue.providers import LLMClient, LLMResponse
from accrue.schemas.base import UsageInfo
from typing import Any


class MyClient:
    """Custom LLM provider adapter."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse:
        # Call your API here
        content = await my_api_call(messages, model, temperature, max_tokens)
        return LLMResponse(
            content=content,
            usage=UsageInfo(input_tokens=100, output_tokens=50),
        )


# Use it
LLMStep("analyze", fields={...}, client=MyClient())
```

The protocol is structural (duck typing). No inheritance required -- just implement the `complete` method with the right signature.

For batch API support, also implement the `BatchCapableLLMClient` protocol, which adds `submit_batch()`, `poll_batch()`, and `cancel_batch()` methods.

## provider_kwargs

The `provider_kwargs` parameter is an escape hatch for provider-specific features not yet exposed as first-class parameters. The dict is merged into the API call.

```python
# Anthropic extended thinking
LLMStep("deep_analysis",
    fields={"analysis": "Detailed strategic analysis"},
    model="claude-sonnet-4-5-20250929",
    provider_kwargs={"thinking": {"type": "adaptive"}},
)

# OpenAI reduced effort
LLMStep("quick_check",
    fields={"is_valid": "Is this a valid company?"},
    provider_kwargs={"effort": "low"},
)

# OpenAI disable storage
LLMStep("sensitive",
    fields={"pii_check": "Check for PII"},
    provider_kwargs={"store": False},
)
```

`provider_kwargs` works for both realtime and batch execution. They are not included in cache keys, so you can iterate on provider-specific features without invalidating your cache.

## Gotchas

- Anthropic and Google require separate installs (`accrue[anthropic]`, `accrue[google]`). Import errors are deferred to instantiation, so you get a clear error message at runtime, not at import time.
- When using `base_url`, the OpenAI adapter switches from Responses API to Chat Completions. Features that rely on Responses API (native web search) are not available.
- Structured outputs are disabled automatically when grounding is active on Anthropic and Google, because those providers do not support tool use and structured output simultaneously.
- The `client=` parameter overrides `api_key=` and `base_url=`. If you pass a pre-configured client, those other parameters are ignored.

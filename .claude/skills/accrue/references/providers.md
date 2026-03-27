# Model Selection & Provider Configuration

## Model Comparison for Enrichment

> **Last verified: 2026-03-26.** Pricing changes frequently — confirm at [OpenAI](https://openai.com/api/pricing/), [Anthropic](https://docs.anthropic.com/en/docs/about-claude/pricing), [Google](https://ai.google.dev/gemini-api/docs/pricing) before relying on estimates.

| Model | Provider | Cost (in/out per MTok) | Context | Best For |
|-------|----------|----------------------|---------|----------|
| `gpt-4.1-nano` | OpenAI | $0.10 / $0.40 | 1M | Simple classification, tagging |
| `gpt-4.1-mini` | OpenAI | $0.40 / $1.60 | 1M | **Default for most enrichment** |
| `gpt-4.1` | OpenAI | $2.00 / $8.00 | 1M | Complex inference, high-quality extraction |
| `claude-haiku-4-5` | Anthropic | $1.00 / $5.00 | 200K | High-volume structured extraction |
| `claude-sonnet-4-6` | Anthropic | $3.00 / $15.00 | 1M | Quality + reasoning, adaptive thinking |
| `claude-opus-4-6` | Anthropic | $5.00 / $25.00 | 1M | Maximum quality, complex analysis |
| `gemini-2.5-flash` | Google | $0.30 / $2.50 | 1M | Google Search grounding (note: output price includes thinking tokens) |
| `o4-mini` | OpenAI | $1.10 / $4.40 | 200K | Tasks requiring step-by-step reasoning |

### Selection Decision Tree

1. **Is it simple classification/tagging?** → `gpt-4.1-nano` (cheapest)
2. **Standard enrichment (company research, qualification)?** → `gpt-4.1-mini` (default)
3. **Needs reasoning or nuance?** → `gpt-4.1` or `claude-sonnet-4-6`
4. **Needs latest real-time data?** → Any model + `grounding=True`
5. **High-volume (>10K rows), cost-sensitive?** → `gpt-4.1-nano` or `gemini-2.5-flash`
6. **Anthropic preferred?** → `claude-haiku-4-5` (volume) or `claude-sonnet-4-6` (quality)

## Provider Setup

### OpenAI (default, no extra install)

```python
# Set OPENAI_API_KEY in .env (auto-loaded by Accrue)
LLMStep("analyze", fields={...}, model="gpt-4.1-mini")
```

### Anthropic

```bash
pip install accrue[anthropic]
```

```python
from accrue.providers import AnthropicClient
LLMStep("analyze", fields={...}, model="claude-sonnet-4-6", client=AnthropicClient())
```

Set `ANTHROPIC_API_KEY` in .env.

**Corporate proxy / custom SSL:** If behind a corporate proxy (e.g., Zscaler) that re-signs HTTPS, pass a custom `httpx` client with the corporate CA bundle. **Always pass `api_key` explicitly** when using a custom `http_client` — the SDK won't auto-read env vars with a custom client:

```python
import os, ssl, httpx
from dotenv import load_dotenv
from accrue.providers import AnthropicClient

load_dotenv(override=True)  # Load .env BEFORE constructing clients

ssl_ctx = ssl.create_default_context(cafile="ca-bundle.pem")
client = AnthropicClient(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    http_client=httpx.AsyncClient(verify=ssl_ctx),
)
LLMStep("analyze", fields={...}, model="claude-sonnet-4-6", client=client)
```

### Google

```bash
pip install accrue[google]
```

```python
from accrue.providers import GoogleClient
LLMStep("analyze", fields={...}, model="gemini-2.5-flash", client=GoogleClient())
```

Set `GOOGLE_API_KEY` in .env.

### OpenAI-Compatible (Ollama, Groq, Together, etc.)

```python
LLMStep("analyze", fields={...}, model="llama3",
        base_url="http://localhost:11434/v1")
```

Note: `base_url` switches from Responses API to Chat Completions. No web search. Structured outputs fall back to `json_object`.

## Provider-Specific Features (provider_kwargs)

### OpenAI

```python
# Reasoning effort (ONLY on reasoning models: o3, o4-mini, gpt-5.x — NOT gpt-4.1)
provider_kwargs={"reasoning_effort": "medium"}  # low | medium | high

# Disable storage (privacy-sensitive data)
provider_kwargs={"store": False}
```

### Anthropic

```python
# Adaptive thinking (Opus 4.6 and Sonnet 4.6 only)
# Use for complex inference steps; skip for simple extraction
# IMPORTANT: temperature MUST be set to 1 when thinking is enabled
provider_kwargs={
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "low"},  # low | medium | high | max (Opus only)
}
# When using adaptive thinking, set temperature=1 on the LLMStep

# Hide thinking output (faster time-to-first-token, same cost)
provider_kwargs={
    "thinking": {"type": "adaptive", "display": "omitted"},
}
```

**Prompt caching** is automatic in Accrue — `cache_control: {"type": "ephemeral"}` is added to system messages. ~90% savings on repeated system prompt tokens.

Cache thresholds (minimum tokens for caching to activate):
- Opus 4.6: 4,096 tokens
- Sonnet 4.6: 2,048 tokens
- Sonnet 4.5: 1,024 tokens

### Google

No special provider_kwargs currently. Grounding uses Google Search natively.

## Grounding / Web Search

### When to Enable

| Scenario | Grounding? | Why |
|----------|-----------|-----|
| Time-sensitive data (funding, news, headcount) | Yes | Model knowledge has a cutoff |
| Niche/small companies (<100 employees) | Yes | Model unlikely to know them |
| Well-known companies (FAANG, unicorns) | No | Model knowledge is sufficient |
| Classification from provided context | No | No external data needed |
| Text transformation/extraction | No | Working with existing text |
| Verifying factual claims | Yes | Ground truth matters |

### Configuration

```python
# Simple: enable defaults
LLMStep("research", fields={...}, grounding=True)

# Advanced: restrict to specific domains
LLMStep("research", fields={...}, grounding=GroundingConfig(
    allowed_domains=["crunchbase.com", "linkedin.com", "pitchbook.com"],
    max_searches=3,
))
```

### Provider Behavior

| Provider | Grounding Implementation | Structured Outputs |
|----------|------------------------|-------------------|
| OpenAI | Responses API web search | Works normally |
| Anthropic | `web_search_20250305` server tool (real web search) | **Disabled** (incompatible with web search tool) |
| Google | Google Search | **Disabled** (incompatible) |

**All three providers support real web search.** The tradeoff: on Anthropic and Google, enabling grounding disables structured outputs. The LLM returns JSON in a text block instead of structured tool output. Accrue handles parsing, but quality may be slightly lower. Flag this to the user.

### Citations

When grounding is active, citations are stored in a `sources` field by default:

```python
LLMStep("research", fields={...}, grounding=True, sources_field="sources")
# sources_field=None to suppress citations
```

## Batch API

50% cost savings. Available on OpenAI and Anthropic.

```python
LLMStep("enrich", fields={...}, batch=True)
```

- 24-hour processing window (not for time-critical work)
- Auto-chunks at 50,000 requests
- Cache-aware: only uncached rows go to batch
- Failed batch rows automatically fall back to realtime
- Configure via `EnrichmentConfig(batch_poll_interval=60, batch_timeout=86400)`

### When to Use Batch

| Row Count | Latency OK? | Recommendation |
|-----------|------------|---------------|
| <100 | — | Realtime (batch overhead not worth it) |
| 100-500 | Yes | Consider batch for cost savings |
| >500 | Yes | Definitely batch |
| Any | No, need results fast | Realtime |

## Cost Estimation

**Formula:** `rows x avg_input_tokens x input_price + rows x avg_output_tokens x output_price`

**Typical enrichment tokens per row:**
- Input: ~500 tokens (system prompt + row data + field specs)
- Output: ~200 tokens (structured JSON response)
- With grounding: 2-3x input tokens (search results injected)

**Quick estimates for 1000 rows, single step:**

| Model | Est. Cost (realtime) | Est. Cost (batch) |
|-------|---------------------|-------------------|
| gpt-4.1-nano | ~$0.10 | ~$0.05 |
| gpt-4.1-mini | ~$0.50 | ~$0.25 |
| gpt-4.1 | ~$2.60 | ~$1.30 |
| claude-haiku-4-5 | ~$1.50 | ~$0.75 |
| claude-sonnet-4-6 | ~$4.50 | ~$2.25 |

Multiply by number of steps. Gate patterns reduce cost proportionally to the pass-through rate.

## Feature Comparison

| Feature | OpenAI | Anthropic | Google | OpenAI-Compatible |
|---------|--------|-----------|--------|-------------------|
| Structured outputs | json_schema | Constrained decoding | json_schema | json_object |
| Grounding / web search | Yes | Yes (citations) | Yes (Google Search) | No |
| Batch API | Yes | Yes | No | No |
| Prompt caching | No | Automatic | No | No |
| Reasoning / thinking | o3/o4-mini/gpt-5.x | Adaptive thinking | No | Varies |

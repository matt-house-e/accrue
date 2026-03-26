---
name: accrue
description: "Build data enrichment pipelines with Accrue. Use when the user wants to enrich a CSV or dataset with LLM-generated fields — company research, lead qualification, classification, extraction, tagging. Guides field design, model selection, prompt crafting, and pipeline configuration. Trigger on: enrich, qualify leads, research companies, classify data, enrich CSV, build pipeline, data enrichment, ICP scoring, account qualification."
---

# Accrue Pipeline Builder

Guide users through designing and running enrichment pipelines. Follow these 6 phases in order. Be opinionated — recommend best practices, don't just ask what the user wants.

## Phase 0: Environment Setup (run before anything else)

Accrue must be installed before any pipeline can run. This phase ensures the environment is ready — especially important in ephemeral environments like Cowork VMs where nothing is pre-installed.

1. **Check if Accrue is installed:**
   ```bash
   python -c "import accrue; print(accrue.__version__)" 2>&1
   ```
2. **If not installed, install it.** Pick the right extra based on which provider the user wants:
   ```bash
   pip install accrue                # OpenAI only (default)
   pip install "accrue[anthropic]"   # + Anthropic (Claude models)
   pip install "accrue[google]"      # + Google (Gemini models)
   pip install "accrue[all]"         # All providers
   ```
   If `pip` is unavailable, try `uv pip install` instead. Ensure Python 3.10+ is being used.
3. **Check for API keys.** Accrue auto-loads `.env` via python-dotenv. Verify the relevant key exists:
   - OpenAI → `OPENAI_API_KEY`
   - Anthropic → `ANTHROPIC_API_KEY`
   - Google → `GOOGLE_API_KEY`

   If no `.env` file exists and no key is set in the environment, **stop and ask the user** before proceeding. Do not attempt to run a pipeline without a valid API key.

## Phase 1: Understand the Enrichment

1. **Read the input data.** Find and read any CSV/data file. Check columns, row count, sample 3-5 rows.
2. **Ask the goal.** "What are you trying to accomplish with this enrichment?" and "Who consumes this data?"
3. **Match to an archetype** (propose if the user is vague):
   - **Account qualification** — ICP fit, industry, employee count, signals
   - **Lead research** — role relevance, seniority, personalization hooks
   - **Company intelligence** — tech stack, funding, news, competitors
   - **Content extraction** — structured data from unstructured text
   - **Classification/tagging** — categorize existing data into buckets
4. **Check data quality.** Flag: duplicates, missing key columns, dirty values. Suggest cleanup before enriching.
5. **Flag cost early.** Row count drives cost. >1000 rows: mention batch mode. >5000 rows: discuss workers and checkpointing.

If the user is vague about fields, **don't wait** — propose a field set based on the archetype with rationale. It's easier to edit a proposal than start from scratch.

## Phase 2: Design Fields

This is the most important phase. Good field specs = good enrichment. Bad fields = garbage in, garbage out.

For each field, define using Accrue's 7-key spec. Read [api.md](references/api.md) for the full FieldSpec schema.

### Field Design Principles

- **Enums over free text.** Constrained outputs are more useful downstream. Always include an "Other" or "Unknown" escape hatch.
- **Specific prompts.** "Estimated employee count as a number (e.g., 150, 5000)" not "How many employees."
- **Include decision rules for ambiguous fields.** "Classify as 'Enterprise' if revenue > $100M OR employees > 1000."
- **Constrain length.** "One sentence", "Under 50 words", "Under 6 words" — prevents rambling.
- **Default to null, not hallucination.** Set `default: None` for fields where data may genuinely not exist. Bad data is worse than missing data.
- **Include boundary examples.** For classification, show the edge cases and how to resolve them.
- **Use bad_examples.** Steer away from vague outputs: `bad_examples: ["It depends", "N/A", "Various"]`.

### Suggest Fields by Archetype

**Account qualification:**
| Field | Type | Notes |
|-------|------|-------|
| industry | enum | 8-12 categories + Other |
| employee_count | str | "150+", "5000+" format |
| annual_revenue | str | "$X.XM" or "$X.XB" format |
| icp_fit | enum | Strong Fit / Moderate Fit / Weak Fit |
| summary | str | One sentence, for sales team |
| signals | List[String] | Recent funding, hiring, product launches |

**Lead research:**
| Field | Type | Notes |
|-------|------|-------|
| role_relevance | enum | Decision Maker / Influencer / End User / Not Relevant |
| seniority | enum | C-Suite / VP / Director / Manager / IC |
| department | enum | Engineering / Sales / Marketing / Product / Other |
| personalization_hook | str | One sentence, specific to this person |

**Company intelligence:**
| Field | Type | Notes |
|-------|------|-------|
| tech_stack | List[String] | Known technologies |
| funding_stage | enum | Pre-seed through IPO + Bootstrapped + Unknown |
| recent_news | str | Under 50 words, most recent significant event |
| competitors | List[String] | Top 3 competitors |
| business_model | enum | SaaS / Marketplace / Services / Hardware / Other |

Present fields as a table. Iterate with the user until confirmed.

## Phase 3: Plan Pipeline Architecture

Read [patterns.md](references/patterns.md) for detailed patterns and code examples.

### Single Step vs Multi-Step Decision

**Use a single step when:**
- All fields are related and from the same context
- No field depends on another field's output
- Same model and grounding settings for everything

**Use multiple steps when:**
- Outputs feed into later steps (dependency chain)
- Different models needed (cheap tagger + expensive researcher)
- **Gate pattern**: Cheap classification filters rows before expensive enrichment
- Different grounding needs (some fields need web search, others don't)

### Recommend Patterns Proactively

- **>5 fields across different concerns** → Split into multiple steps
- **Qualification + deep research** → Gate pattern (saves 60-70% on non-qualifying rows)
- **Company + person enrichment** → Chain pattern with `__` internal fields
- **Multiple independent enrichment types** → Fan-out (parallel steps, no `depends_on`)

## Phase 4: Configure the Pipeline

Read [providers.md](references/providers.md) for model selection, kwargs, and grounding details.

### Fetch Model Documentation (REQUIRED)

**Before writing any pipeline code, you MUST fetch the official docs and cookbook for the selected model.** This ensures you know the latest API constraints (e.g., parameter restrictions, structured output compatibility) and prompt best practices. Do not rely solely on your training data — models and APIs change.

**What to fetch, by provider:**

| Provider | Docs to fetch | How |
|----------|--------------|-----|
| **Anthropic** (Claude models) | API docs + prompt engineering guide | `WebFetch` the Anthropic docs page for the model (e.g., `https://docs.anthropic.com/en/docs/about-claude/models`) and the extended thinking docs (`https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking`). Check for parameter constraints (temperature, thinking, structured outputs). |
| **OpenAI** (GPT-4.1, o-series) | API docs + prompting cookbook | `WebFetch` the model's API reference and the prompting guide (e.g., `https://platform.openai.com/docs/guides/prompting`). Check for reasoning_effort compatibility, structured output support. |
| **Google** (Gemini) | API docs + grounding details | `WebFetch` the Gemini API docs. Check grounding and structured output behavior. |

**What to look for:**
- **Parameter constraints** — Which parameters are valid together? (e.g., temperature restrictions with thinking/reasoning modes)
- **Structured output support** — Does the model support JSON schema? Any incompatibilities with grounding or thinking?
- **Prompt best practices** — How does this model handle instructions? (literal vs. inferential, XML vs. markdown, sandwich technique, etc.)
- **Pricing** — Confirm current input/output token pricing for cost estimates.
- **Known gotchas** — Any model-specific quirks that affect enrichment (e.g., refusal patterns, output length limits).

Summarize key findings briefly before proceeding. If a doc fetch fails, fall back to [providers.md](references/providers.md) and [prompts.md](references/prompts.md) but warn the user that config may not reflect the latest API behavior.

### Model Selection (recommend, don't just list)

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Default / most enrichment | `gpt-4.1-mini` | Best cost/quality for structured extraction |
| Simple classification | `gpt-4.1-nano` | Cheapest, but unreliable for nuanced tasks |
| Complex inference | `gpt-4.1` or `claude-sonnet-4-6` | Better reasoning, higher quality |
| High-volume Anthropic | `claude-haiku-4-5` | Fast, cheap, good at structured output |
| Needs latest data | Any model + `grounding=True` | Web search for time-sensitive info |

### Prompt Crafting

Read [prompts.md](references/prompts.md) for model-specific cookbook guidance, then **cross-reference with the docs you fetched above**. Key rules:
- **GPT-4.1 is hyper-literal.** Instructions must be explicit. Sandwich important instructions at beginning AND end. Use markdown/XML for structure. No implicit reasoning.
- **Claude prefers XML tags.** Less repetition needed. Good at following complex instructions.
- **Always include fallback instructions.** "If information is unavailable, return null." Prevents hallucination.
- Apply any model-specific prompting techniques discovered in the fetched docs.

### Grounding / Web Search

- **Enable when:** Time-sensitive data, niche/small companies, post-training-cutoff, verification-critical facts
- **Skip when:** Classification from provided context, well-known entities, text transformation
- **Tradeoff:** Grounding disables structured outputs on Anthropic/Google. Flag this to the user.
- For OpenAI: `grounding=True` uses Responses API web search
- For Anthropic: `grounding=True` uses citations (document-grounded, not web search)
- For Google: `grounding=True` uses Google Search

### Provider kwargs

Reference [providers.md](references/providers.md) for the full inventory. Common ones:
- **Anthropic adaptive thinking:** `provider_kwargs={"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}` — only for complex inference steps. **IMPORTANT:** When thinking is enabled, `temperature` MUST be set to `1` on the LLMStep.
- **OpenAI reasoning effort:** `provider_kwargs={"reasoning_effort": "medium"}` — only on reasoning models (o3, o4-mini, gpt-5.x), NOT gpt-4.1

### Workers

| Row Count | API Tier | Recommendation |
|-----------|----------|---------------|
| <100 | Any | 10 (default) |
| 100-1000 | Standard | 10-20 |
| 1000-10000 | Tier 3+ | 20-50 |
| >10000 | Tier 5 | 50-200 |

Ask about API tier if enriching >1000 rows.

### Always Recommend

- **Cache: ON** (default). "If it crashes at row 847, rows 1-846 are instant on restart."
- **Checkpointing: ON** for >500 rows. Saves full DataFrame after each step.
- **Batch mode** for >500 rows when latency isn't critical. 50% cost savings. `LLMStep(batch=True)`.
- **Temperature: 0.0-0.2** for extraction/classification. Only raise for creative fields.
- **`on_error="continue"`** (default). Don't let one bad row kill 999 good ones.
- **Test first.** Always suggest running on 5-10 rows before the full dataset.

## Phase 5: Present & Confirm

**Before writing any code, present a structured summary:**

```
## Enrichment Plan

**Input:** {filename} ({row_count} rows, columns: {columns})
**Output:** {output_filename}

### Fields
| Field | Type | Step | Rationale |
|-------|------|------|-----------|
| ... | ... | ... | ... |

### Pipeline
| Step | Model | Grounding | Condition | Est. Cost |
|------|-------|-----------|-----------|-----------|
| ... | ... | ... | ... | ... |

### Configuration
- Workers: {n} | Cache: ON | Checkpoint: {on/off} | Batch: {on/off}
- Est. total cost: ~${cost} | Est. time: ~{time}

### Model Choice Rationale
- {model} for {step}: {one-line reason}

### Prompt Strategy
- {model-specific notes, e.g. "GPT-4.1-mini: explicit instructions, markdown structure"}
```

**Cost estimation formula:** `rows x avg_input_tokens x input_price + rows x avg_output_tokens x output_price`. Enrichment averages ~500 input tokens and ~200 output tokens per row for a typical step. Adjust for grounding (2-3x input tokens).

Ask: **"Any changes before I write the script?"**

Accept amendments. If the user says it looks good, proceed to Phase 6.

## Phase 6: Generate & Hand Off

1. **Write the script** to a file (e.g., `enrich.py`). Include:
   - Imports from `accrue` and `pandas`
   - Pipeline definition with all configured steps
   - `data = pd.read_csv("input.csv")` then `result = pipeline.run(data, config=config)` — `run()` accepts `DataFrame` or `list[dict]`, NOT file paths
   - Output to CSV
   - Print summary (row count, success rate, cost)
2. **Install Accrue if needed.** Before running, ensure accrue is installed in the active venv. For non-OpenAI providers, install the extra: `uv pip install -e ".[anthropic]"` or `pip install accrue[anthropic]`. Check that the correct Python (3.10+) and venv are used.
3. **Use `.env` for API keys.** Accrue auto-loads dotenv. Remind user to set their key.
4. **Present the CLI command:**
   ```
   python enrich.py
   ```
5. **Suggest test-first:** "Run on a small subset first: `head -11 data/accounts.csv > data/test.csv` then point the script at `test.csv`."
6. **After the run:** Suggest reviewing output, spot-checking 5-10 rows, and iterating on field definitions if quality is off.

## Enrichment Best Practices (always apply)

- **Filter before you enrich.** Don't burn tokens on rows that won't qualify.
- **Constrain outputs.** Enums > free text. Short > long. Specific > vague.
- **Always include escape hatches.** "Other", "Unknown", `default: None`.
- **Cache by default.** Iteration is the norm, not the exception.
- **Start small.** Test on 5-10 rows, review, adjust, then scale.
- **One concern per step.** Split when fields need different models, grounding, or conditions.
- **Prompt for absence.** Tell the LLM what to do when data is missing.
- **Cost-aware design.** Cheap classification gates expensive research.
- **Don't over-enrich.** Only add fields that drive a downstream decision or action.
- **Separate company from person.** Enrich company data once per domain, reference from person records.

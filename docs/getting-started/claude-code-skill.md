# Using the Claude Code Skill

Accrue ships with a built-in [Claude Code](https://claude.ai/claude-code) skill that guides you through building enrichment pipelines interactively. Instead of writing pipeline code from scratch, you describe what you want and the skill designs fields, picks models, estimates costs, and writes a production-ready script.

## Prerequisites

- [Claude Code](https://claude.ai/claude-code) installed
- Working directory is a clone of the Accrue repo (the skill lives in `.claude/skills/accrue/`)
- An API key for your chosen provider (OpenAI, Anthropic, or Google)

## Quick Start

Open Claude Code in a project that has Accrue installed, then invoke the skill:

```
> /accrue
> I have 500 companies in accounts.csv and need to qualify them for ICP fit
```

The skill walks you through 6 phases:

### Phase 1: Understand

The skill reads your input data, checks columns and row count, and asks about your goal. If you're vague, it proposes an enrichment archetype (account qualification, lead research, company intelligence, etc.) and suggests fields to get the conversation moving.

### Phase 2: Design Fields

The most important phase. The skill proposes fields using Accrue's [7-key field spec](../guides/field-specifications.md) system and iterates with you:

```
Proposed fields for account qualification:

| Field          | Type              | Notes                           |
|----------------|-------------------|---------------------------------|
| industry       | enum (10 + Other) | ICP segmentation                |
| employee_count | str               | "500+" format                   |
| icp_fit        | enum (3 levels)   | Strong Fit / Moderate / Weak    |
| summary        | str               | One sentence, under 25 words    |

Does this look right? Want to add, remove, or change anything?
```

The skill applies enrichment best practices: enums over free text, escape hatches for ambiguity, length constraints, boundary examples for classification, and null defaults where data may not exist.

### Phase 3: Plan Pipeline

The skill decides whether your enrichment needs one step or multiple, and recommends patterns:

- **Single step** for straightforward enrichment
- **Gate pattern** when cheap classification can filter rows before expensive research
- **Chain pattern** when company research feeds into person research
- **Fan-out** when independent enrichment types can run in parallel

### Phase 4: Configure

The skill picks settings with rationale:

- **Model** -- `gpt-4.1-mini` for most enrichment, cheaper/bigger models when appropriate
- **Grounding** -- Web search for time-sensitive or niche data, skipped for classification
- **Workers** -- Based on row count and API tier
- **Cache & checkpoint** -- Always recommended
- **Batch mode** -- For large datasets where latency isn't critical

It references model-specific prompt cookbooks (GPT-4.1 is hyper-literal; Claude prefers XML) and recommends `provider_kwargs` like Anthropic adaptive thinking when relevant.

### Phase 5: Present & Confirm

Before writing any code, the skill presents a structured summary:

```
## Enrichment Plan

Input: accounts.csv (500 rows)
Output: enriched_accounts.csv

### Pipeline
| Step             | Model         | Grounding | Condition              | Est. Cost |
|------------------|---------------|-----------|------------------------|-----------|
| classify         | gpt-4.1-nano  | No        | All rows               | ~$0.05    |
| deep_research    | gpt-4.1-mini  | Web search| icp_fit == "Strong Fit"| ~$0.12    |

### Configuration
Workers: 10 | Cache: ON | Checkpoint: ON | Batch: No
Est. total: ~$0.17 | Est. time: ~60s

Any changes before I write the script?
```

### Phase 6: Hand Off

The skill writes the script, suggests testing on 5-10 rows first, and gives you the run command:

```bash
python enrich.py
```

## What the Skill Knows

The skill has access to reference material covering:

- **[API surface](../../.claude/skills/accrue/references/api.md)** -- Every Pipeline, LLMStep, FunctionStep, and Config parameter
- **[Prompt cookbook](../../.claude/skills/accrue/references/prompts.md)** -- Model-specific prompt guidance, enrichment templates, anti-patterns
- **[Provider details](../../.claude/skills/accrue/references/providers.md)** -- Model comparison, cost tables, provider_kwargs, grounding behavior
- **[Pipeline patterns](../../.claude/skills/accrue/references/patterns.md)** -- Gate, fan-out, chain, conditional, batch, incremental enrichment

## Example Session

```
> /accrue

> I have a CSV of 200 SaaS companies with name and website.
> I need to figure out which ones are a good fit for our developer tools product.

The skill will:
1. Read your CSV, note it has 200 rows with company_name and website columns
2. Propose an "account qualification" field set (industry, size, tech signals, ICP fit)
3. Recommend a 2-step gate pattern: cheap classifier → detailed research on strong fits
4. Pick gpt-4.1-mini with grounding for the research step
5. Present a cost estimate (~$0.30) and configuration summary
6. Write enrich.py with caching enabled, suggest testing on 10 rows first
```

## Tips

- **Be specific about your goal.** "Qualify for ICP fit" gets better results than "enrich this data."
- **Iterate on fields.** The first proposal is a starting point. Add domain-specific categories, adjust enums, tighten prompts.
- **Test first.** The skill always suggests running on a small subset before the full dataset.
- **Re-run cheaply.** With caching enabled, re-running after a crash or prompt tweak only processes new/changed rows.

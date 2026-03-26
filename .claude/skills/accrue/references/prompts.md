# Prompt Cookbook for Enrichment

## The S.P.I.C.E. Framework (adapted for Accrue)

Structure enrichment prompts with these components:

1. **Specifics** — What exact output you want (field names, formats, constraints)
2. **Persona** — Role context via `system_prompt_header` ("You are a B2B sales analyst...")
3. **Instructions** — Step-by-step extraction rules, decision criteria, fallback behavior
4. **Context** — Row data injected automatically via `{column_name}` placeholders
5. **Examples** — Few-shot via `examples` and `bad_examples` in field specs

In Accrue, most of this is handled by the field spec system. The LLM receives an auto-generated prompt containing all field definitions. Use `system_prompt_header` for additional context.

## Model-Specific Guidance

### GPT-4.1 / GPT-4.1-mini (OpenAI)

GPT-4.1 follows instructions **hyper-literally** compared to GPT-4o. Adapt accordingly:

- **Be explicit.** Don't rely on implicit understanding. If you want reasoning, ask for it.
- **Sandwich method.** Place critical instructions at BOTH the beginning and end of long context.
- **Markdown/XML > JSON** for document structure in prompts. JSON performs "particularly poorly" for long-context tasks.
- **No built-in reasoning.** Must explicitly prompt for chain-of-thought if needed.
- **Use the tools API** exclusively for function calling — don't inject tool descriptions into prompts.
- **Structured outputs work natively** via `json_schema` with `strict: true`. Accrue handles this automatically.

**Good GPT-4.1 enrichment prompt style:**
```python
system_prompt_header="""You are a B2B company analyst. For each company, extract structured data.

RULES:
- If data is unavailable, return null — never guess
- Employee counts should be estimates with + suffix (e.g., "500+")
- Industry must be one of the enum values provided
- Summary must be exactly one sentence, under 30 words

IMPORTANT: Return null for any field where you lack confidence."""
```

### Claude (Anthropic)

- **XML tags preferred.** Claude responds well to `<instructions>`, `<context>`, `<output_format>` tags.
- **Less repetition needed.** Claude follows complex instructions well without sandwiching.
- **System prompt for persona.** Claude uses system prompt role-setting effectively.
- **Prompt caching is automatic** in Accrue — system messages get `cache_control`.

**Good Claude enrichment prompt style:**
```python
system_prompt_header="""<role>B2B company analyst</role>

<instructions>
Extract structured company data. Follow these rules:
- Return null when data is genuinely unavailable
- Employee counts: estimates with + suffix (e.g., "500+")
- Summary: exactly one sentence, under 30 words
</instructions>"""
```

### Gemini (Google)

- Follows OpenAI-style prompting conventions
- Strong at grounding with Google Search
- Use clear, direct instructions

## Enrichment Prompt Templates

### Account Qualification

```python
LLMStep("qualify",
    system_prompt_header="You are qualifying B2B SaaS accounts for sales outreach. Be concise and factual.",
    fields={
        "industry": {
            "prompt": "Primary industry category for this company",
            "enum": ["Fintech", "Developer Tools", "Cybersecurity", "Data & Analytics",
                     "Cloud Infrastructure", "HR Tech", "Sales & Marketing", "Healthcare",
                     "E-commerce", "Other"],
        },
        "employee_count": {
            "prompt": "Estimated total employee count. Use format like '500+' or '5000+'",
            "examples": ["150+", "2000+", "50000+"],
            "bad_examples": ["a lot", "large company", "unknown"],
            "default": None,
        },
        "icp_fit": {
            "prompt": "How well does this company fit as a B2B SaaS prospect? Strong Fit = enterprise with >500 employees in tech. Moderate = mid-market or adjacent industry. Weak = SMB, consumer, or no tech buying signals.",
            "enum": ["Strong Fit", "Moderate Fit", "Weak Fit"],
        },
        "summary": {
            "prompt": "One-sentence company summary for a sales team. Under 25 words.",
            "examples": ["Enterprise cloud security platform serving Fortune 500 companies"],
            "bad_examples": ["Stripe is a company that does payments and stuff"],
        },
    },
    model="gpt-4.1-mini",
)
```

### Lead Research

```python
LLMStep("research_lead",
    system_prompt_header="You are researching sales leads. Extract role and relevance data.",
    fields={
        "role_relevance": {
            "prompt": "Is this person relevant to a B2B software sale? Decision Maker = budget authority. Influencer = technical evaluator. End User = daily user. Not Relevant = wrong department or role.",
            "enum": ["Decision Maker", "Influencer", "End User", "Not Relevant"],
        },
        "seniority": {
            "prompt": "Seniority level based on job title",
            "enum": ["C-Suite", "VP", "Director", "Manager", "IC", "Unknown"],
        },
        "personalization_hook": {
            "prompt": "One specific, non-generic thing about this person that could start a conversation. Reference their work, background, or company. Under 20 words.",
            "examples": ["Led the migration from on-prem to AWS at their previous company"],
            "bad_examples": ["They work in tech", "Experienced professional"],
            "default": None,
        },
    },
)
```

### Classification / Tagging

```python
LLMStep("classify",
    system_prompt_header="Classify each record into categories. Be consistent — same input should always produce same output.",
    fields={
        "category": {
            "prompt": "Primary content category",
            "enum": ["Product Update", "Thought Leadership", "Case Study", "Tutorial", "News", "Other"],
        },
        "sentiment": {
            "prompt": "Overall sentiment of the content",
            "enum": ["Positive", "Neutral", "Negative"],
        },
        "relevance_score": {
            "prompt": "How relevant is this to our product (1-5). 5 = directly about our space. 1 = unrelated.",
            "type": "Number",
        },
    },
    model="gpt-4.1-nano",  # Simple classification = cheapest model
    temperature=0.0,         # Deterministic
)
```

### Content Extraction

```python
LLMStep("extract",
    system_prompt_header="""Extract structured data from the provided text.
Rules: Only extract information explicitly stated. Never infer or guess.
If a field is not mentioned in the text, return null.""",
    fields={
        "company_name": {"prompt": "Company name mentioned in text", "default": None},
        "deal_size": {"prompt": "Dollar amount of deal/contract", "format": "$X.XM", "default": None},
        "key_people": {"prompt": "Names and titles of people mentioned", "type": "List[String]"},
    },
)
```

## Prompt Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| "Tell me about this company" | Too vague, LLM rambles | Specific field prompts with length constraints |
| No fallback instruction | LLM hallucinates when unsure | Add `default: None` and "return null if unavailable" |
| Ambiguous enum categories | Inconsistent classification | Add decision rules with boundary examples |
| 20+ fields in one step | Instruction neglect / contextual drift | Split into focused steps |
| "Be creative" in enrichment | Inconsistent outputs | Lower temperature, constrain format |
| No examples | LLM guesses at desired format | Add 2-3 examples and bad_examples |

## When to Use system_prompt vs system_prompt_header

- **`system_prompt_header`** (recommended): Adds context to the auto-generated prompt. Field specs still format automatically. Use for persona, rules, and domain knowledge.
- **`system_prompt`**: Fully replaces the auto-generated prompt. You must instruct the LLM to return JSON with field names. Use only when the auto-generated prompt doesn't fit your needs.

Most enrichment should use `system_prompt_header`. Only reach for `system_prompt` when you need full control.

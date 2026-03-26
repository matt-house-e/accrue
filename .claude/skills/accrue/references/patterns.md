# Pipeline Architecture Patterns

## Single-Step Enrichment

Use when all fields are related, same model, same grounding, no dependencies.

```python
from accrue import Pipeline, LLMStep, EnrichmentConfig

pipeline = Pipeline([
    LLMStep("qualify",
        system_prompt_header="You are qualifying B2B accounts for sales outreach.",
        fields={
            "industry": {"prompt": "Primary industry", "enum": [...]},
            "employee_count": {"prompt": "Estimated headcount", "examples": ["500+"]},
            "icp_fit": {"prompt": "ICP fit assessment", "enum": ["Strong", "Moderate", "Weak"]},
            "summary": {"prompt": "One-sentence summary, under 25 words"},
        },
        model="gpt-4.1-mini",
    ),
])

config = EnrichmentConfig(enable_caching=True, enable_checkpointing=True)
data = pd.read_csv("accounts.csv")
result = pipeline.run(data, config=config)
result.data.to_csv("enriched.csv", index=False)
```

## Gate Pattern (Cost Optimization)

Cheap classification step filters rows before expensive enrichment. If 30% of rows qualify, you save 70% on the expensive step.

```python
pipeline = Pipeline([
    # Step 1: Cheap classifier (runs on ALL rows)
    LLMStep("classify",
        fields={
            "tier": {
                "prompt": "Classify company tier. Enterprise = >1000 employees or >$100M revenue. Mid-Market = 100-1000 employees. SMB = <100 employees.",
                "enum": ["Enterprise", "Mid-Market", "SMB"],
            },
        },
        model="gpt-4.1-nano",  # Cheapest model for simple classification
        temperature=0.0,
    ),
    # Step 2: Expensive research (ONLY on Enterprise rows)
    LLMStep("deep_research",
        fields={
            "competitive_landscape": {"prompt": "Detailed competitive analysis", "default": "Not available"},
            "growth_signals": {"prompt": "Recent growth signals (funding, hiring, expansion)", "type": "List[String]", "default": None},
            "decision_makers": {"prompt": "Key decision makers and their roles", "type": "List[String]", "default": None},
        },
        depends_on=["classify"],
        run_if=lambda row, prior: prior.get("tier") == "Enterprise",
        model="gpt-4.1",
        grounding=True,  # Web search for current data
    ),
])
```

**Cost math:** 1000 rows. Step 1: 1000 x gpt-4.1-nano = ~$0.10. Step 2 (30% qualify): 300 x gpt-4.1 + grounding = ~$1.56. **Total: ~$1.66** vs ~$5.20 without gating.

## Fan-Out (Parallel Independent Steps)

Steps without `depends_on` relationships run in parallel automatically.

```python
pipeline = Pipeline([
    # These 3 steps run in parallel (no depends_on between them)
    LLMStep("firmographics",
        fields={
            "industry": {"prompt": "Primary industry", "enum": [...]},
            "employee_count": {"prompt": "Estimated headcount"},
            "hq_location": {"prompt": "Headquarters city and country"},
        },
        model="gpt-4.1-mini",
    ),
    LLMStep("technographics",
        fields={
            "tech_stack": {"prompt": "Known technologies used", "type": "List[String]"},
            "cloud_provider": {"prompt": "Primary cloud (AWS/GCP/Azure/Other)", "enum": [...]},
        },
        model="gpt-4.1-mini",
        grounding=True,  # Tech stack needs current data
    ),
    LLMStep("news",
        fields={
            "recent_news": {"prompt": "Most recent significant company news, under 50 words", "default": None},
            "funding_stage": {"prompt": "Latest funding round", "enum": ["Seed", "Series A", "Series B", "Series C+", "IPO", "Bootstrapped", "Unknown"]},
        },
        model="gpt-4.1-mini",
        grounding=True,
    ),
    # This step waits for all 3 above
    LLMStep("synthesize",
        fields={
            "icp_fit": {"prompt": "Based on firmographics, tech stack, and news, assess overall ICP fit", "enum": ["Strong", "Moderate", "Weak"]},
            "outreach_angle": {"prompt": "Suggested sales angle based on all available data, one sentence"},
        },
        depends_on=["firmographics", "technographics", "news"],
        model="gpt-4.1",  # Needs reasoning to synthesize
    ),
])
```

## Chain Pattern (Internal Fields)

Pass context between steps using `__` prefixed internal fields. These are available to downstream steps but filtered from final output.

```python
pipeline = Pipeline([
    LLMStep("company_research",
        fields={
            "__company_context": "Comprehensive company background including products, market position, recent developments. 2-3 paragraphs.",
            "industry": {"prompt": "Primary industry", "enum": [...]},
            "company_size": {"prompt": "Company size category", "enum": ["Startup", "SMB", "Mid-Market", "Enterprise"]},
        },
        model="gpt-4.1-mini",
        grounding=True,
    ),
    LLMStep("person_research",
        fields={
            "role_fit": {"prompt": "Based on the company context and this person's role, assess relevance to our product", "enum": ["High", "Medium", "Low"]},
            "personalization": {"prompt": "One sentence personalized to both the person and their company context"},
        },
        depends_on=["company_research"],
        model="gpt-4.1-mini",
        # __company_context is automatically available in prior_results
    ),
])
```

## Conditional Enrichment (run_if / skip_if)

### Skip Bad Data

```python
LLMStep("enrich",
    fields={"summary": "Company summary"},
    skip_if=lambda row, prior: not row.get("company_name") or not row.get("website"),
)
```

### Tiered Enrichment

```python
pipeline = Pipeline([
    LLMStep("classify",
        fields={"priority": {"prompt": "Priority tier", "enum": ["High", "Medium", "Low"]}},
    ),
    LLMStep("basic_info",
        fields={"summary": "One-sentence summary"},
        depends_on=["classify"],
        # Runs for all classified rows (Medium and High)
        skip_if=lambda row, prior: prior.get("priority") == "Low",
    ),
    LLMStep("premium_research",
        fields={
            "competitive_analysis": {"prompt": "Detailed competitive analysis", "default": "N/A"},
            "buying_signals": {"prompt": "Active buying signals", "type": "List[String]", "default": None},
        },
        depends_on=["classify"],
        run_if=lambda row, prior: prior.get("priority") == "High",
        model="gpt-4.1",
        grounding=True,
    ),
])
```

## FunctionStep for Pre/Post Processing

Use FunctionStep for non-LLM operations: data cleaning, API calls, computations.

```python
from accrue import Pipeline, LLMStep, FunctionStep, StepContext

async def clean_domain(row: dict, context: StepContext) -> dict:
    """Extract clean domain from website URL."""
    website = row.get("website", "")
    domain = website.replace("https://", "").replace("http://", "").replace("www.", "").strip("/")
    return {"clean_domain": domain}

async def compute_score(row: dict, context: StepContext) -> dict:
    """Compute composite score from enriched fields."""
    weights = {"Strong Fit": 3, "Moderate Fit": 2, "Weak Fit": 1}
    icp_score = weights.get(context.prior_results.get("icp_fit", ""), 0)
    has_signals = len(context.prior_results.get("signals", []) or []) > 0
    return {"composite_score": icp_score + (1 if has_signals else 0)}

pipeline = Pipeline([
    FunctionStep("clean", fn=clean_domain, fields=["clean_domain"]),
    LLMStep("enrich", fields={...}, depends_on=["clean"]),
    FunctionStep("score", fn=compute_score, fields=["composite_score"], depends_on=["enrich"]),
])
```

## Batch Mode

For large datasets where latency isn't critical. 50% cost savings.

```python
pipeline = Pipeline([
    LLMStep("enrich",
        fields={...},
        model="gpt-4.1-mini",
        batch=True,  # That's it
    ),
])

config = EnrichmentConfig(
    enable_caching=True,
    enable_checkpointing=True,
    batch_poll_interval=60.0,   # Check every 60s (default)
    batch_timeout=86400.0,      # 24h max (default)
)
data = pd.read_csv("large_dataset.csv")
result = pipeline.run(data, config=config)
```

- Cache-aware: only uncached rows are sent to batch
- Auto-chunks at 50,000 requests per batch
- Failed rows automatically fall back to realtime
- Available on OpenAI and Anthropic only

## Incremental Enrichment

Adding new fields to already-enriched data without re-running existing fields.

```python
# Round 1: Basic enrichment
pipeline_v1 = Pipeline([
    LLMStep("basic", fields={"industry": "...", "size": "..."}),
])
result = pipeline_v1.run("accounts.csv", config=EnrichmentConfig(enable_caching=True))
result.data.to_csv("enriched_v1.csv", index=False)

# Round 2: Add new fields (industry and size already cached)
pipeline_v2 = Pipeline([
    LLMStep("basic", fields={"industry": "...", "size": "...", "icp_fit": "..."}),
    # Cache will serve industry and size; only icp_fit hits the API
    # NOTE: Adding a field changes the cache key, so all rows will re-run.
    # To avoid this, add icp_fit as a separate step:
])

# Better approach: new step for new fields
pipeline_v2 = Pipeline([
    LLMStep("basic", fields={"industry": "...", "size": "..."}),  # 100% cache hits
    LLMStep("scoring", fields={"icp_fit": "..."}, depends_on=["basic"]),  # Only this runs
])
```

**Key insight:** Cache keys include the prompt + field spec + model. Changing any of these invalidates the cache for that step. To preserve cache, add new fields as new steps rather than expanding existing ones.

## Configuration Recommendations by Scale

| Rows | Workers | Cache | Checkpoint | Batch | Est. Time (gpt-4.1-mini) |
|------|---------|-------|------------|-------|--------------------------|
| 10 | 10 | Yes | No | No | ~5s |
| 100 | 10 | Yes | No | No | ~30s |
| 1,000 | 20 | Yes | Yes | Consider | ~5 min |
| 10,000 | 50 | Yes | Yes | Yes | ~50 min (batch: hours) |
| 50,000 | 100 | Yes | Yes | Yes | ~4 hours (batch: hours) |

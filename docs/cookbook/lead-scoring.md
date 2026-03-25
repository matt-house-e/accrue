# Lead Scoring

Score and qualify sales leads with conditional execution. This pipeline classifies
leads into tiers, deep-scores only the high-value ones, and generates personalized
outreach -- skipping low-priority leads entirely.

## Full Working Example

```python
from accrue import Pipeline, LLMStep, EnrichmentConfig

# -- Input data ---------------------------------------------------------------

leads = [
    {"name": "Jane Smith", "title": "VP Engineering", "company": "Acme Corp", "employees": 500},
    {"name": "Bob Jones", "title": "Intern", "company": "Startup Inc", "employees": 5},
    {"name": "Alice Chen", "title": "CTO", "company": "MegaCorp", "employees": 10000},
]

# -- Conditional predicates ---------------------------------------------------
# These receive (row, prior_results) and return a bool.
# prior_results contains outputs from all completed dependency steps.

def is_enterprise(row, prior_results):
    """Only run for leads classified as Enterprise tier."""
    return prior_results.get("tier") == "Enterprise"

def is_low_tier(row, prior_results):
    """Skip outreach for leads classified as Low tier."""
    return prior_results.get("tier") == "Low"

# -- Pipeline definition ------------------------------------------------------

pipeline = Pipeline([
    # Step 1: Classify every lead into a tier
    LLMStep(
        "classify",
        fields={
            "tier": {
                "prompt": (
                    "Classify this lead into a sales tier based on their title "
                    "and company size (employee count)"
                ),
                "type": "String",
                "enum": ["Enterprise", "Mid-Market", "SMB", "Low"],
                "examples": ["Enterprise"],
            },
            "icp_fit": {
                "prompt": "Rate how well this lead matches an ideal customer profile for B2B SaaS",
                "type": "String",
                "enum": ["Strong", "Moderate", "Weak"],
                "default": "Weak",
            },
        },
        model="gpt-4.1-mini",
    ),

    # Step 2: Deep score -- only runs for Enterprise leads
    # Rows that don't match run_if get default values (None or field default).
    LLMStep(
        "deep_score",
        fields={
            "authority_score": {
                "prompt": (
                    "Score this person's purchasing authority from 1-10 based on "
                    "their title and company size"
                ),
                "type": "Number",
                "format": "X/10",
                "examples": ["9"],
                "default": 0,
            },
            "urgency_signals": {
                "prompt": "List any signals that suggest urgency or near-term buying intent",
                "type": "String",
                "default": "Not evaluated",
            },
            "deal_size_estimate": {
                "prompt": "Estimate the potential annual contract value",
                "type": "String",
                "enum": ["$100K+", "$50K-$100K", "$10K-$50K", "Under $10K"],
                "default": "Under $10K",
            },
        },
        depends_on=["classify"],
        run_if=is_enterprise,
        model="gpt-4.1-mini",
    ),

    # Step 3: Generate outreach -- skips Low tier leads entirely
    # skip_if is the inverse of run_if. Use whichever reads more naturally.
    LLMStep(
        "outreach",
        fields={
            "email_subject": {
                "prompt": (
                    "Write a compelling email subject line for cold outreach to this "
                    "lead. Reference their role and company. Keep it under 60 characters."
                ),
                "type": "String",
                "default": "",
            },
            "opening_line": {
                "prompt": (
                    "Write a personalized opening line for a sales email. "
                    "Reference something specific about their role or company size."
                ),
                "type": "String",
                "default": "",
            },
        },
        depends_on=["deep_score"],
        skip_if=is_low_tier,
        model="gpt-4.1-mini",
    ),
])

# -- Run -----------------------------------------------------------------------

config = EnrichmentConfig(
    enable_caching=True,
    temperature=0.3,
    max_workers=10,
)

result = pipeline.run(leads, config=config)

# -- Inspect results -----------------------------------------------------------

for row in result.data:
    print(f"\n{'=' * 60}")
    print(f"Lead:           {row['name']} ({row['title']})")
    print(f"Company:        {row['company']} ({row['employees']} employees)")
    print(f"Tier:           {row['tier']}")
    print(f"ICP Fit:        {row['icp_fit']}")
    print(f"Authority:      {row['authority_score']}")
    print(f"Deal Size:      {row['deal_size_estimate']}")
    print(f"Email Subject:  {row['email_subject']}")
    print(f"Opening Line:   {row['opening_line']}")

print(f"\nSuccess rate: {result.success_rate:.0%}")

# Check which steps actually ran for each tier
for step_name, usage in result.cost.steps.items():
    print(f"\n  Step '{step_name}':")
    print(f"    Rows processed: {usage.rows_processed}")
    print(f"    Rows skipped:   {usage.rows_skipped}")
    print(f"    Tokens used:    {usage.total_tokens:,}")
```

## Expected Output

```
============================================================
Lead:           Jane Smith (VP Engineering)
Company:        Acme Corp (500 employees)
Tier:           Mid-Market
ICP Fit:        Moderate
Authority:      0                    # default -- run_if(Enterprise) was False
Deal Size:      Under $10K           # default
Email Subject:  Scaling Acme Corp's engineering team?
Opening Line:   Managing 500 engineers is no small feat -- here's how ...

============================================================
Lead:           Bob Jones (Intern)
Company:        Startup Inc (5 employees)
Tier:           Low
ICP Fit:        Weak
Authority:      0                    # default -- skipped
Deal Size:      Under $10K           # default
Email Subject:                       # default -- skip_if(Low) was True
Opening Line:                        # default

============================================================
Lead:           Alice Chen (CTO)
Company:        MegaCorp (10000 employees)
Tier:           Enterprise
ICP Fit:        Strong
Authority:      9                    # deep_score ran for Enterprise
Deal Size:      $100K+               # deep_score ran
Email Subject:  MegaCorp's next infrastructure move
Opening Line:   As CTO of a 10,000-person organization, you ...

Success rate: 100%

  Step 'classify':
    Rows processed: 3
    Rows skipped:   0
    Tokens used:    2,100

  Step 'deep_score':
    Rows processed: 1
    Rows skipped:   2
    Tokens used:    850

  Step 'outreach':
    Rows processed: 2
    Rows skipped:   1
    Tokens used:    1,400
```

## Key Concepts Demonstrated

**Conditional execution with `run_if`.** The `deep_score` step uses `run_if=is_enterprise`
so it only makes LLM calls for Enterprise-tier leads. Other rows get the field `default`
values without consuming any tokens.

**Conditional execution with `skip_if`.** The `outreach` step uses `skip_if=is_low_tier`
to skip Low-tier leads. `run_if` and `skip_if` are two ways to express the same idea --
use whichever reads more naturally. They are mutually exclusive on a single step.

**Default values for skipped rows.** When a row is skipped by a predicate, each field
gets its `default` value from the field spec. If no default is specified, the field
is set to `None`. This keeps the output schema consistent across all rows.

**Predicate signature.** Both `run_if` and `skip_if` receive `(row, prior_results)`:
- `row` is the original input data for that row
- `prior_results` is a dict of all outputs from completed dependency steps

**Cost savings.** In this example, `deep_score` only runs for 1 of 3 rows and `outreach`
runs for 2 of 3. With a dataset of thousands of leads where most are low-quality,
conditional steps can save 50-80% of API costs by skipping irrelevant rows.

## Filtering Results After Enrichment

```python
import pandas as pd

# Run with DataFrame input for easier filtering
df = pd.DataFrame(leads)
result = pipeline.run(df, config=config)

# Filter to actionable leads
qualified = result.data[result.data["tier"].isin(["Enterprise", "Mid-Market"])]
print(f"Qualified leads: {len(qualified)} of {len(result.data)}")
print(qualified[["name", "company", "tier", "email_subject"]])
```

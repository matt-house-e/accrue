# Content Analysis

Extract structured data from unstructured text. This pipeline parses job descriptions
to pull out skills, requirements, and metadata, then classifies each posting into
standardized categories.

## Full Working Example

```python
from accrue import Pipeline, LLMStep, EnrichmentConfig

# -- Input data ---------------------------------------------------------------

job_posts = [
    {
        "title": "Senior ML Engineer",
        "description": (
            "We're looking for an ML engineer with 5+ years experience in "
            "PyTorch, transformers, and MLOps. Must have experience deploying "
            "models at scale. Remote-friendly, based in SF."
        ),
    },
    {
        "title": "Frontend Developer",
        "description": (
            "Join our team building React applications with TypeScript. "
            "3+ years experience required. Experience with Next.js and "
            "Tailwind preferred. NYC office, hybrid."
        ),
    },
    {
        "title": "Data Engineering Lead",
        "description": (
            "Lead a team of 5 data engineers building our lakehouse on "
            "Databricks and Spark. Strong SQL, Python, and Airflow required. "
            "7+ years experience, 2+ years leading teams. Chicago, on-site."
        ),
    },
]

# -- Pipeline definition ------------------------------------------------------

pipeline = Pipeline([
    # Step 1: Extract structured entities from free text
    LLMStep(
        "extract",
        fields={
            "required_skills": {
                "prompt": "Extract all explicitly required technical skills and tools",
                "type": "List[String]",
                "examples": ["PyTorch, transformers, MLOps"],
            },
            "preferred_skills": {
                "prompt": "Extract skills mentioned as preferred or nice-to-have (not required)",
                "type": "List[String]",
                "examples": ["Next.js, Tailwind"],
            },
            "years_experience": {
                "prompt": "Extract the minimum years of experience required",
                "type": "Number",
                "examples": ["5"],
                "default": 0,
            },
            "location": {
                "prompt": "Extract the job location (city or region)",
                "type": "String",
                "examples": ["San Francisco, CA"],
                "default": "Not specified",
            },
            "work_model": {
                "prompt": "Determine the work arrangement",
                "type": "String",
                "enum": ["Remote", "Hybrid", "On-site", "Unknown"],
                "examples": ["Remote"],
                "default": "Unknown",
            },
            "is_leadership": {
                "prompt": "Does this role involve managing or leading a team?",
                "type": "Boolean",
                "examples": ["false"],
                "default": False,
            },
        },
        model="gpt-4.1-mini",
    ),

    # Step 2: Classify based on extracted data
    # This step sees the original row (title, description) PLUS all
    # extracted fields from step 1 via prior_results.
    LLMStep(
        "classify",
        fields={
            "department": {
                "prompt": (
                    "Classify this role into a department based on the title "
                    "and required skills"
                ),
                "type": "String",
                "enum": [
                    "Engineering - Backend",
                    "Engineering - Frontend",
                    "Engineering - ML/AI",
                    "Engineering - Data",
                    "Engineering - DevOps",
                    "Engineering - Fullstack",
                    "Product",
                    "Design",
                    "Other",
                ],
            },
            "seniority": {
                "prompt": (
                    "Determine the seniority level from the title and "
                    "years of experience required"
                ),
                "type": "String",
                "enum": ["Junior", "Mid", "Senior", "Staff", "Lead", "Principal", "Director"],
            },
            "skill_category": {
                "prompt": (
                    "What is the primary technical domain based on the "
                    "required skills?"
                ),
                "type": "String",
                "examples": ["Machine Learning", "Web Development", "Data Engineering"],
            },
        },
        depends_on=["extract"],
        model="gpt-4.1-mini",
    ),
])

# -- Run -----------------------------------------------------------------------

config = EnrichmentConfig(
    enable_caching=True,
    temperature=0.1,     # Very low temperature for consistent extraction
    max_workers=10,
)

result = pipeline.run(job_posts, config=config)

# -- Inspect results -----------------------------------------------------------

for row in result.data:
    print(f"\n{'=' * 60}")
    print(f"Title:            {row['title']}")
    print(f"Department:       {row['department']}")
    print(f"Seniority:        {row['seniority']}")
    print(f"Skill Category:   {row['skill_category']}")
    print(f"Required Skills:  {', '.join(row['required_skills'])}")
    print(f"Preferred Skills: {', '.join(row['preferred_skills'])}")
    print(f"Experience:       {row['years_experience']}+ years")
    print(f"Location:         {row['location']}")
    print(f"Work Model:       {row['work_model']}")
    print(f"Leadership Role:  {row['is_leadership']}")

# Summary statistics
print(f"\nTotal tokens: {result.cost.total_tokens:,}")
print(f"Success rate: {result.success_rate:.0%}")
```

## Expected Output

```
============================================================
Title:            Senior ML Engineer
Department:       Engineering - ML/AI
Seniority:        Senior
Skill Category:   Machine Learning
Required Skills:  PyTorch, transformers, MLOps
Preferred Skills:
Experience:       5+ years
Location:         San Francisco, CA
Work Model:       Remote
Leadership Role:  False

============================================================
Title:            Frontend Developer
Department:       Engineering - Frontend
Seniority:        Mid
Skill Category:   Web Development
Required Skills:  React, TypeScript
Preferred Skills: Next.js, Tailwind
Experience:       3+ years
Location:         New York, NY
Work Model:       Hybrid
Leadership Role:  False

============================================================
Title:            Data Engineering Lead
Department:       Engineering - Data
Seniority:        Lead
Skill Category:   Data Engineering
Required Skills:  Databricks, Spark, SQL, Python, Airflow
Preferred Skills:
Experience:       7+ years
Location:         Chicago, IL
Work Model:       On-site
Leadership Role:  True

Total tokens: 5,800
Success rate: 100%
```

## Key Concepts Demonstrated

**List fields.** The `required_skills` and `preferred_skills` fields use `type: "List[String]"`
to extract variable-length lists. Accrue generates a JSON schema with `{"type": "array", "items": {"type": "string"}}` and the LLM returns a proper JSON array.

**Boolean fields.** The `is_leadership` field uses `type: "Boolean"` for yes/no extraction.
The LLM returns `true` or `false` in JSON, and Accrue validates it as a Python `bool`.

**Rich enum classification.** The `department` field uses a long enum list to map free-text
job descriptions into a controlled taxonomy. Enums are included in the JSON schema and the
system prompt, so the LLM is constrained to valid values.

**Two-phase extraction.** Step 1 extracts raw entities, step 2 classifies based on those
entities. This separation keeps each LLM call focused and improves accuracy compared to
doing everything in one shot.

**Low temperature for extraction.** Extraction tasks benefit from very low temperature
(0.1) since the answers are factual and should not vary between runs. Classification
tasks can use slightly higher temperature (0.2-0.3) if you want more nuanced judgment.

## Scaling to Larger Datasets

For processing hundreds or thousands of job postings, add checkpointing to recover
from interruptions:

```python
config = EnrichmentConfig(
    enable_caching=True,
    enable_checkpointing=True,
    checkpoint_interval=50,    # Save progress every 50 rows
    max_workers=20,            # Higher concurrency for Tier 2+ accounts
    temperature=0.1,
)

result = pipeline.run(job_posts, config=config)
```

If the pipeline is interrupted, re-running with the same config automatically resumes
from the last checkpoint. Cached rows are skipped entirely.

## Using Field Defaults for Robustness

Fields with `default` values handle edge cases gracefully. If the LLM cannot determine
the work model from the description, the `"Unknown"` default is used instead of
producing an error. This is especially useful at scale where some input rows may have
incomplete or ambiguous data.

```python
"work_model": {
    "prompt": "Determine the work arrangement",
    "type": "String",
    "enum": ["Remote", "Hybrid", "On-site", "Unknown"],
    "default": "Unknown",   # Used when the LLM gives a refusal-like answer
},
```

Accrue detects common refusal patterns (like "unable to determine" or "not specified")
and automatically substitutes the field default.

"""Accrue Quickstart: Account Enrichment

Enrich a list of companies with structured data using LLMs.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""

from accrue import LLMStep, Pipeline

companies = [
    {
        "company": "Stripe",
        "website": "stripe.com",
        "description": "Online payment processing for internet businesses",
    },
    {
        "company": "Notion",
        "website": "notion.so",
        "description": "All-in-one workspace for notes, docs, and project management",
    },
    {
        "company": "Figma",
        "website": "figma.com",
        "description": "Collaborative interface design tool for teams",
    },
    {
        "company": "Datadog",
        "website": "datadoghq.com",
        "description": "Cloud monitoring and security platform",
    },
    {
        "company": "Canva",
        "website": "canva.com",
        "description": "Online design and visual communication platform",
    },
    {
        "company": "Plaid",
        "website": "plaid.com",
        "description": "Financial data connectivity for fintech applications",
    },
    {
        "company": "Airtable",
        "website": "airtable.com",
        "description": "Low-code platform for building collaborative apps",
    },
    {
        "company": "Vercel",
        "website": "vercel.com",
        "description": "Frontend cloud platform for web developers",
    },
]

pipeline = Pipeline(
    [
        LLMStep(
            "enrich",
            fields={
                "industry": {
                    "prompt": "Classify the company's primary industry",
                    "enum": [
                        "Fintech",
                        "Developer Tools",
                        "Productivity",
                        "Design",
                        "Infrastructure",
                        "Other",
                    ],
                },
                "icp_fit": {
                    "prompt": "Rate fit as a B2B SaaS prospect for a sales intelligence platform",
                    "enum": ["Strong Fit", "Moderate Fit", "Weak Fit"],
                },
                "employee_estimate": {
                    "prompt": "Estimate the number of employees",
                    "type": "Number",
                },
            },
        )
    ]
)

if __name__ == "__main__":
    result = pipeline.run(companies)

    print(result.data.to_string(index=False))
    print(f"\nAccounts enriched: {len(result.data)}")
    print(f"Success rate:      {result.success_rate:.0%}")
    print(f"Total tokens:      {result.cost.total_tokens:,}")

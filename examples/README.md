# Accrue Examples

## Prerequisites

- Python 3.10+
- `pip install accrue`
- `export OPENAI_API_KEY=sk-...`

## Start Here

**[quickstart.ipynb](quickstart.ipynb)** — Enrich 8 accounts with industry, ICP fit, and employee estimates in under 20 lines. (~$0.05 in API costs)

Or run without Jupyter:
```bash
python examples/quickstart.py
```

## Go Deeper

**[advanced_pipeline.ipynb](advanced_pipeline.ipynb)** — Production-grade pipeline with multi-step execution, conditional logic, caching, lifecycle hooks, multi-provider support, and grounding. (~$0.30 in API costs)

## Reference Data

- `sample_data.csv` — 15 tech companies for testing
- `field_categories.csv` — Example field specs for `load_fields()` CSV workflow

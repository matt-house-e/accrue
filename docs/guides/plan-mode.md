# Plan mode — `pipeline.plan()`

`Pipeline.run()` is all-or-nothing: it either runs the whole dataset or it
doesn't. For a new pipeline that's a real risk — a mis-built prompt or a 10k-row
CSV can turn into an accidental \$50 run. `pipeline.plan()` is the dry run:
it previews what each step will do and estimates the full cost **before** you
commit, modelled on `terraform plan` and Claude Code's plan mode.

```python
plan = pipeline.plan(df, sample_size=3)
print(plan.summary())
```

## What a plan does

`plan()` runs the pipeline over the first `sample_size` rows — real calls, but
capped — and reports:

- **Resolved prompts** — the system prompt each LLM step will send, rendered for
  the first sample row.
- **JSON schemas** — the structured-output schema each step generates from its
  field specs.
- **Sample outputs** — the actual enriched rows from the capped run.
- **Estimated full-run cost** — the sample's measured token usage extrapolated
  to the full row count.

`plan.summary()` returns a human-readable preview as a string (print it, log it,
or assert on it in a test). The structured fields are also available directly on
the `PipelinePlan`: `steps`, `sample_rows`, `sample_outputs`, `sample_errors`,
`total_rows`, `sample_size`, `sample_cost`, and `estimated_cost`.

## Cost estimation

The estimate extrapolates the sample's *measured* tokens to the full dataset.
Crucially, it divides by the rows that actually called the API — cache misses
(or every processed row when caching is off) — so **cached sample rows don't
deflate the estimate**, and re-planning an already-cached dataset doesn't
invent phantom cost.

Cost is reported in **tokens** (prompt / completion / total), matching how
`result.summary()` and `result.report()` express cost throughout Accrue. On a
fresh dataset the estimate lands within roughly ±20% of the actual full run;
treat it as a guardrail against surprises, not an invoice.

## Confirm before a big run

For interactive scripts, `run(..., confirm=True)` wires the preview straight into
a gate: it prints the plan and asks `y/n` before running the rest of the dataset.

```python
# Prints the plan, then prompts. Answering "n" raises RuntimeError without
# running the full dataset.
result = pipeline.run(df, confirm=True)
```

The capped sample inside the plan still makes real (small) calls, so you see
true sample outputs before deciding whether to spend on the whole set.

## Out of scope

- **Per-row cost preview** — use `run()` and inspect `result.cost` for that.
- **Interactive UI** — `plan()` is plain stdout; wrap it yourself if you want more.

# Run reports — `result.report()`

After a pipeline run, `result.report()` produces a heuristic-driven
Markdown or HTML summary you can paste into Slack, save to a file, or
attach to a stakeholder update.

```python
result = pipeline.run(df)

print(result.report())                                # Markdown for terminal/Slack
result.report(output_format="html", path="run-report.html")  # self-contained HTML
```

## What's in the report

- **Headline numbers** — rows succeeded / errored, wall time, total tokens.
- **Flagged patterns** — what looks suspicious, with a probable cause and a
  suggested action (the headline section).
- **Per-step table** — model, rows processed, skipped count, cache hit rate,
  tokens, wall time.
- **Errors** — count and most-common error types.

## Heuristics — judgement, not statistics

The point of a run report is to tell you what looks *wrong*, not to make you
read histograms. Each builtin heuristic encodes a pattern accrue has seen
go bad in real pipelines, and emits a finding with a suggested next step.

| Code | Severity | Fires when… |
|---|---|---|
| `enum_collapse`     | warning | One enum value covers > 60% of rows. |
| `numeric_clipping`  | warning | A min/max value covers > 25% of numeric outputs. |
| `length_anomaly`    | warning | Mean output length is < 30% or > 200% of the field's `format` range hint (e.g. `"80-120 words"`). |
| `retry_storm`       | warning | A step's post-retry error rate exceeds 5%. |
| `refusal_pattern`   | warning | Outputs match refusal phrases (`"I cannot determine"`, …) on > 5% of rows. |
| `cache_thrash`      | info    | Cache had some hits, but hit rate < 10% (suggests churning input). |
| `cost_outlier`      | info    | A single step accounts for > 60% of pipeline tokens. |

## Silencing noisy heuristics

If a heuristic is wrong for your domain — e.g. you genuinely expect every
row to fall into a default category — pass its code to `disable`:

```python
result.report(disable=["enum_collapse", "cache_thrash"])
```

## Saving to a file

`path=` writes the rendered report and still returns the string:

```python
text = result.report(output_format="html", path="reports/run-2026-05-10.html")
```

## Heuristics use field specs

`enum_collapse`, `numeric_clipping`, `length_anomaly`, and `refusal_pattern`
look at the field specs collected from your steps. If you build a pipeline
out of `FunctionStep`s only — no `LLMStep` with a `FieldSpec` — those
heuristics have nothing to inspect; the rest of the report still renders.

## Writing a custom heuristic

A heuristic is just a function from a `ReportContext` to a list of
`Finding`s. The same building blocks the builtins use are exported, so
you can run your own alongside them and feed the combined findings to the
existing renderer:

```python
from accrue.pipeline.report import (
    Finding,
    ReportContext,
    render_markdown,
    run_heuristics,
)

def all_blank(ctx: ReportContext) -> list[Finding]:
    """Flag any field whose every row came back empty."""
    findings: list[Finding] = []
    for col in ctx.field_specs:
        values = ctx.values(col)              # non-null values, row order
        if not values or all(v == "" for v in values):
            findings.append(
                Finding(
                    code="all_blank",
                    message=f"Field {col!r} is empty for every row.",
                )
            )
    return findings

# ReportContext mirrors the PipelineResult attributes report() reads:
ctx = ReportContext(
    data=result.data,
    cost=result.cost,
    errors=result.errors,
    field_specs=result.field_specs,
    pipeline_elapsed_seconds=result.pipeline_elapsed_seconds,
    step_elapsed_seconds=result.step_elapsed_seconds,
)
findings = run_heuristics(ctx) + all_blank(ctx)    # builtins + your own
print(render_markdown(ctx, findings))
```

`render_html` works the same way if you want a shareable document.

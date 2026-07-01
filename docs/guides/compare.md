# Comparing two runs — `accrue.compare()`

The way you actually iterate on a pipeline is: run it, eyeball the output,
tweak a prompt, run it again, and ask *"did v2 change anything meaningful,
and are the changed rows better?"* You have no labels — just two runs and a
hunch. `accrue.compare()` makes that one line.

```python
import accrue

result_a = pipeline.run(df)   # v1
# ... tweak a prompt ...
result_b = pipeline.run(df)   # v2

diff = accrue.compare(result_a, result_b, label_a="v1", label_b="v2")

print(diff.summary())                       # Markdown for terminal/Slack
diff.report(output_format="html", path="v1-vs-v2.html")  # self-contained HTML
```

The two results don't have to come from the same pipeline definition — only
the overlap of their output fields is compared. If the field-spec schemas
differ, `compare()` warns (it never raises) and compares the overlap only.

## What you get back

`compare()` returns a `ComparisonResult` with four methods:

| Method | Returns |
|---|---|
| `changed_rows(field=None)` | A before/after DataFrame of rows that differ — on any field, or on one named field. |
| `distribution_shift()` | Per-field churn keyed by field name: enum frequency tables, numeric mean/std deltas, text differs-counts. |
| `cost_delta()` | Token, cache-hit-rate, and wall-time deltas between the two runs. |
| `summary()` / `report()` | A rendered Markdown (or HTML) report combining all of the above. |

```python
diff.changed_rows()             # every row where any output field moved
diff.changed_rows("category")   # only rows where `category` changed
diff.distribution_shift()       # {"category": FieldChurn(...), "score": ...}
diff.cost_delta()               # CostDelta(total_tokens_delta=..., ...)
```

## How rows are paired

- If both runs' data carry the **default positional index** (the common
  case — `list[dict]` input, or a freshly-built DataFrame), rows are
  aligned by position. Differing row counts compare the shorter prefix and
  warn about the rest.
- If both carry a **real, unique index**, rows are aligned by label, on the
  intersection. Labels present on only one side are warned about and
  skipped.
- A non-unique index, or one default and one custom index, falls back to
  positional alignment with a warning — never a silent misalignment.

## How fields are classified

Each compared field is classified from its spec, exactly as
`result.report()` does:

- **Enum** (`enum` set) — before/after frequency table, surfaced in the
  summary as `'AI' 22% → 31%`.
- **Number** (`type: "Number"`) — mean and std, plus the mean delta.
  Non-numeric values coerce to `NaN` rather than erroring.
- **Everything else** (String / Boolean / Date / JSON / `List[String]`) —
  a "differs" count only. String fields additionally get an approximate
  token-length delta (`len(str) // 4`). Free-text fields are intentionally
  **not** fuzzy-matched or scored for semantic similarity — that's out of
  scope.

## Edge cases, handled

- **Different row counts** → compare the intersection, warn about the rest.
- **One run errored on a row, the other didn't** → flagged as a change (not
  a value diff), cross-referenced by each side's `result.errors`.
- **Identical runs** → `summary()` short-circuits to a one-line "no
  differences" message. Still useful as confirmation.

## What's deliberately not here

- **No `$`/pricing.** There's no pricing data anywhere in Accrue, so
  `cost_delta()` reports tokens, cache efficiency, and wall time — never a
  dollar figure it would have to invent.
- **No statistical significance testing** (chi-square, etc.). Distribution
  shift numbers are the right level of detail for eyeballing a prompt change.
- **No N-way comparison.** `compare()` is pairwise; compare runs two at a time.

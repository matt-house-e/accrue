# Merging repeated runs — `accrue.merge_runs()`

Run the same pipeline over the same rows twice and some structured cells come
back different, even at temperature 0. The usual cause is the source, not the
sampling: a clause that can be read two ways gets read two ways. One run is a
sample. `merge_runs()` takes two or more independent runs, votes per cell, and
tells you which cells the runs could not settle, so a person checks those and
not the whole table.

```python
import accrue
from accrue import EnrichmentConfig

runs = [
    pipeline.run(df, config=EnrichmentConfig(cache_dir=f".accrue/run-{i}",
                                             checkpoint_dir=f".accrue/run-{i}"))
    for i in range(3)
]
merged = accrue.merge_runs(runs)

merged.data            # one answer per cell
merged.needs_review    # the cells a person should check
print(merged.summary())
```

## Keep the runs independent

With the same pipeline, rows and config, run 2 has the same cache key as run 1
and replays it: no API call, the same answers, and a merge that reports perfect
agreement. Give each run its own `cache_dir`, as above. Each run still caches
itself, so rerunning the script costs nothing and returns the same N reads;
`enable_caching=False` also gives a fresh read, but pays again on every rerun.

Use a separate `checkpoint_dir` per run too if checkpointing is on. A run that
ends with errors keeps its checkpoint, and the next run over the same rows
would resume from it.

`merge_runs()` warns when a run was served entirely from cache and matches an
earlier run on every cell, because that is what a replay looks like.

## How a cell is decided

Each run whose row did not error casts one vote. If the votes agree, the cell
is `unanimous`. If they disagree, the field's resolver is asked first (see
below), then the field's rule:

| Rule | Default for | On disagreement |
|---|---|---|
| `"majority"` | enums, `Number`, `Boolean`, `Date`, `JSON`, `List[String]`, fields with no spec (FunctionStep outputs) | A strict majority wins (2 of 3). Without one, the earliest of the largest groups is kept and the cell is flagged. With two runs, any disagreement is flagged. |
| `"first"` | free-text `String` fields with no `enum` | Run 0's answer is kept and the cell is not flagged. Two runs rarely word prose the same way, so voting on it would flag almost every cell. Agreement is still recorded. |

A cell only one run read, because the row errored in the others, is flagged as
`single_read`: it has had no second opinion. A row that errored in every run is
`no_reads`, with a `None` value.

Change a field's rule with `resolve=`: `resolve={"summary": "majority"}` votes
on a free-text field, `resolve="first"` turns voting off everywhere.

### Comparing values: `key=`

Values are compared with the same equality as `accrue.compare()`: two nulls
agree, `3 == 3.0`, lists and dicts compare by content. To compare on something
looser, pass a key function. It normalises before comparing; the merged value
is still the raw value from the earliest run in the winning group.

```python
merged = accrue.merge_runs(runs, key={
    "start_date": parse_date,                       # "2026-01-01" == "1 Jan 2026"
    "governing_law": lambda v: str(v).split(",")[0].strip().lower(),
})
```

### Your own rule: resolvers

A resolver is `resolver(field, candidates) -> Candidate | None`, called only
when the runs disagree. Each `Candidate` has `run` (its index), `value`, and
`row`, the run's whole output row, so the resolver can read companion columns
accrue knows nothing about. Return the candidate to keep, or `None` to fall
back to the field's rule.

Prefer the run with the better evidence, here a quote that a later step checked
against the source:

```python
RANK = {"verified": 2, "unverified": 1}

def better_evidence(field, candidates):
    ranks = [RANK.get(c.row.get(f"{field}_quote_check"), 0) for c in candidates]
    best = [c for c, r in zip(candidates, ranks) if r == max(ranks)]
    return best[0] if len(best) == 1 else None   # tied evidence: let the vote decide

merged = accrue.merge_runs(runs, resolve={"notice_months": better_evidence})
```

Pass a resolver directly (`resolve=better_evidence`) to use it for every field.
A cell a resolver settled has `basis == "resolver"` and is not flagged.

## What you get back

`merge_runs()` returns a `MergedResult`:

| Attribute | |
|---|---|
| `data` | Run 0's rows with every merged field replaced by its merged value. Other columns (inputs, anything outside `fields=`) come from run 0. Rows missing from any run are dropped, with a warning. |
| `agreement` | One row per merged cell: `row`, `field`, `value`, `votes`, `voters`, `agreed`, `needs_review`, `basis`, `values` (each run's raw answer, `None` where it errored). |
| `needs_review` | The rows of `agreement` with `needs_review` set. |
| `errored_rows` | Row label → the runs in which that row errored. |
| `cost` | A `CostSummary` summed across runs, per step. |
| `elapsed_seconds` | Wall time summed across runs. |
| `rules` | Field → `"majority"`, `"first"` or `"resolver"`. |
| `summary()` | Markdown: cells unanimous, cells needing review, and a per-field table. |

`basis` is one of `unanimous`, `majority`, `resolver`, `first`, `no_majority`,
`single_read`, `no_reads`. Rows are aligned to run 0 the way
[`compare()`](compare.md#how-rows-are-paired) aligns two runs.

```python
review = merged.needs_review
review.groupby("field").size()            # which fields the runs fight over
merged.agreement.pivot(index="row", columns="field", values="votes")
```

## Derived fields

A field computed from other fields (a deadline from a start date and a notice
period) should be computed from the merged values, not voted on. Merge the
read, then run the step that derives it over the merged rows:

```python
derive = accrue.Pipeline([
    accrue.FunctionStep("deadline", fn=compute_deadline, fields=["notice_deadline"]),
])
final = derive.run(merged.data)
```

If the derived field is in the runs' outputs it is merged too; its agreement is
still a useful signal of which rows moved, and `derive.run()` overwrites the
value.

## Not included

- **Running the passes for you.** There is no `Pipeline(..., passes=3)`. A run is
  already a `pipeline.run()` call; looping over it keeps the cost visible and
  each run's config, cache and run log your own.
- **Domain rules.** Which evidence outranks which, or when a disagreement does
  not matter, belongs in a resolver or `key=` in your project.
- **Fuzzy text agreement.** Free text is not scored for similarity. Use `key=`
  to normalise it, or keep it on `"first"`.

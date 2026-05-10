---
description: Sweep open issues — apply labels, ask for repro on bugs, link near-duplicates
---

Run a triage pass on open issues in the accrue repo.

## Step 1 — pull the queue

```bash
gh issue list --state open --limit 50 --json number,title,body,labels,createdAt,comments
```

## Step 2 — for each issue

Apply this decision tree:

1. **Already labelled with a `type:*` label?** Skip the type step.

2. **Otherwise classify type** by reading title + body. Use exactly one of:
   - `type:bug` — something is broken
   - `type:task` — small concrete change
   - `type:story` — user-facing feature
   - `type:epic` — multi-issue effort
   - `type:spike` — research / investigation

3. **Component** (also exactly one):
   - `component:core` — `accrue/core/`
   - `component:steps` — `accrue/steps/`
   - `component:pipeline` — `accrue/pipeline/`
   - `component:data` — `accrue/data/`, schemas
   - `component:testing` — `tests/`
   - `component:docs` — `docs/`, `README.md`
   - `component:infra` — CI, packaging, releases

4. **Priority** if not set — read severity:
   - `priority:critical` — blocks usage of the library
   - `priority:high` — affects most users
   - `priority:medium` — default
   - `priority:low` — nice-to-have

5. **Bugs without repro** — if `type:bug` and the body has no minimal example, post:
   > Thanks for the report. Could you share a minimal pipeline that reproduces this — ideally a 5–10 line snippet plus the accrue version (`pip show accrue`)?

   Add the `needs info` label.

6. **Possible duplicates** — search closed and open issues for similar titles. If you find one with >70% confidence, comment linking it. Add `duplicate` label only if the maintainer should close — otherwise just link and leave the label off.

## Step 3 — apply labels

Use `gh issue edit <num> --add-label <label1>,<label2>,...`. Do not remove existing labels unless they're plainly wrong.

## Step 4 — summarise

End with a table:

| # | Title | Labels added | Action |
|---|---|---|---|

Action is one of: `labelled`, `asked for repro`, `flagged duplicate`, `no change`.

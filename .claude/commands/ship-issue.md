---
description: Pick up a GitHub issue end-to-end — read it, plan, code, test, open a PR
argument-hint: <issue-number>
---

You are shipping issue #$ARGUMENTS in the accrue repo.

Follow the full lifecycle. Be conservative — small PRs beat big ones.

## 0. Overview first

Before any planning, branching, or code, post a brief feature overview to the user so they can grok the change instantly without reading the PR body or diff. Keep it under 8 lines:

```
**What it does:** <one sentence>
**Value:** <one or two sentences — who benefits, what gets simpler>
**Before / After:** <minimal code snippet showing the API change>
```

Then continue.

## 1. Understand

- `gh issue view $ARGUMENTS --comments` — read the issue and any discussion.
- Check labels: `type:*` tells you scope, `component:*` tells you where.
- If the issue is vague or missing repro, **stop and ask in a comment**. Don't guess.

## 2. Plan

- Read `CLAUDE.md` and `AGENTS.md` if you haven't this session.
- Search the codebase for the relevant module under `accrue/`.
- Sketch the diff in 3-5 bullets: which files, what changes, what tests.
- Verify the plan respects the project's invariants:
  - **Async-only steps.** No sync duplicates.
  - **Step purity.** `dict[str, Any]` in/out, no pandas inside steps.
  - **Internal fields use `__` prefix.**
  - **Minimal deps.** No litellm, no langfuse.

## 3. Branch

```bash
git checkout -b feature/issue-$ARGUMENTS-<short-slug>
```

Follow `feature/description` naming from `CLAUDE.md`.

## 4. Code

- Edit existing files where possible — don't create new ones unless the layout demands it.
- Add or update tests under `tests/` for every behaviour change.
- If you change architecture, update the relevant file in `docs/guides/`.

## 5. Verify

Run these and fix anything that fails before opening the PR:

```bash
ruff check .
ruff format --check .
pytest -x -q
```

## 6. Commit

Use the format from `CLAUDE.md`:

```
<type>: <brief description>

- Detail 1
- Detail 2

Closes #$ARGUMENTS

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`.

## 7. Open the PR

Use `.github/pull_request_template.md`. PR title under 70 chars. Body covers Summary, Changes, Testing, Checklist. End with `Closes #$ARGUMENTS`.

## When to bail

- The issue is actually two issues — comment, ask the maintainer to split.
- The change requires a design decision the issue doesn't resolve — comment with options, don't pick.
- Tests fail in a way you don't understand — push the branch, open a draft PR, ask for input.

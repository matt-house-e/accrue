# Accrue

Composable enrichment pipeline engine. The gap between Instructor (single LLM call) and Clay (full SaaS platform). v1.3.0, Python 3.10+.

## Commands

```bash
pytest                          # Run all tests (791)
pytest tests/unit/              # Unit tests only
pytest -x -q                    # Fast fail
python -m build                 # Build package
pip install -e ".[dev]"         # Dev install
pip install -e ".[anthropic]"   # With Anthropic provider
pip install -e ".[google]"      # With Google provider
```

## Code Style

- **Async-only steps.** Sync API is `Pipeline.run()` wrapping `asyncio.run()`. No sync/async duplication.
- **Step data**: `dict[str, Any]` not `pd.Series`. Steps are pure, no pandas inside.
- **Internal fields**: `__` prefix (e.g. `__web_context`) for inter-step data, filtered from output.
- **Prompt split**: `build_prompt()` returns `PromptParts(system, user)`. Keep the `system` half row-independent — providers cache on an exact prefix match, so any per-row content in it turns caching into a 1.25x surcharge (#107).
- **Minimal deps**: Base: `openai`, `pydantic`, `pandas`, `tqdm`, `python-dotenv`. Never add litellm/langfuse.
- See `docs/guides/` for architecture, providers, caching, grounding details.

## Git Workflow

- **`main`** — Production-ready code
- **`feature/description`** — Feature branches

### Commit Format
```
type: Brief description

- Detail 1
- Detail 2

Co-Authored-By: Claude <noreply@anthropic.com>
```
Types: `feat`, `fix`, `docs`, `refactor`, `test`

### What Gets Committed
- Source code (`accrue/`), Tests (`tests/`), Examples (`examples/`), Docs (`.md`)
- Never: `data/`, `.env`, `.vscode/`, `.idea/`, `.notes/`

### Merging PRs

- **A PR from a fork by a non-collaborator must get a review pass before merge.**
  `claude-review.yml` triggers on `pull_request`, which for fork PRs runs without
  repository secrets — the automatic review never fires on external contributions.
  Comment `@claude review` (that path runs on `issue_comment`, in base-repo
  context, so it works on forks), read the result, then merge.
- Green CI is not a review. `main` requires `test (3.10-3.13)`, which proves the
  tests pass — not that the diff does what it says.
- Never merge with `--admin`. If branch protection blocks a merge, fix the cause.

## GitHub Issues

**Always include labels.** Format: `[Type]: [Component] Description`

Labels that actually exist on the repo (`gh label list` is the source of truth):

- Type: `bug`, `enhancement`, `documentation`, `question`, `duplicate`, `invalid`, `wontfix`
- Priority: `priority:{high,medium,low}`
- Other: `backlog` (not on the near-term roadmap), `meta` (dev tooling, not user-facing),
  `good first issue`, `help wanted`, `needs info`, `release`

There is no `type:*` or `component:*` namespace — `gh issue create --label type:task`
fails. Create the label first if you want a new one.

See `docs/guides/` for details.

## Build Status

| Phase | Status |
|-------|--------|
| 1-5 (Core engine through DX) | COMPLETE |
| 6A Ship: examples, README, PyPI | COMPLETE |
| 6B Power user: conditional steps, grounding (done); batch API (#62, OpenAI+Anthropic); waterfall, chunked, CLI (remaining) | IN PROGRESS |

Full design: `docs/instructions/PIPELINE_DESIGN.md`

## Keeping Docs in Sync

**When making architectural decisions or design changes, update:**

1. `CLAUDE.md` — This file. Commands, style, gotchas.
2. `docs/` — Guides, reference, and technical design.
3. GitHub issues — Close stale issues, update epics.

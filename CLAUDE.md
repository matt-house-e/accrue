# Accrue

Composable enrichment pipeline engine. The gap between Instructor (single LLM call) and Clay (full SaaS platform). v1.4.0, Python 3.10+.

## Scope

Accrue is a composable pipeline **engine**, not a platform. The positioning — the gap
between Instructor (single LLM call) and Clay (full SaaS platform) — is a constraint,
not a slogan. Check a proposal against it before writing code.

**IN**

- New step types, provider adapters, field-spec features
- Caching, batching, cost accounting, retries, checkpointing
- Anything making a pipeline cheaper, more reliable, or easier to reason about
- Run observability *as data* — the run-log contract is accrue's half of that

**OUT**

- **Bundled third-party integrations** (Sheets, Airtable, Salesforce, CRM connectors).
  `pipeline.serve()` (#34) is the story for connecting accrue to live systems: it makes
  a pipeline callable by anything speaking HTTP/MCP without accrue owning one connector.
- **Hosted anything** — no accounts, no server we run, no scheduler.
- **The dashboard.** Accrue Watch lives in
  [accrue-ui](https://github.com/matt-house-e/accrue-ui); core ships the run log and the
  `accrue watch` handoff, nothing more.
- **Orchestrator ambitions.** Dagster/Prefect/Airflow call accrue, not the reverse.

**Why minimal deps.** Users embed accrue in their own stacks, so every dependency is
theirs too — a version conflict we import is one they have to solve. Base deps are
`openai`, `pydantic`, `pandas`, `tqdm`, `python-dotenv`; provider SDKs are extras.

**Out-of-scope proposals** get labelled `wontfix` or `backlog` with a one-line rationale.
Don't build them, and don't leave them open to dilute the roadmap.

## Commands

```bash
pytest                          # Run all tests — tests/integration excluded via norecursedirs
pytest tests/integration/       # Integration tests, opt-in only
pytest -x -q                    # Fast fail
python -m tests.test_public_api --update   # Regenerate the public API snapshot
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
- **Minimal deps**: Base: `openai`, `pydantic`, `pandas`, `tqdm`, `python-dotenv`. Never add litellm/langfuse — see [Scope](#scope) for why.
- **Public API is under contract.** `tests/public_api_snapshot.json` pins every name exported from `accrue`, `accrue.providers`, `accrue.data` and `accrue.core.exceptions`. Any change to that surface — including a purely additive one — fails `tests/test_public_api.py`. That is deliberate: it puts the change in the PR diff as readable JSON. To change the API on purpose, run `python -m tests.test_public_api --update`, commit the regenerated snapshot in the same PR, and add a `CHANGELOG.md` entry under `[Unreleased]`. Never hand-edit the snapshot (#121).
- See `docs/guides/` for architecture, providers, caching, grounding, run-log details.
- **Run-log contract v1**: `run(..., run_log=True)` emits JSONL via `JsonlRunLogger` (a hooks consumer, `accrue/core/runlog.py`). Additive changes only within v1 — never rename/remove fields without bumping `SCHEMA_VERSION`; golden fixture `tests/fixtures/run_small.jsonl` is what accrue-ui tests against. `pipeline_start` carries a nested `manifest` object (the run's definition: steps+types+models, field schema, config — built by `accrue/core/manifest.py`, #138) that must stay row-independent and deterministic (no timestamps/rng); regenerate the fixture with `python tests/fixtures/generate_run_fixture.py` and keep `test_regeneration_is_stable` green.

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
  Nothing reviews PRs automatically — the auto-review workflow was dropped in #136,
  so reviews run ad hoc. Either review the diff locally with Claude Code, or comment
  `@claude review` on the PR: that path is `claude.yml`, which fires on
  `issue_comment` in base-repo context and so has repository secrets even for fork
  PRs (a `pull_request`-triggered job would not). Read the result, then merge.
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
| 6B Power user: conditional steps, grounding, batch API (#62, OpenAI+Anthropic), run log + `accrue watch` CLI (done); waterfall, chunked (remaining) | IN PROGRESS |

## Keeping Docs in Sync

**When making architectural decisions or design changes, update:**

1. `CLAUDE.md` — This file. Commands, style, gotchas.
2. `docs/` — Guides, reference, and technical design.
3. GitHub issues — Close stale issues, update epics.

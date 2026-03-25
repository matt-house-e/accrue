# Accrue

Composable enrichment pipeline engine. The gap between Instructor (single LLM call) and Clay (full SaaS platform). v1.0.0, Python 3.10+.

## Commands

```bash
pytest                          # Run all tests (483)
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

## GitHub Issues

**Always include labels.** Format: `[Type]: [Component] Description`

Labels: `type:{epic,story,task,bug,spike}`, `priority:{critical,high,medium,low}`, `component:{core,steps,pipeline,data,testing,docs,infra}`

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

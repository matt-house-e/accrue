---
description: Find drift between code and docs, propose patches
---

Audit the accrue repo for drift between `accrue/` (the source) and `docs/` + `README.md` (the docs).

## What to check

1. **Public API surface.** For each public class/function exported from `accrue/__init__.py`, confirm it appears in at least one of `README.md`, `docs/getting-started/`, or `docs/guides/`. Flag anything exported but undocumented.

2. **Step types.** Every concrete step under `accrue/steps/` (e.g. `LLMStep`, `FunctionStep`) should have a corresponding section in `docs/guides/`. New step types without docs is the most common drift.

3. **Provider docs.** `accrue/steps/providers/` should match `docs/guides/providers.md`. Check that every provider listed in code is in the doc, with the right install extra.

4. **CLAUDE.md.** Specifically:
   - The `Build Status` table — is it still accurate vs. closed/open issues?
   - The `pip install` extras list — does it match `pyproject.toml`'s `[project.optional-dependencies]`?
   - The Commands block — do the listed commands still work?

5. **Examples drift.** Files under `examples/` should still run against current accrue. Don't run them, but read the imports — flag any that reference symbols that no longer exist.

6. **`docs/guides/` cross-refs.** Anchor links between guides should resolve. Flag broken ones.

## How to report

For each piece of drift, output:

- **Where**: file path and line, or section name
- **What**: one sentence describing the mismatch
- **Fix**: a proposed Edit (old/new strings) the user can apply

Don't apply the patches yourself — surface them so the maintainer can review the batch.

End with a one-line verdict: **"X drift findings, Y critical."** Critical = public API undocumented or example broken.

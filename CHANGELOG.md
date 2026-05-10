# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Agentic SDLC scaffolding: GitHub Actions for `@claude` mention, PR review, weekly maintenance, and dogfooded issue triage. Repo-scoped slash commands under `.claude/commands/`. Local `PostToolUse` hook for ruff-on-edit. See `AGENTS.md`.

### Changed
- **`EnrichmentConfig.enable_caching` now defaults to `True`** (was `False`). Re-runs of unchanged inputs no longer re-pay the API cost. Opt out with `EnrichmentConfig(enable_caching=False)` for one-off runs.

### Fixed
- `LLMStep.parse_response` now strips markdown code fences before `json.loads`. Fixes parse failures with Claude Haiku + grounding tools, where the structured-output constraint is disabled and the model wraps JSON in ` ``` ` fences. (#7)
- `accrue/__init__.py` `__version__` was out of sync with `pyproject.toml`.

## [1.2.0] - 2026-04-25

### Added
- `http_client` support for provider clients.
- Pipeline result summary printed at end of `run()`.

### Fixed
- Internal field (`__`-prefixed) handling in step output filtering.

### Changed
- Updated `/accrue` skill documentation.

## [1.1.0] - 2026-04-15

### Added
- Auto-loading of `.env` files via `python-dotenv` on import.
- Publish workflow now validates that the git tag matches `pyproject.toml`'s version before pushing to PyPI.

## [1.0.0] - 2026-04

Initial public release.

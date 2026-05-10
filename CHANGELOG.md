# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Agentic SDLC scaffolding: GitHub Actions for `@claude` mention, PR review, weekly maintenance, and dogfooded issue triage. Repo-scoped slash commands under `.claude/commands/`. Local `PostToolUse` hook for ruff-on-edit. See `AGENTS.md`.

### Fixed
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

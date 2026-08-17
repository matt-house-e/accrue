# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`accrue.compare(result_a, result_b)`** — diff two pipeline runs. Aligns rows explicitly (positionally for default `RangeIndex`, by label for real indexes, with a warning and positional fallback on non-unique or mixed-kind indexes) rather than via a blind index intersection that silently yields garbage. `changed_rows(field=None)` returns a before/after frame of differing rows; `distribution_shift()` reports per-field churn — frequency tables for enums, mean/std for numbers, differs-count plus an approximate token delta otherwise. Cell equality is null-safe and container-safe. (#36)
- **`Pipeline.plan(data, sample_size=3)`** — dry-run preview returning a `PipelinePlan`. Introspects each step's resolved system prompt and JSON schema, runs a capped real sample, and extrapolates measured token usage to the full dataset; `summary()` renders a terraform-plan-style preview. Cost is extrapolated only over rows that actually called the API, so cached sample rows don't deflate the estimate. Also adds `run(..., confirm=True)` to print the plan and prompt before the full run. See `docs/guides/plan-mode.md`. (#12)
- **`output_file` / `PipelineResult.save(path)`** — write enriched data to disk, inferring format from the extension (`.csv` / `.json` / `.parquet`). When `output_file` is passed to `run()` / `run_async()`, the save happens *before* the call returns, so an error in the caller's own post-processing can no longer discard a completed, already-paid-for run. (#10)
- **`PipelineResult.report(format, path, disable)`** — heuristic-driven Markdown/HTML run summary. Headlines suspicious patterns (enum collapse, numeric clipping, length anomalies, retry storms, refusal phrases, cache thrash, cost outliers) with a probable cause and a suggested action; backed by a per-step stats table. Pass `disable=[...]` to mute individual heuristic codes. See `docs/guides/report.md`. (#32)
- `PipelineResult` now exposes `pipeline_elapsed_seconds`, `step_elapsed_seconds`, and `field_specs`, populated by `Pipeline.run_async()` so reporters and custom heuristics can introspect.

### Changed
- **`Pipeline.execute()` returns a 4-tuple** `(accumulated, errors, cost, step_elapsed_seconds)` instead of a 3-tuple. Internal API (not exported in `accrue.__all__`); callers using `Pipeline.run()` / `run_async()` are unaffected.

### Fixed
- **`batch=True` no longer crashes mid-run against an OpenAI-compatible gateway.** `LLMStep.is_batch_eligible` gated the batch path on `isinstance(client, BatchCapableLLMClient)`, but that Protocol is `runtime_checkable`, so the check only confirmed the three batch methods *exist*. `OpenAIClient` defines them unconditionally — including when a `base_url` points at OpenRouter, Groq, Together, vLLM, or Ollama, none of which serve the Batch API. Any such step reported itself eligible and then died inside `client.files.create()`, after earlier steps in the pipeline had already been paid for. Batch capability is now resolved by `is_batch_capable(client)`, which combines the structural check with an optional `supports_batch` property; `OpenAIClient.supports_batch` is `False` whenever `base_url` is set. Such steps degrade to realtime execution and warn once that they are paying non-batch prices, matching the documented fallback for non-batch-capable clients. Calling `submit_batch()` / `poll_batch()` on a `base_url` client directly now raises a `StepError` naming `batch=False` as the fix, instead of a 404 from the gateway. Custom adapters that omit `supports_batch` are still treated as capable, so no existing client changes behaviour. `is_batch_capable` is exported from `accrue`.
- **The Claude 5 family works again.** The Anthropic adapter always sent `temperature` and no code path could omit it — `llm.py` substituted a literal `0.2` when none was given, and `EnrichmentConfig.temperature` itself defaults to `0.2`, so `None` never reached the adapter. Claude 5 models reject an explicit temperature, so every call to `claude-sonnet-5`, `claude-opus-5`, or `claude-fable-5` returned a 400 with an error that named the model rather than accrue. The parameter is now omitted for models that reject it (matched by family, so unreleased Claude 5 variants and `anthropic.` / `us.anthropic.` prefixes are covered), on both the realtime and batch paths. Dropping an explicit value warns once, and temperature-related API errors carry an actionable hint. Also covers `claude-opus-4-7` and `claude-opus-4-8`, which removed the sampling parameters too. (#109)
- **Prompt caching now actually hits.** The row's own data was built into the same system message the Anthropic adapter marks `cache_control: {"type": "ephemeral"}`, so the cached prefix changed on every row: every call wrote a fresh entry at 1.25x input and none was ever read at 0.1x — a standing ~25% surcharge on every input token for a feature that never fired. The prompt builder now returns the prompt already split (`build_prompt() -> PromptParts(system, user)`): instructions and `<field_specifications>` stay in the cacheable system block, `<row_data>` and `<prior_results>` move to the user message. The reminder stays last in the user message, preserving the sandwich pattern. (#107)
- Anthropic: `cache_creation_input_tokens` and `cache_read_input_tokens` are now carried through `UsageInfo`, `StepUsage`, and `CostSummary`, and counted in `total_tokens`. Cost reports previously omitted prompt-cache tokens, under-reporting the bill by up to ~63x on cached workloads. The pipeline summary now shows cache write/read totals so silent prompt-cache failures are visible. (#108)

### Internal
- CI: Markdown is excluded from ruff (`extend-exclude = ["*.md"]`). The dev extras are unbounded, so CI resolved ruff 0.16, which began formatting Python code blocks inside Markdown and failed `ruff format --check` on 24 untouched files across `README.md`, `docs/`, and `.claude/skills/`. The network-blocking test fixture also now patches `httpx2` as well as `httpx`, since `openai>=3` moved to the `httpx2` package and CI installs only `.[dev]`, which pulls neither on its own. `main` had been red since both landed. (#113)

## [1.3.0] - 2026-05-19

### Added
- Agentic SDLC scaffolding: GitHub Actions for `@claude` mention, PR review, weekly maintenance, and dogfooded issue triage. Repo-scoped slash commands under `.claude/commands/`. Local `PostToolUse` hook for ruff-on-edit. See `AGENTS.md`.

### Changed
- **`EnrichmentConfig.enable_caching` now defaults to `True`** (was `False`). Re-runs of unchanged inputs no longer re-pay the API cost. Opt out with `EnrichmentConfig(enable_caching=False)` for one-off runs.
- **`LLMStep` auto-detects provider from model name.** `claude-*` → `AnthropicClient`, `gemini-*` → `GoogleClient`, anything else (or any model with `base_url` set) → `OpenAIClient`. Previously every Claude/Gemini user had to pass `client=AnthropicClient()` or `client=GoogleClient()` explicitly. Explicit `client=` still wins. (#23)

### Fixed
- `LLMStep.parse_response` now strips markdown code fences before `json.loads`. Fixes parse failures with Claude Haiku + grounding tools, where the structured-output constraint is disabled and the model wraps JSON in ` ``` ` fences. (#7)
- `accrue/__init__.py` `__version__` was out of sync with `pyproject.toml`.
- Google provider: Option-B rate-limit fallback now uses a word-boundary regex (`\brate[\s_-]?limit\b`) instead of bare `"rate" in exc_str`, preventing false positives on words like "iterate" or "accelerator". (#80)
- Google provider: `DeadlineExceeded` errors now set `status_code=408` (both typed Option-A and substring Option-B paths) for consistency with the Anthropic adapter. (#80)
- Checkpoint: pre-1.2.1 files using the legacy `__type__` sentinel are now detected, logged as a `WARNING`, and discarded so pipelines resume cleanly instead of silently misreading typed values. (#80)
- Pipeline: submit-batch cleanup now logs a `WARNING` for any `cancel_batch` failures so users know if orphaned batches may still be billable after a submit error. (#80)

### Performance
- **Streaming worker pool** replaces eager `asyncio.Task` materialization in the realtime pipeline path. Previously, running a step across N rows created N pending tasks up-front; for 50k rows × multi-step pipelines that meant hundreds of thousands of objects in the event-loop's ready queue. The new design keeps a fixed pool of `max_workers` tasks pulling from a bounded `asyncio.Queue`, so memory and scheduling overhead is O(max_workers) regardless of row count. All existing semantics are preserved: `on_error="raise"/"continue"`, hooks, caching, `run_if`/`skip_if` predicates, checkpointing, and cancellation drain. (#78)

### Internal
- `AnthropicClient._warned_grounding_schema` is per-client-instance scope (not per-process). `LLMStep` constructs a fresh client per `Pipeline.run_async()` call, so the grounding-schema incompatibility warning fires once per run. (#80)

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

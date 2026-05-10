# Agents in the Accrue repo

Accrue is built using an agentic software development lifecycle modelled on Anthropic's own internal practice — colloquially called **"antfooding"** (Anthropic's technical staff are "ants"; antfooding is their version of dogfooding). Internally, ~70–80% of Anthropic engineers use Claude Code daily, and they use it to build Claude Code itself. This file describes the equivalent setup for the accrue repo: GitHub Actions wired to the official `claude-code-action`, repo-scoped slash commands, native Claude Code hooks, and one dogfooded pipeline that uses accrue itself to triage its own issues.

The patterns here are taken from the official documentation, not guessed. Citations are at the bottom of this file.

## TL;DR

| What | Trigger | Where |
|---|---|---|
| `@claude` mention bot | Comment containing `@claude` on any issue, PR, or review | `.github/workflows/claude.yml` |
| Auto PR review (inline) | Every non-draft PR opened/updated against `main` | `.github/workflows/claude-review.yml` |
| Weekly maintenance | Sunday 02:00 UTC + manual `workflow_dispatch` | `.github/workflows/weekly-maintenance.yml` |
| Accrue-triages-accrue | New issue opened | `.github/workflows/accrue-triage.yml` |
| Lint-on-edit (local) | After every Edit/Write tool call by Claude Code | `.claude/settings.json` + `.claude/hooks/lint-changed.sh` |
| `/ship-issue <num>` | Local Claude Code | `.claude/commands/ship-issue.md` |
| `/release <version>` | Local Claude Code | `.claude/commands/release.md` |
| `/sync-docs` | Local Claude Code | `.claude/commands/sync-docs.md` |
| `/triage-issues` | Local Claude Code | `.claude/commands/triage-issues.md` |

## Required secrets

Add this in **Settings → Secrets and variables → Actions**:

- `ANTHROPIC_API_KEY` — used by every workflow, including the dogfooded triage pipeline (which runs against Claude Haiku via accrue's Anthropic provider).

Workflows fail closed without it — no agent runs, no PR comments, no surprise API spend.

## Action version pin

All workflows pin `anthropics/claude-code-action@v1`. Older examples on the web use `@beta` and the deprecated `direct_prompt` / `mode` / `custom_instructions` / `max_turns` / `model` inputs — those were collapsed into `prompt` and `claude_args` in v1.0. Don't copy `@beta` examples without translating.

## How the pieces fit together

### 1. Local development — slash commands + hooks

When working in this repo with Claude Code, four project-scoped commands appear in the `/` menu. They encode accrue's specific workflow (pytest, ruff, version-tag matching, the `docs/guides/` layout) so you don't have to re-explain it every session.

- **`/ship-issue <num>`** — read the issue, plan, code, test, lint, open a PR.
- **`/release <version>`** — bump `pyproject.toml`, draft `CHANGELOG.md` from commits since the last tag, create the tag.
- **`/sync-docs`** — diff `accrue/` against `docs/` + `README.md`, surface drift, propose patches.
- **`/triage-issues`** — sweep open issues, apply labels, ask for repro on bug reports, link near-duplicates.

`.claude/commands/` is where these live. It's still the supported location, though Claude Code is moving toward `.claude/skills/` as the unified home for commands+skills — both work today.

`.claude/settings.json` adds a **PostToolUse hook** that runs `ruff check` + `ruff format --check` on every Python file Claude edits. If it fails, the hook exits 2, stderr is fed back to Claude, and Claude fixes it before stopping. This is the "self-healing dev loop" pattern — one of the most-used hook configurations in the docs.

### 2. CI — agents on GitHub

- **`@claude` mention bot.** Comment `@claude implement #9` on any issue or PR and the action picks it up. v1 auto-detects mode from event context — no `mode:` input needed.
- **Auto PR review.** Runs on every non-draft PR. Uses `track_progress: true` for a visible tracking comment, and the `mcp__github_inline_comment__create_inline_comment` MCP tool so feedback lands as inline review comments rather than one wall of text. Tools are scoped via `claude_args: --allowedTools` for safety.
- **Weekly maintenance.** Cron + `workflow_dispatch`. Sweeps stale issues, doc drift, version coherence, examples sanity. Posts one summary issue per run.

### 3. Dogfooded triage — the meta move

When an issue is opened, **`accrue-triage.yml`** runs `.github/scripts/triage_issue.py`, which builds a one-row Accrue pipeline over the issue's title and body and emits structured fields (`kind`, `needs_info`, `good_first_issue`). The workflow then filters those against the repo's actual labels and applies what matches.

This is on purpose: the easiest way to dogfood a data-enrichment library is to use it on the data your repo already produces. The official action also offers a built-in issue-triage recipe (see solutions.md) — we're not using it for *this one workflow* because the dogfooding is the point. The two are complementary.

## Adding a new agent or command

1. **New slash command** — drop a markdown file in `.claude/commands/`. Filename is the command name. Frontmatter format: `description`, optional `argument-hint`. Body is the prompt.
2. **New workflow** — add a YAML file under `.github/workflows/`. Always pin `@v1`. Use `prompt:` (not the deprecated `direct_prompt:`). Scope tools via `claude_args: --allowedTools "..."`.
3. **New hook** — extend `.claude/settings.json`. Useful events for this repo: `PostToolUse` (file-level checks), `Stop` (turn-end checks), `PreToolUse` with `if: "Bash(rm *)"` (block destructive shell). Hook scripts go in `.claude/hooks/`.
4. **Update this file** — every new entry-point gets a row in the TL;DR table above.

## Costs and guardrails

- All agentic workflows are gated: `@claude` requires explicit mention, review runs only on PRs, triage runs only on new issues, maintenance runs weekly. No infinite loops, no scheduled spend.
- The dogfooded triage pipeline uses `claude-haiku-4-5-20251001` and processes one row per issue — pennies per run.
- The PR review action uses `--allowedTools` to constrain which Bash commands and MCP tools it can call. Don't widen this without a reason.
- To temporarily disable a workflow, comment out its `on:` block — don't delete the file.

## Sources (verified, not guessed)

- **`claude-code-action` repo** — [`github.com/anthropics/claude-code-action`](https://github.com/anthropics/claude-code-action). Authoritative for inputs, version tags, MCP tool names.
- **Solutions guide** — [`solutions.md`](https://github.com/anthropics/claude-code-action/blob/main/docs/solutions.md). PR review, issue triage, scheduled maintenance, doc-sync recipes are lifted from here.
- **Usage guide** — [`usage.md`](https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md). v0→v1 migration table, full input reference.
- **Claude Code GitHub Actions docs** — [`code.claude.com/docs/en/github-actions`](https://code.claude.com/docs/en/github-actions).
- **Hooks reference** — [`code.claude.com/docs/en/hooks`](https://code.claude.com/docs/en/hooks). Authoritative for `.claude/settings.json` schema, event names, exit-code semantics.
- **"Antfooding" practice** — described publicly by the Claude Code team on the [Every podcast](https://every.to/podcast/how-to-use-claude-code-like-the-people-who-built-it). Source of the 70–80% internal-usage figure.
- **Awesome Claude Code** — [`github.com/hesreallyhim/awesome-claude-code`](https://github.com/hesreallyhim/awesome-claude-code). Curated real-world commands/hooks/skills for cross-reference.

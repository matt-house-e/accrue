#!/usr/bin/env bash
# Lint the file Claude just edited. Runs after every Edit/Write, scoped
# to the single file via the tool input passed on stdin. Exits 2 on
# failure so Claude is told (via stderr) to fix it before continuing —
# the "self-healing dev loop" pattern from Claude Code's hook docs.
#
# Wired in .claude/settings.json under hooks.PostToolUse.

set -u

# Pick a ruff invocation. Prefer ruff on PATH (fastest); fall back to
# `uvx ruff` which downloads and runs it on demand (works without a
# venv activation). If neither tool is available, silently no-op so
# we don't block every turn on a missing dev dep.
if command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
elif command -v uvx >/dev/null 2>&1; then
  RUFF=(uvx ruff)
else
  exit 0
fi

file=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null || true)

case "$file" in
  *.py)
    if ! "${RUFF[@]}" check "$file" 1>&2; then exit 2; fi
    if ! "${RUFF[@]}" format --check "$file" 1>&2; then exit 2; fi
    ;;
esac

exit 0

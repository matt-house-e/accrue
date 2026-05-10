#!/usr/bin/env bash
# Lint the file Claude just edited. Runs after every Edit/Write, scoped
# to the single file via the tool input passed on stdin. Exits 2 on
# failure so Claude is told (via stderr) to fix it before continuing —
# the "self-healing dev loop" pattern from Claude Code's hook docs.
#
# Wired in .claude/settings.json under hooks.PostToolUse.

set -u

# Skip gracefully if ruff isn't on PATH (e.g. venv not activated).
# Silent no-op rather than blocking every turn for setup reasons.
command -v ruff >/dev/null 2>&1 || exit 0

file=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null || true)

case "$file" in
  *.py)
    if ! ruff check "$file" 1>&2; then exit 2; fi
    if ! ruff format --check "$file" 1>&2; then exit 2; fi
    ;;
esac

exit 0

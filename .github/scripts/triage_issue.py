"""Dogfood: triage a GitHub issue using accrue itself.

Reads issue title + body from env (set by the calling workflow), runs a
one-row accrue pipeline, prints the labels to apply on stdout. The
workflow filters those against the repo's actual labels before applying,
so missing labels fail soft.

Usage (in CI):
    ISSUE_TITLE="..." ISSUE_BODY="..." python .github/scripts/triage_issue.py

Output (stdout, one line):
    label1,label2,label3
"""

from __future__ import annotations

import os
import sys

from accrue import LLMStep, Pipeline
from accrue.providers import AnthropicClient


def main() -> int:
    title = os.environ.get("ISSUE_TITLE", "").strip()
    body = os.environ.get("ISSUE_BODY", "").strip()

    if not title:
        print("error: ISSUE_TITLE env var is empty", file=sys.stderr)
        return 1

    # accrue's LLMStep defaults to OpenAIClient regardless of model name
    # (see accrue/steps/llm.py::_resolve_client). We pass AnthropicClient
    # explicitly so this works against ANTHROPIC_API_KEY only.
    pipeline = Pipeline(
        [
            LLMStep(
                "triage",
                model="claude-haiku-4-5-20251001",
                client=AnthropicClient(),
                fields={
                    "kind": {
                        "prompt": (
                            "Classify the issue. 'bug' if something is broken or "
                            "behaves unexpectedly; 'enhancement' for a new feature "
                            "or improvement; 'documentation' for doc-only work; "
                            "'question' for a usage question with no defect."
                        ),
                        "enum": ["bug", "enhancement", "documentation", "question"],
                    },
                    "needs_info": {
                        "prompt": (
                            "True only if this is a bug report AND the issue lacks "
                            "either a code snippet, error message, or version number. "
                            "False for enhancements and documentation issues."
                        ),
                        "enum": ["true", "false"],
                    },
                    "good_first_issue": {
                        "prompt": (
                            "True if this is small, well-scoped, and a newcomer "
                            "could reasonably tackle it without deep codebase "
                            "knowledge. False otherwise."
                        ),
                        "enum": ["true", "false"],
                    },
                },
            )
        ]
    )

    result = pipeline.run(
        [
            {
                "title": title,
                "body": body or "(no body)",
            }
        ]
    )

    # Surface row-level errors so silent LLM failures don't masquerade
    # as "pipeline produced no rows".
    if result.has_errors():
        for err in result.errors[:5]:
            print(f"row error: {err!r}", file=sys.stderr)

    if not result.success_rate:
        print(
            "error: pipeline produced no successful rows "
            "(see row errors above)",
            file=sys.stderr,
        )
        return 1

    row = result.data.iloc[0].to_dict()
    labels: list[str] = []

    kind = row.get("kind", "").strip().lower()
    if kind in {"bug", "enhancement", "documentation", "question"}:
        labels.append(kind)

    if str(row.get("needs_info", "")).lower() == "true":
        labels.append("needs info")

    if str(row.get("good_first_issue", "")).lower() == "true":
        labels.append("good first issue")

    print(",".join(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())

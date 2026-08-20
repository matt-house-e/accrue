"""Generate the captured run-log fixtures for capture-tier / attempt tests (#134).

A small deterministic pipeline — 3 rows, one ``LLMStep`` driven by a scripted
fake client so there is **no network, no API key, no real LLM** — exercised at
``capture="prompts"``.  The script makes some cells retry so the run emits
``row_attempt`` records of every kind:

* ``acme``    — succeeds on the first try               → 1 parse attempt
* ``globex``  — one rate-limit, then succeeds           → 1 api + 1 parse attempt
* ``initech`` — one unparseable reply, then succeeds     → 2 parse attempts

The scripted API key embedded in ``globex``'s row is there on purpose: it must
come back ``***REDACTED***`` in the sidecar, proving captured bodies are
sanitised the same way error text is.

Two files are written and committed:

* ``run_captured.jsonl``          — the main log (``row_attempt`` + ``row_complete``)
* ``run_captured.prompts.jsonl``  — the prompt sidecar the ``prompt_ref``s point into

Regenerate with::

    python tests/fixtures/generate_captured_fixture.py

``latency_ms`` and ``backoff_s`` vary run to run (wall clock / jitter); the
record *sequence*, kinds, statuses, and prompt-ref resolution are stable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from accrue import EnrichmentConfig, LLMStep, Pipeline
from accrue.schemas.base import UsageInfo
from accrue.steps.providers.base import LLMAPIError, LLMResponse

FIXTURE_PATH = Path(__file__).resolve().parent / "run_captured.jsonl"
SIDECAR_PATH = Path(__file__).resolve().parent / "run_captured.prompts.jsonl"

FIELD = "grade"

# A secret planted in one row's data — it must be redacted in the sidecar.
PLANTED_SECRET = "sk-fixturesecret0123456789abcdef"


def build_rows() -> list[dict]:
    """3 rows; first column is a string so display_key resolves to 'company'."""
    return [
        {"company": "acme", "note": "clean"},
        {"company": "globex", "note": f"api_key={PLANTED_SECRET}"},
        {"company": "initech", "note": "clean"},
    ]


class _ScriptedClient:
    """Deterministic fake LLM client: scripted retries keyed off the row.

    Each row's company name appears in the rendered user prompt; the client
    counts attempts per company and follows a fixed script, so the same run
    reproduces the same record sequence with no network involved.
    """

    #: company -> list of behaviours, one per attempt.
    _SCRIPT: dict[str, list[str]] = {
        "acme": ["ok"],
        "globex": ["rate_limit", "ok"],
        "initech": ["bad_json", "ok"],
    }

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def _company(self, messages: list[dict[str, Any]]) -> str:
        # messages[1] is the original user prompt carrying the row data; it is
        # rebuilt fresh on an API retry and left intact across parse retries.
        haystack = " ".join(m.get("content", "") for m in messages)
        for name in self._SCRIPT:
            if name in haystack:
                return name
        return "acme"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse:
        company = self._company(messages)
        n = self._counts.get(company, 0)
        self._counts[company] = n + 1
        script = self._SCRIPT[company]
        behaviour = script[n] if n < len(script) else "ok"

        if behaviour == "rate_limit":
            raise LLMAPIError("rate limited", status_code=429, retry_after=0.0, is_rate_limit=True)
        usage = UsageInfo(
            prompt_tokens=11, completion_tokens=3, total_tokens=14, model="fake-model"
        )
        if behaviour == "bad_json":
            return LLMResponse(content="not json at all", usage=usage)
        grade = {"acme": "A", "globex": "B", "initech": "C"}[company]
        return LLMResponse(content=json.dumps({FIELD: grade}), usage=usage)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            LLMStep(
                name="classify",
                fields={FIELD: "Assign a one-letter grade to the company"},
                client=_ScriptedClient(),
                max_retries=2,
                cache=False,
            )
        ]
    )


def generate(path: Path | str = FIXTURE_PATH, sidecar: Path | str = SIDECAR_PATH) -> Path:
    """Run the scripted pipeline at capture=prompts; return the main-log path."""
    path = Path(path)
    sidecar = Path(sidecar)
    # Both files are append-only; start from scratch for a reproducible fixture.
    for p in (path, sidecar):
        if p.exists():
            p.unlink()
    # Seed the RNG so the api-retry backoff jitter is stable across regens.
    random.seed(1234)
    config = EnrichmentConfig(
        enable_caching=False,
        enable_progress_bar=False,
        max_workers=1,
        max_retries=2,
        retry_base_delay=0.001,
    )
    build_pipeline().run(build_rows(), config=config, run_log=str(path), capture="prompts")
    return path


if __name__ == "__main__":
    out = generate()
    print(f"wrote {out}")
    print(f"wrote {SIDECAR_PATH}")

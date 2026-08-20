"""Generate the golden run-log fixture: ``tests/fixtures/run_small.jsonl``.

A small deterministic pipeline — 12 rows, 3 function-based steps, no LLM
providers, no network, no API keys.  Row 7 errors in step ``score``; row 3
is skipped by ``run_if`` in step ``flag``.  Published for accrue-ui's
contract tests (issue #128); the contract itself is specified in
``docs/guides/run-log.md``.

Regenerate with:

    python tests/fixtures/generate_run_fixture.py

Regeneration is stable: the record sequence is identical apart from
timestamps (``t``, ``started_at``, ``elapsed_*``) and ``run_id``.
Determinism comes from ``max_workers=1`` (rows complete in order),
``enable_caching=False`` (``from_cache`` is always false), and a linear
step DAG (no same-level parallelism to interleave records).
"""

from __future__ import annotations

from pathlib import Path

from accrue import EnrichmentConfig, FunctionStep, Pipeline
from accrue.steps.base import StepContext

FIXTURE_PATH = Path(__file__).resolve().parent / "run_small.jsonl"

NUM_ROWS = 12
ERROR_ROW = 7  # errors in step 'score'
SKIP_ROW = 3  # skipped by run_if in step 'flag'


def build_rows() -> list[dict]:
    """12 rows; first column is a string so display_key resolves to 'company'."""
    return [{"company": f"company-{i:02d}", "employees": (i + 1) * 10} for i in range(NUM_ROWS)]


def _normalize(ctx: StepContext) -> dict:
    return {"name_upper": ctx.row["company"].upper()}


def _score(ctx: StepContext) -> dict:
    if ctx.row["company"] == f"company-{ERROR_ROW:02d}":
        raise ValueError(f"cannot score {ctx.row['company']}")
    return {"score": ctx.row["employees"] / 10}


def _flag(ctx: StepContext) -> dict:
    score = ctx.prior_results.get("score")
    return {"flagged": bool(score is not None and score > 6)}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            FunctionStep("normalize", _normalize, fields=["name_upper"]),
            FunctionStep("score", _score, fields=["score"], depends_on=["normalize"]),
            FunctionStep(
                "flag",
                _flag,
                fields=["flagged"],
                depends_on=["score"],
                run_if=lambda row, prior: row["company"] != f"company-{SKIP_ROW:02d}",
            ),
        ]
    )


def generate(path: Path | str = FIXTURE_PATH) -> Path:
    """Run the pipeline with run_log pointed at *path* and return the path."""
    path = Path(path)
    # Run logs are append-only; start from scratch for a reproducible fixture.
    if path.exists():
        path.unlink()
    config = EnrichmentConfig(
        enable_caching=False,
        enable_progress_bar=False,
        max_workers=1,
    )
    build_pipeline().run(build_rows(), config=config, run_log=str(path))
    return path


if __name__ == "__main__":
    print(f"wrote {generate()}")

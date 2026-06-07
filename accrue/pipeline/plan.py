"""Pipeline.plan() — a dry-run preview with cost extrapolation.

A plan answers "what will this pipeline do, and what will the full run cost?"
*before* you commit to spending on the whole dataset.  It introspects each
step's resolved prompt and JSON schema, runs a small capped sample (real
calls), and extrapolates the sample's token usage to the full row count.

Modelled on ``terraform plan`` / Claude Code's plan mode.  See issue #12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.exceptions import RowError
from ..schemas.base import CostSummary, StepUsage


@dataclass
class StepPlan:
    """Preview of a single step within a :class:`PipelinePlan`.

    Attributes:
        name: Step name.
        kind: ``"llm"``, ``"function"``, or ``"other"`` (custom Step).
        depends_on: Names of upstream steps this one consumes.
        model: Model identifier (LLM steps only).
        system_prompt: The resolved system prompt for the first sample row
            (LLM steps only).  Prior-step fields are empty in this preview —
            it shows the prompt *template* applied to one row, not a
            mid-pipeline render.
        response_format: The ``response_format`` the step will send — a JSON
            schema dict, or ``{"type": "json_object"}`` for freeform — (LLM
            steps only).
    """

    name: str
    kind: str
    depends_on: list[str] = field(default_factory=list)
    model: str | None = None
    system_prompt: str | None = None
    response_format: dict[str, Any] | None = None


@dataclass
class PipelinePlan:
    """A dry-run preview returned by :meth:`Pipeline.plan`.

    Carries the per-step prompts/schemas, the sample rows and their real
    (capped) outputs, the sample's measured cost, and the cost extrapolated
    to the full dataset.  Call :meth:`summary` for a human-readable preview.
    """

    steps: list[StepPlan]
    sample_rows: list[dict[str, Any]]
    sample_outputs: list[dict[str, Any]]
    sample_errors: list[RowError]
    total_rows: int
    sample_size: int
    sample_cost: CostSummary
    estimated_cost: CostSummary

    def summary(self) -> str:
        """Render a human-readable preview — prompts, schemas, sample
        outputs, and estimated full-run cost.

        Returns the text so it's easy to test or log; print it to preview::

            print(pipeline.plan(df).summary())
        """
        lines: list[str] = []
        rule = "─" * 60
        lines.append("")
        lines.append(rule)
        lines.append(
            f"  Pipeline Plan — {len(self.steps)} step(s), "
            f"sampled {self.sample_size} of {self.total_rows} rows"
        )
        lines.append(rule)

        # Per-step prompts + schemas
        for sp in self.steps:
            header = f"  • {sp.name}  [{sp.kind}"
            if sp.model:
                header += f": {sp.model}"
            header += "]"
            if sp.depends_on:
                header += f"  ← {', '.join(sp.depends_on)}"
            lines.append("")
            lines.append(header)
            if sp.response_format is not None:
                fields = _schema_fields(sp.response_format)
                if fields:
                    lines.append(f"      schema fields: {', '.join(fields)}")
                else:
                    lines.append("      schema: json_object (freeform)")
            if sp.system_prompt:
                lines.append("      system prompt:")
                for pl in _indent_block(sp.system_prompt, "        "):
                    lines.append(pl)

        # Sample outputs
        lines.append("")
        lines.append(f"  Sample outputs ({len(self.sample_outputs)} row(s)):")
        for row in self.sample_outputs:
            lines.append(f"    {row}")
        if self.sample_errors:
            lines.append(f"    {len(self.sample_errors)} sample row(s) errored")

        # Cost
        lines.append("")
        lines.append("  Cost")
        lines.append(
            f"    sample (measured):  {_fmt_tokens(self.sample_cost.total_tokens)} tokens"
            f"  ({self.sample_cost.total_prompt_tokens:,} in / "
            f"{self.sample_cost.total_completion_tokens:,} out)"
        )
        lines.append(
            f"    full run (est.):    {_fmt_tokens(self.estimated_cost.total_tokens)} tokens"
            f"  ({self.estimated_cost.total_prompt_tokens:,} in / "
            f"{self.estimated_cost.total_completion_tokens:,} out)"
        )
        lines.append(
            "    estimate extrapolates measured sample tokens to "
            f"{self.total_rows} rows (cached rows excluded)."
        )
        lines.append(rule)
        lines.append("")
        return "\n".join(lines)


def extrapolate_cost(sample_cost: CostSummary, total_rows: int) -> CostSummary:
    """Extrapolate a capped sample's token usage to the full dataset.

    Per step, the per-row token cost is computed over the rows that actually
    called the API (cache misses, or every processed row when caching is off) —
    cached sample rows incur no tokens, so counting them would understate the
    estimate — then scaled to ``total_rows``.
    A step whose sample rows were all cached (or which uses no tokens, e.g. a
    FunctionStep) contributes zero, since re-running cached rows is free.
    """
    est = CostSummary()
    for name, usage in sample_cost.steps.items():
        # Rows that actually consumed tokens: cache misses when caching is on,
        # else every processed row (caching off reports misses as 0).
        executed = usage.cache_misses or usage.rows_processed
        if executed <= 0:
            scaled = StepUsage(model=usage.model, rows_processed=total_rows)
        else:
            factor = total_rows / executed
            scaled = StepUsage(
                prompt_tokens=round(usage.prompt_tokens * factor),
                completion_tokens=round(usage.completion_tokens * factor),
                total_tokens=round(usage.total_tokens * factor),
                rows_processed=total_rows,
                cache_misses=total_rows,
                model=usage.model,
            )
        est.steps[name] = scaled
        est.total_prompt_tokens += scaled.prompt_tokens
        est.total_completion_tokens += scaled.completion_tokens
        est.total_tokens += scaled.total_tokens
    return est


def _schema_fields(response_format: dict[str, Any]) -> list[str]:
    """Pull the top-level property names out of a json_schema response_format."""
    schema = response_format.get("json_schema", {})
    props = schema.get("schema", {}).get("properties", {})
    return list(props.keys())


def _fmt_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def _indent_block(text: str, indent: str, max_lines: int = 12) -> list[str]:
    """Indent a block of text, truncating very long prompts for the preview."""
    raw = text.splitlines()
    shown = raw[:max_lines]
    out = [f"{indent}{line}" for line in shown]
    if len(raw) > max_lines:
        out.append(f"{indent}… ({len(raw) - max_lines} more lines)")
    return out

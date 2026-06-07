"""Heuristic-driven run reports for ``PipelineResult``.

The point of this module is *judgment*, not statistics. Each builtin
heuristic encodes a pattern that suggests the pipeline is misbehaving
(enum collapse, numeric clipping, refusal phrases, cache thrash, …)
and emits a :class:`Finding` whose message names a probable cause and
a suggested action.

Renderers are pure: they take a :class:`ReportContext` plus the list
of findings and produce Markdown or HTML. No I/O, no globals.
"""

from __future__ import annotations

import html as _html
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


SEVERITY_ICON = {"critical": "⛔", "warning": "⚠️", "info": "ℹ️"}
SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Finding:
    """A single heuristic flag — what looks wrong, probable cause, suggested action.

    Attributes:
        code: Stable heuristic identifier (e.g. ``"enum_collapse"``).
            Pass to ``report(disable=[...])`` to mute a noisy heuristic.
        severity: ``"critical"``, ``"warning"``, or ``"info"``.
        subject: The field or step the finding is about (e.g. ``"category"``).
        message: One sentence: probable cause + suggested action.
        evidence: Supporting numbers (percentages, counts) for the renderer.
    """

    code: str
    severity: str
    subject: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report context — what heuristics see
# ---------------------------------------------------------------------------


@dataclass
class ReportContext:
    """Snapshot of a pipeline run, passed to every heuristic.

    Constructed by :meth:`PipelineResult.report`; not part of the public
    API but documented here so users can write custom heuristics against
    the same shape.
    """

    data: pd.DataFrame | list[dict[str, Any]]
    cost: Any
    errors: list[Any]
    field_specs: dict[str, dict[str, Any]]
    pipeline_elapsed_seconds: float
    step_elapsed_seconds: dict[str, float]

    @property
    def num_rows(self) -> int:
        return len(self.data)

    def values(self, field_name: str) -> list[Any]:
        """All non-null values for *field_name*, preserving row order."""
        if isinstance(self.data, pd.DataFrame):
            if field_name not in self.data.columns:
                return []
            series = self.data[field_name]
            return [v for v in series.tolist() if not _is_null(v)]
        return [
            row[field_name]
            for row in self.data
            if field_name in row and not _is_null(row[field_name])
        ]


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


HeuristicFn = Callable[[ReportContext], list[Finding]]


# ---------------------------------------------------------------------------
# Builtin heuristics
# ---------------------------------------------------------------------------


_ENUM_COLLAPSE_THRESHOLD = 0.60
_NUMERIC_CLIP_THRESHOLD = 0.25
_REFUSAL_THRESHOLD = 0.05
_RETRY_STORM_THRESHOLD = 0.05
_CACHE_THRASH_THRESHOLD = 0.10
_COST_OUTLIER_THRESHOLD = 0.60
_LENGTH_LOWER_FRACTION = 0.30
_LENGTH_UPPER_FRACTION = 2.00

_REFUSAL_PATTERNS = [
    r"\bI cannot determine\b",
    r"\bI can[' ]?t determine\b",
    r"\binsufficient information\b",
    r"\bI don[' ]?t have\b",
    r"\bunable to (?:determine|find|provide)\b",
    r"\bnot enough (?:information|context|data)\b",
    r"\bno (?:information|data) available\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

_LENGTH_HINT_RE = re.compile(
    r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*(words?|chars?|characters?|sentences?|tokens?)",
    re.IGNORECASE,
)


def _enum_collapse(ctx: ReportContext) -> list[Finding]:
    """Top value of an enum/categorical field exceeds 60% of non-null rows."""
    findings: list[Finding] = []
    for name, spec in ctx.field_specs.items():
        if not spec.get("enum"):
            continue
        values = ctx.values(name)
        if not values:
            continue
        counter = Counter(values)
        top_value, top_count = counter.most_common(1)[0]
        ratio = top_count / len(values)
        if ratio >= _ENUM_COLLAPSE_THRESHOLD:
            pct = round(ratio * 100)
            findings.append(
                Finding(
                    code="enum_collapse",
                    severity="warning",
                    subject=name,
                    message=(
                        f"`{name}` collapsed to `{top_value}` on {pct}% of rows. "
                        "Likely cause: the enum is too narrow or the prompt isn't "
                        "differentiating between options."
                    ),
                    evidence={
                        "top_value": top_value,
                        "ratio": round(ratio, 3),
                        "total": len(values),
                    },
                )
            )
    return findings


def _numeric_clipping(ctx: ReportContext) -> list[Finding]:
    """A single value (likely a bound) is hit by >25% of numeric outputs."""
    findings: list[Finding] = []
    for name, spec in ctx.field_specs.items():
        if spec.get("type") != "Number":
            continue
        raw = ctx.values(name)
        nums: list[float] = []
        for v in raw:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                continue
        if len(nums) < 4:
            continue
        counter = Counter(nums)
        top_value, top_count = counter.most_common(1)[0]
        ratio = top_count / len(nums)
        if ratio < _NUMERIC_CLIP_THRESHOLD:
            continue
        # Only flag if the dominant value sits at the edge of observed range.
        lo, hi = min(nums), max(nums)
        if top_value not in (lo, hi):
            continue
        pct = round(ratio * 100)
        edge = "max" if top_value == hi else "min"
        pretty = int(top_value) if top_value.is_integer() else top_value
        findings.append(
            Finding(
                code="numeric_clipping",
                severity="warning",
                subject=name,
                message=(
                    f"`{name}` hit {edge} ({pretty}) on {pct}% of rows. "
                    "Model isn't actually scoring — consider rephrasing or "
                    "breaking the criterion into sub-fields."
                ),
                evidence={
                    "top_value": pretty,
                    "edge": edge,
                    "ratio": round(ratio, 3),
                    "total": len(nums),
                },
            )
        )
    return findings


def _length_anomaly(ctx: ReportContext) -> list[Finding]:
    """String field's mean output is far outside a parseable length hint."""
    findings: list[Finding] = []
    for name, spec in ctx.field_specs.items():
        if spec.get("type", "String") != "String":
            continue
        fmt = spec.get("format")
        if not fmt:
            continue
        m = _LENGTH_HINT_RE.search(fmt)
        if not m:
            continue
        lo_hint, hi_hint, unit = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        values = [str(v) for v in ctx.values(name) if v]
        if not values:
            continue
        if unit.startswith("word"):
            lengths = [len(v.split()) for v in values]
            unit_label = "words"
        elif unit.startswith("sentence"):
            lengths = [max(1, len(re.split(r"[.!?]+", v.strip()))) for v in values]
            unit_label = "sentences"
        elif unit.startswith("token"):
            lengths = [max(1, len(v) // 4) for v in values]
            unit_label = "tokens (approx)"
        else:
            lengths = [len(v) for v in values]
            unit_label = "chars"

        mean_len = sum(lengths) / len(lengths)
        target_mid = (lo_hint + hi_hint) / 2
        if target_mid == 0:
            continue
        ratio = mean_len / target_mid
        if ratio >= _LENGTH_LOWER_FRACTION and ratio <= _LENGTH_UPPER_FRACTION:
            continue
        direction = "truncated or the model is being lazy" if ratio < 1 else "verbose"
        findings.append(
            Finding(
                code="length_anomaly",
                severity="warning",
                subject=name,
                message=(
                    f"`{name}` averaged {mean_len:.0f} {unit_label}; spec asked for "
                    f"{lo_hint}–{hi_hint}. Outputs are likely {direction}."
                ),
                evidence={
                    "mean": round(mean_len, 1),
                    "hint_lo": lo_hint,
                    "hint_hi": hi_hint,
                    "unit": unit_label,
                },
            )
        )
    return findings


def _retry_storm(ctx: ReportContext) -> list[Finding]:
    """A step's error rate exceeds 5% — best proxy we have for retry exhaustion."""
    findings: list[Finding] = []
    if not ctx.errors or not ctx.cost or not ctx.cost.steps:
        return findings
    by_step: Counter[str] = Counter(e.step_name for e in ctx.errors)
    for step_name, usage in ctx.cost.steps.items():
        total = usage.rows_processed + usage.rows_skipped
        if total == 0:
            continue
        err_count = by_step.get(step_name, 0)
        ratio = err_count / total
        if ratio < _RETRY_STORM_THRESHOLD:
            continue
        pct = round(ratio * 100)
        findings.append(
            Finding(
                code="retry_storm",
                severity="warning",
                subject=step_name,
                message=(
                    f"{pct}% of `{step_name}` rows errored after retries. "
                    "Schema is probably too strict for the prompt, or the "
                    "model is hitting rate limits — inspect `result.errors`."
                ),
                evidence={"errors": err_count, "total": total, "ratio": round(ratio, 3)},
            )
        )
    return findings


def _cache_thrash(ctx: ReportContext) -> list[Finding]:
    """Cache had some hits but mostly missed — likely a re-run with churned input."""
    findings: list[Finding] = []
    if not ctx.cost or not ctx.cost.steps:
        return findings
    for step_name, usage in ctx.cost.steps.items():
        total = usage.cache_hits + usage.cache_misses
        if total == 0 or usage.cache_hits == 0:
            continue
        if usage.cache_hit_rate >= _CACHE_THRASH_THRESHOLD:
            continue
        pct = round(usage.cache_hit_rate * 100)
        findings.append(
            Finding(
                code="cache_thrash",
                severity="info",
                subject=step_name,
                message=(
                    f"`{step_name}` cache hit rate was {pct}%. Some entries existed "
                    "but most were stale — input fields are probably churning between "
                    "runs (whitespace, ordering, IDs)."
                ),
                evidence={
                    "hits": usage.cache_hits,
                    "misses": usage.cache_misses,
                    "ratio": round(usage.cache_hit_rate, 3),
                },
            )
        )
    return findings


def _refusal_patterns(ctx: ReportContext) -> list[Finding]:
    """A free-text field hits refusal phrases on more than 5% of rows."""
    findings: list[Finding] = []
    for name, spec in ctx.field_specs.items():
        if spec.get("type", "String") != "String":
            continue
        values = ctx.values(name)
        if not values:
            continue
        hits = sum(1 for v in values if isinstance(v, str) and _REFUSAL_RE.search(v))
        if hits == 0:
            continue
        ratio = hits / len(values)
        if ratio < _REFUSAL_THRESHOLD:
            continue
        pct = round(ratio * 100)
        findings.append(
            Finding(
                code="refusal_pattern",
                severity="warning",
                subject=name,
                message=(
                    f'`{name}` matched a refusal phrase ("I cannot determine", '
                    f'"insufficient information", …) on {pct}% of rows. The model '
                    "doesn't think it has the data — check upstream context or grounding."
                ),
                evidence={"hits": hits, "total": len(values), "ratio": round(ratio, 3)},
            )
        )
    return findings


def _cost_outlier(ctx: ReportContext) -> list[Finding]:
    """A single step accounts for >60% of total token spend."""
    findings: list[Finding] = []
    if not ctx.cost or not ctx.cost.steps:
        return findings
    # Skip single-step pipelines: by definition the only step holds 100%
    # of cost, so flagging it carries no information.
    if len(ctx.cost.steps) < 2:
        return findings
    total = ctx.cost.total_tokens
    if total < 1000:
        return findings
    for step_name, usage in ctx.cost.steps.items():
        ratio = usage.total_tokens / total
        if ratio < _COST_OUTLIER_THRESHOLD:
            continue
        pct = round(ratio * 100)
        findings.append(
            Finding(
                code="cost_outlier",
                severity="info",
                subject=step_name,
                message=(
                    f"`{step_name}` consumed {pct}% of pipeline tokens "
                    f"({usage.total_tokens:,} of {total:,}). If this step is the "
                    "bottleneck, try a smaller model, shorter prompt, or batch API."
                ),
                evidence={
                    "step_tokens": usage.total_tokens,
                    "total_tokens": total,
                    "ratio": round(ratio, 3),
                },
            )
        )
    return findings


BUILTIN_HEURISTICS: dict[str, HeuristicFn] = {
    "enum_collapse": _enum_collapse,
    "numeric_clipping": _numeric_clipping,
    "length_anomaly": _length_anomaly,
    "retry_storm": _retry_storm,
    "cache_thrash": _cache_thrash,
    "refusal_pattern": _refusal_patterns,
    "cost_outlier": _cost_outlier,
}


def run_heuristics(
    ctx: ReportContext,
    disable: list[str] | None = None,
) -> list[Finding]:
    """Run every enabled builtin heuristic and return the combined findings."""
    disabled = set(disable or ())
    findings: list[Finding] = []
    for code, fn in BUILTIN_HEURISTICS.items():
        if code in disabled:
            continue
        findings.extend(fn(ctx))
    findings.sort(key=lambda f: (SEVERITY_RANK.get(f.severity, 99), f.code, f.subject))
    return findings


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _format_seconds(s: float) -> str:
    if s >= 60:
        m, sec = divmod(s, 60)
        return f"{int(m)}m {sec:.1f}s"
    return f"{s:.2f}s"


def render_markdown(ctx: ReportContext, findings: list[Finding]) -> str:
    """Render the report as Markdown — pasteable into Slack, GitHub, terminals."""
    lines: list[str] = []
    lines.append("# Pipeline Run Report")
    lines.append("")

    # Headline numbers
    total = ctx.num_rows
    err = len(ctx.errors)
    ok = total - err
    lines.append(
        f"**Rows:** {ok:,}/{total:,} succeeded  "
        f"**Errors:** {err}  "
        f"**Wall time:** {_format_seconds(ctx.pipeline_elapsed_seconds)}"
    )
    cost = ctx.cost
    if cost is not None and cost.total_tokens:
        lines.append(
            f"**Tokens:** {_format_tokens(cost.total_tokens)} "
            f"({cost.total_prompt_tokens:,} in / {cost.total_completion_tokens:,} out)"
        )
    lines.append("")

    # Findings (headline)
    if findings:
        lines.append("## Flagged patterns")
        lines.append("")
        for f in findings:
            icon = SEVERITY_ICON.get(f.severity, "•")
            lines.append(f"- {icon} {f.message} _(`{f.code}`)_")
        lines.append("")
    else:
        lines.append("## ✅ No flagged patterns")
        lines.append("")
        lines.append(
            "Heuristics ran cleanly. See per-step stats below "
            "(or pass `disable=[...]` to silence specific heuristics)."
        )
        lines.append("")

    # Per-step
    if cost is not None and cost.steps:
        lines.append("## Per-step")
        lines.append("")
        lines.append("| Step | Model | Rows | Skipped | Cache | Tokens | Wall time |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for name, usage in cost.steps.items():
            total_cache = usage.cache_hits + usage.cache_misses
            cache_str = f"{round(usage.cache_hit_rate * 100)}%" if total_cache > 0 else "—"
            elapsed = ctx.step_elapsed_seconds.get(name, 0.0)
            lines.append(
                f"| `{name}` "
                f"| {usage.model or '—'} "
                f"| {usage.rows_processed:,} "
                f"| {usage.rows_skipped:,} "
                f"| {cache_str} "
                f"| {_format_tokens(usage.total_tokens)} "
                f"| {_format_seconds(elapsed)} |"
            )
        lines.append("")

    # Errors
    if ctx.errors:
        lines.append("## Errors")
        lines.append("")
        type_counts = Counter(e.error_type for e in ctx.errors)
        summary = ", ".join(f"{t} (×{n})" for t, n in type_counts.most_common(5))
        lines.append(f"{len(ctx.errors)} rows errored. Most common: {summary}.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


_HTML_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #222; }
h1 { border-bottom: 1px solid #ddd; padding-bottom: .3rem; }
h2 { margin-top: 2rem; }
.headline { color: #555; }
.findings { list-style: none; padding-left: 0; }
.findings li { padding: .5rem .75rem; margin-bottom: .35rem;
               border-left: 3px solid #888; background: #f7f7f7; }
.findings li.warning { border-color: #d97706; background: #fff7ed; }
.findings li.info    { border-color: #2563eb; background: #eff6ff; }
.findings li.critical{ border-color: #dc2626; background: #fef2f2; }
.code { color: #777; font-size: .85em; font-family: ui-monospace, monospace; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: .35rem .5rem; border-bottom: 1px solid #eee; text-align: left; }
th { background: #fafafa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code { background: #f1f1f1; padding: 0 .2em; border-radius: 3px; }
""".strip()


def render_html(ctx: ReportContext, findings: list[Finding]) -> str:
    """Render the report as a self-contained HTML document."""
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head>')
    parts.append('<meta charset="utf-8">')
    parts.append("<title>Pipeline Run Report</title>")
    parts.append(f"<style>{_HTML_STYLE}</style>")
    parts.append("</head><body>")
    parts.append("<h1>Pipeline Run Report</h1>")

    total = ctx.num_rows
    err = len(ctx.errors)
    ok = total - err
    cost = ctx.cost
    elapsed_s = _html.escape(_format_seconds(ctx.pipeline_elapsed_seconds))
    head_bits = [
        f"<strong>Rows:</strong> {ok:,}/{total:,} succeeded",
        f"<strong>Errors:</strong> {err}",
        f"<strong>Wall time:</strong> {elapsed_s}",
    ]
    if cost is not None and cost.total_tokens:
        head_bits.append(
            f"<strong>Tokens:</strong> {_format_tokens(cost.total_tokens)} "
            f"({cost.total_prompt_tokens:,} in / {cost.total_completion_tokens:,} out)"
        )
    parts.append('<p class="headline">' + " &middot; ".join(head_bits) + "</p>")

    if findings:
        parts.append("<h2>Flagged patterns</h2>")
        parts.append('<ul class="findings">')
        for f in findings:
            sev = _html.escape(f.severity)
            msg = _md_inline_to_html(f.message)
            parts.append(
                f'<li class="{sev}">{msg} <span class="code">({_html.escape(f.code)})</span></li>'
            )
        parts.append("</ul>")
    else:
        parts.append("<h2>✅ No flagged patterns</h2>")
        parts.append("<p>Heuristics ran cleanly.</p>")

    if cost is not None and cost.steps:
        parts.append("<h2>Per-step</h2>")
        parts.append("<table>")
        parts.append(
            "<thead><tr>"
            "<th>Step</th><th>Model</th><th>Rows</th><th>Skipped</th>"
            "<th>Cache</th><th>Tokens</th><th>Wall time</th>"
            "</tr></thead><tbody>"
        )
        for name, usage in cost.steps.items():
            total_cache = usage.cache_hits + usage.cache_misses
            cache_str = f"{round(usage.cache_hit_rate * 100)}%" if total_cache > 0 else "—"
            elapsed = ctx.step_elapsed_seconds.get(name, 0.0)
            parts.append(
                "<tr>"
                f"<td><code>{_html.escape(name)}</code></td>"
                f"<td>{_html.escape(usage.model or '—')}</td>"
                f'<td class="num">{usage.rows_processed:,}</td>'
                f'<td class="num">{usage.rows_skipped:,}</td>'
                f'<td class="num">{cache_str}</td>'
                f'<td class="num">{_format_tokens(usage.total_tokens)}</td>'
                f'<td class="num">{_html.escape(_format_seconds(elapsed))}</td>'
                "</tr>"
            )
        parts.append("</tbody></table>")

    if ctx.errors:
        parts.append("<h2>Errors</h2>")
        type_counts = Counter(e.error_type for e in ctx.errors)
        summary = ", ".join(f"{_html.escape(t)} (×{n})" for t, n in type_counts.most_common(5))
        parts.append(f"<p>{len(ctx.errors)} rows errored. Most common: {summary}.</p>")

    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _md_inline_to_html(text: str) -> str:
    """Tiny markdown-inline → HTML: only handles backticks, since that's all we emit."""
    escaped = _html.escape(text)
    # Re-apply backtick → <code>; we escaped first so HTML inside content is safe.
    return _BACKTICK_RE.sub(r"<code>\1</code>", escaped)

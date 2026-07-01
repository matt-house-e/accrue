"""``accrue.compare()`` — diff two ``PipelineResult`` objects.

The Accrue user's actual iteration loop is "tweak a prompt, re-run, did
anything meaningful change?" — not "what's the F1 score against ground
truth." They have no labels, just two runs. ``compare()`` answers that
question directly: align the two result sets, classify each output field
by its spec (enum / numeric / text), and report what moved.

This module reuses the rendering primitives from :mod:`.report` (token/
seconds formatting, the HTML stylesheet, inline-markdown escaping, and the
null-safe check) so a comparison report looks and feels like
``result.report()``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .pipeline import PipelineResult
from .report import _HTML_STYLE, _format_seconds, _format_tokens, _is_null, _md_inline_to_html

__all__ = ["compare", "ComparisonResult"]


# ---------------------------------------------------------------------------
# Row alignment
# ---------------------------------------------------------------------------


def _is_default_range_index(idx: pd.Index) -> bool:
    """True if *idx* is the plain positional ``RangeIndex(0, len(idx))``.

    This is how both the ``list[dict]`` input path and a freshly-built
    DataFrame index look — i.e. "no caller-supplied index". A caller who
    sliced/reindexed a DataFrame (even into another RangeIndex with a
    different start/step) gets treated as a "real" index instead.
    """
    return isinstance(idx, pd.RangeIndex) and idx.equals(pd.RangeIndex(len(idx)))


def _align_rows(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[list[int], list[int], Literal["positional", "label"]]:
    """Pair up rows of *df_a* and *df_b* for comparison.

    Returns ``(positions_a, positions_b, mode)`` where ``positions_a[i]``
    (an integer position, suitable for ``.iloc``) pairs with
    ``positions_b[i]``, and ``mode`` is ``"positional"`` or ``"label"``.

    A blind ``df_a.index & df_b.index`` is not safe here: it silently
    produces an empty (or wrong) result when the indexes are different
    dtypes, or double-counts/misaligns when either index has duplicate
    labels. So alignment is by explicit case:

    - Both frames have the default positional RangeIndex (the common case
      for ``list[dict]`` input, or any freshly-built DataFrame) -> align by
      position, on the shorter length.
    - Both frames carry a real (non-default) index -> align by label, but
      only after confirming each index is unique.
    - Anything else (one default + one real, or a non-unique real index on
      either side) -> warn and fall back to positional alignment.
    """
    default_a = _is_default_range_index(df_a.index)
    default_b = _is_default_range_index(df_b.index)

    if default_a and default_b:
        n = min(len(df_a), len(df_b))
        if len(df_a) != len(df_b):
            warnings.warn(
                f"compare(): row counts differ (A={len(df_a)}, B={len(df_b)}); "
                f"comparing the first {n} row(s) positionally and ignoring the rest.",
                stacklevel=3,
            )
        return list(range(n)), list(range(n)), "positional"

    if not default_a and not default_b:
        if df_a.index.is_unique and df_b.index.is_unique:
            labels_b = set(df_b.index)
            common = [lab for lab in df_a.index if lab in labels_b]
            unmatched_a = len(df_a) - len(common)
            unmatched_b = len(df_b) - len(common)
            if unmatched_a or unmatched_b:
                warnings.warn(
                    f"compare(): {unmatched_a} row(s) in A and {unmatched_b} row(s) in B "
                    "have no matching index label on the other side; comparing the "
                    "intersection only.",
                    stacklevel=3,
                )
            pos_a = [df_a.index.get_loc(lab) for lab in common]
            pos_b = [df_b.index.get_loc(lab) for lab in common]
            return pos_a, pos_b, "label"

        warnings.warn(
            "compare(): index is non-unique on one or both sides; "
            "falling back to positional row alignment.",
            stacklevel=3,
        )
        n = min(len(df_a), len(df_b))
        return list(range(n)), list(range(n)), "positional"

    # Mixed: one default RangeIndex, one real index.
    warnings.warn(
        "compare(): one side has a default positional index and the other a custom "
        "index; falling back to positional row alignment.",
        stacklevel=3,
    )
    n = min(len(df_a), len(df_b))
    return list(range(n)), list(range(n)), "positional"


# ---------------------------------------------------------------------------
# Cell equality
# ---------------------------------------------------------------------------


def _stable_repr(v: Any) -> str:
    """Deterministic string form for values that can't be compared/hashed directly."""
    try:
        return json.dumps(v, sort_keys=True, default=str)
    except TypeError:
        return repr(v)


def _cells_equal(x: Any, y: Any) -> bool:
    """True if *x* and *y* should be treated as the same output value.

    A naive ``x != y`` has two problems here: ``NaN != NaN`` is ``True``
    (so two byte-identical runs with missing values would be reported as
    fully changed), and it can raise outright on container cells (JSON /
    ``List[String]`` fields hold ``dict``/``list`` values, and an
    elementwise numpy comparison of two such cells can come back as an
    array, which blows up on ``bool()``).

    Both-null compares equal (reusing :func:`_is_null`, the same null
    check ``report()`` uses). Otherwise try direct equality; if that
    raises or doesn't resolve to a plain boolean, fall back to a
    sort-keyed JSON (or ``repr``, if not JSON-able) comparison.
    """
    x_null, y_null = _is_null(x), _is_null(y)
    if x_null or y_null:
        return x_null and y_null
    try:
        return bool(x == y)
    except (ValueError, TypeError):
        return _stable_repr(x) == _stable_repr(y)


def _value_counts_safe(series: pd.Series) -> pd.Series:
    """``value_counts()`` that tolerates unhashable cells (lists/dicts).

    Container cells (JSON / ``List[String]`` fields) raise
    ``TypeError: unhashable type`` from the normal hash-based counting
    path, so non-scalar cells are stringified first via :func:`_stable_repr`.
    """
    stringified = [
        v if isinstance(v, (str, int, float, bool)) else _stable_repr(v)
        for v in series.tolist()
        if not _is_null(v)
    ]
    return pd.Series(stringified, dtype=object).value_counts()


def _frequency_table(series: pd.Series) -> dict[str, float]:
    """value -> fraction of non-null rows, for an enum/categorical column."""
    counts = _value_counts_safe(series)
    total = int(counts.sum())
    if total == 0:
        return {}
    return {str(k): v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Per-field churn
# ---------------------------------------------------------------------------


@dataclass
class FieldChurn:
    """Before/after churn for a single output field.

    ``kind`` selects which of the type-specific attributes are populated:

    - ``"enum"``: ``freq_a``/``freq_b`` — value -> fraction frequency
      tables.
    - ``"numeric"``: ``mean_a``/``mean_b``/``std_a``/``std_b``/``mean_delta``.
    - ``"text"``: ``avg_len_tokens_a``/``avg_len_tokens_b``/``len_delta_tokens``
      (populated only for ``type: String``; ``None`` for Boolean/Date/JSON/
      List[String], which only get a differs-count).

    Numeric fields with no coercible values, or text fields with no rows,
    leave their type-specific attributes as ``None`` rather than raising.
    """

    field: str
    kind: Literal["enum", "numeric", "text"]
    changed: int
    total: int
    freq_a: dict[str, float] | None = None
    freq_b: dict[str, float] | None = None
    mean_a: float | None = None
    mean_b: float | None = None
    std_a: float | None = None
    std_b: float | None = None
    mean_delta: float | None = None
    avg_len_tokens_a: float | None = None
    avg_len_tokens_b: float | None = None
    len_delta_tokens: float | None = None

    @property
    def changed_pct(self) -> float:
        return (self.changed / self.total * 100) if self.total else 0.0


@dataclass
class CostDelta:
    """Token/cache/latency delta between two pipeline runs.

    No pricing fields by design -- there is no pricing data anywhere in
    Accrue, so "cost" here means tokens, cache efficiency, and wall time
    only.
    """

    prompt_tokens_a: int
    prompt_tokens_b: int
    prompt_tokens_delta: int
    completion_tokens_a: int
    completion_tokens_b: int
    completion_tokens_delta: int
    total_tokens_a: int
    total_tokens_b: int
    total_tokens_delta: int
    cache_hit_rate_a: float
    cache_hit_rate_b: float
    cache_hit_rate_delta: float
    elapsed_seconds_a: float
    elapsed_seconds_b: float
    elapsed_seconds_delta: float


# ---------------------------------------------------------------------------
# Text fragments shared by summary() (Markdown) and report() (HTML)
#
# Each of these returns plain text with `backtick` spans only -- no other
# markdown -- so the same string can be dropped into a Markdown bullet list
# as-is, or passed through report.py's _md_inline_to_html() for HTML.
# ---------------------------------------------------------------------------


def _models_str(cost: Any) -> str:
    """Distinct, sorted model names used across a run's steps (or '—')."""
    models = sorted({s.model for s in cost.steps.values() if s.model})
    return ", ".join(models) if models else "—"


def _format_header(label_a: str, cost_a: Any, label_b: str, cost_b: Any) -> str:
    return f"Comparing `{label_a}` ({_models_str(cost_a)}) vs. `{label_b}` ({_models_str(cost_b)})"


_ENUM_SHIFT_LIMIT = 5


def _format_enum_shift(fc: FieldChurn) -> str:
    """'<value>' <pctA>% → <pctB>%' for the values with the largest shift, biggest first."""
    freq_a, freq_b = fc.freq_a or {}, fc.freq_b or {}
    keys = sorted(set(freq_a) | set(freq_b))
    if not keys:
        return ""
    ranked = sorted(keys, key=lambda k: abs(freq_b.get(k, 0.0) - freq_a.get(k, 0.0)), reverse=True)
    shown = ranked[:_ENUM_SHIFT_LIMIT]
    parts = [
        f"{k!r} {freq_a.get(k, 0.0) * 100:.0f}% → {freq_b.get(k, 0.0) * 100:.0f}%" for k in shown
    ]
    if len(ranked) > _ENUM_SHIFT_LIMIT:
        parts.append("…")
    return ", ".join(parts)


def _format_field_churn_line(fc: FieldChurn) -> str:
    """One line per field for the 'Per-field churn' section, no leading bullet marker."""
    head = f"`{fc.field}`: {fc.changed} changed ({fc.changed_pct:.1f}%)"
    if fc.kind == "enum":
        shift = _format_enum_shift(fc)
        return f"{head} — distribution shifted: {shift}" if shift else head
    if fc.kind == "numeric":
        if fc.mean_delta is None:
            return head
        sign = "+" if fc.mean_delta >= 0 else ""
        return f"{head} — mean delta {sign}{fc.mean_delta:.2f}"
    # text (String, or differs-only for Boolean/Date/JSON/List[String])
    head = f"`{fc.field}`: {fc.changed} changed (text)"
    if fc.len_delta_tokens is None:
        return head
    sign = "+" if fc.len_delta_tokens >= 0 else ""
    return f"{head} — avg length {sign}{fc.len_delta_tokens:.0f} tokens"


def _format_cost_delta_lines(cd: CostDelta, label_b: str) -> list[str]:
    """Lines for the 'Cost delta' section: token deltas, cache hit rate, wall time."""

    def _pct(delta: int, base: int) -> str:
        if base == 0:
            return "n/a" if delta == 0 else "new"
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta / base * 100:.0f}%"

    return [
        f"`{label_b}` used {_pct(cd.prompt_tokens_delta, cd.prompt_tokens_a)} input tokens, "
        f"{_pct(cd.completion_tokens_delta, cd.completion_tokens_a)} output tokens "
        f"({_format_tokens(cd.total_tokens_a)} → {_format_tokens(cd.total_tokens_b)} total).",
        f"Cache hit rate: {cd.cache_hit_rate_a * 100:.0f}% → {cd.cache_hit_rate_b * 100:.0f}%.",
        f"Wall time: {_format_seconds(cd.elapsed_seconds_a)} → "
        f"{_format_seconds(cd.elapsed_seconds_b)}.",
    ]


# ---------------------------------------------------------------------------
# compare() + ComparisonResult
# ---------------------------------------------------------------------------


def _to_dataframe(data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


@dataclass
class ComparisonResult:
    """Diff between two :class:`~accrue.pipeline.pipeline.PipelineResult` runs.

    Built by :func:`compare`; not meant to be constructed directly.

    Attributes:
        result_a: The first (``label_a``) pipeline result.
        result_b: The second (``label_b``) pipeline result.
        label_a: Display label for ``result_a`` (default ``"A"``).
        label_b: Display label for ``result_b`` (default ``"B"``).
        fields: Output field names compared — present as a column in both
            ``result_a.data`` and ``result_b.data``, and declared in the
            field specs of at least one side.
        field_specs: Field-spec dict per compared field, used to classify
            it as enum / numeric / text. Prefers ``result_a``'s spec,
            falling back to ``result_b``'s.
        aligned_a: ``result_a``'s data, sliced to the aligned row set and
            reset to a plain ``0..n-1`` index. Row ``i`` pairs with
            ``aligned_b`` row ``i``.
        aligned_b: ``result_b``'s data, aligned the same way.
        align_mode: ``"positional"`` or ``"label"`` — how rows were paired
            (see :func:`_align_rows`).
        pos_a: Original integer position in ``result_a.data`` for each
            aligned row (lines up with ``RowError.row_index``).
        pos_b: Original integer position in ``result_b.data`` for each
            aligned row.
    """

    result_a: PipelineResult
    result_b: PipelineResult
    label_a: str
    label_b: str
    fields: list[str]
    field_specs: dict[str, dict[str, Any]]
    aligned_a: pd.DataFrame
    aligned_b: pd.DataFrame
    align_mode: Literal["positional", "label"]
    pos_a: list[int]
    pos_b: list[int]

    def changed_rows(self, field: str | None = None) -> pd.DataFrame:
        """Before/after values for aligned rows that changed.

        Args:
            field: If given, only consider this output field (must be in
                ``.fields``). If ``None`` (default), a row counts as
                changed if *any* compared field differs.

        Returns:
            A DataFrame indexed by aligned-row position, with two columns
            per considered field: ``"{field}_{label_a}"`` and
            ``"{field}_{label_b}"``. Empty if nothing changed.

        A row that errored on exactly one side counts as changed even if
        its (typically ``None``-filled) values happen to compare equal —
        an error means that side's output for the row isn't trustworthy,
        independent of which field you're filtering to.
        """
        if field is not None and field not in self.fields:
            raise KeyError(f"{field!r} is not a comparable output field; available: {self.fields}")
        target_fields = [field] if field is not None else self.fields

        error_pos_a = {e.row_index for e in self.result_a.errors}
        error_pos_b = {e.row_index for e in self.result_b.errors}

        col_a = {f: self.aligned_a[f].tolist() for f in target_fields}
        col_b = {f: self.aligned_b[f].tolist() for f in target_fields}

        changed_idx: list[int] = []
        for i in range(len(self.aligned_a)):
            errored_a = self.pos_a[i] in error_pos_a
            errored_b = self.pos_b[i] in error_pos_b
            if errored_a != errored_b:
                changed_idx.append(i)
                continue
            if any(not _cells_equal(col_a[f][i], col_b[f][i]) for f in target_fields):
                changed_idx.append(i)

        suffix_a, suffix_b = f"_{self.label_a}", f"_{self.label_b}"
        before = self.aligned_a.loc[changed_idx, target_fields].add_suffix(suffix_a)
        after = self.aligned_b.loc[changed_idx, target_fields].add_suffix(suffix_b)
        ordered_cols = [c for f in target_fields for c in (f + suffix_a, f + suffix_b)]
        return pd.concat([before, after], axis=1)[ordered_cols]

    def distribution_shift(self) -> dict[str, FieldChurn]:
        """Per-field churn, classified by each field's spec.

        - Enum fields (``spec["enum"]`` truthy): before/after frequency
          tables.
        - Numeric fields (``spec["type"] == "Number"``): mean/std and the
          mean delta, coercing with ``pd.to_numeric(errors="coerce")`` so
          stray non-numeric values don't raise.
        - Everything else (String/Boolean/Date/JSON/List[String]): a
          differs-count; String fields additionally get an approximate
          token-length delta (``len(str(v)) // 4``, matching ``report.py``'s
          length-anomaly heuristic).

        Returns:
            ``field name -> FieldChurn``, in the same order as ``.fields``.
        """
        result: dict[str, FieldChurn] = {}
        n = len(self.aligned_a)
        for f in self.fields:
            spec = self.field_specs.get(f, {})
            series_a, series_b = self.aligned_a[f], self.aligned_b[f]
            vals_a, vals_b = series_a.tolist(), series_b.tolist()
            changed = sum(1 for x, y in zip(vals_a, vals_b) if not _cells_equal(x, y))

            if spec.get("enum"):
                result[f] = FieldChurn(
                    field=f,
                    kind="enum",
                    changed=changed,
                    total=n,
                    freq_a=_frequency_table(series_a),
                    freq_b=_frequency_table(series_b),
                )
            elif spec.get("type") == "Number":
                num_a = pd.to_numeric(series_a, errors="coerce")
                num_b = pd.to_numeric(series_b, errors="coerce")
                mean_a = float(num_a.mean()) if num_a.notna().any() else None
                mean_b = float(num_b.mean()) if num_b.notna().any() else None
                std_a = float(num_a.std()) if num_a.notna().sum() > 1 else None
                std_b = float(num_b.std()) if num_b.notna().sum() > 1 else None
                mean_delta = mean_b - mean_a if mean_a is not None and mean_b is not None else None
                result[f] = FieldChurn(
                    field=f,
                    kind="numeric",
                    changed=changed,
                    total=n,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    std_a=std_a,
                    std_b=std_b,
                    mean_delta=mean_delta,
                )
            else:
                len_a = len_b = len_delta = None
                if spec.get("type", "String") == "String":
                    lens_a = [len(str(v)) // 4 for v in vals_a if not _is_null(v)]
                    lens_b = [len(str(v)) // 4 for v in vals_b if not _is_null(v)]
                    len_a = sum(lens_a) / len(lens_a) if lens_a else None
                    len_b = sum(lens_b) / len(lens_b) if lens_b else None
                    len_delta = len_b - len_a if len_a is not None and len_b is not None else None
                result[f] = FieldChurn(
                    field=f,
                    kind="text",
                    changed=changed,
                    total=n,
                    avg_len_tokens_a=len_a,
                    avg_len_tokens_b=len_b,
                    len_delta_tokens=len_delta,
                )
        return result

    def cost_delta(self) -> CostDelta:
        """Token, cache-hit-rate, and wall-time delta between the two runs.

        ``StepUsage.cache_hit_rate`` is a per-step property only -- there
        is no top-level aggregate on ``CostSummary``, so the overall rate
        for each side is computed here as
        ``sum(cache_hits) / sum(cache_hits + cache_misses)`` across that
        side's ``cost.steps`` (0.0 if neither ever ran, instead of
        dividing by zero).

        No ``$``/pricing fields -- see :class:`CostDelta`.
        """
        cost_a, cost_b = self.result_a.cost, self.result_b.cost

        def _cache_rate(cost: Any) -> float:
            hits = sum(s.cache_hits for s in cost.steps.values())
            total = sum(s.cache_hits + s.cache_misses for s in cost.steps.values())
            return hits / total if total > 0 else 0.0

        rate_a, rate_b = _cache_rate(cost_a), _cache_rate(cost_b)
        elapsed_a = self.result_a.pipeline_elapsed_seconds
        elapsed_b = self.result_b.pipeline_elapsed_seconds

        return CostDelta(
            prompt_tokens_a=cost_a.total_prompt_tokens,
            prompt_tokens_b=cost_b.total_prompt_tokens,
            prompt_tokens_delta=cost_b.total_prompt_tokens - cost_a.total_prompt_tokens,
            completion_tokens_a=cost_a.total_completion_tokens,
            completion_tokens_b=cost_b.total_completion_tokens,
            completion_tokens_delta=(
                cost_b.total_completion_tokens - cost_a.total_completion_tokens
            ),
            total_tokens_a=cost_a.total_tokens,
            total_tokens_b=cost_b.total_tokens,
            total_tokens_delta=cost_b.total_tokens - cost_a.total_tokens,
            cache_hit_rate_a=rate_a,
            cache_hit_rate_b=rate_b,
            cache_hit_rate_delta=rate_b - rate_a,
            elapsed_seconds_a=elapsed_a,
            elapsed_seconds_b=elapsed_b,
            elapsed_seconds_delta=elapsed_b - elapsed_a,
        )

    def _row_counts(self) -> tuple[int, int, int]:
        """``(total, changed, identical)`` aligned-row counts."""
        total = len(self.aligned_a)
        changed_n = len(self.changed_rows())
        return total, changed_n, total - changed_n

    def summary(self) -> str:
        """Markdown summary — headline numbers, per-field churn, cost delta.

        Mirrors the shape of ``PipelineResult.report()``: pasteable into
        Slack/GitHub. Identical runs short-circuit to a one-line "no
        differences" message instead of an empty churn/cost section.
        """
        total, changed_n, identical_n = self._row_counts()

        lines: list[str] = [
            "# Pipeline Comparison Report",
            "",
            _format_header(self.label_a, self.result_a.cost, self.label_b, self.result_b.cost),
            "",
            f"**{total:,} rows compared**",
            f"- {changed_n:,} rows changed in at least one field",
            f"- {identical_n:,} rows identical",
            "",
        ]

        if changed_n == 0:
            lines.append("No differences detected between the two runs.")
            return "\n".join(lines).rstrip() + "\n"

        if self.fields:
            lines.append("**Per-field churn:**")
            shifts = self.distribution_shift()
            for f in self.fields:
                lines.append(f"- {_format_field_churn_line(shifts[f])}")
            lines.append("")

        lines.append("**Cost delta:**")
        for line in _format_cost_delta_lines(self.cost_delta(), self.label_b):
            lines.append(f"- {line}")
        lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def report(self, output_format: str = "markdown", path: str | None = None) -> str:
        """Render the comparison — Markdown (mirrors :meth:`summary`) or self-contained HTML.

        Mirrors :meth:`PipelineResult.report`'s signature.

        Args:
            output_format: ``"markdown"`` (default) or ``"html"``.
            path: If given, also write the rendered report to this path
                (UTF-8).

        Returns:
            The rendered report as a string.

        Raises:
            ValueError: If ``output_format`` isn't ``"markdown"`` or ``"html"``.
        """
        fmt = output_format.lower()
        if fmt == "markdown":
            text = self.summary()
        elif fmt == "html":
            text = self._render_html()
        else:
            raise ValueError(
                f"Unknown report format {output_format!r}; expected 'markdown' or 'html'."
            )

        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def _render_html(self) -> str:
        """Self-contained HTML rendering of the same content as :meth:`summary`."""
        total, changed_n, identical_n = self._row_counts()

        header = _format_header(self.label_a, self.result_a.cost, self.label_b, self.result_b.cost)
        parts: list[str] = [
            "<!doctype html>",
            '<html lang="en"><head>',
            '<meta charset="utf-8">',
            "<title>Pipeline Comparison Report</title>",
            f"<style>{_HTML_STYLE}</style>",
            "</head><body>",
            "<h1>Pipeline Comparison Report</h1>",
            f'<p class="headline">{_md_inline_to_html(header)}</p>',
            f"<p><strong>{total:,} rows compared</strong> &middot; "
            f"{changed_n:,} changed &middot; {identical_n:,} identical</p>",
        ]

        if changed_n == 0:
            parts.append("<h2>No differences detected</h2>")
            parts.append("<p>The two runs produced identical output on every compared field.</p>")
            parts.append("</body></html>")
            return "\n".join(parts) + "\n"

        if self.fields:
            parts.append("<h2>Per-field churn</h2>")
            parts.append('<ul class="findings">')
            shifts = self.distribution_shift()
            for f in self.fields:
                line = _md_inline_to_html(_format_field_churn_line(shifts[f]))
                parts.append(f"<li>{line}</li>")
            parts.append("</ul>")

        parts.append("<h2>Cost delta</h2>")
        parts.append('<ul class="findings">')
        for line in _format_cost_delta_lines(self.cost_delta(), self.label_b):
            parts.append(f"<li>{_md_inline_to_html(line)}</li>")
        parts.append("</ul>")

        parts.append("</body></html>")
        return "\n".join(parts) + "\n"


def compare(
    a: PipelineResult,
    b: PipelineResult,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> ComparisonResult:
    """Diff two pipeline runs — the prompt-iteration loop in one call.

    ``result_a``/``result_b`` need not come from the same pipeline
    definition; only the overlap of their output fields is compared.

    Args:
        a: The "before" run.
        b: The "after" run.
        label_a: Display label for *a* in summaries/reports.
        label_b: Display label for *b* in summaries/reports.

    Returns:
        A :class:`ComparisonResult`. Call ``.summary()`` for a Markdown
        readout, ``.changed_rows()`` for the rows that differ, or
        ``.report()`` to render/save the full comparison.
    """
    df_a = _to_dataframe(a.data)
    df_b = _to_dataframe(b.data)

    pos_a, pos_b, align_mode = _align_rows(df_a, df_b)
    aligned_a = df_a.iloc[pos_a].reset_index(drop=True)
    aligned_b = df_b.iloc[pos_b].reset_index(drop=True)

    spec_keys_a = set(a.field_specs)
    spec_keys_b = set(b.field_specs)
    if spec_keys_a != spec_keys_b:
        only_a = sorted(spec_keys_a - spec_keys_b)
        only_b = sorted(spec_keys_b - spec_keys_a)
        warnings.warn(
            f"compare(): field-spec schemas differ between {label_a!r} and {label_b!r} "
            f"(only in {label_a}: {only_a or 'none'}; only in {label_b}: {only_b or 'none'}); "
            "comparing the overlapping output fields only.",
            stacklevel=2,
        )

    candidate_fields = spec_keys_a | spec_keys_b
    fields = sorted(
        f
        for f in candidate_fields
        if not f.startswith("__") and f in aligned_a.columns and f in aligned_b.columns
    )
    field_specs = {
        f: (a.field_specs[f] if f in a.field_specs else b.field_specs.get(f, {})) for f in fields
    }

    return ComparisonResult(
        result_a=a,
        result_b=b,
        label_a=label_a,
        label_b=label_b,
        fields=fields,
        field_specs=field_specs,
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        align_mode=align_mode,
        pos_a=pos_a,
        pos_b=pos_b,
    )

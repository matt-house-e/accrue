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
from typing import Any

import pandas as pd

from .pipeline import PipelineResult
from .report import _is_null

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


def _align_rows(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[list[int], list[int], str]:
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
    align_mode: str
    pos_a: list[int]
    pos_b: list[int]


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

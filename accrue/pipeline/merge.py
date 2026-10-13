"""``accrue.merge_runs()`` — combine N independent runs into one answer per cell.

Reading the same rows twice changes a share of the structured cells even at
temperature 0: the source is ambiguous, not the sampling. A single run is a
sample. ``merge_runs()`` takes several independent runs of the same pipeline
over the same rows and, per cell, votes on the answer and records how much
the runs agreed, so a person reviews the cells the runs could not settle
instead of every cell.

Row alignment, null handling and cell equality are shared with
:func:`accrue.compare` (:mod:`.compare`), so two runs that ``compare()``
calls identical merge as unanimous.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import pandas as pd

from ..schemas.base import CostSummary, StepUsage
from .compare import _align_rows, _cells_equal, _to_dataframe
from .pipeline import PipelineResult

__all__ = ["merge_runs", "MergedResult", "Candidate"]


@dataclass(frozen=True)
class Candidate:
    """One run's answer for one cell, as a resolver sees it.

    Attributes:
        run: Index of the run in the list passed to :func:`merge_runs`.
        value: The run's value for the field.
        row: The run's whole output row, so a resolver can read companion
            columns (a quote, a confidence, a source) without accrue
            knowing what they mean.
    """

    run: int
    value: Any
    row: dict[str, Any]


Resolver = Callable[[str, list[Candidate]], Union[Candidate, None]]
"""``resolver(field, candidates) -> Candidate | None``.

Called only when the runs disagree. Return the candidate to keep, or
``None`` to fall back to the field's default rule.
"""

Rule = Literal["majority", "first"]

Basis = Literal[
    "unanimous", "majority", "resolver", "first", "no_majority", "single_read", "no_reads"
]


def _default_rule(spec: dict[str, Any]) -> Rule:
    """``"first"`` for free text, ``"majority"`` for everything else.

    Free text is a declared ``String`` field without an ``enum``: two runs
    almost never word prose the same way, so voting on it would flag nearly
    every cell. A field with an empty spec (FunctionStep outputs, list-style
    LLMStep fields) has no declared type and is voted on.
    """
    if spec.get("type") == "String" and not spec.get("enum"):
        return "first"
    return "majority"


def _sum_costs(costs: Sequence[CostSummary]) -> CostSummary:
    """Token usage summed across runs, step by step."""
    steps: dict[str, StepUsage] = {}
    int_fields = [
        name
        for name, info in StepUsage.model_fields.items()
        if info.annotation is int  # prompt/completion/cache counters, rows
    ]
    for cost in costs:
        for name, usage in cost.steps.items():
            if name not in steps:
                steps[name] = usage.model_copy()
                continue
            acc = steps[name]
            for f in int_fields:
                setattr(acc, f, getattr(acc, f) + getattr(usage, f))
            acc.model = acc.model or usage.model
            # The tokens cover every run, so the batch ids must too; batch_id
            # stays the first of them.  Rebind, never extend: model_copy() is
            # shallow, so the list is still run 0's own.
            acc.batch_id = acc.batch_id or usage.batch_id
            acc.batch_ids = [*acc.batch_ids, *usage.batch_ids]
    return CostSummary(
        total_prompt_tokens=sum(c.total_prompt_tokens for c in costs),
        total_completion_tokens=sum(c.total_completion_tokens for c in costs),
        total_tokens=sum(c.total_tokens for c in costs),
        total_cache_write_tokens=sum(c.total_cache_write_tokens for c in costs),
        total_cache_read_tokens=sum(c.total_cache_read_tokens for c in costs),
        steps=steps,
    )


def _fully_cached(result: PipelineResult) -> bool:
    """True if every step that ran was served entirely from accrue's cache."""
    steps = result.cost.steps.values()
    return any(s.cache_hits for s in steps) and not any(s.cache_misses for s in steps)


@dataclass
class MergedResult:
    """N runs merged into one answer per cell. Built by :func:`merge_runs`.

    Attributes:
        data: The merged rows: run 0's rows (aligned rows only) with every
            merged field replaced by the merged value. Columns that were not
            merged, inputs and anything outside ``fields``, come from run 0.
        agreement: One row per merged cell: ``row`` (run 0's index label),
            ``field``, ``value`` (the merged value), ``votes`` (runs whose
            answer matches it), ``voters`` (runs that produced the row
            without error), ``agreed`` (every voter gave the same answer),
            ``needs_review``, ``basis`` (how the value was chosen: one of
            ``unanimous``, ``majority``, ``resolver``, ``first``,
            ``no_majority``, ``single_read``, ``no_reads``) and ``values``
            (a tuple of each run's raw answer; ``None`` where the row errored).
        errored_rows: Row label -> indexes of the runs in which that row
            errored. Those runs do not vote on the row.
        cost: Token usage summed across all runs.
        elapsed_seconds: Wall time summed across all runs.
        fields: The merged field names.
        rules: Field -> how disagreements are settled: ``"majority"``,
            ``"first"`` or ``"resolver"`` (a resolver, falling back to the
            field's default rule when it declines).
        runs: The input runs, in order.
    """

    data: pd.DataFrame
    agreement: pd.DataFrame
    errored_rows: dict[Any, list[int]]
    cost: CostSummary
    elapsed_seconds: float
    fields: list[str]
    rules: dict[str, str]
    runs: list[PipelineResult] = field(repr=False)

    @property
    def needs_review(self) -> pd.DataFrame:
        """The rows of :attr:`agreement` a person should check."""
        return self.agreement[self.agreement["needs_review"]].reset_index(drop=True)

    def summary(self) -> str:
        """Markdown: per-field agreement and the number of cells needing review."""
        a = self.agreement
        rows = a["row"].nunique() if len(a) else len(self.data)
        lines = [
            "# Merged runs",
            "",
            f"**{len(self.runs)} runs, {rows:,} rows, {len(a):,} cells**",
            f"- {int(a['agreed'].sum()):,} cells unanimous",
            f"- {int(a['needs_review'].sum()):,} cells need review",
            f"- {len(self.errored_rows):,} rows errored in at least one run",
            "",
        ]
        if len(a):
            lines += [
                "| Field | Rule | Unanimous | Needs review |",
                "|---|---|---|---|",
            ]
            for f, g in a.groupby("field", sort=False):
                rule = self.rules.get(f, "")
                lines.append(
                    f"| `{f}` | {rule} | {g['agreed'].mean() * 100:.0f}% "
                    f"| {int(g['needs_review'].sum())} |"
                )
            lines.append("")
        return "\n".join(lines)


def _per_field(arg: Any, fields: list[str], name: str) -> dict[str, Any]:
    """Expand a callable-or-mapping argument into ``field -> value``."""
    if arg is None:
        return {}
    if isinstance(arg, Mapping):
        unknown = sorted(set(arg) - set(fields))
        if unknown:
            raise ValueError(
                f"merge_runs(): {name}= names field(s) that are not merged: {unknown}; "
                f"merged fields: {fields}"
            )
        return dict(arg)
    return {f: arg for f in fields}


def merge_runs(
    runs: Sequence[PipelineResult],
    *,
    fields: Sequence[str] | None = None,
    key: Callable[[Any], Any] | Mapping[str, Callable[[Any], Any]] | None = None,
    resolve: Resolver | Mapping[str, Resolver | Rule] | Rule | None = None,
) -> MergedResult:
    """Combine several independent runs over the same rows into one answer per cell.

    Per cell, each run whose row did not error casts one vote. If every vote
    agrees the cell is ``unanimous``. Otherwise the field's resolver (if any)
    is asked; if it declines, the field's rule decides:

    - ``"majority"`` (default for enums, numbers, booleans, dates, JSON,
      lists and fields with no spec): a strict majority of voters wins;
      without one the earliest of the largest groups is kept and the cell is
      flagged ``needs_review``.
    - ``"first"`` (default for free-text ``String`` fields): run 0's answer
      (the earliest run that read the row) is kept and the cell is not
      flagged. Agreement is still recorded.

    A cell read by fewer than two runs (the row errored elsewhere) is flagged
    ``single_read``; one no run read is ``no_reads``.

    Make the runs independent before merging: with the same config, run 2
    replays run 1 from accrue's cache. Give each run its own ``cache_dir``
    (see ``docs/guides/merge-runs.md``).

    Args:
        runs: Two or more results of the same pipeline over the same rows.
            Rows are aligned to run 0 exactly as :func:`accrue.compare`
            aligns two runs; rows missing from any run are left out.
        fields: Fields to merge. Default: every output field (from the
            runs' field specs) present in all runs.
        key: Normalise a value before comparing it, e.g. parse a date or
            lower-case a name. One callable for every field, or a mapping
            ``field -> callable``. The merged value is the raw value from the
            earliest run in the winning group.
        resolve: Settle disagreements. A resolver
            ``(field, candidates) -> Candidate | None`` for every field, a
            rule name (``"majority"`` / ``"first"``) for every field, or a
            mapping ``field -> resolver | rule name``.

    Returns:
        A :class:`MergedResult`.

    Raises:
        ValueError: Fewer than two runs, or ``key=`` / ``resolve=`` name a
            field that is not merged.
        KeyError: A field in ``fields=`` is missing from a run.
        TypeError: A resolver returned something other than one of the
            candidates it was given, or ``None``.
    """
    runs = list(runs)
    if len(runs) < 2:
        raise ValueError(f"merge_runs() needs at least two runs, got {len(runs)}")

    frames = [_to_dataframe(r.data) for r in runs]
    base = frames[0]

    # Align every run to run 0; keep the rows present in all of them.
    maps: list[dict[int, int]] = [{p: p for p in range(len(base))}]
    for df in frames[1:]:
        pos0, posi, _mode = _align_rows(base, df, caller="merge_runs")
        maps.append(dict(zip(pos0, posi)))
    common = [p for p in range(len(base)) if all(p in m for m in maps)]

    specs: dict[str, dict[str, Any]] = {}
    for r in runs:
        for f, s in r.field_specs.items():
            specs.setdefault(f, s)
    if fields is None:
        fields = sorted(
            f for f in specs if not f.startswith("__") and all(f in df.columns for df in frames)
        )
    else:
        fields = list(fields)
        for i, df in enumerate(frames):
            missing = [f for f in fields if f not in df.columns]
            if missing:
                raise KeyError(f"merge_runs(): run {i} has no column(s) {missing}")

    keys = _per_field(key, fields, "key")
    if isinstance(resolve, str):
        resolve = {f: resolve for f in fields}
    resolvers = _per_field(resolve, fields, "resolve")
    for f, r in resolvers.items():
        if isinstance(r, str) and r not in ("majority", "first"):
            raise ValueError(f"merge_runs(): unknown rule {r!r} for {f!r}")

    errored = [{e.row_index for e in r.errors} for r in runs]
    merged = base.iloc[common].copy()
    columns: dict[str, list[Any]] = {f: [] for f in fields}
    records: list[dict[str, Any]] = []
    errored_rows: dict[Any, list[int]] = {}
    # (i, j) pairs of runs that differed on at least one cell.
    differed: set[tuple[int, int]] = set()

    for p0 in common:
        label = base.index[p0]
        positions = [m[p0] for m in maps]
        bad = [i for i, pos in enumerate(positions) if pos in errored[i]]
        if bad:
            errored_rows[label] = bad
        rows = [
            None if i in bad else frames[i].iloc[pos].to_dict() for i, pos in enumerate(positions)
        ]
        for f in fields:
            values = tuple(None if row is None else row[f] for row in rows)
            k = keys.get(f) or (lambda v: v)
            keyed = [None if row is None else k(row[f]) for row in rows]
            for i in range(len(runs)):
                for j in range(i):
                    if (
                        rows[i] is not None
                        and rows[j] is not None
                        and not _cells_equal(keyed[i], keyed[j])
                    ):
                        differed.add((j, i))

            candidates = [
                Candidate(run=i, value=row[f], row=row)
                for i, row in enumerate(rows)
                if row is not None
            ]
            # Group candidates by keyed equality, in run order.
            groups: list[list[Candidate]] = []
            for c in candidates:
                for g in groups:
                    if _cells_equal(keyed[c.run], keyed[g[0].run]):
                        g.append(c)
                        break
                else:
                    groups.append([c])

            voters = len(candidates)
            rule = resolvers.get(f)
            default = _default_rule(specs.get(f, {}))
            chosen: list[Candidate] | None = None
            basis: Basis
            needs_review = False
            if voters == 0:
                basis, needs_review, value = "no_reads", True, None
            elif len(groups) == 1:
                chosen, basis = groups[0], "unanimous"
                needs_review = voters < 2
                if needs_review:
                    basis = "single_read"
            else:
                pick = rule(f, candidates) if callable(rule) else None
                if pick is not None:
                    if not any(pick is c for c in candidates):
                        raise TypeError(
                            f"merge_runs(): resolver for {f!r} must return one of the "
                            f"candidates it was given, or None; got {pick!r}"
                        )
                    chosen = next(g for g in groups if any(pick is c for c in g))
                    chosen = [pick] + [c for c in chosen if c is not pick]
                    basis = "resolver"
                else:
                    policy = rule if isinstance(rule, str) else default
                    if policy == "first":
                        chosen, basis = groups[0], "first"
                    else:
                        largest = max(groups, key=len)  # earliest on ties
                        chosen = largest
                        if len(largest) * 2 > voters:
                            basis = "majority"
                        else:
                            basis, needs_review = "no_majority", True
            if chosen is not None:
                value = chosen[0].value
            votes = len(chosen) if chosen else 0
            columns[f].append(value)
            records.append(
                {
                    "row": label,
                    "field": f,
                    "value": value,
                    "votes": votes,
                    "voters": voters,
                    "agreed": voters > 0 and votes == voters,
                    "needs_review": needs_review,
                    "basis": basis,
                    "values": values,
                }
            )

    for i in range(1, len(runs)):
        if not _fully_cached(runs[i]):
            continue
        for j in range(i):
            if (j, i) not in differed:
                warnings.warn(
                    f"merge_runs(): run {i} was served entirely from cache and matches "
                    f"run {j} on every cell; it is probably a replay of run {j}, not an "
                    "independent read. Give each run its own cache_dir.",
                    stacklevel=2,
                )
                break

    for f in fields:
        merged[f] = columns[f]

    agreement = pd.DataFrame(
        records,
        columns=[
            "row",
            "field",
            "value",
            "votes",
            "voters",
            "agreed",
            "needs_review",
            "basis",
            "values",
        ],
    )
    return MergedResult(
        data=merged,
        agreement=agreement,
        errored_rows=errored_rows,
        cost=_sum_costs([r.cost for r in runs]),
        elapsed_seconds=sum(r.pipeline_elapsed_seconds for r in runs),
        fields=fields,
        rules={
            f: "resolver"
            if callable(resolvers.get(f))
            else (resolvers.get(f) or _default_rule(specs.get(f, {})))
            for f in fields
        },
        runs=runs,
    )

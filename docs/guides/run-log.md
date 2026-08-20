# Run Log (contract v1)

A run log is an append-only JSONL file describing one pipeline run, one JSON object per line. It is the wire contract between accrue and anything that watches a run — dashboards, `tail -f`, log shippers, accrue-ui. This document is the v1 specification; the golden fixture consumers test against lives at `tests/fixtures/run_small.jsonl`.

## Enabling it

```python
result = pipeline.run(data, run_log=True)
# -> .accrue/runs/2026-08-19-143512.jsonl  (relative to the CWD)

result = pipeline.run(data, run_log="logs/tonight.jsonl")   # explicit path
result = pipeline.run(data, run_log=True, display_key="company_name")
```

- `run_log=True` writes to `.accrue/runs/<run_id>.jsonl` under the current working directory. `run_id` is a local timestamp, `YYYY-MM-DD-HHMMSS`.
- `run_log=<str | Path>` writes to exactly that file. Parent directories are created.
- `display_key` names the column a UI should use to label rows. If omitted, it defaults to the first column with string/object dtype, else the first column, else `null`.
- The returned `PipelineResult` is unchanged; the log is a side channel.
- User `hooks=` still fire — the log's hooks are merged with yours, not substituted.

Both `run()` and `run_async()` accept the same keywords. `retry_failed()` / `retry_failed_async()` do too — pointed at an existing log, a retry appends to it under the same `run_id` (see [`retry_start` / `retry_end`](#retry_start--retry_end--a-failed-only-retry-appended-to-the-run)).

### Standalone logger

`run_log=True` is sugar for wiring up a `JsonlRunLogger`, which is a plain `EnrichmentHooks` consumer you can also drive yourself:

```python
from accrue import JsonlRunLogger

logger = JsonlRunLogger("my_run.jsonl", pipeline=pipeline, display_key="company", data=data)
result = pipeline.run(data, hooks=logger.hooks)
```

Pass `pipeline=` so the `pipeline_start` record can carry per-step level/mode/model; without it those entries are `null`. Pass `data=` so each `row_complete` can carry the row's display value as `key`; without it `key` is `null` (the log holds only step outputs, so the value can't be recovered later).

## Envelope

Every line is a JSON object carrying the same three envelope fields, followed by the record's own fields:

```json
{"v": 1, "t": 0.0142, "type": "row_complete", ...}
```

| Field | Type | Description |
|-------|------|-------------|
| `v` | `int` | Schema version. `1` for everything in this document. |
| `t` | `float` | Seconds since `pipeline_start`, measured on a monotonic clock. Non-decreasing across the file. In an appended retry segment it continues from the file's last `t` -- it is a within-file ordering clock, not wall-clock elapsed. |
| `type` | `string` | One of `pipeline_start`, `step_start`, `row_complete`, `step_end`, `pipeline_end`, `retry_start`, `retry_end`. |

## Record types

### `pipeline_start` — once, always the first line

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `string` | Local-timestamp id, `YYYY-MM-DD-HHMMSS`. Matches the default filename. |
| `started_at` | `string` | Wall-clock start, ISO-8601 UTC. |
| `num_rows` | `int` | Rows in the input data. |
| `display_key` | `string \| null` | Label column for UIs (see above). |
| `steps` | `array` | One `{name, level, mode, model}` per step, in execution order. `level` is the 0-based DAG level; `mode` is `"realtime"` or `"batch"`; `model` is the step's model id when it exposes one (LLM steps), else `null`. |
| `plan` | `object \| null` | Reserved for a `Pipeline.plan()` snapshot. Always `null` in current emitters. |

### `step_start` — per step, before its rows

| Field | Type | Description |
|-------|------|-------------|
| `step` | `string` | Step name. |
| `level` | `int` | 0-based DAG level. |
| `mode` | `string` | `"realtime"` or `"batch"`. |
| `num_rows` | `int` | Rows this step will process. |

### `row_complete` — per row, per step

| Field | Type | Description |
|-------|------|-------------|
| `step` | `string` | Step name. |
| `row` | `int` | 0-based row index. |
| `key` | `string \| null` | The row's display value: `str()` of the row's `display_key` column in the **input** data, resolved once per row. `null` when it can't be resolved — no display key, column absent, or a null value. Additive within v1: the log otherwise carries only step *outputs*, so without this field row labels degrade to "row *n*". |
| `status` | `string` | `"ok"`, `"error"`, or `"skipped"` (`run_if`/`skip_if`). |
| `from_cache` | `bool` | Result served from the SQLite cache. |
| `values` | `object \| null` | The row's output values, internal `__`-prefixed keys included — consumers filter those; the log stays complete. Non-JSON values (datetimes, etc.) degrade to strings. `null` when the row errored or produced no values; skipped rows carry their skip-default values. |
| `error` | `object \| null` | `{type, msg}` — exception class name and message. `null` unless `status` is `"error"`. |
| `usage` | `object \| null` | `{in, out, cost}` token usage for this row. `null` when unavailable: function steps, cache hits, skipped rows, and batch mode. |
| `elapsed_ms` | `float \| null` | Wall-clock milliseconds for this row (excludes queue/semaphore wait). `null` in batch mode. |

### `step_end` — per step, after all its rows

| Field | Type | Description |
|-------|------|-------------|
| `step` | `string` | Step name. |
| `num_errors` | `int` | Row errors in this step. |
| `usage` | `object` | `{in, out, cost}` aggregated over the step's rows. |
| `elapsed_s` | `float` | Wall-clock seconds for the step. |
| `batch_id` | `string \| null` | Provider batch job id (batch mode only). |

### `pipeline_end` — once, always the last line

Fires even when the run errors.

| Field | Type | Description |
|-------|------|-------------|
| `num_rows` | `int` | Rows processed. |
| `total_errors` | `int` | Row errors across all steps. Equals the sum of `step_end.num_errors`. |
| `cost` | `object` | `{in, out, cost}` aggregated over the whole run. |
| `elapsed_s` | `float` | Wall-clock seconds for the run. |

### `retry_start` / `retry_end` — a failed-only retry appended to the run

`pipeline.retry_failed(data, config=config, run_log=<the run's log>)` re-runs only the cells that errored and appends its events to the same file, keeping the original `run_id` and `display_key`. The segment is framed by `retry_start` … `retry_end` rather than a second `pipeline_start` / `pipeline_end`, so a consumer that only knows the records above ignores the frame (per the versioning policy) and still applies the recovered rows. Between the frame, `step_start` / `row_complete` / `step_end` are exactly as documented — only the retried steps report, and only for the retried rows.

`retry_start`:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `string` | The **original** run's id, unchanged. |
| `started_at` | `string` | Wall-clock start of the retry, ISO-8601 UTC. |
| `num_rows` | `int` | Rows in the input data (unchanged from the run). |
| `num_cells` | `int` | `(step, row)` cells this retry will re-execute. |
| `cells` | `array` | One `{step, row}` per cell, so a UI can mark them in flight. |

`retry_end` carries `pipeline_end`'s fields — `num_rows`, `total_errors`, `cost`, `elapsed_s` — plus `num_cells`. `total_errors` counts only the cells this retry executed; a cell that fails again appears as an ordinary `row_complete` with `status: "error"`.

```json
{"v": 1, "t": 0.011385, "type": "pipeline_end", "num_rows": 12, "total_errors": 1, "cost": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.011243}
{"v": 1, "t": 0.011385, "type": "retry_start", "run_id": "2026-08-20-080226", "started_at": "2026-08-20T07:14:02.881904+00:00", "num_rows": 12, "num_cells": 1, "cells": [{"step": "score", "row": 7}]}
{"v": 1, "t": 0.011502, "type": "step_start", "step": "score", "level": 1, "mode": "realtime", "num_rows": 1}
{"v": 1, "t": 0.012219, "type": "row_complete", "step": "score", "row": 7, "key": "company-07", "status": "ok", "from_cache": false, "values": {"score": 70}, "error": null, "usage": null, "elapsed_ms": 0.281}
{"v": 1, "t": 0.012884, "type": "step_end", "step": "score", "num_errors": 0, "usage": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.001204, "batch_id": null}
{"v": 1, "t": 0.013001, "type": "retry_end", "num_rows": 12, "total_errors": 0, "cost": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.001616, "num_cells": 1}
```

See the [checkpointing guide](caching.md#retrying-failed-cells) for what `retry_failed()` re-runs and what it reuses.

### Usage objects and `cost`

Everywhere a `{in, out, cost}` object appears: `in` is prompt tokens, `out` is completion tokens, and `cost` is reserved for dollar cost. Accrue deliberately ships no model pricing data (see `compare.py`), so **current emitters always write `cost: null`** — consumers price the tokens themselves, and must tolerate a number appearing here from a future emitter.

## Example

A 12-row, 3-step run — one row errors in `score`, one is skipped in `flag` (abridged from `tests/fixtures/run_small.jsonl`):

```json
{"v": 1, "t": 0.0, "type": "pipeline_start", "run_id": "2026-08-20-080226", "started_at": "2026-08-20T07:02:26.020167+00:00", "num_rows": 12, "display_key": "company", "steps": [{"name": "normalize", "level": 0, "mode": "realtime", "model": null}, {"name": "score", "level": 1, "mode": "realtime", "model": null}, {"name": "flag", "level": 2, "mode": "realtime", "model": null}], "plan": null}
{"v": 1, "t": 0.00243, "type": "step_start", "step": "normalize", "level": 0, "mode": "realtime", "num_rows": 12}
{"v": 1, "t": 0.003022, "type": "row_complete", "step": "normalize", "row": 0, "key": "company-00", "status": "ok", "from_cache": false, "values": {"name_upper": "COMPANY-00"}, "error": null, "usage": null, "elapsed_ms": 0.324}
{"v": 1, "t": 0.005766, "type": "step_end", "step": "normalize", "num_errors": 0, "usage": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.003241, "batch_id": null}
{"v": 1, "t": 0.005797, "type": "step_start", "step": "score", "level": 1, "mode": "realtime", "num_rows": 12}
{"v": 1, "t": 0.007665, "type": "row_complete", "step": "score", "row": 7, "key": "company-07", "status": "error", "from_cache": false, "values": null, "error": {"type": "ValueError", "msg": "cannot score company-07"}, "usage": null, "elapsed_ms": 0.214}
{"v": 1, "t": 0.008748, "type": "step_end", "step": "score", "num_errors": 1, "usage": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.002896, "batch_id": null}
{"v": 1, "t": 0.008775, "type": "step_start", "step": "flag", "level": 2, "mode": "realtime", "num_rows": 12}
{"v": 1, "t": 0.00968, "type": "row_complete", "step": "flag", "row": 3, "key": "company-03", "status": "skipped", "from_cache": false, "values": {"flagged": null}, "error": null, "usage": null, "elapsed_ms": 0.007}
{"v": 1, "t": 0.011343, "type": "step_end", "step": "flag", "num_errors": 0, "usage": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.002512, "batch_id": null}
{"v": 1, "t": 0.011385, "type": "pipeline_end", "num_rows": 12, "total_errors": 1, "cost": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.011243}
```

An LLM row looks like:

```json
{"v": 1, "t": 3.412, "type": "row_complete", "step": "analyze", "row": 4, "key": "Initech", "status": "ok", "from_cache": false, "values": {"market_size": 12.5, "risk_level": "Low"}, "error": null, "usage": {"in": 512, "out": 64, "cost": null}, "elapsed_ms": 1841.2}
```

## Guarantees

- **Append-only.** The file is opened in append mode and never truncated; pointing `run_log` at an existing file adds to it.
- **Flush per line.** Every record is flushed as soon as it is written. A crashed or killed run leaves a valid prefix of the stream — the tail is always complete lines, so `tail -f` and crash-recovery readers work.
- **Ordering.** `pipeline_start` is first; `pipeline_end` closes the run and fires even on error. Each step's `step_start` precedes all of its `row_complete` records, which precede its `step_end`. Steps at the same DAG level run in parallel, so *their* records may interleave; `t` stays non-decreasing throughout.
- **Segments.** A file holds one run, optionally followed by one `retry_start` … `retry_end` segment per `retry_failed()` call, each carrying the run's original `run_id`. Nothing follows a `pipeline_end` except retry segments.
- **Completeness.** Every row of every executed step gets exactly one `row_complete` — including cached, skipped, and errored rows. In a retry segment, "executed" means the retried cells: a row served from the checkpoint produces no record, and its state is whatever the earlier segment last said.
- **Never crashes the run.** The logger is an ordinary hooks consumer; hook errors are caught and logged, and non-JSON values degrade to strings rather than raise.

## Versioning policy

- `v` is bumped **only for breaking changes** (removing/renaming a field, changing a field's meaning or type).
- Within v1, changes are **additive only**: new record types or new fields may appear. Consumers MUST ignore unknown fields and unknown record types.
- Fields documented as nullable may start carrying values in newer emitters (e.g. `plan`, per-row `usage` in batch mode, `cost` in dollars). Consumers MUST NOT treat `null` as a guarantee.

## Where logs live

`run_log=True` writes to `.accrue/runs/` relative to the process CWD — alongside accrue's SQLite cache default (`.accrue/`). One file per run, named by `run_id`. Accrue never rotates or deletes run logs; clean the directory yourself if it grows.

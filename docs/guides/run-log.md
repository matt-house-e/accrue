# Run Log (contract v1)

A run log is an append-only JSONL file describing one pipeline run, one JSON object per line. It is the wire contract between accrue and anything that watches a run — dashboards, `tail -f`, log shippers, accrue-ui. This document is the v1 specification; the golden fixture consumers test against lives at `tests/fixtures/run_small.jsonl`.

## Enabling it

```python
result = pipeline.run(data, run_log=True)
# -> .accrue/runs/2026-08-19-143512-4f1c9a.jsonl  (relative to the CWD)

result = pipeline.run(data, run_log="logs/tonight.jsonl")   # explicit path
result = pipeline.run(data, run_log=True, display_key="company_name")
```

- `run_log=True` writes to `.accrue/runs/<run_id>.jsonl` under the current working directory. `run_id` is a UTC timestamp plus a short random suffix, `YYYY-MM-DD-HHMMSS-xxxxxx` — treat it as opaque. (The suffix is what keeps two runs started in the same second from sharing one id, and one file.)
- `run_log=<str | Path>` writes to exactly that file. Parent directories are created. Log files are created mode `0600`: they carry your row values and error text.
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
| `type` | `string` | One of `pipeline_start`, `step_start`, `row_attempt`, `row_complete`, `step_end`, `pipeline_end`, `retry_start`, `retry_end`. |

## Record types

### `pipeline_start` — once, always the first line

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `string` | Opaque run id, `YYYY-MM-DD-HHMMSS-xxxxxx` (UTC timestamp + random suffix). Matches the default filename. |
| `started_at` | `string` | Wall-clock start, ISO-8601 UTC. |
| `num_rows` | `int` | Rows in the input data. |
| `display_key` | `string \| null` | Label column for UIs (see above). |
| `steps` | `array` | One `{name, level, mode, model}` per step, in execution order. `level` is the 0-based DAG level; `mode` is `"realtime"` or `"batch"`; `model` is the step's model id when it exposes one (LLM steps), else `null`. |
| `manifest` | `object \| null` | The run's **definition** — see [Manifest](#manifest) below. `null` only if introspection failed. |
| `plan` | `object \| null` | Reserved for a `Pipeline.plan()` snapshot. Always `null` in current emitters. |

#### Manifest

The `manifest` object (issue #138) describes *what the pipeline is*, for a dashboard's read-only Overview: the steps and their types, the model and params per step, the enrichment-field schema, and the run config. It is introspected **once at run start** from the real `Pipeline` / `Step` / response-model objects — row-independent, deterministic (no timestamps or rng), and additive to schema v1. It never touches the cached `system` prompt half, so it cannot affect provider prompt caching (#107).

```jsonc
"manifest": {
  "accrue_version": "1.3.0",
  "config": { "max_workers": 6, "caching": false, "checkpointing": true, "batch": false, "capture": "prompts" },
  "steps": [
    {
      "name": "classify",
      "type": "LLMStep",              // type(step).__name__ — LLMStep / FunctionStep / …
      "model": { "id": "gpt-4.1-mini", "provider": "openai", "temperature": 0.0, "max_tokens": 512 },
      "produces": ["category", "icp_fit"],
      "depends_on": [],
      "condition": null               // a literal condition string when a step exposes one, else null
    }
    // FunctionStep: "model": null, "type": "FunctionStep"; still lists produces/depends_on
  ],
  "fields": [
    { "name": "category", "type": "str",  "enum": null,                     "description": "One short industry category.", "step": "classify", "internal": false },
    { "name": "icp_fit",  "type": "enum", "enum": ["strong","good","weak"], "description": "Fit as a customer.",           "step": "classify", "internal": false }
    // internal `__`-prefixed inter-step fields: "internal": true
  ]
}
```

How each part is introspected:

- **`config`** mirrors the run's `EnrichmentConfig` (`max_workers`, `caching`, `checkpointing`) plus the `capture` tier; `batch` is `true` when any step opts into the batch API (there is no config-level batch flag).
- **`steps[].type`** is `type(step).__name__`. **`steps[].model`** carries the LLM step's model `id`, the `provider` (from `base_url`'s host, e.g. `openrouter.ai` → `"openrouter"`, else the model-name prefix: `claude-*` → `anthropic`, `gemini-*` → `google`, else `openai`), and the **effective** `temperature` / `max_tokens` the runtime would send (the step's value, or the config fallback). FunctionSteps and any step with no model report `"model": null`.
- **`steps[].condition`** is a literal expression string only when a step exposes one; `run_if` / `skip_if` predicates are lambdas and can't be introspected to a string, so predicate-gated steps report `null`.
- **`fields[].type`** / `enum` / `description` come from introspecting each step's response model. Inline `FieldSpec`s map `String`/`Date` → `str`, `Number` → `float`, `Boolean` → `bool`, `List[String]` → `list`, `JSON` → `json`, and any `enum` → `type:"enum"` with its members; a custom Pydantic `schema=` is read by field annotation (`int`/`float`/`bool`/`str`, `Literal`/`Enum` → `enum`, `list`/`dict`). Fields with no introspectable source — FunctionStep outputs, `__` inter-step fields, or LLM `list` fields on the permissive default schema — report `type:"unknown"` rather than a fabricated value. `internal` is `true` for `__`-prefixed fields.

### `step_start` — per step, before its rows

| Field | Type | Description |
|-------|------|-------------|
| `step` | `string` | Step name. |
| `level` | `int` | 0-based DAG level. |
| `mode` | `string` | `"realtime"` or `"batch"`. |
| `num_rows` | `int` | Rows this step will process. |

### `row_attempt` — per LLM provider attempt (additive, v1)

Emitted once per provider call an LLM step makes, **before** the row's single `row_complete`: a cell that retries yields several `row_attempt` records then one `row_complete`; a cell that succeeds first try yields one `row_attempt` then its `row_complete`. Only LLM steps emit it — function steps have no provider attempts. Additive within v1 (issue #134): a consumer that only knows `row_complete` ignores these lines and still applies the row.

| Field | Type | Description |
|-------|------|-------------|
| `step` | `string` | Step name. |
| `row` | `int` | 0-based row index. |
| `attempt` | `int` | 1-based attempt number, counted across both of the step's retry loops (API-error and parse/validation). |
| `kind` | `string` | `"api"` — the provider call raised (rate limit, timeout, 5xx); or `"parse"` — the call returned and was parsed/validated (whether or not that parse succeeded). |
| `status` | `string` | Short outcome: `"ok"`, `"rate_limited"`, `"timeout"`, `"api_error"`, `"parse_error"`, `"validation_error"`. |
| `latency_ms` | `float \| null` | Wall-clock milliseconds for this attempt's provider call. |
| `backoff_s` | `float \| null` | Seconds slept before the *next* attempt (set on a retrying `api` attempt), else `null`. |
| `error` | `object \| null` | `{type, msg}` — exception class and message, secret patterns redacted as `***REDACTED***`. `null` on a successful attempt. |
| `prompt_ref` | `object \| null` | `{off, len}` byte offset + length into the [prompt sidecar](#prompt-sidecar) at `capture >= "prompts"`, else `null`. Bodies are never inlined into the main log. |

A cell that is rate-limited once, then parses cleanly:

```json
{"v": 1, "t": 0.481, "type": "row_attempt", "step": "classify", "row": 1, "attempt": 1, "kind": "api", "status": "rate_limited", "latency_ms": 12.4, "backoff_s": 0.83, "error": {"type": "LLMAPIError", "msg": "rate limited"}, "prompt_ref": null}
{"v": 1, "t": 1.402, "type": "row_attempt", "step": "classify", "row": 1, "attempt": 2, "kind": "parse", "status": "ok", "latency_ms": 640.2, "backoff_s": null, "error": null, "prompt_ref": {"off": 1276, "len": 1283}}
{"v": 1, "t": 1.404, "type": "row_complete", "step": "classify", "row": 1, "key": "Globex", "status": "ok", "from_cache": false, "values": {"grade": "B"}, "error": null, "usage": {"in": 11, "out": 3, "cost": null}, "elapsed_ms": 652.8}
```

### `row_complete` — per row, per step

| Field | Type | Description |
|-------|------|-------------|
| `step` | `string` | Step name. |
| `row` | `int` | 0-based row index. |
| `key` | `string \| null` | The row's display value: `str()` of the row's `display_key` column in the **input** data, resolved once per row. `null` when it can't be resolved — no display key, column absent, or a null value. Additive within v1: the log otherwise carries only step *outputs*, so without this field row labels degrade to "row *n*". |
| `status` | `string` | `"ok"`, `"error"`, or `"skipped"` (`run_if`/`skip_if`). |
| `from_cache` | `bool` | Result served from the SQLite cache. |
| `values` | `object \| null` | The row's output values, internal `__`-prefixed keys included — consumers filter those; the log stays complete. Non-JSON values (datetimes, etc.) degrade to strings; `NaN` / `±inf` become `null` (a bare `NaN` literal is not JSON). A value that cannot be encoded at all — non-string dict keys, a reference cycle — is replaced wholesale by `{"__unserializable__": true}` so the record is still emitted. `null` when the row errored or produced no values; skipped rows carry their skip-default values. |
| `error` | `object \| null` | `{type, msg}` — exception class name and message, with known secret patterns redacted as `***REDACTED***`. `null` unless `status` is `"error"`. |
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
{"v": 1, "t": 0.011385, "type": "retry_start", "run_id": "2026-08-20-080226-4f1c9a", "started_at": "2026-08-20T07:14:02.881904+00:00", "num_rows": 12, "num_cells": 1, "cells": [{"step": "score", "row": 7}]}
{"v": 1, "t": 0.011502, "type": "step_start", "step": "score", "level": 1, "mode": "realtime", "num_rows": 1}
{"v": 1, "t": 0.012219, "type": "row_complete", "step": "score", "row": 7, "key": "company-07", "status": "ok", "from_cache": false, "values": {"score": 70}, "error": null, "usage": null, "elapsed_ms": 0.281}
{"v": 1, "t": 0.012884, "type": "step_end", "step": "score", "num_errors": 0, "usage": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.001204, "batch_id": null}
{"v": 1, "t": 0.013001, "type": "retry_end", "num_rows": 12, "total_errors": 0, "cost": {"in": 0, "out": 0, "cost": null}, "elapsed_s": 0.001616, "num_cells": 1}
```

See the [checkpointing guide](caching.md#retrying-failed-cells) for what `retry_failed()` re-runs and what it reuses.

### Usage objects and `cost`

Everywhere a `{in, out, cost}` object appears: `in` is prompt tokens, `out` is completion tokens, and `cost` is reserved for dollar cost. Accrue deliberately ships no model pricing data (see `compare.py`), so **current emitters always write `cost: null`** — consumers price the tokens themselves, and must tolerate a number appearing here from a future emitter.

## Capture tiers

`run(..., capture=...)` controls how much of each LLM attempt the log persists. It defaults to `"metadata"`, and both `run()` and `run_async()` accept it.

```python
result = pipeline.run(data, run_log=True)                     # capture="metadata" (default)
result = pipeline.run(data, run_log=True, capture="prompts")  # + rendered prompt/response bodies
```

| Tier | What lands on disk |
|------|--------------------|
| `"metadata"` (default) | `row_attempt` records with attempt metadata only — no prompt or response bodies. `prompt_ref` is `null` everywhere and no sidecar file is created. |
| `"prompts"` | Everything in `metadata`, **plus** each LLM attempt's rendered system/user messages and the raw response/parsed object, written to the [prompt sidecar](#prompt-sidecar) and pointed to by `prompt_ref`. |
| `"full"` | Intended to also capture raw provider request/response payloads. The built-in provider adapters return a normalised response, so those payloads are not cheaply available — **`"full"` currently behaves exactly like `"prompts"`**. Recorded separately so a future emitter can add them without a version bump. |

**Why the default is metadata.** Enriched rows are frequently PII — names, emails, firmographics — and the rendered prompt embeds the row verbatim, while the response is the model's output about that row. Persisting bodies by default would write that to a file that outlives the process, so it is strictly opt-in. `row_attempt` metadata (timings, statuses, retry shape) carries no row content and is always safe to keep, which is why it is emitted at every tier. When you do opt in, captured bodies pass through the same `_sanitize_secrets` redaction as error text — an `api_key=…` or `sk-…` in a prompt lands as `***REDACTED***` on disk.

## Prompt sidecar

At `capture >= "prompts"`, attempt bodies are appended to a sidecar beside the main log — `<run>.prompts.jsonl` (e.g. `.accrue/runs/<id>.jsonl` → `.accrue/runs/<id>.prompts.jsonl`; `logs/tonight.jsonl` → `logs/tonight.prompts.jsonl`). The main log never inlines a body: it stays scannable, and a 500 MB capture is seek-only.

- **One JSON object per line**, each a body `{"messages": [{role, content}, …], "response": <raw text|null>, "parsed": <object|null>}`. `messages` is what was sent to the provider for that attempt (including the appended correction turns on a parse retry); `response` is the raw completion text; `parsed` is the validated object on a successful attempt, else `null`.
- **Referenced by byte span.** Each `row_attempt.prompt_ref` is `{"off": <byte offset>, "len": <byte length>}` pointing at the body's JSON (newline excluded). Append-only, flushed per line, created mode `0600` — the same hardening as the main log (bodies are often PII).
- **Resolving a ref.** `accrue.read_prompt_ref(sidecar_path, off, len)` seeks and parses one body; a record's ref splats straight in:

```python
from accrue import read_prompt_ref

sidecar = "logs/tonight.prompts.jsonl"
for rec in records:
    if rec.get("type") == "row_attempt" and rec["prompt_ref"]:
        body = read_prompt_ref(sidecar, **rec["prompt_ref"])
        print(body["messages"], body["response"])
```

The golden captured fixture for consumers lives at `tests/fixtures/run_captured.jsonl` with its `tests/fixtures/run_captured.prompts.jsonl` sidecar.

## Example

A 12-row, 3-step run — one row errors in `score`, one is skipped in `flag` (abridged from `tests/fixtures/run_small.jsonl`):

```json
{"v": 1, "t": 0.0, "type": "pipeline_start", "run_id": "2026-08-20-080226-4f1c9a", "started_at": "2026-08-20T07:02:26.020167+00:00", "num_rows": 12, "display_key": "company", "steps": [{"name": "normalize", "level": 0, "mode": "realtime", "model": null}, {"name": "score", "level": 1, "mode": "realtime", "model": null}, {"name": "flag", "level": 2, "mode": "realtime", "model": null}], "manifest": {"accrue_version": "1.3.0", "config": {"max_workers": 1, "caching": false, "checkpointing": false, "batch": false, "capture": "metadata"}, "steps": [{"name": "normalize", "type": "FunctionStep", "model": null, "produces": ["name_upper"], "depends_on": [], "condition": null}, {"name": "score", "type": "FunctionStep", "model": null, "produces": ["score"], "depends_on": ["normalize"], "condition": null}, {"name": "flag", "type": "FunctionStep", "model": null, "produces": ["flagged"], "depends_on": ["score"], "condition": null}], "fields": [{"name": "name_upper", "type": "unknown", "enum": null, "description": null, "step": "normalize", "internal": false}, {"name": "score", "type": "unknown", "enum": null, "description": null, "step": "score", "internal": false}, {"name": "flagged", "type": "unknown", "enum": null, "description": null, "step": "flag", "internal": false}]}, "plan": null}
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

- **Append-only.** The file is opened in append mode and never truncated; pointing `run_log` at an existing file adds to it. A second run appended that way opens its own `pipeline_start` … `pipeline_end` frame with its own `run_id`, and its `t` continues from the file's last `t` rather than restarting at zero — `t` is a within-file ordering clock, not wall-clock elapsed. One file per run is still the recommended layout; readers that assume a single run per file should key on `run_id`.
- **Flush per line.** Every record is flushed as soon as it is written. A crashed or killed run leaves a valid prefix of the stream — the tail is always complete lines, so `tail -f` and crash-recovery readers work.
- **Ordering.** `pipeline_start` is first; `pipeline_end` closes the run and fires even on error. Each step's `step_start` precedes all of its `row_complete` records, which precede its `step_end`. Steps at the same DAG level run in parallel, so *their* records may interleave; `t` stays non-decreasing throughout.
- **Segments.** A file holds one run, optionally followed by one `retry_start` … `retry_end` segment per `retry_failed()` call, each carrying the run's original `run_id`. Nothing follows a `pipeline_end` except retry segments.
- **Completeness.** Every row of every executed step gets exactly one `row_complete` — including cached, skipped, and errored rows, and including rows whose values cannot be encoded (those carry the `__unserializable__` placeholder). In a retry segment, "executed" means the retried cells: a row served from the checkpoint produces no record, and its state is whatever the earlier segment last said.
  The same applies to a run that **auto-resumed from a checkpoint**: steps served from the checkpoint are never executed, so they emit no `step_start` / `row_complete` / `step_end` at all. Such a log covers only the steps (and failed cells) that actually ran — a consumer reading it alone sees partial coverage of the dataset, not a complete run.
- **Never crashes the run.** The logger is an ordinary hooks consumer; hook errors are caught and logged, and values JSON cannot represent degrade (to strings, to `null`, or to the `__unserializable__` placeholder) rather than raise.
- **Every line is strict JSON.** No `NaN` / `Infinity` literals, so strict parsers — `JSON.parse`, `serde_json`, `jq` — read the file without special-casing.

## Versioning policy

- `v` is bumped **only for breaking changes** (removing/renaming a field, changing a field's meaning or type).
- Within v1, changes are **additive only**: new record types or new fields may appear. Consumers MUST ignore unknown fields and unknown record types.
- Fields documented as nullable may start carrying values in newer emitters (e.g. `plan`, per-row `usage` in batch mode, `cost` in dollars). Consumers MUST NOT treat `null` as a guarantee.

## Where logs live

`run_log=True` writes to `.accrue/runs/` relative to the process CWD — alongside accrue's SQLite cache default (`.accrue/`). One file per run, named by `run_id`. Accrue never rotates or deletes run logs; clean the directory yourself if it grows.

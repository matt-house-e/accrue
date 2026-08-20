"""Tests for the v0.2 run-log additions: RowAttemptEvent, capture tiers, and
the prompt sidecar (issue #134).

Validates the committed captured fixtures (``tests/fixtures/run_captured.jsonl``
and its ``.prompts.jsonl`` sidecar) and exercises ``Pipeline.run(..., capture=)``
end-to-end with a scripted fake LLM client — no network, no API key, no real
LLM.  The default (``metadata``) tier and the ``prompts`` tier are both checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accrue import (
    EnrichmentConfig,
    EnrichmentHooks,
    LLMStep,
    Pipeline,
    RowAttemptEvent,
    read_prompt_ref,
)
from accrue.core.runlog import SCHEMA_VERSION, prompt_sidecar_path
from tests.fixtures.generate_captured_fixture import (
    PLANTED_SECRET,
    _ScriptedClient,
    build_pipeline,
    build_rows,
    generate,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_captured.jsonl"
SIDECAR = Path(__file__).resolve().parent / "fixtures" / "run_captured.prompts.jsonl"

ROW_ATTEMPT_KEYS = {
    "step",
    "row",
    "attempt",
    "kind",
    "status",
    "latency_ms",
    "backoff_s",
    "error",
    "prompt_ref",
}


def _load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, f"{path} is empty"
    return [json.loads(line) for line in lines]


def _quiet_config(**overrides) -> EnrichmentConfig:
    defaults = dict(
        enable_caching=False,
        enable_progress_bar=False,
        max_workers=1,
        max_retries=2,
        retry_base_delay=0.001,
    )
    defaults.update(overrides)
    return EnrichmentConfig(**defaults)


# ---------------------------------------------------------------------------
# Golden captured fixture
# ---------------------------------------------------------------------------


class TestCapturedFixture:
    def test_every_line_parses_with_envelope(self):
        for rec in _load(FIXTURE):
            assert rec["v"] == SCHEMA_VERSION
            assert isinstance(rec["t"], (int, float)) and rec["t"] >= 0
            assert "type" in rec

    def test_t_non_decreasing(self):
        ts = [rec["t"] for rec in _load(FIXTURE)]
        assert ts == sorted(ts)

    def test_row_attempt_records_validate(self):
        attempts = [r for r in _load(FIXTURE) if r["type"] == "row_attempt"]
        assert attempts, "fixture should contain row_attempt records"
        for r in attempts:
            assert ROW_ATTEMPT_KEYS <= set(r.keys())
            assert r["kind"] in ("api", "parse")
            assert isinstance(r["attempt"], int) and r["attempt"] >= 1
            assert isinstance(r["status"], str) and r["status"]
            assert r["latency_ms"] is None or isinstance(r["latency_ms"], (int, float))
            assert r["backoff_s"] is None or isinstance(r["backoff_s"], (int, float))

    def test_attempt_counts_and_kinds(self):
        records = _load(FIXTURE)
        attempts = [r for r in records if r["type"] == "row_attempt"]
        completes = [r for r in records if r["type"] == "row_complete"]

        # 3 rows: 1 + 2 + 2 attempts, one row_complete each.
        assert len(attempts) == 5
        assert len(completes) == 3

        by_row: dict[int, list[dict]] = {}
        for a in attempts:
            by_row.setdefault(a["row"], []).append(a)

        # row 0: single parse/ok attempt.
        assert [(a["kind"], a["status"]) for a in by_row[0]] == [("parse", "ok")]
        # row 1: an api rate-limit, then a parse success.
        assert [(a["kind"], a["status"]) for a in by_row[1]] == [
            ("api", "rate_limited"),
            ("parse", "ok"),
        ]
        # row 2: an unparseable reply, then a parse success.
        assert [(a["kind"], a["status"]) for a in by_row[2]] == [
            ("parse", "parse_error"),
            ("parse", "ok"),
        ]

    def test_attempts_precede_their_row_complete(self):
        records = _load(FIXTURE)
        for row in (0, 1, 2):
            positions = [
                i
                for i, r in enumerate(records)
                if r.get("row") == row and r["type"] in ("row_attempt", "row_complete")
            ]
            kinds = [records[i]["type"] for i in positions]
            complete_at = kinds.index("row_complete")
            # every row_attempt for this row comes before its row_complete
            assert all(k == "row_attempt" for k in kinds[:complete_at])
            assert kinds[complete_at:] == ["row_complete"]

    def test_api_attempt_has_null_prompt_ref(self):
        attempts = [r for r in _load(FIXTURE) if r["type"] == "row_attempt"]
        api = [a for a in attempts if a["kind"] == "api"]
        assert api and all(a["prompt_ref"] is None for a in api)
        assert all(a["error"] is not None for a in api)

    def test_backoff_recorded_on_retrying_api_attempt(self):
        attempts = [r for r in _load(FIXTURE) if r["type"] == "row_attempt"]
        api = [a for a in attempts if a["kind"] == "api"]
        # The rate-limited attempt schedules a retry, so it carries a backoff.
        assert all(a["backoff_s"] is not None and a["backoff_s"] >= 0 for a in api)

    def test_every_prompt_ref_resolves_to_its_body(self):
        attempts = [r for r in _load(FIXTURE) if r["type"] == "row_attempt"]
        refs = [a for a in attempts if a["prompt_ref"] is not None]
        # Every parse attempt (ok or failed) captured a body.
        assert len(refs) == 4
        for a in refs:
            body = read_prompt_ref(SIDECAR, **a["prompt_ref"])
            assert set(body) == {"messages", "response", "parsed"}
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"
            # A parse/ok attempt carries the parsed object; a failed one does not.
            if a["status"] == "ok":
                assert isinstance(body["parsed"], dict)
            else:
                assert body["parsed"] is None

    def test_sidecar_line_count_matches_refs(self):
        refs = [
            r["prompt_ref"]
            for r in _load(FIXTURE)
            if r["type"] == "row_attempt" and r["prompt_ref"] is not None
        ]
        assert len(_load(SIDECAR)) == len(refs)

    def test_planted_secret_is_redacted_in_sidecar(self):
        raw = SIDECAR.read_text(encoding="utf-8")
        assert PLANTED_SECRET not in raw
        assert "***REDACTED***" in raw

    def test_main_log_never_inlines_bodies(self):
        # No row_attempt / row_complete record carries a prompt body inline.
        for r in _load(FIXTURE):
            assert "body" not in r
            assert "messages" not in r


# ---------------------------------------------------------------------------
# Capture tiers end-to-end
# ---------------------------------------------------------------------------


class TestCaptureTiers:
    def test_default_capture_is_metadata_no_sidecar(self, tmp_path):
        log = tmp_path / "run.jsonl"
        build_pipeline().run(build_rows(), config=_quiet_config(), run_log=str(log))
        sidecar = prompt_sidecar_path(log)
        assert not sidecar.exists(), "metadata (default) must write no sidecar"
        # row_attempt records are still emitted, every prompt_ref null.
        attempts = [r for r in _load(log) if r["type"] == "row_attempt"]
        assert attempts
        assert all(a["prompt_ref"] is None for a in attempts)

    def test_metadata_capture_explicit_no_sidecar(self, tmp_path):
        log = tmp_path / "run.jsonl"
        build_pipeline().run(
            build_rows(), config=_quiet_config(), run_log=str(log), capture="metadata"
        )
        assert not prompt_sidecar_path(log).exists()

    def test_prompts_capture_writes_resolvable_sidecar(self, tmp_path):
        log = tmp_path / "run.jsonl"
        build_pipeline().run(
            build_rows(), config=_quiet_config(), run_log=str(log), capture="prompts"
        )
        sidecar = prompt_sidecar_path(log)
        assert sidecar.exists()
        refs = [
            r["prompt_ref"]
            for r in _load(log)
            if r["type"] == "row_attempt" and r["prompt_ref"] is not None
        ]
        assert refs, "capture=prompts should attach prompt_refs"
        for ref in refs:
            body = read_prompt_ref(sidecar, ref["off"], ref["len"])
            assert body["messages"][0]["role"] == "system"
            assert "<row_data>" in body["messages"][1]["content"]

    def test_full_capture_behaves_like_prompts(self, tmp_path):
        log = tmp_path / "run.jsonl"
        build_pipeline().run(build_rows(), config=_quiet_config(), run_log=str(log), capture="full")
        sidecar = prompt_sidecar_path(log)
        assert sidecar.exists()
        # Same body shape as prompts — raw provider payloads aren't captured.
        body = _load(sidecar)[0]
        assert set(body) == {"messages", "response", "parsed"}

    def test_invalid_capture_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="capture must be"):
            build_pipeline().run(
                build_rows(),
                config=_quiet_config(),
                run_log=str(tmp_path / "run.jsonl"),
                capture="bogus",
            )

    def test_sidecar_created_owner_only(self, tmp_path):
        import os
        import stat

        log = tmp_path / "run.jsonl"
        build_pipeline().run(
            build_rows(), config=_quiet_config(), run_log=str(log), capture="prompts"
        )
        mode = stat.S_IMODE(os.stat(prompt_sidecar_path(log)).st_mode)
        assert mode == 0o600

    def test_secret_in_prompt_is_redacted_on_disk(self, tmp_path):
        secret = "sk-livesecretABCDEFGHIJKLMNOP0123456789"
        rows = [{"company": "acme", "note": f"api_key={secret}"}]
        log = tmp_path / "run.jsonl"
        build_pipeline().run(rows, config=_quiet_config(), run_log=str(log), capture="prompts")
        raw = prompt_sidecar_path(log).read_text(encoding="utf-8")
        assert secret not in raw
        assert "***REDACTED***" in raw

    def test_regeneration_of_fixture_is_structurally_stable(self, tmp_path):
        # Regenerating into a temp dir reproduces the record kinds/statuses
        # (timestamps and jitter aside).
        log = tmp_path / "run_captured.jsonl"
        sidecar = tmp_path / "run_captured.prompts.jsonl"
        generate(log, sidecar)
        regen = [(r["type"], r.get("row"), r.get("kind"), r.get("status")) for r in _load(log)]
        committed = [
            (r["type"], r.get("row"), r.get("kind"), r.get("status")) for r in _load(FIXTURE)
        ]
        assert regen == committed


# ---------------------------------------------------------------------------
# RowAttemptEvent as a plain hook (no run log)
# ---------------------------------------------------------------------------


class TestRowAttemptEvent:
    @pytest.mark.asyncio
    async def test_event_fires_right_count_and_kind_for_retrying_cell(self):
        events: list[RowAttemptEvent] = []

        async def collect(e: RowAttemptEvent) -> None:
            events.append(e)

        hooks = EnrichmentHooks(on_row_attempt=collect)
        await build_pipeline().run_async(build_rows(), config=_quiet_config(), hooks=hooks)

        by_row: dict[int, list[RowAttemptEvent]] = {}
        for e in events:
            by_row.setdefault(e.row_index, []).append(e)

        assert len(events) == 5
        assert [(e.kind, e.status) for e in by_row[0]] == [("parse", "ok")]
        assert [(e.kind, e.status) for e in by_row[1]] == [
            ("api", "rate_limited"),
            ("parse", "ok"),
        ]
        assert [(e.kind, e.status) for e in by_row[2]] == [
            ("parse", "parse_error"),
            ("parse", "ok"),
        ]
        # 1-based attempt numbers, and the api attempt carries the error.
        assert all(e.attempt >= 1 for e in events)
        api = [e for e in events if e.kind == "api"]
        assert api and all(e.error is not None for e in api)

    @pytest.mark.asyncio
    async def test_no_attempt_hook_means_no_overhead(self):
        # With no on_row_attempt hook the step still runs to completion (the
        # emitter is None and every emit short-circuits).
        result = await build_pipeline().run_async(build_rows(), config=_quiet_config())
        grades = [row["grade"] for row in result.data]
        assert grades == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_bodies_absent_at_metadata_present_at_prompts(self, tmp_path):
        # Without a run log the event body is None at metadata; the LLM step
        # only builds a body when the capture tier calls for it.
        meta_bodies: list = []
        prompt_bodies: list = []

        async def collect_meta(e: RowAttemptEvent) -> None:
            meta_bodies.append(e.body)

        async def collect_prompts(e: RowAttemptEvent) -> None:
            prompt_bodies.append(e.body)

        step = LLMStep(
            name="classify",
            fields={"grade": "grade it"},
            client=_ScriptedClient(),
            max_retries=2,
            cache=False,
        )
        pipe = Pipeline([step])
        await pipe.run_async(
            build_rows(), config=_quiet_config(), hooks=EnrichmentHooks(on_row_attempt=collect_meta)
        )
        assert all(b is None for b in meta_bodies)

        pipe2 = Pipeline(
            [
                LLMStep(
                    name="classify",
                    fields={"grade": "grade it"},
                    client=_ScriptedClient(),
                    max_retries=2,
                    cache=False,
                )
            ]
        )
        await pipe2.run_async(
            build_rows(),
            config=_quiet_config(),
            hooks=EnrichmentHooks(on_row_attempt=collect_prompts),
            capture="prompts",
        )
        # Parse attempts (ok / failed) carry a body; the api attempt does not.
        assert any(b is not None for b in prompt_bodies)
        assert any(b is None for b in prompt_bodies)  # the api attempt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_prompt_sidecar_path_swaps_suffix(self):
        assert prompt_sidecar_path("logs/tonight.jsonl") == Path("logs/tonight.prompts.jsonl")
        assert prompt_sidecar_path(Path(".accrue/runs/2026-01-01-000000-abcdef.jsonl")) == Path(
            ".accrue/runs/2026-01-01-000000-abcdef.prompts.jsonl"
        )

    def test_read_prompt_ref_splat(self):
        rec = next(
            r for r in _load(FIXTURE) if r["type"] == "row_attempt" and r["prompt_ref"] is not None
        )
        body = read_prompt_ref(SIDECAR, **rec["prompt_ref"])
        assert "messages" in body

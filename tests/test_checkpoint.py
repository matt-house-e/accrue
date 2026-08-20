"""Tests for the per-step CheckpointManager."""

import json
import logging
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from accrue.core.checkpoint import (
    CheckpointData,
    CheckpointManager,
    CheckpointSession,
    derive_data_identifier,
)
from accrue.core.config import EnrichmentConfig
from accrue.core.exceptions import RowError

# -- helpers -----------------------------------------------------------------


def _make_mgr(tmp_path: Path, *, enabled=True, auto_resume=True) -> CheckpointManager:
    config = EnrichmentConfig(
        enable_checkpointing=enabled,
        auto_resume=auto_resume,
        checkpoint_dir=str(tmp_path),
    )
    return CheckpointManager(config)


FIELDS = {"company_type": {"prompt": "Classify", "type": "String"}}


# -- save / load round-trip --------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_single_step(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        row_results = [{"company_type": "B2B"}, {"company_type": "B2C"}]

        ok = mgr.save_step(
            data_identifier="test_data",
            category="info",
            step_name="classify",
            step_row_results=row_results,
            total_rows=2,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )
        assert ok is True

        cp = mgr.load("test_data", "info")
        assert cp is not None
        assert isinstance(cp, CheckpointData)
        assert cp.category == "info"
        assert cp.total_rows == 2
        assert cp.completed_steps == ["classify"]
        assert cp.step_results["classify"] == row_results
        assert cp.fields_dict == FIELDS

    def test_multiple_steps(self, tmp_path):
        mgr = _make_mgr(tmp_path)

        step1_results = [{"f1": "a"}, {"f1": "b"}]
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="step1",
            step_row_results=step1_results,
            total_rows=2,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        step2_results = [{"f2": "x"}, {"f2": "y"}]
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="step2",
            step_row_results=step2_results,
            total_rows=2,
            fields_dict=FIELDS,
            existing_completed=["step1"],
            existing_results={"step1": step1_results},
        )

        cp = mgr.load("data", "cat")
        assert cp is not None
        assert cp.completed_steps == ["step1", "step2"]
        assert cp.step_results["step1"] == step1_results
        assert cp.step_results["step2"] == step2_results


# -- load returns None -------------------------------------------------------


class TestLoadReturnsNone:
    def test_no_file(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        assert mgr.load("nonexistent", "cat") is None

    def test_disabled(self, tmp_path):
        mgr = _make_mgr(tmp_path, enabled=False)
        assert mgr.load("data", "cat") is None

    def test_auto_resume_false(self, tmp_path):
        # Save with a fully-enabled manager, then try to load with auto_resume=False
        mgr_save = _make_mgr(tmp_path)
        mgr_save.save_step(
            data_identifier="data",
            category="cat",
            step_name="s",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        mgr_no_resume = _make_mgr(tmp_path, auto_resume=False)
        assert mgr_no_resume.load("data", "cat") is None

    def test_category_mismatch(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.save_step(
            data_identifier="data",
            category="cat_a",
            step_name="s",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        # Load with a different category — but same identifier gives same file path,
        # so the file exists but category doesn't match
        assert mgr.load("data", "cat_b") is None


# -- cleanup -----------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_file(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="s",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        # File should exist
        files = list(tmp_path.glob("*_checkpoint.json"))
        assert len(files) == 1

        mgr.cleanup("data", "cat")

        # File should be gone
        files = list(tmp_path.glob("*_checkpoint.json"))
        assert len(files) == 0

    def test_cleanup_noop_when_disabled(self, tmp_path):
        mgr = _make_mgr(tmp_path, enabled=False)
        assert mgr.cleanup("data", "cat") is True


# -- list_checkpoints -------------------------------------------------------


class TestListCheckpoints:
    def test_finds_checkpoints(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.save_step(
            data_identifier="alpha",
            category="cat1",
            step_name="s",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )
        mgr.save_step(
            data_identifier="beta",
            category="cat2",
            step_name="s",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        found = mgr.list_checkpoints()
        assert len(found) == 2
        # Check that category info is available
        categories = {v["category"] for v in found.values()}
        assert "cat1" in categories
        assert "cat2" in categories

    def test_empty_when_disabled(self, tmp_path):
        mgr = _make_mgr(tmp_path, enabled=False)
        assert mgr.list_checkpoints() == {}


# -- save_step returns True when disabled (no-op) ---------------------------


class TestSaveStepDisabled:
    def test_returns_true(self, tmp_path):
        mgr = _make_mgr(tmp_path, enabled=False)
        result = mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="s",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )
        assert result is True

        # No file should have been written
        files = list(tmp_path.glob("*_checkpoint.json"))
        assert len(files) == 0


# -- atomic write ------------------------------------------------------------


class TestAtomicWrite:
    def test_original_file_intact_after_failed_write(self, tmp_path):
        """If json.dump raises mid-write, the original checkpoint file is untouched."""
        mgr = _make_mgr(tmp_path)

        # Write a valid checkpoint first
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="step1",
            step_row_results=[{"v": 1}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )
        checkpoint_file = tmp_path / "data_cat_checkpoint.json"
        original_content = checkpoint_file.read_text()

        # Now simulate a failure during json.dump — the tmp file is left behind
        # but the original must remain intact.
        with patch("json.dump", side_effect=OSError("simulated disk failure")):
            ok = mgr.save_step(
                data_identifier="data",
                category="cat",
                step_name="step2",
                step_row_results=[{"v": 2}],
                total_rows=1,
                fields_dict=FIELDS,
                existing_completed=["step1"],
                existing_results={"step1": [{"v": 1}]},
            )

        assert ok is False  # save reported failure
        # Original file must be readable and unchanged
        assert checkpoint_file.read_text() == original_content
        # Parsed content should still match the first successful write
        saved = json.loads(original_content)
        assert saved["completed_steps"] == ["step1"]


# -- strict resume validation ------------------------------------------------


class TestStrictResumeValidation:
    def _save(self, mgr, *, total_rows=10, fields=None):
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="s",
            step_row_results=[{}] * total_rows,
            total_rows=total_rows,
            fields_dict=fields or FIELDS,
            existing_completed=[],
            existing_results={},
        )

    def test_rejects_mismatched_row_count(self, tmp_path, caplog):
        mgr = _make_mgr(tmp_path)
        self._save(mgr, total_rows=10)

        with caplog.at_level(logging.WARNING, logger="accrue.core.checkpoint"):
            result = mgr.load("data", "cat", expected_total_rows=5)

        assert result is None
        assert any("row count mismatch" in r.message.lower() for r in caplog.records)

    def test_rejects_mismatched_fields(self, tmp_path, caplog):
        mgr = _make_mgr(tmp_path)
        self._save(mgr, fields={"a": {}, "b": {}})

        with caplog.at_level(logging.WARNING, logger="accrue.core.checkpoint"):
            result = mgr.load("data", "cat", expected_fields={"a": {}, "b": {}, "c": {}})

        assert result is None
        assert any("fields mismatch" in r.message.lower() for r in caplog.records)

    def test_rejects_unknown_steps(self, tmp_path, caplog):
        mgr = _make_mgr(tmp_path)
        # Checkpoint contains step "old_step" that is no longer in the pipeline.
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="old_step",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        with caplog.at_level(logging.WARNING, logger="accrue.core.checkpoint"):
            result = mgr.load("data", "cat", expected_steps=["new_step"])

        assert result is None
        assert any("unknown to the current pipeline" in r.message for r in caplog.records)

    def test_accepts_when_expected_kwargs_are_none(self, tmp_path):
        """Backwards-compat: omitting expected_* kwargs skips those checks."""
        mgr = _make_mgr(tmp_path)
        self._save(mgr, total_rows=10)

        # No expected_* kwargs — should load fine
        result = mgr.load("data", "cat")
        assert result is not None
        assert result.total_rows == 10

    def test_accepts_matching_validation(self, tmp_path):
        """All expected_* kwargs match — checkpoint is accepted."""
        mgr = _make_mgr(tmp_path)
        self._save(mgr, total_rows=10, fields={"a": {}, "b": {}})

        result = mgr.load(
            "data",
            "cat",
            expected_total_rows=10,
            expected_fields={"a": {}, "b": {}},
            expected_steps=["s", "other"],
        )
        assert result is not None


# -- typed serializer --------------------------------------------------------


class TestTypedSerializer:
    def _round_trip(self, mgr, value, tmp_path):
        """Save a checkpoint containing *value* in step_results; return loaded value."""
        mgr.save_step(
            data_identifier="typed",
            category="cat",
            step_name="s",
            step_row_results=[{"val": value}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )
        cp = mgr.load("typed", "cat")
        assert cp is not None
        return cp.step_results["s"][0]["val"]

    def test_datetime_round_trip(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        dt = datetime(2024, 6, 15, 12, 30, 45)
        result = self._round_trip(mgr, dt, tmp_path)
        assert isinstance(result, datetime)
        assert result == dt

    def test_decimal_round_trip(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        d = Decimal("3.14159")
        result = self._round_trip(mgr, d, tmp_path)
        assert isinstance(result, Decimal)
        assert result == d

    def test_set_round_trip(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        s = {"apple", "banana", "cherry"}
        result = self._round_trip(mgr, s, tmp_path)
        assert isinstance(result, set)
        assert result == s

    def test_unknown_type_raises_on_save(self, tmp_path):
        """Saving an unserializable type must raise TypeError, not silently stringify."""
        mgr = _make_mgr(tmp_path)

        class _Unserializable:
            pass

        with pytest.raises(TypeError, match="not JSON serializable"):
            mgr.save_step(
                data_identifier="data",
                category="cat",
                step_name="s",
                step_row_results=[{"val": _Unserializable()}],
                total_rows=1,
                fields_dict=FIELDS,
                existing_completed=[],
                existing_results={},
            )

    def test_user_dict_with_dunder_type_key_round_trips_unchanged(self, tmp_path):
        """User data containing '__type__' must survive save+load without interpretation."""
        mgr = _make_mgr(tmp_path)
        user_record = {"__type__": "user_tag", "value": "x"}

        result = self._round_trip(mgr, user_record, tmp_path)

        assert result == user_record
        assert result["__type__"] == "user_tag"

    def test_tmp_file_cleaned_up_on_encoder_failure(self, tmp_path):
        """A TypeError mid-encode must unlink the .tmp file before re-raising."""
        mgr = _make_mgr(tmp_path)

        with pytest.raises(TypeError):
            mgr.save_step(
                data_identifier="data",
                category="cat",
                step_name="s",
                step_row_results=[{"val": complex(1, 2)}],
                total_rows=1,
                fields_dict=FIELDS,
                existing_completed=[],
                existing_results={},
            )

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned .tmp files found: {tmp_files}"


# -- partial checkpoint (partial=True) ---------------------------------------


class TestPartialCheckpoint:
    def test_partial_true_does_not_add_to_completed_steps(self, tmp_path):
        """save_step(partial=True) must NOT append the step to completed_steps."""
        mgr = _make_mgr(tmp_path)
        row_results = [{"f": "partial_val"}, {}]

        ok = mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="step1",
            step_row_results=row_results,
            total_rows=2,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
            partial=True,
        )
        assert ok is True

        cp = mgr.load("data", "cat")
        assert cp is not None
        # Step must NOT appear in completed_steps so a resumed run re-executes it.
        assert "step1" not in cp.completed_steps
        assert cp.completed_steps == []
        # Partial data is still persisted in step_results for tracking purposes.
        assert cp.step_results["step1"] == row_results

    def test_partial_false_default_adds_to_completed_steps(self, tmp_path):
        """save_step(partial=False) (the default) DOES append to completed_steps."""
        mgr = _make_mgr(tmp_path)
        row_results = [{"f": "val"}]

        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="step1",
            step_row_results=row_results,
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        cp = mgr.load("data", "cat")
        assert cp is not None
        assert "step1" in cp.completed_steps

    def test_partial_preserves_prior_completed_steps_unchanged(self, tmp_path):
        """Partial save must not disturb already-completed steps."""
        mgr = _make_mgr(tmp_path)

        # First step is fully completed.
        step1_results = [{"f1": "done"}]
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="step1",
            step_row_results=step1_results,
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

        # Second step is only partially done (e.g. cancelled mid-flight).
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="step2",
            step_row_results=[{}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=["step1"],
            existing_results={"step1": step1_results},
            partial=True,
        )

        cp = mgr.load("data", "cat")
        assert cp is not None
        # Only step1 is completed; step2 must NOT appear.
        assert cp.completed_steps == ["step1"]
        assert "step2" not in cp.completed_steps


# -- Legacy checkpoint format (fix #80) -------------------------------------


class TestLegacyCheckpointDetection:
    def test_legacy_type_sentinel_returns_none_and_warns(self, tmp_path, caplog):
        """Pre-1.2.1 checkpoint using __type__ must be discarded with a WARNING."""
        mgr = _make_mgr(tmp_path)

        # Write a checkpoint file containing the old __type__ sentinel
        legacy_payload = {
            "timestamp": 1000.0,
            "category": "cat",
            "total_rows": 1,
            "fields_dict": {"col": {}},
            "completed_steps": ["step1"],
            "step_results": {
                "step1": [{"val": {"__type__": "datetime", "value": "2024-01-01T00:00:00"}}]
            },
        }
        path = mgr._get_path("data", "cat")
        path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="accrue.core.checkpoint"):
            result = mgr.load("data", "cat")

        assert result is None, "Legacy checkpoint must be discarded (return None)"
        assert "Legacy checkpoint format" in caplog.text, "A WARNING must be logged"


# -- XDG defaults + 0o600 permissions ----------------------------------------


class TestXdgCheckpointDirDefault:
    def test_xdg_state_home_respected(self, monkeypatch, tmp_path):
        """XDG_STATE_HOME is used as the base for checkpoint_dir."""
        xdg_state = str(tmp_path / "xdg_state")
        monkeypatch.setenv("XDG_STATE_HOME", xdg_state)
        config = EnrichmentConfig()
        assert config.checkpoint_dir == os.path.join(xdg_state, "accrue")

    def test_fallback_to_local_state_when_xdg_unset(self, monkeypatch):
        """Falls back to ~/.local/state/accrue when XDG_STATE_HOME is unset."""
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        config = EnrichmentConfig()
        expected = os.path.join(os.path.expanduser("~"), ".local", "state", "accrue")
        assert config.checkpoint_dir == expected

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod")
    def test_checkpoint_file_has_0o600_permissions(self, tmp_path):
        """Checkpoint files are written with 0o600 permissions."""
        mgr = _make_mgr(tmp_path)
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="s",
            step_row_results=[{"v": 1}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )
        checkpoint_files = list(tmp_path.glob("*_checkpoint.json"))
        assert len(checkpoint_files) == 1
        stat = checkpoint_files[0].stat()
        assert stat.st_mode & 0o777 == 0o600

    def test_custom_checkpoint_dir_constructor_arg_still_works(self, tmp_path):
        """Regression: explicit checkpoint_dir= overrides the XDG default."""
        custom = str(tmp_path / "custom_checkpoints")
        config = EnrichmentConfig(checkpoint_dir=custom)
        assert config.checkpoint_dir == custom


# -- data identifier covers content, not just shape --------------------------


class TestDataIdentifierCoversContent:
    """The identifier keyed only ``{columns, num_rows}``.

    With ``checkpoint_dir`` defaulting to one global XDG state directory, two
    unrelated datasets that happened to share a shape resolved to the same
    checkpoint file — and the second silently received the first one's
    enrichments, with no error anywhere.
    """

    def test_same_shape_different_content_differs(self):
        a = [{"company": "Acme"}, {"company": "Globex"}]
        b = [{"company": "Initech"}, {"company": "Umbrella"}]
        assert derive_data_identifier(["company"], 2, a) != derive_data_identifier(
            ["company"], 2, b
        )

    def test_same_content_is_stable_across_calls(self):
        rows = [{"company": f"c-{i}", "n": i} for i in range(20)]
        first = derive_data_identifier(["company", "n"], 20, rows)
        second = derive_data_identifier(["company", "n"], 20, [dict(r) for r in rows])
        assert first == second
        assert first.startswith("df_")
        assert len(first) == len("df_") + 16

    def test_only_the_sampled_rows_are_hashed(self):
        """First and last 5 rows: an edit in the middle of a big set can slip through."""
        rows = [{"n": i} for i in range(20)]
        edited = [dict(r) for r in rows]
        edited[10]["n"] = 999
        assert derive_data_identifier(["n"], 20, rows) == derive_data_identifier(["n"], 20, edited)

        edited_edge = [dict(r) for r in rows]
        edited_edge[0]["n"] = 999
        assert derive_data_identifier(["n"], 20, rows) != derive_data_identifier(
            ["n"], 20, edited_edge
        )

    def test_heterogeneous_list_dict_rows(self):
        """``list[dict]`` rows need not share keys, and values need not be JSON."""
        a = [{"x": 1}, {"y": datetime(2026, 1, 1)}, {"z": {1: "int key"}}]
        b = [{"x": 2}, {"y": datetime(2026, 1, 2)}, {"z": {1: "int key"}}]
        id_a = derive_data_identifier(["x"], 3, a)
        assert id_a == derive_data_identifier(["x"], 3, [dict(r) for r in a])
        assert id_a != derive_data_identifier(["x"], 3, b)

    def test_key_order_does_not_change_the_identifier(self):
        a = [{"x": 1, "y": 2}]
        b = [{"y": 2, "x": 1}]
        assert derive_data_identifier(["x", "y"], 1, a) == derive_data_identifier(["x", "y"], 1, b)

    def test_rows_omitted_still_works(self):
        """Callers that pass no rows keep a shape-only identifier."""
        assert derive_data_identifier(["x"], 2).startswith("df_")

    def test_two_same_shaped_datasets_do_not_share_a_checkpoint(self, tmp_path):
        """End-to-end: dataset B must not receive dataset A's enrichments."""
        from accrue.core.config import EnrichmentConfig as Config
        from accrue.pipeline.pipeline import Pipeline
        from accrue.steps.function import FunctionStep

        seen: list[str] = []

        def upper(ctx):
            seen.append(ctx.row["company"])
            if ctx.row["company"] == "Globex":
                raise ValueError("boom")  # keeps A's checkpoint on disk
            return {"upper": ctx.row["company"].upper()}

        pipeline = Pipeline([FunctionStep("up", upper, fields=["upper"])])
        config = Config(
            enable_checkpointing=True,
            checkpoint_dir=str(tmp_path),
            enable_caching=False,
            enable_progress_bar=False,
        )

        a = [{"company": "Acme"}, {"company": "Globex"}]
        b = [{"company": "Initech"}, {"company": "Umbrella"}]

        pipeline.run(a, config=config)
        assert list(tmp_path.glob("*_checkpoint.json")), "A's checkpoint is kept (it errored)"

        seen.clear()
        result = pipeline.run(b, config=config)

        assert seen == ["Initech", "Umbrella"], "B must actually run, not resume A"
        assert [row["upper"] for row in result.data] == ["INITECH", "UMBRELLA"]


# -- concurrent writers ------------------------------------------------------


class TestTmpFileIsUniquePerWriter:
    def _save(self, mgr, value):
        mgr.save_step(
            data_identifier="data",
            category="cat",
            step_name="s",
            step_row_results=[{"v": value}],
            total_rows=1,
            fields_dict=FIELDS,
            existing_completed=[],
            existing_results={},
        )

    def test_tmp_name_carries_pid_and_a_random_suffix(self, tmp_path):
        """A constant tmp name let two processes interleave one atomic write."""
        mgr = _make_mgr(tmp_path)
        seen: list[str] = []
        real_replace = os.replace

        def spy(src, dst):
            seen.append(str(src))
            return real_replace(src, dst)

        with patch("accrue.core.checkpoint.os.replace", side_effect=spy):
            self._save(mgr, 1)
            self._save(mgr, 2)

        assert len(seen) == 2
        assert seen[0] != seen[1], "each write needs its own tmp file"
        assert all(str(os.getpid()) in name and name.endswith(".tmp") for name in seen)

    def test_a_concurrent_writers_tmp_file_is_left_alone(self, tmp_path):
        """A tmp file from another writer must not be consumed by this one."""
        mgr = _make_mgr(tmp_path)
        self._save(mgr, 1)
        checkpoint = next(tmp_path.glob("*_checkpoint.json"))
        foreign = checkpoint.with_suffix(".json.99999.deadbeef.tmp")
        foreign.write_text("half written", encoding="utf-8")

        self._save(mgr, 2)

        assert json.loads(checkpoint.read_text())["step_results"]["s"] == [{"v": 2}]
        assert foreign.read_text() == "half written"


# -- CheckpointSession write-back --------------------------------------------


def _session(tmp_path, *, checkpoint=None, retry_cells=None, total_rows=3):
    return CheckpointSession(
        _make_mgr(tmp_path),
        data_identifier="data",
        category="cat",
        total_rows=total_rows,
        fields_dict=FIELDS,
        checkpoint=checkpoint,
        retry_cells=retry_cells,
    )


def _saved(tmp_path) -> dict:
    return json.loads((tmp_path / "data_cat_checkpoint.json").read_text(encoding="utf-8"))


class TestPersistedErrorMessages:
    def test_message_is_capped(self, tmp_path):
        """A provider can return a multi-megabyte body; the checkpoint keeps 500 chars."""
        session = _session(tmp_path)
        session.on_step_complete(
            "s",
            [{"v": None}, {"v": 1}, {"v": 2}],
            [RowError(row_index=0, step_name="s", error=ValueError("x" * 5000))],
        )
        msg = _saved(tmp_path)["row_errors"]["s"]["0"]["msg"]
        assert len(msg) == 500

    def test_short_message_is_untouched(self, tmp_path):
        session = _session(tmp_path)
        session.on_step_complete(
            "s",
            [{"v": None}] * 3,
            [RowError(row_index=0, step_name="s", error=ValueError("boom"))],
        )
        assert _saved(tmp_path)["row_errors"]["s"]["0"]["msg"] == "boom"


class TestSkippedRetriedCellStaysUnresolved:
    def test_a_skipped_retried_cell_keeps_its_recorded_error(self, tmp_path):
        """A predicate that now skips the cell has not healed it."""
        checkpoint = CheckpointData(
            timestamp=1.0,
            category="cat",
            total_rows=3,
            fields_dict=FIELDS,
            completed_steps=["s"],
            step_results={"s": [{"v": None}, {"v": 1}, {"v": 2}]},
            row_errors={"s": {0: {"type": "ValueError", "msg": "boom"}}},
        )
        session = _session(tmp_path, checkpoint=checkpoint, retry_cells={"s": [0]})

        # The retry ran the step, but row 0 was skipped by run_if/skip_if.
        session.on_step_complete("s", [{"v": None}, {"v": 1}, {"v": 2}], [], {0})

        assert session.row_errors["s"][0]["msg"] == "boom"
        assert session.has_failed_cells, "an unresolved cell must keep the checkpoint alive"
        assert _saved(tmp_path)["row_errors"]["s"]["0"]["msg"] == "boom"

    def test_a_retried_cell_that_ran_clears_its_error(self, tmp_path):
        checkpoint = CheckpointData(
            timestamp=1.0,
            category="cat",
            total_rows=3,
            fields_dict=FIELDS,
            completed_steps=["s"],
            step_results={"s": [{"v": None}, {"v": 1}, {"v": 2}]},
            row_errors={"s": {0: {"type": "ValueError", "msg": "boom"}}},
        )
        session = _session(tmp_path, checkpoint=checkpoint, retry_cells={"s": [0]})

        session.on_step_complete("s", [{"v": 0}, {"v": 1}, {"v": 2}], [], set())

        assert session.row_errors == {}
        assert not session.has_failed_cells


class TestPartialCheckpointMidRetry:
    def _checkpoint(self):
        return CheckpointData(
            timestamp=1.0,
            category="cat",
            total_rows=3,
            fields_dict=FIELDS,
            completed_steps=["s"],
            step_results={"s": [{"v": None}, {"v": 1}, {"v": 2}]},
            row_errors={"s": {0: {"type": "ValueError", "msg": "boom"}}},
        )

    def test_retried_step_keeps_its_completed_status_and_errors(self, tmp_path):
        """A crash mid-retry used to demote the step to 'never ran'.

        The next retry then re-executed every row of it at full price instead
        of the one cell still failing.
        """
        session = _session(tmp_path, checkpoint=self._checkpoint(), retry_cells={"s": [0]})

        session.on_partial_checkpoint("s", [{"v": None}, {"v": 1}, {"v": 2}], 1)

        saved = _saved(tmp_path)
        assert saved["completed_steps"] == ["s"], "results were seeded — the step is still done"
        assert saved["row_errors"]["s"]["0"]["msg"] == "boom"

        # A fresh retry reading that file re-runs exactly the failed cell.
        resumed = _make_mgr(tmp_path).resume_failed("data", "cat")
        assert resumed is not None
        _, cells = resumed
        assert cells == {"s": [0]}

    def test_a_first_run_step_is_still_marked_unfinished(self, tmp_path):
        """Unchanged for a fresh step: its unrun rows hold {} and must re-run."""
        session = _session(tmp_path)

        session.on_partial_checkpoint("s", [{"v": 0}, {}, {}], 1)

        assert _saved(tmp_path)["completed_steps"] == []


class TestCheckpointFailureDoesNotKillTheRun:
    def test_unserializable_value_degrades_to_no_checkpoint(self, tmp_path, caplog):
        """A UUID/Path/Enum value used to crash the run *after* the API spend."""
        import uuid as _uuid

        from accrue.pipeline.pipeline import Pipeline
        from accrue.steps.function import FunctionStep

        calls: list[int] = []

        def step_one(ctx):
            calls.append(ctx.row["i"])
            return {"token": _uuid.uuid4()}  # not JSON-encodable

        def step_two(ctx):
            return {"label": f"row-{ctx.row['i']}"}

        pipeline = Pipeline(
            [
                FunctionStep("one", step_one, fields=["token"]),
                FunctionStep("two", step_two, fields=["label"], depends_on=["one"]),
            ]
        )
        config = EnrichmentConfig(
            enable_checkpointing=True,
            checkpoint_dir=str(tmp_path),
            enable_caching=False,
            enable_progress_bar=False,
        )

        with caplog.at_level(logging.ERROR, logger="accrue.pipeline.pipeline"):
            result = pipeline.run([{"i": 0}, {"i": 1}], config=config)

        assert len(calls) == 2, "the run completed"
        assert [row["label"] for row in result.data] == ["row-0", "row-1"]
        assert not result.has_errors
        assert "Checkpoint save failed" in caplog.text
        assert list(tmp_path.glob("*_checkpoint.json")) == []

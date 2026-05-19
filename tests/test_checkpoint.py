"""Tests for the per-step CheckpointManager."""

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from accrue.core.checkpoint import CheckpointData, CheckpointManager
from accrue.core.config import EnrichmentConfig

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

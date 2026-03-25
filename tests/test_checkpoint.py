"""Tests for the per-step CheckpointManager."""

from pathlib import Path

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

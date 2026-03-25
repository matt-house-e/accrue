"""Tests for Step protocol, StepContext, and StepResult."""

import pytest

from accrue.schemas.base import UsageInfo
from accrue.steps.base import Step, StepContext, StepResult

# -- StepContext ---------------------------------------------------------


class TestStepContext:
    def test_creation(self):
        ctx = StepContext(
            row={"company": "Acme"},
            fields={"market_size": {"prompt": "Estimate market size"}},
            prior_results={"funding": "$10M"},
        )
        assert ctx.row == {"company": "Acme"}
        assert "market_size" in ctx.fields
        assert ctx.prior_results == {"funding": "$10M"}
        assert ctx.config is None

    def test_frozen(self):
        ctx = StepContext(row={}, fields={}, prior_results={})
        with pytest.raises(AttributeError):
            ctx.row = {"changed": True}

    def test_config_passthrough(self):
        sentinel = object()
        ctx = StepContext(row={}, fields={}, prior_results={}, config=sentinel)
        assert ctx.config is sentinel


# -- StepResult ----------------------------------------------------------


class TestStepResult:
    def test_creation_minimal(self):
        r = StepResult(values={"market_size": "Large"})
        assert r.values == {"market_size": "Large"}
        assert r.usage is None
        assert r.metadata == {}

    def test_creation_full(self):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15, model="gpt-4o")
        r = StepResult(
            values={"a": 1},
            usage=usage,
            metadata={"attempt": 1},
        )
        assert r.usage.total_tokens == 15
        assert r.metadata["attempt"] == 1


# -- Step Protocol -------------------------------------------------------


class _DummyStep:
    """Minimal class that satisfies the Step protocol via duck typing."""

    def __init__(self):
        self.name = "dummy"
        self.fields = ["f1"]
        self.depends_on = []

    async def run(self, ctx: StepContext) -> StepResult:
        return StepResult(values={"f1": "val"})


class TestStepProtocol:
    def test_isinstance_check(self):
        step = _DummyStep()
        assert isinstance(step, Step)

    def test_non_step_rejected(self):
        assert not isinstance("not a step", Step)
        assert not isinstance(42, Step)

    def test_missing_run_rejected(self):
        class NoRun:
            name = "x"
            fields = []
            depends_on = []

        assert not isinstance(NoRun(), Step)

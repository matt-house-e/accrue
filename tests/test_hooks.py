"""Tests for lifecycle hooks (EnrichmentHooks + event dataclasses)."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from accrue.core.hooks import (
    EnrichmentHooks,
    PipelineEndEvent,
    PipelineStartEvent,
    RowCompleteEvent,
    StepEndEvent,
    StepStartEvent,
    _fire_hook,
)
from accrue.pipeline.pipeline import Pipeline
from accrue.steps.base import StepContext, StepResult

# ---------------------------------------------------------------------------
# _fire_hook unit tests
# ---------------------------------------------------------------------------


class TestFireHook:
    @pytest.mark.asyncio
    async def test_none_hook_is_noop(self):
        await _fire_hook(None, "event")  # Should not raise

    @pytest.mark.asyncio
    async def test_sync_callback_called(self):
        mock = MagicMock()
        event = PipelineStartEvent(step_names=["a"], num_rows=10, config=None)
        await _fire_hook(mock, event)
        mock.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self):
        mock = AsyncMock()
        event = PipelineStartEvent(step_names=["a"], num_rows=10, config=None)
        await _fire_hook(mock, event)
        mock.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_exception_caught_and_logged(self, caplog):
        def bad_hook(event):
            raise ValueError("hook error")

        with caplog.at_level(logging.WARNING):
            await _fire_hook(bad_hook, "event")

        assert "raised an exception" in caplog.text

    @pytest.mark.asyncio
    async def test_async_exception_caught(self, caplog):
        async def bad_async_hook(event):
            raise RuntimeError("async hook error")

        with caplog.at_level(logging.WARNING):
            await _fire_hook(bad_async_hook, "event")

        assert "raised an exception" in caplog.text


# ---------------------------------------------------------------------------
# EnrichmentHooks dataclass
# ---------------------------------------------------------------------------


class TestEnrichmentHooks:
    def test_defaults_are_none(self):
        hooks = EnrichmentHooks()
        assert hooks.on_pipeline_start is None
        assert hooks.on_pipeline_end is None
        assert hooks.on_step_start is None
        assert hooks.on_step_end is None
        assert hooks.on_row_complete is None

    def test_can_set_hooks(self):
        fn = lambda e: None
        hooks = EnrichmentHooks(on_pipeline_start=fn)
        assert hooks.on_pipeline_start is fn


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------


class _MockStep:
    """Minimal step for hook testing."""

    def __init__(self, name: str, fields: list[str], depends_on: list[str] | None = None):
        self.name = name
        self.fields = fields
        self.depends_on = depends_on or []
        self.cache = False  # disable caching for simplicity

    async def run(self, ctx: StepContext) -> StepResult:
        return StepResult(values={f: f"val_{f}" for f in self.fields})


class _ErrorStep:
    """Step that always errors."""

    def __init__(self, name: str, fields: list[str], depends_on: list[str] | None = None):
        self.name = name
        self.fields = fields
        self.depends_on = depends_on or []
        self.cache = False

    async def run(self, ctx: StepContext) -> StepResult:
        raise ValueError(f"step {self.name} failed")


# ---------------------------------------------------------------------------
# Integration tests — hooks fired during pipeline execution
# ---------------------------------------------------------------------------


class TestHooksIntegration:
    @pytest.mark.asyncio
    async def test_pipeline_start_fires(self):
        events: list[PipelineStartEvent] = []
        hooks = EnrichmentHooks(on_pipeline_start=lambda e: events.append(e))

        pipeline = Pipeline([_MockStep("s1", fields=["f1"])])
        await pipeline.run_async([{"x": 1}, {"x": 2}], hooks=hooks)

        assert len(events) == 1
        assert events[0].num_rows == 2
        assert events[0].step_names == ["s1"]

    @pytest.mark.asyncio
    async def test_pipeline_end_fires(self):
        events: list[PipelineEndEvent] = []
        hooks = EnrichmentHooks(on_pipeline_end=lambda e: events.append(e))

        pipeline = Pipeline([_MockStep("s1", fields=["f1"])])
        await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert len(events) == 1
        assert events[0].num_rows == 1
        assert events[0].total_errors == 0
        assert events[0].elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_pipeline_end_fires_on_error(self):
        """on_pipeline_end should fire even when a step errors."""
        events: list[PipelineEndEvent] = []
        hooks = EnrichmentHooks(on_pipeline_end=lambda e: events.append(e))

        pipeline = Pipeline([_ErrorStep("s1", fields=["f1"])])
        await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert len(events) == 1
        assert events[0].total_errors == 1

    @pytest.mark.asyncio
    async def test_step_start_and_end_fire(self):
        starts: list[StepStartEvent] = []
        ends: list[StepEndEvent] = []
        hooks = EnrichmentHooks(
            on_step_start=lambda e: starts.append(e),
            on_step_end=lambda e: ends.append(e),
        )

        pipeline = Pipeline(
            [
                _MockStep("s1", fields=["f1"]),
                _MockStep("s2", fields=["f2"], depends_on=["s1"]),
            ]
        )
        await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert len(starts) == 2
        assert starts[0].step_name == "s1"
        assert starts[0].level == 0
        assert starts[1].step_name == "s2"
        assert starts[1].level == 1

        assert len(ends) == 2
        assert ends[0].step_name == "s1"
        assert ends[1].step_name == "s2"
        assert ends[0].elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_row_complete_fires_per_row_per_step(self):
        events: list[RowCompleteEvent] = []
        hooks = EnrichmentHooks(on_row_complete=lambda e: events.append(e))

        pipeline = Pipeline([_MockStep("s1", fields=["f1"])])
        await pipeline.run_async([{"x": 1}, {"x": 2}, {"x": 3}], hooks=hooks)

        assert len(events) == 3
        row_indices = sorted(e.row_index for e in events)
        assert row_indices == [0, 1, 2]
        for e in events:
            assert e.step_name == "s1"
            assert e.error is None
            assert e.from_cache is False
            assert e.values == {"f1": "val_f1"}

    @pytest.mark.asyncio
    async def test_row_complete_with_error(self):
        events: list[RowCompleteEvent] = []
        hooks = EnrichmentHooks(on_row_complete=lambda e: events.append(e))

        pipeline = Pipeline([_ErrorStep("s1", fields=["f1"])])
        await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert len(events) == 1
        assert events[0].error is not None
        assert events[0].from_cache is False

    @pytest.mark.asyncio
    async def test_hook_ordering(self):
        """step_start → row_complete (per row) → step_end."""
        log: list[str] = []
        hooks = EnrichmentHooks(
            on_step_start=lambda e: log.append(f"step_start:{e.step_name}"),
            on_step_end=lambda e: log.append(f"step_end:{e.step_name}"),
            on_row_complete=lambda e: log.append(f"row:{e.step_name}:{e.row_index}"),
        )

        pipeline = Pipeline([_MockStep("s1", fields=["f1"])])
        await pipeline.run_async([{"x": 1}, {"x": 2}], hooks=hooks)

        assert log[0] == "step_start:s1"
        assert log[-1] == "step_end:s1"
        # Row events should be between start and end
        row_events = [e for e in log if e.startswith("row:")]
        assert len(row_events) == 2

    @pytest.mark.asyncio
    async def test_hook_error_does_not_crash_pipeline(self):
        """A hook that raises should not break the pipeline."""

        def bad_hook(event):
            raise RuntimeError("hook exploded")

        hooks = EnrichmentHooks(on_row_complete=bad_hook)
        pipeline = Pipeline([_MockStep("s1", fields=["f1"])])
        result = await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_async_hooks_work(self):
        events: list[str] = []

        async def on_start(e):
            await asyncio.sleep(0)
            events.append("start")

        async def on_end(e):
            await asyncio.sleep(0)
            events.append("end")

        hooks = EnrichmentHooks(on_pipeline_start=on_start, on_pipeline_end=on_end)
        pipeline = Pipeline([_MockStep("s1", fields=["f1"])])
        await pipeline.run_async([{"x": 1}], hooks=hooks)

        assert events == ["start", "end"]

    @pytest.mark.asyncio
    async def test_hooks_with_list_dict_input(self):
        events: list[PipelineStartEvent] = []
        hooks = EnrichmentHooks(on_pipeline_start=lambda e: events.append(e))

        pipeline = Pipeline([_MockStep("s1", fields=["f1"])])
        result = await pipeline.run_async(
            [{"company": "Acme"}, {"company": "Beta"}],
            hooks=hooks,
        )

        assert len(events) == 1
        assert events[0].num_rows == 2
        assert isinstance(result.data, list)

    @pytest.mark.asyncio
    async def test_multi_step_row_events(self):
        """Two steps × 2 rows = 4 row_complete events."""
        events: list[RowCompleteEvent] = []
        hooks = EnrichmentHooks(on_row_complete=lambda e: events.append(e))

        pipeline = Pipeline(
            [
                _MockStep("s1", fields=["f1"]),
                _MockStep("s2", fields=["f2"], depends_on=["s1"]),
            ]
        )
        await pipeline.run_async([{"x": 1}, {"x": 2}], hooks=hooks)

        assert len(events) == 4
        s1_events = [e for e in events if e.step_name == "s1"]
        s2_events = [e for e in events if e.step_name == "s2"]
        assert len(s1_events) == 2
        assert len(s2_events) == 2

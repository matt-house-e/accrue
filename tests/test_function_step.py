"""Tests for FunctionStep."""

import pytest

from accrue.steps.base import Step, StepContext, StepResult
from accrue.steps.function import FunctionStep


def _make_ctx(**overrides):
    defaults = dict(row={"company": "Acme"}, fields={}, prior_results={})
    defaults.update(overrides)
    return StepContext(**defaults)


# -- Construction --------------------------------------------------------


class TestFunctionStepConstruction:
    def test_satisfies_protocol(self):
        step = FunctionStep(name="fn", fn=lambda ctx: {}, fields=["a"])
        assert isinstance(step, Step)

    def test_defaults(self):
        step = FunctionStep(name="fn", fn=lambda ctx: {}, fields=["a"])
        assert step.depends_on == []
        assert step.name == "fn"
        assert step.fields == ["a"]

    def test_depends_on(self):
        step = FunctionStep(name="fn", fn=lambda ctx: {}, fields=["a"], depends_on=["prev"])
        assert step.depends_on == ["prev"]


# -- Sync function execution --------------------------------------------


class TestFunctionStepSync:
    @pytest.mark.asyncio
    async def test_sync_fn(self):
        def my_fn(ctx):
            return {"greeting": f"Hello, {ctx.row['company']}"}

        step = FunctionStep(name="greet", fn=my_fn, fields=["greeting"])
        result = await step.run(_make_ctx())

        assert isinstance(result, StepResult)
        assert result.values == {"greeting": "Hello, Acme"}
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_sync_fn_filters_to_declared_fields(self):
        def my_fn(ctx):
            return {"keep": 1, "drop": 2}

        step = FunctionStep(name="fn", fn=my_fn, fields=["keep"])
        result = await step.run(_make_ctx())

        assert result.values == {"keep": 1}
        assert "drop" not in result.values


# -- Async function execution -------------------------------------------


class TestFunctionStepAsync:
    @pytest.mark.asyncio
    async def test_async_fn(self):
        async def my_fn(ctx):
            return {"result": 42}

        step = FunctionStep(name="fn", fn=my_fn, fields=["result"])
        result = await step.run(_make_ctx())

        assert result.values == {"result": 42}

    @pytest.mark.asyncio
    async def test_async_fn_filters_fields(self):
        async def my_fn(ctx):
            return {"a": 1, "b": 2, "c": 3}

        step = FunctionStep(name="fn", fn=my_fn, fields=["a", "c"])
        result = await step.run(_make_ctx())

        assert result.values == {"a": 1, "c": 3}


# -- Prior results -------------------------------------------------------


class TestFunctionStepPriorResults:
    @pytest.mark.asyncio
    async def test_accesses_prior_results(self):
        def my_fn(ctx):
            prefix = ctx.prior_results.get("prefix", "")
            return {"full_name": f"{prefix} {ctx.row['company']}"}

        step = FunctionStep(name="fn", fn=my_fn, fields=["full_name"], depends_on=["prev"])
        ctx = _make_ctx(prior_results={"prefix": "Inc."})
        result = await step.run(ctx)

        assert result.values == {"full_name": "Inc. Acme"}

"""FunctionStep — wraps any sync or async callable as a pipeline step."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..core.exceptions import PipelineError
from .base import StepContext, StepResult


class FunctionStep:
    """Wraps a user-supplied callable as a pipeline step.

    The callable receives a ``StepContext`` and must return
    ``dict[str, Any]`` mapping field names to values.

    Sync functions are executed via ``run_in_executor`` so they
    never block the event loop.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        fields: list[str],
        depends_on: list[str] | None = None,
        cache: bool = True,
        cache_version: str | None = None,
        run_if: Callable[..., Any] | None = None,
        skip_if: Callable[..., Any] | None = None,
    ):
        """Configure a function-based pipeline step.

        Args:
            name: Unique step name used in logs, cache keys, and
                ``depends_on`` references.
            fn: Sync or async callable that receives a :class:`StepContext`
                and returns ``dict[str, Any]`` mapping field names to values.
            fields: Field names this step produces (``list[str]`` only —
                FunctionStep does not accept dict field specs).
            depends_on: Names of steps whose outputs this step needs.
            cache: Enable input-hash caching for this step (default True).
            cache_version: Bump to invalidate cached results when the
                function logic changes (e.g. ``"v2"``).
            run_if: Predicate ``(row, prior_results) -> bool``.  When set,
                the step only runs for rows where the predicate returns True.
                Mutually exclusive with ``skip_if``.
            skip_if: Predicate ``(row, prior_results) -> bool``.  When set,
                the step is skipped for rows where the predicate returns True.
                Mutually exclusive with ``run_if``.
        """
        if run_if is not None and skip_if is not None:
            raise PipelineError(
                f"Step '{name}' has both run_if and skip_if set. "
                f"These are mutually exclusive — use one or the other."
            )
        self.name = name
        self.fn = fn
        self.fields = fields
        self.depends_on = depends_on or []
        self.cache = cache
        self.cache_version = cache_version
        self.run_if = run_if
        self.skip_if = skip_if
        self._is_async = asyncio.iscoroutinefunction(fn)

    async def run(self, ctx: StepContext) -> StepResult:
        """Execute the wrapped callable and filter output to declared fields.

        Sync functions run via ``run_in_executor`` to avoid blocking the
        event loop.  The returned dict is filtered to only include keys
        listed in ``self.fields``.
        """
        if self._is_async:
            raw = await self.fn(ctx)
        else:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, self.fn, ctx)

        # Filter to declared fields only
        values = {k: v for k, v in raw.items() if k in self.fields}
        return StepResult(values=values)

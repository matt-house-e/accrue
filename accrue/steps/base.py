"""Step protocol and data classes for the Accrue pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from ..schemas.base import UsageInfo

if TYPE_CHECKING:
    from ..core.config import EnrichmentConfig
    from ..core.hooks import RowAttemptEvent


@dataclass(frozen=True)
class StepContext:
    """Immutable context passed to each step's ``run()`` method.

    Attributes:
        row: Original row data as ``dict[str, Any]``.
        fields: Resolved field specs for THIS step only, always
            ``dict[str, dict]`` (even when the step constructor received
            ``list[str]`` — the pipeline resolves specs before slicing).
        prior_results: Merged outputs from all dependency steps for the
            current row.
        config: Optional :class:`EnrichmentConfig` for reading runtime
            settings (temperature, max_tokens, etc.).
        row_index: 0-based index of this row in the input data, or ``None``
            when a step is run outside the pipeline.  Carried so a step can
            label the attempt events it fires (#134).
        capture: Run-log capture tier — ``"metadata"`` (default),
            ``"prompts"``, or ``"full"``.  At ``>= "prompts"`` an LLM step
            captures the rendered prompt/response and attaches it to the
            attempt event it fires (#134).
        on_attempt: Async callback a step calls once per provider attempt
            with a :class:`~accrue.core.hooks.RowAttemptEvent`.  ``None`` when
            nothing is listening (no run log, no ``on_row_attempt`` hook), so
            a metadata run pays nothing.
    """

    row: dict[str, Any]
    fields: dict[str, dict[str, Any]]
    prior_results: dict[str, Any]
    config: EnrichmentConfig | None = None
    row_index: int | None = None
    capture: str = "metadata"
    on_attempt: Callable[[RowAttemptEvent], Awaitable[None]] | None = None


@dataclass
class StepResult:
    """Output from a single step execution.

    Attributes:
        values: Field name -> produced value.
        usage: Token usage (LLM steps only).
        metadata: Arbitrary metadata for logging/debugging.
    """

    values: dict[str, Any]
    usage: Optional[UsageInfo] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Step(Protocol):
    """Protocol all steps must satisfy.

    Async-only to eliminate the sync/async duplication from v0.2.
    Implementations can be plain classes — no inheritance required.

    Attributes:
        name: Unique step identifier.
        fields: Field names this step produces, always ``list[str]`` on
            instances regardless of constructor input form (LLMStep
            normalizes ``dict`` fields to a list of names).
        depends_on: Names of steps whose outputs this step requires.
    """

    name: str
    fields: list[str]
    depends_on: list[str]

    async def run(self, ctx: StepContext) -> StepResult: ...

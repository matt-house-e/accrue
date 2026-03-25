"""
Custom exceptions for the Accrue enrichment tool.

Provides specific exception types for different failure modes
with helpful error messages and context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class EnrichmentError(Exception):
    """Base exception for all enrichment-related errors.

    Attributes:
        message: Human-readable error description.
        row_index: Row that triggered the error (``None`` for non-row errors).
        field: Field name involved (``None`` if not field-specific).
    """

    def __init__(self, message: str, row_index: int | None = None, field: str | None = None):
        self.message = message
        self.row_index = row_index
        self.field = field

        # Build descriptive error message
        error_parts = [message]
        if row_index is not None:
            error_parts.append(f"Row: {row_index}")
        if field is not None:
            error_parts.append(f"Field: {field}")

        super().__init__(" | ".join(error_parts))


class FieldValidationError(EnrichmentError):
    """Raised when field definitions are invalid or missing.

    Common causes:
        - Unknown keys in a field spec (only ``prompt``, ``type``, ``format``,
          ``enum``, ``examples``, ``bad_examples``, ``default`` are allowed).
        - Missing required ``prompt`` key in a dict field spec.
        - Invalid ``type`` or ``enum`` values.
    """

    pass


class ConfigurationError(EnrichmentError):
    """Raised when configuration is invalid."""

    pass


class StepError(EnrichmentError):
    """Raised when a pipeline step fails after exhausting retries.

    Typical causes: all parse/validation retries exhausted, or all API
    retries exhausted for an LLMStep.

    Attributes:
        step_name: Name of the step that failed (``None`` if unknown).
    """

    def __init__(self, message: str, step_name: str | None = None, **kwargs: Any):
        self.step_name = step_name
        super().__init__(message, **kwargs)


class PipelineError(EnrichmentError):
    """Raised when pipeline construction or execution fails.

    Common causes:
        - Duplicate step names.
        - A step's ``depends_on`` references a step that doesn't exist.
        - The dependency graph contains a cycle.
    """

    pass


@dataclass
class RowError:
    """Per-row failure record for partial result tracking."""

    row_index: int
    step_name: str
    error: BaseException
    error_type: str = ""

    def __post_init__(self) -> None:
        if not self.error_type:
            self.error_type = type(self.error).__name__

    def __str__(self) -> str:
        return (
            f"RowError(row={self.row_index}, step='{self.step_name}', "
            f"{self.error_type}: {self.error})"
        )

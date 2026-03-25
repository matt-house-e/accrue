"""FieldSpec — Pydantic model for the 7-key field specification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

VALID_TYPES = Literal["String", "Number", "Boolean", "Date", "List[String]", "JSON"]


class FieldSpec(BaseModel):
    """Strict schema for a single field's specification.

    The 7 supported keys:
      - prompt (required): The extraction instruction for this field.
      - type: Expected data type (default: ``"String"``).
      - format: Output format pattern (e.g. ``"YYYY-MM-DD"``, ``"$X.XB"``).
      - enum: Constrained value list (e.g. ``["Low", "Medium", "High"]``).
      - examples: Good output examples showing expected style.
      - bad_examples: Anti-patterns to avoid.
      - default: Fallback value when data is insufficient (enforced in Python).

    Unknown keys are rejected (``extra="forbid"``).

    Use ``model_fields_set`` to detect whether ``default`` was explicitly provided::

        spec = FieldSpec(prompt="...", default=None)
        has_default = "default" in spec.model_fields_set  # True
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    type: VALID_TYPES = "String"
    format: str | None = None
    enum: list[str] | None = None
    examples: list[str] | None = None
    bad_examples: list[str] | None = None
    default: Any = None

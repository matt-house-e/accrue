"""Dynamic Pydantic model builder for OpenAI structured outputs.

Converts 7-key FieldSpec definitions into a Pydantic model whose
JSON Schema is used with ``response_format={"type": "json_schema", ...}``
and ``strict: true``.  This constrains token generation at the OpenAI API
level, eliminating most parse/validation retries.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, create_model

from ..schemas.field_spec import FieldSpec

# FieldSpec.type → Python annotation
_TYPE_MAP: dict[str, type] = {
    "String": str,
    "Number": float,
    "Boolean": bool,
    "Date": str,
    "List[String]": list[str],
    "JSON": dict[str, Any],
}


def build_response_model(
    field_specs: dict[str, FieldSpec],
) -> type[BaseModel]:
    """Dynamically create a Pydantic model from field specifications.

    The generated model:
      - Has one field per spec, typed according to ``spec.type``.
      - Enum specs produce ``Literal["val1", "val2", ...]`` types.
      - Uses ``extra="forbid"`` → ``additionalProperties: false`` in JSON Schema.
      - Field descriptions combine prompt, format, and examples for
        better schema-level guidance to the model.

    Returns:
        A Pydantic ``BaseModel`` subclass.
    """
    field_definitions: dict[str, Any] = {}

    from pydantic import Field

    for name, spec in field_specs.items():
        # Skip __ internal fields — Pydantic forbids leading underscores.
        # These fields are still included in the prompt and parsed from the
        # raw JSON response, but they bypass schema validation.
        if name.startswith("__"):
            continue
        annotation = _resolve_type(spec)
        description = _build_description(spec)
        field_definitions[name] = (annotation, Field(description=description))

    model = create_model(
        "EnrichmentResponse",
        __config__=ConfigDict(extra="forbid"),
        **field_definitions,
    )
    return model


def build_json_schema(field_specs: dict[str, FieldSpec]) -> dict[str, Any]:
    """Build the ``response_format`` dict for OpenAI structured outputs.

    Returns:
        ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}``
    """
    model = build_response_model(field_specs)
    schema = model.model_json_schema()

    # Ensure additionalProperties: false at top level for strict mode
    schema["additionalProperties"] = False

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "enrichment_result",
            "schema": schema,
            "strict": True,
        },
    }


def _resolve_type(spec: FieldSpec) -> type:
    """Map a FieldSpec to a Python type annotation.

    Enum fields → ``Literal["val1", "val2", ...]``.
    """
    if spec.enum is not None and len(spec.enum) > 0:
        return Literal[tuple(spec.enum)]  # type: ignore[valid-type]

    return _TYPE_MAP.get(spec.type, str)


def _build_description(spec: FieldSpec) -> str:
    """Combine prompt, format, and examples into a field description."""
    parts = [spec.prompt]

    if spec.format is not None:
        parts.append(f"Format: {spec.format}")

    if spec.examples is not None:
        examples_str = "; ".join(spec.examples)
        parts.append(f"Examples: {examples_str}")

    return ". ".join(parts)

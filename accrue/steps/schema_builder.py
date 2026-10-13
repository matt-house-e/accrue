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


def build_json_schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    """Build a strict ``response_format`` dict from a user-supplied Pydantic model.

    Strict structured outputs require ``additionalProperties: false`` on *every*
    object in the schema, not just the top level — nested models land in
    ``$defs`` and would otherwise be rejected by the provider.

    Args:
        model: Any ``BaseModel`` subclass (a custom ``schema=`` on an LLMStep).

    Returns:
        ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}``
    """
    schema = model.model_json_schema()
    _strictify(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": schema,
            "strict": True,
        },
    }


def describe_from_specs(
    model: type[BaseModel],
    field_specs: dict[str, FieldSpec | str | dict[str, Any]],
) -> type[BaseModel]:
    """Return ``model`` with each field spec's prompt written into its property description.

    With ``LLMStep(schema=..., structured_outputs=True)`` the constrained decoder
    reads the JSON schema ``description`` next to each value, and a custom schema
    does not get the spec text automatically.  This is the opt-in fix: every
    top-level field whose name matches a spec gets the same description
    ``build_json_schema`` would have written (prompt, format, examples).

    A field that already carries a description containing the spec's prompt is left
    alone; otherwise its description is set (an existing unrelated description is
    replaced).  The input model is not modified: a subclass with the same name is
    returned.  Names with no matching field are ignored.
    """
    from pydantic.fields import FieldInfo

    overrides: dict[str, Any] = {}
    for name, raw in field_specs.items():
        info = model.model_fields.get(name)
        if info is None:
            continue
        spec = raw if isinstance(raw, FieldSpec) else (
            FieldSpec(prompt=raw) if isinstance(raw, str) else FieldSpec.model_validate(raw)
        )
        if info.description and spec.prompt in info.description:
            continue
        merged = FieldInfo.merge_field_infos(info, description=_build_description(spec))
        overrides[name] = (info.annotation, merged)

    if not overrides:
        return model
    return create_model(model.__name__, __base__=model, __module__=model.__module__, **overrides)


def undescribed_fields(
    model: type[BaseModel],
    field_specs: dict[str, FieldSpec],
) -> list[str]:
    """Top-level properties of ``model`` matching a spec whose schema description
    is missing or does not contain the spec's prompt."""
    properties = model.model_json_schema().get("properties", {})
    missing: list[str] = []
    for name, spec in field_specs.items():
        prop = properties.get(name)
        if prop is None:
            continue
        description = prop.get("description") or ""
        if spec.prompt not in description:
            missing.append(name)
    return missing


def _strictify(node: Any) -> None:
    """Recursively set ``additionalProperties: false`` on every object node."""
    if isinstance(node, list):
        for item in node:
            _strictify(item)
        return
    if not isinstance(node, dict):
        return

    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False

    for key in ("properties", "$defs", "definitions", "patternProperties"):
        sub = node.get(key)
        if isinstance(sub, dict):
            for value in sub.values():
                _strictify(value)

    for key in ("items", "prefixItems", "anyOf", "oneOf", "allOf", "additionalItems", "not"):
        if key in node:
            _strictify(node[key])


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

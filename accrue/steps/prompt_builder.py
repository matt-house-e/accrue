"""Dynamic system prompt builder for LLMStep.

Follows the OpenAI GPT-4.1 cookbook pattern:
  - Markdown headers (``#``) for instruction sections
  - XML tags for data boundaries (``<row_data>``, ``<field_specifications>``)
  - Static content at top (enables OpenAI prompt caching)
  - Variable content at bottom (row data, field specs, prior results)
  - Sandwich pattern: key constraints reiterated after dynamic content

Only describes field specification keys that are actually used across the
step's fields — avoids confusing GPT-4.1's literal instruction-following
with irrelevant sections.
"""

from __future__ import annotations

import json
from typing import Any

from ..schemas.field_spec import FieldSpec


def build_system_message(
    field_specs: dict[str, FieldSpec],
    row: dict[str, Any],
    prior_results: dict[str, Any] | None = None,
    custom_system_prompt: str | None = None,
    system_prompt_header: str | None = None,
) -> str:
    """Build the full system message for an LLM enrichment call.

    Args:
        field_specs: Mapping of field name → validated FieldSpec.
        row: The input row data.
        prior_results: Merged outputs from dependency steps (or None).
        custom_system_prompt: If provided, replaces the auto-generated
            instruction portion (Role + Keys + Output Rules). Data sections
            are still appended. When set, ``system_prompt_header`` is ignored.
        system_prompt_header: Domain context injected as a ``# Context``
            section between Role and Field Specification Keys. Ignored when
            ``custom_system_prompt`` is set.

    Returns:
        Complete system message string.
    """
    field_names = list(field_specs.keys())

    if custom_system_prompt is not None:
        instructions = custom_system_prompt
    else:
        instructions = _build_instructions(field_specs, field_names, system_prompt_header)

    data = _build_data_section(field_specs, row, prior_results)
    reminder = _build_reminder(field_names)

    return f"{instructions}\n\n{data}\n\n{reminder}"


# ---------------------------------------------------------------------------
# Instruction portion (static — cacheable by OpenAI)
# ---------------------------------------------------------------------------


def _build_instructions(
    field_specs: dict[str, FieldSpec],
    field_names: list[str],
    system_prompt_header: str | None = None,
) -> str:
    """Build the Role + Context + Field Specification Keys + Output Rules sections."""
    parts: list[str] = []

    # -- Role
    parts.append(
        "# Role\n"
        "You are a structured data enrichment engine. Given one input row, "
        "a set of field specifications, and optional context from prior "
        "processing steps, produce a JSON object with exactly the requested "
        "fields as keys."
    )

    # -- Context (system_prompt_header)
    if system_prompt_header:
        parts.append(f"# Context\n{system_prompt_header}")

    # -- Field Specification Keys (dynamic: only describe keys actually used)
    keys_section = _build_keys_section(field_specs)
    if keys_section:
        parts.append(keys_section)

    # -- Output Rules
    parts.append(_build_output_rules(field_specs, field_names))

    return "\n\n".join(parts)


def _build_keys_section(field_specs: dict[str, FieldSpec]) -> str:
    """Describe only the field spec keys that appear across all fields."""
    used = _detect_used_keys(field_specs)

    lines = ["# Field Specification Keys"]
    lines.append("Each field below is described using these keys:")
    # prompt is always present
    lines.append("- **prompt**: the extraction instruction for this field")

    if "type" in used:
        lines.append(
            "- **type**: expected output data type "
            "(String, Number, Boolean, Date, List[String], JSON)"
        )
    if "format" in used:
        lines.append("- **format**: output format pattern to follow")
    if "enum" in used:
        lines.append(
            "- **enum**: constrained value list — the output MUST be one of these options exactly"
        )
    if "examples" in used:
        lines.append("- **examples**: good output examples showing expected style")
    if "bad_examples" in used:
        lines.append("- **bad_examples**: anti-patterns to avoid")
    if "default" in used:
        lines.append("- **default**: last-resort fallback if you truly cannot determine the value")

    return "\n".join(lines)


def _build_output_rules(
    field_specs: dict[str, FieldSpec],
    field_names: list[str],
) -> str:
    """Build the Output Rules section with conditional rules."""
    used = _detect_used_keys(field_specs)
    names_str = ", ".join(field_names)

    lines = ["# Output Rules"]
    lines.append(
        "- Return ONLY a single valid JSON object. No prose, no code fences, no explanations."
    )
    lines.append(f"- Top-level keys MUST be exactly: {names_str}")
    lines.append("- Keep outputs concise and information-dense.")

    if "enum" in used:
        lines.append(
            "- For enum fields, the value MUST match one of the listed "
            "options exactly. Do not paraphrase or combine options."
        )
    if "format" in used:
        lines.append("- For fields with a format, follow the specified pattern precisely.")
    lines.append(
        "- Always attempt to determine a value using the row data, "
        "prior results, and your general knowledge."
    )
    if "default" in used:
        lines.append(
            "- Only if you genuinely cannot determine a value after "
            "exhausting all available information, you may use the "
            "field's default value as a last resort."
        )
    else:
        lines.append(
            "- If you genuinely cannot determine a value, return "
            '"Unable to determine" for String fields, null for other types.'
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data section (variable — row-specific)
# ---------------------------------------------------------------------------


def _build_data_section(
    field_specs: dict[str, FieldSpec],
    row: dict[str, Any],
    prior_results: dict[str, Any] | None,
) -> str:
    """Build the XML-delimited data: row, field specs, prior results."""
    parts: list[str] = []

    # Row data
    parts.append(f"<row_data>\n{json.dumps(row, default=str)}\n</row_data>")

    # Field specifications as XML
    parts.append(_build_field_specs_xml(field_specs))

    # Prior results (only if present)
    if prior_results:
        parts.append(f"<prior_results>\n{json.dumps(prior_results, default=str)}\n</prior_results>")

    return "\n\n".join(parts)


def _build_field_specs_xml(field_specs: dict[str, FieldSpec]) -> str:
    """Build ``<field_specifications>`` XML block.

    Each field gets a ``<field name="...">`` block containing only the
    keys that are actually defined (no empty tags).
    """
    lines = ["<field_specifications>"]

    for name, spec in field_specs.items():
        lines.append(f'<field name="{name}">')
        lines.append(f"  <prompt>{spec.prompt}</prompt>")

        if spec.type != "String":
            lines.append(f"  <type>{spec.type}</type>")
        if spec.format is not None:
            lines.append(f"  <format>{spec.format}</format>")
        if spec.enum is not None:
            lines.append(f"  <enum>{', '.join(spec.enum)}</enum>")
        if spec.examples is not None:
            for ex in spec.examples:
                lines.append(f"  <example>{ex}</example>")
        if spec.bad_examples is not None:
            for ex in spec.bad_examples:
                lines.append(f"  <bad_example>{ex}</bad_example>")
        if "default" in spec.model_fields_set:
            lines.append(f"  <default>{spec.default}</default>")

        lines.append("</field>")

    lines.append("</field_specifications>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reminder (sandwich pattern — key constraints after dynamic content)
# ---------------------------------------------------------------------------


def _build_reminder(field_names: list[str]) -> str:
    """Final reminder reiterating key constraints after all dynamic content."""
    names_str = ", ".join(field_names)
    return f"# Reminder\nReturn ONLY the JSON object with keys: {names_str}. No additional text."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_used_keys(field_specs: dict[str, FieldSpec]) -> set[str]:
    """Detect which optional field spec keys are used across all fields."""
    used: set[str] = set()
    for spec in field_specs.values():
        if spec.type != "String":
            used.add("type")
        if spec.format is not None:
            used.add("format")
        if spec.enum is not None:
            used.add("enum")
        if spec.examples is not None:
            used.add("examples")
        if spec.bad_examples is not None:
            used.add("bad_examples")
        if "default" in spec.model_fields_set:
            used.add("default")
    return used

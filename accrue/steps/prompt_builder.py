"""Dynamic prompt builder for LLMStep.

Follows the OpenAI GPT-4.1 cookbook pattern:
  - Markdown headers (``#``) for instruction sections
  - XML tags for data boundaries (``<row_data>``, ``<field_specifications>``)
  - Static content first (enables prompt caching)
  - Variable content last (row data, prior results)
  - Sandwich pattern: key constraints reiterated after dynamic content

Only describes field specification keys that are actually used across the
step's fields — avoids confusing GPT-4.1's literal instruction-following
with irrelevant sections.

**The static/variable boundary is load-bearing, not cosmetic.**
:func:`build_prompt` returns the prompt already split in two:

  - :attr:`PromptParts.system` is derived only from step configuration, so it
    is byte-identical for every row of a given step. Providers cache on a
    prefix match (Anthropic explicitly via ``cache_control``, OpenAI
    automatically), and a prefix that changes per row can never be re-read —
    it is only ever re-written, at 1.25x base input rates.
  - :attr:`PromptParts.user` carries everything row-specific.

Keep row data out of ``system``. Anything moved across that line silently
turns prompt caching from a discount into a surcharge.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

from ..schemas.field_spec import FieldSpec

#: Task instruction closing the user message, before the final reminder.
TASK_INSTRUCTION = "Analyze the provided data and return the requested fields as JSON."


class PromptParts(NamedTuple):
    """A prompt split at the static/variable boundary.

    Attributes:
        system: Static instructions and field specifications. Identical for
            every row of a given step — this is the cacheable prefix.
        user: Row data, prior results, and the closing reminder. Differs per
            row.
    """

    system: str
    user: str


def build_prompt(
    field_specs: dict[str, FieldSpec],
    row: dict[str, Any],
    prior_results: dict[str, Any] | None = None,
    custom_system_prompt: str | None = None,
    system_prompt_header: str | None = None,
) -> PromptParts:
    """Build the system and user messages for an LLM enrichment call.

    The two halves are returned together so the static/variable split cannot
    be applied by accident or half-applied — see the module docstring.

    Args:
        field_specs: Mapping of field name → validated FieldSpec.
        row: The input row data.
        prior_results: Merged outputs from dependency steps (or None).
        custom_system_prompt: If provided, replaces the auto-generated
            instruction portion (Role + Keys + Output Rules). Field
            specifications are still appended. When set,
            ``system_prompt_header`` is ignored.
        system_prompt_header: Domain context injected as a ``# Context``
            section between Role and Field Specification Keys. Ignored when
            ``custom_system_prompt`` is set.

    Returns:
        :class:`PromptParts` with the cacheable ``system`` half and the
        row-specific ``user`` half.
    """
    field_names = list(field_specs.keys())

    if custom_system_prompt is not None:
        instructions = custom_system_prompt
    else:
        instructions = _build_instructions(field_specs, field_names, system_prompt_header)

    # Static half: instructions + field specs. Both derive from step config
    # alone, so this string is identical across every row of the step.
    system = f"{instructions}\n\n{_build_field_specs_xml(field_specs)}"

    # Variable half: row data, prior results, then the task line and the
    # reminder. The reminder stays last (sandwich pattern) — it is one line,
    # so leaving it uncached costs nothing, while hoisting it into the cached
    # prefix would put row data after it and weaken the constraint.
    user_parts = [f"<row_data>\n{json.dumps(row, default=str)}\n</row_data>"]
    if prior_results:
        user_parts.append(
            f"<prior_results>\n{json.dumps(prior_results, default=str)}\n</prior_results>"
        )
    user_parts.append(TASK_INSTRUCTION)
    user_parts.append(_build_reminder(field_names))

    return PromptParts(system=system, user="\n\n".join(user_parts))


# ---------------------------------------------------------------------------
# Instruction portion (static — cacheable)
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
# Field specifications (static — derived from step config, never from the row)
# ---------------------------------------------------------------------------


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

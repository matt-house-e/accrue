"""Tests for the dynamic system prompt builder."""

from __future__ import annotations

from accrue.schemas.field_spec import FieldSpec
from accrue.steps.prompt_builder import (
    _build_field_specs_xml,
    _detect_used_keys,
    build_system_message,
)

# -- key detection -------------------------------------------------------


class TestDetectUsedKeys:
    def test_prompt_only(self):
        specs = {"f1": FieldSpec(prompt="test")}
        assert _detect_used_keys(specs) == set()

    def test_enum_detected(self):
        specs = {"f1": FieldSpec(prompt="test", enum=["A", "B"])}
        assert "enum" in _detect_used_keys(specs)

    def test_format_detected(self):
        specs = {"f1": FieldSpec(prompt="test", format="$X.XB")}
        assert "format" in _detect_used_keys(specs)

    def test_examples_detected(self):
        specs = {"f1": FieldSpec(prompt="test", examples=["ex1"])}
        assert "examples" in _detect_used_keys(specs)

    def test_bad_examples_detected(self):
        specs = {"f1": FieldSpec(prompt="test", bad_examples=["bad"])}
        assert "bad_examples" in _detect_used_keys(specs)

    def test_default_detected(self):
        specs = {"f1": FieldSpec(prompt="test", default="N/A")}
        assert "default" in _detect_used_keys(specs)

    def test_non_string_type_detected(self):
        specs = {"f1": FieldSpec(prompt="test", type="Number")}
        assert "type" in _detect_used_keys(specs)

    def test_string_type_not_detected(self):
        """Default type 'String' doesn't trigger key description."""
        specs = {"f1": FieldSpec(prompt="test", type="String")}
        assert "type" not in _detect_used_keys(specs)

    def test_mixed_fields(self):
        specs = {
            "f1": FieldSpec(prompt="test"),
            "f2": FieldSpec(prompt="test", enum=["A"], format="$X"),
        }
        used = _detect_used_keys(specs)
        assert "enum" in used
        assert "format" in used
        assert "examples" not in used


# -- XML field specs -----------------------------------------------------


class TestFieldSpecsXML:
    def test_prompt_only_field(self):
        specs = {"market_size": FieldSpec(prompt="Estimate TAM")}
        xml = _build_field_specs_xml(specs)
        assert '<field name="market_size">' in xml
        assert "<prompt>Estimate TAM</prompt>" in xml
        assert "</field>" in xml
        # No type tag for default String
        assert "<type>" not in xml

    def test_non_string_type_included(self):
        specs = {"revenue": FieldSpec(prompt="Estimate", type="Number")}
        xml = _build_field_specs_xml(specs)
        assert "<type>Number</type>" in xml

    def test_enum_field(self):
        specs = {"risk": FieldSpec(prompt="Rate risk", enum=["Low", "Medium", "High"])}
        xml = _build_field_specs_xml(specs)
        assert "<enum>Low, Medium, High</enum>" in xml

    def test_examples_field(self):
        specs = {"f1": FieldSpec(prompt="test", examples=["ex1", "ex2"])}
        xml = _build_field_specs_xml(specs)
        assert "<example>ex1</example>" in xml
        assert "<example>ex2</example>" in xml

    def test_bad_examples_field(self):
        specs = {"f1": FieldSpec(prompt="test", bad_examples=["bad1"])}
        xml = _build_field_specs_xml(specs)
        assert "<bad_example>bad1</bad_example>" in xml

    def test_default_field(self):
        specs = {"f1": FieldSpec(prompt="test", default="Unknown")}
        xml = _build_field_specs_xml(specs)
        assert "<default>Unknown</default>" in xml

    def test_format_field(self):
        specs = {"f1": FieldSpec(prompt="test", format="$X.XB")}
        xml = _build_field_specs_xml(specs)
        assert "<format>$X.XB</format>" in xml


# -- full system message -------------------------------------------------


class TestBuildSystemMessage:
    def test_basic_structure(self):
        specs = {"market_size": FieldSpec(prompt="Estimate TAM")}
        row = {"company": "Acme"}
        msg = build_system_message(specs, row)

        # Has all main sections
        assert "# Role" in msg
        assert "# Field Specification Keys" in msg
        assert "# Output Rules" in msg
        assert "<row_data>" in msg
        assert "<field_specifications>" in msg
        assert "# Reminder" in msg

    def test_row_data_in_xml(self):
        specs = {"f1": FieldSpec(prompt="test")}
        row = {"company": "Acme", "industry": "Tech"}
        msg = build_system_message(specs, row)

        assert "<row_data>" in msg
        assert '"company": "Acme"' in msg
        assert "</row_data>" in msg

    def test_field_names_in_output_rules(self):
        specs = {
            "market_size": FieldSpec(prompt="Estimate TAM"),
            "risk": FieldSpec(prompt="Rate risk"),
        }
        msg = build_system_message(specs, {"x": 1})

        assert "market_size, risk" in msg

    def test_prior_results_included_when_present(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1}, prior_results={"context": "data"})

        assert "<prior_results>" in msg
        assert "context" in msg

    def test_prior_results_omitted_when_empty(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1}, prior_results=None)
        assert "<prior_results>" not in msg

    def test_prior_results_omitted_when_empty_dict(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1}, prior_results={})
        assert "<prior_results>" not in msg

    def test_enum_rules_in_output_section(self):
        specs = {"f1": FieldSpec(prompt="test", enum=["A", "B"])}
        msg = build_system_message(specs, {"x": 1})

        assert "MUST match one of the listed options" in msg

    def test_default_rules_in_output_section(self):
        specs = {"f1": FieldSpec(prompt="test", default="N/A")}
        msg = build_system_message(specs, {"x": 1})

        assert "default value" in msg

    def test_no_default_fallback_rule(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1})

        assert "Unable to determine" in msg

    def test_custom_system_prompt_override(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1}, custom_system_prompt="Custom instructions.")

        assert msg.startswith("Custom instructions.")
        # Data sections still appended
        assert "<row_data>" in msg
        assert "<field_specifications>" in msg
        # Dynamic instructions NOT present
        assert "# Role" not in msg

    def test_sandwich_pattern(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1})

        # Reminder at the end reiterates field names
        assert msg.strip().endswith("No additional text.")
        last_section = msg.split("# Reminder")[-1]
        assert "f1" in last_section

    def test_only_used_keys_described(self):
        """Keys not used by any field should not appear in key descriptions."""
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1})

        # Only prompt should be described (type=String is default, not described)
        instructions = msg.split("<row_data>")[0]
        assert "**prompt**" in instructions
        assert "**enum**" not in instructions
        assert "**format**" not in instructions
        assert "**examples**" not in instructions
        assert "**bad_examples**" not in instructions
        assert "**default**" not in instructions


# -- system_prompt_header ------------------------------------------------


class TestSystemPromptHeader:
    def test_header_injected_between_role_and_keys(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(
            specs, {"x": 1}, system_prompt_header="Analyzing B2B SaaS companies."
        )

        assert "# Context" in msg
        assert "Analyzing B2B SaaS companies." in msg

        # Verify ordering: Role < Context < Keys
        role_pos = msg.index("# Role")
        context_pos = msg.index("# Context")
        keys_pos = msg.index("# Field Specification Keys")
        assert role_pos < context_pos < keys_pos

    def test_header_omitted_when_none(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1}, system_prompt_header=None)
        assert "# Context" not in msg

    def test_header_omitted_when_empty(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(specs, {"x": 1}, system_prompt_header="")
        assert "# Context" not in msg

    def test_header_ignored_when_custom_system_prompt_set(self):
        specs = {"f1": FieldSpec(prompt="test")}
        msg = build_system_message(
            specs,
            {"x": 1},
            custom_system_prompt="Custom prompt.",
            system_prompt_header="Should be ignored.",
        )

        assert "# Context" not in msg
        assert "Should be ignored." not in msg
        assert msg.startswith("Custom prompt.")

    def test_header_multiline(self):
        specs = {"f1": FieldSpec(prompt="test")}
        header = "Line one.\nLine two.\nLine three."
        msg = build_system_message(specs, {"x": 1}, system_prompt_header=header)

        assert "# Context\nLine one.\nLine two.\nLine three." in msg

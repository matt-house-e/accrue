"""Tests for the dynamic prompt builder."""

from __future__ import annotations

from accrue.schemas.field_spec import FieldSpec
from accrue.steps.prompt_builder import (
    TASK_INSTRUCTION,
    _build_field_specs_xml,
    _detect_used_keys,
    build_prompt,
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


# -- full prompt ---------------------------------------------------------


class TestBuildPrompt:
    def test_basic_structure(self):
        specs = {"market_size": FieldSpec(prompt="Estimate TAM")}
        row = {"company": "Acme"}
        parts = build_prompt(specs, row)

        # Static half carries the instructions and the field specs
        assert "# Role" in parts.system
        assert "# Field Specification Keys" in parts.system
        assert "# Output Rules" in parts.system
        assert "<field_specifications>" in parts.system

        # Variable half carries the row and the closing reminder
        assert "<row_data>" in parts.user
        assert "# Reminder" in parts.user

    def test_row_data_in_xml(self):
        specs = {"f1": FieldSpec(prompt="test")}
        row = {"company": "Acme", "industry": "Tech"}
        parts = build_prompt(specs, row)

        assert "<row_data>" in parts.user
        assert '"company": "Acme"' in parts.user
        assert "</row_data>" in parts.user

    def test_task_instruction_in_user_message(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1})

        assert TASK_INSTRUCTION in parts.user
        assert TASK_INSTRUCTION not in parts.system

    def test_field_names_in_output_rules(self):
        specs = {
            "market_size": FieldSpec(prompt="Estimate TAM"),
            "risk": FieldSpec(prompt="Rate risk"),
        }
        parts = build_prompt(specs, {"x": 1})

        assert "market_size, risk" in parts.system

    def test_prior_results_included_when_present(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1}, prior_results={"context": "data"})

        assert "<prior_results>" in parts.user
        assert "context" in parts.user

    def test_prior_results_omitted_when_empty(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1}, prior_results=None)
        assert "<prior_results>" not in parts.user

    def test_prior_results_omitted_when_empty_dict(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1}, prior_results={})
        assert "<prior_results>" not in parts.user

    def test_enum_rules_in_output_section(self):
        specs = {"f1": FieldSpec(prompt="test", enum=["A", "B"])}
        parts = build_prompt(specs, {"x": 1})

        assert "MUST match one of the listed options" in parts.system

    def test_default_rules_in_output_section(self):
        specs = {"f1": FieldSpec(prompt="test", default="N/A")}
        parts = build_prompt(specs, {"x": 1})

        assert "default value" in parts.system

    def test_no_default_fallback_rule(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1})

        assert "Unable to determine" in parts.system

    def test_custom_system_prompt_override(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1}, custom_system_prompt="Custom instructions.")

        assert parts.system.startswith("Custom instructions.")
        # Field specs still appended to the static half
        assert "<field_specifications>" in parts.system
        # Row data still travels in the user half
        assert "<row_data>" in parts.user
        # Dynamic instructions NOT present
        assert "# Role" not in parts.system

    def test_sandwich_pattern(self):
        """The reminder is the last thing the model reads, after the row data."""
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1})

        assert parts.user.strip().endswith("No additional text.")
        last_section = parts.user.split("# Reminder")[-1]
        assert "f1" in last_section
        # Reminder trails the row data and the task instruction
        assert parts.user.index("<row_data>") < parts.user.index("# Reminder")
        assert parts.user.index(TASK_INSTRUCTION) < parts.user.index("# Reminder")

    def test_only_used_keys_described(self):
        """Keys not used by any field should not appear in key descriptions."""
        specs = {"f1": FieldSpec(prompt="test")}
        instructions = build_prompt(specs, {"x": 1}).system.split("<field_specifications>")[0]

        # Only prompt should be described (type=String is default, not described)
        assert "**prompt**" in instructions
        assert "**enum**" not in instructions
        assert "**format**" not in instructions
        assert "**examples**" not in instructions
        assert "**bad_examples**" not in instructions
        assert "**default**" not in instructions


# -- cacheable prefix (issue #107) ---------------------------------------


class TestCacheablePrefix:
    """The system half must be byte-identical across rows.

    Providers cache on a prefix match.  If any row-specific content leaks into
    the system message the prefix changes every call, so every call writes a
    fresh cache entry (billed at 1.25x input) and none ever reads one — the
    caching "optimisation" becomes a permanent surcharge.
    """

    SPECS = {
        "market_size": FieldSpec(
            prompt="Estimate TAM in billions USD",
            format="$X.XB",
            examples=["$4.2B"],
            bad_examples=["a lot"],
        ),
        "risk": FieldSpec(prompt="Rate risk", enum=["Low", "High"], default="Low"),
    }
    HEADER = "You are analysing European B2B SaaS companies.\nUse 2024 figures."

    def _build(self, row, prior_results=None):
        return build_prompt(
            self.SPECS,
            row,
            prior_results=prior_results,
            system_prompt_header=self.HEADER,
        )

    def test_system_byte_identical_across_rows(self):
        a = self._build({"company": "Acme", "hq": "Berlin"})
        b = self._build({"company": "Globex", "hq": "Lisbon", "extra": 42})

        assert a.system == b.system

    def test_user_differs_across_rows(self):
        a = self._build({"company": "Acme"})
        b = self._build({"company": "Globex"})

        assert a.user != b.user
        assert "Acme" in a.user
        assert "Globex" in b.user

    def test_system_stable_across_differing_prior_results(self):
        a = self._build({"company": "Acme"}, prior_results={"search": "one"})
        b = self._build({"company": "Acme"}, prior_results={"search": "two"})

        assert a.system == b.system
        assert a.user != b.user

    def test_system_stable_whether_or_not_prior_results_exist(self):
        a = self._build({"company": "Acme"})
        b = self._build({"company": "Acme"}, prior_results={"search": "ctx"})

        assert a.system == b.system

    def test_field_specifications_stay_in_the_cached_prefix(self):
        """The trap: field specs are static but used to sit *after* the row data.

        They carry every prompt, format, enum, example and bad_example, so on a
        rich step they are the largest static block.  A naive "split at
        ``<row_data>``" evicts them from the cache.
        """
        parts = self._build({"company": "Acme"})

        assert "<field_specifications>" in parts.system
        assert "<field_specifications>" not in parts.user
        for fragment in (
            "Estimate TAM in billions USD",
            "<format>$X.XB</format>",
            "<example>$4.2B</example>",
            "<bad_example>a lot</bad_example>",
            "<enum>Low, High</enum>",
        ):
            assert fragment in parts.system

    def test_reminder_stays_out_of_the_cached_prefix(self):
        """One line, so it costs nothing uncached — and it must stay last."""
        parts = self._build({"company": "Acme"})

        assert "# Reminder" in parts.user
        assert "# Reminder" not in parts.system

    def test_no_row_content_leaks_into_system(self):
        parts = self._build({"company": "Acme", "hq": "Berlin"}, prior_results={"s": "ctx"})

        assert "<row_data>" not in parts.system
        assert "<prior_results>" not in parts.system
        assert "Acme" not in parts.system
        assert "Berlin" not in parts.system
        assert "ctx" not in parts.system

    def test_system_changes_when_step_config_changes(self):
        """The prefix is stable per config, not globally — caching must still
        invalidate when the step itself changes."""
        base = self._build({"company": "Acme"})
        different_header = build_prompt(
            self.SPECS, {"company": "Acme"}, system_prompt_header="A different header."
        )
        different_fields = build_prompt(
            {"market_size": FieldSpec(prompt="Estimate TAM in billions USD")},
            {"company": "Acme"},
            system_prompt_header=self.HEADER,
        )

        assert base.system != different_header.system
        assert base.system != different_fields.system

    def test_custom_system_prompt_prefix_also_stable(self):
        a = build_prompt(self.SPECS, {"company": "Acme"}, custom_system_prompt="Do the thing.")
        b = build_prompt(self.SPECS, {"company": "Globex"}, custom_system_prompt="Do the thing.")

        assert a.system == b.system
        assert a.user != b.user


# -- system_prompt_header ------------------------------------------------


class TestSystemPromptHeader:
    def test_header_injected_between_role_and_keys(self):
        specs = {"f1": FieldSpec(prompt="test")}
        system = build_prompt(
            specs, {"x": 1}, system_prompt_header="Analyzing B2B SaaS companies."
        ).system

        assert "# Context" in system
        assert "Analyzing B2B SaaS companies." in system

        # Verify ordering: Role < Context < Keys
        role_pos = system.index("# Role")
        context_pos = system.index("# Context")
        keys_pos = system.index("# Field Specification Keys")
        assert role_pos < context_pos < keys_pos

    def test_header_omitted_when_none(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1}, system_prompt_header=None)
        assert "# Context" not in parts.system

    def test_header_omitted_when_empty(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(specs, {"x": 1}, system_prompt_header="")
        assert "# Context" not in parts.system

    def test_header_ignored_when_custom_system_prompt_set(self):
        specs = {"f1": FieldSpec(prompt="test")}
        parts = build_prompt(
            specs,
            {"x": 1},
            custom_system_prompt="Custom prompt.",
            system_prompt_header="Should be ignored.",
        )

        assert "# Context" not in parts.system
        assert "Should be ignored." not in parts.system
        assert parts.system.startswith("Custom prompt.")

    def test_header_multiline(self):
        specs = {"f1": FieldSpec(prompt="test")}
        header = "Line one.\nLine two.\nLine three."
        parts = build_prompt(specs, {"x": 1}, system_prompt_header=header)

        assert "# Context\nLine one.\nLine two.\nLine three." in parts.system

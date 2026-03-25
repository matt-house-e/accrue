"""Tests for FieldSpec — the 7-key field specification model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from accrue.schemas.field_spec import FieldSpec


class TestFieldSpecValid:
    def test_prompt_only(self):
        spec = FieldSpec(prompt="Estimate TAM")
        assert spec.prompt == "Estimate TAM"
        assert spec.type == "String"
        assert spec.format is None
        assert spec.enum is None
        assert spec.examples is None
        assert spec.bad_examples is None
        assert spec.default is None
        assert "default" not in spec.model_fields_set

    def test_all_keys(self):
        spec = FieldSpec(
            prompt="Assess risk",
            type="String",
            format="X/10",
            enum=["Low", "Medium", "High"],
            examples=["High", "Medium"],
            bad_examples=["Moderately high", "3/5"],
            default="Unknown",
        )
        assert spec.prompt == "Assess risk"
        assert spec.type == "String"
        assert spec.format == "X/10"
        assert spec.enum == ["Low", "Medium", "High"]
        assert spec.examples == ["High", "Medium"]
        assert spec.bad_examples == ["Moderately high", "3/5"]
        assert spec.default == "Unknown"

    def test_valid_types(self):
        for t in ("String", "Number", "Boolean", "Date", "List[String]", "JSON"):
            spec = FieldSpec(prompt="test", type=t)
            assert spec.type == t

    def test_model_validate_from_dict(self):
        spec = FieldSpec.model_validate(
            {
                "prompt": "test",
                "enum": ["A", "B"],
            }
        )
        assert spec.prompt == "test"
        assert spec.enum == ["A", "B"]


class TestFieldSpecDefault:
    def test_default_not_set(self):
        spec = FieldSpec(prompt="test")
        assert "default" not in spec.model_fields_set

    def test_default_explicitly_none(self):
        spec = FieldSpec(prompt="test", default=None)
        assert "default" in spec.model_fields_set
        assert spec.default is None

    def test_default_string(self):
        spec = FieldSpec(prompt="test", default="Unknown")
        assert "default" in spec.model_fields_set
        assert spec.default == "Unknown"

    def test_default_numeric(self):
        spec = FieldSpec(prompt="test", type="Number", default=0)
        assert spec.default == 0


class TestFieldSpecRejection:
    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            FieldSpec(prompt="test", instructions="extra")

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            FieldSpec(prompt="test", type="InvalidType")

    def test_prompt_required(self):
        with pytest.raises(ValidationError, match="prompt"):
            FieldSpec(type="String")  # type: ignore[call-arg]

    def test_enum_must_be_list(self):
        with pytest.raises(ValidationError):
            FieldSpec(prompt="test", enum="not a list")  # type: ignore[arg-type]

    def test_examples_must_be_list(self):
        with pytest.raises(ValidationError):
            FieldSpec(prompt="test", examples="not a list")  # type: ignore[arg-type]


class TestFieldSpecSerialization:
    def test_model_dump_exclude_none(self):
        spec = FieldSpec(prompt="test", enum=["A", "B"])
        d = spec.model_dump(exclude_none=True)
        assert d == {"prompt": "test", "type": "String", "enum": ["A", "B"]}
        assert "format" not in d
        assert "examples" not in d

    def test_model_dump_full(self):
        spec = FieldSpec(prompt="test")
        d = spec.model_dump()
        assert d["prompt"] == "test"
        assert d["format"] is None

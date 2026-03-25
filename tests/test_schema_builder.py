"""Tests for the dynamic schema builder (structured outputs)."""

from __future__ import annotations

from typing import get_args, get_origin

import pytest
from pydantic import ValidationError

from accrue.schemas.field_spec import FieldSpec
from accrue.steps.schema_builder import (
    _build_description,
    _resolve_type,
    build_json_schema,
    build_response_model,
)

# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


class TestResolveType:
    def test_string(self):
        assert _resolve_type(FieldSpec(prompt="test", type="String")) is str

    def test_number(self):
        assert _resolve_type(FieldSpec(prompt="test", type="Number")) is float

    def test_boolean(self):
        assert _resolve_type(FieldSpec(prompt="test", type="Boolean")) is bool

    def test_date(self):
        assert _resolve_type(FieldSpec(prompt="test", type="Date")) is str

    def test_list_string(self):
        t = _resolve_type(FieldSpec(prompt="test", type="List[String]"))
        assert get_origin(t) is list
        assert get_args(t) == (str,)

    def test_json(self):
        t = _resolve_type(FieldSpec(prompt="test", type="JSON"))
        assert get_origin(t) is dict

    def test_enum_becomes_literal(self):
        t = _resolve_type(FieldSpec(prompt="test", enum=["Low", "Medium", "High"]))
        assert get_args(t) == ("Low", "Medium", "High")


# ---------------------------------------------------------------------------
# Description builder
# ---------------------------------------------------------------------------


class TestBuildDescription:
    def test_prompt_only(self):
        desc = _build_description(FieldSpec(prompt="Estimate TAM"))
        assert desc == "Estimate TAM"

    def test_with_format(self):
        desc = _build_description(FieldSpec(prompt="Estimate TAM", format="$X.XB"))
        assert "Format: $X.XB" in desc

    def test_with_examples(self):
        desc = _build_description(FieldSpec(prompt="test", examples=["ex1", "ex2"]))
        assert "Examples: ex1; ex2" in desc

    def test_combined(self):
        desc = _build_description(
            FieldSpec(
                prompt="Estimate revenue",
                format="$X.XM",
                examples=["$2.5M", "$10.0M"],
            )
        )
        assert "Estimate revenue" in desc
        assert "Format: $X.XM" in desc
        assert "Examples: $2.5M; $10.0M" in desc


# ---------------------------------------------------------------------------
# build_response_model
# ---------------------------------------------------------------------------


class TestBuildResponseModel:
    def test_basic_string_fields(self):
        specs = {
            "market_size": FieldSpec(prompt="Estimate TAM"),
            "risk": FieldSpec(prompt="Rate risk"),
        }
        model = build_response_model(specs)

        # Can validate correct data
        instance = model.model_validate({"market_size": "Large", "risk": "High"})
        assert instance.market_size == "Large"
        assert instance.risk == "High"

    def test_number_field(self):
        specs = {"revenue": FieldSpec(prompt="Estimate", type="Number")}
        model = build_response_model(specs)

        instance = model.model_validate({"revenue": 42.5})
        assert instance.revenue == 42.5

    def test_boolean_field(self):
        specs = {"active": FieldSpec(prompt="Is active?", type="Boolean")}
        model = build_response_model(specs)

        instance = model.model_validate({"active": True})
        assert instance.active is True

    def test_list_string_field(self):
        specs = {"tags": FieldSpec(prompt="List tags", type="List[String]")}
        model = build_response_model(specs)

        instance = model.model_validate({"tags": ["a", "b"]})
        assert instance.tags == ["a", "b"]

    def test_json_field(self):
        specs = {"meta": FieldSpec(prompt="Metadata", type="JSON")}
        model = build_response_model(specs)

        instance = model.model_validate({"meta": {"k": "v"}})
        assert instance.meta == {"k": "v"}

    def test_enum_field_validates(self):
        specs = {"risk": FieldSpec(prompt="Rate risk", enum=["Low", "Medium", "High"])}
        model = build_response_model(specs)

        instance = model.model_validate({"risk": "High"})
        assert instance.risk == "High"

    def test_enum_field_rejects_invalid(self):
        specs = {"risk": FieldSpec(prompt="Rate risk", enum=["Low", "Medium", "High"])}
        model = build_response_model(specs)

        with pytest.raises(ValidationError):
            model.model_validate({"risk": "Critical"})

    def test_extra_fields_rejected(self):
        specs = {"f1": FieldSpec(prompt="test")}
        model = build_response_model(specs)

        with pytest.raises(ValidationError):
            model.model_validate({"f1": "ok", "extra": "bad"})

    def test_wrong_type_rejected(self):
        specs = {"score": FieldSpec(prompt="Score", type="Number")}
        model = build_response_model(specs)

        with pytest.raises(ValidationError):
            model.model_validate({"score": "not a number"})

    def test_model_dump(self):
        specs = {
            "f1": FieldSpec(prompt="test"),
            "f2": FieldSpec(prompt="test2", type="Number"),
        }
        model = build_response_model(specs)
        instance = model.model_validate({"f1": "hello", "f2": 3.14})
        dumped = instance.model_dump()
        assert dumped == {"f1": "hello", "f2": 3.14}

    def test_field_descriptions(self):
        specs = {"f1": FieldSpec(prompt="Estimate TAM", format="$X.XB")}
        model = build_response_model(specs)
        schema = model.model_json_schema()
        desc = schema["properties"]["f1"].get("description", "")
        assert "Estimate TAM" in desc
        assert "Format: $X.XB" in desc


# ---------------------------------------------------------------------------
# build_json_schema
# ---------------------------------------------------------------------------


class TestBuildJsonSchema:
    def test_structure(self):
        specs = {"f1": FieldSpec(prompt="test")}
        result = build_json_schema(specs)

        assert result["type"] == "json_schema"
        js = result["json_schema"]
        assert js["name"] == "enrichment_result"
        assert js["strict"] is True
        assert "schema" in js

    def test_additional_properties_false(self):
        specs = {"f1": FieldSpec(prompt="test")}
        result = build_json_schema(specs)
        schema = result["json_schema"]["schema"]
        assert schema["additionalProperties"] is False

    def test_required_fields(self):
        specs = {
            "f1": FieldSpec(prompt="test"),
            "f2": FieldSpec(prompt="test2"),
        }
        result = build_json_schema(specs)
        schema = result["json_schema"]["schema"]
        assert set(schema["required"]) == {"f1", "f2"}

    def test_enum_in_schema(self):
        specs = {"risk": FieldSpec(prompt="Rate risk", enum=["Low", "High"])}
        result = build_json_schema(specs)
        schema = result["json_schema"]["schema"]
        risk_prop = schema["properties"]["risk"]
        assert "enum" in risk_prop
        assert set(risk_prop["enum"]) == {"Low", "High"}

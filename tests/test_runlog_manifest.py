"""Tests for the run-log pipeline manifest (issue #138).

The manifest is a nested, additive object on the ``pipeline_start`` record
describing the run's *definition* — steps + types + models, the enrichment-
field schema, and the run config — for a dashboard's read-only Overview.
Additive to run-log schema v1: ``SCHEMA_VERSION`` stays 1.

These cover manifest introspection directly (:func:`build_manifest`) and its
appearance on a real run's ``pipeline_start`` record, with no network — the
LLM steps are never executed, only introspected.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import accrue
from accrue import EnrichmentConfig, FunctionStep, Pipeline
from accrue.core.manifest import build_manifest
from accrue.core.runlog import SCHEMA_VERSION
from accrue.steps.llm import LLMStep

# ---------------------------------------------------------------------------
# Fixtures — pipelines built (never run) for introspection
# ---------------------------------------------------------------------------


class _Custom(BaseModel):
    """A custom response schema with a real ``int`` annotation (no FieldSpec has one)."""

    headcount: int = Field(description="Number of employees.")
    tier: Literal["A", "B", "C"] = Field(description="Named tier.")
    ratio: float = Field(description="A ratio.")
    active: bool = Field(description="Whether active.")


def _mixed_pipeline() -> Pipeline:
    """An LLMStep with inline field specs, a custom-schema LLMStep, and a FunctionStep."""
    return Pipeline(
        [
            LLMStep(
                "classify",
                fields={
                    "category": "One short industry category.",
                    "icp_fit": {"prompt": "Fit as a customer.", "enum": ["strong", "good", "weak"]},
                },
                model="gpt-4.1-mini",
                temperature=0.0,
                max_tokens=512,
            ),
            LLMStep(
                "route",
                fields=["headcount", "tier", "ratio", "active"],
                schema=_Custom,
                model="claude-3-5-haiku",
                depends_on=["classify"],
            ),
            FunctionStep(
                "finalize",
                lambda ctx: {"done": True, "__scratch": 1},
                fields=["done", "__scratch"],
                depends_on=["route"],
            ),
        ]
    )


def _steps_by_name(manifest: dict) -> dict[str, dict]:
    return {s["name"]: s for s in manifest["steps"]}


def _fields_by_name(manifest: dict) -> dict[str, dict]:
    return {f["name"]: f for f in manifest["fields"]}


# ---------------------------------------------------------------------------
# build_manifest — structure & versioning
# ---------------------------------------------------------------------------


class TestManifestShape:
    def test_top_level_keys(self):
        m = build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata")
        assert set(m.keys()) == {"accrue_version", "config", "steps", "fields"}

    def test_accrue_version_is_the_package_version(self):
        m = build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata")
        assert m["accrue_version"] == accrue.__version__

    def test_schema_version_unchanged(self):
        # The manifest is additive — the run-log schema version must not bump.
        assert SCHEMA_VERSION == 1

    def test_none_pipeline_still_renders_config(self):
        m = build_manifest(None, EnrichmentConfig(max_workers=4), "metadata")
        assert m["steps"] == []
        assert m["fields"] == []
        assert m["config"]["max_workers"] == 4


# ---------------------------------------------------------------------------
# Steps — types, models, produces, depends_on
# ---------------------------------------------------------------------------


class TestManifestSteps:
    def test_step_types(self):
        steps = _steps_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert steps["classify"]["type"] == "LLMStep"
        assert steps["route"]["type"] == "LLMStep"
        assert steps["finalize"]["type"] == "FunctionStep"

    def test_llmstep_model_params_are_the_declared_values(self):
        steps = _steps_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        model = steps["classify"]["model"]
        assert model == {
            "id": "gpt-4.1-mini",
            "provider": "openai",
            "temperature": 0.0,
            "max_tokens": 512,
        }

    def test_llmstep_model_params_fall_back_to_config(self):
        # ``route`` leaves temperature / max_tokens unset — the manifest reports
        # the effective value the runtime would send (the config fallback).
        cfg = EnrichmentConfig(temperature=0.7, max_tokens=1234)
        steps = _steps_by_name(build_manifest(_mixed_pipeline(), cfg, "metadata"))
        model = steps["route"]["model"]
        assert model["temperature"] == 0.7
        assert model["max_tokens"] == 1234

    def test_functionstep_has_no_model(self):
        steps = _steps_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert steps["finalize"]["model"] is None

    def test_produces_and_depends_on(self):
        steps = _steps_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert steps["classify"]["produces"] == ["category", "icp_fit"]
        assert steps["classify"]["depends_on"] == []
        assert steps["route"]["depends_on"] == ["classify"]
        assert steps["finalize"]["produces"] == ["done", "__scratch"]

    def test_condition_is_null_for_predicate_gated_steps(self):
        # run_if is a lambda — not introspectable to an expression string.
        p = Pipeline(
            [
                FunctionStep(
                    "gate",
                    lambda ctx: {"x": 1},
                    fields=["x"],
                    run_if=lambda row, prior: True,
                )
            ]
        )
        steps = _steps_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        assert steps["gate"]["condition"] is None

    @pytest.mark.parametrize(
        "model, base_url, expected",
        [
            ("gpt-4.1-mini", None, "openai"),
            ("claude-3-5-haiku", None, "anthropic"),
            ("gemini-2.5-flash", None, "google"),
            ("google/gemini-3.5-flash-lite", "https://openrouter.ai/api/v1", "openrouter"),
            ("llama-3", "https://api.groq.com/openai/v1", "groq"),
        ],
    )
    def test_provider_inference(self, model, base_url, expected):
        p = Pipeline([LLMStep("s", fields=["a"], model=model, base_url=base_url)])
        steps = _steps_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        assert steps["s"]["model"]["provider"] == expected


# ---------------------------------------------------------------------------
# System prompt — the row-independent cached prefix per LLM step (#140)
# ---------------------------------------------------------------------------


class _LeakySystemStep(LLMStep):
    """An LLMStep that (wrongly) folds row data into its ``system`` half.

    This is the exact #107 caching bug the manifest guard exists to surface:
    the cached prefix is supposed to be byte-identical for every row.
    """

    def _build_prompt(self, ctx):
        base = super()._build_prompt(ctx)
        return base._replace(system=base.system + f"\n<leak>{ctx.row!r}</leak>")


class TestManifestSystemPrompt:
    def test_llmstep_carries_the_row_independent_system_half(self):
        steps = _steps_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        sp = steps["classify"]["system_prompt"]
        assert isinstance(sp, str)
        # Instruction + field-spec content live in the system (cacheable) half...
        assert "structured data enrichment engine" in sp
        assert "category" in sp
        assert "strong" in sp  # the icp_fit enum option
        # ...and the row-specific user half does not.
        assert "<row_data>" not in sp

    def test_system_prompt_matches_the_step_accessor(self):
        p = _mixed_pipeline()
        classify = p.get_step("classify")
        steps = _steps_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        # The manifest value is exactly the step's row-independent system half.
        assert steps["classify"]["system_prompt"] == classify.row_independent_system()

    def test_functionstep_has_no_system_prompt(self):
        steps = _steps_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert steps["finalize"]["system_prompt"] is None

    def test_custom_system_prompt_is_used(self):
        p = Pipeline(
            [LLMStep("s", fields=["a"], model="gpt-4.1-mini", system_prompt="BESPOKE-INSTRUCTION")]
        )
        steps = _steps_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        assert "BESPOKE-INSTRUCTION" in steps["s"]["system_prompt"]

    def test_row_leaking_step_nulls_and_warns(self):
        # A step whose system half depends on the row leaks its cached prefix
        # (#107). The manifest must NOT embed per-row content: it nulls + warns.
        p = Pipeline([_LeakySystemStep("leaky", fields={"x": "do x"}, model="gpt-4.1-mini")])
        with pytest.warns(UserWarning, match="row-independent"):
            steps = _steps_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        assert steps["leaky"]["system_prompt"] is None

    def test_row_independent_step_does_not_warn(self):
        # The happy path must stay silent — no spurious caching warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            steps = _steps_by_name(
                build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata")
            )
        assert steps["classify"]["system_prompt"] is not None

    def test_secret_in_system_prompt_is_redacted(self):
        secret = "sk-" + "A" * 40
        p = Pipeline(
            [
                LLMStep(
                    "s",
                    fields=["a"],
                    model="gpt-4.1-mini",
                    system_prompt=f"Authenticate with {secret} before answering.",
                )
            ]
        )
        steps = _steps_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        sp = steps["s"]["system_prompt"]
        assert secret not in sp
        assert "***REDACTED***" in sp

    def test_building_the_system_half_makes_no_provider_call(self):
        # No API key set, no network — pure prompt assembly at run start.
        p = Pipeline([LLMStep("s", fields={"a": "describe a"}, model="gpt-4.1-mini")])
        steps = _steps_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        assert isinstance(steps["s"]["system_prompt"], str)


# ---------------------------------------------------------------------------
# Fields — type / enum / description / step / internal
# ---------------------------------------------------------------------------


class TestManifestFields:
    def test_string_fieldspec_field(self):
        fields = _fields_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        cat = fields["category"]
        assert cat["type"] == "str"
        assert cat["enum"] is None
        assert cat["description"] == "One short industry category."
        assert cat["step"] == "classify"
        assert cat["internal"] is False

    def test_enum_fieldspec_field(self):
        fields = _fields_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        icp = fields["icp_fit"]
        assert icp["type"] == "enum"
        assert icp["enum"] == ["strong", "good", "weak"]

    def test_int_field_from_custom_schema(self):
        fields = _fields_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert fields["headcount"]["type"] == "int"
        assert fields["headcount"]["description"] == "Number of employees."

    def test_literal_field_from_custom_schema_is_enum(self):
        fields = _fields_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        tier = fields["tier"]
        assert tier["type"] == "enum"
        assert tier["enum"] == ["A", "B", "C"]

    def test_float_and_bool_fields_from_custom_schema(self):
        fields = _fields_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert fields["ratio"]["type"] == "float"
        assert fields["active"]["type"] == "bool"

    def test_internal_fields_flagged(self):
        fields = _fields_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert fields["__scratch"]["internal"] is True
        assert fields["done"]["internal"] is False

    def test_functionstep_fields_are_unknown(self):
        # A FunctionStep has no response model to introspect.
        fields = _fields_by_name(build_manifest(_mixed_pipeline(), EnrichmentConfig(), "metadata"))
        assert fields["done"]["type"] == "unknown"

    def test_llmstep_list_fields_on_default_schema_are_unknown(self):
        # list fields + the permissive default EnrichmentResult → no per-field type.
        p = Pipeline([LLMStep("s", fields=["note"], model="gpt-4.1-mini")])
        fields = _fields_by_name(build_manifest(p, EnrichmentConfig(), "metadata"))
        assert fields["note"]["type"] == "unknown"


class _Sourced(BaseModel):
    value: str
    source: str


class _SourcedSchema(BaseModel):
    """amana's shape (#170): every field a nested ``{value, source}`` object."""

    term: _Sourced
    notice: _Sourced | None = None
    clauses: list[_Sourced]
    extra: dict[str, str]
    owner: str
    either: int | str
    status: str


class TestManifestCustomSchemaTypes:
    """A custom ``schema=`` owns the field type; the field spec keeps enum + description (#170)."""

    @staticmethod
    def _fields() -> dict[str, dict]:
        step = LLMStep(
            "extract",
            fields={
                "term": "Contract term.",
                "notice": "Notice period.",
                "clauses": "Key clauses.",
                "extra": "Anything else.",
                "owner": "Contract owner.",
                "either": {"prompt": "Either.", "type": "Number"},
                "status": {"prompt": "Status.", "enum": ["active", "expired"]},
            },
            schema=_SourcedSchema,
            model="gpt-4.1-mini",
        )
        return _fields_by_name(build_manifest(Pipeline([step]), EnrichmentConfig(), "metadata"))

    def test_nested_model_is_object_not_the_spec_string(self):
        term = self._fields()["term"]
        assert term["type"] == "object"
        assert term["enum"] is None
        assert term["description"] == "Contract term."

    def test_optional_nested_model_is_object(self):
        assert self._fields()["notice"]["type"] == "object"

    def test_list_of_models_is_list(self):
        assert self._fields()["clauses"]["type"] == "list"

    def test_dict_is_object(self):
        assert self._fields()["extra"]["type"] == "object"

    def test_scalar_annotation_still_maps(self):
        assert self._fields()["owner"]["type"] == "str"

    def test_unknown_annotation_falls_back_to_the_spec_type(self):
        # A multi-arm Union says less than the spec does.
        assert self._fields()["either"]["type"] == "float"

    def test_spec_enum_kept_on_a_str_annotation(self):
        status = self._fields()["status"]
        assert status["type"] == "enum"
        assert status["enum"] == ["active", "expired"]
        assert status["description"] == "Status."

    def test_schema_only_nested_model_uses_the_pydantic_description(self):
        class _Described(BaseModel):
            term: _Sourced = Field(description="Term with its source.")

        step = LLMStep("s", fields=["term"], schema=_Described, model="gpt-4.1-mini")
        fields = _fields_by_name(build_manifest(Pipeline([step]), EnrichmentConfig(), "metadata"))
        assert fields["term"]["type"] == "object"
        assert fields["term"]["description"] == "Term with its source."

    def test_plain_fieldspec_step_is_unchanged(self):
        # No custom schema → exactly what the manifest reported before #170.
        step = LLMStep(
            "classify",
            fields={
                "category": "One short industry category.",
                "icp_fit": {"prompt": "Fit as a customer.", "enum": ["strong", "good", "weak"]},
                "employees": {"prompt": "Headcount.", "type": "Number"},
                "public": {"prompt": "Listed?", "type": "Boolean"},
                "tags": {"prompt": "Tags.", "type": "List[String]"},
                "meta": {"prompt": "Extra.", "type": "JSON"},
                "founded": {"prompt": "Founding date.", "type": "Date"},
            },
            model="gpt-4.1-mini",
        )
        manifest = build_manifest(Pipeline([step]), EnrichmentConfig(), "metadata")
        expected = [
            ("category", "str", None, "One short industry category."),
            ("icp_fit", "enum", ["strong", "good", "weak"], "Fit as a customer."),
            ("employees", "float", None, "Headcount."),
            ("public", "bool", None, "Listed?"),
            ("tags", "list", None, "Tags."),
            ("meta", "json", None, "Extra."),
            ("founded", "str", None, "Founding date."),
        ]
        assert manifest["fields"] == [
            {
                "name": name,
                "type": type_str,
                "enum": field_enum,
                "description": description,
                "step": "classify",
                "internal": False,
            }
            for name, type_str, field_enum, description in expected
        ]


# ---------------------------------------------------------------------------
# Config — mirrors EnrichmentConfig + capture
# ---------------------------------------------------------------------------


class TestManifestConfig:
    def test_config_mirrors_enrichment_config(self):
        cfg = EnrichmentConfig(max_workers=6, enable_caching=False, enable_checkpointing=True)
        m = build_manifest(_mixed_pipeline(), cfg, "prompts")
        assert m["config"] == {
            "max_workers": 6,
            "caching": False,
            "checkpointing": True,
            "batch": False,
            "capture": "prompts",
        }

    def test_batch_true_when_any_step_opts_in(self):
        p = Pipeline(
            [
                LLMStep("a", fields=["x"], model="gpt-4.1-mini"),
                LLMStep("b", fields=["y"], model="gpt-4.1-mini", batch=True, depends_on=["a"]),
            ]
        )
        m = build_manifest(p, EnrichmentConfig(), "metadata")
        assert m["config"]["batch"] is True

    def test_capture_tier_recorded(self):
        m = build_manifest(_mixed_pipeline(), EnrichmentConfig(), "full")
        assert m["config"]["capture"] == "full"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestManifestDeterminism:
    def test_repeated_builds_are_identical(self):
        p = _mixed_pipeline()
        cfg = EnrichmentConfig()
        a = build_manifest(p, cfg, "metadata")
        b = build_manifest(p, cfg, "metadata")
        assert a == b
        # No timestamps / rng leaked in — JSON-serialisable and stable.
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# End-to-end — manifest on a real run's pipeline_start
# ---------------------------------------------------------------------------


class TestManifestEndToEnd:
    def _run(self, tmp_path: Path) -> dict:
        pipeline = Pipeline(
            [
                FunctionStep(
                    "upper",
                    lambda ctx: {"upper": ctx.row["name"].upper()},
                    fields=["upper"],
                )
            ]
        )
        log = tmp_path / "run.jsonl"
        config = EnrichmentConfig(enable_caching=False, enable_progress_bar=False, max_workers=3)
        pipeline.run([{"name": "acme"}], config=config, run_log=str(log))
        return json.loads(log.read_text(encoding="utf-8").splitlines()[0])

    def test_manifest_present_on_pipeline_start(self, tmp_path):
        start = self._run(tmp_path)
        assert start["type"] == "pipeline_start"
        manifest = start["manifest"]
        assert manifest["accrue_version"] == accrue.__version__
        assert [s["name"] for s in manifest["steps"]] == ["upper"]

    def test_config_reflects_the_run(self, tmp_path):
        start = self._run(tmp_path)
        assert start["manifest"]["config"]["max_workers"] == 3
        assert start["manifest"]["config"]["caching"] is False

"""Pipeline manifest — the run's *definition*, introspected at run start.

The manifest is an additive, nested object on the run-log ``pipeline_start``
record (issue #138).  It answers "what is this pipeline?" for a dashboard's
read-only Overview: the steps and their types, the model and params per step,
the enrichment-field schema (types / enums / descriptions / producing step),
and the run configuration.

It is built **once, at run start**, from the real
:class:`~accrue.pipeline.pipeline.Pipeline` / :class:`~accrue.steps.base.Step`
and response-model objects — never per row, so it costs nothing on the hot
path and cannot leak per-row content into the cached ``system`` prompt half
(#107).  Every value is introspected deterministically (no timestamps, no
rng), so regenerating a fixture is byte-stable.

Introspection is defensive by contract: where a source can't be introspected
cheaply the field falls back to ``"unknown"`` (fields) or ``null`` (model /
condition) rather than raising — a manifest must never take a run down or
fabricate a value it cannot read.  Schema stays run-log **v1** (additive).
"""

from __future__ import annotations

import enum as _enum
import types as _types
import typing
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..core.config import EnrichmentConfig
    from ..pipeline.pipeline import Pipeline

# FieldSpec.type (the 7-key spec) → the manifest's short type name.  FieldSpec
# has no ``int`` — ``Number`` is float; an ``int`` field only ever arrives via
# a custom Pydantic ``schema=`` and is picked up by annotation introspection.
_FIELDSPEC_TYPE: dict[str, str] = {
    "String": "str",
    "Number": "float",
    "Boolean": "bool",
    "Date": "str",
    "List[String]": "list",
    "JSON": "json",
}


def build_manifest(
    pipeline: Pipeline | None,
    config: EnrichmentConfig | None,
    capture: str,
) -> dict[str, Any]:
    """Introspect *pipeline* / *config* into the ``pipeline_start.manifest`` object.

    Returns the nested manifest dict (``accrue_version``, ``config``, ``steps``,
    ``fields``).  With ``pipeline is None`` the ``steps`` / ``fields`` arrays are
    empty — the manifest still renders its version and config.
    """
    # Lazy import: ``accrue/__init__`` imports this package, so importing the
    # version at module load would be circular.  By run start it is available.
    try:
        from .. import __version__ as accrue_version
    except Exception:  # pragma: no cover — defensive
        accrue_version = None

    return {
        "accrue_version": accrue_version,
        "config": _config_section(pipeline, config, capture),
        "steps": _step_entries(pipeline, config),
        "fields": _field_entries(pipeline),
    }


# -- config -----------------------------------------------------------------


def _config_section(
    pipeline: Pipeline | None,
    config: EnrichmentConfig | None,
    capture: str,
) -> dict[str, Any]:
    """The run configuration a dashboard shows: mirrors ``EnrichmentConfig`` + capture."""
    # There is no config-level batch flag — batch is declared per step. Report
    # whether *any* step opts into batch, i.e. whether this is a batch run.
    batch = any(getattr(step, "batch", False) for step in _iter_steps(pipeline))
    return {
        "max_workers": getattr(config, "max_workers", None),
        "caching": getattr(config, "enable_caching", None),
        "checkpointing": getattr(config, "enable_checkpointing", None),
        "batch": batch,
        # ``capture`` is a run() argument, not an EnrichmentConfig field.
        "capture": capture,
    }


# -- steps ------------------------------------------------------------------


def _step_entries(
    pipeline: Pipeline | None,
    config: EnrichmentConfig | None,
) -> list[dict[str, Any]]:
    """One entry per step, in execution (topological) order."""
    entries: list[dict[str, Any]] = []
    for step in _iter_steps(pipeline):
        # A step may expose a literal ``condition`` string (a future
        # ConditionalStep); ``run_if`` / ``skip_if`` predicates are lambdas
        # and cannot be introspected to an expression string cheaply, so a
        # predicate-gated step reports ``condition: null``.
        condition = getattr(step, "condition", None)
        if not isinstance(condition, str):
            condition = None
        entries.append(
            {
                "name": getattr(step, "name", None),
                "type": type(step).__name__,
                "model": _model_info(step, config),
                "produces": list(getattr(step, "fields", []) or []),
                "depends_on": list(getattr(step, "depends_on", []) or []),
                "condition": condition,
            }
        )
    return entries


def _model_info(step: Any, config: EnrichmentConfig | None) -> dict[str, Any] | None:
    """Model id / provider / effective temperature+max_tokens for an LLM step.

    ``None`` for steps that expose no model (FunctionSteps and any Step that
    has no ``model`` attribute).  ``temperature`` / ``max_tokens`` report the
    *effective* value the runtime would send: the step's own value, or the
    config fallback that :meth:`LLMStep.build_messages` applies when the step
    leaves it unset — so the Overview shows what actually goes to the provider.
    """
    model_id = getattr(step, "model", None)
    if not isinstance(model_id, str) or not model_id:
        return None

    temperature = getattr(step, "temperature", None)
    if temperature is None:
        temperature = getattr(config, "temperature", None)
    max_tokens = getattr(step, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(config, "max_tokens", None)

    return {
        "id": model_id,
        "provider": _provider(step),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _provider(step: Any) -> str:
    """Provider label, mirroring ``LLMStep._resolve_client`` selection.

    A ``base_url`` means an OpenAI-compatible gateway — the host names the
    provider (e.g. ``https://openrouter.ai/api/v1`` → ``"openrouter"``).
    Otherwise the model-name prefix decides, as in the client resolver.
    """
    base_url = getattr(step, "base_url", None)
    if base_url:
        return _provider_from_base_url(base_url)
    model = getattr(step, "model", "") or ""
    if isinstance(model, str) and model.startswith("claude-"):
        return "anthropic"
    if isinstance(model, str) and model.startswith("gemini-"):
        return "google"
    return "openai"


def _provider_from_base_url(base_url: str) -> str:
    """Short provider label from a base URL's host — ``openrouter.ai`` → ``openrouter``.

    Heuristic: strip a ``www.`` / ``api.`` prefix and take the second-level
    domain label.  Falls back to the raw host when it has no dotted form.
    """
    host = urlparse(base_url).hostname or base_url
    host = host.removeprefix("www.").removeprefix("api.")
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host


# -- fields -----------------------------------------------------------------


def _field_entries(pipeline: Pipeline | None) -> list[dict[str, Any]]:
    """The enrichment-field schema — one entry per field, first producer wins.

    Fields are visited in execution order; a name seen twice (two steps
    declaring it) keeps its first producing step, so the schema has one row
    per field name.
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in _iter_steps(pipeline):
        for field_name in getattr(step, "fields", []) or []:
            if field_name in seen:
                continue
            seen.add(field_name)
            type_str, field_enum, description = _introspect_field(step, field_name)
            entries.append(
                {
                    "name": field_name,
                    "type": type_str,
                    "enum": field_enum,
                    "description": description,
                    "step": getattr(step, "name", None),
                    # ``__``-prefixed fields are internal inter-step data,
                    # filtered from output but carried between steps.
                    "internal": field_name.startswith("__"),
                }
            )
    return entries


def _introspect_field(step: Any, field_name: str) -> tuple[str, list[str] | None, str | None]:
    """Resolve ``(type, enum, description)`` for one field of a step.

    Sources, in order of richness:

    1. An inline :class:`FieldSpec` (LLMStep ``dict`` fields) — the cleanest
       source: its declared type, enum, and prompt.
    2. A custom Pydantic response model (``schema=MyModel``) — field annotation
       (``str`` / ``int`` / ``float`` / ``bool`` / ``Literal`` / enum / ``list``)
       and the Pydantic field description.
    3. Nothing introspectable (FunctionStep fields, ``__`` inter-step fields,
       or LLMStep ``list`` fields on the permissive default ``EnrichmentResult``
       schema) → ``("unknown", None, None)``.

    Any introspection error degrades to ``"unknown"`` rather than raising.
    """
    # 1. Inline FieldSpec.
    field_specs = getattr(step, "_field_specs", None)
    if isinstance(field_specs, dict) and field_name in field_specs:
        spec = field_specs[field_name]
        try:
            spec_enum = getattr(spec, "enum", None)
            description = getattr(spec, "prompt", None)
            if spec_enum:
                return "enum", list(spec_enum), description
            type_str = _FIELDSPEC_TYPE.get(getattr(spec, "type", None), "unknown")
            return type_str, None, description
        except Exception:  # pragma: no cover — defensive
            return "unknown", None, None

    # 2. Custom Pydantic response model.  The default ``EnrichmentResult`` is
    #    permissive (extra="allow", no declared fields), so its model_fields is
    #    empty and this branch is skipped for it.
    model_fields = getattr(getattr(step, "schema", None), "model_fields", None)
    if isinstance(model_fields, dict) and field_name in model_fields:
        try:
            info = model_fields[field_name]
            type_str, field_enum = _annotation_to_type(info.annotation)
            return type_str, field_enum, getattr(info, "description", None)
        except Exception:  # pragma: no cover — defensive
            return "unknown", None, None

    # 3. No cheap source.
    return "unknown", None, None


def _annotation_to_type(annotation: Any) -> tuple[str, list[str] | None]:
    """Map a Pydantic field annotation to ``(type, enum_members | None)``.

    Handles ``Optional[X]`` / ``X | None`` (unwrapped to X), ``Literal[...]``
    and ``enum.Enum`` subclasses (→ ``"enum"`` with members), containers
    (``list`` / ``dict``), and the scalar types.  Unrecognised annotations →
    ``"unknown"``.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[X] / Union — unwrap a single non-None arm (e.g. ``str | None``).
    if origin is typing.Union or origin is getattr(_types, "UnionType", None):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_type(non_none[0])
        return "unknown", None

    if origin is typing.Literal:
        return "enum", [str(a) for a in args]

    if isinstance(annotation, type) and issubclass(annotation, _enum.Enum):
        return "enum", [str(member.value) for member in annotation]

    if origin in (list, tuple, set, frozenset) or annotation in (list, tuple, set, frozenset):
        return "list", None
    if origin is dict or annotation is dict:
        return "json", None

    # ``bool`` before ``int`` — ``bool`` is a subclass of ``int``.  Identity
    # checks (``is``) keep them distinct without the subclass trap.
    if annotation is bool:
        return "bool", None
    if annotation is int:
        return "int", None
    if annotation is float:
        return "float", None
    if annotation is str:
        return "str", None

    return "unknown", None


# -- shared -----------------------------------------------------------------


def _iter_steps(pipeline: Pipeline | None) -> list[Any]:
    """Step objects in execution (topological) order, or ``[]`` for no pipeline.

    Uses ``execution_levels`` + ``get_step`` — the same ordering the existing
    ``pipeline_start.steps`` metadata uses — so the manifest agrees with it.
    """
    if pipeline is None:
        return []
    steps: list[Any] = []
    for level in pipeline.execution_levels:
        for name in level:
            steps.append(pipeline.get_step(name))
    return steps

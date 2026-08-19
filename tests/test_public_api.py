"""A CI contract on the public API surface.

Accrue ships to PyPI with downstream users, and an increasing share of pull
requests are authored by agents. Nothing else in CI notices when the exported
surface changes: a renamed keyword argument, a dropped export, or a reordered
positional parameter sails through ``ruff`` and ``pytest`` as long as the
(mostly mocked) unit tests still pass. This module rebuilds the surface at test
time and diffs it against a committed snapshot, so changing the public API
becomes an *intentional* act with a reviewable diff.

The point is not to block change. It is to force every surface change into the
pull request as readable JSON, where ``- "signature": ...`` sits next to the
``+`` line and a reviewer can see what a caller would have to rewrite.

Scope is the four modules users are actually told to import from — see
``MODULES``. ``docs/`` and ``examples/`` import ``AnthropicClient``,
``load_fields`` and ``ConfigurationError`` directly, so a snapshot limited to
``accrue.__all__`` would leave documented names ungated.

Regenerate after an intentional change::

    python -m tests.test_public_api --update

Design notes, because the naive implementations are all subtly wrong:

* Members are filtered to those *defined in accrue*. ``inspect.getmembers`` on
  a Pydantic model reports 28 inherited names (``model_dump``,
  ``model_validate``, ...) and none of accrue's own, so an unfiltered snapshot
  would record Pydantic's API and turn every Pydantic release into a CI
  failure.
* Pydantic models are captured by ``model_fields``, not by methods — the fields
  *are* the API.
* Protocols are captured by their methods. ``Step`` and
  ``BatchCapableLLMClient`` both render as ``(*args, **kwargs)`` at class
  level, which says nothing about what an implementer must provide.
* Defaults come from ``inspect.signature``, whose ``<factory>`` sentinel is
  stable, and never from ``dataclasses.fields()``, whose raw defaults render as
  ``<_MISSING_TYPE object at 0x7f...>`` and would differ between runs.

The rendered surface is byte-identical on Python 3.10 through 3.13 and across
Pydantic 2.0 through 2.13, which is why this test runs on the whole CI matrix
rather than being pinned to one interpreter.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

try:  # pydantic is a base dependency; guard only so a partial env fails clearly
    import pydantic
except ImportError:  # pragma: no cover - base dependency, present in every env
    pydantic = None  # type: ignore[assignment]

SNAPSHOT_PATH = Path(__file__).with_name("public_api_snapshot.json")

REGEN_HINT = (
    "If this change was intentional:\n"
    "  1. Regenerate the snapshot:  python -m tests.test_public_api --update\n"
    "  2. Commit the updated tests/public_api_snapshot.json in this same PR\n"
    "  3. Add a CHANGELOG.md entry under [Unreleased] describing the change\n"
    "If it was not intentional, you have changed accrue's public API by "
    "accident — restore the old names and signatures instead."
)

#: Modules whose public surface is under contract. ``exports_from_all`` is
#: False for modules that define no ``__all__``; those fall back to the public
#: names *defined in that module*, which keeps re-imported helpers such as
#: ``typing.Any`` out of the snapshot.
MODULES: tuple[tuple[str, bool], ...] = (
    ("accrue", True),
    ("accrue.providers", True),
    ("accrue.data", True),
    ("accrue.core.exceptions", False),
)


def _render(text: str | None) -> str | None:
    """Normalise a rendered signature so diffs stay legible.

    Modules using ``from __future__ import annotations`` render annotations as
    quoted source text (``name: 'str'``) while Pydantic models render evaluated
    objects (``name: str``). Both forms are stable, but mixing them in one file
    makes the snapshot harder to read than it needs to be.
    """
    if text is None:
        return None
    text = re.sub(r": '([^']+)'", r": \1", text)
    text = re.sub(r"-> '([^']+)'", r"-> \1", text)
    text = text.replace("typing.", "")
    text = re.sub(r"\baccrue(?:\.\w+)+\.(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _signature(obj: Any) -> str | None:
    try:
        return _render(str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return None


def _kind(obj: Any) -> str:
    if inspect.isclass(obj):
        if getattr(obj, "_is_protocol", False):
            return "protocol"
        if issubclass(obj, BaseException):
            return "exception"
        if pydantic is not None and issubclass(obj, pydantic.BaseModel):
            return "pydantic_model"
        if dataclasses.is_dataclass(obj):
            return "dataclass"
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    return type(obj).__name__


def _members(cls: type) -> dict[str, dict[str, Any]]:
    """Public methods, properties and attributes *defined in accrue*.

    Uses ``getattr_static`` so descriptors are inspected rather than invoked,
    and filters on ``__module__`` so inherited third-party API (Pydantic's, in
    particular) never enters the snapshot. Methods inherited from another
    accrue class are kept — they are part of this class's surface.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue
        try:
            value = inspect.getattr_static(cls, name)
        except AttributeError:  # pragma: no cover - defensive
            continue

        if isinstance(value, property):
            func = value.fget
            module = getattr(func, "__module__", None)
            entry = {"kind": "property", "signature": _signature(func) if func else None}
        elif isinstance(value, (staticmethod, classmethod)):
            module = getattr(value.__func__, "__module__", None)
            entry = {"kind": type(value).__name__, "signature": _signature(value.__func__)}
        elif callable(value):
            module = getattr(value, "__module__", None)
            entry = {"kind": "method", "signature": _signature(value)}
        else:
            module = getattr(type(value), "__module__", None)
            entry = {"kind": "attribute"}

        if module and module.split(".")[0] == "accrue":
            out[name] = entry
    return out


def _describe(obj: Any) -> dict[str, Any]:
    kind = _kind(obj)
    entry: dict[str, Any] = {"kind": kind}

    # A Protocol's own signature is a meaningless ``(*args, **kwargs)``.
    if kind != "protocol":
        entry["signature"] = _signature(obj)

    if inspect.isclass(obj):
        entry["members"] = _members(obj)
        if kind == "exception":
            entry["bases"] = [b.__name__ for b in obj.__mro__[1:] if b is not object]
        if kind == "pydantic_model":
            entry["fields"] = {
                name: {
                    "annotation": _render(str(field.annotation)),
                    "required": field.is_required(),
                    "default": None if field.is_required() else repr(field.default),
                }
                for name, field in sorted(obj.model_fields.items())
            }
    return entry


def _public_names(module: Any, from_all: bool) -> list[str]:
    if from_all:
        return sorted(module.__all__)
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and getattr(value, "__module__", None) == module.__name__
    )


def build_surface() -> dict[str, Any]:
    """Rebuild the public API surface from the live package."""
    surface: dict[str, Any] = {}
    for module_name, from_all in MODULES:
        module = importlib.import_module(module_name)
        names = _public_names(module, from_all)
        entries: dict[str, Any] = {}
        for name in names:
            # A name in __all__ that is not actually importable is itself a
            # defect — `from accrue import *` would raise for downstream users.
            assert hasattr(module, name), (
                f"{module_name}.__all__ lists {name!r} but the module has no such "
                f"attribute. `from {module_name} import *` would fail for users."
            )
            entries[name] = _describe(getattr(module, name))
        surface[module_name] = {"exports": names, "names": entries}
    return surface


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted paths so diffs can point at one field."""
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def _describe_change(path: str, old: Any, new: Any) -> str:
    """Render one changed value.

    Lists (the per-module ``exports``) are rendered as element deltas. Printing
    two 33-name lists in full and leaving the reader to spot the difference is
    exactly the kind of unreadable failure this test exists to avoid.
    """
    if isinstance(old, list) and isinstance(new, list):
        gone = sorted(set(old) - set(new))
        came = sorted(set(new) - set(old))
        parts = [f"    ~ {path}:"]
        if gone:
            parts.append(f"        no longer exported: {', '.join(gone)}")
        if came:
            parts.append(f"        newly exported:     {', '.join(came)}")
        if not gone and not came:  # same members, different order
            parts.append(f"        reordered: {new!r}")
        return "\n".join(parts)
    return f"    ~ {path}:\n        was: {old!r}\n        now: {new!r}"


def _format_delta(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    old, new = _flatten(expected), _flatten(actual)
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])

    lines: list[str] = ["accrue's public API no longer matches the committed snapshot.", ""]
    if removed:
        lines.append(f"REMOVED ({len(removed)}) — breaking for downstream users:")
        lines += [f"    - {k}: {old[k]!r}" for k in removed]
        lines.append("")
    if changed:
        lines.append(f"CHANGED ({len(changed)}) — check whether existing callers still work:")
        lines += [_describe_change(k, old[k], new[k]) for k in changed]
        lines.append("")
    if added:
        lines.append(f"ADDED ({len(added)}) — additive, but still a public API change:")
        lines += [f"    + {k}: {new[k]!r}" for k in added]
        lines.append("")
    lines.append(REGEN_HINT)
    return "\n".join(lines)


def write_snapshot() -> Path:
    """Write the current surface to the snapshot file. Used by ``--update``."""
    payload = json.dumps(build_surface(), indent=2, sort_keys=True)
    SNAPSHOT_PATH.write_text(payload + "\n", encoding="utf-8")
    return SNAPSHOT_PATH


@pytest.mark.skipif(not SNAPSHOT_PATH.exists(), reason="not running from a source checkout")
def test_public_api_matches_snapshot() -> None:
    """The exported surface must match the committed snapshot exactly.

    Additions fail too, deliberately. A subset check would let new exports land
    unreviewed and leave the snapshot permanently stale, so it would stop being
    a truthful answer to "what is accrue's public API?".
    """
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    actual = build_surface()
    if actual != expected:
        # pytest.fail rather than a bare assert: pytest's own introspection
        # would append a truncated dict-vs-dict dump underneath the curated
        # delta, which is precisely the unreadable output this replaces.
        pytest.fail(_format_delta(expected, actual), pytrace=False)


def test_snapshot_is_canonically_formatted() -> None:
    """The committed file must be exactly what ``--update`` writes.

    Otherwise a hand-edited snapshot could pass the diff above while producing
    a spurious whole-file diff the next time someone regenerates it.
    """
    on_disk = SNAPSHOT_PATH.read_text(encoding="utf-8")
    canonical = json.dumps(json.loads(on_disk), indent=2, sort_keys=True) + "\n"
    assert on_disk == canonical, (
        "tests/public_api_snapshot.json is not in canonical form. Do not edit it "
        "by hand — regenerate it:\n"
        "  python -m tests.test_public_api --update"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite tests/public_api_snapshot.json from the live package",
    )
    args = parser.parse_args()
    if not args.update:
        parser.error("nothing to do — pass --update to regenerate the snapshot")
    path = write_snapshot()
    print(f"Wrote {path}")
    print("Review the diff, then commit it with a CHANGELOG.md entry under [Unreleased].")

"""Repo-metadata invariants that used to be checked by a monthly LLM sweep.

The scheduled maintenance workflow ran "version coherence" and "CHANGELOG
presence" ten times over ten weeks against files that had not changed in
months, and never produced a finding. Both are cheap, exact assertions —
they belong here, where they run on every commit and cannot be misreported.

The version is read with a regex rather than ``tomllib`` because ``tomllib``
is 3.11+ and this suite runs on 3.10. The regex matches the same static
``version = "..."`` key that ``.github/workflows/publish.yml`` reads to verify
a release tag, so the two checks agree by construction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import accrue

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def _pyproject_version() -> str:
    match = _VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    assert match is not None, (
        'No static `version = "..."` key in pyproject.toml. publish.yml reads '
        "this key via tomllib to verify the git tag matches; if it has been "
        'replaced with `dynamic = ["version"]`, update that workflow too or '
        "the next release will fail with a KeyError."
    )
    return match.group(1)


@pytest.mark.skipif(not PYPROJECT.exists(), reason="not running from a source checkout")
def test_version_matches_package() -> None:
    """pyproject.toml and accrue.__version__ must not drift apart.

    They are two hand-maintained copies of the same number, and they have
    silently diverged before (see the 1.3.0 entry in CHANGELOG.md).
    """
    assert _pyproject_version() == accrue.__version__, (
        f"pyproject.toml says {_pyproject_version()!r} but "
        f"accrue.__version__ is {accrue.__version__!r}. Both must be bumped "
        "together — see the release skill."
    )


@pytest.mark.skipif(not CHANGELOG.exists(), reason="not running from a source checkout")
def test_changelog_has_section_for_current_version() -> None:
    """The version being shipped must have release notes.

    1.3.0 was tagged and published to PyPI with its notes still sitting under
    ``[Unreleased]``, and nothing caught it for three months. If this fails on
    a version-bump commit, the fix is to write the section — not to skip the
    test. The release flow bumps the version and drafts the notes together, so
    they land in the same commit.
    """
    version = accrue.__version__
    body = CHANGELOG.read_text(encoding="utf-8")
    assert re.search(rf"^## \[{re.escape(version)}\]", body, re.MULTILINE), (
        f"CHANGELOG.md has no `## [{version}]` section, but that is the "
        "version this package ships. Move the relevant entries out of "
        "[Unreleased] into a dated section for it."
    )


@pytest.mark.skipif(not CHANGELOG.exists(), reason="not running from a source checkout")
def test_changelog_unreleased_section_exists() -> None:
    """Keep-a-Changelog format requires a landing spot for the next change."""
    body = CHANGELOG.read_text(encoding="utf-8")
    assert re.search(r"^## \[Unreleased\]", body, re.MULTILINE), (
        "CHANGELOG.md has no `## [Unreleased]` heading. New entries have nowhere to go without one."
    )

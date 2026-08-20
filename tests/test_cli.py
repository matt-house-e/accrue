"""Tests for the ``accrue`` console CLI (accrue/cli.py).

The ``watch`` subcommand delegates to the optional accrue-ui package,
which is deliberately NOT installed in this test environment — the
ImportError hint path is exercised for real.  Delegation is covered by
injecting a fake ``accrue_ui.cli`` module into ``sys.modules``.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
import subprocess
import sys
import types

import pytest

from accrue.cli import INSTALL_HINT, USAGE, main

ACCRUE_UI_INSTALLED = importlib.util.find_spec("accrue_ui") is not None


# ---------------------------------------------------------------------------
# In-process
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_subcommand_prints_usage_and_exits_2(self, capsys):
        assert main([]) == 2
        assert capsys.readouterr().err == USAGE

    def test_unknown_subcommand_prints_usage_and_exits_2(self, capsys):
        assert main(["frobnicate"]) == 2
        assert capsys.readouterr().err == USAGE

    def test_help_prints_usage_and_exits_0(self, capsys):
        assert main(["--help"]) == 0
        assert capsys.readouterr().out == USAGE
        assert main(["-h"]) == 0

    @pytest.mark.skipif(ACCRUE_UI_INSTALLED, reason="accrue-ui installed; hint path unreachable")
    def test_watch_without_accrue_ui_hints_and_exits_1(self, capsys):
        """The lead hint must be an install line that actually works.

        It used to be ``pip install 'accrue[ui]'``, which fails outright:
        accrue-ui is not on PyPI, and the extra has been removed.
        """
        assert main(["watch"]) == 1
        err = capsys.readouterr().err
        lines = err.strip().splitlines()
        assert len(lines) == 2
        assert INSTALL_HINT in lines[0]
        assert INSTALL_HINT == "pip install git+https://github.com/matt-house-e/accrue-ui"
        assert "accrue[ui]" not in err
        assert "not on PyPI" in err

    def test_watch_distinguishes_a_broken_accrue_ui_from_a_missing_one(self, monkeypatch, capsys):
        """accrue-ui installed but raising on import must not print an install hint."""
        fake_pkg = types.ModuleType("accrue_ui")
        fake_pkg.__spec__ = importlib.machinery.ModuleSpec("accrue_ui", None)
        # No `cli` submodule and no __path__: importing accrue_ui.cli raises
        # ImportError from inside an installed package.
        monkeypatch.setitem(sys.modules, "accrue_ui", fake_pkg)
        monkeypatch.delitem(sys.modules, "accrue_ui.cli", raising=False)

        assert main(["watch"]) == 1
        err = capsys.readouterr().err
        assert "installed but failed to import" in err
        assert INSTALL_HINT not in err

    def test_watch_delegates_remaining_argv(self, monkeypatch):
        seen: list[list[str]] = []

        fake_cli = types.ModuleType("accrue_ui.cli")
        fake_cli.main = lambda argv: (seen.append(argv), 0)[1]
        fake_pkg = types.ModuleType("accrue_ui")
        fake_pkg.cli = fake_cli
        monkeypatch.setitem(sys.modules, "accrue_ui", fake_pkg)
        monkeypatch.setitem(sys.modules, "accrue_ui.cli", fake_cli)

        assert main(["watch", "--run", "x.jsonl"]) == 0
        assert seen == [["--run", "x.jsonl"]]

    def test_watch_propagates_delegate_exit_code(self, monkeypatch):
        fake_cli = types.ModuleType("accrue_ui.cli")
        fake_cli.main = lambda argv: 3
        fake_pkg = types.ModuleType("accrue_ui")
        fake_pkg.cli = fake_cli
        monkeypatch.setitem(sys.modules, "accrue_ui", fake_pkg)
        monkeypatch.setitem(sys.modules, "accrue_ui.cli", fake_cli)
        assert main(["watch"]) == 3

    def test_watch_treats_none_return_as_success(self, monkeypatch):
        fake_cli = types.ModuleType("accrue_ui.cli")
        fake_cli.main = lambda argv: None
        fake_pkg = types.ModuleType("accrue_ui")
        fake_pkg.cli = fake_cli
        monkeypatch.setitem(sys.modules, "accrue_ui", fake_pkg)
        monkeypatch.setitem(sys.modules, "accrue_ui.cli", fake_cli)
        assert main(["watch"]) == 0

    def test_watch_treats_any_non_int_return_as_success(self, monkeypatch):
        """``or 0`` turned a falsy-but-meaningful return into 0; only ints are codes."""
        fake_cli = types.ModuleType("accrue_ui.cli")
        fake_cli.main = lambda argv: "quit"
        fake_pkg = types.ModuleType("accrue_ui")
        fake_pkg.cli = fake_cli
        monkeypatch.setitem(sys.modules, "accrue_ui", fake_pkg)
        monkeypatch.setitem(sys.modules, "accrue_ui.cli", fake_cli)
        assert main(["watch"]) == 0


# ---------------------------------------------------------------------------
# Console entry points
# ---------------------------------------------------------------------------


class TestConsoleEntry:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "accrue.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_module_invocation_no_args_exits_2(self):
        proc = self._run()
        assert proc.returncode == 2
        assert "usage: accrue" in proc.stderr

    def test_module_invocation_unknown_exits_2(self):
        assert self._run("bogus").returncode == 2

    @pytest.mark.skipif(ACCRUE_UI_INSTALLED, reason="accrue-ui installed; hint path unreachable")
    def test_module_invocation_watch_exits_1_with_hint(self):
        proc = self._run("watch")
        assert proc.returncode == 1
        assert INSTALL_HINT in proc.stderr
        assert "accrue[ui]" not in proc.stderr

    @pytest.mark.skipif(
        shutil.which("accrue") is None, reason="accrue script not on PATH (not pip-installed)"
    )
    def test_installed_script_no_args_exits_2(self):
        proc = subprocess.run(["accrue"], capture_output=True, text=True)
        assert proc.returncode == 2
        assert "usage: accrue" in proc.stderr

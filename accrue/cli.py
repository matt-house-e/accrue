"""``accrue`` console entry point.

Skeleton for the planned 6B CLI: ``accrue <command> [args...]``.  Each
subcommand is a callable in :data:`COMMANDS` taking the remaining argv and
returning an exit code, so future subcommands (waterfall, chunked, ...)
slot in without restructuring.

The only subcommand today is ``watch``, which delegates to the optional
`accrue-ui <https://github.com/matt-house-e/accrue-ui>`_ package.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Callable

USAGE = """\
usage: accrue <command> [args...]

commands:
  watch    Watch pipeline runs live (requires the accrue-ui package)
"""

#: accrue-ui is not on PyPI, so the git URL is the install line that works.
INSTALL_HINT = "pip install git+https://github.com/matt-house-e/accrue-ui"


def _accrue_ui_present() -> bool:
    """True when ``accrue_ui`` is importable as a distribution."""
    try:
        return importlib.util.find_spec("accrue_ui") is not None
    except (ImportError, ValueError):
        return False


def _cmd_watch(argv: list[str]) -> int:
    """Delegate to accrue-ui's CLI, or explain how to install it."""
    try:
        from accrue_ui.cli import main as ui_main
    except ImportError as exc:
        if _accrue_ui_present():
            # Installed, but its own import failed — printing an install hint
            # here would send the user to fix something that is not broken.
            print(
                f"accrue watch: accrue-ui is installed but failed to import: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"accrue watch requires the accrue-ui package: {INSTALL_HINT}", file=sys.stderr)
            print(
                "(accrue-ui is not on PyPI yet, so install it from the repo.)",
                file=sys.stderr,
            )
        return 1
    rc = ui_main(argv)
    return rc if isinstance(rc, int) else 0


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "watch": _cmd_watch,
}


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (``[project.scripts] accrue``)."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(USAGE, end="", file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())

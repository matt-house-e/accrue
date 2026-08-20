"""``accrue`` console entry point.

Skeleton for the planned 6B CLI: ``accrue <command> [args...]``.  Each
subcommand is a callable in :data:`COMMANDS` taking the remaining argv and
returning an exit code, so future subcommands (waterfall, chunked, ...)
slot in without restructuring.

The only subcommand today is ``watch``, which delegates to the optional
`accrue-ui <https://github.com/matt-house-e/accrue-ui>`_ package.
"""

from __future__ import annotations

import sys
from typing import Callable

USAGE = """\
usage: accrue <command> [args...]

commands:
  watch    Watch pipeline runs live (requires the accrue-ui package)
"""


def _cmd_watch(argv: list[str]) -> int:
    """Delegate to accrue-ui's CLI, or explain how to install it."""
    try:
        from accrue_ui.cli import main as ui_main
    except ImportError:
        print(
            "accrue watch requires the accrue-ui package: pip install 'accrue[ui]'",
            file=sys.stderr,
        )
        print(
            "(accrue-ui is not yet on PyPI -- until then: "
            "pip install git+https://github.com/matt-house-e/accrue-ui)",
            file=sys.stderr,
        )
        return 1
    return ui_main(argv) or 0


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

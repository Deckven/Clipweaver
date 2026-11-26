"""
Rename analyzed videos to their suggested names (dry run unless --apply).

The same rename as ``analyzer --rename``, with a safer default: both go through
one planner (naming.rename_from_catalog), which reserves every video and catalog
destination across the batch and never moves a file onto another.
"""

from __future__ import annotations

import argparse

from . import config
from .log import add_verbose_flag, setup_logging
from .naming import rename_from_catalog


def main(argv: list[str] | None = None) -> None:
    """CLI: rename videos to their suggested_filename (dry run unless --apply)."""
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--game", default=config.DEFAULT_GAME, choices=list(config.GAMES))
    ap.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run)")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    rename_from_catalog(args.game, dry_run=not args.apply)
    if not args.apply:
        print("\nRe-run with --apply to perform renames.")


if __name__ == "__main__":
    main()

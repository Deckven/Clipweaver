"""
Rename helper for analyzed videos.

Reads catalog JSONs and renames source videos to `suggested_filename`.
By default runs in dry-run mode; pass --apply to actually rename.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import config
from . import catalog as _catalog
from .log import add_verbose_flag, setup_logging
from .naming import _warn_about_curation

log = logging.getLogger(__name__)


def iter_records(game: str):
    """(json path, entry) for each readable catalog file at the top of the game's folder."""
    subdir = config.GAMES.get(game, {}).get("catalog_subdir", f"{game}_videos")
    catalog_dir = config.CATALOG_DIR / subdir
    if not catalog_dir.exists():
        log.warning("No catalog at %s", catalog_dir)
        return
    for jf in sorted(catalog_dir.glob("*.json")):
        try:
            yield jf, _catalog.read_entry(jf)
        except (OSError, ValueError) as e:
            log.warning("skip %s: %s", jf.name, e)


def unique_target(target: Path) -> Path:
    """target, or target_2, target_3... - the first name not taken on disk."""
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 2
    while True:
        cand = target.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def main(argv: list[str] | None = None) -> None:
    """CLI: rename videos to their suggested_filename (dry run unless --apply)."""
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--game", default=config.DEFAULT_GAME, choices=list(config.GAMES))
    ap.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run)")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    planned = []
    for jf, rec in iter_records(args.game):
        if not rec.get("source_path"):
            log.warning("skip %s: no source_path", jf.name)
            continue
        src = Path(rec["source_path"])
        suggested = rec.get("suggested_filename")
        if not suggested or not src.exists():
            continue
        if src.name == suggested:
            continue
        target = unique_target(src.with_name(suggested))
        planned.append((src, target, jf))

    if not planned:
        print("Nothing to rename.")
        return

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: {len(planned)} file(s)")
    _warn_about_curation({src.name for src, _, _ in planned})
    for src, tgt, jf in planned:
        print(f"  {src.name}\n    -> {tgt.name}")
        if args.apply:
            src.rename(tgt)
            rec = _catalog.read_entry(jf)
            rec["source_path"] = str(tgt)
            rec["original_name"] = tgt.name
            jf.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            new_jf = jf.with_name(f"{tgt.stem}.json")
            if new_jf != jf:
                jf.rename(new_jf)

    if not args.apply:
        print("\nRe-run with --apply to perform renames.")


if __name__ == "__main__":
    main()

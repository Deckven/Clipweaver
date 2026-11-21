"""
Rename helper for analyzed videos.

Reads catalog JSONs and renames source videos to `suggested_filename`.
By default runs in dry-run mode; pass --apply to actually rename.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def iter_records(game: str):
    subdir = config.GAMES.get(game, {}).get("catalog_subdir", f"{game}_videos")
    catalog_dir = config.CATALOG_DIR / subdir
    if not catalog_dir.exists():
        print(f"No catalog at {catalog_dir}")
        return
    for jf in sorted(catalog_dir.glob("*.json")):
        try:
            yield jf, json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {jf.name}: {e}")


def unique_target(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 2
    while True:
        cand = target.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=config.DEFAULT_GAME, choices=list(config.GAMES))
    ap.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run)")
    args = ap.parse_args()

    planned = []
    for jf, rec in iter_records(args.game):
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
    for src, tgt, jf in planned:
        print(f"  {src.name}\n    -> {tgt.name}")
        if args.apply:
            src.rename(tgt)
            rec = json.loads(jf.read_text(encoding="utf-8"))
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

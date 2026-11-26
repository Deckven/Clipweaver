"""Regex grep over a game's VLM catalog: timeline captions/OCR and scene text.

The semantic index (embed_index) answers "show me a picture like this"; this
answers "where does this exact word appear on screen" — a place name, a code,
an item name, a number. Each hit is a moment (timecode), not just a file.

Usage:
    python -m clipmaker.ocrgrep "<regex>" --game <game> [--per-file 4] [--file <name substring>]

Output, one block per matching video:
    <original_name> [<duration>s] @ 01:10, 01:25, 02:00 (+3)
        <first matching caption or scene description>
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

from . import config
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

CAPTION_PREVIEW = 150
DEFAULT_PER_FILE = 4


@dataclass(frozen=True)
class Match:
    """One video whose OCR or scene text matched, with the timecodes of the hits."""
    name: str
    duration_sec: float | None
    stamps: tuple[str, ...]
    preview: str


def catalog_dir(game: str) -> Path:
    """The game's catalog folder; ValueError for an unknown game."""
    game_cfg = config.GAMES.get(game)
    if not game_cfg:
        raise ValueError(f"unknown game {game!r}; known: {', '.join(config.GAMES)}")
    return config.CATALOG_DIR / game_cfg.get("catalog_subdir", f"{game}_videos")


def iter_entries(cat_dir: Path) -> Iterator[dict]:
    """Catalog entries in path order. Non-object JSON is skipped quietly: the
    catalog holds sidecar files (embeddings, debug dumps) that are not entries.
    A file that cannot be read at all is skipped with a warning."""
    for path in sorted(cat_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            log.warning("skipped unreadable catalog file %s: %s", path, exc)
            continue
        if isinstance(data, dict):
            yield data


def _as_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value or "")


def entry_hits(entry: dict, pattern: re.Pattern) -> list[dict]:
    """Timeline items whose caption/OCR match, then scenes whose description does."""
    hits = [t for t in entry.get("timeline") or []
            if isinstance(t, dict)
            and pattern.search(_as_text(t.get("caption")) + " " + _as_text(t.get("ocr")))]
    hits += [s for s in entry.get("scenes") or []
             if isinstance(s, dict) and pattern.search(_as_text(s.get("description")))]
    return hits


def search(cat_dir: Path, pattern: re.Pattern,
           name_filter: str | None = None) -> list[Match]:
    """Videos whose OCR or scene text matches the pattern, in catalog order."""
    needle = (name_filter or "").lower()
    matches = []
    for entry in iter_entries(cat_dir):
        name = str(entry.get("original_name") or Path(str(entry.get("source_path", "?"))).name)
        if needle and needle not in name.lower():
            continue
        hits = entry_hits(entry, pattern)
        if not hits:
            continue
        stamps = tuple(str(h.get("start", h.get("timestamp", "?"))) for h in hits)
        preview = _as_text(hits[0].get("caption") or hits[0].get("description"))
        matches.append(Match(name, entry.get("duration_sec"), stamps,
                             preview[:CAPTION_PREVIEW]))
    return matches


def format_match(m: Match, per_file: int) -> str:
    """One result as two lines: name, duration and timecodes, then a preview."""
    shown = ", ".join(m.stamps[:per_file])
    more = f" (+{len(m.stamps) - per_file})" if len(m.stamps) > per_file else ""
    return f"{m.name} [{m.duration_sec}s] @ {shown}{more}\n    {m.preview}"


def main(argv: list[str] | None = None) -> None:
    """CLI: regex over the OCR and scene text of a game's catalog."""
    ap = argparse.ArgumentParser(prog="clipmaker.ocrgrep",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("regex", help="case-insensitive regular expression")
    ap.add_argument("--game", required=True, choices=list(config.GAMES))
    ap.add_argument("--per-file", type=int, default=DEFAULT_PER_FILE,
                    help="timecodes shown per video (default %(default)s)")
    ap.add_argument("--file", default=None,
                    help="only videos whose name contains this substring")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    if args.per_file <= 0:
        ap.error("--per-file must be positive")
    try:
        pattern = re.compile(args.regex, re.I)
    except re.error as exc:
        ap.error(f"bad regex: {exc}")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cat_dir = catalog_dir(args.game)
    if not cat_dir.is_dir():
        sys.exit(f"No catalog folder {cat_dir}")
    matches = search(cat_dir, pattern, args.file)
    for m in matches:
        print(format_match(m, args.per_file))
    print(f"-- {len(matches)} files")


if __name__ == "__main__":
    main()

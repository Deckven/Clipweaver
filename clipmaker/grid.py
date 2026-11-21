"""Frame grid of one footage file, found by name, for eyeballing a range.

Checking an event's range before it goes into vegas_events.json: give the clip
by (part of) its name and a time range, get one labelled sheet. The range ends
are included, so the first and the last tile are exactly the slot's edges.

Usage:
    python -m clipmaker.grid NAME --from S --to E --out grid.jpg [--count 12] [--game G] [--dir D ...]
    python -m clipmaker.grid NAME --at T [T ...] --out grid.jpg [--game G] [--dir D ...]

NAME is a path, a full file stem, or a fragment of one. Lookup order: an
existing path; an exact stem; a stem ending in " NAME" (so "Trading 1" finds
"<game prefix> Trading 1" but not "... Trading 11"); a unique substring.
Footage is searched in --dir folders and, with --game, in that game's
video_dirs from config (recursively). Times: seconds, M:SS or H:MM:SS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .stills import parse_time, video_sheet

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
DEFAULT_COUNT = 12
DEFAULT_COLS = 4
DEFAULT_TILE_WIDTH = 480
AMBIGUOUS_SHOWN = 8


def footage_files(dirs: list[Path], recursive: bool = True) -> list[Path]:
    files: set[Path] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        found = d.rglob("*") if recursive else d.iterdir()
        files.update(p for p in found if p.suffix.lower() in VIDEO_EXTS and p.is_file())
    return sorted(files)


def find_clip(name: str, files: list[Path]) -> Path:
    """Resolve NAME against candidate files (see module docstring for order).
    Raises LookupError listing the candidates when nothing or several match."""
    if Path(name).is_file():
        return Path(name)
    wanted = Path(name).stem.lower() if Path(name).suffix.lower() in VIDEO_EXTS else name.lower()
    for rule in (lambda s: s == wanted,
                 lambda s: s.endswith(" " + wanted),
                 lambda s: wanted in s):
        hits = [p for p in files if rule(p.stem.lower())]
        if len(hits) == 1:
            return hits[0]
        if hits:
            # A stronger rule with several hits (the same stem in two folders)
            # is ambiguous; falling through to a weaker rule would only add more.
            shown = ", ".join(str(p) for p in hits[:AMBIGUOUS_SHOWN])
            raise LookupError(f"{name!r}: {len(hits)} matches: {shown}")
    raise LookupError(f"{name!r}: no footage file matches")


def range_times(start: float, end: float, count: int) -> list[float]:
    """``count`` times from start to end inclusive (one tile -> the start)."""
    if end < start:
        raise ValueError(f"range ends before it starts: {start} > {end}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count == 1:
        return [round(start, 3)]
    step = (end - start) / (count - 1)
    return [round(start + i * step, 3) for i in range(count)]


def search_dirs(game: str | None, extra: list[Path]) -> list[Path]:
    dirs = list(extra)
    if game:
        game_cfg = config.GAMES.get(game)
        if not game_cfg:
            raise ValueError(f"unknown game {game!r}; known: {', '.join(config.GAMES)}")
        dirs += [Path(d) for d in game_cfg.get("video_dirs", [])]
    return dirs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clipmaker.grid", description=__doc__.split("\n")[0])
    ap.add_argument("name", help="footage path, file stem, or a fragment of one")
    ap.add_argument("--from", dest="start", help="range start")
    ap.add_argument("--to", dest="end", help="range end (included)")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--at", nargs="+", help="explicit times instead of a range")
    ap.add_argument("--out", required=True)
    ap.add_argument("--game", default=None, help="search this game's video_dirs from config")
    ap.add_argument("--dir", action="append", type=Path, default=[],
                    help="extra footage folder (repeatable)")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS)
    ap.add_argument("--tile-width", type=int, default=DEFAULT_TILE_WIDTH)
    args = ap.parse_args(argv)
    if bool(args.at) == bool(args.start is not None or args.end is not None):
        ap.error("give either --at or --from/--to")
    if not args.at and (args.start is None or args.end is None):
        ap.error("--from and --to go together")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        dirs = search_dirs(args.game, args.dir)
        if not dirs and not Path(args.name).is_file():
            ap.error("NAME is not a file: pass --game and/or --dir to search footage")
        src = find_clip(args.name, footage_files(dirs))
        times = ([parse_time(t) for t in args.at] if args.at
                 else range_times(parse_time(args.start), parse_time(args.end), args.count))
        video_sheet(str(src), times, Path(args.out), args.cols, args.tile_width)
    except (LookupError, ValueError, RuntimeError, OSError) as exc:
        sys.exit(f"error: {exc}")
    print(f"{args.out} <- {src}")


if __name__ == "__main__":
    main()

"""One reader for the catalog: every module loads catalog JSONs the same way.

Before this module there were four copies of the loader (report, build_storyboard,
clip_extractor, layout). They had drifted: report read plain utf-8, so a catalog
file with a byte-order mark (files saved from PowerShell have one) failed to parse
and was skipped without a word — its search hits never reached the storyboard.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from . import config

_reported: set[str] = set()


def _report_once(key: str, message: str) -> None:
    if key not in _reported:
        _reported.add(key)
        print(f"[catalog] {message}")


def read_entry(path: Path) -> dict:
    """One catalog JSON. utf-8-sig accepts a leading BOM as well as plain utf-8."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def catalog_dir(game: str) -> Path | None:
    """The game's catalog folder, or None when the game is unknown."""
    game_cfg = config.GAMES.get(game)
    if not game_cfg:
        return None
    return config.CATALOG_DIR / game_cfg["catalog_subdir"]


def load_entries(games: Iterable[str] | None = None) -> list[dict]:
    """Every catalog entry of `games` (all configured games by default).

    Each entry gets ``_game`` and ``_catalog_path`` (relative to CATALOG_DIR,
    forward slashes). A file that cannot be read is reported, not skipped
    silently. Order is stable: games as given, files by path.
    """
    entries: list[dict] = []
    for game in (list(games) if games is not None else list(config.GAMES)):
        cat_dir = catalog_dir(game)
        if cat_dir is None or not cat_dir.is_dir():
            continue
        for path in sorted(cat_dir.rglob("*.json")):
            rel = path.relative_to(config.CATALOG_DIR).as_posix()
            try:
                data = read_entry(path)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                _report_once(rel, f"skipped unreadable catalog file {rel}: {exc}")
                continue
            if not isinstance(data, dict):
                _report_once(rel, f"skipped {rel}: not a JSON object")
                continue
            data["_game"] = game
            data["_catalog_path"] = rel
            entries.append(data)
    return entries


def index_by_name(entries: Iterable[dict]) -> dict[str, dict]:
    """Entries by ``original_name`` — the key timings and curation files use.

    When two catalog files carry the same name, the first one wins (games in
    the order given, then by path) and the clash is reported once, instead of
    the later file silently replacing the earlier one.
    """
    index: dict[str, dict] = {}
    for entry in entries:
        name = entry.get("original_name") or ""
        if not name:
            continue
        kept = index.get(name)
        if kept is None:
            index[name] = entry
        elif kept.get("_catalog_path") != entry.get("_catalog_path"):
            _report_once(
                "name:" + name,
                f"{name!r} is in two catalog files: {kept.get('_catalog_path')} (used) and "
                f"{entry.get('_catalog_path')} (ignored) — rename one so a pick is unambiguous",
            )
    return index


def load_index(games: Iterable[str] | None = None) -> dict[str, dict]:
    """``index_by_name(load_entries(games))``."""
    return index_by_name(load_entries(games))


def map_path(path: str | Path) -> Path | None:
    """Where `path` lives after a move listed in config.PATH_MAP, or None when
    no entry matches. Case and slash direction do not matter (Windows paths)."""
    text = str(path).replace("\\", "/")
    low = text.lower()
    for old, new in config.PATH_MAP:
        o = old.lower()
        if low == o or low.startswith(o + "/"):
            return Path(new + text[len(old):])
    return None


def resolve_source(video_name: str, footage_dir: Path,
                   catalog: dict[str, dict]) -> Path | None:
    """The video file for a catalog name. Tried in order: the project's
    footage/ override; the catalog source_path as-is; source_path's folder +
    the name (files renamed to the descriptive name leave source_path pointing
    at the old raw name); then both again under config.PATH_MAP."""
    entry = catalog.get(video_name, {})
    orig = entry.get("source_path", "")
    candidates = [footage_dir / video_name]
    if orig:
        for base in (Path(orig), map_path(orig)):
            if base is not None:
                candidates += [base, base.parent / video_name]
    return next((c for c in candidates if c.exists()), None)


def existing_path(path: Path) -> Path:
    """`path` itself, or its PATH_MAP location when only that one exists."""
    if path.exists():
        return path
    moved = map_path(path)
    return moved if moved is not None and moved.exists() else path

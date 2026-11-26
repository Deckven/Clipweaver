"""Suggested filenames for analyzed videos, and renaming videos to them.

A suggested name is "<Game display name> <content tokens>.ext", built from the
catalog record (location folder, title words, star systems from OCR, the most
frequent activity tags). Renaming appends " N" to keep duplicates apart and
moves the catalog JSON along with the video.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import catalog as _catalog
from . import config

log = logging.getLogger(__name__)


# Words that should never end up in tags or filenames (too generic)
_NOISE_TOKENS = {
    "elden", "ring", "er", "gameplay", "video", "footage", "clip",
    "game", "play", "playing", "scene", "moment", "the", "and", "with",
    "of", "in", "on", "to", "a", "an", "for", "from",
    "snake_case", "snake", "case", "lowercase", "specific", "tags",
    "t1", "t2", "tag", "example", "title",
}


# Tags matching these regex patterns are stripped from VLM output — they come
# from OCR chrome (corp ids, local counts, etc.) and are noise, not entities.
_NOISE_TAG_PATTERNS = [
    re.compile(p) for p in (
        r"^corp(?:oration)?_?\d+$",
        r"^local_?\d+$",
        r"^\d+$",
        r"^channel_\d+$",
    )
]


def _is_noise_tag(tag: str) -> bool:
    t = str(tag).lower()
    if t in _NOISE_TOKENS:
        return True
    return any(p.match(t) for p in _NOISE_TAG_PATTERNS)


def _to_tag(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# System/entity extraction patterns for building unique filenames.
# Codes: F67E-0, 6E-578, JH-MZW, MZ-CFI, Y9g-KS (2-5 chars, optional digit+letter mix, dash, 2-5 more)
_SYSTEM_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{0,3}-[A-Z0-9]{2,5}\b")


# Named system + sec status: "Postouvin 0.7", "Niballe 0,6", "Larryn 8.6"
_SYSTEM_WITH_SEC_RE = re.compile(r"\b([A-Z][a-z]{3,12})\s+[0-9][,. ][0-9]")


# Location folders are usually long and have filler words we can drop.
_LOCATION_FILLER = {"the", "a", "for", "and", "president"}


def _primary_systems(ocr_text: list[str], limit: int = 2) -> list[str]:
    """Return the most-frequent star system references across OCR text."""
    from collections import Counter
    codes: list[str] = []
    names: list[str] = []
    for line in ocr_text:
        codes.extend(_SYSTEM_CODE_RE.findall(line))
        names.extend(_SYSTEM_WITH_SEC_RE.findall(line))
    freq = Counter(codes + names)
    return [s for s, _ in freq.most_common(limit)]


def _primary_activity_tags(timeline: list[dict], exclude: set[str], limit: int = 2) -> list[str]:
    """Most-frequent vocab tag across the timeline, excluding given tags
    (e.g. the location tag, already in the filename)."""
    from collections import Counter
    c: Counter = Counter()
    for seg in timeline or []:
        for t in seg.get("tags") or []:
            if t and t not in exclude and not _is_noise_tag(t):
                c[t] += 1
    return [t for t, _ in c.most_common(limit)]


def _game_display_name(game: str) -> str:
    cfg = config.GAMES.get(game, {})
    return cfg.get("display_name") or (game.replace("_", " ").title() if game else "")


def _titleize_vocab_tag(tag: str) -> str:
    """Turn a vocab-derived tag (e.g. 'warp', 'memorial_research_site',
    'gallente_shuttle') into a human-readable Title Case string."""
    return tag.replace("_", " ").title()


def _norm_word(w: str) -> str:
    """Fold a word for duplicate detection: lowercase, curly quotes -> straight,
    surrounding punctuation dropped. Without this, the folder's 'Assassin’s'
    never matches the display name's "Assassin's" and the game name lands in
    the filename twice. '_' is ShadowPlay's stand-in for a colon in the folder
    name ('Onimusha_ Way of the Sword')."""
    return w.replace("’", "'").strip(".,:;!?()[]\"_").lower()


# A title is one sentence; only the leading words carry the discriminating
# content ("Cape Bonavista Naval Engagement" -> all 4 kept).
_TITLE_MAX_WORDS = 6


def _content_tokens(analysis: dict, skip_words: set[str] | None = None) -> list[str]:
    """Human-readable Title-case tokens describing what's in the video:
    shortened location + title words + up to 2 primary star systems + up to 2
    activities (mission / warp / mining / combat).

    The title matters most: location is per-folder, so for a game whose footage
    all sits in one directory it is identical across every video and the names
    collapse into each other."""
    out: list[str] = []
    seen_lower: set[str] = {_norm_word(w) for w in (skip_words or ())}

    def add(tok: str) -> None:
        t = tok.strip()
        if not t:
            return
        n = _norm_word(t)
        if not n or n in seen_lower:
            return
        seen_lower.add(n)
        out.append(t)

    ph = analysis.get("path_hints") or {}
    location = ph.get("location") or ""
    for part in location.split():
        if _norm_word(part) in _LOCATION_FILLER:
            continue
        w = part.strip()
        add(w if w.isupper() and len(w) >= 2 else w.capitalize())

    kept = 0
    for part in str(analysis.get("title") or "").split():
        if kept >= _TITLE_MAX_WORDS:
            break
        n = _norm_word(part)
        if not n or n in _LOCATION_FILLER or _is_noise_tag(n):
            continue
        before = len(out)
        add(part if part.isupper() and len(part) >= 2 else part.capitalize())
        if len(out) > before:
            kept += 1

    for s in _primary_systems(analysis.get("ocr_text") or [], limit=2):
        add(s.strip())

    loc_tag_set = {_to_tag(location)} | {_to_tag(t) for t in out}
    for tag in _primary_activity_tags(analysis.get("timeline") or [], loc_tag_set, limit=2):
        add(_titleize_vocab_tag(tag))

    return out


# Characters Windows forbids in a filename. display_name legitimately contains
# a colon ("Assassin's Creed IV: Black Flag"), so the base MUST be scrubbed or
# the rename fails with OSError on every file.
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename_base(base: str) -> str:
    cleaned = _ILLEGAL_FILENAME_CHARS.sub(" ", base)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Windows also rejects a trailing dot or space.
    return cleaned.rstrip(". ")


def suggest_filename(analysis: dict, original: Path) -> str:
    """Base filename in the form "<Game display> <content tokens>.ext".

    The sequence number (" 1", " 2", …) used to disambiguate duplicates is
    appended at rename time, so the stored value here is the stable base."""
    game = analysis.get("game") or ""
    game_display = _game_display_name(game)
    skip = set(game_display.split()) if game_display else set()
    content = _content_tokens(analysis, skip_words=skip)

    parts = []
    if game_display:
        parts.append(game_display)
    parts.extend(content)

    base = _safe_filename_base(" ".join(p for p in parts if p))
    if not base:
        base = "Video"
    return f"{base[:120].rstrip('. ')}{original.suffix.lower()}"


def regenerate_suggestions(game: str, dry_run: bool = False) -> int:
    """Re-run suggest_filename over every cached JSON without touching the
    video files. Returns number of JSONs updated (or, on a dry run, that
    would be updated — nothing is written then)."""
    game_cfg = config.game_config(game)
    subdir = game_cfg.get("catalog_subdir", f"{game}_videos")
    cat_dir = config.CATALOG_DIR / subdir
    if not cat_dir.is_dir():
        log.warning("No catalog dir: %s", cat_dir)
        return 0

    updated = 0
    paths = _catalog.entry_paths(game)
    for p in paths:
        try:
            rec = _catalog.read_entry(p)
        except (OSError, ValueError) as e:
            log.warning("SKIP (bad json): %s - %s", p.name, e)
            continue
        original = Path(rec.get("source_path") or rec.get("original_name") or p.stem)
        new_name = suggest_filename(rec, original)
        if rec.get("suggested_filename") != new_name:
            if dry_run:
                print(f"  {p.name}: {rec.get('suggested_filename')!r} -> {new_name!r}")
            else:
                _catalog.write_entry({**rec, "suggested_filename": new_name}, p)
            updated += 1
    total = len(paths)
    verb = "DRY-RUN: would regenerate" if dry_run else "Regenerated"
    print(f"{verb} suggestions for {updated}/{total} JSONs")
    return updated


def _rename_plan(game: str) -> list[tuple[Path, Path, Path, Path]]:
    """Build a rename plan. Each video gets `" N"` appended before the
    extension, where N disambiguates videos sharing the same base name.
    Already-numbered videos keep their number, so the plan is idempotent and
    only ever touches new entries. Returns a list of
    (video_old, video_new, json_old, json_new)."""
    config.game_config(game)  # an unknown game is an error, not an empty plan

    # Gather (src, base_name, json_path) for every entry with a live video.
    entries: list[tuple[Path, str, Path]] = []
    for jp in _catalog.entry_paths(game):
        try:
            rec = _catalog.read_entry(jp)
        except (OSError, ValueError) as e:
            log.warning("SKIP (bad json): %s - %s", jp.name, e)
            continue
        src = Path(rec.get("source_path") or "")
        if not src.is_file():
            log.warning("SKIP (video missing): %s", src)
            continue
        base = rec.get("suggested_filename") or ""
        if not base:
            continue
        entries.append((src, base, jp))

    # Group by base name and assign seq in chronological order.
    from collections import defaultdict
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for src, base, jp in entries:
        groups[base].append((src, jp))
    for items in groups.values():
        items.sort(key=lambda t: t[0].name)

    plan: list[tuple[Path, Path, Path, Path]] = []
    claimed: set[Path] = set()       # video destinations reserved by this batch
    claimed_json: set[Path] = set()  # catalog destinations reserved by this batch
    for base, items in groups.items():
        stem, ext = Path(base).stem, Path(base).suffix
        seq_re = re.compile(re.escape(stem) + " ([0-9]+)" + re.escape(ext) + "$")

        # A video that already carries a sequence number for this base keeps it.
        # The number only disambiguates duplicates, so re-deriving it from the
        # current filename order renumbers settled files for nothing — and once
        # a group passes nine entries, string order alone cascades ("10" sorts
        # before "2", pushing 2->12, 3->13, ...).
        taken: set[int] = set()
        pending: list[tuple[Path, Path]] = []
        for src, jp in items:
            m = seq_re.match(src.name)
            if m:
                taken.add(int(m.group(1)))
                claimed.add(src)
            else:
                pending.append((src, jp))

        # Fresh entries take the lowest free numbers, in recording order
        # (items is sorted by filename, and un-renamed names embed the timestamp).
        next_n = 1
        for src, jp in pending:
            while True:
                new_video = src.with_name(f"{stem} {next_n}{ext}")
                new_json = jp.with_name(f"{new_video.stem}.json")
                if (next_n not in taken and _free(new_video, src, claimed)
                        and _free(new_json, jp, claimed_json)):
                    break
                next_n += 1
            taken.add(next_n)
            claimed.add(new_video)
            claimed_json.add(new_json)
            plan.append((src, new_video, jp, new_json))

    plan.sort(key=lambda p: p[0].name)
    return plan


def _free(dest: Path, current: Path, claimed: set[Path]) -> bool:
    """True when dest is neither reserved by this batch nor another file on disk."""
    return dest not in claimed and (dest == current or not dest.exists())


def _move(src: Path, dest: Path) -> None:
    """Rename that never replaces another file: os.rename overwrites on POSIX.

    A case-only rename (same file on a case-insensitive disk) is allowed."""
    if dest.exists() and not (src.exists() and src.samefile(dest)):
        raise FileExistsError(f"{dest} already exists")
    src.rename(dest)


def curation_files_naming(names: set[str]) -> dict[Path, list[str]]:
    """Project curation.json files that pick any of these video names, with the names."""
    root = config.PROJECTS_DIR
    hits: dict[Path, list[str]] = {}
    if not root.is_dir():
        return hits
    for path in sorted(root.rglob("curation.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            log.warning("cannot read %s to check it against the renames: %s", path, exc)
            continue
        picked = {spec if isinstance(spec, str) else spec[0]
                  for spec in (data.values() if isinstance(data, dict) else [])
                  if isinstance(spec, str) or (isinstance(spec, list) and spec)}
        found = sorted(names & picked)
        if found:
            hits[path] = found
    return hits


def _warn_about_curation(old_names: set[str]) -> None:
    """Renaming a video breaks every curation.json pick that names it: say which."""
    for path, names in curation_files_naming(old_names).items():
        log.warning("%s picks %d video(s) this renames (%s); update it after renaming, "
                    "or build_storyboard loses those picks", path, len(names),
                    ", ".join(names[:3]) + (" ..." if len(names) > 3 else ""))


def rename_from_catalog(game: str, dry_run: bool) -> int:
    """Use suggested_filename in each catalog JSON to rename the matching
    video (and its JSON). Returns number of renames executed (or planned)."""
    plan = _rename_plan(game)
    if not plan:
        print("Nothing to rename.")
        return 0

    print(f"{'DRY-RUN' if dry_run else 'RENAMING'}: {len(plan)} video(s)")
    for old_v, new_v, _, _ in plan:
        print(f"  {old_v.name}  ->  {new_v.name}")
    _warn_about_curation({old_v.name for old_v, _, _, _ in plan})

    if dry_run:
        return len(plan)

    executed = 0
    for old_v, new_v, old_j, new_j in plan:
        try:
            _move(old_v, new_v)
        except OSError as e:
            log.error("FAILED video rename %s: %s", old_v.name, e)
            continue
        # Update JSON contents to reflect new paths, then rename the JSON.
        try:
            rec = _catalog.read_entry(old_j)
            rec["source_path"] = str(new_v)
            rec["original_name"] = new_v.name
            _catalog.write_entry(rec, old_j)
            if new_j != old_j:
                _move(old_j, new_j)
        except (OSError, ValueError) as e:
            # The video already carries its new name: the entry now points at a
            # file that is gone, so say exactly which one to fix by hand.
            log.error("video renamed to %s but its catalog entry %s was not updated: %s",
                      new_v.name, old_j, e)
        executed += 1
    print(f"Renamed {executed}/{len(plan)}")
    return executed

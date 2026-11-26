"""Build final timings: manual curation + semantic moments + audio timings.

Merges three per-project inputs into ``timings.json``:

  curation.json   hand-picked video (and optional clip range) per fragment —
                  the editor's decision, always wins. Lives in the project
                  folder, not in this module: curating a project must never
                  mean editing pipeline code.
  moments.json    machine choice — the best semantic hit per fragment that
                  report.py found while building the storyboard (shot_query ->
                  embed_index). Used for every fragment curation doesn't cover,
                  so the moment chosen at storyboard time is what gets cut,
                  instead of being re-derived later by token overlap (which
                  scored 0 across the RU narration / EN catalog gap and picked
                  effectively random clips).
  timings_v1.json audio anchors from align.py — when each fragment is spoken.

Each timings.json entry records where its clip range came from
(``clip_source``: curation / moment / curation+moment) so a wrong clip can be
traced to the decision that produced it.

Usage:
    python -m clipmaker.build_storyboard --project <name>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config
from . import catalog as _catalog
from .align import fmt_ts, load_transcript
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

CLIP_PAD_SEC = 5.0


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_curation(project_dir: Path) -> dict[int, tuple[str, float | None, float | None]]:
    """``curation.json``: {"<fragment>": "video.mp4"} or ["video.mp4", start, end].

    Hand-edited, so every entry is checked: a ValueError names the file, the
    fragment and what is wrong, instead of a KeyError or a negative-length cut
    surfacing three steps later in clip_extractor or the Vegas loader.
    """
    path = project_dir / "curation.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of fragment -> video, got {type(raw).__name__}")
    return {_fragment_number(key, path): _curation_entry(key, spec)
            for key, spec in raw.items()}


def _fragment_number(key: str, path: Path) -> int:
    """A curation key as a fragment number; ValueError for anything else."""
    if not key.strip().isdigit() or int(key) < 1:
        raise ValueError(f"{path}: key {key!r} is not a fragment number (\"1\", \"2\", ...)")
    return int(key)


def _curation_entry(key: str, spec: object) -> tuple[str, float | None, float | None]:
    """(video, start, end) of one curation entry, checked."""
    where = f"curation.json fragment {key}"
    if isinstance(spec, str):
        if not spec.strip():
            raise ValueError(f"{where}: empty file name")
        return spec, None, None
    if not (isinstance(spec, list) and len(spec) == 3):
        raise ValueError(f"{where}: expected a file name or [name, start, end], got {spec!r}")
    name, start, end = spec
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}: the first item must be the video file name, got {name!r}")
    for label, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: {label} must be a number of seconds, got {value!r}")
    if start < 0 or end <= start:
        raise ValueError(f"{where}: range {start}-{end} s is empty or negative; "
                         "the end must come after the start")
    return name, float(start), float(end)


def load_moments(project_dir: Path) -> dict[int, list[dict]]:
    """Semantic candidates per fragment from the project's moments.json ({} if absent)."""
    path = project_dir / "moments.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {f["fragment"]: f.get("candidates", []) for f in data.get("fragments", [])}


def load_catalog_index(games: list[str] | None = None) -> dict[str, dict]:
    """Catalog entries by original_name. All games by default — a storyboard
    may mix titles (AC lore episodes span three catalogs in one script)."""
    return _catalog.load_index(games or None)


# ---------------------------------------------------------------------------
# clip range selection
# ---------------------------------------------------------------------------

def pick_moment(candidates: list[dict], video: str | None = None) -> dict | None:
    """Best usable semantic candidate: has a start, interval not backwards.

    Candidates arrive sorted by score from report.py. When ``video`` is given,
    only that file's candidates count (used to attach a moment to a curated
    video that has no hand-picked range).
    """
    for c in candidates:
        if video and c.get("original_name") != video:
            continue
        if c.get("start_sec") is None:
            continue
        if c.get("timing_ok") is False:
            continue
        return c
    return None


def derive_clip_range(start: float, end: float | None, frag_dur: float,
                      video_dur: float | None, pad: float = CLIP_PAD_SEC
                      ) -> tuple[float, float]:
    """Grow a ~10s semantic moment into a clip that covers the fragment.

    The moment anchors the start; the end extends to fragment duration plus
    editing headroom. Only when the video is too short to fit does the start
    shift back from the anchor.
    """
    needed = frag_dur + pad
    clip_end = max(end if end is not None else start, start + needed)
    clip_start = start
    if video_dur:
        clip_end = min(clip_end, video_dur)
        if clip_end - clip_start < needed:
            clip_start = max(0.0, clip_end - needed)
    return round(clip_start, 2), round(clip_end, 2)


def choose_clip(num: int, frag_dur: float,
                curation: dict[int, tuple[str, float | None, float | None]],
                moments: dict[int, list[dict]],
                catalog: dict[str, dict]) -> dict:
    """One fragment's video + clip range + provenance."""
    candidates = moments.get(num, [])

    if num in curation:
        video, start, end = curation[num]
        if start is not None:
            return {"primary_video": video, "clip_start": start,
                    "clip_end": end, "clip_source": "curation"}
        moment = pick_moment(candidates, video=video)
        if moment:
            video_dur = catalog.get(video, {}).get("duration_sec")
            cs, ce = derive_clip_range(moment["start_sec"], moment.get("end_sec"),
                                       frag_dur, video_dur)
            return {"primary_video": video, "clip_start": cs, "clip_end": ce,
                    "clip_source": "curation+moment"}
        return {"primary_video": video, "clip_source": "curation"}

    moment = pick_moment(candidates)
    if moment:
        video = moment["original_name"]
        video_dur = (moment.get("duration_sec")
                     or catalog.get(video, {}).get("duration_sec"))
        cs, ce = derive_clip_range(moment["start_sec"], moment.get("end_sec"),
                                   frag_dur, video_dur)
        return {"primary_video": video, "clip_start": cs, "clip_end": ce,
                "clip_source": "moment"}

    return {"primary_video": None, "clip_source": None}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def get_transcript_for_range(segments: list[dict], start_sec: float, end_sec: float) -> str:
    """Transcript text of the segments overlapping [start_sec, end_sec]."""
    parts = []
    for seg in segments:
        if seg["end"] < start_sec:
            continue
        if seg["start"] > end_sec:
            break
        parts.append(seg["text"].strip())
    return " ".join(parts)


def run(project: str) -> Path:
    """Merge curation, semantic moments and audio anchors into timings.json; returns its path."""
    project_dir = config.project_path(project)
    timings_v1_path = project_dir / "timings_v1.json"
    transcript_path = project_dir / "transcript.json"
    out_path = project_dir / "timings.json"

    if not timings_v1_path.exists():
        raise FileNotFoundError(f"No timings_v1.json in {project_dir} - run align first.")

    timings_v1 = json.loads(timings_v1_path.read_text(encoding="utf-8"))
    curation = load_curation(project_dir)
    moments = load_moments(project_dir)
    catalog = load_catalog_index()

    transcript_segs = load_transcript(transcript_path) if transcript_path.exists() else []

    if not curation and not moments:
        log.warning("neither curation.json nor moments.json in the project - "
                    "every fragment will be uncovered.")
    log.info("Fragments: %d, curated: %d, with semantic moments: %d, catalog: %d videos",
             len(timings_v1), len(curation), len(moments), len(catalog))

    results = []
    for t in timings_v1:
        num = t["fragment"]
        start_sec = t.get("start_sec", 0)
        end_sec = t.get("end_sec", 0)
        duration = t.get("duration_sec", 0)

        choice = choose_clip(num, duration, curation, moments, catalog)
        video_name = choice.get("primary_video")
        entry = catalog.get(video_name, {}) if video_name else {}
        if video_name and video_name not in catalog:
            log.warning("Frag %2d: %s not in catalog!", num, video_name)

        transcript_text = get_transcript_for_range(transcript_segs, start_sec, end_sec)
        result = {
            "fragment": num,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration,
            "start_ts": fmt_ts(start_sec),
            "end_ts": fmt_ts(end_sec),
            "transcript_preview": transcript_text[:80] + "..." if len(transcript_text) > 80 else transcript_text,
            "video_title": entry.get("title", ""),
            "video_game": entry.get("_game", ""),
            "text_preview": t.get("text_preview", ""),
        }
        result.update(choice)
        results.append(result)

        range_label = (f"{choice.get('clip_start')}-{choice.get('clip_end')}s"
                       if "clip_start" in choice else "no range")
        log.info("Frag %2d [%s-%s] %-15s %-18s %s", num, fmt_ts(start_sec), fmt_ts(end_sec),
                 choice.get("clip_source") or "UNCOVERED", range_label,
                 ascii((video_name or "-")[:50]))

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    covered = sum(1 for r in results if r.get("primary_video"))
    log.info("Saved %d fragments to %s (%d covered, %d uncovered)",
             len(results), out_path, covered, len(results) - covered)
    return out_path


def main(argv: list[str] | None = None) -> None:
    """CLI: build a project's timings.json."""
    ap = argparse.ArgumentParser(description="Merge curation.json, moments.json and "
                                 "timings_v1.json into timings.json")
    ap.add_argument("--project", default=None)
    ap.add_argument("project_pos", nargs="?", help="project name (positional)")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    name = args.project or args.project_pos
    if not name:
        ap.error("no project given: python -m clipmaker.build_storyboard --project <name>")
    try:
        print(f"Timings: {run(name)}")
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

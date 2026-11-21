"""Compute the final Vegas timeline layout from a project's timings.json.

The Vegas side of the pipeline used to be a hand-written .cs per project with
positions, durations and the audio offset hardcoded — every desync was invisible
until the user saw it on the timeline. This module moves the whole layout
decision into Python, where it can be validated and reviewed BEFORE Vegas runs:

  timings.json  ->  layout.txt        (machine: consumed by import_layout.cs)
                    layout_review.md  (human: table to eyeball before Vegas)

Every event is quantized to whole frames at the declared fps and stored as
integer milliseconds — the loader uses Timecode.FromMilliseconds, so no fps
guessing happens on the C# side. All times are relative to the narration start;
the loader finds the narration audio event on the timeline and adds its start,
so moving the mp3 never desyncs the video again.

Validation fails loudly (exit 1, no layout written) on: missing media files,
overlapping events, non-positive event lengths. Clips shorter than their
fragment are allowed but flagged as gaps in the review.

The Vegas project frame rate and the narration file come from the project's
vegas.json (see clipmaker.vegas_takes), so they are not typed per run; --fps and
--audio override it. Its raw-take lag table is copied into the layout.txt
header as take= lines: when the timeline holds the raw takes instead of the
narration mp3, the loader anchors on them.

Usage:
    python -m clipmaker.layout --project <name> [--fps 60] [--audio narration.mp3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .vegas_takes import TAKES_COMMENT, TakeSegment, VegasConfig, find_vegas_config

LAYOUT_VERSION = 2
DEFAULT_TRACK = "B-Roll (layout)"
DEFAULT_FADE_MS = 400


@dataclass(frozen=True)
class LayoutEvent:
    fragment: int
    start_ms: int         # narration-relative, frame-quantized
    length_ms: int        # frame-quantized
    take_offset_ms: int   # offset into the media file
    path: Path
    label: str
    gap_ms: int = 0       # uncovered tail of the fragment (clip too short)


@dataclass
class LayoutResult:
    events: list[LayoutEvent] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)  # fragments without video
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------

def quantize_ms(sec: float, fps: float) -> int:
    """Seconds -> milliseconds snapped to the nearest whole frame at fps."""
    frames = round(sec * fps)
    return round(frames / fps * 1000)


def fmt_tc(ms: int, fps: float) -> str:
    """HH:MM:SS:FF for the review table (nearest frame; display only).

    Seconds come from the wall clock, frames from the remainder: counting
    frames and dividing by the nominal rate drifts at 59.94/29.97 (half a
    second by the ninth minute), and the table stops matching the timeline."""
    fpsi = max(1, round(fps))
    total_sec, rem_ms = divmod(ms, 1000)
    ff = min(round(rem_ms / 1000 * fps), fpsi - 1)
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"


# --------------------------------------------------------------------------
# media resolution
# --------------------------------------------------------------------------

def probe_duration_sec(path: Path) -> float | None:
    """Container duration via ffprobe; None when the probe fails."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def load_catalog_index(games: list[str]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for game in games:
        game_cfg = config.GAMES.get(game)
        if not game_cfg:
            continue
        cat_dir = config.CATALOG_DIR / game_cfg["catalog_subdir"]
        if not cat_dir.exists():
            continue
        for p in cat_dir.rglob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8-sig"))
                index[data.get("original_name", "")] = data
            except (json.JSONDecodeError, OSError):
                continue
    return index


def resolve_source(video_name: str, footage_dir: Path,
                   catalog: dict[str, dict]) -> Path | None:
    """Same resolution order clip_extractor uses: footage/, catalog source_path,
    source_path's folder + the (possibly renamed) original_name."""
    entry = catalog.get(video_name, {})
    orig = entry.get("source_path", "")
    candidates = [footage_dir / video_name]
    if orig:
        candidates.append(Path(orig))
        candidates.append(Path(orig).parent / video_name)
    return next((c for c in candidates if c.exists()), None)


def find_precut_clip(clips_dir: Path, fragment: int) -> Path | None:
    if not clips_dir.is_dir():
        return None
    matches = sorted(clips_dir.glob(f"{fragment:02d}_*.mp4"))
    return matches[0] if matches else None


# --------------------------------------------------------------------------
# layout construction
# --------------------------------------------------------------------------

def build_layout(timings: list[dict], fps: float, clips_dir: Path,
                 footage_dir: Path, catalog: dict[str, dict],
                 probe=probe_duration_sec) -> LayoutResult:
    res = LayoutResult()

    for t in sorted(timings, key=lambda x: x["start_sec"]):
        num = t["fragment"]
        # Fragment length comes from the quantized start/end boundaries, not
        # from duration_sec: ends equal the next fragment's start, so lengths
        # derived this way cannot create rounding overlaps between neighbours.
        start_ms = quantize_ms(float(t["start_sec"]), fps)
        frag_end = float(t.get("end_sec", t["start_sec"] + t["duration_sec"]))
        frag_len_ms = quantize_ms(frag_end, fps) - start_ms
        preview = (t.get("text_preview") or t.get("transcript_preview") or "")
        label = (t.get("label") or f"fr {num:02d} {preview}").replace("|", "/").strip()[:80]
        video_name = t.get("primary_video")

        if not video_name:
            res.uncovered.append(f"fr {num:02d} [{t.get('start_ts', '?')}] no video assigned")
            continue

        precut = find_precut_clip(clips_dir, num)
        if precut is not None:
            path = precut
            take_offset_sec = 0.0
            clip_dur = probe(precut)
            if clip_dur is None:
                res.errors.append(f"fr {num:02d}: cannot probe duration of {precut.name}")
                continue
        else:
            path = resolve_source(video_name, footage_dir, catalog)
            if path is None:
                res.errors.append(f"fr {num:02d}: media not found: {video_name}")
                continue
            if "clip_start" not in t or "clip_end" not in t:
                res.errors.append(
                    f"fr {num:02d}: no pre-cut clip and no clip_start/clip_end "
                    f"in timings — nothing defines which part of "
                    f"{video_name} to show")
                continue
            take_offset_sec = float(t["clip_start"])
            clip_dur = float(t["clip_end"]) - take_offset_sec
            source_dur = catalog.get(video_name, {}).get("duration_sec")
            if source_dur and t["clip_end"] > source_dur + 1:
                res.warnings.append(
                    f"fr {num:02d}: clip_end {t['clip_end']}s is past the end "
                    f"of {video_name} ({source_dur}s)")

        length_ms = min(frag_len_ms, quantize_ms(clip_dur, fps))
        gap_ms = max(0, frag_len_ms - length_ms)

        if length_ms <= 0:
            res.errors.append(f"fr {num:02d}: non-positive event length "
                              f"({length_ms / 1000:.2f}s)")
            continue
        if gap_ms > 500:
            res.warnings.append(
                f"fr {num:02d}: clip covers only {length_ms / 1000:.1f}s of a "
                f"{frag_len_ms / 1000:.1f}s fragment — "
                f"{gap_ms / 1000:.1f}s gap at the end")

        res.events.append(LayoutEvent(
            fragment=num, start_ms=start_ms, length_ms=length_ms,
            take_offset_ms=quantize_ms(take_offset_sec, fps),
            path=path, label=label, gap_ms=gap_ms,
        ))

    res.errors.extend(find_overlaps(res.events, fps))
    return res


def events_to_timings(events: list[dict]) -> list[dict]:
    """Explicit placements -> rows build_layout understands.

    Map clips and document stills go on the spoken second, across fragment
    boundaries and several per fragment, which the one-event-per-fragment
    timings.json cannot express. Each placement is {start_sec, end_sec, path,
    clip_start, label?}; it becomes its own pseudo-fragment, so the same
    validation (missing media, overlaps) applies.
    """
    rows = []
    ordered = sorted(events, key=lambda e: float(e["start_sec"]))
    for i, e in enumerate(ordered, 1):
        start, end = float(e["start_sec"]), float(e["end_sec"])
        clip_start = float(e.get("clip_start", 0.0))
        rows.append({
            "fragment": i,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "primary_video": str(e["path"]),
            "clip_start": clip_start,
            "clip_end": clip_start + (end - start),
            "label": e.get("label") or Path(e["path"]).stem,
        })
    return rows


def find_overlaps(events: list[LayoutEvent], fps: float) -> list[str]:
    """Events sharing timeline time on the single track. One frame of slack."""
    frame_ms = 1000 / fps
    problems = []
    ordered = sorted(events, key=lambda e: e.start_ms)
    for prev, cur in zip(ordered, ordered[1:]):
        overlap_ms = prev.start_ms + prev.length_ms - cur.start_ms
        if overlap_ms > frame_ms:
            problems.append(
                f"OVERLAP: fr {prev.fragment:02d} runs {overlap_ms / 1000:.2f}s "
                f"into fr {cur.fragment:02d} (fr {prev.fragment:02d} ends "
                f"{(prev.start_ms + prev.length_ms) / 1000:.2f}s, "
                f"fr {cur.fragment:02d} starts {cur.start_ms / 1000:.2f}s)")
    return problems


# --------------------------------------------------------------------------
# output rendering
# --------------------------------------------------------------------------

def render_layout_txt(res: LayoutResult, project: str, audio: str,
                      fps: float, track: str, fade_ms: int,
                      takes: tuple[TakeSegment, ...] = ()) -> str:
    lines = [
        f"# clipmaker layout v{LAYOUT_VERSION} — {project}",
        "# times are milliseconds relative to the narration start;",
        "# the loader adds the narration audio event's own timeline position.",
        f"project={project}",
        f"audio={audio}",
        f"fps={fps:g}",
        f"track={track}",
        f"fade_ms={fade_ms}",
    ]
    if takes:
        lines += [TAKES_COMMENT] + [t.line() for t in takes]
    lines.append("---")
    for e in res.events:
        lines.append(f"{e.start_ms}|{e.length_ms}|{e.take_offset_ms}"
                     f"|{e.label}|{e.path}")
    return "\n".join(lines) + "\n"


def render_review_md(res: LayoutResult, project: str, fps: float) -> str:
    lines = [
        f"# Layout review: {project}",
        "",
        "Times are relative to the narration start (the loader adds the mp3's",
        "position on the Vegas timeline). Check every row BEFORE running the",
        "Vegas script — this table is exactly what will be placed.",
        "",
        "| Frag | Timeline start | Length | Media | Take offset | Note |",
        "|---|---|---|---|---|---|",
    ]
    for e in res.events:
        note = f"{e.gap_ms / 1000:.1f}s gap at end" if e.gap_ms > 500 else ""
        lines.append(
            f"| {e.fragment:02d} | {fmt_tc(e.start_ms, fps)} "
            f"| {e.length_ms / 1000:.2f}s | {e.path.name} "
            f"| {e.take_offset_ms / 1000:.1f}s | {note} |")
    if res.uncovered:
        lines += ["", "## Fragments without video", ""]
        lines += [f"- {u}" for u in res.uncovered]
    if res.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in res.warnings]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def resolve_fps(cli_fps: float | None, cfg: VegasConfig | None) -> float:
    """--fps wins over vegas.json; one of them must say it."""
    cfg_fps = cfg.fps if cfg else None
    if cli_fps is not None:
        if cfg_fps is not None and abs(cli_fps - cfg_fps) > 1e-6:
            print(f"  WARN  --fps {cli_fps:g} overrides fps {cfg_fps:g} from {cfg.path}")
        return cli_fps
    if cfg_fps is not None:
        return cfg_fps
    sys.exit('No frame rate: put "fps": <Vegas project fps> into the project\'s '
             "vegas.json (see clipmaker.vegas_takes), or pass --fps.")


def detect_audio(project_dir: Path, override: str | None) -> str:
    if override:
        return override
    mp3s = sorted(p.name for p in project_dir.glob("*.mp3"))
    if len(mp3s) == 1:
        return mp3s[0]
    if not mp3s:
        sys.exit("No .mp3 in the project folder — pass --audio <file.mp3> "
                 "(the loader matches it against the narration event in Vegas).")
    sys.exit("Several .mp3 files in the project folder, pass --audio with one of:\n  "
             + "\n  ".join(mp3s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--fps", type=float, default=None,
                    help="Vegas project frame rate (e.g. 60, 59.94) — events are "
                         "quantized to whole frames at this rate. Default: fps "
                         "from the project's vegas.json")
    ap.add_argument("--audio", default=None,
                    help="Narration file name as it appears in the Vegas project "
                         "(default: audio from vegas.json, else the folder's only .mp3)")
    ap.add_argument("--track", default=DEFAULT_TRACK)
    ap.add_argument("--fade-ms", type=int, default=DEFAULT_FADE_MS)
    ap.add_argument("--events", default=None,
                    help="JSON list of explicit placements {start_sec, end_sec, "
                         "path, clip_start, label} (relative to the project dir) "
                         "used instead of timings.json — for map clips and stills "
                         "placed on the spoken second")
    args = ap.parse_args()

    project_dir = config.project_path(args.project)
    if not project_dir.is_dir():
        sys.exit(f"No project folder {project_dir}")
    try:
        vegas_cfg = find_vegas_config(project_dir)
    except (ValueError, OSError) as exc:
        sys.exit(f"Bad vegas.json: {exc}")
    fps = resolve_fps(args.fps, vegas_cfg)
    takes = vegas_cfg.takes if vegas_cfg else ()
    clips_dir = project_dir / "clips"
    if args.events:
        events_path = project_dir / args.events
        if not events_path.exists():
            sys.exit(f"No {events_path}")
        timings = events_to_timings(json.loads(events_path.read_text(encoding="utf-8")))
        # pseudo-fragment numbers must not pick up pre-cut NN_*.mp4 clips
        clips_dir = project_dir / "__no_precut__"
    else:
        timings_path = project_dir / "timings.json"
        if not timings_path.exists():
            sys.exit(f"No timings.json in {project_dir} — run build_storyboard first.")
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
    games = sorted({t.get("video_game") for t in timings if t.get("video_game")})
    catalog = load_catalog_index(games)
    audio = detect_audio(project_dir, args.audio or (vegas_cfg.audio if vegas_cfg else None))

    res = build_layout(timings, fps, clips_dir,
                       project_dir / "footage", catalog)

    for w in res.warnings:
        print(f"  WARN  {w}")
    if res.errors:
        print(f"\nLayout NOT written — {len(res.errors)} error(s):")
        for e in res.errors:
            print(f"  ERROR {e}")
        sys.exit(1)

    layout_path = project_dir / "layout.txt"
    review_path = project_dir / "layout_review.md"
    layout_path.write_text(
        render_layout_txt(res, args.project, audio, fps,
                          args.track, args.fade_ms, takes), encoding="utf-8")
    review_path.write_text(render_review_md(res, args.project, fps),
                           encoding="utf-8")

    print(f"\nEvents: {len(res.events)}, uncovered fragments: {len(res.uncovered)}, "
          f"warnings: {len(res.warnings)}, fps: {fps:g}, take segments: {len(takes)}")
    print(f"Layout:  {layout_path}")
    print(f"Review:  {review_path}")
    print("Review the table, then run vegas_scripts/import_layout.cs in Vegas "
          "and pick this layout.txt.")


if __name__ == "__main__":
    main()

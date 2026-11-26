"""
Stage 1 - VLM video analyzer.

Takes a video file, samples frames, runs Qwen2-VL to produce a structured
JSON description (title, summary, tags, timeline) and writes it to the
per-game catalog folder. Optionally suggests a new filename.

The model step lives in vlm.py, filename suggestion and renaming in naming.py;
this module holds the per-video pipeline, the batch and the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import catalog as _catalog
from . import config
from .game_vocab import match_entities, match_region, render_vocab_hint, vocab_for
from .log import add_verbose_flag, setup_logging
from .naming import _is_noise_tag, _to_tag, regenerate_suggestions, rename_from_catalog, suggest_filename
from . import vlm
from .vlm import _fmt_mmss, clamp_scenes

log = logging.getLogger(__name__)


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _cuda_available() -> bool:
    """True when torch sees a CUDA device (imports torch: the vlm extra)."""
    import torch
    return torch.cuda.is_available()


def _run_tool(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    """Run ffmpeg/ffprobe; RuntimeError carrying the tool's own stderr on failure.

    ``check=True`` used to raise CalledProcessError, whose message is only the
    command line and exit code: "returned non-zero exit status 1" says nothing
    about the moov atom that is missing or the file that is not a video.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError(f"{cmd[0]} is not on PATH (needed to {what})") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "no output on stderr"
        raise RuntimeError(f"{cmd[0]} failed to {what} (exit {result.returncode}): "
                           f"{stderr[-800:]}")
    return result


def get_video_duration(video_path: Path) -> float:
    """Duration in seconds from ffprobe; RuntimeError with ffprobe's message otherwise."""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = _run_tool(cmd, f"read the duration of {video_path.name}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"ffprobe reported no duration for {video_path.name}: "
                           f"{result.stdout[:200]!r}") from exc


def extract_frames(video_path: Path, output_dir: Path, interval: int, max_size: int) -> list[tuple[Path, int]]:
    """Extract frames at given interval. Returns [(frame_path, timestamp_sec), ...]."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("frame_*.jpg"):
        old.unlink()

    pattern = output_dir / "frame_%06d.jpg"
    vf = (
        f"fps=1/{interval},"
        f"scale='if(gt(iw,ih),{max_size},-2)':'if(gt(ih,iw),{max_size},-2)'"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", vf,
        "-q:v", "3",
        "-y",
        str(pattern),
    ]
    _run_tool(cmd, f"extract frames from {video_path.name}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return [(f, idx * interval) for idx, f in enumerate(frames)]


def remove_frames(work_dir: Path) -> None:
    """Delete the extracted frames and their folder (kept if anything else is in it)."""
    for frame in work_dir.glob("frame_*.jpg"):
        frame.unlink(missing_ok=True)
    try:
        work_dir.rmdir()
    except OSError:
        pass  # missing, or holds files that are not ours


def extract_path_hints(video_path: Path, game: str) -> dict:
    """Derive authoritative location/entity labels from the filesystem path.

    - location: parent folder name (under the game's root) matched against
      the region vocab. Falls back to the raw folder name.
    - entities: bosses / enemies / concepts mentioned in the filename.
    """
    vocab = vocab_for(game) or {}
    hints: dict[str, Any] = {"location": None, "location_folder": None, "entities": []}

    # Find the closest game root from config; use first parent under it.
    roots = config.GAMES.get(game, {}).get("video_dirs", [])
    parent_folder = None
    for root in roots:
        try:
            rel = video_path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) > 1:
            parent_folder = parts[0]
            break

    if parent_folder is None and video_path.parent != video_path:
        parent_folder = video_path.parent.name

    if parent_folder:
        hints["location_folder"] = parent_folder
        matched = match_region(parent_folder, vocab)
        hints["location"] = matched or parent_folder

    # Match entities in filename against vocab categories that carry
    # recognisable proper nouns.
    filename_text = video_path.stem.replace("_", " ")
    entity_items = []
    for cat in ("bosses", "enemies", "regions", "ships", "factions", "events", "landmarks"):
        entity_items.extend(vocab.get(cat, []))
    hints["entities"] = match_entities(filename_text, entity_items)
    hints["game_description"] = vocab.get("description", "")

    return hints


def _parse_mmss(s: str) -> int | None:
    m = re.match(r"^\s*(\d+):(\d{1,2})\s*$", str(s))
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", str(s))
    if m:
        return int(m.group(1))
    return None


def _match_vocab_in_line(line: str, vocab_items: list[str]) -> list[str]:
    """Return canonical vocab items that appear in the OCR line (substring,
    case-insensitive). Longer items take precedence."""
    if not line or not vocab_items:
        return []
    norm_line = line.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for item in sorted(vocab_items, key=len, reverse=True):
        key = item.lower()
        if key in seen or len(key) < 3:
            continue
        if key in norm_line:
            hits.append(item)
            seen.add(key)
    return hits


def build_ocr_timeline(per_frame_ocr: list[list[str]], interval: int,
                       duration_sec: int, vocab: dict | None) -> list[dict]:
    """Build a deterministic per-frame timeline from OCR text.

    Each frame at timestamp i*interval produces one segment. Adjacent frames
    with identical caption are merged. Captions go through chrome filtering
    and phrase collapsing so UI clutter (Route/General/Scan Results/etc.) is
    dropped in favour of signal (star systems, POIs, warp status)."""
    from .ocr import filter_chrome

    vocab_items: list[str] = []
    if vocab:
        for items in vocab.values():
            if isinstance(items, list):
                vocab_items.extend(items)

    segments: list[dict] = []
    for i, lines in enumerate(per_frame_ocr):
        start = i * interval
        end = min((i + 1) * interval, duration_sec)
        if end <= start:
            continue

        cleaned = filter_chrome(lines) if lines else []
        caption = "; ".join(cleaned)

        seg_tags: list[str] = []
        seen_tags: set[str] = set()
        for line in cleaned:
            for hit in _match_vocab_in_line(line, vocab_items):
                tag = _to_tag(hit)
                if tag and tag not in seen_tags:
                    seg_tags.append(tag)
                    seen_tags.add(tag)

        if segments and segments[-1]["caption"] == caption:
            segments[-1]["end"] = _fmt_mmss(end)
            continue

        segments.append({
            "start": _fmt_mmss(start),
            "end": _fmt_mmss(end),
            "caption": caption,
            "tags": seg_tags,
        })
    return segments


def clean_analysis(analysis: dict, duration_sec: int) -> dict:
    """Post-filter: drop segments past video duration, normalize timestamps,
    deduplicate identical consecutive captions, strip noise tags."""
    tl = analysis.get("timeline") or []
    cleaned = []
    last_caption = None
    for seg in tl:
        start = _parse_mmss(seg.get("start", ""))
        end = _parse_mmss(seg.get("end", ""))
        if start is None or end is None:
            continue
        if start >= duration_sec:
            break
        end = min(end, duration_sec)
        if end <= start:
            continue
        caption = str(seg.get("caption", "")).strip()
        if caption and caption == last_caption:
            cleaned[-1]["end"] = _fmt_mmss(end)
            continue
        seg_tags = [t for t in seg.get("tags", [])
                    if t and not _is_noise_tag(t)]
        cleaned.append({
            "start": _fmt_mmss(start),
            "end": _fmt_mmss(end),
            "caption": caption,
            "tags": seg_tags,
        })
        last_caption = caption

    analysis["timeline"] = cleaned

    # Clean top-level tags
    analysis["tags"] = [
        t for t in analysis.get("tags", [])
        if t and not _is_noise_tag(t)
    ]
    return analysis


def inject_path_facts(analysis: dict, path_hints: dict) -> dict:
    """Merge location + confirmed entities into tags (both top-level and
    per-segment), so path-derived ground truth cannot be lost even if the
    VLM hallucinated or ignored the hint."""
    forced: list[str] = []
    if path_hints.get("location"):
        forced.append(_to_tag(path_hints["location"]))
    for ent in path_hints.get("entities") or []:
        forced.append(_to_tag(ent))

    tags = list(analysis.get("tags") or [])
    for t in forced:
        if t and t not in tags:
            tags.insert(0, t)
    analysis["tags"] = tags

    for seg in analysis.get("timeline") or []:
        seg_tags = list(seg.get("tags") or [])
        for t in forced:
            if t and t not in seg_tags:
                seg_tags.append(t)
        seg["tags"] = seg_tags

    # Persist only the path-derived facts, not the static game description.
    analysis["path_hints"] = {
        k: v for k, v in path_hints.items() if k != "game_description"
    }
    return analysis


def frame_interval(duration_sec: int, interval: int | None, max_frames: int | None) -> int:
    """Seconds between sampled frames: the one asked for, else the smallest that
    keeps the video within the frame cap (never below VLM_FRAME_INTERVAL_SEC)."""
    if interval is not None:
        return interval
    frame_cap = max_frames or config.VLM_FRAME_MAX_FRAMES
    needed = -(-duration_sec // max(1, frame_cap))  # ceil div
    return max(config.VLM_FRAME_INTERVAL_SEC, needed)


def analyze_video(
    video_path: Path,
    game: str,
    interval: int | None = None,
    max_frames: int | None = None,
    device: str = "cuda",
    keep_frames: bool = False,
) -> dict:
    """Catalog record of one video: frames -> OCR -> VLM -> cleaned, path-grounded JSON."""
    duration_sec = int(get_video_duration(video_path))
    interval = frame_interval(duration_sec, interval, max_frames)

    work_dir = config.VLM_WORK_DIR / video_path.stem
    log.info("Duration %ds, sampling 1 frame / %ds", duration_sec, interval)
    try:
        return _analyze_frames(video_path, game, work_dir, duration_sec, interval, device)
    finally:
        # Also on failure: a crashed or interrupted VLM call used to leave every
        # extracted frame of the video in _vlm_frames/.
        if not keep_frames:
            remove_frames(work_dir)


def _read_screen_text(frame_paths: list[Path]) -> tuple[list[str], list[list[str]]]:
    """(unique OCR lines of the video, OCR lines per frame); empty with OCR off."""
    if not config.OCR_ENABLED:
        return [], [[] for _ in frame_paths]
    from .ocr import dedupe_preserving_order, extract_frame_text, filter_noise
    from .ocr import release as release_ocr

    raw_per_frame = extract_frame_text(
        frame_paths,
        languages=config.OCR_LANGUAGES,
        min_confidence=config.OCR_MIN_CONFIDENCE,
    )
    per_frame_ocr = [filter_noise(lines) for lines in raw_per_frame]
    ocr_lines = dedupe_preserving_order(per_frame_ocr)
    frames_with_text = sum(1 for ln in per_frame_ocr if ln)
    log.info("OCR: %d unique lines  |  %d/%d frames with text",
             len(ocr_lines), frames_with_text, len(frame_paths))
    release_ocr()
    return ocr_lines, per_frame_ocr


def _analyze_frames(video_path: Path, game: str, work_dir: Path, duration_sec: int,
                    interval: int, device: str) -> dict:
    """The part of analyze_video that works on extracted frames."""
    frames = extract_frames(video_path, work_dir, interval, config.VLM_FRAME_MAX_SIZE)
    if not frames:
        raise RuntimeError("No frames extracted")
    log.info("%d frames -> VLM", len(frames))

    frame_paths = [f for f, _ in frames]
    vocab = vocab_for(game)
    path_hints = extract_path_hints(video_path, game)
    if path_hints["location"] or path_hints["entities"]:
        log.info("Path hints: location=%r  entities=%s",
                 path_hints["location"], path_hints["entities"])
    ocr_lines, per_frame_ocr = _read_screen_text(frame_paths)

    request = vlm.VlmRequest(
        game_hint=game.replace("_", " "),
        vocab_hint=render_vocab_hint(vocab) if vocab else "",
        name_hint=video_path.stem,
        path_hints=path_hints,
        ocr_lines=ocr_lines[:60],
        device=device,
    )
    reply = vlm.ask_vlm(frame_paths, duration_sec, interval, request, config.VLM_CHUNK_SIZE)
    analysis = _ground_analysis(reply, duration_sec, interval, per_frame_ocr, vocab, path_hints)
    return _build_record(video_path, game, duration_sec, interval, len(frames), ocr_lines, analysis)


def _ground_analysis(reply: dict, duration_sec: int, interval: int,
                     per_frame_ocr: list[list[str]], vocab: dict | None,
                     path_hints: dict) -> dict:
    """The VLM reply with clean tags, the OCR timeline, clamped scenes and path facts.

    The timeline comes from OCR, never from the model: VLM timestamps drift.
    """
    scenes = reply.get("scenes", [])
    analysis = clean_analysis({k: v for k, v in reply.items() if k not in ("scenes", "timeline")},
                              duration_sec)
    analysis["timeline"] = build_ocr_timeline(per_frame_ocr, interval, duration_sec, vocab)
    if scenes:
        analysis["scenes"] = clamp_scenes(scenes, duration_sec)
    return inject_path_facts(analysis, path_hints)


def _build_record(video_path: Path, game: str, duration_sec: int, interval: int,
                  n_frames: int, ocr_lines: list[str], analysis: dict) -> dict:
    """The catalog JSON: provenance fields first, then the analysis, then the name."""
    record = {
        "source_path": str(video_path),
        "original_name": video_path.name,
        "duration_sec": duration_sec,
        "frame_interval_sec": interval,
        "n_frames_analyzed": n_frames,
        "model": config.VLM_MODEL,
        "analyzed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "game": game,
        "ocr_text": ocr_lines,
        **analysis,
    }
    # Must run on the assembled record, not on `analysis` - the game key lives
    # here, and regenerate_suggestions() feeds it the record too. Passing
    # `analysis` silently dropped the game display name from every new name.
    return {**record, "suggested_filename": suggest_filename(record, video_path)}


_DIR_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")


def _sanitize_dirname(name: str) -> str:
    """Lowercase, spaces->underscores, drop non-alnum. Returns '' if empty."""
    s = re.sub(r"\s+", "_", name.strip().lower())
    s = _DIR_SANITIZE_RE.sub("", s)
    return s


def _catalog_subpath_for(video_path: Path, game: str) -> Path:
    """Mirror the source folder structure under the game's catalog subdir.

    E.g. `<RAW_FOOTAGE_DIR>/My Game/Season 2 Bosses/foo.mp4` (game root is
    `My Game`) -> catalog/my_game_videos/season_2 (if an override is
    configured, else the sanitized source folder name).
    """
    cfg = config.GAMES.get(game, {})
    overrides: dict[str, str] = cfg.get("subdir_overrides") or {}
    rel_parent: Path | None = None
    for root in cfg.get("video_dirs", []):
        try:
            rel = video_path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            continue
        rel_parent = rel.parent
        break

    if rel_parent is None or str(rel_parent) in (".", ""):
        return Path()

    parts: list[str] = []
    for raw in rel_parent.parts:
        name = overrides.get(raw) or _sanitize_dirname(raw)
        if name:
            parts.append(name)
    return Path(*parts) if parts else Path()


def catalog_json_path(video_path: Path, game: str) -> Path:
    """Where the catalog entry for a video lives. Pure: creates nothing."""
    subdir = config.GAMES.get(game, {}).get("catalog_subdir", f"{game}_videos")
    out_dir = config.CATALOG_DIR / subdir / _catalog_subpath_for(video_path, game)
    return out_dir / f"{video_path.stem}.json"


def catalog_path_for(video_path: Path, game: str) -> Path:
    """catalog_json_path with its folder created, ready to write."""
    out = catalog_json_path(video_path, game)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def collect_videos(roots: list[Path]) -> list[Path]:
    """Video files under the given files and folders, recursively, sorted and unique."""
    out = []
    for r in roots:
        if not r.exists():
            log.warning("not found: %s", r)
            continue
        if r.is_file() and r.suffix.lower() in VIDEO_EXTS:
            out.append(r)
            continue
        for ext in VIDEO_EXTS:
            out.extend(r.rglob(f"*{ext}"))
    return sorted(set(out))


def catalog_collisions(game: str, videos: list[Path]) -> list[str]:
    """Videos that would share one catalog JSON, as messages; [] when none do.

    The entry is named after the file stem, so "Clip.mp4" and "Clip.mkv" in one
    folder map to the same Clip.json: the second either overwrote the first or
    was skipped as "cached" under the other video's description. Also caught:
    an existing entry that already describes a different file of that stem.
    """
    by_json: dict[Path, list[Path]] = {}
    for v in videos:
        by_json.setdefault(catalog_json_path(v, game), []).append(v)
    problems = [f"{out.name}: " + ", ".join(str(v) for v in group)
                for out, group in by_json.items() if len(group) > 1]
    for out, (video, *rest) in by_json.items():
        if rest or not out.exists():
            continue
        try:
            owner = str(_catalog.read_entry(out).get("original_name") or "")
        except (OSError, ValueError):
            continue  # unreadable entries are re-analyzed or reported elsewhere
        if owner and owner.lower() != video.name.lower():
            problems.append(f"{out.name} already describes {owner}, not {video}")
    return problems


def index_plan(game: str, videos: list[Path], force: bool) -> list[tuple[Path, Path, bool]]:
    """(video, catalog json, will_analyze) per video. Pure: touches nothing."""
    plan = []
    for v in videos:
        out = catalog_json_path(v, game)
        plan.append((v, out, force or not out.exists()))
    return plan


def print_index_plan(plan: list[tuple[Path, Path, bool]]) -> int:
    """Dry-run report of an indexing batch. Returns the number to analyze."""
    todo = sum(1 for _, _, run in plan if run)
    print(f"DRY-RUN: {todo} of {len(plan)} video(s) would be analyzed "
          f"(no model is loaded, nothing is written)")
    for v, out, run in plan:
        print(f"  {'ANALYZE' if run else 'skip   '}  {v.name}  ->  {out}")
    return todo


def run_batch(game: str, paths: list[Path] | None, force: bool,
              interval: int | None, max_frames: int | None = None,
              dry_run: bool = False) -> int:
    """Analyze every video of a batch into the catalog; returns how many failed.

    One failing video does not stop the batch, but it is logged as an error
    and counted, so the caller can exit non-zero instead of reporting success.
    """
    game_cfg = config.game_config(game)

    # An explicit path list is a caller's precise instruction, so a path that does
    # not resolve is a bug in the caller, not something to skip past. collect_videos
    # only warns; continuing on the survivors once cost a catalog-wide --force
    # re-analysis, because a mangled argument list still contained "." and that
    # expands to the whole tree.
    if paths:
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            shown = "\n".join(f"  {p}" for p in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} of {len(paths)} paths do not exist, e.g.:\n{shown}")

    roots = paths or game_cfg["video_dirs"]
    videos = collect_videos(roots)
    if not videos:
        log.warning("No videos found.")
        return 0

    collisions = catalog_collisions(game, videos)
    if collisions:
        raise ValueError("these videos share a catalog entry name; rename one of each "
                         "pair before indexing:\n  " + "\n  ".join(collisions))

    # A dry run stops here: no device probe, no model, no catalog folders.
    if dry_run:
        print_index_plan(index_plan(game, videos, force))
        return 0

    device = "cuda" if _cuda_available() else "cpu"
    log.info("Device: %s  |  Game: %s  |  Videos: %d", device, game, len(videos))

    failed = 0
    for i, v in enumerate(videos, 1):
        out = catalog_path_for(v, game)
        if out.exists() and not force:
            log.info("[%d/%d] skip (cached): %s", i, len(videos), v.name)
            continue
        log.info("[%d/%d] %s", i, len(videos), v.name)
        try:
            rec = analyze_video(v, game=game, interval=interval,
                                max_frames=max_frames, device=device)
            _catalog.write_entry(rec, out)
            log.info("-> %s  | suggested: %s", out.name, rec["suggested_filename"])
        except Exception as e:  # noqa: BLE001 - batch boundary: keep the other videos going
            failed += 1
            log.error("FAILED %s: %s", v.name, e)
            log.debug("traceback of the failure", exc_info=True)
    if failed:
        log.error("%d of %d video(s) failed", failed, len(videos))
    return failed


def main(argv: list[str] | None = None) -> None:
    """CLI: index videos into the catalog, or rename them from their entries."""
    ap = argparse.ArgumentParser(description="VLM video analyzer (Stage 1)")
    ap.add_argument("paths", nargs="*", type=Path,
                    help="Video files or directories. If empty, uses GAMES[game].video_dirs.")
    ap.add_argument("--game", default=config.DEFAULT_GAME, choices=list(config.GAMES))
    ap.add_argument("--interval", type=int, default=None,
                    help=f"Seconds between sampled frames (default {config.VLM_FRAME_INTERVAL_SEC})")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Max frames to extract (overrides adaptive cap; VLM processes in chunks of 50)")
    ap.add_argument("--force", action="store_true", help="Re-analyze even if catalog entry exists")
    ap.add_argument("--regenerate-suggestions", action="store_true",
                    help="Re-run suggest_filename on every cached JSON (no VLM call)")
    ap.add_argument("--rename", action="store_true",
                    help="Rename video files to match suggested_filename (and update JSONs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the plan without doing anything: with --rename the "
                         "renames, with --regenerate-suggestions the changed names, "
                         "otherwise which videos would be analyzed. Never loads the "
                         "model and never writes.")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    if args.regenerate_suggestions:
        regenerate_suggestions(args.game, dry_run=args.dry_run)
        return
    if args.rename:
        rename_from_catalog(args.game, dry_run=args.dry_run)
        return

    try:
        failed = run_batch(args.game, args.paths or None, force=args.force,
                           interval=args.interval, max_frames=args.max_frames,
                           dry_run=args.dry_run)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

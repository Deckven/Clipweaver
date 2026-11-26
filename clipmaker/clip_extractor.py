"""Extract one clip per storyboard fragment for Sony Vegas editing.

Cuts exactly the ranges recorded in ``timings.json`` (curation or the semantic
moment propagated by build_storyboard). A fragment without a range is reported
as uncut — the old token-overlap fallback scored 0 across the RU narration /
EN catalog gap and picked effectively random clips, so guessing here is worse
than an honest gap.

Usage:
    python -m clipmaker.clip_extractor <project-name>
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path

from . import config
from . import catalog as _catalog
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)



def _sec_to_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_catalog_index(game: str) -> dict[str, dict]:
    """Catalog entries of one game by original_name."""
    return _catalog.load_index([game])


def build_cut_cmd(source: Path, start_sec: float, end_sec: float, out: Path,
                  vcodec: str) -> list[str]:
    """Frame-accurate cut command.

    ``-ss`` before ``-i`` with a re-encode is exact: ffmpeg seeks to the
    keyframe before the point, decodes forward and drops frames until the
    target. The old ``-c copy`` path snapped the start back to the previous
    keyframe — on long-GOP game captures that shifted clips by up to ~10 s,
    which surfaced as "the video does not match the narration".
    """
    codec_args = {
        "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"],
        "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"],
    }[vcodec]
    return [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(source),
        "-t", f"{end_sec - start_sec:.3f}",
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]


def extract_clip(source: Path, start_sec: float, end_sec: float, out: Path) -> bool:
    """Cut [start_sec, end_sec) frame-accurately; NVENC first, libx264 fallback."""
    last_err = ""
    for vcodec in ("h264_nvenc", "libx264"):
        cmd = build_cut_cmd(source, start_sec, end_sec, out, vcodec)
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_err = str(exc)
            continue
        if result.returncode == 0:
            return True
        last_err = result.stderr.decode(errors="replace")[-400:]
    log.error("ffmpeg failed for %s, stderr tail:\n%s", out.name, last_err)
    return False


resolve_source = _catalog.resolve_source



def clear_fragment_clips(clips_dir: Path, fragment: int) -> list[str]:
    """Delete every earlier cut of this fragment before cutting it again.

    Without this a changed pick left the old NN_<old>.mp4 next to the new one,
    and the old one could reach the timeline. Returns the deleted names."""
    if not clips_dir.is_dir():
        return []
    removed = []
    for old in sorted(clips_dir.glob(f"{fragment:02d}_*.mp4")):
        old.unlink()
        removed.append(old.name)
    return removed


def run(project: str) -> Path:
    """Cut every fragment's range into clips/ and write _cut_list.md; returns its path."""
    project_dir = config.project_path(project)
    clips_dir = project_dir / "clips"
    footage_dir = project_dir / "footage"
    timings_path = project_dir / "timings.json"

    if not timings_path.exists():
        raise SystemExit(f"No timings.json in {project_dir} — "
                         f"run build_storyboard first.")

    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    games = sorted({t.get("video_game") for t in timings if t.get("video_game")})
    catalog: dict[str, dict] = {}
    for g in games:
        catalog.update(load_catalog_index(g))
    log.info("%d fragments, games=%s, catalog has %d entries",
             len(timings), games, len(catalog))

    clips_dir.mkdir(parents=True, exist_ok=True)
    cut_lines = ["# Cut List", ""]
    total_clips = 0
    total_sec = 0.0

    for t in timings:
        num = t["fragment"]
        duration = t["duration_sec"]
        for old in clear_fragment_clips(clips_dir, num):
            log.info("Frag %2d: removed old clip %s", num, old)
        video_name = t.get("primary_video")
        preview = t.get("text_preview", "")

        cut_lines.append(f"## Fragment {num} ({t['start_ts']} - {t['end_ts']}, "
                         f"{duration:.0f}s)")
        cut_lines.append(f"> {preview}")
        cut_lines.append("")

        if not video_name:
            cut_lines.append("NO LOCAL VIDEO")
            cut_lines.append("")
            log.info("Frag %2d: no video", num)
            continue

        source = resolve_source(video_name, footage_dir, catalog)
        if source is None:
            cut_lines.append(f"MISSING: {video_name}")
            cut_lines.append("")
            log.warning("Frag %2d: MISSING %s", num, video_name)
            continue

        if "clip_start" not in t or "clip_end" not in t:
            cut_lines.append(f"NO RANGE: {video_name} — no curated range and no "
                             f"semantic moment; add one to curation.json")
            cut_lines.append("")
            log.warning("Frag %2d: NO RANGE for %s", num, video_name)
            continue

        start, end = float(t["clip_start"]), float(t["clip_end"])
        clip_dur = end - start
        if clip_dur <= 0:
            cut_lines.append(f"BAD RANGE: {video_name} {start:.3f}-{end:.3f}s — the end is "
                             f"not after the start; fix curation.json")
            cut_lines.append("")
            log.warning("Frag %2d: BAD RANGE %.3f-%.3f", num, start, end)
            continue
        clip_name = f"{num:02d}_{Path(video_name).stem}.mp4"
        clip_path = clips_dir / clip_name

        if extract_clip(source, start, end, clip_path):
            total_clips += 1
            total_sec += clip_dur
            entry = catalog.get(video_name, {})
            log.info("Frag %2d: %s (%s-%s, %.1fs) [narr %.0fs, %s]", num, clip_name,
                     _sec_to_ts(start), _sec_to_ts(end), clip_dur, duration,
                     t.get("clip_source", "?"))
            cut_lines.append(f"**Clip:** `{clip_name}`")
            cut_lines.append(f"**Source:** {video_name} "
                             f"(chosen by {t.get('clip_source', '?')})")
            cut_lines.append(f"**Range:** {_sec_to_ts(start)} - {_sec_to_ts(end)} "
                             f"({clip_dur:.1f}s) — cut exactly at "
                             f"{start:.3f}-{end:.3f}s")
            cut_lines.append(f"**Content:** {entry.get('title', '')}")
            cut_lines.append("")
        else:
            cut_lines.append("FAILED to extract")
            cut_lines.append("")
            log.error("Frag %2d: FAILED ffmpeg", num)

    cut_path = clips_dir / "_cut_list.md"
    cut_path.write_text("\n".join(cut_lines), encoding="utf-8")

    total = int(total_sec)
    log.info("Done: %d clips, %ds total (%dm%02ds)", total_clips, total, total // 60, total % 60)
    return cut_path


def main(argv: list[str] | None = None) -> None:
    """CLI: cut a project's clips."""
    ap = argparse.ArgumentParser(
        description="Cut the ranges in a project's timings.json into clips/ (+ _cut_list.md)")
    ap.add_argument("project", help="project folder under projects/")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    print(f"Cut list: {run(args.project)}")


if __name__ == "__main__":
    main()

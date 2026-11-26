"""Contact sheet: verify clip choices BEFORE they reach Vegas.

One HTML page per project: for every fragment — the narration text, the chosen
clip (file, range, why it was chosen) and a few frames pulled from exactly the
range that will be cut. Five minutes of scrolling catches what used to be
discovered fragment-by-fragment on the Vegas timeline, at the user's expense.

Frames come from the pre-cut clip when clips/ has one (that is literally what
will be placed), otherwise from the source at the timings range.

Usage:
    python -m clipmaker.contact_sheet --project <name> [--frames 4]
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

from . import config
from .clip_extractor import resolve_source
from .layout import find_precut_clip
from .log import add_verbose_flag, setup_logging


DEFAULT_FRAMES = 4
FRAME_WIDTH = 480


# ---------------------------------------------------------------------------
# frame planning (pure)
# ---------------------------------------------------------------------------

def frame_times(start: float, end: float, count: int) -> list[float]:
    """Evenly spaced sample points, biased off the exact edges — the first and
    last frames of a cut are fades/transitions, not representative content.
    One frame is the middle of the range, not its first frame."""
    if count < 1:
        raise ValueError(f"frame count must be at least 1, got {count}")
    span = max(0.0, end - start)
    if span == 0:
        return [start]
    step = span / (count + 1)
    return [round(start + step * (i + 1), 2) for i in range(count)]


def sheet_row(t: dict, clips_dir: Path, footage_dir: Path,
              catalog: dict[str, dict], frames: int) -> dict:
    """Decide what to sample for one fragment. Returns a render/extract plan."""
    num = t["fragment"]
    row = {
        "fragment": num,
        "start_ts": t.get("start_ts", "?"),
        "end_ts": t.get("end_ts", "?"),
        "duration": t.get("duration_sec", 0),
        "text": t.get("text_preview", ""),
        "transcript": t.get("transcript_preview", ""),
        "source_label": t.get("clip_source") or "—",
        "video": t.get("primary_video"),
        "title": t.get("video_title", ""),
        "problem": None,
        "media": None,
        "times": [],
        "range_label": "",
    }
    if not t.get("primary_video"):
        row["problem"] = "no video assigned"
        return row

    precut = find_precut_clip(clips_dir, num, t["primary_video"])
    if precut is not None:
        clip_len = max(0.0, float(t.get("clip_end", 0)) - float(t.get("clip_start", 0)))
        span = clip_len or float(t.get("duration_sec", 0)) or 30.0
        row["media"] = precut
        row["times"] = frame_times(0.0, span, frames)
        row["range_label"] = f"pre-cut clip {precut.name}"
        return row

    if "clip_start" not in t or "clip_end" not in t:
        row["problem"] = "NO RANGE — no curated range and no semantic moment"
        return row

    source = resolve_source(t["primary_video"], footage_dir, catalog)
    if source is None:
        row["problem"] = f"media not found: {t['primary_video']}"
        return row

    start, end = float(t["clip_start"]), float(t["clip_end"])
    row["media"] = source
    row["times"] = frame_times(start, end, frames)
    row["range_label"] = f"source {start:.1f}–{end:.1f}s"
    return row


# ---------------------------------------------------------------------------
# frame extraction
# ---------------------------------------------------------------------------

def extract_frame(media: Path, at_sec: float, out: Path) -> bool:
    """Write one scaled frame of media at at_sec to out; False if ffmpeg fails."""
    cmd = ["ffmpeg", "-y", "-ss", f"{at_sec:.3f}", "-i", str(media),
           "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-1", "-q:v", "3",
           str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and out.exists()


# ---------------------------------------------------------------------------
# HTML rendering (pure)
# ---------------------------------------------------------------------------

def render_html(project: str, rows: list[dict]) -> str:
    """The contact sheet page: per fragment, its narration and the frames to be placed."""
    problems = sum(1 for r in rows if r["problem"])
    out = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Contact sheet: {html.escape(project)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f5f5f2;color:#222}",
        ".frag{background:#fff;border:1px solid #ddd;border-radius:8px;"
        "padding:16px;margin-bottom:16px}",
        ".frag.bad{border-color:#c33;background:#fff6f6}",
        ".head{font-weight:600;margin-bottom:4px}",
        ".meta{color:#666;font-size:13px;margin-bottom:8px}",
        ".text{margin:8px 0;font-size:14px}",
        ".transcript{color:#555;font-size:13px;font-style:italic;margin-bottom:8px}",
        ".frames{display:flex;gap:8px;flex-wrap:wrap}",
        ".frames figure{margin:0;font-size:11px;color:#666;text-align:center}",
        ".frames img{max-width:240px;border-radius:4px;display:block}",
        ".problem{color:#c33;font-weight:600}",
        "</style>",
        f"<h1>Contact sheet: {html.escape(project)}</h1>",
        f"<p>{len(rows)} fragments, {problems} with problems. "
        "Frames are sampled from exactly the range that will be placed — "
        "if a picture does not match its text, fix that fragment in "
        "curation.json and rebuild.</p>",
    ]
    for r in rows:
        cls = "frag bad" if r["problem"] else "frag"
        out.append(f"<div class='{cls}'>")
        out.append(f"<div class='head'>Fragment {r['fragment']:02d} "
                   f"&middot; {r['start_ts']}–{r['end_ts']} "
                   f"({r['duration']:.0f}s)</div>")
        meta = [f"chosen by: {r['source_label']}"]
        if r["video"]:
            meta.append(html.escape(str(r["video"])))
        if r["range_label"]:
            meta.append(html.escape(r["range_label"]))
        if r["title"]:
            meta.append(html.escape(r["title"]))
        out.append(f"<div class='meta'>{' &middot; '.join(meta)}</div>")
        out.append(f"<div class='text'>{html.escape(r['text'])}</div>")
        if r["transcript"]:
            out.append(f"<div class='transcript'>spoken: "
                       f"{html.escape(r['transcript'])}</div>")
        if r["problem"]:
            out.append(f"<div class='problem'>{html.escape(r['problem'])}</div>")
        elif r.get("images"):
            out.append("<div class='frames'>")
            for img, at in r["images"]:
                out.append(f"<figure><img src='{img}' alt=''>"
                           f"<figcaption>{at:.1f}s</figcaption></figure>")
            out.append("</div>")
        out.append("</div>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI: build contact_sheet.html for a project."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    project_dir = config.project_path(args.project)
    timings_path = project_dir / "timings.json"
    if not timings_path.exists():
        sys.exit(f"No timings.json in {project_dir} — run build_storyboard first.")

    timings = json.loads(timings_path.read_text(encoding="utf-8"))

    from .build_storyboard import load_catalog_index
    catalog = load_catalog_index()

    sheet_dir = project_dir / "contact_sheet"
    sheet_dir.mkdir(exist_ok=True)

    rows = []
    for t in timings:
        row = sheet_row(t, project_dir / "clips", project_dir / "footage",
                        catalog, args.frames)
        if row["media"] is not None:
            images = []
            for k, at in enumerate(row["times"], 1):
                out = sheet_dir / f"fr{row['fragment']:02d}_{k}.jpg"
                if extract_frame(row["media"], at, out):
                    images.append((out.name, at))
            if images:
                row["images"] = images
            else:
                row["problem"] = "frame extraction failed"
        rows.append(row)
        state = row["problem"] or f"{len(row.get('images', []))} frames"
        print(f"  fr {row['fragment']:02d}: {state}")

    html_path = sheet_dir / "contact_sheet.html"
    html_path.write_text(render_html(args.project, rows), encoding="utf-8")
    problems = sum(1 for r in rows if r["problem"])
    print(f"\n{len(rows)} fragments, {problems} problem(s)")
    print(f"Open: {html_path}")


if __name__ == "__main__":
    main()

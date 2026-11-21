"""Extract sections from structured transcript into Vegas-compatible JSON.

Parses *_structured.md to extract section headings with timestamps,
then writes a sections.json for the Vegas import script.

Usage:
    python -m clipmaker.vegas_export catalog/.../Video_structured.md
    python -m clipmaker.vegas_export catalog/.../Video_structured.md --video "path/to/video.mp4"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_timestamp(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(parts[0])


def extract_sections(md_text: str) -> list[dict]:
    heading_re = re.compile(
        r"^(#{2,3})\s+(.+?)(?:\s+\[(\d{2}:\d{2}:\d{2})\])?\s*$",
        re.MULTILINE,
    )
    ts_range_re = re.compile(
        r"^\*\[(\d{2}:\d{2}:\d{2})\s*[—–-]\s*(\d{2}:\d{2}:\d{2})\]\*"
    )
    ts_inline_re = re.compile(
        r"^\[(\d{2}:\d{2}:\d{2})\]"
    )

    sections: list[dict] = []
    lines = md_text.split("\n")
    i = 0

    while i < len(lines):
        m = heading_re.match(lines[i])
        if not m:
            i += 1
            continue

        level = len(m.group(1))
        title = m.group(2).strip()
        start_sec = parse_timestamp(m.group(3)) if m.group(3) else None

        if title.lower() in ("summary", "table of contents"):
            i += 1
            continue

        end_sec = None
        for j in range(i + 1, min(i + 6, len(lines))):
            line = lines[j].strip()
            if not line:
                continue
            range_match = ts_range_re.match(line)
            if range_match:
                start_sec = start_sec or parse_timestamp(range_match.group(1))
                end_sec = parse_timestamp(range_match.group(2))
                break
            inline_match = ts_inline_re.match(line)
            if inline_match:
                start_sec = start_sec or parse_timestamp(inline_match.group(1))
                break

        if start_sec is not None:
            sections.append({
                "title": title,
                "level": level,
                "start_sec": start_sec,
                "end_sec": end_sec,
            })

        i += 1

    for idx in range(len(sections) - 1):
        if sections[idx]["end_sec"] is None:
            sections[idx]["end_sec"] = sections[idx + 1]["start_sec"]

    if sections and sections[-1]["end_sec"] is None:
        sections[-1]["end_sec"] = sections[-1]["start_sec"] + 60

    return sections


def find_catalog_json(structured_path: Path) -> dict | None:
    stem = structured_path.stem.removesuffix("_structured")
    catalog_json = structured_path.parent / f"{stem}.json"
    if catalog_json.exists():
        data = json.loads(catalog_json.read_text(encoding="utf-8"))
        return data
    return None


def run(structured_path: Path, video_override: str | None = None) -> Path:
    if not structured_path.exists():
        print(f"File not found: {structured_path}")
        sys.exit(1)

    md_text = structured_path.read_text(encoding="utf-8")
    sections = extract_sections(md_text)

    if not sections:
        print("No sections with timestamps found.")
        sys.exit(1)

    video_path = video_override
    if not video_path:
        catalog = find_catalog_json(structured_path)
        if catalog and "source_path" in catalog:
            video_path = catalog["source_path"]

    stem = structured_path.stem.removesuffix("_structured")
    output = {
        "source_video": video_path or "",
        "source_structured": structured_path.name,
        "sections": sections,
    }

    out_path = structured_path.parent / f"{stem}_vegas_sections.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Extracted {len(sections)} sections -> {out_path}")

    for s in sections:
        end_label = f" — {s['end_sec']:.0f}s" if s["end_sec"] else ""
        level_prefix = "  " if s["level"] == 3 else ""
        print(f"  {level_prefix}[{s['start_sec']:6.0f}s{end_label}] {s['title']}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract sections from structured transcript for Vegas import",
    )
    ap.add_argument("input", type=Path, help="Path to *_structured.md file")
    ap.add_argument("--video", type=str, default=None,
                    help="Override source video path (auto-detected from catalog JSON)")
    args = ap.parse_args()
    run(args.input, video_override=args.video)


if __name__ == "__main__":
    main()

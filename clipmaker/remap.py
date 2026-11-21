"""Carry B-roll event starts from an old narration to a re-recorded one.

When the narration is re-voiced, every event in vegas_events.json was timed to
the old mp3. Word-level transcripts of both recordings give anchors: runs of
identical words (at least --min-run long) pair an old time with a new one. Each
event start moves by its nearest preceding anchor's shift; each event then
lasts until the next event's new start (the last one until the new end).

Usage:
    python -m clipmaker.remap --project <p> --old <old_raw_transcript.json>
        --new <new_raw_transcript.json> [--events vegas_events.json] [--out remap.json]

Paths are relative to the project folder. Flags in the printed table:
    LONGER+x  the event's slot grew by more than 0.5 s — is the clip long enough?
    TOO-SHORT the slot shrank under 1.5 s
    FARx      the nearest anchor is more than 3 s away — the passage was rewritten
    NO-ANCHOR the event sits before the first anchor and was not moved
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import json
import math
import re
import sys
from pathlib import Path

from . import config

MIN_RUN = 3
LONGER_SLACK_SEC = 0.5
TOO_SHORT_SEC = 1.5
FAR_ANCHOR_SEC = 3.0
_NON_WORD = re.compile(r"[^\w]")


def normalize_word(word: str) -> str:
    return _NON_WORD.sub("", word.lower().replace("ё", "е"))


def transcript_words(data: dict | list) -> list[tuple[str, float]]:
    """(normalized word, start sec) over a faster-whisper transcript JSON."""
    segments = data["segments"] if isinstance(data, dict) else data
    out = []
    for seg in segments:
        for w in seg.get("words") or []:
            token = normalize_word(w["word"])
            if token:
                out.append((token, float(w["start"])))
    return out


def transcript_end(data: dict | list) -> float:
    segments = data["segments"] if isinstance(data, dict) else data
    return max((float(s["end"]) for s in segments), default=0.0)


def anchor_pairs(old: list[tuple[str, float]], new: list[tuple[str, float]],
                 min_run: int = MIN_RUN) -> list[tuple[float, float]]:
    """(old time, new time) for every word inside an identical run >= min_run."""
    sm = difflib.SequenceMatcher(a=[w for w, _ in old], b=[w for w, _ in new], autojunk=False)
    pairs = []
    for tag, i1, i2, j1, _ in sm.get_opcodes():
        if tag == "equal" and i2 - i1 >= min_run:
            pairs += [(old[i1 + k][1], new[j1 + k][1]) for k in range(i2 - i1)]
    return sorted(pairs)


def map_time(t: float, pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """(new time, distance to the nearest anchor). Before the first anchor the
    time is kept and the distance is infinite."""
    olds = [p[0] for p in pairs]
    i = bisect.bisect_right(olds, t) - 1
    if i < 0:
        return t, math.inf
    j = min(i + 1, len(olds) - 1)
    near = min(abs(t - olds[i]), abs(olds[j] - t))
    return pairs[i][1] + (t - olds[i]), near


def remap_events(events: list[dict], pairs: list[tuple[float, float]],
                 new_end: float) -> list[dict]:
    """Events (sorted by start) with new_start_sec / new_len_sec / old_len_sec /
    anchor_dist_sec added; input dicts are not modified."""
    ordered = sorted(events, key=lambda e: float(e["start_sec"]))
    starts = []
    for e in ordered:
        new_start, near = map_time(float(e["start_sec"]), pairs)
        starts.append((round(new_start, 2), near))
    rows = []
    for k, (e, (new_start, near)) in enumerate(zip(ordered, starts)):
        nxt = starts[k + 1][0] if k + 1 < len(starts) else new_end
        rows.append({**e,
                     "new_start_sec": new_start,
                     "new_len_sec": round(nxt - new_start, 2),
                     "old_len_sec": round(float(e["end_sec"]) - float(e["start_sec"]), 2),
                     "anchor_dist_sec": None if math.isinf(near) else round(near, 1)})
    return rows


def row_flags(row: dict) -> str:
    flags = ""
    if row["new_len_sec"] > row["old_len_sec"] + LONGER_SLACK_SEC:
        flags += f" LONGER+{row['new_len_sec'] - row['old_len_sec']:.1f}"
    if row["new_len_sec"] < TOO_SHORT_SEC:
        flags += " TOO-SHORT"
    if row["anchor_dist_sec"] is None:
        flags += " NO-ANCHOR"
    elif row["anchor_dist_sec"] > FAR_ANCHOR_SEC:
        flags += f" FAR{row['anchor_dist_sec']}"
    return flags


def _load_json(path: Path):
    if not path.exists():
        sys.exit(f"No {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clipmaker.remap", description=__doc__.split("\n")[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--old", required=True, help="word-level transcript of the old narration")
    ap.add_argument("--new", required=True, help="word-level transcript of the new narration")
    ap.add_argument("--events", default="vegas_events.json")
    ap.add_argument("--out", default="remap.json")
    ap.add_argument("--min-run", type=int, default=MIN_RUN,
                    help="shortest identical word run used as an anchor (default %(default)s)")
    ap.add_argument("--new-end", type=float, default=None,
                    help="end of the new narration, s (default: transcript end)")
    args = ap.parse_args(argv)

    project_dir = config.PROJECTS_DIR / args.project
    if not project_dir.is_dir():
        sys.exit(f"No project folder {project_dir}")
    old = _load_json(project_dir / args.old)
    new = _load_json(project_dir / args.new)
    events = _load_json(project_dir / args.events)

    pairs = anchor_pairs(transcript_words(old), transcript_words(new), args.min_run)
    if not pairs:
        sys.exit("No anchors: the transcripts share no word run "
                 f"of {args.min_run}+ words. Wrong files?")
    new_end = args.new_end if args.new_end is not None else transcript_end(new)
    rows = remap_events(events, pairs, new_end)

    out_path = project_dir / args.out
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for k, r in enumerate(rows, 1):
        print(f"{k:3d} old {float(r['start_sec']):7.1f} -> new {r['new_start_sec']:7.1f} "
              f"len {r['old_len_sec']:5.1f}->{r['new_len_sec']:5.1f}{row_flags(r)} "
              f"| {str(r.get('label', ''))[:60]}")
    print(f"\n{len(pairs)} anchor words, {len(rows)} events -> {out_path}")


if __name__ == "__main__":
    main()

"""Align storyboard text fragments to audio timestamps via transcription.

Reads the storyboard fragments and the Whisper transcription JSON,
finds where each fragment starts/ends in the audio, and writes
a timecoded fragment map for clip_extractor and the Vegas script.

Usage:
    python -m clipmaker.align <project>            # e.g. my_game/drafts/<slug>
    python -m clipmaker.align --project <project>  # same thing
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

from . import config
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

# Below this share of matched words a fragment's anchor is flagged: the start
# may be right, but nothing in the transcript proves it.
WEAK_MATCH_RATIO = 0.5


def _fragment_quote(block: str) -> str:
    """The fragment's full narration quote — every `>` line of the leading
    blockquote, not just the first, plus Markdown "lazy" continuation lines
    (a wrapped line right under a `>` line, before any blank line; storyboards
    written before 2026-11 kept the narration's own line breaks). Stops at the
    first content line after a blank line, so the `> **Unconfirmed**` note
    inside "My videos" is never swallowed.
    """
    lines: list[str] = []
    in_quote = False
    for line in block.splitlines():
        s = line.strip()
        if s.startswith(">"):
            lines.append(s.lstrip("> ").strip())
            in_quote = True
            continue
        if not s:
            if lines:
                in_quote = False
            continue
        if in_quote:
            lines.append(s)  # lazy continuation of the quote
            continue
        break  # first content line after (or before) the quote
    return " ".join(ln for ln in lines if ln)


def parse_storyboard(path: Path) -> list[dict]:
    """Fragments of a storyboard.md: number, narration quote and listed videos."""
    text = path.read_text(encoding="utf-8")
    fragments = []
    for block in re.split(r"^## Fragment (\d+)", text, flags=re.MULTILINE):
        block = block.strip()
        if not block:
            continue
        if block.isdigit():
            fragments.append({"num": int(block), "text": "", "videos": []})
            continue
        if not fragments:
            continue
        frag = fragments[-1]
        frag["text"] = _fragment_quote(block)
        frag["videos"] = re.findall(r"- File: `(.+?)`", block)
    return fragments


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def load_transcript(path: Path) -> list[dict]:
    """Segments of a transcript.json: a list, or an object with "segments".

    Each segment needs numeric start/end and a text; "words" (Whisper word
    timestamps) is optional but must carry word and start when present. A
    ValueError names the segment, where a KeyError deep inside the alignment
    said nothing about which file was wrong.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    segments = raw.get("segments") if isinstance(raw, dict) else raw
    if not isinstance(segments, list):
        raise ValueError(f"{path}: expected a list of segments or an object with "
                         f"\"segments\", got {type(raw).__name__}")
    for i, seg in enumerate(segments):
        _check_segment(seg, f"{path.name} segment {i}")
    return segments


def _check_segment(seg: object, where: str) -> None:
    """ValueError unless seg is a transcript segment the aligner can read."""
    if not isinstance(seg, dict):
        raise ValueError(f"{where}: expected an object, got {seg!r}")
    for key in ("start", "end"):
        if isinstance(seg.get(key), bool) or not isinstance(seg.get(key), (int, float)):
            raise ValueError(f"{where}: {key!r} must be a number of seconds, got {seg.get(key)!r}")
    if seg["end"] < seg["start"]:
        raise ValueError(f"{where}: ends ({seg['end']}) before it starts ({seg['start']})")
    if not isinstance(seg.get("text"), str):
        raise ValueError(f"{where}: missing \"text\"")
    for w in seg.get("words") or []:
        if not (isinstance(w, dict) and isinstance(w.get("word"), str)
                and isinstance(w.get("start"), (int, float))):
            raise ValueError(f"{where}: bad word entry {w!r} (needs word and start)")


def transcript_words(segments: list[dict]) -> tuple[list[str], list[float]]:
    """Flatten the transcript into parallel lists of words and their start times.

    Uses Whisper's word-level timestamps; falls back to spreading a segment's
    words evenly across its span when they are missing.
    """
    words: list[str] = []
    times: list[float] = []

    for seg in segments:
        stamped = seg.get("words") or []
        if stamped:
            for w in stamped:
                token = _normalize(w["word"])
                if token:
                    words.append(token)
                    times.append(float(w["start"]))
            continue

        tokens = _normalize(seg["text"]).split()
        if not tokens:
            continue
        span = (seg["end"] - seg["start"]) / len(tokens)
        for i, token in enumerate(tokens):
            words.append(token)
            times.append(seg["start"] + i * span)

    return words, times


def anchor_starts(frag_texts: list[str],
                  segments: list[dict]) -> tuple[list[float], list[dict]]:
    """Find each fragment's start time by aligning script words to spoken words.

    The narrator reads the script start to finish but paraphrases it — words are
    dropped, numbers are said differently, sentences are rearranged. So instead
    of matching each fragment independently (which mis-fires on a weak local
    match and then drags every later fragment along with it), align the *whole*
    script against the *whole* transcript as two word sequences. The matching
    blocks are common-subsequence runs, so they are ordered by construction:
    each fragment's first word maps to a spoken word at or after the previous
    fragment's, and paraphrased stretches are simply skipped over.

    Returns the starts and a per-fragment confidence record: what share of the
    fragment's words the transcript actually matched (``match_ratio``) and how
    the anchor was found (``anchor``: exact — the opening word matched;
    paraphrased — snapped forward to the next matched block; unmatched — no
    block at or after the fragment, clamped to the end). A low ratio means the
    start (and the previous fragment's end) is a guess to be checked, not a
    fact — a silent bad anchor corrupts two fragments.
    """
    spoken_words, spoken_times = transcript_words(segments)
    if not spoken_words:
        empty = [{"match_ratio": 0.0, "anchor": "unmatched"} for _ in frag_texts]
        return [0.0] * len(frag_texts), empty

    script_words: list[str] = []
    frag_offsets: list[int] = []  # index in script_words where each fragment opens
    for text in frag_texts:
        frag_offsets.append(len(script_words))
        script_words.extend(_normalize(text).split())
    frag_ends = frag_offsets[1:] + [len(script_words)]

    matcher = SequenceMatcher(None, script_words, spoken_words, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]

    starts: list[float] = []
    stats: list[dict] = []
    for offset, frag_end in zip(frag_offsets, frag_ends, strict=True):
        spoken_idx = len(spoken_times) - 1
        anchor = "unmatched"
        for block in blocks:
            if block.a <= offset < block.a + block.size:  # word matched directly
                spoken_idx = block.b + (offset - block.a)
                anchor = "exact"
                break
            if block.a > offset:  # opening was paraphrased — take the next match
                spoken_idx = block.b
                anchor = "paraphrased"
                break
        starts.append(spoken_times[min(spoken_idx, len(spoken_times) - 1)])

        matched = sum(min(b.a + b.size, frag_end) - max(b.a, offset)
                      for b in blocks
                      if b.a < frag_end and b.a + b.size > offset)
        span = max(1, frag_end - offset)
        stats.append({"match_ratio": round(matched / span, 2), "anchor": anchor})

    for i in range(1, len(starts)):  # guard against any residual backtracking
        starts[i] = max(starts[i], starts[i - 1])

    return starts, stats


def _preview(text: str, width: int = 60) -> str:
    """The start of a text, with "..." only when something was cut."""
    return text if len(text) <= width else text[:width] + "..."


def fmt_ts(sec: float) -> str:
    """Seconds as HH:MM:SS."""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def align(project: str) -> Path:
    """Anchor each storyboard fragment in the narration transcript and write
    timings_v1.json; returns its path. FileNotFoundError if an input is missing."""
    # Never config.get_project_dir here: it creates the folder, and a mistyped
    # name used to leave stray directories like projects/--project behind.
    project_dir = config.project_path(project)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"No project folder {project_dir}")
    storyboard_path = project_dir / "storyboard.md"
    transcript_path = project_dir / "transcript.json"
    out_path = project_dir / "timings_v1.json"

    if not storyboard_path.exists():
        raise FileNotFoundError(f"No storyboard.md in {project_dir}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"No transcript.json in {project_dir}")

    fragments = parse_storyboard(storyboard_path)
    segments = load_transcript(transcript_path)
    log.info("Fragments: %d, Transcript segments: %d", len(fragments), len(segments))

    spoken = [f for f in fragments if f["text"]]

    # Pass 1: anchor every fragment start in one global word-sequence alignment.
    starts, stats = anchor_starts([f["text"] for f in spoken], segments)

    transcript_end = segments[-1]["end"] if segments else 0.0

    # Pass 2: each fragment ends where the next one starts; last ends at EOF.
    timings = []
    weak = []
    for i, frag in enumerate(spoken):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(spoken) else transcript_end
        if end <= start:  # degenerate anchor — fall back to a wpm estimate
            end = start + len(frag["text"].split()) / config.WORDS_PER_MIN_RU * 60
        dur = end - start
        stat = stats[i]
        timings.append({
            "fragment": frag["num"],
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "duration_sec": round(dur, 2),
            "start_ts": fmt_ts(start),
            "end_ts": fmt_ts(end),
            "match_ratio": stat["match_ratio"],
            "anchor": stat["anchor"],
            "videos": frag["videos"],
            "text_preview": _preview(frag["text"]),
        })
        flag = ""
        if stat["match_ratio"] < WEAK_MATCH_RATIO or stat["anchor"] == "unmatched":
            weak.append(frag["num"])
            flag = f"  <-- WEAK ({stat['anchor']}, {stat['match_ratio']:.0%} matched)"
        log.info("Frag %2d: %s - %s (%.0fs) %s%s", frag["num"], fmt_ts(start),
                 fmt_ts(end), dur, frag["text"][:50], flag)

    out_path.write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %d fragment timings to %s", len(timings), out_path)
    if weak:
        log.warning("WEAK ANCHORS on fragments %s: the transcript matched under "
                    "%.0f%% of their words. Their starts (and the previous fragments' "
                    "ends) are guesses - check these rows on the contact sheet, or the "
                    "mp3 may be an older draft of the text (a known failure: identical "
                    "timings in a run mean unvoiced text).", weak, WEAK_MATCH_RATIO * 100)
    return out_path


def parse_args(argv: list[str] | None = None) -> str:
    """Project from either the positional argument or --project (the form the
    other CLIs use). Both given must agree; a missing project is an error, not
    a silent default."""
    return _parse(argv)[0]


def _parse(argv: list[str] | None) -> tuple[str, bool]:
    """(project, verbose) from the command line."""
    ap = argparse.ArgumentParser(
        prog="clipmaker.align",
        description="Align storyboard fragments to the narration transcript "
                    "-> timings_v1.json")
    ap.add_argument("project_pos", nargs="?", metavar="project",
                    help="project folder under projects/, e.g. my_game/drafts/<slug>")
    ap.add_argument("--project", default=None, help="same as the positional argument")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    if args.project and args.project_pos and args.project != args.project_pos:
        ap.error(f"two different projects given: {args.project_pos!r} and {args.project!r}")
    project = args.project or args.project_pos
    if not project:
        ap.error("no project given")
    return project, args.verbose


def main(argv: list[str] | None = None) -> None:
    """CLI: align a project's storyboard to its transcript."""
    project, verbose = _parse(argv)
    setup_logging(verbose)
    try:
        print(f"Timings: {align(project)}")
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

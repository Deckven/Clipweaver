"""Transcribe audio with word-level timestamps using faster-whisper.

Usage:
    python -m clipmaker.transcribe --project diablo4-lord-of-hatred
    python -m clipmaker.transcribe --file "path/to/video.mp4" --lang en --out transcript.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)


def transcribe_file(audio_path: Path, out_path: Path, language: str = "ru",
                    model_size: str = "medium") -> Path:
    """Whisper segments with word timestamps -> out_path (JSON list); returns out_path."""
    from faster_whisper import WhisperModel

    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    log.info("Transcribing: %s  (lang=%s)", audio_path.name, language)
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    result = []
    for seg in segments:
        entry = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        }
        if seg.words:
            entry["words"] = [
                {"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in seg.words
            ]
        result.append(entry)
        preview = seg.text.strip()[:80]
        log.info("[%7.1fs - %7.1fs] %s", seg.start, seg.end, preview)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %d segments to %s", len(result), out_path)
    return out_path


def transcribe(project: str) -> Path:
    """Transcribe a project's narration (<project>.mp3, else narration.mp3) to transcript.json."""
    project_dir = config.project_path(project)
    audio_path = project_dir / f"{project}.mp3"
    if not audio_path.exists():
        audio_path = project_dir / "narration.mp3"
    out_path = project_dir / "transcript.json"
    return transcribe_file(audio_path, out_path, language="ru")


def main(argv: list[str] | None = None) -> None:
    """CLI: transcribe a file or a project's narration."""
    ap = argparse.ArgumentParser(description="Transcribe audio/video with word-level timestamps")
    ap.add_argument("--project", type=str, default=None,
                    help="Project name (uses project audio + Russian)")
    ap.add_argument("--file", type=Path, default=None,
                    help="Direct path to audio/video file")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default: next to input file)")
    ap.add_argument("--lang", type=str, default="ru",
                    help="Language code (default: ru)")
    ap.add_argument("--model", type=str, default="medium",
                    help="Whisper model size (default: medium)")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    if not args.file and not args.project:
        ap.error("give --file or --project")
    try:
        if args.file:
            out = args.out or args.file.with_suffix(".transcript.json")
            print(transcribe_file(args.file, out, language=args.lang, model_size=args.model))
        else:
            print(transcribe(args.project))
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

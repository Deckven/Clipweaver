"""Transcribe audio with word-level timestamps using faster-whisper.

Usage:
    python -m clipmaker.transcribe --project diablo4-lord-of-hatred
    python -m clipmaker.transcribe --file "path/to/video.mp4" --lang en --out transcript.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def transcribe_file(audio_path: Path, out_path: Path, language: str = "ru",
                    model_size: str = "medium"):
    from faster_whisper import WhisperModel

    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print(f"Transcribing: {audio_path.name}  (lang={language})")
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
        print(f"  [{seg.start:7.1f}s - {seg.end:7.1f}s] {preview}")

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(result)} segments to {out_path}")


def transcribe(project: str):
    project_dir = config.get_project_dir(project)
    audio_path = project_dir / f"{project}.mp3"
    if not audio_path.exists():
        audio_path = project_dir / "narration.mp3"
    out_path = project_dir / "transcript.json"
    transcribe_file(audio_path, out_path, language="ru")


def main():
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
    args = ap.parse_args()

    if args.file:
        out = args.out or args.file.with_suffix(".transcript.json")
        transcribe_file(args.file, out, language=args.lang, model_size=args.model)
    elif args.project:
        transcribe(args.project)
    else:
        transcribe("diablo4-lord-of-hatred")


if __name__ == "__main__":
    main()

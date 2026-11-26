"""Video transcriber: raw verbatim transcript + structured text via local LLM.

Two-stage pipeline:
  1. Raw transcription (faster-whisper) with word-level timestamps
  2. Structured text generation (Ollama local LLM)

Usage:
    python -m clipmaker.video_transcriber video.mp4
    python -m clipmaker.video_transcriber video.mp4 --lang en --ollama-model qwen2.5:14b
    python -m clipmaker.video_transcriber video.mp4 --stage raw
    python -m clipmaker.video_transcriber raw_transcript.json --stage structure
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)


class TranscriberError(RuntimeError):
    """A stage failed for a reason the user can fix (ffmpeg, Ollama, input file)."""

AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts"})

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "mistral-small"
DEFAULT_WHISPER_MODEL = "medium"
# Char budget per LLM request. Kept modest because local models (e.g. mistral-small)
# degrade — summarising instead of structuring — when handed a very large input at once.
MAX_CHUNK_CHARS = 6_000
PAUSE_THRESHOLD_SEC = 3.0

STRUCTURE_SYSTEM_PROMPT = """\
You are a professional transcription editor. Transform the raw verbatim \
transcript into a well-structured document.

Rules:
1. Infer a title from the content.
2. Write a 2-3 sentence summary.
3. Identify main topics and create section headings with timestamp ranges.
4. Clean speech artifacts: filler words (um, uh, like, you know, so, \
значит, ну, типа, вот, э-э, короче), false starts, repetitions.
5. Fix incomplete sentences while preserving the speaker's meaning.
6. Preserve important or notable quotes verbatim using > blockquote.
7. Add key takeaways as bullet lists where appropriate.
8. Output in the SAME LANGUAGE as the transcript.

Output format:
# [Title]

## Summary
[2-3 sentences]

## Table of Contents
1. [Section] [HH:MM:SS]

---

## [Section Name]
*[HH:MM:SS — HH:MM:SS]*

[Clean structured text]

**Key points:**
- ...\
"""

STRUCTURE_CONTINUATION_PROMPT = """\
Continue structuring the transcript. Create section headings with timestamps, \
clean filler words, fix incomplete sentences. Same language as input. \
No title/summary/TOC — just sections.\
"""


def format_timestamp(seconds: float) -> str:
    """Seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """16 kHz mono WAV of a video's audio track; TranscriberError with ffmpeg's stderr."""
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", str(audio_path),
    ]
    log.info("Extracting audio from %s...", video_path.name)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise TranscriberError(f"ffmpeg could not extract audio from {video_path.name} "
                               f"(exit {result.returncode}):\n{stderr[-1500:]}")


# Keeps loaded WhisperModel objects referenced for the lifetime of the process.
# Deallocating a CUDA WhisperModel triggers a native CTranslate2/CUDA context
# teardown that segfaults on some GPUs (observed on RTX 5070 Ti / Blackwell with
# long audio), killing the process with exit code 9 *after* transcription but
# *before* outputs are written. Holding the reference here prevents the
# garbage-collector from running that teardown when transcribe_raw() returns;
# main() then terminates via os._exit() to skip the destructor entirely.
_MODEL_KEEPALIVE: list = []


def _collect_segments(segments_iter) -> list[dict]:
    """Drain a faster-whisper segment iterator into our serializable dicts."""
    segments: list[dict] = []
    for seg in segments_iter:
        entry: dict = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        }
        if seg.words:
            entry["words"] = [
                {"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in seg.words
            ]
        segments.append(entry)
        log.info("[%7.1fs - %7.1fs] %s", seg.start, seg.end, seg.text.strip()[:80])
    return segments


def _looks_truncated(segments: list[dict], total_duration: float) -> bool:
    """Heuristic for VAD over-filtering: the transcript ends far before the audio.

    Stylized or music-bedded narration can be misread as non-speech by Silero
    VAD, silently dropping most of the voiceover. Flag it when the last segment
    ends well before the end of the audio.
    """
    if not segments or total_duration <= 0:
        return False
    last_end = segments[-1]["end"]
    return (total_duration - last_end) > 60.0 and last_end < 0.5 * total_duration


def transcribe_raw(
    audio_path: Path,
    language: str | None = None,
    model_size: str = DEFAULT_WHISPER_MODEL,
    *,
    use_vad: bool = True,
    vad_fallback: bool = True,
) -> tuple[list[dict], str]:
    """Transcribe audio to segments with timestamps.

    The Silero VAD filter (``use_vad``) can wrongly classify stylized or
    music-bedded narration as non-speech and drop most of it. When ``use_vad``
    is on and the result looks truncated (ends far before the audio), a loud
    warning is printed; if ``vad_fallback`` is also on, the audio is
    re-transcribed without VAD and the wider-coverage result is kept.

    Returns (segments, detected_language).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    _MODEL_KEEPALIVE.append(model)  # prevent GC-time CUDA teardown segfault

    lang_label = "auto" if language is None else language
    vad_label = "on" if use_vad else "off"
    log.info("Transcribing: %s  (lang=%s, model=%s, vad=%s)",
             audio_path.name, lang_label, model_size, vad_label)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=use_vad,
    )

    detected = info.language
    log.info("Detected language: %s (probability: %.2f)", detected, info.language_probability)

    segments = _collect_segments(segments_iter)
    log.info("Transcription complete: %d segments", len(segments))

    total = float(getattr(info, "duration", 0.0))
    if use_vad and _looks_truncated(segments, total):
        last_end = segments[-1]["end"]
        log.warning("speech ends at %.0fs but the audio is %.0fs long - the VAD "
                    "filter may have dropped narration.", last_end, total)
        if vad_fallback:
            log.info("Retrying without VAD (disable with --no-vad-fallback)...")
            nv_iter, _ = model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True,
                vad_filter=False,
            )
            nv_segments = _collect_segments(nv_iter)
            nv_last = nv_segments[-1]["end"] if nv_segments else 0.0
            log.info("No-VAD pass complete: %d segments (ends at %.0fs)",
                     len(nv_segments), nv_last)
            if nv_last > last_end + 30.0:
                log.info("Using no-VAD result (covers more of the audio).")
                segments = nv_segments
            else:
                log.info("Keeping VAD result (no-VAD did not add coverage).")
        else:
            log.warning("Re-run with --no-vad to capture the full audio.")

    return segments, detected


def format_raw_markdown(segments: list[dict], source_name: str, language: str) -> str:
    """The raw transcript as Markdown, one timestamped line per segment."""
    if not segments:
        return f"# Raw Transcript: {source_name}\n\nNo speech detected.\n"

    duration = format_timestamp(segments[-1]["end"])
    lines = [
        f"# Raw Transcript: {source_name}",
        "",
        f"**Language:** {language} | **Duration:** {duration} | **Segments:** {len(segments)}",
        "",
        "---",
        "",
    ]

    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"**[{ts}]** {seg['text']}")
        lines.append("")

    return "\n".join(lines)


def _group_into_paragraphs(
    segments: list[dict],
    pause_sec: float = PAUSE_THRESHOLD_SEC,
) -> list[list[dict]]:
    if not segments:
        return []

    groups: list[list[dict]] = [[segments[0]]]
    for seg in segments[1:]:
        prev_end = groups[-1][-1]["end"]
        if seg["start"] - prev_end > pause_sec:
            groups.append([seg])
        else:
            groups[-1].append(seg)
    return groups


def _paragraph_lines(para: list[dict], max_chars: int) -> list[str]:
    """Render a paragraph as one timestamp-prefixed line, or several if it exceeds
    max_chars.

    Continuous narration (few >PAUSE_THRESHOLD_SEC pauses) groups into huge
    paragraphs that would otherwise become a single oversized chunk; splitting such a
    paragraph by segment keeps every line within budget.
    """
    line = f"[{format_timestamp(para[0]['start'])}] " + " ".join(seg["text"] for seg in para)
    if len(line) <= max_chars:
        return [line]

    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for seg in para:
        seg_line = f"[{format_timestamp(seg['start'])}] {seg['text']}"
        if current_len + len(seg_line) > max_chars and current:
            lines.append("\n".join(current))
            current = []
            current_len = 0
        current.append(seg_line)
        current_len += len(seg_line) + 1
    if current:
        lines.append("\n".join(current))
    return lines


def chunk_for_llm(segments: list[dict], max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split transcript into text chunks that fit in LLM context.

    Pause-delimited paragraph boundaries are preferred, but a paragraph longer than
    max_chars is itself split so no single chunk overflows the budget.
    """
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for para in _group_into_paragraphs(segments):
        for line in _paragraph_lines(para, max_chars):
            if current_len + len(line) > max_chars and current_lines:
                chunks.append("\n\n".join(current_lines))
                current_lines = []
                current_len = 0
            current_lines.append(line)
            current_len += len(line)

    if current_lines:
        chunks.append("\n\n".join(current_lines))

    return chunks


def query_ollama(prompt: str, system: str, model: str) -> str:
    """The model's reply; TranscriberError if Ollama is unreachable or refuses."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(  # noqa: S310 - OLLAMA_URL is a fixed http:// constant
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 - fixed http:// URL
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TranscriberError(f"Ollama connection error ({OLLAMA_URL}): {exc}. "
                               "Make sure Ollama is running: ollama serve") from exc

    if "error" in data:
        raise TranscriberError(f"Ollama error: {data['error']}")

    return data["message"]["content"]


def structure_transcript(
    segments: list[dict],
    language: str,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> str:
    """Send raw transcript to Ollama for structuring."""
    chunks = chunk_for_llm(segments)
    total = len(chunks)
    log.info("Structuring with %s (%d chunk%s)...", ollama_model, total, "s" if total != 1 else "")

    # Smaller local models drift to English mid-document despite rule #8; naming the
    # language explicitly and forbidding translation keeps the whole output consistent.
    lang_lock = (
        f"\n\nCRITICAL: Write the ENTIRE output — title, summary, headings, sections, "
        f"key points — strictly in the SAME language as the transcript ({language}). "
        f"NEVER translate or switch to another language. Keep proper nouns and domain "
        f"terms exactly as they appear in the source."
    )

    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        is_first = i == 0
        base_system = STRUCTURE_SYSTEM_PROMPT if is_first else STRUCTURE_CONTINUATION_PROMPT
        system = base_system + lang_lock
        label = "transcript" if is_first else "continuation"
        prompt = f"Here is the {label} (language: {language}):\n\n{chunk}"

        log.info("Chunk %d/%d...", i + 1, total)
        result = query_ollama(prompt, system, ollama_model)
        parts.append(result)

    return "\n\n".join(parts)


def run(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    language: str | None = None,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    stage: str = "all",
    use_vad: bool = True,
    vad_fallback: bool = True,
) -> dict[str, Path]:
    """Run the transcription pipeline.

    Returns dict mapping output type to file path. FileNotFoundError for a
    missing input, TranscriberError when ffmpeg or Ollama fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    out_dir = output_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    outputs: dict[str, Path] = {}

    segments: list[dict] = []
    detected_lang: str = "en"

    if stage in ("all", "raw"):
        suffix = input_path.suffix.lower()
        is_audio = suffix in AUDIO_EXTENSIONS

        tmp_dir = None if is_audio else Path(tempfile.mkdtemp(prefix="clipmaker_"))
        try:
            if tmp_dir is None:
                audio_path = input_path
            else:
                audio_path = tmp_dir / f"{stem}.wav"
                extract_audio(input_path, audio_path)

            segments, detected_lang = transcribe_raw(
                audio_path, language=language, model_size=whisper_model,
                use_vad=use_vad, vad_fallback=vad_fallback,
            )
        finally:
            # A failed extraction or transcription must not leave a WAV of the
            # whole video behind in the temp folder.
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        json_path = out_dir / f"{stem}_raw_transcript.json"
        json_data = {
            "source": input_path.name,
            "language": detected_lang,
            "segments": segments,
        }
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        outputs["raw_json"] = json_path
        log.info("Saved: %s", json_path)

        md_path = out_dir / f"{stem}_raw_transcript.md"
        md_path.write_text(
            format_raw_markdown(segments, input_path.name, detected_lang),
            encoding="utf-8",
        )
        outputs["raw_md"] = md_path
        log.info("Saved: %s", md_path)

    if stage in ("all", "structure"):
        if stage == "structure":
            raw_data = json.loads(input_path.read_text(encoding="utf-8"))
            segments = raw_data["segments"]
            detected_lang = raw_data.get("language", "en")
            stem = stem.removesuffix("_raw_transcript")

        if not segments:
            log.warning("No segments to structure.")
            return outputs

        structured = structure_transcript(segments, detected_lang, ollama_model)

        structured_path = out_dir / f"{stem}_structured.md"
        structured_path.write_text(structured, encoding="utf-8")
        outputs["structured_md"] = structured_path
        log.info("Saved: %s", structured_path)

    return outputs


def main(argv: list[str] | None = None) -> None:
    """CLI: transcribe a video or audio file and structure the transcript."""
    ap = argparse.ArgumentParser(
        description="Video transcriber: raw transcript + structured text via local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s video.mp4                              Full pipeline
  %(prog)s video.mp4 --stage raw                   Transcribe only
  %(prog)s video.mp4 --lang en                     Force English
  %(prog)s video.mp4 --no-vad                       Disable VAD (stylized narration)
  %(prog)s raw_transcript.json --stage structure    Structure existing transcript
  %(prog)s video.mp4 --ollama-model gemma2:9b      Use different LLM
""",
    )
    ap.add_argument("input", type=Path,
                    help="Video/audio file, or *_raw_transcript.json for --stage structure")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: same as input file)")
    ap.add_argument("--lang", type=str, default=None,
                    help="Language code, e.g. ru, en (default: auto-detect)")
    ap.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL,
                    help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})")
    ap.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL,
                    help=f"Ollama model for structuring (default: {DEFAULT_OLLAMA_MODEL})")
    ap.add_argument("--stage", choices=["all", "raw", "structure"], default="all",
                    help="all = transcribe + structure, raw = transcribe only, "
                         "structure = process existing raw_transcript.json")
    ap.add_argument("--no-vad", action="store_true",
                    help="Disable the VAD filter. Use for stylized/music-bedded "
                         "narration that VAD wrongly drops as non-speech.")
    ap.add_argument("--no-vad-fallback", action="store_true",
                    help="Disable the automatic no-VAD retry when a VAD transcript "
                         "looks truncated (ends far before the audio).")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    exit_code = 0
    try:
        outputs = run(
            args.input,
            output_dir=args.out_dir,
            language=args.lang,
            whisper_model=args.whisper_model,
            ollama_model=args.ollama_model,
            stage=args.stage,
            use_vad=not args.no_vad,
            vad_fallback=not args.no_vad_fallback,
        )
        for path in outputs.values():
            print(path)
    except (FileNotFoundError, TranscriberError) as exc:
        log.error("%s", exc)
        exit_code = 1

    sys.stdout.flush()
    sys.stderr.flush()

    # If a CUDA WhisperModel was loaded, terminate with os._exit() to bypass the
    # native teardown segfault (see _MODEL_KEEPALIVE). All outputs are already on
    # disk by now, so skipping interpreter cleanup loses nothing. Stages that
    # load no model (e.g. --stage structure) keep the normal exit path.
    if _MODEL_KEEPALIVE:
        os._exit(exit_code)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

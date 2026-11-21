---
name: video-transcriber
description: >
  Transcribe any video or audio file to raw verbatim transcript with timestamps,
  then structure it into a clean topic-segmented document via local LLM (Ollama).
  Two-stage pipeline: faster-whisper → Ollama. Supports Russian + English with
  auto-detection. Use when the user needs a transcript, structured summary,
  or wants to prepare text from a video/audio source.
---

# Video Transcriber

Two-stage pipeline: raw transcription (faster-whisper) → structured text (Ollama local LLM).

## Quick Reference

```bash
# Full pipeline (transcribe + structure)
python -m clipmaker.video_transcriber video.mp4

# Transcribe only (Stage 1)
python -m clipmaker.video_transcriber video.mp4 --stage raw

# Structure existing raw transcript (Stage 2 only)
python -m clipmaker.video_transcriber raw_transcript.json --stage structure

# Force language + custom LLM
python -m clipmaker.video_transcriber video.mp4 --lang en --ollama-model mistral-nemo

# Custom output directory
python -m clipmaker.video_transcriber video.mp4 --out-dir path/to/output
```

## Stage 1: Raw Transcription

**Tool:** faster-whisper (CUDA, float16)
**Model:** `medium` by default (override with `--whisper-model`)

Input: video (.mp4, .mkv, .avi, .mov, .webm) or audio (.mp3, .wav, .flac, .aac, .ogg)

Process:
1. Extract audio from video via ffmpeg → temp WAV (16kHz mono)
2. Transcribe with word-level timestamps, VAD filter enabled
3. Auto-detect language (or accept `--lang` override)

Output files (in same directory as input, or `--out-dir`):
- `{name}_raw_transcript.json` — segments with start/end timestamps and word-level detail
- `{name}_raw_transcript.md` — human-readable markdown with timestamps

### JSON format

```json
{
  "source": "video.mp4",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 4.52,
      "text": "Welcome to EVE Fanfest 2026",
      "words": [
        {"word": "Welcome", "start": 0.0, "end": 0.42},
        {"word": "to", "start": 0.42, "end": 0.56}
      ]
    }
  ]
}
```

## Stage 2: Structured Text

**Tool:** Ollama (local LLM, no cloud API)
**Model:** `mistral-small` by default (14B, best bilingual on this machine)

Available models on this machine:
- `mistral-small` (14B) — default, best for bilingual RU/EN
- `mistral-nemo` (7B) — faster, slightly lower quality
- `llama3.1:8b` (4.7B) — fastest, English-biased

Process:
1. Group segments into paragraphs by natural pauses (>3 sec gaps)
2. Chunk paragraphs to fit LLM context (~12K chars per chunk)
3. First chunk: full system prompt (title, summary, TOC, sections)
4. Continuation chunks: lighter prompt (sections only)
5. Combine all chunks into final document

Output: `{name}_structured.md`

### Structured output format

```markdown
# [Inferred Title]

## Summary
[2-3 sentences]

## Table of Contents
1. [Section] [HH:MM:SS]

---

## [Section Name]
*[HH:MM:SS — HH:MM:SS]*

[Clean structured text — no filler words, fixed sentences]

**Key points:**
- ...
```

## Prerequisites

1. **ffmpeg** in PATH (for video → audio extraction)
2. **faster-whisper** Python package with CUDA support
3. **Ollama** running locally (`ollama serve`) for Stage 2

Verify Ollama: `curl http://localhost:11434/api/tags`

## When to Use Each Stage

| Scenario | Stage |
|----------|-------|
| Need full transcript + structured doc | `--stage all` (default) |
| Only need raw timestamps for mapping | `--stage raw` |
| Already have raw JSON, want structured | `--stage structure` |
| B-roll pipeline Step 2 | `--stage raw` |
| B-roll pipeline Step 3 | `--stage structure` |

## Design Decisions

- **Standalone module** — not coupled to project-based narration workflow. Works with arbitrary files.
- **Chunking by pause gaps** — preserves topical coherence (not arbitrary char splits).
- **urllib over requests** — no external dependency, stdlib only.
- **Temp audio cleanup** — extracted WAV is in tempfile dir, auto-deleted.
- **Output next to input** — files land beside the source video unless `--out-dir` is specified.

## Known Limitations

- **No diarization** — speaker identification is not implemented. Structured LLM may infer speakers from context.
- **Very long videos (2+ hours)** — not tested. May produce disjointed structured sections.
- **num_ctx not tuned** — Ollama uses default context window. For very long transcripts, may need `num_ctx` parameter.
- **No `--stage all` with JSON input** — if input is JSON, only `--stage structure` is valid.

## Real-World Results

### Expansion Unlocked Cradle of War (38 min, English)
- Raw: 304 segments, language auto-detected (p=1.0)
- Structured: 3 chunks via mistral-small
- Quality: clean sections, accurate speaker attribution (FC Okami, FC Mercury, FC K1P1, FC Havron)
- Catalog JSON updated with 27 timeline entries from transcript

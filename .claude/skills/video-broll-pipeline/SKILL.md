---
name: video-broll-pipeline
description: >
  B-roll assembly pipeline for video production. Takes a source video (keynote, presentation,
  gameplay) and a narration audio, produces a Sony Vegas script that places B-roll clips
  synced to the narration. Uses VLM visual analysis, speech transcription, and structured
  transcript as three mandatory data sources. Use when the user needs to create a B-roll
  timeline, map narration to source video, or generate Vegas import scripts.
---

# Video B-Roll Pipeline

Assemble B-roll clips from a source video onto a timeline synced with narration audio.

## Pipeline Steps (ALL MANDATORY — DO NOT SKIP ANY)

### Step 1: VLM Frame Analysis

**Tool:** `python -m clipmaker.analyzer <video.mp4> --game <game> --interval <N>`

Produces: `catalog/<game>/<video>.json` with:
- `scenes[]` — visual scene descriptions with `type` (gameplay/speaker/slide/trailer/concept_art/ui/video_clip) and `description`
- `timeline[]` — OCR text per frame
- `ocr_text[]` — all text visible on screen
- `tags[]`, `summary`, `title`

**Critical:** This is the ONLY source of truth for what is VISUALLY shown at each timestamp.
Speech transcripts describe what is SAID, not what is SHOWN — these often diverge,
especially in keynotes where a speaker talks about topic A while the screen shows topic B.

**Interval guidance:**
- 30s: fast, low resolution, good for first pass
- 10s: medium, usually sufficient
- 5s: high resolution, produces `scenes` with type/description (recommended for B-roll)

**If scenes field is missing:** the VLM was run before the scenes feature was added.
Re-run with `--force` and interval ≤10s.

### Step 2: Audio Transcription (source video)

**Tool:** `python -m clipmaker.video_transcriber <video.mp4> --stage raw`

Produces: `<video>_raw_transcript.json` + `<video>_raw_transcript.md`

This captures what the SPEAKER SAYS with timestamps. Used for topic identification,
NOT for visual content mapping.

### Step 3: Structuring (source video)

**Tool:** `python -m clipmaker.video_transcriber <raw_transcript.json> --stage structure`

Produces: `<video>_structured.md` — topic-segmented document with headings and timestamps.

Useful for understanding the presentation structure and topic flow.

### Step 4: Vegas Export

**Tool:** `python -m clipmaker.vegas_export <structured.md>`

Produces: `<video>_vegas_sections.json` — sections extracted from structured transcript.

### Step 5: Narration Transcription

**Tool:** `python -m clipmaker.video_transcriber <narration.mp3> --stage raw`

Produces: raw transcript of the narrator's audio — what TOPICS the narrator covers and WHEN.

### Step 6: Timings Mapping (THE CRITICAL STEP)

**Input:** Three data sources:
1. **VLM scenes** (Step 1) — what is VISUALLY SHOWN at each timestamp
2. **Source structured transcript** (Step 3) — what the speaker SAYS at each timestamp
3. **Narration transcript** (Step 5) — what the narrator covers and when

**Algorithm:**
1. Split narration into fragments by topic (from narration transcript)
2. For each fragment, identify the narration topic
3. Find VLM scenes where the VISUAL content matches the narration topic
4. Prefer scene types: gameplay > video_clip > slide > concept_art > speaker
5. Assign `clip_start` / `clip_end` from VLM scene timestamps
6. Verify: clip duration ≥ narration fragment duration
7. Verify: no overlapping clip ranges between fragments

**Output:** `timings.json` or hardcoded values in C# script

### RULES FOR STEP 6:

```
NEVER map clips using only the structured transcript section names.
  → Speech timestamps describe what is SAID, not what is SHOWN.
  → "Minmatar Weapons [40:25]" in transcript ≠ Minmatar weapon footage at 40:25.

ALWAYS cross-reference VLM scenes to find where the visual content actually appears.
  → The VLM scene at 32:50 with type=slide, description="Amarr Laser Rifle and Minmatar SMG"
    is the correct source for a weapons fragment, regardless of when the speaker
    talks about weapons.

ONE automated pass, then user verifies in Vegas.
  → Do NOT iterate blindly. The user sees the video, you do not.
  → If the user reports a problem, fix surgically by fragment number.
  → Never rebuild all timings from scratch — fix only the reported fragments.
```

### Step 7: Vegas Script

Generate a C# script for Vegas Pro that:
1. Removes existing B-Roll tracks (prevent stacking)
2. Removes old markers/regions from previous runs
3. Imports source video from media pool
4. Creates a new video track ("B-Roll v2")
5. Places clips at `AUDIO_OFFSET_SEC + fragment.start_sec`
6. Sets `take.Offset` to `clip_start` for each fragment
7. Adds fade in/out (0.5s default)
8. Creates markers and regions
9. Shows debug MessageBox with all clip offsets

**Template:** Follow existing pattern in `vegas_scripts/import_*.cs`

## Checklist Before Delivering Timings

- [ ] VLM analysis exists with `scenes` field (not just OCR timeline)
- [ ] Every fragment's clip range points to a VLM scene with matching visual content
- [ ] No two fragments share overlapping clip ranges
- [ ] Every clip range duration ≥ narration fragment duration
- [ ] Vegas script includes cleanup of old tracks/markers
- [ ] Vegas script includes debug MessageBox

## Known Issues

### VLM Chunking Quality
- Processing >50 frames triggers chunked mode (chunks of 50)
- With 5s interval on a 46-min video: 550 frames = 11 chunks
- Some chunks may fail to parse (retry with different temperature)
- Specific frame ranges can crash CUDA — fill manually from original VLM

### Speech ≠ Visuals (keynotes)
- Presenters often show gameplay/slides while talking about unrelated topics
- Structured transcript timestamps are speech-based, not visual-based
- The ONLY reliable source of visual content is VLM scene analysis

### Vegas Script Caching
- Vegas may cache compiled scripts between runs
- Use hardcoded timings in C# (not external JSON) to avoid read-path issues
- Always include debug MessageBox to verify the correct script is running

## Example: Successful Pipeline (Expansion Unlocked)

1. VLM: 76 frames at 30s → rich timeline with OCR
2. Transcription: 304 segments, English
3. Structuring: clean sections with speaker attribution
4. Vegas export: 31 sections
5. Narration: 266 segments, Russian
6. Mapping: sequential topic matching (speech ≈ visuals for this video)
7. Vegas script: 19 fragments → user reported "edited very little"

**Why it worked:** Expansion Unlocked is a visual-heavy presentation where
speech closely matches screen content. Sequential mapping was sufficient.

## Example: Failed Pipeline (Vanguard Keynote)

Same steps 1-5 completed. Step 6 failed THREE TIMES because:
- v1: mapped by speech transcript section names → wrong visual content
- v2: guessed "gameplay" timestamps from section titles → still wrong
- v3: used VLM scenes for mapping → correct visual content

**Why it failed initially:** Vanguard Keynote has extensive speaker-on-stage
footage where speech topics diverge from screen visuals. Speech-based mapping
produced wrong clips. VLM scene-based mapping was required.

**Lesson:** Always use VLM scenes. If it works without them (like Expansion Unlocked),
that's luck — not a process you can rely on.

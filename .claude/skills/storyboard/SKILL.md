---
name: storyboard
description: >
  Generate a storyboard from narration text: split into fragments, search local
  catalog and YouTube for matching footage per fragment, produce a storyboard
  document. Then align fragments to audio timestamps and optionally extract clips.
  Use when starting video assembly from a finished narration script.
---

# Storyboard

Text → fragments → footage search → audio alignment → clip extraction.

## Before you trust the search results

`report.py` generates queries from the narration. That finds explainers and
retrospectives; it does **not** find recordings of the events themselves, because
events are titled with place names, codes and group names, not with narration
phrases. Its empty result is not evidence that the footage is missing.

So after `report.py`, always run a second pass by hand:

1. Pull the proper nouns out of the narration — places, names, codes, dates.
2. Search each one: `yt-dlp --skip-download --flat-playlist --print
   "%(duration)s|%(title)s|%(webpage_url)s" "ytsearch5:<term>"`.
3. Confirm each hit with `--print "%(upload_date)s|%(duration)s|%(width)sx%(height)s"`.
   The same regions get fought over for decades — an upload from 2009 or 2017 looks
   identical in a title.
4. If you keep research notes for the topic, read them first:
   community channels post the links directly, with dates.

Only after that pass may the storyboard say a fragment has no footage.

## Full Pipeline

### Step 1: Project Setup

```
projects/<project-name>/
  text.md          ← narration text (required)
  transcript.json  ← Whisper transcript of audio (needed for steps 3-4)
```

Get the transcript via `/video-transcriber`:
```bash
python -m clipmaker.video_transcriber narration.mp3 --stage raw --out-dir projects/<project>/
# Rename *_raw_transcript.json → transcript.json
```

### Step 2: Generate Storyboard

```bash
python -m clipmaker.report <project-name> [game]

# Example
python -m clipmaker.report my-episode assassins_creed_4

# List projects
python -m clipmaker.report
```

**What it does:**
1. Splits `text.md` into fragments (by sentence, ~40+ words each)
2. Searches VLM catalog + YouTube for each fragment
3. Writes `storyboard.md` with search results
4. Writes `moments.json` — the same candidates machine-readable, with
   timecodes. This is what build_storyboard cuts from, so the moment chosen
   here IS the clip that lands on the timeline.

**Output:** `projects/<project>/storyboard.md` + `moments.json`

```markdown
## Fragment 1
> Narration text for this fragment

**Search query:** `diablo iv boss fight`

### My videos
1. **Diablo IV Boss Fight** (2:15)
   - File: `Diablo IV Vessel Hatred Seasons Death 1.mp4`
   - Tags: boss, combat, death

### YouTube videos
1. **Diablo IV All Boss Fights**
   - Channel: GameClips | Duration: 45:00 | Views: 1.2M
```

### Step 3: Manual Curation (optional)

Open `storyboard.md` and override fragments where the semantic pick is wrong.
Record choices in `projects/<project>/curation.json` — NEVER in module code:

```json
{
  "1": "Diablo IV Vessel Hatred Seasons Death 1.mp4",
  "13": ["Diablo IV Butcher Season Death Champion 1.mp4", 0, 126]
}
```

A bare name means "this video, range from its own semantic moment".
Fragments absent from curation.json use the top moment from `moments.json`.

### Step 4: Audio Alignment

```bash
python -m clipmaker.align <project-name>
```

The project is a positional argument (e.g. `my_game/drafts/<slug>`);
`--project <project-name>` works too.
is merged; before that it fails with "No storyboard.md".

**What it does:**
1. Reads fragments from `storyboard.md`
2. Matches each to Whisper transcript via SequenceMatcher
3. Determines start_sec / end_sec for each fragment

**Output:** `projects/<project>/timings_v1.json`

### Step 5: Build Final Timings

```bash
python -m clipmaker.build_storyboard --project <project-name>
```

**What it does:**
1. Merges `curation.json` (wins) with `moments.json` (fills the rest) and
   audio timings (`timings_v1.json`)
2. Each entry records `clip_source` (curation / moment / curation+moment) so
   a wrong clip traces to the decision that produced it

**Output:** `projects/<project>/timings.json`

### Step 6: Extract Clips (optional)

```bash
python -m clipmaker.clip_extractor <project-name>
```

Cuts exactly the ranges in `timings.json`, frame-accurately (re-encode; no
keyframe snapping). A fragment without a range is listed as `NO RANGE` in the
cut list — fix it in `curation.json`, nothing gets cut by guesswork.

**Output:** `projects/<project>/clips/` + `_cut_list.md`

### Step 7: Contact Sheet (verify BEFORE Vegas)

```bash
python -m clipmaker.contact_sheet --project <project-name>
```

One HTML page: per fragment — narration text, chosen clip and source of the
choice, 4 frames from exactly the range that will be placed. Give it to the
user to review; a wrong picture is fixed in `curation.json` and rebuilt.
Five minutes here replaces discovering mismatches on the Vegas timeline.

**Output:** `projects/<project>/contact_sheet/contact_sheet.html`

### Step 8: Vegas Layout

```bash
python -m clipmaker.layout --project <project-name> --fps <project fps>
```

Validates the placement (overlaps, missing media, gaps) and writes
`layout.txt` + `layout_review.md`. Review the table with the user, then they
run `vegas_scripts/import_layout.cs` and pick the layout.txt.

## Module Map

```
storyboard.py        → split text into fragments
report.py            → catalog + YouTube search → storyboard.md + moments.json
align.py             → align fragments to audio → timings_v1.json
build_storyboard.py  → curation.json + moments.json + timings_v1 → timings.json
clip_extractor.py    → cut exact ranges from timings → clips/
contact_sheet.py     → per-fragment text + frames → contact_sheet.html
layout.py            → validated Vegas layout → layout.txt (+ review)
```

## Storyboard vs B-roll Pipeline

| | Storyboard | B-roll Pipeline |
|---|---|---|
| **Source** | Multiple videos from catalog | Single source video |
| **Curation** | curation.json + semantic moments | Automatic (VLM scenes) |
| **Output** | Extracted clips for Vegas | C# script for Vegas |
| **Use case** | Authored video (review, essay) | News-style (keynote recap) |

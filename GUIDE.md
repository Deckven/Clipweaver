# Clipmaker user guide

What each part of Clipmaker does, and how to go from your own gameplay captures
and a narration to a B-roll timeline in Vegas Pro, step by step.

Clipmaker is a pre-production assistant. It finds the right moments in hours of
footage, lays them against your narration and places them on the timeline. The
final edit — pacing, cuts, music, the last picks — stays with you.

- [1. What it does](#1-what-it-does)
- [2. What you need](#2-what-you-need)
- [3. Folders and files](#3-folders-and-files)
- [4. Ten-minute tour on the sample (no GPU)](#4-ten-minute-tour-on-the-sample-no-gpu)
- [5. Step by step: captures + narration → Vegas timeline](#5-step-by-step-captures--narration--vegas-timeline)
- [6. When the narration is re-recorded](#6-when-the-narration-is-re-recorded)
- [7. Other tools](#7-other-tools)
- [8. Project folder reference](#8-project-folder-reference)
- [9. Limits and troubleshooting](#9-limits-and-troubleshooting)

## 1. What it does

| Job | Module | Needs |
|---|---|---|
| Describe every capture: title, summary, tags, a caption with on-screen text every few seconds, scenes with start and end | `analyzer` | CUDA GPU, 12 GB+ |
| Read on-screen text (UI, system names, numbers) for the captions | `ocr` (used by `analyzer`) | GPU recommended |
| Find a moment by exact on-screen text | `ocrgrep` | — |
| Find a moment by describing the picture, in any language | `embed_index` | CPU is enough for search |
| Turn a narration paragraph into short picture descriptions | `shot_query` | Ollama (optional) |
| Storyboard: for every narration fragment, the best moments from your catalog plus YouTube | `report` | network for YouTube |
| Transcribe the narration with word timings | `transcribe`, `video_transcriber` | CUDA GPU |
| Find where each fragment is spoken in the audio | `align` | — |
| Merge your manual picks with the machine picks into final timings | `build_storyboard` | — |
| A page of frames per fragment, to check picks before Vegas | `contact_sheet` | ffmpeg |
| Cut each chosen range into its own file | `clip_extractor` | ffmpeg |
| Validated timeline layout for Vegas | `layout` + `vegas_scripts/import_layout.cs` | Vegas Pro 22 |
| Stills and contact sheets for articles | `stills`, `grid` | ffmpeg; yt-dlp for remote video |

It does not edit the video, choose music or judge pacing.

## 2. What you need

| | Required for | Notes |
|---|---|---|
| Python 3.11+, ffmpeg and ffprobe on `PATH` | everything | |
| CUDA GPU, 12 GB+ VRAM | indexing (`analyzer`), transcription | tested on 16 GB; the 7B vision model in NF4 takes about 5 GB, a 10-frame call peaks at 9.8 GB |
| [Ollama](https://ollama.com) with `mistral-small` | picture queries (`shot_query`), structured transcripts | without it the search embeds the paragraph itself — worse, but it works |
| Vegas Pro 22 | the timeline loader | the rest of the pipeline does not need it |
| Pexels / Pixabay API keys | stock footage in `production_plan` | optional |

Install and configure as in the [README](README.md#install): torch first, then
`pip install -e ".[all,test]"`, then `cp .env.example .env`.

## 3. Folders and files

```
footage/raw/<Game>/        # your captures (CLIPMAKER_RAW_FOOTAGE_DIR)
catalog/<game>_videos/     # one JSON per capture, written by the analyzer (CLIPMAKER_CATALOG_DIR)
catalog/_index/            # semantic index, written by embed_index build
projects/<project>/        # one folder per video you are making (CLIPMAKER_PROJECTS_DIR)
```

All four locations can be moved with the variables in `.env`.

A game is a key in `GAMES`. Out of the box it holds one, the sample
`assassins_creed_4`. Put your own games in `clipmaker/games_local.py`; when that
file exists, `clipmaker/config.py` loads it instead of the sample, and your
list stays out of the files you pull:

```python
from pathlib import Path


def games(RAW_FOOTAGE_DIR: Path, EDIT_PROJECTS_DIR: Path) -> dict:
    return {
        "my_game": {
            "video_dirs": [RAW_FOOTAGE_DIR / "My Game"],   # where its captures are
            "catalog_subdir": "my_game_videos",           # folder under catalog/
            "display_name": "My Game",
        },
    }


DEFAULT_GAME = "my_game"
```

Keep `assassins_creed_4` in your list if you still want to run the sample tour.

Optionally give the game a vocabulary in `clipmaker/game_vocab.py`: a
dictionary like `ELDEN_RING` (description, regions, bosses, enemies, concepts,
actions — 50–80 names in all) registered in `vocab_for()`. It helps the vision
model and the picture queries use the right names.

## 4. Ten-minute tour on the sample (no GPU)

`examples/sample/` holds four catalog entries of the author's own Assassin's
Creed IV captures (no video) and a four-paragraph narration. Run from the
repository root:

```bash
export CLIPMAKER_CATALOG_DIR=examples/sample/catalog      # PowerShell: $env:CLIPMAKER_CATALOG_DIR="examples/sample/catalog"
export CLIPMAKER_PROJECTS_DIR=/tmp/clipmaker-projects     # PowerShell: $env:CLIPMAKER_PROJECTS_DIR="$env:TEMP\clipmaker-projects"
```

**1. Exact on-screen text.** Which moments show the harbourmaster menu:

```bash
python -m clipmaker.ocrgrep "harbou?rmaster" --game assassins_creed_4
```

One block per video, with timecodes of every hit:

```
Assassin's Creed IV Black Flag Resynced Ship Upgrade Trading 1.mp4 [220s] @ 00:10, 00:15, 00:20, 00:50 (+7)
    HARBOURMASTER; Armored Hull IV; Upgrades; ...
```

**2. Describe the picture.** Build the semantic index once (downloads the
multilingual E5 model, ~1 GB; about 20 s on CPU for four clips), then search:

```bash
python -m clipmaker.embed_index build --game assassins_creed_4
python -m clipmaker.embed_index search "sailing ship firing cannons at enemy ships in a stormy sea" --game assassins_creed_4 --min-z 0
```

Each hit is a moment: score, level (`scene` or `video`), start–end, file, and
the scene description that matched. `--min-z 0` switches off the outlier
cutoff, which only means something on a full-size catalog.

**3. A storyboard from a narration.**

```bash
mkdir -p /tmp/clipmaker-projects/ac4_demo
cp examples/sample/narration.md /tmp/clipmaker-projects/ac4_demo/text.md
python -m clipmaker.report ac4_demo assassins_creed_4
```

`report` writes `storyboard.md` (readable) and `moments.json` (what the next
steps cut from) into the project folder. Paragraphs are merged into fragments
of at least 40 words, so this short sample becomes one fragment. With Ollama
running, the storyboard also shows the picture queries written for it; without
it, the fragment text itself is searched. On four clips the top pick can be the
wrong one — that is what the curation and the contact sheet in section 5 are for.

The rest of the pipeline needs your own footage and a recorded narration.

## 5. Step by step: captures + narration → Vegas timeline

Every module prints its options with `python -m clipmaker.<module> --help`.
`<project>` below is a folder name under `projects/`; nested names such as
`my_game/drafts/episode_1` work too.

### Step 1. Index your captures (once per new footage)

```bash
python -m clipmaker.analyzer --game my_game --interval 5
```

Without paths it walks the game's `video_dirs`; you can also pass files or
folders. Already indexed files are skipped (`--force` re-indexes). `--interval`
is seconds between sampled frames: 5 gives scene descriptions with start and
end and is the right choice for B-roll; 30 is a fast first pass. Output: one
JSON per capture in `catalog/my_game_videos/`.

Tidy names (optional): `--regenerate-suggestions` proposes names from the
catalog, `rename_videos --game my_game` shows the renames, `--apply` does them.

### Step 2. Build the semantic index

```bash
python -m clipmaker.embed_index build --game my_game
python -m clipmaker.embed_index stats
```

Run again after indexing new footage; only new or changed entries are embedded
(`--force` rebuilds all).

### Step 3. Start a project

```
projects/<project>/
  text.md          # the narration script, paragraphs separated by blank lines
```

Optional: `queries.json` — your own search query for a fragment number, when
the automatic one misses (`{"3": "harbourmaster upgrade menu"}`).

### Step 4. Storyboard

```bash
python -m clipmaker.report <project> my_game
```

Several games in one video: `"game_a,game_b"`. Add `--images` (or
`--images=N`) for still images from game wikis and Wikimedia Commons for
fragments without footage.

Read `storyboard.md`: for each fragment — the text, the picture queries, your
best moments with timecodes and why they matched, and YouTube results.
`moments.json` keeps the same picks in machine form.

An empty "My videos" section does not prove the footage is missing: events are
often titled with codes and names the narration never uses. Search those names
directly with `ocrgrep` or `embed_index search` before giving up.

### Step 5. Fix the picks you disagree with

Write your choices into `projects/<project>/curation.json`, never into code:

```json
{
  "1": "My Game Boss Fight 1.mp4",
  "4": ["My Game Harbour 2.mp4", 95, 130]
}
```

A bare file name means "this video, at its best moment"; a list sets the range
in seconds. Fragments not listed keep the pick from `moments.json`. Your choice
always wins.

### Step 6. Record and transcribe the narration

Put the recording into the project folder as `narration.mp3` (or
`<project>.mp3`), then:

```bash
python -m clipmaker.transcribe --project <project>
```

This writes `transcript.json` with word timings. The `--project` mode assumes a
Russian narration; for any other language transcribe the file directly:

```bash
python -m clipmaker.transcribe --file projects/<project>/narration.mp3 --lang en --out projects/<project>/transcript.json
```

### Step 7. Align the text to the audio

```bash
python -m clipmaker.align <project>
```

Finds where each fragment is spoken and writes `timings_v1.json`. A warning
`WEAK ANCHORS on fragments …` means the recording differs from `text.md` there
(an ad-lib, a rewrite) — check those fragments.

### Step 8. Final timings

```bash
python -m clipmaker.build_storyboard --project <project>
```

Merges `curation.json`, `moments.json` and `timings_v1.json` into
`timings.json`. Every entry records where its clip came from (`curation`,
`moment` or both), so a wrong clip traces back to the decision that made it.

### Step 9. Check before Vegas

```bash
python -m clipmaker.contact_sheet --project <project>
```

Open `contact_sheet/contact_sheet.html`: per fragment, the narration, the
chosen clip and four frames from exactly the range that will be placed. Fix a
wrong picture in `curation.json`, then rerun steps 8–9. Five minutes here saves
finding mismatches one by one on the timeline.

Cut the ranges into separate files if you want them (optional):

```bash
python -m clipmaker.clip_extractor <project>
```

Output: `clips/` and `_cut_list.md`. A fragment without a range is listed as
`NO RANGE` — nothing is cut by guesswork.

### Step 10. Layout and Vegas

Tell the layout the Vegas frame rate and the narration file, once per project,
in `projects/<project>/vegas.json`:

```json
{"fps": 59.94, "audio": "narration.mp3"}
```

or pass `--fps` and `--audio` on the command line. Then:

```bash
python -m clipmaker.layout --project <project>
```

It checks the placement (overlaps, missing media, gaps), quantizes every event
to whole frames and writes `layout.txt` for Vegas plus `layout_review.md` for
you. Read the review table first.

In Vegas Pro 22:

1. Open the project with the narration mp3 on the timeline.
2. Tools → Scripting → Run Script → `vegas_scripts/import_layout.cs`.
3. Pick the project's `layout.txt`.
4. Confirm the summary. A B-roll track appears; a track with the same name
   from an earlier run is replaced.

All times in `layout.txt` are relative to the start of the narration, so the
clips follow the mp3 wherever it sits on the timeline.

## 6. When the narration is re-recorded

| Situation | Tool |
|---|---|
| New recording, same script: move existing events to the new audio | `remap --project <p> --old <old transcript> --new <new transcript>` — flags slots that grew, shrank under 1.5 s or lost their anchor |
| Raw takes on the timeline instead of one mixed mp3 | `take_lags <mix.mp3> <takes...> --json takes.json` measures where each take sits; copy the list into `takes` in `vegas.json`, run `layout` (or `vegas_takes sync --project <p>` to refresh an existing `layout.txt`), and the loader anchors on the takes |
| Check the take table against the saved `.veg` without opening Vegas | `veg_scan <project.veg> --project <p>` |
| YouTube chapters computed on the narration, but the render is shifted | `render_offset <narration.mp3> <render.mp4>` |

## 7. Other tools

| Module | What for |
|---|---|
| `stills grab / sheet / images / fetch` | A frame at an exact second from a local or remote video, letterbox removed; contact sheets to choose from |
| `grid NAME --from S --to E --out grid.jpg` | A labelled frame grid of one capture, found by part of its name, to check a range |
| `image_search fandom|commons <query>` | Still images from game wikis (publisher artwork — editorial use) and Wikimedia Commons (real licence carried through) |
| `video_transcriber <video>` | Transcript of a source video plus a structured, topic-segmented version (Ollama) |
| `vegas_export <video>_structured.md` | Sections with timestamps from that structured transcript, for a one-source B-roll edit (keynotes, presentations) |
| `production_plan <project> [game]` | Shot type and duration per fragment, coverage from your catalog, YouTube and stock, and a shooting list for the gaps |
| `youtube_search`, `stock_search` | Search YouTube (yt-dlp) or Pexels / Pixabay from the command line |
| `retrieval_eval` | Recall of each search layer against picks from finished videos; the numbers in the README come from it |

## 8. Project folder reference

| File | Written by | Read by |
|---|---|---|
| `text.md` | you | `report`, `production_plan` |
| `queries.json` | you, optional | `report` |
| `storyboard.md`, `moments.json` | `report` | you; `build_storyboard` |
| `curation.json` | you, optional | `build_storyboard` |
| `narration.mp3` | you | `transcribe`, Vegas |
| `transcript.json` | `transcribe` | `align` |
| `timings_v1.json` | `align` | `build_storyboard` |
| `timings.json` | `build_storyboard` | `contact_sheet`, `clip_extractor`, `layout` |
| `contact_sheet/contact_sheet.html` | `contact_sheet` | you |
| `clips/`, `_cut_list.md` | `clip_extractor` | Vegas, you |
| `vegas.json` | you; `vegas_takes` | `layout` |
| `layout.txt`, `layout_review.md` | `layout` | `import_layout.cs`, you |

## 9. Limits and troubleshooting

- **Narration says what a video is about; the catalog says what is on screen.**
  Searching a paragraph directly often misses. That is why `shot_query` rewrites
  it into picture descriptions — keep Ollama running for real projects.
- **A paragraph with nothing to film gets no picture queries.** The storyboard
  says so instead of forcing a random clip; fill that fragment by hand.
- **Scene descriptions need `--interval` ≤ 10 s.** Captures indexed at 30 s
  only have per-frame captions; re-index with `--force --interval 5`.
- **Indexing long captures** splits the frames into chunks of 50; on 16 GB the
  default batch is 12 frames per call.
- **No GPU?** Indexing and transcription need CUDA. Search, storyboard,
  curation, alignment, contact sheet and layout run on CPU — with a catalog and
  a `transcript.json` made elsewhere, the rest of the pipeline works.

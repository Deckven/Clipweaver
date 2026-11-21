---
name: shooting-list
description: >
  Generate a shooting list from narration text + audio: determine what game footage
  needs to be recorded, with exact durations. Transcribes audio for timing,
  searches catalog for existing footage, and outputs a prioritized task list
  of clips to capture in-game. Use when preparing to record gameplay footage
  for a video project.
---

# Shooting List

Input: narration text + audio track. Output: list of game footage to record, with durations.

## Pipeline

### Step 1: Transcribe Audio

```bash
python -m clipmaker.video_transcriber narration.mp3 --stage raw
```

Produces timecodes for each narration segment — this determines the **duration** of each clip.

### Step 2: Split Text into Fragments

The narration text is split into fragments by sentence. Each fragment = one shot in the final video.

For each fragment, determine:
- **Duration** — from audio transcription timecodes (step 1)
- **Shot type** — via `shot_classifier.py`
- **What should be on screen** — text content analysis

### Step 3: Search Existing Catalog

For each fragment, search the VLM catalog for matching footage:

```python
from clipmaker.report import _load_catalog, search_catalog
catalog = _load_catalog("assassins_creed_4")
matches = search_catalog(fragment_text, catalog, max_results=3)
```

**Three outcomes:**
- **Exact match** (score ≥ 0.5) — footage exists, no recording needed
- **Approximate match** (score 0.3–0.5) — usable but better to record fresh
- **No match** (score < 0.3) — **must record**

### Step 4: Classify Missing Footage

For each fragment without footage, determine the type:

| Type | What to record |
|------|---------------|
| `gameplay` | Active gameplay: combat, farming, exploration |
| `cinematic` | Use official trailer (don't record) |
| `broll_abstract` | Atmospheric shots: landscapes, flights, stations |
| `screen_recording` | UI capture: menus, store, skill trees |
| `talking_head` | Narrator on camera (webcam, not in-game) |
| `text_overlay` | Compositing in Vegas (don't record) |

### Step 5: Generate the List

Output format:

```markdown
## Shooting List

Total footage to record: **4 min 30 sec** (12 clips)

### Priority: HIGH (no existing footage)

1. **[gameplay] Mephisto boss fight** — 25s
   Fragment 6: "VoH: Spiritborn, Nahantu, Kurast..."
   Record: Mephisto fight in Lord of Hatred, epic moment

2. **[gameplay] Infernal Hordes run** — 18s
   Fragment 9: "Season 8 Belial, season 9 nightmare dungeons..."
   Record: active Infernal Hordes gameplay, mob waves

### Priority: MEDIUM (similar footage exists but not ideal)

3. **[screen_recording] Steam page with reviews** — 8s
   Fragment 17: "Did LoH deliver? 64.5K Steam, Metacritic 84..."
   Record: screenshot of Diablo IV Steam page with reviews

### No recording needed:
- Fragment 1: ✅ in catalog (Diablo IV VoH Seasons...)
- Fragment 3: ✅ in catalog (Diablo Trailers 1.mp4)
- Fragment 14: 🎬 text overlay (composit in Vegas)
```

## Automated via Production Plan

Most of this pipeline is already implemented in `production_plan.py`:

```bash
# Full production plan includes a shooting list section
python -m clipmaker.production_plan <project-name> <game>
```

The `plan.md` output contains a **Shooting List** section with:
- Fragment number
- Shot type
- Description
- Estimated duration

## Manual Workflow (without production_plan)

If you only need the shooting list without a full plan:

1. Transcribe audio → get durations per fragment
2. Split text into fragments
3. For each fragment: `grep -rl "keyword" catalog/<game>/ --include="*.json"`
4. No matches → add to shooting list
5. Clip duration = fragment duration + 5s buffer

## Recording Tips

- **Record extra:** add 10-15s buffer per clip, trim in Vegas
- **Multiple takes:** for combat and action — 2-3 takes, pick the best
- **Clean UI:** disable overlays (FPS counter, chat) unless needed for the shot
- **Resolution:** record at the same resolution as the final video
- **File naming:** name files by fragment number: `F06_mephisto_boss.mp4`
- **After recording:** index new footage via `/vlm-catalog-index`

## Related Skills

- `/vlm-catalog-index` — index recorded footage into the catalog
- `/production-plan` — full production plan (includes shooting list)
- `/catalog-search` — manual catalog search
- `/storyboard` — storyboard with audio-aligned timings

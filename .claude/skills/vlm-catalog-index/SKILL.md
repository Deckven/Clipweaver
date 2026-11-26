---
name: vlm-catalog-index
description: >
  Index source videos into the VLM catalog via analyzer.py. Extracts frames,
  runs Qwen2-VL vision model to generate structured JSON (title, summary, tags,
  timeline, scenes). Use when adding new videos to the catalog or re-indexing
  existing ones with better settings (interval, model).
---

# VLM Catalog Indexer

Index videos into the catalog via VLM frame analysis.

## Quick Start

```bash
# Single video (default interval from config.py)
python -m clipmaker.analyzer video.mp4 --game assassins_creed_4

# With explicit interval
python -m clipmaker.analyzer video.mp4 --game assassins_creed_4 --interval 5

# Re-index (overwrite existing JSON)
python -m clipmaker.analyzer video.mp4 --game assassins_creed_4 --interval 5 --force

# Every capture in the game's video_dirs (already indexed files are skipped)
python -m clipmaker.analyzer --game assassins_creed_4 --interval 5
```

## Frame Interval

| Interval | Frames for 46 min | Quality | When to use |
|----------|-------------------|---------|-------------|
| 30s | ~92 | Low | Quick first pass, bulk indexing |
| 10s | ~275 | Medium | Standard indexing, sufficient for most videos |
| 5s | ~550 | High | B-roll pipeline, produces `scenes` with type/description |
| 4s | ~690 | Maximum | Detailed analysis, current default in config.py |

## Model Selection

| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~8 GB | Generic descriptions ("man in black shirt") | Fast |
| `Qwen/Qwen2.5-VL-7B-Instruct` | ~16 GB | Specific ("Amarr Laser Rifle", named locations) | Medium |

**Current model in config.py:** 7B (recommended).

To switch: edit `VLM_MODEL` in [config.py](clipmaker/config.py).

## Output JSON Structure

```json
{
  "title": "EVE Fanfest: Vanguard Keynote",
  "summary": "Presentation about EVE Vanguard...",
  "tags": ["eve_vanguard", "fanfest", "fps", "shooter"],
  "duration_sec": 2760,
  "timeline": [
    {"start": "00:00", "end": "00:05", "caption": "Speaker on stage", "tags": ["speaker"]}
  ],
  "scenes": [
    {"start": "00:00", "end": "00:45", "type": "speaker", "description": "CCP developer presenting..."}
  ],
  "ocr_text": ["EVE Vanguard", "Fanfest 2026"]
}
```

## Chunked Processing

When there are more frames than `VLM_CHUNK_SIZE` (12, sized for 16 GB VRAM),
the analyzer splits them into chunks of that size.

**Known issues:**
- A chunk that fails all three attempts (greedy, then two sampling temperatures)
  is listed in the entry as `failed_chunks`, and the batch ends with the list of
  such entries. `--dry-run` marks them; `--retry-incomplete` analyzes them again.
  A video where no chunk answered is reported as FAILED and not saved.
- Specific frame ranges can crash CUDA — fill manually from adjacent chunks
- The old chunk size of 50 degraded OCR on 275+ frames and left scenes empty;
  that is why the size is 12 now

**`max_new_tokens` formula:** `60 * n_frames`, but may be insufficient for 7B model
(more verbose responses). Cap defined in `VLM_MAX_NEW_TOKENS_MAX` (currently 2500).

## VRAM Management

- **Before running:** ensure VRAM is free (close games, unload other models)
- **After running:** model stays in VRAM. To unload — restart the Python process
- **RTX 5070 Ti (16 GB):** 7B model fits, but not with a game running

## Supported Games

Game list and catalog paths are in `config.py → GAMES`. Each game has:
- `video_dirs` — where to find source videos
- `catalog_subdir` — folder under `catalog/` for JSONs
- `subdir_overrides` — subfolder mappings (Fanfest, seasons, etc.)

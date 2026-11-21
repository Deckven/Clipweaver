---
name: catalog-search
description: >
  Search across the VLM video catalog (277+ indexed videos) for specific visual
  content by tags, scenes, OCR text, and descriptions. Use when looking for
  footage of a specific subject (weapon, location, character, UI element)
  across all indexed games.
---

# Catalog Search

Find specific visual content across all indexed videos.

## Catalog Structure

```
catalog/
  <game>_videos/              — one folder per key in GAMES (catalog_subdir)
    <subfolder>/              — mirrors the capture subfolders (subdir_overrides renames them)
  _index/                     — semantic index (embed_index build)
```

The bundled sample: `examples/sample/catalog/assassins_creed_4_videos/` (4 entries).

## Searchable Fields

Each catalog JSON contains:

| Field | Content | Example query |
|-------|---------|---------------|
| `tags[]` | Thematic tags | "amarr", "laser", "fleet battle" |
| `summary` | Video description | "Space fleet engagement near station" |
| `title` | VLM-generated title | "Elden Ring Radahn Boss Fight" |
| `timeline[].caption` | Per-frame description | "Player fights giant with sword" |
| `timeline[].tags[]` | Per-frame tags | "boss", "combat", "health_bar" |
| `scenes[].description` | Scene description (interval ≤10s only) | "Amarr Laser Rifle and Minmatar SMG on display" |
| `scenes[].type` | Scene type | gameplay/speaker/slide/trailer/concept_art/ui/video_clip |
| `ocr_text[]` | On-screen text | "EVE Vanguard", "Press F to interact" |

## Semantic Search (preferred)

Dense retrieval over the catalog. Returns the **moment**, not just the file, and
works across languages — a Russian query finds English scene descriptions.

```bash
python -m clipmaker.embed_index build              # once, ~9 s on GPU
python -m clipmaker.embed_index search "harbour menu with ship upgrades and their prices" --game assassins_creed_4
# any language works: "меню гавани, улучшения корабля" (Russian for the same) finds the same clip
python -m clipmaker.embed_index stats
```

```python
from clipmaker import embed_index

for hit in embed_index.search("diving underwater past a reef", game="assassins_creed_4", k=5):
    print(hit.score, hit.timecode, hit.original_name, hit.text)
```

**Write queries as descriptions of the picture, not of the topic.**
"ship firing broadsides in a storm" works; "the economy of piracy" does not —
the catalog describes what is on screen. To go from narration to such a query,
use `shot_query.build_shot_queries(fragment, game=...)` (needs Ollama).

Rejection is by z-score (`EMBED_MIN_ZSCORE = 3.0`), not raw cosine — cosine is
compressed and passes nonsense. Zero results is a real answer: no footage matches.

Rebuild after indexing new videos — incremental, only changed files are re-embedded:
```bash
python -m clipmaker.embed_index build --game assassins_creed_4
```

## Token-Overlap Search (fallback)

Still used when no index exists. Matches only words shared verbatim, so it finds
the right video by proper noun and never identifies a moment.

```python
from clipmaker.report import _load_catalog, search_catalog

catalog = _load_catalog("assassins_creed_4")
results = search_catalog("harbourmaster upgrade", catalog, max_results=5)
```

## Manual Search (grep)

```bash
# Find videos with a specific tag
grep -rl "harbormaster" catalog/assassins_creed_4_videos/ --include="*.json"

# Find OCR text
grep -rl "HARBOURMASTER" catalog/ --include="*.json"

# Find scenes of a specific type
grep -l "\"ui\"" catalog/assassins_creed_4_videos/ --include="*.json"
```

## Tips

- **Scenes vs Timeline:** `scenes` give semantic descriptions (type + what's shown),
  `timeline` gives per-frame OCR + tags. For visual content search, prefer `scenes`.
- **Missing scenes:** if a video was indexed at interval >10s or before the scenes feature,
  re-index with `/vlm-catalog-index` using `--interval 5 --force`.
- **Cross-game search:** for universal footage (space, medieval settings),
  search across all catalog directories.

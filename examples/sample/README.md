# Sample catalog

Four catalog entries from the author's own Assassin's Creed IV Black Flag
(Resynced) gameplay captures, exactly as the VLM indexer wrote them: title,
summary, tags, a caption with the on-screen text every 5 seconds, and scene
descriptions with start and end times. In-game dialogue and item flavour text
were removed from the on-screen text; menus, objectives and names are as indexed.
No video is included — `source_path`
points into `footage/raw/`, where the capture would sit on your machine.

| Clip | Length | On screen |
|---|---|---|
| Battle Against HMS Ipswich 1 | 8:44 | naval battle in a storm, Maynard's armada |
| Anotto Bay Exploration 1 | 14:59 | exploring the bay, diving, the smugglers' den |
| Ship Upgrade Trading 1 | 3:40 | harbourmaster menu: hull armor, weapons, costs |
| Harpooning Bull Shark 1 | 2:59 | crafting menu, harpooning a bull shark |

`narration.md` is a short synthetic narration for these clips, one paragraph
per clip. `shots.json` holds, for each paragraph, a description of the picture
and the clip that should come back first. The two exist side by side because a
narration paragraph says what the video is *about*, while the catalog says what
is *on screen*: embedded directly, paragraph 1 (a sea battle) comes back as the
upgrade menu; described as a picture, all four land on the right clip. In the full pipeline a local LLM writes those picture
descriptions (`clipmaker.shot_query`).

Try it from the repository root — no GPU and no video needed. Install with
`pip install -e ".[search]"` after a CPU torch (see the
[guide](../../GUIDE.md#4-ten-minute-tour-on-the-sample-no-gpu)), then:

```bash
export CLIPMAKER_CATALOG_DIR=examples/sample/catalog   # PowerShell: $env:CLIPMAKER_CATALOG_DIR="examples/sample/catalog"

# Exact on-screen text (OCR): which moments show the harbourmaster menu
python -m clipmaker.ocrgrep "harbou?rmaster" --game assassins_creed_4

# Semantic search: describe the picture, get the moment.
# Downloads the multilingual E5 model once (~1 GB); CPU is fine for four clips.
python -m clipmaker.embed_index build --game assassins_creed_4
python -m clipmaker.embed_index search "sailing ship firing cannons at enemy ships in a stormy sea" --game assassins_creed_4 --min-z 0
```

`--min-z 0` switches off the outlier cutoff, which needs a full-size catalog to
mean anything (see `EMBED_MIN_ZSCORE` in `clipmaker/config.py`).

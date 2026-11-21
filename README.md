# Clipmaker

A pre-production assistant for game video essays: it indexes your own gameplay
footage with a local vision-language model, finds the moments your narration
needs, and hands you a storyboard and a timeline layout. The final edit stays
in your hands.

**New here?** The [user guide](GUIDE.md) explains every module and walks the
full path from your captures and a narration to a Vegas timeline, step by step.

## What it does — and what it does not

**Does**

- **Catalog.** Watches each video with Qwen2.5-VL-7B (4-bit, local GPU) and
  writes a JSON entry: title, summary, tags, a caption with the on-screen text
  every few seconds (EasyOCR), and scene descriptions with start and end times.
- **Search.** Finds *moments*, not files: semantic search over the catalog
  (multilingual E5, so a Russian query finds English descriptions) plus regex
  over the on-screen text for exact names, tickers and numbers.
- **Storyboard.** Splits the narration into fragments, turns each into
  descriptions of the picture (local LLM via Ollama), and picks a clip and a
  range for every fragment. A contact sheet shows the picks before anything
  touches the editor.
- **Timeline.** Aligns fragments to the recorded voice-over (faster-whisper),
  writes a frame-quantized `layout.txt`, and a Vegas Pro script places the
  clips relative to the narration.

**Does not**

- Edit the video. Pacing, cuts, music and the last 20% of picks are yours.
- Work with footage you do not have. It searches your captures; it does not
  download or generate b-roll.
- Run without a GPU for indexing. Search, storyboard and layout run on CPU.

## How well it works

Measured on the author's catalog (over 1 000 indexed videos in ten game catalogs)
against clips actually placed in finished videos. A hit means the right moment
(±5 s) is in the top 10 results of at least one search layer.

| Set | Queries | Recall@10, moment |
|---|---|---|
| Storyboard with picture queries written before searching | 40 | 70% |
| All three finished videos, incl. labels written after the edit | 266 | 35% |

Only 44 of the 346 query–clip pairs are confirmed by hand so far; treat the
numbers as a floor with wide error bars.

Caption accuracy on a hand-labelled sample, indexing time per hour of footage
and peak VRAM per stage are being measured; they will be added here.

Hardware the numbers come from: one RTX 5070 Ti (16 GB). The 7B model in NF4
takes about 5 GB; a call over 10 frames peaks at 9.8 GB, 20 frames at 14.1 GB,
so the default batch is 12 frames.

## Try it without a GPU or video

`examples/sample/` holds four catalog entries from the author's own
Assassin's Creed IV captures — no video, just what the indexer wrote.

```bash
pip install -e .
export CLIPMAKER_CATALOG_DIR=examples/sample/catalog   # PowerShell: $env:CLIPMAKER_CATALOG_DIR="examples/sample/catalog"
python -m clipmaker.ocrgrep "harbou?rmaster" --game assassins_creed_4
```

Semantic search over the same four clips is in
[examples/sample/README.md](examples/sample/README.md).

## Install

**Requires:** Python 3.11+, ffmpeg and ffprobe on `PATH`. For indexing, a CUDA
GPU with 12 GB or more (tested on 16 GB). Optional: [Ollama](https://ollama.com) for picture queries
and structured transcripts; Vegas Pro 22 for the timeline loader.

```bash
# torch first, from the index that matches your CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all,test]"
cp .env.example .env    # then set your footage folders
pytest -q
```

Extras: `vlm` (indexing), `ocr`, `search` (E5 embeddings), `transcribe`
(Whisper), `youtube` (yt-dlp), `all`.

## Configure

Machine-specific settings live in `.env` (see `.env.example`): where your raw
captures and editor projects are, and optional stock-footage API keys. Games
and their capture folders are listed in `GAMES` in `clipmaker/config.py` —
add yours there.

## Pipeline

```
your captures ──► analyzer (VLM + OCR) ──► catalog/<game>_videos/*.json
                                                │
narration ──► shot_query (Ollama) ──► embed_index / ocrgrep ──► report (storyboard)
                                                │
voice-over ──► transcribe ──► align ──► build_storyboard ──► contact_sheet (check)
                                                │
                                   layout ──► layout.txt ──► vegas_scripts/import_layout.cs
```

Each step is a module: `python -m clipmaker.<module> --help`. Step by step, with
the files each step reads and writes: [GUIDE.md](GUIDE.md).

## Claude Code skills

`.claude/skills/` holds the instructions the author's Claude Code sessions use
for each step: indexing, catalog search, storyboard, b-roll pipeline,
production plan, shooting list, transcription. They describe the working
order and the rules learned the hard way; read them as documentation even if
you do not use Claude Code.

## Structure

```
clipmaker/          # the pipeline, one module per step
tests/              # pytest; the Vegas loader test runs only where Vegas is installed
vegas_scripts/      # import_layout.cs — the one generic Vegas Pro loader
examples/sample/    # four sanitized catalog entries + a synthetic narration
.claude/skills/     # Claude Code instructions per step
```

## License

MIT — see [LICENSE](LICENSE).

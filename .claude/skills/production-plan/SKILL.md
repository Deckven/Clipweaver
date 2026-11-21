---
name: production-plan
description: >
  Generate a production plan from narration text: classify fragments by shot type,
  search local catalog + YouTube + stock footage, produce gap analysis and shooting
  priorities. Use when starting a new video project and need to understand what
  footage is available and what needs to be recorded.
---

# Production Plan

Analyze narration text → classify shots → search footage → gap analysis.

## Quick Start

```bash
# Generate plan for a project
python -m clipmaker.production_plan <project-name> [game]

# Example
python -m clipmaker.production_plan diablo4-lord-of-hatred diablo4

# List available projects
python -m clipmaker.production_plan
```

## Project Setup

Before generating a plan, create the project folder:

```
projects/<project-name>/
  text.md     ← narration text (required)
```

Text may contain YAML frontmatter — it will be stripped automatically.

## What the Plan Does

For each text fragment:

1. **Classification** — determines shot type:
   - `gameplay` — active gameplay, fights, farming
   - `cinematic` — trailers, cutscenes, intros
   - `broll_abstract` — atmospheric shots, transitions
   - `broll_gaming` — generic gaming context
   - `talking_head` — narrator on camera
   - `screen_recording` — UI, menus, store pages, reviews
   - `comparison` — side-by-side before/after
   - `text_overlay` — numbers, dates, facts on screen

2. **Footage search** across three sources:
   - Local catalog (VLM-indexed videos)
   - YouTube (via yt-dlp)
   - Stock APIs (Pexels, Pixabay — requires API keys in config.py)

3. **Recommendation** — best source + confidence level (high/medium/low)

4. **Gap analysis** — fragments with low confidence

5. **Shooting list** — what needs to be recorded manually

## Output Files

```
projects/<project-name>/
  plan.json   ← full plan (machine-readable)
  plan.md     ← report (human-readable)
```

### plan.md Structure

```markdown
# Production Plan: <project>

## Summary
| Covered by local catalog | 45% |
| Covered by YouTube       | 20% |
| Needs shooting           | 35% |

## Fragment Breakdown
### [+] Fragment 1 — local (high)
### [~] Fragment 5 — youtube (medium)
### [!] Fragment 8 — shoot (low)

## Gap Analysis
| # | Fragment | Priority | Issue |

## Shooting List
Total footage to record: 180s (3.0 min)
```

## Configuration (config.py)

- `WORDS_PER_MIN_RU = 150` — speech rate for duration estimation
- `PLAN_LOCAL_RESULTS = 3` — catalog results per fragment
- `PLAN_YT_RESULTS = 3` — YouTube results per fragment
- `PLAN_STOCK_RESULTS = 3` — stock results per fragment
- `PEXELS_API_KEY` / `PIXABAY_API_KEY` — stock API keys (None = skip)

## Module Dependencies

```
production_plan.py
  ├── shot_classifier.py    (fragment classification, RU→EN translation)
  ├── youtube_search.py     (YouTube search via yt-dlp)
  ├── stock_search.py       (Pexels + Pixabay API)
  └── report.py             (catalog search)
```

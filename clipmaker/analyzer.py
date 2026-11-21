"""
Stage 1 — VLM video analyzer.

Takes a video file, samples frames, runs Qwen2-VL to produce a structured
JSON description (title, summary, tags, timeline) and writes it to the
per-game catalog folder. Optionally suggests a new filename.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from . import config
from .game_vocab import vocab_for, render_vocab_hint, match_entities, match_region


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# Words that should never end up in tags or filenames (too generic)
_NOISE_TOKENS = {
    "elden", "ring", "er", "gameplay", "video", "footage", "clip",
    "game", "play", "playing", "scene", "moment", "the", "and", "with",
    "of", "in", "on", "to", "a", "an", "for", "from",
    "snake_case", "snake", "case", "lowercase", "specific", "tags",
    "t1", "t2", "tag", "example", "title",
}

# Tags matching these regex patterns are stripped from VLM output — they come
# from OCR chrome (corp ids, local counts, etc.) and are noise, not entities.
_NOISE_TAG_PATTERNS = [
    re.compile(p) for p in (
        r"^corp(?:oration)?_?\d+$",
        r"^local_?\d+$",
        r"^\d+$",
        r"^channel_\d+$",
    )
]


def _is_noise_tag(tag: str) -> bool:
    t = str(tag).lower()
    if t in _NOISE_TOKENS:
        return True
    return any(p.match(t) for p in _NOISE_TAG_PATTERNS)


# ---------- ffmpeg helpers ----------

def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_frames(video_path: Path, output_dir: Path, interval: int, max_size: int) -> list[tuple[Path, int]]:
    """Extract frames at given interval. Returns [(frame_path, timestamp_sec), ...]."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("frame_*.jpg"):
        old.unlink()

    pattern = output_dir / "frame_%06d.jpg"
    vf = (
        f"fps=1/{interval},"
        f"scale='if(gt(iw,ih),{max_size},-2)':'if(gt(ih,iw),{max_size},-2)'"
    )
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", vf,
        "-q:v", "3",
        "-y",
        str(pattern),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return [(f, idx * interval) for idx, f in enumerate(frames)]


# ---------- VLM ----------

_vlm_cache: dict = {}


def load_vlm(device: str = "cuda"):
    if "model" in _vlm_cache:
        return _vlm_cache["model"], _vlm_cache["processor"]

    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_kwargs: dict = {"device_map": device}

    if config.VLM_QUANTIZE_4BIT:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading {config.VLM_MODEL} (bnb-4bit NF4)...")
    else:
        dtype = torch.float16 if config.VLM_DTYPE == "float16" else torch.bfloat16
        load_kwargs["dtype"] = dtype
        print(f"Loading {config.VLM_MODEL} ({dtype})...")

    model = AutoModelForImageTextToText.from_pretrained(
        config.VLM_MODEL, **load_kwargs,
    ).eval()
    processor = AutoProcessor.from_pretrained(config.VLM_MODEL)
    _vlm_cache["model"] = model
    _vlm_cache["processor"] = processor
    return model, processor


SYSTEM_PROMPT = (
    "You analyze gameplay video footage and produce a structured JSON "
    "description in English. Be concrete and specific: name visible game "
    "locations, bosses, enemies, weapons, spells, and items by their "
    "proper names when you can identify them. Never repeat the same "
    "caption across segments — describe what is actually different."
)


def _fmt_mmss(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"


def extract_path_hints(video_path: Path, game: str) -> dict:
    """Derive authoritative location/entity labels from the filesystem path.

    - location: parent folder name (under the game's root) matched against
      the region vocab. Falls back to the raw folder name.
    - entities: bosses / enemies / concepts mentioned in the filename.
    """
    vocab = vocab_for(game) or {}
    hints = {"location": None, "location_folder": None, "entities": []}

    # Find the closest game root from config; use first parent under it.
    roots = config.GAMES.get(game, {}).get("video_dirs", [])
    parent_folder = None
    for root in roots:
        try:
            rel = video_path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) > 1:
            parent_folder = parts[0]
            break

    if parent_folder is None and video_path.parent != video_path:
        parent_folder = video_path.parent.name

    if parent_folder:
        hints["location_folder"] = parent_folder
        matched = match_region(parent_folder, vocab)
        hints["location"] = matched or parent_folder

    # Match entities in filename against vocab categories that carry
    # recognisable proper nouns.
    filename_text = video_path.stem.replace("_", " ")
    entity_items = []
    for cat in ("bosses", "enemies", "regions", "ships", "factions", "events", "landmarks"):
        entity_items.extend(vocab.get(cat, []))
    hints["entities"] = match_entities(filename_text, entity_items)
    hints["game_description"] = vocab.get("description", "")

    return hints


def _build_user_prompt(n_frames: int, duration_sec: int, interval: int,
                       game_hint: str, vocab_hint: str, name_hint: str,
                       path_hints: dict, ocr_lines: list[str] | None = None) -> str:
    end_ts = _fmt_mmss(duration_sec)

    parts = [
        f"You are given {n_frames} frames sampled from a {duration_sec}-second "
        f"gameplay video (total duration {end_ts}).",
        "",
        f"Game: {game_hint}",
    ]

    game_desc = (path_hints or {}).get("game_description") or ""
    if game_desc:
        parts.append("")
        parts.append(f"Game description (ground the visuals in this — do not "
                     f"default to a different genre):")
        parts.append(game_desc)

    fact_lines = []
    location = path_hints.get("location")
    if location:
        fact_lines.append(f'Region: {location}  <-- this IS the location.')
    entities = path_hints.get("entities") or []
    if entities:
        fact_lines.append(
            "Confirmed entities in this video (from file naming): "
            + ", ".join(entities)
            + ". Use their exact names."
        )
    if fact_lines:
        parts.append("")
        parts.append("AUTHORITATIVE FACTS:")
        parts.extend(f"- {line}" for line in fact_lines)

    if ocr_lines:
        parts.append("")
        parts.append(
            "Text captured by OCR across these frames (star systems, ships, "
            "stations, events, status messages — use these verbatim when you "
            "name things in your output, do NOT invent similar-looking names):"
        )
        parts.extend(f"- {line}" for line in ocr_lines)

    if name_hint:
        parts.append(f'Original filename: "{name_hint}"')
    if vocab_hint:
        parts.append(
            "Known entities for this game (USE these exact spellings when "
            "they match):\n" + vocab_hint
        )
    parts.extend([
        "",
        "Produce ONE JSON object with exactly these fields:",
        "  title   : short specific title naming the main activity/place/event",
        "            (e.g. 'Gallente President Election route through Postouvin').",
        "  summary : 1-3 sentences describing what happens overall. Name specific",
        "            systems, ships, stations, events that you can confirm from",
        "            the OCR list or the vocab. Do NOT invent names.",
        "  tags    : 6-12 snake_case tags covering concrete entities and",
        "            activities (e.g. postouvin, warp_drive, gallente_shuttle,",
        "            roden_for_president_yc128, memorial_research_site).",
        "",
        "  scenes : array of objects, each with:",
        "            start : MM:SS timestamp where this scene begins",
        "            end   : MM:SS timestamp where this scene ends",
        '            type  : one of "gameplay", "speaker", "slide", "trailer",',
        '                    "concept_art", "ui", "video_clip"',
        "            description : 1 sentence describing what is VISUALLY shown",
        "                          (not what is being said). Name specific things",
        "                          you see: weapons, characters, environments, UI elements.",
        "          Group consecutive frames showing the same scene into one entry.",
        "          Aim for 4-10 scene entries per response.",
        "",
        "RULES:",
        "- Only use names that appear in the OCR list, the vocab above, or the",
        "  authoritative facts. If unsure, describe generically rather than guess.",
        f"- Do NOT include '{game_hint.replace(' ', '_')}' or 'gameplay' in tags.",
        "- Output JSON only, starting with { and ending with }. No markdown, no prose.",
    ])
    return "\n".join(parts)


def build_messages(frame_paths: list[Path], duration_sec: int, interval: int,
                   game_hint: str, vocab_hint: str, name_hint: str,
                   path_hints: dict, ocr_lines: list[str] | None = None):
    content: list[dict] = [
        {
            "type": "image",
            "image": f"file://{p.as_posix()}",
            "max_pixels": config.VLM_FRAME_MAX_PIXELS,
        }
        for p in frame_paths
    ]
    content.append({
        "type": "text",
        "text": _build_user_prompt(len(frame_paths), duration_sec, interval,
                                   game_hint, vocab_hint, name_hint, path_hints,
                                   ocr_lines),
    })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _is_degenerate(text: str) -> bool:
    """Detect VLM output that is pure repetition of a single character."""
    stripped = text.strip()
    if not stripped:
        return True
    return len(set(stripped)) <= 2


def run_vlm(frame_paths: list[Path], duration_sec: int, interval: int,
            game_hint: str, vocab_hint: str, name_hint: str,
            path_hints: dict, ocr_lines: list[str] | None = None,
            device: str = "cuda",
            do_sample: bool = False,
            temperature: float = 1.0,
            repetition_penalty: float = 1.0) -> str:
    from qwen_vl_utils import process_vision_info

    model, processor = load_vlm(device)
    messages = build_messages(frame_paths, duration_sec, interval,
                              game_hint, vocab_hint, name_hint, path_hints,
                              ocr_lines)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    max_new = max(
        config.VLM_MAX_NEW_TOKENS_MIN,
        min(
            config.VLM_MAX_NEW_TOKENS_MAX,
            config.VLM_MAX_NEW_TOKENS_ENVELOPE
            + config.VLM_MAX_NEW_TOKENS_PER_FRAME * len(frame_paths),
        ),
    )
    gen_kwargs: dict = {"max_new_tokens": max_new}
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    print(f"  VLM input tokens: {inputs.input_ids.shape[1]}  |  max_new: {max_new}"
          f"  |  sample={do_sample}  rep_pen={repetition_penalty}")
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ---------- Output parsing & filenames ----------

def _repair_truncated_json(snippet: str) -> str:
    """If generation was cut mid-array, trim back to the last complete `}`
    inside the `timeline` array and close the structure."""
    # Remove any trailing partial line inside timeline
    # Find the `"timeline": [` opener
    tl_match = re.search(r'"timeline"\s*:\s*\[', snippet)
    if not tl_match:
        return snippet
    prefix = snippet[: tl_match.end()]
    rest = snippet[tl_match.end():]

    # Collect complete top-level {...} objects within the timeline array
    items, depth, start = [], 0, None
    for i, ch in enumerate(rest):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                items.append(rest[start : i + 1])
                start = None

    if not items:
        return snippet
    return prefix + ", ".join(items) + "]}"


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] (invalid JSON but common VLM output)."""
    return re.sub(r",\s*([}\]])", r"\1", s)


def _normalize_keys(d: dict) -> dict:
    """Map common VLM key mistakes to expected field names."""
    key_map = {"description": "summary", "strings": "tags"}
    return {key_map.get(k, k): v for k, v in d.items()}


def parse_json_response(raw: str) -> dict:
    """Extract the first JSON object from VLM output, tolerating truncation."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in VLM output:\n{raw[:500]}")
    end = raw.rfind("}")
    candidate = raw[start : end + 1] if end != -1 else raw[start:]

    for repair_fn in (lambda s: s, _strip_trailing_commas, _repair_truncated_json):
        try:
            result = json.loads(repair_fn(candidate))
            return _normalize_keys(result)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"Could not parse JSON from VLM output:\n{candidate[:500]}")


def _parse_mmss(s: str) -> int | None:
    m = re.match(r"^\s*(\d+):(\d{1,2})\s*$", str(s))
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", str(s))
    if m:
        return int(m.group(1))
    return None


def _match_vocab_in_line(line: str, vocab_items: list[str]) -> list[str]:
    """Return canonical vocab items that appear in the OCR line (substring,
    case-insensitive). Longer items take precedence."""
    if not line or not vocab_items:
        return []
    norm_line = line.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for item in sorted(vocab_items, key=len, reverse=True):
        key = item.lower()
        if key in seen or len(key) < 3:
            continue
        if key in norm_line:
            hits.append(item)
            seen.add(key)
    return hits


def build_ocr_timeline(per_frame_ocr: list[list[str]], interval: int,
                       duration_sec: int, vocab: dict | None) -> list[dict]:
    """Build a deterministic per-frame timeline from OCR text.

    Each frame at timestamp i*interval produces one segment. Adjacent frames
    with identical caption are merged. Captions go through chrome filtering
    and phrase collapsing so UI clutter (Route/General/Scan Results/etc.) is
    dropped in favour of signal (star systems, POIs, warp status)."""
    from .ocr import filter_chrome

    vocab_items: list[str] = []
    if vocab:
        for cat, items in vocab.items():
            if isinstance(items, list):
                vocab_items.extend(items)

    segments: list[dict] = []
    for i, lines in enumerate(per_frame_ocr):
        start = i * interval
        end = min((i + 1) * interval, duration_sec)
        if end <= start:
            continue

        cleaned = filter_chrome(lines) if lines else []
        caption = "; ".join(cleaned)

        seg_tags: list[str] = []
        seen_tags: set[str] = set()
        for line in cleaned:
            for hit in _match_vocab_in_line(line, vocab_items):
                tag = _to_tag(hit)
                if tag and tag not in seen_tags:
                    seg_tags.append(tag)
                    seen_tags.add(tag)

        if segments and segments[-1]["caption"] == caption:
            segments[-1]["end"] = _fmt_mmss(end)
            continue

        segments.append({
            "start": _fmt_mmss(start),
            "end": _fmt_mmss(end),
            "caption": caption,
            "tags": seg_tags,
        })
    return segments


def clean_analysis(analysis: dict, duration_sec: int) -> dict:
    """Post-filter: drop segments past video duration, normalize timestamps,
    deduplicate identical consecutive captions, strip noise tags."""
    tl = analysis.get("timeline") or []
    cleaned = []
    last_caption = None
    for seg in tl:
        start = _parse_mmss(seg.get("start", ""))
        end = _parse_mmss(seg.get("end", ""))
        if start is None or end is None:
            continue
        if start >= duration_sec:
            break
        end = min(end, duration_sec)
        if end <= start:
            continue
        caption = str(seg.get("caption", "")).strip()
        if caption and caption == last_caption:
            cleaned[-1]["end"] = _fmt_mmss(end)
            continue
        seg_tags = [t for t in seg.get("tags", [])
                    if t and not _is_noise_tag(t)]
        cleaned.append({
            "start": _fmt_mmss(start),
            "end": _fmt_mmss(end),
            "caption": caption,
            "tags": seg_tags,
        })
        last_caption = caption

    analysis["timeline"] = cleaned

    # Clean top-level tags
    analysis["tags"] = [
        t for t in analysis.get("tags", [])
        if t and not _is_noise_tag(t)
    ]
    return analysis


def _to_tag(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def inject_path_facts(analysis: dict, path_hints: dict) -> dict:
    """Merge location + confirmed entities into tags (both top-level and
    per-segment), so path-derived ground truth cannot be lost even if the
    VLM hallucinated or ignored the hint."""
    forced: list[str] = []
    if path_hints.get("location"):
        forced.append(_to_tag(path_hints["location"]))
    for ent in path_hints.get("entities") or []:
        forced.append(_to_tag(ent))

    tags = list(analysis.get("tags") or [])
    for t in forced:
        if t and t not in tags:
            tags.insert(0, t)
    analysis["tags"] = tags

    for seg in analysis.get("timeline") or []:
        seg_tags = list(seg.get("tags") or [])
        for t in forced:
            if t and t not in seg_tags:
                seg_tags.append(t)
        seg["tags"] = seg_tags

    # Persist only the path-derived facts, not the static game description.
    analysis["path_hints"] = {
        k: v for k, v in path_hints.items() if k != "game_description"
    }
    return analysis


_SAFE = re.compile(r"[^a-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return [t for t in _SAFE.sub("_", s.lower()).split("_") if t and len(t) > 1]


def _dedup_keep_order(items):
    seen = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            yield x


# System/entity extraction patterns for building unique filenames.
# Codes: F67E-0, 6E-578, JH-MZW, MZ-CFI, Y9g-KS (2-5 chars, optional digit+letter mix, dash, 2-5 more)
_SYSTEM_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{0,3}-[A-Z0-9]{2,5}\b")
# Named system + sec status: "Postouvin 0.7", "Niballe 0,6", "Larryn 8.6"
_SYSTEM_WITH_SEC_RE = re.compile(r"\b([A-Z][a-z]{3,12})\s+[0-9][,. ][0-9]")
# Timestamp portion of original Eve filename: 2026.04.11 ... 12.59.23.08
_FILE_TS_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2}).*?(\d{2})\.(\d{2})\.(\d{2})")

# Location folders are usually long and have filler words we can drop.
_LOCATION_FILLER = {"the", "a", "for", "and", "president"}


def _extract_file_timestamp(stem: str) -> str:
    m = _FILE_TS_RE.search(stem)
    if not m:
        return ""
    y, mo, d, h, mi, s = m.groups()
    return f"{y}{mo}{d}_{h}{mi}{s}"


def _primary_systems(ocr_text: list[str], limit: int = 2) -> list[str]:
    """Return the most-frequent star system references across OCR text."""
    from collections import Counter
    codes: list[str] = []
    names: list[str] = []
    for line in ocr_text:
        codes.extend(_SYSTEM_CODE_RE.findall(line))
        names.extend(_SYSTEM_WITH_SEC_RE.findall(line))
    freq = Counter(codes + names)
    return [s for s, _ in freq.most_common(limit)]


def _primary_activity_tags(timeline: list[dict], exclude: set[str], limit: int = 2) -> list[str]:
    """Most-frequent vocab tag across the timeline, excluding given tags
    (e.g. the location tag, already in the filename)."""
    from collections import Counter
    c: Counter = Counter()
    for seg in timeline or []:
        for t in seg.get("tags") or []:
            if t and t not in exclude and not _is_noise_tag(t):
                c[t] += 1
    return [t for t, _ in c.most_common(limit)]


def _game_display_name(game: str) -> str:
    cfg = config.GAMES.get(game, {})
    return cfg.get("display_name") or (game.replace("_", " ").title() if game else "")


def _titleize_vocab_tag(tag: str) -> str:
    """Turn a vocab-derived tag (e.g. 'warp', 'memorial_research_site',
    'gallente_shuttle') into a human-readable Title Case string."""
    return tag.replace("_", " ").title()


def _norm_word(w: str) -> str:
    """Fold a word for duplicate detection: lowercase, curly quotes -> straight,
    surrounding punctuation dropped. Without this, the folder's 'Assassin’s'
    never matches the display name's "Assassin's" and the game name lands in
    the filename twice. '_' is ShadowPlay's stand-in for a colon in the folder
    name ('Onimusha_ Way of the Sword')."""
    return w.replace("’", "'").strip(".,:;!?()[]\"_").lower()


# A title is one sentence; only the leading words carry the discriminating
# content ("Cape Bonavista Naval Engagement" -> all 4 kept).
_TITLE_MAX_WORDS = 6


def _content_tokens(analysis: dict, skip_words: set[str] | None = None) -> list[str]:
    """Human-readable Title-case tokens describing what's in the video:
    shortened location + title words + up to 2 primary star systems + up to 2
    activities (mission / warp / mining / combat).

    The title matters most: location is per-folder, so for a game whose footage
    all sits in one directory it is identical across every video and the names
    collapse into each other."""
    out: list[str] = []
    seen_lower: set[str] = {_norm_word(w) for w in (skip_words or ())}

    def add(tok: str) -> None:
        t = tok.strip()
        if not t:
            return
        n = _norm_word(t)
        if not n or n in seen_lower:
            return
        seen_lower.add(n)
        out.append(t)

    ph = analysis.get("path_hints") or {}
    location = ph.get("location") or ""
    for part in location.split():
        if _norm_word(part) in _LOCATION_FILLER:
            continue
        w = part.strip()
        add(w if w.isupper() and len(w) >= 2 else w.capitalize())

    kept = 0
    for part in str(analysis.get("title") or "").split():
        if kept >= _TITLE_MAX_WORDS:
            break
        n = _norm_word(part)
        if not n or n in _LOCATION_FILLER or _is_noise_tag(n):
            continue
        before = len(out)
        add(part if part.isupper() and len(part) >= 2 else part.capitalize())
        if len(out) > before:
            kept += 1

    for s in _primary_systems(analysis.get("ocr_text") or [], limit=2):
        add(s.strip())

    loc_tag_set = {_to_tag(location)} | {_to_tag(t) for t in out}
    for tag in _primary_activity_tags(analysis.get("timeline") or [], loc_tag_set, limit=2):
        add(_titleize_vocab_tag(tag))

    return out


# Characters Windows forbids in a filename. display_name legitimately contains
# a colon ("Assassin's Creed IV: Black Flag"), so the base MUST be scrubbed or
# the rename fails with OSError on every file.
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename_base(base: str) -> str:
    cleaned = _ILLEGAL_FILENAME_CHARS.sub(" ", base)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Windows also rejects a trailing dot or space.
    return cleaned.rstrip(". ")


def suggest_filename(analysis: dict, original: Path) -> str:
    """Base filename in the form "<Game display> <content tokens>.ext".

    The sequence number (" 1", " 2", …) used to disambiguate duplicates is
    appended at rename time, so the stored value here is the stable base."""
    game = analysis.get("game") or ""
    game_display = _game_display_name(game)
    skip = set(game_display.split()) if game_display else set()
    content = _content_tokens(analysis, skip_words=skip)

    parts = []
    if game_display:
        parts.append(game_display)
    parts.extend(content)

    base = _safe_filename_base(" ".join(p for p in parts if p))
    if not base:
        base = "Video"
    return f"{base[:120].rstrip('. ')}{original.suffix.lower()}"


# ---------- Per-video pipeline ----------

# Per-chunk decoding attempts. Greedy first — it is the fastest and parses most
# of the time — then progressively hotter sampling. A retry MUST change the
# decoding params: greedy is deterministic, so re-running it reproduces the same
# unparseable output verbatim.
_CHUNK_RETRY_PROFILES: tuple[dict, ...] = (
    {"do_sample": False},
    {"do_sample": True, "temperature": 0.2},
    {"do_sample": True, "temperature": 0.5, "ocr_limit": 30},
)


def _parse_chunk_ts(ts: str) -> int:
    """Seconds from a chunk-relative timestamp, for "SS", "MM:SS" or "HH:MM:SS".

    Qwen emits HH:MM:SS. The previous implementation was `parts[0] * 60 +
    parts[-1]`, which silently DROPPED the minutes field on 3-part input:
    "00:04:10" parsed as 10 s instead of 250 s. Every scene in a chunk
    therefore collapsed into a 0-59 s band, producing end < start whenever a
    scene crossed a minute boundary."""
    parts = [int(p) for p in ts.strip().split(".")[0].split(":")]
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec


def clamp_scenes(scenes: list[dict], duration_sec: int) -> list[dict]:
    """Drop scenes starting past the video and clamp the rest to its duration.

    The chunked path already bounds each scene to its own chunk, but a video
    short enough for a single VLM call skips that path entirely. Qwen then
    sometimes answers with a uniform synthetic grid ("every 12 s") that runs
    well past the real end - a 77 s clip came back with scenes up to 02:00.
    Those timestamps reach the search index, so a hit would hand Vegas a range
    the file does not contain.
    """
    clamped = []
    for scene in scenes:
        try:
            start = _parse_chunk_ts(scene.get("start", "00:00"))
            end = _parse_chunk_ts(scene.get("end", "00:00"))
        except (ValueError, IndexError):
            continue
        if start >= duration_sec:
            continue
        end = min(max(end, start), duration_sec)
        if end <= start:
            continue
        clamped.append({**scene, "start": _fmt_mmss(start), "end": _fmt_mmss(end)})
    return clamped


# Han, kana and CJK compatibility blocks. Qwen is bilingual and occasionally
# answers a chunk in Chinese despite SYSTEM_PROMPT asking for English.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _chunk_has_cjk(chunk: dict) -> bool:
    """True if any user-visible field of a chunk result is not English.

    Chunk fields are concatenated across the whole video, so a single Chinese
    chunk contaminates the entire record — and a title like "Exploring奖励系统"
    also reaches suggested_filename.
    """
    fields = [chunk.get("title", ""), chunk.get("summary", "")]
    fields += [str(t) for t in chunk.get("tags") or []]
    fields += [s.get("description", "") for s in chunk.get("scenes") or []]
    return any(_has_cjk(f) for f in fields)


def _pick_title(candidates: list[dict]) -> str:
    """Title of the whole video, chosen by gameplay weight rather than order.

    The rule used to be "take chunk 1". A clip that opens on a menu or a
    loading screen was therefore named after its interface: a 15-minute
    Kingston mission came back as "Menu Screen Transition". Ties fall back to
    the earliest chunk, so a video that is uniform throughout keeps its
    opening title.
    """
    offered = [c for c in candidates if (c.get("title") or "").strip()]
    if not offered:
        return ""
    clean = [c for c in offered if not _has_cjk(c["title"])] or offered
    return max(clean, key=lambda c: (c.get("gameplay", 0), -c.get("index", 0)))["title"]


def _gameplay_scene_count(chunk: dict) -> int:
    """Scenes in a chunk that show the game rather than its interface."""
    return sum(1 for s in chunk.get("scenes") or []
               if (s.get("type") or "").lower() != "ui")


def _run_vlm_chunked(
    frame_paths: list[Path],
    duration_sec: int,
    interval: int,
    game_hint: str,
    vocab_hint: str,
    name_hint: str,
    path_hints: dict,
    ocr_lines: list[str] | None,
    device: str,
    chunk_size: int,
) -> dict:
    """Process frames in chunks of chunk_size, merge VLM results."""
    all_tags: list[str] = []
    all_summaries: list[str] = []
    all_scenes: list[dict] = []
    title_candidates: list[dict] = []
    n_chunks = -(-len(frame_paths) // chunk_size)

    for ci in range(0, len(frame_paths), chunk_size):
        if ci > 0:
            torch.cuda.empty_cache()
        chunk_frames = frame_paths[ci:ci + chunk_size]
        chunk_idx = ci // chunk_size + 1
        chunk_start_sec = ci * interval
        chunk_end_sec = min((ci + len(chunk_frames)) * interval, duration_sec)
        chunk_dur = chunk_end_sec - chunk_start_sec
        print(f"  VLM chunk {chunk_idx}/{n_chunks}: frames {ci+1}-{ci+len(chunk_frames)} "
              f"({_fmt_mmss(chunk_start_sec)}-{_fmt_mmss(chunk_end_sec)})")

        chunk_result = None
        cjk_fallback = None
        raw = ""
        for attempt, profile in enumerate(_CHUNK_RETRY_PROFILES):
            gen = dict(profile)  # module-level constant: never mutate in place
            ocr_for_attempt = (ocr_lines or [])[:gen.pop("ocr_limit", 60)]
            if attempt > 0:
                print(f"    chunk {chunk_idx} retry {attempt}: "
                      f"sample={gen.get('do_sample')} temp={gen.get('temperature')} "
                      f"ocr_lines={len(ocr_for_attempt)}")
            raw = run_vlm(
                chunk_frames, chunk_dur, interval,
                game_hint=game_hint,
                vocab_hint=vocab_hint,
                name_hint=f"{name_hint} (part {chunk_idx}/{n_chunks})",
                path_hints=path_hints,
                ocr_lines=ocr_for_attempt,
                device=device,
                **gen,
            )
            if _is_degenerate(raw):
                print(f"    chunk {chunk_idx} attempt {attempt + 1}: degenerate output")
                continue
            try:
                parsed = parse_json_response(raw)
            except Exception:
                print(f"    chunk {chunk_idx} attempt {attempt + 1}: bad JSON")
                continue
            if _chunk_has_cjk(parsed):
                # Keep the first one: a Chinese chunk still carries usable
                # scene timings, and losing it would blank that stretch of the
                # video. Retry first — a different decoding usually answers in
                # English.
                cjk_fallback = cjk_fallback or parsed
                print(f"    chunk {chunk_idx} attempt {attempt + 1}: non-English output")
                continue
            chunk_result = parsed
            break

        if chunk_result is None and cjk_fallback is not None:
            # Every attempt answered in Chinese. Take the scenes anyway — their
            # timings are sound and the stretch would otherwise go undescribed —
            # but drop the text that feeds search: a Chinese tag never matches a
            # query, it only pollutes the record.
            print(f"    chunk {chunk_idx} stayed non-English after "
                  f"{len(_CHUNK_RETRY_PROFILES)} attempts, keeping scenes only")
            chunk_result = {**cjk_fallback, "title": "", "summary": "",
                            "tags": [t for t in cjk_fallback.get("tags") or []
                                     if not _has_cjk(str(t))]}

        if chunk_result is None:
            # Losing one chunk costs chunk_size*interval seconds of coverage —
            # keep the rest of the video rather than failing it outright.
            debug_path = (config.CATALOG_DIR / "_debug" /
                          f"{name_hint}_chunk{chunk_idx}_raw.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw, encoding="utf-8")
            print(f"    chunk {chunk_idx} failed after "
                  f"{len(_CHUNK_RETRY_PROFILES)} attempts, skipping "
                  f"(raw -> {debug_path.name})")
            continue

        title_candidates.append({
            "index": chunk_idx - 1,
            "title": chunk_result.get("title", ""),
            "gameplay": _gameplay_scene_count(chunk_result),
        })
        summary = chunk_result.get("summary", "")
        if summary and not _has_cjk(summary):
            all_summaries.append(summary)
        all_tags.extend(t for t in chunk_result.get("tags", [])
                        if not _has_cjk(str(t)))

        for scene in chunk_result.get("scenes", []):
            try:
                start = _parse_chunk_ts(scene.get("start", "00:00"))
                end = _parse_chunk_ts(scene.get("end", "00:00"))
                if end < start:
                    end = start
                # The model occasionally times a scene past the frames it was
                # given; clamp so a scene never claims footage from the next
                # chunk.
                span = chunk_end_sec - chunk_start_sec
                start = min(start, span)
                end = min(end, span)
                scene["start"] = _fmt_mmss(start + chunk_start_sec)
                scene["end"] = _fmt_mmss(end + chunk_start_sec)
                all_scenes.append(scene)
            except (ValueError, IndexError):
                continue

    seen: set[str] = set()
    deduped_tags = []
    for t in all_tags:
        if t and t not in seen:
            seen.add(t)
            deduped_tags.append(t)

    return {
        "title": _pick_title(title_candidates),
        "summary": " ".join(all_summaries),
        "tags": deduped_tags,
        "scenes": all_scenes,
    }


def analyze_video(
    video_path: Path,
    game: str,
    interval: int | None = None,
    max_frames: int | None = None,
    device: str = "cuda",
    keep_frames: bool = False,
) -> dict:
    duration = get_video_duration(video_path)
    duration_sec = int(duration)

    frame_cap = max_frames or config.VLM_FRAME_MAX_FRAMES
    min_interval = config.VLM_FRAME_INTERVAL_SEC
    if interval is not None:
        chosen_interval = interval
    else:
        needed = -(-duration_sec // max(1, frame_cap))  # ceil div
        chosen_interval = max(min_interval, needed)
    interval = chosen_interval

    work_dir = config.VLM_WORK_DIR / video_path.stem
    print(f"  Duration {duration_sec}s, sampling 1 frame / {interval}s")
    frames = extract_frames(video_path, work_dir, interval, config.VLM_FRAME_MAX_SIZE)
    if not frames:
        raise RuntimeError("No frames extracted")
    print(f"  {len(frames)} frames -> VLM")

    frame_paths = [f for f, _ in frames]
    vocab = vocab_for(game)
    vocab_hint = render_vocab_hint(vocab) if vocab else ""
    path_hints = extract_path_hints(video_path, game)
    if path_hints["location"] or path_hints["entities"]:
        print(f"  Path hints: location={path_hints['location']!r}  entities={path_hints['entities']}")

    ocr_lines: list[str] = []
    per_frame_ocr: list[list[str]] = [[] for _ in frame_paths]
    if config.OCR_ENABLED:
        from .ocr import (
            extract_frame_text, dedupe_preserving_order,
            filter_noise, release as release_ocr,
        )
        raw_per_frame = extract_frame_text(
            frame_paths,
            languages=config.OCR_LANGUAGES,
            min_confidence=config.OCR_MIN_CONFIDENCE,
        )
        per_frame_ocr = [filter_noise(lines) for lines in raw_per_frame]
        ocr_lines = dedupe_preserving_order(per_frame_ocr)
        frames_with_text = sum(1 for l in per_frame_ocr if l)
        print(f"  OCR: {len(ocr_lines)} unique lines  |  "
              f"{frames_with_text}/{len(frame_paths)} frames with text")
        release_ocr()

    prompt_ocr = ocr_lines[:60]
    vlm_chunk_size = config.VLM_CHUNK_SIZE

    retry_profiles = [
        {"do_sample": True, "temperature": 0.1},
        {"do_sample": True, "temperature": 0.4},
        {"do_sample": True, "temperature": 0.7, "ocr_limit": 30},
    ]

    if len(frame_paths) > vlm_chunk_size:
        print(f"  Chunked VLM: {len(frame_paths)} frames in chunks of {vlm_chunk_size}")
        analysis = _run_vlm_chunked(
            frame_paths, duration_sec, interval,
            game_hint=game.replace("_", " "),
            vocab_hint=vocab_hint,
            name_hint=video_path.stem,
            path_hints=path_hints,
            ocr_lines=prompt_ocr,
            device=device,
            chunk_size=vlm_chunk_size,
        )
    else:
        analysis = None
        for attempt, profile in enumerate(retry_profiles):
            ocr_for_attempt = prompt_ocr[:profile.pop("ocr_limit", 60)]
            if attempt > 0:
                print(f"  Retry {attempt}: sample={profile.get('do_sample')}  "
                      f"rep_pen={profile.get('repetition_penalty')}  "
                      f"ocr_lines={len(ocr_for_attempt)}")
            raw = run_vlm(
                frame_paths, duration_sec, interval,
                game_hint=game.replace("_", " "),
                vocab_hint=vocab_hint,
                name_hint=video_path.stem,
                path_hints=path_hints,
                ocr_lines=ocr_for_attempt,
                device=device,
                **profile,
            )
            if _is_degenerate(raw):
                print(f"  Attempt {attempt + 1}/{len(retry_profiles)}: degenerate output")
                continue
            try:
                analysis = parse_json_response(raw)
                break
            except Exception:
                print(f"  Attempt {attempt + 1}/{len(retry_profiles)}: bad JSON")
                continue

        if analysis is None:
            debug_path = config.CATALOG_DIR / "_debug" / f"{video_path.stem}_raw.txt"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw, encoding="utf-8")
            raise RuntimeError(
                f"All {len(retry_profiles)} attempts failed  (raw saved to {debug_path})"
            )

    scenes = analysis.pop("scenes", [])
    analysis.pop("timeline", None)
    analysis = clean_analysis(analysis, duration_sec)
    analysis["timeline"] = build_ocr_timeline(per_frame_ocr, interval,
                                              duration_sec, vocab)
    if scenes:
        analysis["scenes"] = clamp_scenes(scenes, duration_sec)
    analysis = inject_path_facts(analysis, path_hints)

    record = {
        "source_path": str(video_path),
        "original_name": video_path.name,
        "duration_sec": duration_sec,
        "frame_interval_sec": interval,
        "n_frames_analyzed": len(frames),
        "model": config.VLM_MODEL,
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game": game,
        "ocr_text": ocr_lines,
        **analysis,
    }
    # Must run on the assembled record, not on `analysis` — the game key lives
    # here, and regenerate_suggestions() feeds it the record too. Passing
    # `analysis` silently dropped the game display name from every new name.
    record["suggested_filename"] = suggest_filename(record, video_path)

    if not keep_frames:
        for f in frame_paths:
            f.unlink(missing_ok=True)
        try:
            work_dir.rmdir()
        except OSError:
            pass

    return record


_DIR_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")


def _sanitize_dirname(name: str) -> str:
    """Lowercase, spaces->underscores, drop non-alnum. Returns '' if empty."""
    s = re.sub(r"\s+", "_", name.strip().lower())
    s = _DIR_SANITIZE_RE.sub("", s)
    return s


def _catalog_subpath_for(video_path: Path, game: str) -> Path:
    """Mirror the source folder structure under the game's catalog subdir.

    E.g. `<RAW_FOOTAGE_DIR>\\Eve Online\\Gallente President Election\\foo.mp4` (game root
    is `Eve Online`) -> catalog/eve_online_videos/gallente_elections (if an
    override is configured, else sanitized source folder name).
    """
    cfg = config.GAMES.get(game, {})
    overrides: dict[str, str] = cfg.get("subdir_overrides") or {}
    rel_parent: Path | None = None
    for root in cfg.get("video_dirs", []):
        try:
            rel = video_path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            continue
        rel_parent = rel.parent
        break

    if rel_parent is None or str(rel_parent) in (".", ""):
        return Path()

    parts: list[str] = []
    for raw in rel_parent.parts:
        name = overrides.get(raw) or _sanitize_dirname(raw)
        if name:
            parts.append(name)
    return Path(*parts) if parts else Path()


def catalog_json_path(video_path: Path, game: str) -> Path:
    """Where the catalog entry for a video lives. Pure: creates nothing."""
    subdir = config.GAMES.get(game, {}).get("catalog_subdir", f"{game}_videos")
    out_dir = config.CATALOG_DIR / subdir / _catalog_subpath_for(video_path, game)
    return out_dir / f"{video_path.stem}.json"


def catalog_path_for(video_path: Path, game: str) -> Path:
    """catalog_json_path with its folder created, ready to write."""
    out = catalog_json_path(video_path, game)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_record(record: dict, out_path: Path):
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- Batch / CLI ----------

def collect_videos(roots: list[Path]) -> list[Path]:
    out = []
    for r in roots:
        if not r.exists():
            print(f"  WARNING: not found: {r}")
            continue
        if r.is_file() and r.suffix.lower() in VIDEO_EXTS:
            out.append(r)
            continue
        for ext in VIDEO_EXTS:
            out.extend(r.rglob(f"*{ext}"))
    return sorted(set(out))


def regenerate_suggestions(game: str, dry_run: bool = False) -> int:
    """Re-run suggest_filename over every cached JSON without touching the
    video files. Returns number of JSONs updated (or, on a dry run, that
    would be updated — nothing is written then)."""
    game_cfg = config.GAMES.get(game)
    if not game_cfg:
        print(f"Unknown game '{game}'. Known: {list(config.GAMES)}")
        sys.exit(1)
    subdir = game_cfg.get("catalog_subdir", f"{game}_videos")
    cat_dir = config.CATALOG_DIR / subdir
    if not cat_dir.is_dir():
        print(f"No catalog dir: {cat_dir}")
        return 0

    updated = 0
    for p in sorted(cat_dir.rglob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP (bad json): {p.name} — {e}")
            continue
        original = Path(rec.get("source_path") or rec.get("original_name") or p.stem)
        new_name = suggest_filename(rec, original)
        if rec.get("suggested_filename") != new_name:
            if dry_run:
                print(f"  {p.name}: {rec.get('suggested_filename')!r} -> {new_name!r}")
            else:
                save_record({**rec, "suggested_filename": new_name}, p)
            updated += 1
    total = len(list(cat_dir.rglob('*.json')))
    verb = "DRY-RUN: would regenerate" if dry_run else "Regenerated"
    print(f"{verb} suggestions for {updated}/{total} JSONs")
    return updated


def _rename_plan(game: str) -> list[tuple[Path, Path, Path, Path]]:
    """Build a rename plan. Each video gets `" N"` appended before the
    extension, where N disambiguates videos sharing the same base name.
    Already-numbered videos keep their number, so the plan is idempotent and
    only ever touches new entries. Returns a list of
    (video_old, video_new, json_old, json_new)."""
    game_cfg = config.GAMES[game]
    subdir = game_cfg.get("catalog_subdir", f"{game}_videos")
    cat_dir = config.CATALOG_DIR / subdir

    # Gather (src, base_name, json_path) for every entry with a live video.
    entries: list[tuple[Path, str, Path]] = []
    for jp in sorted(cat_dir.rglob("*.json")):
        try:
            rec = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        src = Path(rec.get("source_path") or "")
        if not src.is_file():
            print(f"  SKIP (video missing): {src}")
            continue
        base = rec.get("suggested_filename") or ""
        if not base:
            continue
        entries.append((src, base, jp))

    # Group by base name and assign seq in chronological order.
    from collections import defaultdict
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for src, base, jp in entries:
        groups[base].append((src, jp))
    for base, items in groups.items():
        items.sort(key=lambda t: t[0].name)

    plan: list[tuple[Path, Path, Path, Path]] = []
    claimed: set[Path] = set()
    for base, items in groups.items():
        stem, ext = Path(base).stem, Path(base).suffix
        seq_re = re.compile(re.escape(stem) + " ([0-9]+)" + re.escape(ext) + "$")

        # A video that already carries a sequence number for this base keeps it.
        # The number only disambiguates duplicates, so re-deriving it from the
        # current filename order renumbers settled files for nothing — and once
        # a group passes nine entries, string order alone cascades ("10" sorts
        # before "2", pushing 2->12, 3->13, ...).
        taken: set[int] = set()
        pending: list[tuple[Path, Path]] = []
        for src, jp in items:
            m = seq_re.match(src.name)
            if m:
                taken.add(int(m.group(1)))
                claimed.add(src)
            else:
                pending.append((src, jp))

        # Fresh entries take the lowest free numbers, in recording order
        # (items is sorted by filename, and un-renamed names embed the timestamp).
        next_n = 1
        for src, jp in pending:
            while True:
                new_video = src.with_name(f"{stem} {next_n}{ext}")
                if (next_n not in taken and new_video not in claimed
                        and not new_video.exists()):
                    break
                next_n += 1
            taken.add(next_n)
            claimed.add(new_video)
            new_json = jp.with_name(f"{new_video.stem}.json")
            plan.append((src, new_video, jp, new_json))

    plan.sort(key=lambda p: p[0].name)
    return plan


def rename_from_catalog(game: str, dry_run: bool) -> int:
    """Use suggested_filename in each catalog JSON to rename the matching
    video (and its JSON). Returns number of renames executed (or planned)."""
    plan = _rename_plan(game)
    if not plan:
        print("Nothing to rename.")
        return 0

    print(f"{'DRY-RUN' if dry_run else 'RENAMING'}: {len(plan)} video(s)")
    for old_v, new_v, _, _ in plan:
        print(f"  {old_v.name}  ->  {new_v.name}")

    if dry_run:
        return len(plan)

    executed = 0
    for old_v, new_v, old_j, new_j in plan:
        try:
            old_v.rename(new_v)
        except OSError as e:
            print(f"  FAILED video rename {old_v.name}: {e}")
            continue
        # Update JSON contents to reflect new paths, then rename the JSON.
        try:
            rec = json.loads(old_j.read_text(encoding="utf-8"))
            rec["source_path"] = str(new_v)
            rec["original_name"] = new_v.name
            save_record(rec, old_j)
            if new_j != old_j:
                old_j.rename(new_j)
        except Exception as e:
            print(f"  WARN json update {old_j.name}: {e}")
        executed += 1
    print(f"Renamed {executed}/{len(plan)}")
    return executed


def index_plan(game: str, videos: list[Path], force: bool) -> list[tuple[Path, Path, bool]]:
    """(video, catalog json, will_analyze) per video. Pure: touches nothing."""
    plan = []
    for v in videos:
        out = catalog_json_path(v, game)
        plan.append((v, out, force or not out.exists()))
    return plan


def print_index_plan(plan: list[tuple[Path, Path, bool]]) -> int:
    """Dry-run report of an indexing batch. Returns the number to analyze."""
    todo = sum(1 for _, _, run in plan if run)
    print(f"DRY-RUN: {todo} of {len(plan)} video(s) would be analyzed "
          f"(no model is loaded, nothing is written)")
    for v, out, run in plan:
        print(f"  {'ANALYZE' if run else 'skip   '}  {v.name}  ->  {out}")
    return todo


def run_batch(game: str, paths: list[Path] | None, force: bool,
              interval: int | None, max_frames: int | None = None,
              dry_run: bool = False):
    game_cfg = config.GAMES.get(game)
    if not game_cfg:
        print(f"Unknown game '{game}'. Known: {list(config.GAMES)}")
        sys.exit(1)

    # An explicit path list is a caller's precise instruction, so a path that does
    # not resolve is a bug in the caller, not something to skip past. collect_videos
    # only warns; continuing on the survivors once cost a catalog-wide --force
    # re-analysis, because a mangled argument list still contained "." and that
    # expands to the whole tree.
    if paths:
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            print(f"{len(missing)} of {len(paths)} paths do not exist, e.g.:")
            for p in missing[:5]:
                print(f"  {p}")
            sys.exit(1)

    roots = paths or game_cfg["video_dirs"]
    videos = collect_videos(roots)
    if not videos:
        print("No videos found.")
        return

    # A dry run stops here: no device probe, no model, no catalog folders.
    if dry_run:
        print_index_plan(index_plan(game, videos, force))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Game: {game}  |  Videos: {len(videos)}")

    for i, v in enumerate(videos, 1):
        out = catalog_path_for(v, game)
        if out.exists() and not force:
            print(f"[{i}/{len(videos)}] skip (cached): {v.name}")
            continue
        print(f"[{i}/{len(videos)}] {v.name}")
        try:
            rec = analyze_video(v, game=game, interval=interval,
                                max_frames=max_frames, device=device)
            save_record(rec, out)
            print(f"  -> {out.name}  | suggested: {rec['suggested_filename']}")
        except Exception as e:
            print(f"  FAILED: {e}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="VLM video analyzer (Stage 1)")
    ap.add_argument("paths", nargs="*", type=Path,
                    help="Video files or directories. If empty, uses GAMES[game].video_dirs.")
    ap.add_argument("--game", default=config.DEFAULT_GAME, choices=list(config.GAMES))
    ap.add_argument("--interval", type=int, default=None,
                    help=f"Seconds between sampled frames (default {config.VLM_FRAME_INTERVAL_SEC})")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Max frames to extract (overrides adaptive cap; VLM processes in chunks of 50)")
    ap.add_argument("--force", action="store_true", help="Re-analyze even if catalog entry exists")
    ap.add_argument("--regenerate-suggestions", action="store_true",
                    help="Re-run suggest_filename on every cached JSON (no VLM call)")
    ap.add_argument("--rename", action="store_true",
                    help="Rename video files to match suggested_filename (and update JSONs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the plan without doing anything: with --rename the "
                         "renames, with --regenerate-suggestions the changed names, "
                         "otherwise which videos would be analyzed. Never loads the "
                         "model and never writes.")
    args = ap.parse_args(argv)

    if args.regenerate_suggestions:
        regenerate_suggestions(args.game, dry_run=args.dry_run)
        return
    if args.rename:
        rename_from_catalog(args.game, dry_run=args.dry_run)
        return

    run_batch(args.game, args.paths or None, force=args.force,
              interval=args.interval, max_frames=args.max_frames,
              dry_run=args.dry_run)


if __name__ == "__main__":
    main()

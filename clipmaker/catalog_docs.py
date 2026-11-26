"""Catalog entry -> embeddable documents for the semantic index.

Three levels (see embed_index): the whole video, each VLM scene, and windows
of OCR timeline captions. Pure functions over catalog JSON - no model, no
files - so the document rules can be read and tested on their own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Document:
    """One embeddable unit of the catalog."""

    doc_id: str
    text: str
    level: str  # video | scene | window
    game: str
    catalog_path: str
    original_name: str
    source_path: str
    title: str
    start_sec: float | None
    end_sec: float | None
    scene_type: str | None


def fmt_time(seconds: float) -> str:
    """Seconds as MM:SS, or H:MM:SS from one hour on."""
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


_TIMESTAMP = re.compile(r"\s*(\d+(?:\s*:\s*\d+)*)(\.\d+)?\s*")


def parse_timestamp(value: str | int | float | None) -> float | None:
    """Parse "mm:ss" / "hh:mm:ss" / seconds into seconds; a fraction is kept
    ("01:02.5" -> 62.5). None for anything else."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _TIMESTAMP.fullmatch(str(value))
    if not m:
        return None
    seconds = 0.0
    for part in m.group(1).split(":"):
        seconds = seconds * 60 + int(part)
    return seconds + float(m.group(2) or 0)


def clean_caption(caption: str) -> str:
    """Collapse the "; "-joined OCR soup in timeline captions.

    A raw caption looks like "Capsule; Capsul; Capsule; Approal; No signatures"
    — the same string re-read from consecutive frames. Keeps first occurrences
    in order and drops fragments too short to carry meaning.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for chunk in re.split(r"[;\n]+", caption):
        piece = chunk.strip()
        if len(piece) < 3:
            continue
        key = piece.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(piece)
    return ", ".join(kept)


def _video_text(entry: dict) -> str:
    parts = [
        entry.get("title", ""),
        entry.get("summary", ""),
        " ".join(entry.get("tags", []) or []),
    ]
    return " ".join(p for p in parts if p).strip()


def _scene_documents(entry: dict, base: dict) -> list[Document]:
    # Scenes indexed in chunks before the 2026-07-30 parser fix carry timecodes
    # squeezed into the first minute of each chunk: a search would point at the
    # wrong moment. Leave them out; the caller falls back to timeline windows,
    # whose timecodes come from frame positions and are right. Re-running the
    # analyzer writes a fresh record without the flag.
    if entry.get("scene_timecodes_unreliable"):
        return []
    docs = []
    title = entry.get("title", "")
    for i, scene in enumerate(entry.get("scenes") or []):
        description = (scene.get("description") or "").strip()
        if len(description) < config.EMBED_MIN_CHARS:
            continue
        scene_type = scene.get("type")
        # The video title is prepended so a scene keeps its subject when the
        # description is generic ("the player approaches a large structure").
        text = f"{title}. {description}" if title else description
        docs.append(Document(
            doc_id=f"{base['catalog_path']}#scene{i}",
            text=text,
            level="scene",
            start_sec=parse_timestamp(scene.get("start")),
            end_sec=parse_timestamp(scene.get("end")),
            scene_type=scene_type,
            **base,
        ))
    return docs


def _group_timeline(timeline: list[dict]) -> list[list[dict]]:
    """Cut the timeline into windows bounded by both entry count and duration.

    Entry granularity is uneven across the catalog (median 8 s, up to 128 s on
    coarsely indexed videos), so a fixed entry count yields windows spanning
    minutes on exactly the videos where a precise moment matters most.
    """
    max_entries = max(1, config.EMBED_WINDOW_FRAMES)
    max_span = config.EMBED_WINDOW_MAX_SEC

    groups: list[list[dict]] = []
    current: list[dict] = []
    start: float | None = None

    for frame in timeline:
        if len(current) >= max_entries:
            groups.append(current)
            current, start = [], None

        if current and start is not None and max_span:
            end = parse_timestamp(frame.get("end"))
            if end is not None and end - start > max_span:
                groups.append(current)
                current, start = [], None

        if not current:
            start = parse_timestamp(frame.get("start"))
        current.append(frame)

    if current:
        groups.append(current)
    return groups


# Bump when iter_documents builds documents differently: an index built under
# other rules is rebuilt in full, since the catalog files themselves did not
# change and the incremental build would otherwise keep the old documents.
# 2: speech windows for videos with scenes (2026-11-22).
DOC_RULES = 2


# A window counts as speech when its OCR has at least this many lowercase words
# of 3+ letters: subtitles are sentences ("You're not aiming high enough"),
# HUD and menus are capitalised labels ("FOLLOW; Green; Follow").
SPEECH_MIN_WORDS = 6


def looks_like_speech(text: str) -> bool:
    """True for OCR text that reads like a subtitle sentence, not HUD or menu labels."""
    words = re.findall(r"[^\W\d_]{3,}", text)
    return sum(1 for w in words if w[0].islower()) >= SPEECH_MIN_WORDS


def _window_documents(entry: dict, base: dict, speech_only: bool = False) -> list[Document]:
    timeline = entry.get("timeline") or []
    if not timeline:
        return []
    title = entry.get("title", "")
    docs = []
    for i, chunk in enumerate(_group_timeline(timeline)):
        captions = [clean_caption(f.get("caption", "")) for f in chunk]
        if speech_only and not looks_like_speech(" ".join(captions)):
            continue
        tags = {t for f in chunk for t in (f.get("tags") or [])}
        body = ", ".join(c for c in captions if c)
        if tags:
            body = f"{body}. Tags: {', '.join(sorted(tags))}" if body else ", ".join(sorted(tags))
        if len(body) < config.EMBED_MIN_CHARS:
            continue
        text = f"{title}. {body}" if title else body
        docs.append(Document(
            doc_id=f"{base['catalog_path']}#win{i}",
            text=text,
            level="window",
            start_sec=parse_timestamp(chunk[0].get("start")),
            end_sec=parse_timestamp(chunk[-1].get("end")),
            scene_type=None,
            **base,
        ))
    return docs


def iter_documents(entry: dict, catalog_path: str) -> list[Document]:
    """Build every document for one catalog JSON."""
    base = {
        "game": entry.get("game", ""),
        "catalog_path": catalog_path,
        "original_name": entry.get("original_name", ""),
        "source_path": entry.get("source_path", ""),
        "title": entry.get("title", ""),
    }

    docs: list[Document] = []
    video_text = _video_text(entry)
    if len(video_text) >= config.EMBED_MIN_CHARS:
        docs.append(Document(
            doc_id=f"{catalog_path}#video",
            text=video_text,
            level="video",
            start_sec=None,
            end_sec=None,
            scene_type=None,
            **base,
        ))

    scene_docs = _scene_documents(entry, base)
    docs.extend(scene_docs)
    # Every window where scenes are missing; with scenes, only the speech
    # windows (subtitles) — see the module docstring.
    docs.extend(_window_documents(entry, base, speech_only=bool(scene_docs)))
    return docs

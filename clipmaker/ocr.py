"""
OCR helper: extract visible on-screen text from VLM frames using EasyOCR.

Used by the analyzer to surface UI overlays (system names, timer values,
notifications) that the 3B VLM cannot read. Results are deduplicated across
frames and injected into the VLM prompt as a separate "Text on screen" block,
and persisted in the output JSON.
"""
from __future__ import annotations

from pathlib import Path

_ocr_cache: dict = {}


def _imread_unicode(path: Path):
    """Read an image into a BGR numpy array, tolerating non-ASCII paths.

    cv2.imread (used internally by EasyOCR when given a path string) calls a
    narrow-char fopen on Windows and silently fails on Unicode paths — e.g. a
    frame dir named after a video whose title contains an em-dash. Reading the
    bytes via np.fromfile and decoding with cv2.imdecode sidesteps that.
    Returns None if the file is missing or cannot be decoded.
    """
    import cv2
    import numpy as np
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _get_reader(languages=("en",), gpu: bool = True):
    key = (tuple(languages), gpu)
    if key in _ocr_cache:
        return _ocr_cache[key]
    import easyocr
    reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
    _ocr_cache[key] = reader
    return reader


def extract_frame_text(
    frame_paths: list[Path],
    languages=("en",),
    gpu: bool = True,
    min_confidence: float = 0.5,
    min_length: int = 2,
) -> list[list[str]]:
    """Run OCR on each frame. Returns a list aligned with frame_paths,
    each entry being the raw text lines detected on that frame (filtered
    by confidence and length)."""
    reader = _get_reader(languages, gpu)
    per_frame: list[list[str]] = []
    for fp in frame_paths:
        img = _imread_unicode(fp)
        if img is None:
            per_frame.append([])
            continue
        try:
            detections = reader.readtext(img)
        except Exception:
            per_frame.append([])
            continue
        lines: list[str] = []
        for _bbox, text, conf in detections:
            t = (text or "").strip()
            if not t or len(t) < min_length:
                continue
            if conf < min_confidence:
                continue
            lines.append(t)
        per_frame.append(lines)
    return per_frame


_ALNUM = __import__("re").compile(r"[A-Za-z0-9]")
_LETTERS = __import__("re").compile(r"[A-Za-z]")


def _is_noise(line: str) -> bool:
    """Drop lines that are clearly OCR garbage and not useful for grounding."""
    s = line.strip()
    if len(s) < 3:
        return True
    letters = len(_LETTERS.findall(s))
    # Pure numeric / punctuation (e.g. "3171", "55.71 AU" is OK since it has letters)
    if letters == 0:
        return True
    # Very short alpha only (e.g. "ann", "mam") — keep if >=4 chars
    alnum = len(_ALNUM.findall(s))
    if alnum < 3:
        return True
    # Reject if less than half the chars are alphanumeric (lots of noise)
    if alnum / len(s) < 0.5:
        return True
    return False


def filter_noise(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not _is_noise(ln)]


# ---- Phrase normalization & chrome filtering ----

# Multi-word UI phrases that OCR often splits across detections. If all parts
# appear on a frame (in any order), they collapse to the canonical form and the
# original fragments are removed.
_PHRASE_GROUPS = [
    (("warp", "drive", "active"), "Warp Drive Active"),
    (("establishing", "warp", "vector"), "Establishing Warp Vector"),
    (("establishing", "warp"), "Establishing Warp"),
    (("warp", "drive"), "Warp Drive"),
    (("scan", "results"), "Scan Results"),
    (("no", "object", "selected"), "No Object Selected"),
    (("nothing", "found"), "Nothing Found"),
    (("plex", "vault"), "PLEX Vault"),
    (("memorial", "research", "site"), "Memorial Research Site"),
    (("wave", "observatory"), "Wave Observatory"),
    (("duvolle", "gravitational", "wave"), "Duvolle Gravitational Wave"),
    (("duvolle", "gravitational"), "Duvolle Gravitational"),
    (("control", "tower"), "Control Tower"),
    (("combat", "site"), "Combat Site"),
    (("cosmic", "signature"), "Cosmic Signature"),
    (("research", "facility"), "Research Facility"),
]

# OCR commonly misspells these — normalize to canonical before phrase matching.
_SPELL_FIXES = {
    "geneial": "general",
    "watp": "warp",
    "warpto": "warp",
    "sran": "scan",
    "stan": "scan",
    "cumbal": "combat",
    "combal": "combat",
    "dre_site": "combat site",
    "signatlte": "signature",
    "cosic": "cosmic",
    "vaull": "vault",
    "cumbal sill": "combat site",
    "hining": "mining",
    "watp to": "warp to",
    "warp drive active": "warp drive active",
    "observatorv": "observatory",
    "gravitationa": "gravitational",
}

# Chrome tokens — always drop from captions (kept in ocr_text for reference).
_CHROME = {
    "route", "general", "channel", "filters", "filtets", "result", "results",
    "destination", "eve system", "eve sustem", "eve susten", "eve'sustem",
    "agents", "encounters", "enccunters", "encolnters", "corp",
    "local", "target", "orbc", "orbiting", "cont", "contto", "control",
    "buy", "sell", "trade", "trading", "gallente space", "space",
    "receive", "comes", "field", "proxiity", "proximity",
    "presidng", "presiding", "checrel changed", "channel changed",
    "points", "points eutned", "1075polnts",
    "transpot ships", "pilot", "poitot",  # "poitot" is actually a system — keep as code
}

# Chrome is single-word match only — keep Poitot etc.
_CHROME_STRICT = {
    "route", "general", "filters", "filtets", "result", "results",
    "destination", "agents", "encounters", "enccunters", "encolnters",
    "orbc", "orbiting", "cont", "contto", "control",
    "receive", "comes", "field", "proxiity", "proximity",
    "trading", "space", "channel", "corp",
    "points", "target", "eve system", "eve sustem", "eve susten", "eve'sustem",
    "gallente space",
}

# Chrome patterns as regex (case-insensitive).
_CHROME_PATTERNS = [
    __import__("re").compile(p, __import__("re").IGNORECASE)
    for p in (
        r"^corp\s*\[\d+\]$",
        r"^local\s*\[\d+\a?\]?$",
        r"^loca\s*\[\d+\]?$",
        r"^com\s*\[\d+\]$",
        r"^\d+(?:[.,]\d+)?\s*(?:au|av|kn)$",  # distances like "63 3 AU", "23 kn"
        r"^[a-z]+\s*changed$",
    )
]


def _norm_tok(s: str) -> str:
    s = s.strip().lower()
    return _SPELL_FIXES.get(s, s)


def _collapse_phrases(lines: list[str]) -> list[str]:
    """Merge split OCR fragments that form a known multi-word phrase."""
    norm = [_norm_tok(ln) for ln in lines]
    used: set[int] = set()
    out: list[str] = []

    # Try longest phrases first.
    for parts, canonical in sorted(_PHRASE_GROUPS, key=lambda p: -len(p[0])):
        idxs = []
        for part in parts:
            for i, n in enumerate(norm):
                if i in used or i in idxs:
                    continue
                if n == part or part in n.split():
                    idxs.append(i)
                    break
            else:
                idxs = []
                break
        if idxs:
            out.append(canonical)
            used.update(idxs)

    for i, original in enumerate(lines):
        if i not in used:
            out.append(original)
    return out


def _is_chrome(line: str) -> bool:
    s = _norm_tok(line)
    if s in _CHROME_STRICT:
        return True
    for pat in _CHROME_PATTERNS:
        if pat.match(line.strip()):
            return True
    return False


def filter_chrome(lines: list[str]) -> list[str]:
    """Remove UI-chrome tokens after phrase collapsing. Keeps captions readable."""
    collapsed = _collapse_phrases(lines)
    return [ln for ln in collapsed if not _is_chrome(ln)]


def dedupe_preserving_order(per_frame_text: list[list[str]]) -> list[str]:
    """Flatten per-frame text lists and return unique lines in first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for lines in per_frame_text:
        for line in lines:
            key = line.casefold().strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(line.strip())
    return out


def release():
    """Free the EasyOCR reader (and its GPU buffers) so downstream models
    can claim the VRAM. Safe to call multiple times."""
    import gc
    try:
        import torch
    except ImportError:
        torch = None
    _ocr_cache.clear()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

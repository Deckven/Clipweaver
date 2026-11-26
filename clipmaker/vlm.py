"""The vision-language model step of the analyzer.

Loads Qwen2.5-VL, builds the prompt from frames, OCR lines and path hints,
parses its JSON reply (tolerating truncation and stray commas), retries with
hotter decoding when a reply is unusable, and merges per-chunk replies for
videos with more frames than one call can hold.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


class _LazyTorch:
    """torch, imported on first use: --help, dry runs and the tests that only
    touch naming and catalog logic work without the `vlm` extra installed."""

    def __getattr__(self, name):
        import torch as _torch
        return getattr(_torch, name)


torch = _LazyTorch()


def _empty_cuda_cache() -> None:
    """Free cached GPU memory between chunks. Nothing to free if torch was never
    loaded (tests with a stubbed VLM, machines without the vlm extra)."""
    loaded = sys.modules.get("torch")
    if loaded is not None and loaded.cuda.is_available():
        loaded.cuda.empty_cache()


_vlm_cache: dict = {}


def load_vlm(device: str = "cuda"):
    """The VLM and its processor, loaded once per process (4-bit NF4 by default)."""
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
        log.info("Loading %s (bnb-4bit NF4)...", config.VLM_MODEL)
    else:
        dtype = torch.float16 if config.VLM_DTYPE == "float16" else torch.bfloat16
        load_kwargs["dtype"] = dtype
        log.info("Loading %s (%s)...", config.VLM_MODEL, dtype)

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
        parts.append("Game description (ground the visuals in this — do not "
                     "default to a different genre):")
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
        "            (e.g. 'Naval battle against the HMS Ipswich in a storm').",
        "  summary : 1-3 sentences describing what happens overall. Name specific",
        "            places, characters, vehicles, items, events that you can confirm",
        "            from the OCR list or the vocab. Do NOT invent names.",
        "  tags    : 6-12 snake_case tags covering concrete entities and",
        "            activities (e.g. hms_ipswich, naval_battle, storm, mortar,",
        "            ship_upgrade, harbourmaster).",
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
    """Chat messages for the VLM: the frames as images, then the text prompt."""
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
    """One VLM generation over the frames; returns the raw reply text."""
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

    log.debug("VLM input tokens: %d  |  max_new: %d  |  sample=%s  rep_pen=%s",
              inputs.input_ids.shape[1], max_new, do_sample, repetition_penalty)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids, strict=True)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


_CLOSERS = {"{": "}", "[": "]"}


def _repair_truncated_json(snippet: str) -> str:
    """Close a reply cut off at max_new_tokens: keep it up to the last complete
    element of an open array and close every bracket still open.

    Works for any array (``scenes``, ``tags``, the old ``timeline``) and skips
    brackets inside strings. Returns the snippet unchanged when no array
    element was complete before the cut.
    """
    stack: list[str] = []
    cut: tuple[int, list[str]] | None = None
    in_string = escaped = False
    for i, ch in enumerate(snippet):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                if stack and stack[-1] == "[":
                    cut = (i + 1, list(stack))  # a complete string element
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if stack and stack[-1] == "[":
                cut = (i + 1, list(stack))  # a complete object or array element
    if cut is None:
        return snippet
    end, still_open = cut
    return snippet[:end] + "".join(_CLOSERS[b] for b in reversed(still_open))


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
            return _normalize_keys(json.loads(repair_fn(candidate)))
        except ValueError:  # json.JSONDecodeError is a ValueError
            continue
    raise ValueError(f"Could not parse JSON from VLM output:\n{candidate[:500]}")


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


@dataclass(frozen=True)
class VlmRequest:
    """What every VLM call of one video shares: the prompt context and the device."""

    game_hint: str
    vocab_hint: str
    name_hint: str
    path_hints: dict
    ocr_lines: list[str]
    device: str


@dataclass(frozen=True)
class _Attempts:
    """Outcome of a retry ladder: the parsed reply, a non-English one kept as a
    fallback, and the last raw text (saved for debugging when nothing parsed)."""

    result: dict | None
    cjk_fallback: dict | None
    last_raw: str


# Decoding ladder of a single-call video (short enough for one VLM request).
_SINGLE_RETRY_PROFILES: tuple[dict, ...] = (
    {"do_sample": True, "temperature": 0.1},
    {"do_sample": True, "temperature": 0.4},
    {"do_sample": True, "temperature": 0.7, "ocr_limit": 30},
)


def _decoding(profile: dict, ocr_lines: list[str]) -> tuple[dict, list[str]]:
    """Generation kwargs of a retry profile and the OCR lines it allows."""
    gen = {k: v for k, v in profile.items() if k != "ocr_limit"}
    return gen, ocr_lines[:profile.get("ocr_limit", 60)]


def _ask_with_retries(profiles: tuple[dict, ...], ask: Callable[[dict, list[str]], str],
                      ocr_lines: list[str], label: str, reject_cjk: bool) -> _Attempts:
    """Walk the decoding ladder until a reply parses (and, with reject_cjk, is English).

    ``ask(gen_kwargs, ocr_lines)`` makes one VLM call and returns its raw text.
    """
    cjk_fallback: dict | None = None
    raw = ""
    for attempt, profile in enumerate(profiles):
        gen, ocr = _decoding(profile, ocr_lines)
        if attempt > 0:
            log.info("%s retry %d: sample=%s temp=%s ocr_lines=%d", label, attempt,
                     gen.get("do_sample"), gen.get("temperature"), len(ocr))
        raw = ask(gen, ocr)
        if _is_degenerate(raw):
            log.info("%s attempt %d/%d: degenerate output", label, attempt + 1, len(profiles))
            continue
        try:
            parsed = parse_json_response(raw)
        except ValueError as exc:
            log.info("%s attempt %d/%d: bad JSON", label, attempt + 1, len(profiles))
            log.debug("%s", exc)
            continue
        if reject_cjk and _chunk_has_cjk(parsed):
            # Keep the first one: a Chinese chunk still carries usable scene
            # timings, and losing it would blank that stretch of the video.
            # Retry first - a different decoding usually answers in English.
            cjk_fallback = cjk_fallback or parsed
            log.info("%s attempt %d/%d: non-English output", label, attempt + 1, len(profiles))
            continue
        return _Attempts(parsed, cjk_fallback, raw)
    return _Attempts(None, cjk_fallback, raw)


def _save_raw(raw: str, filename: str) -> Path:
    """Keep an unparseable reply under catalog/_debug for inspection."""
    debug_path = config.CATALOG_DIR / "_debug" / filename
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(raw, encoding="utf-8")
    return debug_path


def _run_vlm_single(frame_paths: list[Path], duration_sec: int, interval: int,
                    req: VlmRequest) -> dict:
    """One VLM call for the whole video; RuntimeError when no attempt parses."""
    def ask(gen: dict, ocr: list[str]) -> str:
        return run_vlm(frame_paths, duration_sec, interval,
                       game_hint=req.game_hint, vocab_hint=req.vocab_hint,
                       name_hint=req.name_hint, path_hints=req.path_hints,
                       ocr_lines=ocr, device=req.device, **gen)

    attempts = _ask_with_retries(_SINGLE_RETRY_PROFILES, ask, req.ocr_lines, "VLM",
                                 reject_cjk=False)
    if attempts.result is None:
        debug_path = _save_raw(attempts.last_raw, f"{req.name_hint}_raw.txt")
        raise RuntimeError(
            f"All {len(_SINGLE_RETRY_PROFILES)} attempts failed  (raw saved to {debug_path})")
    return attempts.result


def _chunk_spans(n_frames: int, chunk_size: int, interval: int,
                 duration_sec: int) -> list[tuple[int, int, int]]:
    """(first frame index, start second, end second) of every chunk."""
    return [(first, first * interval,
             min((first + min(chunk_size, n_frames - first)) * interval, duration_sec))
            for first in range(0, n_frames, chunk_size)]


def _ask_chunk(frames: list[Path], idx: int, n_chunks: int, start: int, end: int,
               interval: int, req: VlmRequest) -> dict | None:
    """One chunk's reply, a scenes-only reply if it stayed non-English, or None."""
    def ask(gen: dict, ocr: list[str]) -> str:
        return run_vlm(frames, end - start, interval,
                       game_hint=req.game_hint, vocab_hint=req.vocab_hint,
                       name_hint=f"{req.name_hint} (part {idx}/{n_chunks})",
                       path_hints=req.path_hints, ocr_lines=ocr, device=req.device, **gen)

    attempts = _ask_with_retries(_CHUNK_RETRY_PROFILES, ask, req.ocr_lines, f"chunk {idx}",
                                 reject_cjk=True)
    if attempts.result is not None:
        return attempts.result
    if attempts.cjk_fallback is not None:
        # Every attempt answered in Chinese. Take the scenes anyway - their
        # timings are sound and the stretch would otherwise go undescribed -
        # but drop the text that feeds search: a Chinese tag never matches a
        # query, it only pollutes the record.
        log.warning("chunk %d stayed non-English after %d attempts, keeping scenes only",
                    idx, len(_CHUNK_RETRY_PROFILES))
        fallback = attempts.cjk_fallback
        return {**fallback, "title": "", "summary": "",
                "tags": [t for t in fallback.get("tags") or [] if not _has_cjk(str(t))]}
    # Losing one chunk costs chunk_size*interval seconds of coverage - keep the
    # rest of the video rather than failing it outright.
    debug_path = _save_raw(attempts.last_raw, f"{req.name_hint}_chunk{idx}_raw.txt")
    log.warning("chunk %d failed after %d attempts, skipping (raw -> %s)",
                idx, len(_CHUNK_RETRY_PROFILES), debug_path)
    return None


def _offset_scenes(scenes: list[dict], start: int, end: int) -> list[dict]:
    """Chunk-relative scene times as video times, clamped to the chunk.

    The model occasionally times a scene past the frames it was given; clamping
    keeps a scene from claiming footage of the next chunk.
    """
    span = end - start
    placed = []
    for scene in scenes:
        try:
            s = _parse_chunk_ts(scene.get("start", "00:00"))
            e = _parse_chunk_ts(scene.get("end", "00:00"))
        except (ValueError, IndexError):
            continue
        e = max(e, s)
        placed.append({**scene, "start": _fmt_mmss(min(s, span) + start),
                       "end": _fmt_mmss(min(e, span) + start)})
    return placed


def _merge_chunks(answered: list[tuple[int, dict, int, int]]) -> dict:
    """One analysis from the chunks that answered: (index, reply, start, end) each."""
    titles: list[dict] = []
    summaries: list[str] = []
    tags: list[str] = []
    scenes: list[dict] = []
    for index, chunk, start, end in answered:
        titles.append({"index": index, "title": chunk.get("title", ""),
                       "gameplay": _gameplay_scene_count(chunk)})
        summary = chunk.get("summary", "")
        if summary and not _has_cjk(summary):
            summaries.append(summary)
        tags.extend(t for t in chunk.get("tags", []) if not _has_cjk(str(t)))
        scenes.extend(_offset_scenes(chunk.get("scenes", []), start, end))
    return {
        "title": _pick_title(titles),
        "summary": " ".join(summaries),
        "tags": list(dict.fromkeys(t for t in tags if t)),
        "scenes": scenes,
    }


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
    req = VlmRequest(game_hint, vocab_hint, name_hint, path_hints, list(ocr_lines or []), device)
    spans = _chunk_spans(len(frame_paths), chunk_size, interval, duration_sec)
    answered = []
    failed: list[dict] = []
    for idx, (first, start, end) in enumerate(spans, 1):
        if first > 0:
            _empty_cuda_cache()
        frames = frame_paths[first:first + chunk_size]
        log.info("VLM chunk %d/%d: frames %d-%d (%s-%s)", idx, len(spans),
                 first + 1, first + len(frames), _fmt_mmss(start), _fmt_mmss(end))
        reply = _ask_chunk(frames, idx, len(spans), start, end, interval, req)
        if reply is None:
            failed.append({"start": _fmt_mmss(start), "end": _fmt_mmss(end)})
        else:
            answered.append((idx - 1, reply, start, end))
    if not answered:
        # An entry with no title, tags or scenes would be saved as done and never
        # retried; failing the video makes the batch report it.
        raise RuntimeError(f"all {len(spans)} chunks failed (raw replies in {config.CATALOG_DIR / '_debug'})")
    merged = _merge_chunks(answered)
    if failed:
        # Kept in the entry so a dry run and --retry-incomplete can find the gap.
        merged["failed_chunks"] = failed
    return merged


def ask_vlm(frame_paths: list[Path], duration_sec: int, interval: int,
            req: VlmRequest, chunk_size: int) -> dict:
    """The video's analysis from the VLM: one call, or chunks when the frames do not fit."""
    if len(frame_paths) <= chunk_size:
        return _run_vlm_single(frame_paths, duration_sec, interval, req)
    log.info("Chunked VLM: %d frames in chunks of %d", len(frame_paths), chunk_size)
    return _run_vlm_chunked(frame_paths, duration_sec, interval, req.game_hint, req.vocab_hint,
                            req.name_hint, req.path_hints, req.ocr_lines, req.device, chunk_size)

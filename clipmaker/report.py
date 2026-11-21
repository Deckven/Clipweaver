"""
Report generator: searches YouTube and local VLM video catalog per text fragment,
writes a storyboard document into the project folder.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from . import config, embed_index, image_search, shot_query
from .youtube_search import (
    split_into_paragraphs,
    extract_search_query,
    search_youtube,
    format_view_count,
)


def _as_games(game: str | list[str] | None) -> list[str]:
    """Normalise the game argument to a list.

    A narration can walk through several titles in one script — the AC lore
    series covers Black Flag, AC3 and Unity in a single episode — so the
    storyboard has to search more than one catalog. Accepts a single key, a
    comma-separated string, or a list.
    """
    if not game:
        return []
    if isinstance(game, str):
        return [g.strip() for g in game.split(",") if g.strip()]
    return [g for g in game if g]


def _load_catalog(game: str | list[str] | None) -> list[dict]:
    entries = []
    for key in _as_games(game):
        game_cfg = config.GAMES.get(key)
        if not game_cfg:
            continue
        cat_dir = config.CATALOG_DIR / game_cfg["catalog_subdir"]
        if not cat_dir.exists():
            continue
        for p in cat_dir.rglob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["_catalog_path"] = str(p.relative_to(config.CATALOG_DIR))
                entries.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return entries


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9']{3,}", text)}


def _build_search_text(entry: dict) -> str:
    parts = [
        entry.get("title", ""),
        entry.get("summary", ""),
        " ".join(entry.get("tags", [])),
    ]
    return " ".join(parts)


def _keyword_search(fragment: str, catalog: list[dict], max_results: int) -> list[dict]:
    """Token-overlap fallback. Only fires on words shared verbatim with the
    catalog, so Russian narration reaches it through proper nouns or not at all."""
    frag_tokens = _tokenize(fragment)
    if not frag_tokens:
        return []

    scored = []
    for entry in catalog:
        entry_tokens = _tokenize(_build_search_text(entry))
        overlap = frag_tokens & entry_tokens
        if len(overlap) >= 2:
            scored.append((len(overlap), entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:max_results]]


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _infer_game(catalog: list[dict]) -> str | None:
    games = {e.get("game") for e in catalog if e.get("game")}
    return games.pop() if len(games) == 1 else None


def _best_hits_across_queries(
    queries: list[shot_query.ShotQuery], max_results: int,
    games: list[str] | None
) -> list[embed_index.Hit]:
    """Search once per shot description, letting every description place a hit.

    Scores are comparable across games because ``embed_index.search`` normalises
    against the whole catalog regardless of the filter, so merging per-game
    result sets ranks them on one scale.

    Selection is round-robin over the shot queries rather than a global top-N.
    A fragment yields several descriptions of differing precision, and pooling
    them by raw score lets whichever query scores highest take every slot. On
    the AC lore episode a paragraph about the French Revolution filled all three
    slots with Black Flag Havana scenes, even though searching its own
    "crowd watching the execution of Louis XVI" reached the Unity footage at
    z=3.54. One slot per description first, extra slots afterwards.
    """
    per_query: list[list[embed_index.Hit]] = []
    for query in queries:
        merged: dict[str, embed_index.Hit] = {}
        for game in (games or [None]):
            for hit in embed_index.search(query.text, k=max_results, game=game):
                tagged = replace(hit, query=query.text if query.source == "llm" else "")
                current = merged.get(hit.catalog_path)
                if current is None or tagged.score > current.score:
                    merged[hit.catalog_path] = tagged
        per_query.append(sorted(merged.values(), key=lambda h: h.score, reverse=True))

    chosen: list[embed_index.Hit] = []
    seen: set[str] = set()
    cursors = [0] * len(per_query)
    while len(chosen) < max_results:
        advanced = False
        for qi, hits in enumerate(per_query):
            if len(chosen) >= max_results:
                break
            while cursors[qi] < len(hits) and hits[cursors[qi]].catalog_path in seen:
                cursors[qi] += 1
            if cursors[qi] < len(hits):
                hit = hits[cursors[qi]]
                cursors[qi] += 1
                seen.add(hit.catalog_path)
                chosen.append(hit)
                advanced = True
        if not advanced:
            break
    return sorted(chosen, key=lambda h: h.score, reverse=True)


def _semantic_search(
    fragment: str, catalog: list[dict], max_results: int,
    game: str | list[str] | None,
    shots: list[shot_query.ShotQuery] | None = None,
) -> list[tuple[dict, embed_index.Hit]] | None:
    """Match through the embedding index, resolving hits back to catalog entries.

    The fragment is first rewritten into visual shot descriptions — embedding a
    narration paragraph directly matches nothing, because the paragraph is about
    the subject and the catalog describes the picture.

    Returns ``None`` when the index is unavailable, so the caller can fall back.
    An empty list is a *result*: either the paragraph has no filmable content or
    every candidate was rejected. That is the honest answer that feeds gap
    analysis in production_plan.
    """
    games = _as_games(game)
    try:
        if embed_index.load_index() is None:
            return None
        # Callers that already built the shot queries pass them in. Rebuilding
        # here instead would issue a second Ollama call whose output differs
        # from the first — the cache key contains the game argument, so a
        # caller holding the full "a,b" game string and this function holding
        # only the primary miss each other, and the storyboard then prints one
        # set of shot queries while searching with another.
        queries = shots
        if queries is None:
            # The vocab hint takes one game; the first is the episode's primary.
            queries = shot_query.build_shot_queries(
                fragment, game=games[0] if games else None
            )
        if not queries:
            return []  # nothing filmable in this paragraph
        hits = _best_hits_across_queries(queries, max_results, games)
    except RuntimeError:  # sentence-transformers absent
        return None

    by_path = {
        _normalize_path(e.get("_catalog_path", "")): e for e in catalog if e.get("_catalog_path")
    }
    by_name = {e.get("original_name", ""): e for e in catalog if e.get("original_name")}

    pairs = []
    for hit in hits:
        entry = by_path.get(_normalize_path(hit.catalog_path)) or by_name.get(hit.original_name)
        if entry is not None:
            pairs.append((entry, hit))
    return pairs


def search_catalog_hits(
    fragment: str,
    catalog: list[dict],
    max_results: int = 3,
    game: str | list[str] | None = None,
    shots: list[shot_query.ShotQuery] | None = None,
) -> list[tuple[dict, embed_index.Hit | None]]:
    """Rank catalog videos for a narration fragment.

    Semantic first — it survives the Russian-narration / English-catalog gap
    and returns the matching *moment*, not just the file.

    A ``hit`` of ``None`` marks a keyword result: the right video, no moment.
    Callers that reason about coverage (production_plan) must treat those as
    weaker evidence than a scored hit — a narration paragraph about patch
    economics matches a video's title while nothing in the footage shows it.
    """
    pairs = _semantic_search(
        fragment, catalog, max_results,
        _as_games(game) or _infer_game(catalog), shots=shots
    )
    if pairs:
        return pairs
    # Semantic found nothing (or is unavailable): keyword still locates the
    # likely video, which beats an empty storyboard section.
    return [(entry, None) for entry in _keyword_search(fragment, catalog, max_results)]


def search_catalog(
    fragment: str,
    catalog: list[dict],
    max_results: int = 3,
    game: str | None = None,
) -> list[dict]:
    """Entry-only view of :func:`search_catalog_hits` (kept for callers that
    only need the video, e.g. production_plan)."""
    return [entry for entry, _ in search_catalog_hits(fragment, catalog, max_results, game)]


# Franchise and sub-title tokens to drop before querying Wikimedia Commons.
# Commons indexes the real world: "Assassin's Creed Unity Louis XVI execution"
# finds nothing, "Louis XVI execution" finds the paintings.
_FRANCHISE_TOKENS = {
    "assassin", "assassin's", "assassins", "creed", "ac",
    "unity", "syndicate", "rogue", "brotherhood", "revelations", "odyssey",
    "origins", "valhalla", "shadows", "chronicles", "black", "flag",
    "iii", "ii", "iv", "the", "fall", "of", "and",
}


def _commons_query(query: str) -> str:
    """Strip franchise branding, leaving the real-world topic (if any).

    Bare numerals go too — they are sequel numbers ("Assassin's Creed 3"), and
    on Commons a stray digit only adds noise.
    """
    kept = [w for w in re.findall(r"[\w'-]+", query)
            if w.lower() not in _FRANCHISE_TOKENS and not w.isdigit()]
    return " ".join(kept)


def _query_entities(query: str) -> set[str]:
    """Proper nouns a curated query names — characters, places, events."""
    return {w for w in re.findall(r"[A-Z][\w'-]+", query)
            if w.lower() not in _FRANCHISE_TOKENS and len(w) > 2}


def _corroborates(entities: set[str], entry: dict, hit) -> bool:
    """Does this hit mention any entity the query asked for?

    Semantic search always returns the catalog's *closest* footage, even when
    the catalog holds nothing about the subject: a paragraph on Alexander II
    lands on "the player stands near a wooden structure in Havana" with a
    perfectly respectable z-score. The z-score cannot catch it — the query is
    on-franchise, only the episode is missing — so corroborate by name instead.
    """
    if not entities:
        return True
    haystack = " ".join([
        str(entry.get("title", "")),
        str(entry.get("original_name", "")),
        " ".join(entry.get("tags") or []),
        getattr(hit, "text", "") or "",
    ]).lower()
    return any(e.lower() in haystack for e in entities)


def _illustration_block(query: str, per_source: int) -> list[str]:
    """Fandom game art + Wikimedia Commons historical stills for one fragment.

    Two different jobs. Fandom carries the franchise's own renders and concept
    art — right for lore beats with no footage, but publisher-copyrighted, so
    it is editorial/commentary material, not cleared stock. Commons carries the
    actual paintings and portraits behind the history the narration retells,
    mostly public domain.
    """
    lines: list[str] = ["### Illustrations", ""]

    art = image_search.fandom_images_for_query(query, limit=per_source)
    lines.append(f"**Game art** — Fandom AC wiki · `{query}` · publisher-licensed, editorial use")
    lines.append("")
    if art:
        for j, img in enumerate(art, 1):
            lines.append(f"{j}. {img['title']} ({img['width']}x{img['height']})"
                         f" — from *{img.get('article', '?')}*")
            lines.append(f"   - {img['file_url']}")
    else:
        lines.append("*No wiki art found*")
    lines.append("")

    hist_query = _commons_query(query)
    lines.append(f"**Historical** — Wikimedia Commons · `{hist_query or '—'}`")
    lines.append("")
    hist = image_search.commons_search(hist_query, limit=per_source) if hist_query else []
    if hist:
        for j, img in enumerate(hist, 1):
            lines.append(f"{j}. {img['title']} ({img['width']}x{img['height']})"
                         f" — {img['license']}")
            lines.append(f"   - {img['file_url']}")
    else:
        lines.append("*No Commons results* — the de-branded query may name lore, not history")
    lines.append("")
    return lines


def _fmt_duration(sec: int) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _detect_game(filename: str) -> str | None:
    name_lower = filename.lower()
    for key, cfg in config.GAMES.items():
        display = cfg.get("display_name", "").lower()
        if display and display in name_lower:
            return key
        if key.replace("_", " ") in name_lower:
            return key
    return None


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def list_projects() -> list[str]:
    """List available project folders."""
    if not config.PROJECTS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in config.PROJECTS_DIR.iterdir()
        if p.is_dir() and (p / "text.md").exists()
    )


def read_project_text(project: str) -> str:
    """Read the source text from a project folder."""
    text_path = config.project_path(project) / "text.md"
    if not text_path.exists():
        raise FileNotFoundError(f"No text.md in project: {project}")
    raw = text_path.read_text(encoding="utf-8")
    return _strip_frontmatter(raw)


def load_query_overrides(project: str) -> dict[int, str]:
    """Read hand-curated search queries from ``queries.json`` in the project.

    Auto-extracted queries degrade badly on transliterated narration: the
    entities are spelled phonetically in Cyrillic ("Пандемик Хорд") and the
    footage is indexed under their English names. The override maps a fragment
    number to the query that actually finds the footage.
    """
    path = config.project_path(project) / "queries.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items() if v}


def generate_report(
    text: str,
    source_name: str,
    game: str | list[str] | None = None,
    yt_per_fragment: int = 3,
    local_per_fragment: int = 3,
    min_words: int = 40,
    query_overrides: dict[int, str] | None = None,
    images_per_source: int = 0,
) -> tuple[str, dict]:
    """Returns the storyboard markdown and the machine-readable moments data.

    The moments mirror the "My videos" sections: for every fragment, the
    scored candidates with their timecodes. build_storyboard consumes this
    (moments.json) so the clip that gets cut is the moment chosen here — the
    decision used to live only in the human-readable markdown and was lost.
    """
    fragments = split_into_paragraphs(text, min_words=min_words)
    if not fragments:
        return "No fragments found in text.", {"fragments": []}

    query_overrides = query_overrides or {}

    games = _as_games(game)
    catalog = _load_catalog(games) if games else []
    catalog_label = f", catalog: {len(catalog)} videos" if catalog else ""
    if len(games) > 1:
        catalog_label += f" ({', '.join(games)})"

    index_info = embed_index.index_stats() if catalog else {"exists": False}
    if index_info.get("exists"):
        catalog_label += (
            f", semantic index: {index_info['documents']} docs ({index_info['model']})"
        )
    elif catalog:
        catalog_label += ", semantic index: none (keyword fallback)"

    lines = [
        f"# Storyboard: {source_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Fragments: {len(fragments)}{catalog_label}",
        "",
        "---",
        "",
    ]
    moment_fragments: list[dict] = []

    for i, fragment in enumerate(fragments, 1):
        query = query_overrides.get(i) or extract_search_query(fragment)
        print(f"[Storyboard] Fragment {i}/{len(fragments)}: query=[{ascii(query)}]")
        lines.append(f"## Fragment {i}")
        lines.append("")
        lines.append("> " + " ".join(fragment.split()))
        lines.append("")
        lines.append(f"**Search query:** `{query}`")
        lines.append("")

        if catalog:
            # Built once and handed to the search below, so the queries printed
            # here are provably the ones that produced the hits.
            shots = shot_query.build_shot_queries(
                fragment, game=games[0] if games else None
            )
            if shots and shots[0].source == "llm":
                lines.append(
                    "**Shot queries:** " + " · ".join(f"`{s.text}`" for s in shots)
                )
            elif not shots:
                lines.append("**Shot queries:** *none — nothing filmable in this fragment*")
            lines.append("")
            matches = search_catalog_hits(
                fragment, catalog, max_results=local_per_fragment, game=games,
                shots=shots,
            )
            moment_fragments.append({
                "fragment": i,
                "text_preview": fragment[:60] + "...",
                "candidates": _moment_candidates(matches),
            })
            lines.append("### My videos")
            lines.append("")
            entities = _query_entities(query)
            if matches and not any(_corroborates(entities, e, h) for e, h in matches):
                lines.append(
                    f"> **Unconfirmed** — nothing here mentions "
                    f"{', '.join(sorted(entities))}. The catalog most likely has no "
                    f"footage of this episode and these are the nearest generic "
                    f"scenes. Prefer the illustrations below."
                )
                lines.append("")
            if matches:
                for j, (entry, hit) in enumerate(matches, 1):
                    dur = _fmt_duration(entry.get("duration_sec", 0))
                    lines.append(f"{j}. **{entry.get('title', 'Untitled')}** ({dur})")
                    lines.append(f"   - File: `{entry.get('original_name', '?')}`")
                    if hit is not None:
                        # The moment, not just the file — this is what goes on the timeline.
                        flag = "" if hit.timing_ok or hit.start_sec is None else " (start only — VLM interval invalid)"
                        lines.append(
                            f"   - Moment: **{hit.timecode}**{flag} "
                            f"({hit.level}{'/' + hit.scene_type if hit.scene_type else ''}, "
                            f"z={hit.score:.2f})"
                        )
                        if hit.query:
                            lines.append(f"   - Found via shot query: `{hit.query}`")
                        lines.append(f"   - Matched: {hit.text[:200]}")
                    else:
                        lines.append(
                            "   - Moment: *keyword match — video only, no moment identified*"
                        )
                    tags = entry.get("tags", [])
                    if tags:
                        lines.append(f"   - Tags: {', '.join(tags[:8])}")
                    lines.append("")
            else:
                lines.append("*No matching local videos*")
                lines.append("")

        lines.append("### YouTube videos")
        lines.append("")
        if query:
            yt_results = search_youtube(query, max_results=yt_per_fragment)
            if yt_results:
                for j, v in enumerate(yt_results, 1):
                    views = format_view_count(v["view_count"])
                    lines.append(f"{j}. **{v['title']}**")
                    lines.append(
                        f"   - Channel: {v['channel']} | Duration: {v['duration']} | Views: {views}"
                    )
                    lines.append(f"   - URL: {v['url']}")
                    lines.append("")
            else:
                lines.append("*No YouTube results found*")
                lines.append("")
        else:
            lines.append("*Could not extract search query*")
            lines.append("")

        if images_per_source and query:
            lines.extend(_illustration_block(query, images_per_source))

        lines.append("---")
        lines.append("")

    moments = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source_name,
        "fragments": moment_fragments,
    }
    return "\n".join(lines), moments


def _moment_candidates(matches: list[tuple[dict, embed_index.Hit | None]]) -> list[dict]:
    """Machine view of one fragment's "My videos" section, best first.

    Keyword-only matches keep null timing — build_storyboard must not treat
    "the right video, no moment" as a cuttable range.
    """
    candidates = []
    for entry, hit in matches:
        c = {
            "original_name": entry.get("original_name", ""),
            "game": (hit.game if hit else entry.get("game", "")) or "",
            "duration_sec": entry.get("duration_sec"),
            "match": "semantic" if hit is not None else "keyword",
            "start_sec": hit.start_sec if hit else None,
            "end_sec": hit.end_sec if hit else None,
        }
        if hit is not None:
            c.update({
                "level": hit.level,
                "scene_type": hit.scene_type,
                "zscore": round(hit.score, 2),
                "timing_ok": hit.timing_ok,
                "shot_query": hit.query,
            })
        candidates.append(c)
    return candidates


def save_report(report_text: str, project: str, moments: dict | None = None) -> Path:
    project_dir = config.get_project_dir(project)
    report_path = project_dir / "storyboard.md"
    report_path.write_text(report_text, encoding="utf-8")
    if moments is not None:
        moments_path = project_dir / "moments.json"
        moments_path.write_text(
            json.dumps(moments, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Storyboard: search the catalog and YouTube for every narration fragment "
                    "of projects/<project>/text.md -> storyboard.md + moments.json")
    ap.add_argument("project", nargs="?", help="project folder under projects/ (omit to list projects)")
    ap.add_argument("game", nargs="?", help="game key from GAMES, several as \"a,b\" "
                    "(default: detected from the project path)")
    ap.add_argument("--images", type=int, nargs="?", const=3, default=0, metavar="N",
                    help="also N still images per source (Fandom wiki, Wikimedia Commons); default N=3")
    args = ap.parse_args()

    if not args.project:
        projects = list_projects()
        if projects:
            print("Available projects:")
            for p in projects:
                print(f"  - {p}")
        else:
            print(f"No projects in {config.PROJECTS_DIR}")
        sys.exit(0)

    images = args.images
    project = args.project
    game = args.game or _detect_game(project)
    text = read_project_text(project)
    print(f"Processing: {project} ({len(text.split())} words)")
    if game:
        print(f"Game(s): {', '.join(_as_games(game))}")
    if images:
        print(f"Illustrations: {images} per source (Fandom wiki + Wikimedia Commons)")

    overrides = load_query_overrides(project)
    if overrides:
        print(f"Query overrides: {len(overrides)} fragments (queries.json)")

    report, moments = generate_report(text, project, game=game,
                                      query_overrides=overrides,
                                      images_per_source=images)
    path = save_report(report, project, moments=moments)
    print(f"Storyboard saved: {path}")
    if moments["fragments"]:
        print(f"Moments saved: {path.parent / 'moments.json'} "
              f"({len(moments['fragments'])} fragments)")

"""
Production Plan generator.

Analyzes source text, classifies each fragment by shot type, estimates
duration, searches local catalog + YouTube + stock footage, and produces
a structured plan with gap analysis and shooting priorities.

Usage:
    python -m clipmaker.production_plan <project> [game]
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from . import config
from .youtube_search import split_into_paragraphs, extract_search_query, search_youtube
from .report import _load_catalog, search_catalog, _detect_game, read_project_text, list_projects
from .stock_search import search_stock
from .shot_classifier import classify_fragment, extract_stock_query, SHOT_TYPES


def _estimate_duration_sec(text: str) -> float:
    word_count = len(text.split())
    return word_count / config.WORDS_PER_MIN_RU * 60


def _score_local_match(fragment: str, entry: dict) -> float:
    frag_tokens = {w.lower() for w in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9']{3,}", fragment)}
    entry_text = " ".join([
        entry.get("title", ""),
        entry.get("summary", ""),
        " ".join(entry.get("tags", [])),
    ])
    entry_tokens = {w.lower() for w in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9']{3,}", entry_text)}
    if not frag_tokens:
        return 0.0
    overlap = frag_tokens & entry_tokens
    return min(len(overlap) / max(len(frag_tokens) * 0.3, 1), 1.0)


def _score_youtube_match(query: str, video: dict) -> float:
    query_tokens = {w.lower() for w in query.split() if len(w) > 2}
    title_tokens = {w.lower() for w in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9']{3,}", video.get("title", ""))}
    if not query_tokens:
        return 0.0
    overlap = query_tokens & title_tokens
    base = len(overlap) / len(query_tokens)
    views = video.get("view_count", 0)
    view_bonus = min(views / 1_000_000, 0.2)
    return min(base + view_bonus, 1.0)


def _recommend_source(local_matches: list, yt_matches: list, stock_matches: list,
                      shot_types: list[tuple[str, float]]) -> tuple[str, str]:
    """Determine best source and confidence level."""
    top_type = shot_types[0][0] if shot_types else "gameplay"

    if top_type == "talking_head":
        return "shoot", "high"
    if top_type == "text_overlay":
        return "composite", "high"
    if top_type == "comparison":
        return "composite", "medium"

    best_local = local_matches[0]["_score"] if local_matches else 0
    best_yt = yt_matches[0]["_score"] if yt_matches else 0
    best_stock = stock_matches[0].get("_score", 0.3) if stock_matches else 0

    if best_local >= 0.5:
        confidence = "high" if best_local >= 0.7 else "medium"
        return "local", confidence
    if best_yt >= 0.5:
        return "youtube", "medium"
    if top_type in ("broll_abstract", "broll_gaming") and stock_matches:
        return "stock", "medium"
    if best_local >= 0.3:
        return "local", "low"
    if best_yt >= 0.3:
        return "youtube", "low"
    if stock_matches:
        return "stock", "low"
    return "shoot", "low"


def generate_plan(
    text: str,
    source_name: str,
    game: str | None = None,
) -> dict:
    """Generate a full production plan from source text."""
    fragments = split_into_paragraphs(text, min_words=30)
    if not fragments:
        return {"error": "No fragments found"}

    catalog = _load_catalog(game) if game else []
    plan_fragments = []
    total_duration = 0.0
    coverage_counts = {"local": 0, "youtube": 0, "stock": 0, "shoot": 0, "composite": 0}

    for i, frag_text in enumerate(fragments, 1):
        duration = _estimate_duration_sec(frag_text)
        total_duration += duration
        shot_types = classify_fragment(frag_text, game=game)

        print(f"[Plan] Fragment {i}/{len(fragments)}: {frag_text[:50]}...")

        local_matches = []
        if catalog:
            raw = search_catalog(frag_text, catalog, max_results=config.PLAN_LOCAL_RESULTS)
            for entry in raw:
                entry["_score"] = _score_local_match(frag_text, entry)
                local_matches.append(entry)

        query = extract_search_query(frag_text)
        yt_matches = []
        if query:
            yt_raw = search_youtube(query, max_results=config.PLAN_YT_RESULTS)
            for v in yt_raw:
                v["_score"] = _score_youtube_match(query, v)
                yt_matches.append(v)

        stock_query = extract_stock_query(frag_text, game=game)
        stock_matches = search_stock(stock_query, max_results=config.PLAN_STOCK_RESULTS)

        recommendation, confidence = _recommend_source(
            local_matches, yt_matches, stock_matches, shot_types
        )
        coverage_counts[recommendation] += 1

        fragment_data = {
            "index": i,
            "text": frag_text,
            "text_preview": frag_text[:80] + ("..." if len(frag_text) > 80 else ""),
            "duration_sec": round(duration, 1),
            "shot_types": [(t, round(s, 2)) for t, s in shot_types[:3]],
            "sources": {
                "local": [
                    {
                        "file": e.get("original_name", "?"),
                        "title": e.get("title", "?"),
                        "score": round(e["_score"], 2),
                        "duration_sec": e.get("duration_sec", 0),
                    }
                    for e in local_matches[:3]
                ],
                "youtube": [
                    {
                        "title": v["title"],
                        "url": v["url"],
                        "score": round(v["_score"], 2),
                        "duration_sec": v.get("duration_sec", 0),
                    }
                    for v in sorted(yt_matches, key=lambda x: x["_score"], reverse=True)[:3]
                ],
                "stock": [
                    {
                        "title": s["title"],
                        "url": s["url"],
                        "source": s["source"],
                        "duration_sec": s.get("duration_sec", 0),
                    }
                    for s in stock_matches[:3]
                ],
            },
            "recommendation": recommendation,
            "confidence": confidence,
        }
        plan_fragments.append(fragment_data)

    total = len(fragments)
    gaps = []
    shooting_list = []

    for frag in plan_fragments:
        if frag["recommendation"] == "shoot":
            shooting_list.append({
                "fragment_index": frag["index"],
                "shot_type": frag["shot_types"][0][0] if frag["shot_types"] else "unknown",
                "description": frag["text_preview"],
                "estimated_duration_sec": frag["duration_sec"],
            })
        if frag["confidence"] == "low":
            gaps.append({
                "fragment_index": frag["index"],
                "recommendation": frag["recommendation"],
                "shot_types": [t for t, _ in frag["shot_types"][:2]],
                "text_preview": frag["text_preview"],
                "priority": "high" if frag["recommendation"] == "shoot" else "medium",
            })

    plan = {
        "source_file": source_name,
        "game": game,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_fragments": total,
        "total_duration_sec": round(total_duration, 1),
        "coverage": {
            "local_catalog": round(coverage_counts["local"] / total, 2) if total else 0,
            "youtube": round(coverage_counts["youtube"] / total, 2) if total else 0,
            "stock": round(coverage_counts["stock"] / total, 2) if total else 0,
            "needs_shooting": round(coverage_counts["shoot"] / total, 2) if total else 0,
            "composite": round(coverage_counts["composite"] / total, 2) if total else 0,
        },
        "fragments": plan_fragments,
        "gaps": gaps,
        "shooting_list": shooting_list,
    }
    return plan


def render_plan_markdown(plan: dict) -> str:
    """Render production plan as a human-readable markdown report."""
    lines = []
    cov = plan["coverage"]
    total_min = plan["total_duration_sec"] / 60

    lines.append(f"# Production Plan: {plan['source_file']}")
    lines.append(f"Generated: {plan['generated_at']}")
    if plan["game"]:
        lines.append(f"Game: {plan['game']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Fragments | {plan['total_fragments']} |")
    lines.append(f"| Total duration | {total_min:.1f} min |")
    lines.append(f"| Covered by local catalog | {cov['local_catalog']:.0%} |")
    lines.append(f"| Covered by YouTube | {cov['youtube']:.0%} |")
    lines.append(f"| Covered by stock footage | {cov['stock']:.0%} |")
    lines.append(f"| Needs shooting/compositing | {cov['needs_shooting'] + cov['composite']:.0%} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Fragment Breakdown")
    lines.append("")

    for frag in plan["fragments"]:
        rec = frag["recommendation"]
        conf = frag["confidence"]
        types_str = ", ".join(f"{t} ({s:.0%})" for t, s in frag["shot_types"][:2])
        icon = {"high": "+", "medium": "~", "low": "!"}[conf]

        lines.append(f"### [{icon}] Fragment {frag['index']} — {rec} ({conf})")
        lines.append("")
        lines.append(f"> {frag['text_preview']}")
        lines.append("")
        lines.append(f"- **Duration:** {frag['duration_sec']:.0f}s")
        lines.append(f"- **Shot types:** {types_str}")
        lines.append(f"- **Recommendation:** `{rec}` (confidence: {conf})")

        if frag["sources"]["local"]:
            lines.append(f"- **Local matches:**")
            for m in frag["sources"]["local"]:
                lines.append(f"  - {m['title']} (score: {m['score']:.2f}, file: `{m['file']}`)")

        if frag["sources"]["youtube"]:
            lines.append(f"- **YouTube:**")
            for v in frag["sources"]["youtube"]:
                lines.append(f"  - {v['title']} (score: {v['score']:.2f}) — {v['url']}")

        if frag["sources"]["stock"]:
            lines.append(f"- **Stock footage:**")
            for s in frag["sources"]["stock"]:
                lines.append(f"  - [{s['source']}] {s['title']} ({s['duration_sec']}s) — {s['url']}")

        lines.append("")

    if plan["gaps"]:
        lines.append("---")
        lines.append("")
        lines.append("## Gap Analysis")
        lines.append("")
        lines.append("| # | Fragment | Priority | Issue |")
        lines.append("|---|----------|----------|-------|")
        for gap in plan["gaps"]:
            lines.append(
                f"| {gap['fragment_index']} | {gap['text_preview'][:40]}... "
                f"| **{gap['priority']}** | {gap['recommendation']}, "
                f"types: {', '.join(gap['shot_types'])} |"
            )
        lines.append("")

    if plan["shooting_list"]:
        lines.append("---")
        lines.append("")
        lines.append("## Shooting List")
        lines.append("")
        total_shoot = sum(s["estimated_duration_sec"] for s in plan["shooting_list"])
        lines.append(f"Total footage to record: **{total_shoot:.0f}s** ({total_shoot/60:.1f} min)")
        lines.append("")
        for item in plan["shooting_list"]:
            shot_desc = SHOT_TYPES.get(item["shot_type"], item["shot_type"])
            lines.append(f"- **Frag {item['fragment_index']}** ({item['estimated_duration_sec']:.0f}s): "
                         f"{shot_desc}")
            lines.append(f"  > {item['description']}")
            lines.append("")

    return "\n".join(lines)


def save_plan(plan: dict, project: str) -> tuple[Path, Path]:
    """Save plan as both JSON and Markdown into the project folder."""
    project_dir = config.get_project_dir(project)

    json_path = project_dir / "plan.json"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = project_dir / "plan.md"
    md_path.write_text(render_plan_markdown(plan), encoding="utf-8")

    return json_path, md_path


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Production plan for projects/<project>/text.md: shot type and duration per "
                    "fragment, coverage from the catalog, YouTube and stock, a shooting list")
    ap.add_argument("project", nargs="?", help="project folder under projects/ (omit to list projects)")
    ap.add_argument("game", nargs="?", help="game key from GAMES (default: detected from the project path)")
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

    project = args.project
    game = args.game or _detect_game(project)
    text = read_project_text(project)

    print(f"Generating production plan: {project} ({len(text.split())} words)")
    if game:
        print(f"Game: {game}")
    print()

    plan = generate_plan(text, project, game=game)
    json_path, md_path = save_plan(plan, project)

    print()
    print(f"Plan saved:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    cov = plan["coverage"]
    print()
    print(f"Coverage: local {cov['local_catalog']:.0%} | "
          f"YouTube {cov['youtube']:.0%} | "
          f"stock {cov['stock']:.0%} | "
          f"needs work {cov['needs_shooting'] + cov['composite']:.0%}")
    print(f"Gaps: {len(plan['gaps'])} | Shooting list: {len(plan['shooting_list'])} items")

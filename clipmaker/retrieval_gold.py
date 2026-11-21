"""
Gold set for the retrieval eval: queries → the clips actually placed.

Two source shapes:

  shots_curation  shots.json (picture queries written BEFORE searching, per
                  fragment) + curation.json (the verified pick per fragment).
                  The canonical storyboard flow; queries are unbiased.
  events          vegas_events.json / footage_events.json, where each event's
                  label describes what is in frame. The label is the query —
                  written AFTER the pick, in Russian, so it is optimistic for
                  wording and pessimistic for the English-only lexical layers.
                  Events with the same label in one video form one item with
                  several acceptable picks.

A reviewer confirms the pairs in eval/retrieval_review.md; apply_review() carries
the marks back into the YAML. Rebuilding the gold keeps confirmed statuses.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path

import yaml

from .retrieval_eval import LAYERS, load_entries, name_key

SOURCES_PATH = Path(__file__).resolve().parent.parent / "eval" / "gold_sources.yaml"


def load_sources(path: Path = SOURCES_PATH) -> list[dict]:
    """The finished videos the gold set is built from, as a YAML list of
    {id, title, game, kind: shots_curation | events, project, events?}; none when
    the file is absent (eval/ is your own)."""
    if not path.exists():
        return []
    return list(yaml.safe_load(path.read_text(encoding="utf-8")) or [])


PREFIXES = ("Assassin's Creed IV Black Flag Resynced ", "Assassin's Creed™ Black Flag Resynced")
WRONG_MARKS = {"x", "х", "✗", "-", "wrong", "нет"}
OK_MARKS = {"", "ok", "ок", "✓", "+"}


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def clean_label(label: str) -> str:
    """Event label → picture query: drop fragment no., spoken quote, tags, notes."""
    s = re.sub(r"^\s*фр\s*\d+\s*", "", label or "")
    s = re.sub(r"\[[^\]]*\]", "", s)                       # [личное], [иллюстрация: …]
    s = re.sub(r"\s*—\s*«[^»]*»", "", s)                   # — «what the narrator says»
    s = re.sub(r"\([^)]*(?:нет в записях|\d+:\d+)[^)]*\)", "", s)  # editor notes, timecodes
    return re.sub(r"\s+", " ", s).strip(" —-,;")


def label_fragment(label: str) -> int | None:
    m = re.match(r"\s*фр\s*(\d+)", label or "")
    return int(m.group(1)) if m else None


def _pick(name: str, start: float, end: float, known: set[str]) -> dict:
    base = re.split(r"[\\/]", name)[-1]
    return {"file": base, "start": round(float(start), 2), "end": round(float(end), 2),
            "in_catalog": name_key(base) in known}


def items_from_shots(video: str, shots: dict, curation: dict, known: set[str]) -> list[dict]:
    items = []
    for frag in sorted((k for k in curation if k.isdigit()), key=int):
        name, start, end = curation[frag][:3]
        items.append({
            "id": f"{video}-f{int(frag):02d}", "video": video, "fragment": int(frag),
            "queries": list(shots.get(frag, [])), "query_source": "shots.json",
            "picks": [_pick(name, start, end, known)], "status": "unreviewed",
        })
    return items


def items_from_events(video: str, events: list[dict], known: set[str]) -> list[dict]:
    groups: dict[str, dict] = {}
    for idx, ev in enumerate(events, 1):
        query = clean_label(ev.get("label", ""))
        if not query:
            continue
        span = float(ev["end_sec"]) - float(ev["start_sec"])
        pick = _pick(ev["path"], ev["clip_start"], float(ev["clip_start"]) + span, known)
        item = groups.setdefault(query.lower(), {
            "id": f"{video}-e{idx:03d}", "video": video, "fragment": label_fragment(ev.get("label", "")),
            "queries": [query], "query_source": "label", "picks": [], "status": "unreviewed",
        })
        item["picks"].append(pick)
    return list(groups.values())


def _git_ref(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%h %D"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_gold(projects_dir: Path, catalog_dir: Path) -> dict:
    videos, items = [], []
    for src in load_sources():
        project = Path(projects_dir) / src["project"]
        known = {name_key(e["original_name"]) for e in load_entries(catalog_dir, src["game"])}
        if src["kind"] == "shots_curation":
            shots = json.loads((project / "shots.json").read_text(encoding="utf-8"))
            curation = json.loads((project / "curation.json").read_text(encoding="utf-8"))
            built = items_from_shots(src["id"], shots, curation, known)
        else:
            events = json.loads((project / src["events"]).read_text(encoding="utf-8"))
            built = items_from_events(src["id"], events, known)
        items.extend({"id": it["id"], "video": it["video"], "game": src["game"], **it} for it in built)
        videos.append({k: src[k] for k in ("id", "title", "game", "project")}
                      | {"query_source": "shots.json" if src["kind"] == "shots_curation" else "label"})
    return {
        "version": 1,
        "built": date.today().isoformat(),
        "projects_ref": _git_ref(Path(projects_dir)),
        "videos": videos,
        "items": items,
    }


def _pick_sig(item: dict) -> list[tuple]:
    return [(p["file"], p["start"], p["end"]) for p in item.get("picks", [])]


def merge_reviews(new: dict, old: dict | None) -> dict:
    """Keep the reviewer's verdicts (and edited queries) from a previous gold file.

    Event ids are positional, so a verdict is carried over only when the item
    still has the same picks — an edited events file never moves a verdict onto
    a different clip.
    """
    if not old:
        return new
    previous = {it["id"]: it for it in old.get("items", [])}
    items = []
    for item in new["items"]:
        prev = previous.get(item["id"])
        if prev and prev.get("status") in ("ok", "wrong") and _pick_sig(prev) == _pick_sig(item):
            item = item | {"status": prev["status"]}
            if prev.get("query_source", "").endswith("+owner"):
                item = item | {"queries": prev["queries"], "query_source": prev["query_source"]}
        items.append(item)
    return new | {"items": items}


def load_gold(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_gold(path: Path, gold: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = ("# Retrieval gold set — built by `python -m clipmaker.retrieval_eval gold`.\n"
              "# status: unreviewed | ok | wrong (set from eval/retrieval_review.md by apply-review).\n"
              "# picks[].in_catalog: false = the source video is not in the VLM catalog; ignored by recall.\n")
    body = yaml.safe_dump(gold, allow_unicode=True, sort_keys=False, width=200)
    Path(path).write_text(header + body, encoding="utf-8")


# --------------------------------------------------------------------------
# review sheet
# --------------------------------------------------------------------------

def _tc(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def _short(name: str) -> str:
    for prefix in PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return re.sub(r"\.mp4$", "", name, flags=re.I).strip(" _") or name


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


LAYER_LETTER = {"semantic": "S", "ocr": "O", "scenes": "C"}


def _found(ranks: dict | None) -> str:
    if not ranks:
        return ""
    parts = []
    for layer in (lay for lay in LAYERS if lay in ranks):   # a layer that did not run is not "·"
        r, letter = ranks[layer], LAYER_LETTER[layer]
        if r.get("moment") and r["moment"] <= 10:
            parts.append(f"{letter}{r['moment']}")
        elif r.get("file") and r["file"] <= 10:
            parts.append(f"{letter}f{r['file']}")
        else:
            parts.append(f"{letter}·")
    return " ".join(parts)


def render_sheet(gold: dict, results: dict | None = None) -> str:
    ranks = {r["id"]: r["ranks"] for r in (results or {}).get("items", [])}
    items = [it for it in gold["items"] if any(p["in_catalog"] for p in it["picks"])]
    skipped = len(gold["items"]) - len(items)
    lines = [
        "# Gold-набор поиска по каталогу — проверка",
        "",
        f"Строк: {len(items)}, ~30 мин. Вопрос к каждой строке один: **описывает ли запрос то, что в кадре "
        "пика?** (а не то, что говорит диктор). Начните с раздела 1 — только там запросы писались до поиска.",
        "",
        "- Колонка «Отметка»: пусто = OK; `x` = неверно (строка уйдёт из оценки); "
        "любой другой текст = исправленный запрос.",
        "- Закончили раздел — поставьте `[x]` в его строке «Раздел просмотрен». Пустые отметки "
        "в непросмотренном разделе не считаются подтверждёнными.",
        "- «Найдено»: S — смысловой поиск, O — OCR, C — сцены VLM; цифра — место момента в выдаче, "
        "`f3` — нашёлся только файл (3-м), `·` — нет в топ-10.",
        "- Потом: `python -m clipmaker.retrieval_eval apply-review` → `run` (см. `eval/retrieval_report.md`).",
        f"- Не в листе: запросы, чьи пики целиком вне VLM-каталога (старые ролики, чужие записи) — {skipped}.",
        "",
    ]
    by_video = {v["id"]: v for v in gold["videos"]}
    for n, vid in enumerate(by_video, 1):
        rows = [it for it in items if it["video"] == vid]
        video = by_video[vid]
        src = "запросы из `shots.json`, написаны до поиска" if video["query_source"] == "shots.json" \
            else "запросы = подписи событий монтажа (после подбора, по-русски)"
        lines += [f"## {n}. {video['title']} · `{vid}`", "",
                  f"Строк: {len(rows)}; {src}.", "", "- [ ] Раздел просмотрен", "",
                  "| id | фр | Запрос | Пик: файл @ диапазон | Найдено | Отметка |",
                  "|---|---|---|---|---|---|"]
        for it in rows:
            picks = [p for p in it["picks"] if p["in_catalog"]]
            pick_txt = "; ".join(f"{_short(p['file'])} @ {_tc(p['start'])}–{_tc(p['end'])}" for p in picks[:3])
            if len(picks) > 3:
                pick_txt += f"; +{len(picks) - 3}"
            mark = "x" if it.get("status") == "wrong" else ""
            lines.append(f"| {it['id']} | {it.get('fragment') or ''} | {_cell(' / '.join(it['queries']))} "
                         f"| {_cell(pick_txt)} | {_found(ranks.get(it['id']))} | {mark} |")
        lines.append("")
    return "\n".join(lines)


def _split_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip())
    return [c.strip().replace("\\|", "|") for c in cells[1:-1]]


def parse_sheet(text: str) -> dict[str, tuple[str, bool]]:
    """id → (mark, section reviewed?) for every item row of the sheet."""
    marks: dict[str, tuple[str, bool]] = {}
    section: list[str] = []
    reviewed = False
    for line in text.splitlines() + ["## end"]:
        if line.startswith("## "):
            for item_id, mark in section:
                marks[item_id] = (mark, reviewed)
            section, reviewed = [], False
        elif re.match(r"^\s*-\s*\[[xXхХ]\]\s*Раздел просмотрен", line):
            reviewed = True
        elif line.startswith("|"):
            cells = _split_row(line)
            if len(cells) >= 6 and re.match(r"^[a-z0-9-]+-[fe]\d+$", cells[0]):
                section.append((cells[0], cells[-1]))
    return marks


def apply_review(gold: dict, sheet_text: str) -> tuple[dict, dict[str, int]]:
    """Return a new gold dict with the reviewer's marks applied, plus counts."""
    marks = parse_sheet(sheet_text)
    counts = {"ok": 0, "wrong": 0, "edited": 0, "cleared": 0, "untouched": 0}
    items = []
    for item in gold["items"]:
        mark, reviewed = marks.get(item["id"], ("", False))
        low = mark.lower()
        if low in WRONG_MARKS:
            item, key = item | {"status": "wrong"}, "wrong"
        elif low in OK_MARKS and reviewed:
            item, key = item | {"status": "ok"}, "ok"
        elif low in OK_MARKS:
            # the sheet pre-fills `x` for wrong rows; a cleared `x` is a retraction
            cleared = item.get("status") == "wrong" and item["id"] in marks
            item, key = (item | {"status": "unreviewed"}, "cleared") if cleared else (item, "untouched")
        else:
            src = item["query_source"].removesuffix("+owner") + "+owner"
            item, key = item | {"status": "ok", "queries": [mark], "query_source": src}, "edited"
        counts[key] += 1
        items.append(item)
    return gold | {"items": items}, counts

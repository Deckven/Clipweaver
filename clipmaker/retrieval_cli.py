"""CLI for the retrieval eval — entry point is `python -m clipmaker.retrieval_eval`."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from . import config
from . import retrieval_eval as rev
from . import retrieval_gold as rg
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

DIAGNOSTIC = "semantic_nocut"   # semantic without z and cosine floors — what the floors cost


def _require_gold(path: str) -> dict:
    gold = rg.load_gold(Path(path))
    if not gold:
        raise SystemExit(f"No gold set at {path} — build it first: python -m clipmaker.retrieval_eval gold")
    return gold


def _cmd_gold(args) -> None:
    gold = rg.build_gold(Path(args.projects_dir), Path(args.catalog_dir))
    gold = rg.merge_reviews(gold, rg.load_gold(Path(args.gold)))
    rg.save_gold(Path(args.gold), gold)
    picks = [p for it in gold["items"] for p in it["picks"]]
    print(f"[gold] {len(gold['items'])} items, {len(picks)} picks "
          f"({sum(p['in_catalog'] for p in picks)} in catalog) -> {args.gold}")
    for v in gold["videos"]:
        its = [it for it in gold["items"] if it["video"] == v["id"]]
        inside = sum(any(p["in_catalog"] for p in it["picks"]) for it in its)
        print(f"  {v['id']}: {len(its)} items, {inside} with an in-catalog pick")


def _index_freshness(catalog_dir: Path, game: str) -> dict[str, int]:
    """How many of the game's catalog files the semantic index has not embedded as-is."""
    from . import embed_index

    index = embed_index.index_files()
    manifest_path = index.manifest if index else None
    files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {}) \
        if manifest_path and manifest_path.exists() else {}
    root = catalog_dir / config.GAMES[game]["catalog_subdir"]
    stale = missing = 0
    for path in root.rglob("*.json"):
        rel = str(path.relative_to(catalog_dir)).replace("\\", "/")
        if rel not in files:
            missing += 1
        elif files[rel] != embed_index._fingerprint(path):
            stale += 1
    return {"missing": missing, "stale": stale}


def _index_info() -> str:
    from . import embed_index

    index = embed_index.index_files()
    if index is None:
        return "?, собран ?"
    model = json.loads(index.manifest.read_text(encoding="utf-8")).get("model", "?") \
        if index.manifest.exists() else "?"
    built = datetime.fromtimestamp(index.vectors.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{model}, собран {built}"


def _layers_for(game: str, catalog_dir: Path, semantic: bool) -> dict:
    entries = rev.load_entries(catalog_dir, game)
    layers: dict = {"ocr": rev.ocr_layer(entries), "scenes": rev.scenes_layer(entries)}
    if semantic:
        layers["semantic"] = rev.SemanticLayer(game)
        layers[DIAGNOSTIC] = rev.SemanticLayer(game, uncut=True)
    log.info("[run] %s: %d videos, ocr docs %d, scene docs %d",
             game, len(entries), len(layers["ocr"]), len(layers["scenes"]))
    return layers


def _cmd_run(args) -> None:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch a GPU busy with indexing
    config.EMBED_DEVICE = args.device
    if args.index_dir:
        config.EMBED_INDEX_DIR = Path(args.index_dir)
    catalog_dir = Path(args.catalog_dir)

    gold = _require_gold(args.gold)
    items = [it for it in gold["items"] if rev.evaluable(it)]
    semantic = not args.no_semantic
    if semantic:
        from . import embed_index

        if embed_index.load_index() is None:
            raise SystemExit(f"No semantic index in {config.EMBED_INDEX_DIR} — pass --index-dir "
                             "or --no-semantic; an empty index would read as 0% recall.")
    rows: list[dict] = []
    freshness: dict[str, dict[str, int]] = {}
    for game in dict.fromkeys(it["game"] for it in items):
        layers = _layers_for(game, catalog_dir, semantic)
        if semantic:
            freshness[game] = _index_freshness(catalog_dir, game)
        rows.extend({"id": it["id"], "video": it["video"],
                     "ranks": rev.evaluate_item(it, layers, tol=args.tolerance)}
                    for it in items if it["game"] == game)
    if semantic and rows and all(r["ranks"]["semantic"]["file"] is None for r in rows):
        log.warning("[run] the semantic layer found no pick file for any query - check the index")
    main_layers = rev.LAYERS if semantic else ("ocr", "scenes")
    results = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "gold": str(args.gold), "catalog_dir": str(catalog_dir),
        "index_dir": str(config.EMBED_INDEX_DIR) if semantic else None,
        "index_info": _index_info() if semantic else None,
        "device": args.device,
        "semantic": "computed" if semantic else "pending",
        "index_freshness": freshness, "tolerance_sec": args.tolerance,
        "summary": rev.summarize(rows, main_layers),
        "diagnostic": [r for r in rev.summarize(rows, (DIAGNOSTIC,)) if r["layer"] == DIAGNOSTIC]
        if semantic else [],
        "attribution_moment_top10": rev.attribution(rows, main_layers),
        "items": rows,
    }
    Path(args.results).write_text(_dump_results(results), encoding="utf-8")
    Path(args.report).write_text(render_report(gold, results), encoding="utf-8")
    print(f"[run] {len(rows)} items evaluated -> {args.results}, {args.report}")
    for r in results["summary"]:
        if r["video"] == "all" and r["granularity"] == "moment":
            print(f"  {r['layer']:9s} moment R@1 {r['r@1']:.2f} R@5 {r['r@5']:.2f} R@10 {r['r@10']:.2f}")


def _dump_results(results: dict) -> str:
    """Pretty header, one line per summary row and per item — readable diffs."""
    head = {k: v for k, v in results.items() if k not in ("summary", "diagnostic", "items")}
    body = json.dumps(head, ensure_ascii=False, indent=1)[:-2]
    parts = [body]
    for key in ("summary", "diagnostic", "items"):
        rows = ",\n  ".join(json.dumps(r, ensure_ascii=False) for r in results[key])
        parts.append(f',\n "{key}": [\n  {rows}\n ]' if rows else f',\n "{key}": []')
    return "".join(parts) + "\n}\n"


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def render_report(gold: dict, results: dict) -> str:
    """retrieval_report.md: recall@k per layer, per video and in total."""
    titles = {v["id"]: v["title"] for v in gold["videos"]}
    sources = {v["id"]: v["query_source"] for v in gold["videos"]}
    reviewed = sum(it.get("status") in ("ok", "wrong") for it in gold["items"])
    lines = [
        "# Retrieval eval — recall@k по слоям каталога", "",
        f"Прогон: {results['run_at']}. Gold: `{results['gold']}` (собран {gold['built']}, "
        f"проекты `{gold['projects_ref']}`), проверено владельцем {reviewed} из {len(gold['items'])}.",
        f"Каталог: `{results['catalog_dir']}`. Смысловой слой: {results['semantic']}"
        + (f" ({results['index_info']}, запросы на {results['device']})" if results.get("index_info") else "")
        + f". Допуск по времени ±{results['tolerance_sec']:g} с.", "",
        "Слои: **semantic** — `embed_index.search` (E5, отсечка z ≥ 3), **ocr** — BM25 по OCR-таймлайну, "
        "**scenes** — BM25 по описаниям сцен VLM, **any** — хотя бы один слой. "
        "**moment** — в топ-k есть момент, пересекающий диапазон пика; **file** — в топ-k видео есть файл пика.",
        "",
    ]
    if results.get("index_freshness"):
        fresh = ", ".join(f"{g}: нет {f['missing']}, устарело {f['stale']}"
                          for g, f in results["index_freshness"].items())
        lines += [f"Свежесть индекса относительно каталога — {fresh}.", ""]
    for gran in ("moment", "file"):
        lines += [f"## {gran}", "", "| Видео | Запросы | n | Слой | R@1 | R@3 | R@5 | R@10 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in results["summary"] + results["diagnostic"]:
            if r["granularity"] != gran:
                continue
            video = "все три" if r["video"] == "all" else titles.get(r["video"], r["video"])
            src = "" if r["video"] == "all" else sources.get(r["video"], "")
            lines.append(f"| {video} | {src} | {r['n']} | {r['layer']} | {_pct(r['r@1'])} | "
                         f"{_pct(r['r@3'])} | {_pct(r['r@5'])} | {_pct(r['r@10'])} |")
        lines.append("")
    lines += ["## Кто нашёл (moment, топ-10)", "", "| Слои | Пиков |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in results["attribution_moment_top10"].items()]
    lines += ["", f"`{DIAGNOSTIC}` — диагностика: смысловой поиск без порогов z и cos, показывает, "
              "сколько находок срезают пороги.", ""]
    lines += ["## Как читать", ""]
    for v in gold["videos"]:
        its = [it for it in gold["items"] if it["video"] == v["id"]]
        picks = [p for it in its for p in it["picks"]]
        out = sum(not p["in_catalog"] for p in picks)
        lines.append(f"- {v['title']}: запросов {len(its)}, пиков {len(picks)}, из них вне каталога {out}.")
    lines += [
        "- Пик в gold один на запрос (то, что встало в ролик). Другой годный кадр в выдаче считается "
        "промахом — recall здесь нижняя граница полезности слоя.",
        "- `label`-запросы — подписи событий монтажа: написаны после подбора и по-русски. ocr и scenes "
        "англоязычные и ловят только латиницу (имена, пункты меню); честное сравнение слоёв — на `shots.json`.",
        "- Строки со статусом `wrong` (после проверки владельцем) в оценку не входят.",
        "",
    ]
    return "\n".join(lines)


def _cmd_sheet(args) -> None:
    path = Path(args.sheet)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists — it may hold the reviewer's marks. "
                         "Run apply-review first, then sheet --force.")
    gold = _require_gold(args.gold)
    results = json.loads(Path(args.results).read_text(encoding="utf-8")) \
        if Path(args.results).exists() else None
    path.write_text(rg.render_sheet(gold, results), encoding="utf-8")
    print(f"[sheet] -> {path}")


def _cmd_apply(args) -> None:
    gold = _require_gold(args.gold)
    new, counts = rg.apply_review(gold, Path(args.sheet).read_text(encoding="utf-8"))
    rg.save_gold(Path(args.gold), new)
    print(f"[apply-review] {counts} -> {args.gold}")


def main(argv: list[str] | None = None) -> None:
    """CLI: build the gold set, run the eval, write and apply the review sheet."""
    p = argparse.ArgumentParser(prog="python -m clipmaker.retrieval_eval",
                                description="Recall@k of catalog search layers against a gold set")
    p.add_argument("--gold", default=str(rev.GOLD_PATH))
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gold", help="build eval/retrieval_gold.yaml from project files")
    g.add_argument("--projects-dir", default=str(config.PROJECTS_DIR))
    g.add_argument("--catalog-dir", default=str(config.CATALOG_DIR))

    r = sub.add_parser("run", help="score every layer, write results + report")
    r.add_argument("--catalog-dir", default=str(config.CATALOG_DIR))
    r.add_argument("--index-dir", default=None, help="semantic index (default config.EMBED_INDEX_DIR)")
    r.add_argument("--device", default="cpu", help="query embedding device (default cpu)")
    r.add_argument("--no-semantic", action="store_true", help="lexical layers only")
    r.add_argument("--tolerance", type=float, default=rev.TOLERANCE_SEC)
    r.add_argument("--results", default=str(rev.RESULTS_PATH))
    r.add_argument("--report", default=str(rev.REPORT_PATH))

    s = sub.add_parser("sheet", help="write the review sheet for confirming pairs")
    s.add_argument("--sheet", default=str(rev.SHEET_PATH))
    s.add_argument("--results", default=str(rev.RESULTS_PATH))
    s.add_argument("--force", action="store_true")

    a = sub.add_parser("apply-review", help="carry the sheet's marks into the gold YAML")
    a.add_argument("sheet", nargs="?", default=str(rev.SHEET_PATH))

    add_verbose_flag(p)
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    {"gold": _cmd_gold, "run": _cmd_run, "sheet": _cmd_sheet, "apply-review": _cmd_apply}[args.command](args)


if __name__ == "__main__":
    main()

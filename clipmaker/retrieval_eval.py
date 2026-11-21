"""
Retrieval evaluation: recall@k of each catalog search layer against a gold set.

The gold set (eval/retrieval_gold.yaml, built by clipmaker.retrieval_gold) holds
picture queries from finished videos and the clips that were actually placed.
Each layer gets the query, returns a ranked list of moments, and we record where
the placed clip shows up.

Layers
  semantic  embed_index.search — E5 over video / scene / window documents, with
            the production z-score cutoff (config.EMBED_MIN_ZSCORE).
  ocr       BM25 over timeline OCR captions (one entry per frame step, 5 s on the
            fine-indexed footage). The ranked form of the per-project
            tools/ocrgrep.py regex greps, which have no ranking of their own.
  scenes    BM25 over VLM scene descriptions (start/end per scene).

Granularity
  file      the pick's video is among the first k distinct videos returned.
  moment    one of the first k moments overlaps the pick range (± tolerance).

recall@k here is the hit rate: an item counts as found when ANY of its queries
puts ANY of its picks within the top k. That is how the queries are used — every
picture description of a fragment is searched, the editor takes the best hit.

CLI (run from the repo root; see eval/README section in the review sheet):
    python -m clipmaker.retrieval_eval gold  --projects-dir P --catalog-dir C
    python -m clipmaker.retrieval_eval run   --catalog-dir C --index-dir I
    python -m clipmaker.retrieval_eval sheet [--force]
    python -m clipmaker.retrieval_eval apply-review [SHEET]
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import config
from .embed_index import clean_caption, parse_timestamp

EVAL_DIR = config.ROOT_DIR / "eval"
GOLD_PATH = EVAL_DIR / "retrieval_gold.yaml"
RESULTS_PATH = EVAL_DIR / "retrieval_results.json"
REPORT_PATH = EVAL_DIR / "retrieval_report.md"
SHEET_PATH = EVAL_DIR / "retrieval_review.md"

KS = (1, 3, 5, 10)
TOLERANCE_SEC = 5.0      # one timeline step; VLM scene times drift by about that
DEFAULT_SPAN_SEC = 5.0   # length assumed for a moment with no usable end
FETCH = 300              # moments pulled per query; grown until 10 distinct files show up
MAX_FETCH = 20000
LAYERS = ("semantic", "ocr", "scenes")


# --------------------------------------------------------------------------
# names and moments
# --------------------------------------------------------------------------

def name_key(name: str) -> str:
    """Comparable video name: basename, curly quotes and ™ folded, case-free.

    Picks carry full Windows paths with a typographic apostrophe in the folder
    ("Assassin’s Creed IV …"); the catalog stores original_name only.
    """
    base = re.split(r"[\\/]", name or "")[-1]
    base = base.replace("™", "").replace("’", "'")
    base = unicodedata.normalize("NFKC", base)
    return re.sub(r"\s+", " ", base).strip().lower()


@dataclass(frozen=True)
class Moment:
    """One ranked result. start is None for whole-video documents."""

    name: str
    start: float | None
    end: float | None
    score: float


@dataclass(frozen=True)
class Pick:
    """A clip range that was placed in the finished video."""

    name: str
    start: float
    end: float


class Layer(Protocol):
    def search(self, query: str, n: int) -> list[Moment]: ...


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------

def _moment_end(m: Moment) -> float:
    if m.end is not None and m.start is not None and m.end > m.start:
        return m.end
    return (m.start or 0.0) + DEFAULT_SPAN_SEC


def overlaps(m: Moment, pick: Pick, tol: float = TOLERANCE_SEC) -> bool:
    if m.start is None or name_key(m.name) != name_key(pick.name):
        return False
    return m.start <= pick.end + tol and _moment_end(m) >= pick.start - tol


def moment_rank(moments: list[Moment], pick: Pick, tol: float = TOLERANCE_SEC) -> int | None:
    """1-based rank of the first moment overlapping the pick, else None."""
    for i, m in enumerate(moments, 1):
        if overlaps(m, pick, tol):
            return i
    return None


def file_rank(moments: list[Moment], pick: Pick) -> int | None:
    """1-based rank of the pick's video among the distinct videos returned."""
    seen: list[str] = []
    want = name_key(pick.name)
    for m in moments:
        key = name_key(m.name)
        if key in seen:
            continue
        seen.append(key)
        if key == want:
            return len(seen)
    return None


def best_rank(ranks: list[int | None]) -> int | None:
    found = [r for r in ranks if r is not None]
    return min(found) if found else None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


# --------------------------------------------------------------------------
# lexical layers
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"[0-9a-zа-я]+")
_STOP = frozenset(
    "a an the of on in at to for with and or by from is are be as it its this that into "
    "over under up out off you your their his her he she they we our not no all some any "
    "и в во на с со к по о об от до за из у а но или не ни что как это то же ли для при "
    "под над про без через фр".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stop words dropped, English plural -s folded."""
    out = []
    for tok in _TOKEN.findall((text or "").lower().replace("ё", "е")):
        if len(tok) < 2 or tok in _STOP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class Bm25Layer:
    """BM25 over short timed documents of one game's catalog."""

    def __init__(self, docs: list[tuple[Moment, str]], k1: float = 1.2, b: float = 0.75):
        self._moments = [m for m, _ in docs]
        self._k1, self._b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        for idx, (_, text) in enumerate(docs):
            tokens = tokenize(text)
            self._lengths.append(len(tokens))
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                self._postings.setdefault(tok, []).append((idx, tf))
        n = len(docs)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        self._idf = {
            tok: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for tok, p in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self._moments)

    def search(self, query: str, n: int) -> list[Moment]:
        scores: dict[int, float] = {}
        k1, b, avgdl = self._k1, self._b, self._avgdl or 1.0
        for tok in set(tokenize(query)):
            idf = self._idf.get(tok)
            if idf is None:
                continue
            for idx, tf in self._postings[tok]:
                norm = tf + k1 * (1 - b + b * self._lengths[idx] / avgdl)
                scores[idx] = scores.get(idx, 0.0) + idf * tf * (k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        return [
            Moment(self._moments[i].name, self._moments[i].start, self._moments[i].end, round(s, 3))
            for i, s in ranked
        ]


def load_entries(catalog_dir: Path, game: str) -> list[dict]:
    """Every catalog JSON of one game, read-only."""
    root = Path(catalog_dir) / config.GAMES[game]["catalog_subdir"]
    entries = []
    for path in sorted(root.rglob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(entry, dict) and entry.get("original_name"):
            entries.append(entry)
    return entries


def ocr_layer(entries: list[dict]) -> Bm25Layer:
    docs = []
    for entry in entries:
        for frame in entry.get("timeline") or []:
            text = f"{clean_caption(frame.get('caption', ''))} {frame.get('ocr', '')}".strip()
            if not text:
                continue
            moment = Moment(entry["original_name"], parse_timestamp(frame.get("start")),
                            parse_timestamp(frame.get("end")), 0.0)
            docs.append((moment, text))
    return Bm25Layer(docs)


def scenes_layer(entries: list[dict]) -> Bm25Layer:
    docs = []
    for entry in entries:
        for scene in entry.get("scenes") or []:
            text = (scene.get("description") or "").strip()
            if not text:
                continue
            moment = Moment(entry["original_name"], parse_timestamp(scene.get("start")),
                            parse_timestamp(scene.get("end")), 0.0)
            docs.append((moment, text))
    return Bm25Layer(docs)


# --------------------------------------------------------------------------
# semantic layer
# --------------------------------------------------------------------------

class SemanticLayer:
    """embed_index.search as the storyboard uses it, minus the per-video collapse.

    The collapse is undone so moment-level recall can see every scene/window of
    a video; file-level recall dedupes the list again (see ranked_moments), which
    reproduces the collapsed ranking.

    ``uncut=True`` drops both production floors (z-score and cosine) — the
    diagnostic that shows what the thresholds cost.
    """

    def __init__(self, game: str, uncut: bool = False):
        self.game = game
        self.uncut = uncut

    def search(self, query: str, n: int) -> list[Moment]:
        from . import embed_index

        floor = config.EMBED_MIN_COSINE
        if self.uncut:
            config.EMBED_MIN_COSINE = -1.0
        try:
            hits = embed_index.search(query, k=n, game=self.game,
                                      min_zscore=-1e9 if self.uncut else None,
                                      per_video_best=False)
        finally:
            config.EMBED_MIN_COSINE = floor
        return [Moment(h.original_name, h.start_sec, h.end_sec, h.score) for h in hits]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def item_picks(item: dict) -> list[Pick]:
    """The item's picks that the catalog can possibly return."""
    return [Pick(p["file"], float(p["start"]), float(p["end"]))
            for p in item.get("picks", []) if p.get("in_catalog", True)]


def evaluable(item: dict) -> bool:
    return item.get("status") != "wrong" and bool(item_picks(item)) and bool(item.get("queries"))


def ranked_moments(layer: Layer, query: str, fetch: int = FETCH,
                   need_files: int = max(KS)) -> list[Moment]:
    """Pull moments until the list covers ``need_files`` distinct videos or runs dry.

    One video can own hundreds of scene/window documents, so a fixed cap could
    stop before the 10th distinct file and understate file-level recall.
    """
    n = fetch
    while True:
        moments = layer.search(query, n)
        files = {name_key(m.name) for m in moments}
        if len(moments) < n or len(files) >= need_files or n >= MAX_FETCH:
            return moments
        n = min(n * 4, MAX_FETCH)


def evaluate_item(item: dict, layers: dict[str, Layer],
                  tol: float = TOLERANCE_SEC, fetch: int = FETCH) -> dict[str, dict]:
    """Best file / moment rank of the item in each layer, plus which query won."""
    picks = item_picks(item)
    out: dict[str, dict] = {}
    for name, layer in layers.items():
        best: dict = {"file": None, "moment": None, "query": None}
        for query in item["queries"]:
            moments = ranked_moments(layer, query, fetch)
            f = best_rank([file_rank(moments, p) for p in picks])
            m = best_rank([moment_rank(moments, p, tol) for p in picks])
            if m is not None and (best["moment"] is None or m < best["moment"]):
                best["moment"], best["query"] = m, query
            best["file"] = best_rank([f, best["file"]])
        out[name] = best
    return out


def summarize(rows: list[dict], layers: tuple[str, ...], ks: tuple[int, ...] = KS) -> list[dict]:
    """recall@k per video (and 'all') × layer (and 'any') × granularity."""
    videos = ["all", *dict.fromkeys(r["video"] for r in rows)]
    table = []
    for video in videos:
        subset = rows if video == "all" else [r for r in rows if r["video"] == video]
        for layer in (*layers, "any"):
            for gran in ("moment", "file"):
                ranks = [
                    best_rank([r["ranks"][lay][gran] for lay in layers]) if layer == "any"
                    else r["ranks"][layer][gran]
                    for r in subset
                ]
                table.append({
                    "video": video, "layer": layer, "granularity": gran, "n": len(ranks),
                    **{f"r@{k}": round(recall_at_k(ranks, k), 3) for k in ks},
                })
    return table


def attribution(rows: list[dict], layers: tuple[str, ...], k: int = 10) -> dict[str, int]:
    """Which layer(s) found each item at moment level within top k."""
    counts: dict[str, int] = {}
    for r in rows:
        found = [lay for lay in layers if (r["ranks"][lay]["moment"] or k + 1) <= k]
        key = "+".join(found) if found else "none"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    from .retrieval_cli import main

    main()

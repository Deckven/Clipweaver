"""
Semantic index over the VLM catalog.

Replaces token-overlap matching (report.search_catalog) with dense retrieval,
so a Russian narration fragment can find English scene descriptions without
sharing a single literal word.

Three document levels, because catalog coverage is uneven:

  video   title + summary + tags. Always available. Says *what* the video is,
          nothing about *when* — same granularity the old keyword search had.
  scene   scenes[].description. Clean prose written by the VLM, carries
          start/end. Only present for videos indexed at interval <= 10s
          (27 of 309 files at the time of writing).
  window  timeline[] captions merged EMBED_WINDOW_FRAMES at a time. OCR-heavy
          and noisy, but it is the only per-time signal the other 282 files
          have. Built *only* when a video has no scenes, so scene-covered
          videos are never diluted with OCR garbage.

Storage is a single .npz of float32 vectors plus a parallel docs.json. At
catalog scale (~4k documents) brute-force cosine is sub-10 ms, so there is no
vector database here on purpose — a server-backed store would add a heavy
dependency and buy nothing.

CLI:
    python -m clipmaker.embed_index build [--game assassins_creed_4] [--force]
    python -m clipmaker.embed_index search "query" [--game G] [-k 8] [--level scene]
    python -m clipmaker.embed_index stats
"""
from __future__ import annotations

import hashlib
import os
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config

_VECTORS_FILE = "vectors.npz"
_DOCS_FILE = "docs.json"
_MANIFEST_FILE = "manifest.json"

_model = None
_index_cache: tuple[int, "Index"] | None = None


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

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


@dataclass(frozen=True)
class Hit:
    """A scored document, carrying enough to cut the clip.

    ``score`` is the level-weighted z-score against the query's own similarity
    distribution — that is what ranking and rejection use. ``cosine`` is kept
    for inspection only; on its own it does not separate a real match from
    nonsense (see config.EMBED_MIN_ZSCORE).
    """

    score: float
    cosine: float
    level: str
    game: str
    catalog_path: str
    original_name: str
    source_path: str
    title: str
    text: str
    start_sec: float | None
    end_sec: float | None
    scene_type: str | None
    query: str = ""  # the shot description that found it, when one was used

    @property
    def timing_ok(self) -> bool:
        """False when the VLM emitted a scene interval that runs backwards."""
        return (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec > self.start_sec
        )

    @property
    def timecode(self) -> str:
        if self.start_sec is None:
            return "full video"
        if not self.timing_ok:
            # VLM scene timestamps drift and sometimes decrease; do not print an
            # interval that would cut a negative-length clip.
            return f"{fmt_time(self.start_sec)}~"
        return f"{fmt_time(self.start_sec)}-{fmt_time(self.end_sec)}"


def fmt_time(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_timestamp(value: str | int | float | None) -> float | None:
    """Parse "mm:ss" / "hh:mm:ss" / seconds into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if not all(p.strip().isdigit() for p in parts if p.strip()):
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


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


def _window_documents(entry: dict, base: dict) -> list[Document]:
    timeline = entry.get("timeline") or []
    if not timeline:
        return []
    title = entry.get("title", "")
    docs = []
    for i, chunk in enumerate(_group_timeline(timeline)):
        captions = [clean_caption(f.get("caption", "")) for f in chunk]
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
    # Windows only where scenes are missing — see module docstring.
    if not scene_docs:
        docs.extend(_window_documents(entry, base))
    return docs


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

def _resolve_device() -> str:
    if config.EMBED_DEVICE:
        return config.EMBED_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        _model = SentenceTransformer(config.EMBED_MODEL, device=_resolve_device())
    return _model


def _needs_e5_prefix() -> bool:
    return "e5" in config.EMBED_MODEL.lower()


def embed_texts(texts: list[str], kind: str) -> np.ndarray:
    """Embed a batch. ``kind`` is "query" or "passage" (E5 requires the prefix)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    prepared = [f"{kind}: {t}" for t in texts] if _needs_e5_prefix() else texts
    vectors = _get_model().encode(
        prepared,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(prepared) > 256,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Index:
    vectors: np.ndarray
    docs: list[dict]
    manifest: dict[str, str]

    def __len__(self) -> int:
        return len(self.docs)


def _index_dir() -> Path:
    return config.EMBED_INDEX_DIR


def load_index() -> Index | None:
    """Load the index, cached by file mtime.

    A storyboard run calls this once per fragment; re-reading the .npz each
    time dominated the search cost.
    """
    global _index_cache
    d = _index_dir()
    vec_path, docs_path = d / _VECTORS_FILE, d / _DOCS_FILE
    if not (vec_path.exists() and docs_path.exists()):
        return None

    stamp = vec_path.stat().st_mtime_ns
    if _index_cache is not None and _index_cache[0] == stamp:
        return _index_cache[1]

    try:
        vectors = np.load(vec_path)["vectors"]
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        manifest_path = d / _MANIFEST_FILE
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists() else {}
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if len(vectors) != len(docs):
        return None
    index = Index(vectors=vectors, docs=docs, manifest=manifest)
    _index_cache = (stamp, index)
    return index


def _save_index(index: Index) -> None:
    """Write the three index files so a crash never leaves them out of step:
    each goes to a temporary name first and is swapped in with os.replace,
    the manifest last."""
    d = _index_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp_vectors = d / ("tmp-" + _VECTORS_FILE)
    np.savez_compressed(tmp_vectors, vectors=index.vectors)
    tmp_docs = d / ("tmp-" + _DOCS_FILE)
    tmp_docs.write_text(json.dumps(index.docs, ensure_ascii=False), encoding="utf-8")
    tmp_manifest = d / ("tmp-" + _MANIFEST_FILE)
    tmp_manifest.write_text(
        json.dumps({
            "model": config.EMBED_MODEL,
            "window_frames": config.EMBED_WINDOW_FRAMES,
            "files": index.manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_vectors, d / _VECTORS_FILE)
    os.replace(tmp_docs, d / _DOCS_FILE)
    os.replace(tmp_manifest, d / _MANIFEST_FILE)


def _fingerprint(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def _catalog_files(game: str | None = None) -> list[Path]:
    root = config.CATALOG_DIR
    if not root.exists():
        return []
    if game:
        cfg = config.GAMES.get(game)
        if not cfg:
            return []
        root = root / cfg["catalog_subdir"]
        if not root.exists():
            return []
    return sorted(p for p in root.rglob("*.json") if _index_dir() not in p.parents)


def _read_entry(path: Path) -> dict | None:
    try:
        # utf-8-sig: analyzer writes a BOM on Windows
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildStats:
    files_seen: int
    files_embedded: int
    files_unchanged: int
    files_dropped: int
    documents: int
    by_level: dict[str, int]


def build_index(game: str | None = None, force: bool = False) -> BuildStats:
    """Build or incrementally refresh the index.

    Files whose content hash is unchanged keep their existing vectors; only
    new, edited and deleted catalog entries are touched.

    Refuses (and leaves the index as it is) when the game is unknown, when no
    catalog file is found — an unplugged footage drive looks exactly like an
    empty catalog — and, unless ``force``, when the build would drop more than
    half of the files indexed before.
    """
    if game is not None and game not in config.GAMES:
        raise ValueError(f"unknown game {game!r}; known: {', '.join(sorted(config.GAMES))}")
    files = _catalog_files(game)
    if not files:
        where = config.CATALOG_DIR / config.GAMES[game]["catalog_subdir"] if game else config.CATALOG_DIR
        raise RuntimeError(f"no catalog files under {where} — is the folder (or its drive) "
                           f"available? The index was left as it is.")
    previous = load_index()
    prev_manifest = dict(previous.manifest.get("files", {})) if previous else {}
    if previous and previous.manifest.get("model") != config.EMBED_MODEL:
        force = True  # vectors from another model are not comparable
        prev_manifest = {}

    fingerprints = {}
    stale: list[Path] = []
    for path in files:
        rel = str(path.relative_to(config.CATALOG_DIR)).replace("\\", "/")
        try:
            fp = _fingerprint(path)
        except OSError:
            continue
        fingerprints[rel] = fp
        if force or prev_manifest.get(rel) != fp:
            stale.append(path)

    # A partial (per-game) build must not drop documents of other games.
    scope = set(fingerprints)
    stale_paths = {
        str(p.relative_to(config.CATALOG_DIR)).replace("\\", "/") for p in stale
    }
    kept_docs, kept_vectors = [], []
    if previous and not force:
        for doc, vec in zip(previous.docs, previous.vectors):
            cp = doc.get("catalog_path", "")
            if cp in stale_paths:
                continue  # re-embedded below
            if game is not None and cp not in scope:
                kept_docs.append(doc)  # belongs to another game, untouched
                kept_vectors.append(vec)
            elif cp in fingerprints:
                kept_docs.append(doc)  # unchanged
                kept_vectors.append(vec)
            # otherwise the catalog file is gone -> its documents die with it

    new_docs: list[Document] = []
    for path in stale:
        entry = _read_entry(path)
        if not entry:
            continue
        rel = str(path.relative_to(config.CATALOG_DIR)).replace("\\", "/")
        new_docs.extend(iter_documents(entry, rel))

    print(
        f"[Index] {len(files)} catalog files, {len(stale)} to embed, "
        f"{len(kept_docs)} documents reused"
    )

    new_vectors = embed_texts([d.text for d in new_docs], "passage") if new_docs else None

    if kept_vectors and new_vectors is not None and len(new_vectors):
        vectors = np.vstack([np.array(kept_vectors, dtype=np.float32), new_vectors])
    elif new_vectors is not None and len(new_vectors):
        vectors = new_vectors
    elif kept_vectors:
        vectors = np.array(kept_vectors, dtype=np.float32)
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)

    docs = kept_docs + [asdict(d) for d in new_docs]

    # Manifest keeps entries for games outside this build's scope.
    manifest = {
        rel: fp for rel, fp in prev_manifest.items()
        if game is not None and rel not in scope
    }
    manifest.update(fingerprints)

    if game is None:
        prev_in_scope = set(prev_manifest)
    else:
        subdir = config.GAMES[game]["catalog_subdir"] + "/"
        prev_in_scope = {r for r in prev_manifest if r.startswith(subdir)}
    would_drop = prev_in_scope - scope
    if not force and prev_in_scope and len(would_drop) > len(prev_in_scope) // 2:
        raise RuntimeError(
            f"this build would drop {len(would_drop)} of {len(prev_in_scope)} indexed catalog "
            f"files — is part of the catalog missing? Re-run with --force if they are "
            f"really gone. The index was left as it is.")

    index = Index(vectors=vectors, docs=docs, manifest=manifest)
    _save_index(index)

    by_level: dict[str, int] = {}
    for doc in docs:
        by_level[doc["level"]] = by_level.get(doc["level"], 0) + 1

    dropped = len(would_drop)

    return BuildStats(
        files_seen=len(files),
        files_embedded=len(stale),
        files_unchanged=len(files) - len(stale),
        files_dropped=dropped,
        documents=len(docs),
        by_level=by_level,
    )


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def search(
    query: str,
    k: int = 8,
    game: str | None = None,
    level: str | None = None,
    min_zscore: float | None = None,
    per_video_best: bool = True,
) -> list[Hit]:
    """Rank catalog documents against ``query``.

    Scoring is the z-score of each document's cosine similarity against the
    **whole catalog's** score distribution, times its level weight. Absolute
    cosine is not used for rejection — E5 packs everything into a narrow band,
    so a query about borscht still scores 0.766 against EVE Online footage. The
    z-score is measured against the *same query's* distribution, which is what
    actually separates a match from a coincidence.

    ``game`` and ``level`` filter which documents are returned, never how they
    are scored, so a hit's z is identical with or without a filter.

    ``per_video_best`` collapses a video to its single strongest moment, which
    is what a storyboard fragment wants — ten hits inside one video are one
    answer, not ten.
    """
    query = (query or "").strip()
    if not query:
        return []
    index = load_index()
    if index is None or not len(index):
        return []

    min_zscore = config.EMBED_MIN_ZSCORE if min_zscore is None else min_zscore

    candidates = [
        i for i, doc in enumerate(index.docs)
        if (not game or doc.get("game") == game)
        and (not level or doc.get("level") == level)
    ]
    # z-scores are meaningless on a handful of documents.
    if len(candidates) < 8:
        return []

    q = embed_texts([query], "query")[0]

    # The reference distribution is ALWAYS the whole catalog, never the filtered
    # subset. The z-score works because it measures a hit against a topically
    # diverse background; a single game is on-topic end to end, so its mean sits
    # far higher and nothing clears the bar. Measured on the author's catalog: the same
    # scene at cos=0.811 scored z=3.78 catalog-wide but z=2.44 under
    # --game assassins_creed_4, while the nonsense control "рецепт борща со
    # сметаной" inverted the other way — rejected at 2.72 catalog-wide, accepted
    # at 3.26 filtered. Normalising per filter made the verdict depend on how
    # the query was scoped rather than on the match.
    sims_all = index.vectors @ q
    std = float(sims_all.std())
    if std < 1e-6:
        return []
    mean = float(sims_all.mean())

    weights = config.EMBED_LEVEL_WEIGHTS
    scored: list[tuple[float, float, dict]] = []
    for doc_idx in candidates:
        cosine = float(sims_all[doc_idx])
        if cosine < config.EMBED_MIN_COSINE:
            continue
        doc = index.docs[doc_idx]
        z = (cosine - mean) / std * weights.get(doc.get("level", ""), 1.0)
        if z < min_zscore:
            continue
        scored.append((z, cosine, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    if per_video_best:
        seen: set[str] = set()
        collapsed = []
        for row in scored:
            key = row[2].get("catalog_path", "")
            if key in seen:
                continue
            seen.add(key)
            collapsed.append(row)
        scored = collapsed

    return [
        Hit(
            score=round(z, 3),
            cosine=round(cosine, 4),
            level=doc.get("level", ""),
            game=doc.get("game", ""),
            catalog_path=doc.get("catalog_path", ""),
            original_name=doc.get("original_name", ""),
            source_path=doc.get("source_path", ""),
            title=doc.get("title", ""),
            text=doc.get("text", ""),
            start_sec=doc.get("start_sec"),
            end_sec=doc.get("end_sec"),
            scene_type=doc.get("scene_type"),
        )
        for z, cosine, doc in scored[:k]
    ]


def index_stats() -> dict:
    index = load_index()
    if index is None:
        return {"exists": False}
    by_level: dict[str, int] = {}
    by_game: dict[str, int] = {}
    for doc in index.docs:
        by_level[doc["level"]] = by_level.get(doc["level"], 0) + 1
        by_game[doc.get("game", "?")] = by_game.get(doc.get("game", "?"), 0) + 1
    return {
        "exists": True,
        "documents": len(index.docs),
        "dimensions": int(index.vectors.shape[1]) if index.vectors.size else 0,
        "model": index.manifest.get("model", "?"),
        "files": len(index.manifest.get("files", {})),
        "by_level": by_level,
        "by_game": by_game,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Semantic index over the VLM catalog")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build or refresh the index")
    p_build.add_argument("--game", default=None, help="limit to one game key")
    p_build.add_argument("--force", action="store_true",
                         help="re-embed everything; also allows dropping most of the indexed files")

    p_search = sub.add_parser("search", help="query the index")
    p_search.add_argument("query")
    p_search.add_argument("--game", default=None)
    p_search.add_argument("--level", default=None, choices=["video", "scene", "window"])
    p_search.add_argument("-k", type=int, default=8)
    p_search.add_argument("--min-z", type=float, default=None,
                          help=f"z-score cutoff (default {config.EMBED_MIN_ZSCORE})")
    p_search.add_argument("--all-hits", action="store_true",
                          help="do not collapse to one hit per video")

    sub.add_parser("stats", help="show index contents")

    args = parser.parse_args()

    if args.command == "build":
        try:
            stats = build_index(game=args.game, force=args.force)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"[Index] {exc}")
        print(
            f"[Index] {stats.documents} documents from {stats.files_seen} files "
            f"({stats.files_embedded} embedded, {stats.files_unchanged} unchanged)"
        )
        for lvl, n in sorted(stats.by_level.items()):
            print(f"  {lvl}: {n}")
        print(f"[Index] saved to {_index_dir()}")

    elif args.command == "search":
        hits = search(
            args.query, k=args.k, game=args.game, level=args.level,
            min_zscore=args.min_z, per_video_best=not args.all_hits,
        )
        if not hits:
            print("No matches above threshold.")
            return
        for i, hit in enumerate(hits, 1):
            print(
                f"{i}. z={hit.score:5.2f} cos={hit.cosine:.3f} "
                f"{hit.level:6s} {hit.timecode:>14s}  {hit.title}"
            )
            print(f"   {hit.original_name}")
            print(f"   {hit.text[:160]}")
            print()

    elif args.command == "stats":
        info = index_stats()
        if not info.get("exists"):
            print("No index. Run: python -m clipmaker.embed_index build")
            return
        print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()

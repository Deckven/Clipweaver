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
  window  timeline[] captions (the OCR read off each frame) merged
          EMBED_WINDOW_FRAMES at a time. Noisy, but the only per-time signal
          videos without scenes have. A video with scenes gets only the
          windows whose OCR reads like speech — cutscene subtitles, where the
          character names are that scene descriptions leave out; HUD and menu
          fragments stay out so they never dilute the scenes.

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

import argparse
import hashlib
import json
import logging
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import catalog as _catalog
from . import config
from .catalog_docs import (  # noqa: F401 - re-exported: retrieval_eval and callers use them here
    DOC_RULES,
    Document,
    clean_caption,
    fmt_time,
    iter_documents,
    looks_like_speech,
    parse_timestamp,
)
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

_VECTORS_FILE = "vectors.npz"
_DOCS_FILE = "docs.json"
_MANIFEST_FILE = "manifest.json"

_model = None
_index_cache: tuple[int, Index] | None = None


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
        """The moment as MM:SS-MM:SS, start only for a backwards interval, or 'full video'."""
        if self.start_sec is None:
            return "full video"
        if self.end_sec is None or not self.timing_ok:
            # VLM scene timestamps drift and sometimes decrease; do not print an
            # interval that would cut a negative-length clip.
            return f"{fmt_time(self.start_sec)}~"
        return f"{fmt_time(self.start_sec)}-{fmt_time(self.end_sec)}"


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
    """The loaded index: one vector per document, the documents, and the build manifest."""
    vectors: np.ndarray
    docs: list[dict]
    manifest: dict[str, Any]  # model, document rules, and "files": {path: fingerprint}

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
        with np.load(vec_path) as npz:
            vectors = npz["vectors"]
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        manifest_path = d / _MANIFEST_FILE
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists() else {}
        )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        # A truncated .npz (a crash mid-write before _save_index went atomic, a
        # copy cut short) raises BadZipFile or EOFError, not OSError.
        log.warning("semantic index in %s is unreadable (%s: %s); rebuild it with "
                    "`python -m clipmaker.embed_index build --force`", d, type(exc).__name__, exc)
        return None
    if len(vectors) != len(docs):
        log.warning("semantic index in %s is out of step: %d vectors for %d documents; "
                    "rebuild it with --force", d, len(vectors), len(docs))
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
            "doc_rules": DOC_RULES,
            "files": index.manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_vectors, d / _VECTORS_FILE)
    os.replace(tmp_docs, d / _DOCS_FILE)
    os.replace(tmp_manifest, d / _MANIFEST_FILE)


def _fingerprint(path: Path) -> str:
    return hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()[:16]


def _catalog_files(game: str | None = None) -> list[Path]:
    """The catalog JSON files a build reads: catalog.entry_paths."""
    return _catalog.entry_paths(game)


def _read_entry(path: Path) -> dict | None:
    """A catalog JSON, or None with a warning when it cannot be read."""
    try:
        return _catalog.read_entry(path)
    except (OSError, ValueError) as exc:
        log.warning("[Index] skipped unreadable catalog file %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildStats:
    """What a build_index run did: files seen, embedded, reused and dropped."""
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
    files = _files_to_index(game)
    previous = load_index()
    prev_manifest: dict[str, str] = dict(previous.manifest.get("files", {})) if previous else {}
    model_changed = previous is not None and previous.manifest.get("model") != config.EMBED_MODEL
    if model_changed:
        force = True  # vectors from another model are not comparable
        prev_manifest = {}
    rebuild_docs = _document_rules_changed(previous, game)

    fingerprints, stale = _scan(files, prev_manifest, refresh_all=force or rebuild_docs)
    # A partial (per-game) build must not drop documents of other games.
    scope = set(fingerprints)
    would_drop = _check_drop(prev_manifest, scope, game, force)

    # force re-embeds every file in scope (all of them are stale), but documents of
    # games outside a per-game build are kept: dropping them while the manifest
    # keeps their fingerprints hid those games until the next full --force build.
    kept_docs, kept_vectors = ([], []) if previous is None or model_changed else _kept_documents(
        previous, {_rel(p) for p in stale}, scope, fingerprints, game)
    new_docs = _new_documents(stale)
    log.info("[Index] %d catalog files, %d to embed, %d documents reused",
             len(files), len(stale), len(kept_docs))
    new_vectors = embed_texts([d.text for d in new_docs], "passage") if new_docs else None

    docs = kept_docs + [asdict(d) for d in new_docs]
    # Manifest keeps entries for games outside this build's scope.
    manifest = {rel: fp for rel, fp in prev_manifest.items()
                if game is not None and rel not in scope}
    manifest.update(fingerprints)
    _save_index(Index(vectors=_stack(kept_vectors, new_vectors), docs=docs, manifest=manifest))

    by_level: dict[str, int] = {}
    for doc in docs:
        by_level[doc["level"]] = by_level.get(doc["level"], 0) + 1
    return BuildStats(
        files_seen=len(files),
        files_embedded=len(stale),
        files_unchanged=len(files) - len(stale),
        files_dropped=len(would_drop),
        documents=len(docs),
        by_level=by_level,
    )


def _rel(path: Path) -> str:
    """Catalog-relative path with forward slashes: the key of manifest and docs."""
    return str(path.relative_to(config.CATALOG_DIR)).replace("\\", "/")


def _files_to_index(game: str | None) -> list[Path]:
    """The catalog files in scope; refuses an unknown game and an empty catalog."""
    if game is not None and game not in config.GAMES:
        raise ValueError(f"unknown game {game!r}; known: {', '.join(sorted(config.GAMES))}")
    files = _catalog_files(game)
    if not files:
        where = config.CATALOG_DIR / config.GAMES[game]["catalog_subdir"] if game else config.CATALOG_DIR
        raise RuntimeError(f"no catalog files under {where} — is the folder (or its drive) "
                           f"available? The index was left as it is.")
    return files


def _document_rules_changed(previous: Index | None, game: str | None) -> bool:
    """True when the index was built under other document rules, so every file
    must be re-read; a per-game build cannot do that and is refused."""
    changed = previous is not None and (
        previous.manifest.get("doc_rules", 1) != DOC_RULES
        or previous.manifest.get("window_frames") != config.EMBED_WINDOW_FRAMES)
    if changed and game is not None:
        raise ValueError("the index was built under other document rules (DOC_RULES or "
                         "EMBED_WINDOW_FRAMES changed); run a full build without --game. "
                         "The index was left as it is.")
    return changed


def _scan(files: list[Path], prev_manifest: dict[str, str],
          refresh_all: bool) -> tuple[dict[str, str], list[Path]]:
    """(fingerprint per catalog file, files to embed again)."""
    fingerprints: dict[str, str] = {}
    stale: list[Path] = []
    for path in files:
        rel = _rel(path)
        try:
            fp = _fingerprint(path)
        except OSError as exc:
            log.warning("[Index] skipped unreadable catalog file %s: %s", path, exc)
            continue
        fingerprints[rel] = fp
        if refresh_all or prev_manifest.get(rel) != fp:
            stale.append(path)
    return fingerprints, stale


def _check_drop(prev_manifest: dict[str, str], scope: set[str], game: str | None,
                force: bool) -> set[str]:
    """Indexed files this build would drop; refuses losing over half without force.

    Runs before any embedding, so a refused build costs no model time.
    """
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
    return would_drop


def _kept_documents(previous: Index, stale_paths: set[str], scope: set[str],
                    fingerprints: dict[str, str], game: str | None) -> tuple[list[dict], list]:
    """Documents and vectors of the previous index that this build reuses."""
    kept_docs, kept_vectors = [], []
    for doc, vec in zip(previous.docs, previous.vectors, strict=True):
        cp = doc.get("catalog_path", "")
        if cp in stale_paths:
            continue  # re-embedded
        if (game is not None and cp not in scope) or cp in fingerprints:
            kept_docs.append(doc)  # another game's, untouched - or unchanged
            kept_vectors.append(vec)
        # otherwise the catalog file is gone -> its documents die with it
    return kept_docs, kept_vectors


def _new_documents(stale: list[Path]) -> list[Document]:
    """Documents of every catalog file that has to be embedded again."""
    docs: list[Document] = []
    for path in stale:
        entry = _read_entry(path)
        if entry:
            docs.extend(iter_documents(entry, _rel(path)))
    return docs


def _stack(kept_vectors: list, new_vectors: np.ndarray | None) -> np.ndarray:
    """Reused vectors followed by the new ones, as one float32 matrix."""
    parts = []
    if kept_vectors:
        parts.append(np.array(kept_vectors, dtype=np.float32))
    if new_vectors is not None and len(new_vectors):
        parts.append(new_vectors)
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return parts[0] if len(parts) == 1 else np.vstack(parts)


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
        log.debug("search %r: no semantic index", query)
        return []

    min_zscore = config.EMBED_MIN_ZSCORE if min_zscore is None else min_zscore

    candidates = [
        i for i, doc in enumerate(index.docs)
        if (not game or doc.get("game") == game)
        and (not level or doc.get("level") == level)
    ]
    # z-scores are meaningless on a handful of documents.
    if len(candidates) < 8:
        log.debug("search %r: %d candidate documents (game=%s, level=%s), fewer than 8",
                  query, len(candidates), game, level)
        return []

    q = embed_texts([query], "query")[0]
    scored = _scored(index, q, candidates, min_zscore)
    if not scored:
        log.debug("search %r: nothing above z=%.2f and cosine %.2f",
                  query, min_zscore, config.EMBED_MIN_COSINE)
    if per_video_best:
        scored = _best_per_video(scored)
    return [_hit(z, cosine, doc) for z, cosine, doc in scored[:k]]


def _scored(index: Index, q: np.ndarray, candidates: list[int],
            min_zscore: float) -> list[tuple[float, float, dict]]:
    """(weighted z, cosine, doc) of every candidate that clears both bars, best first.

    The reference distribution is ALWAYS the whole catalog, never the filtered
    subset. The z-score works because it measures a hit against a topically
    diverse background; a single game is on-topic end to end, so its mean sits
    far higher and nothing clears the bar. Measured on the author's catalog: the same
    scene at cos=0.811 scored z=3.78 catalog-wide but z=2.44 under
    --game assassins_creed_4, while the nonsense control "рецепт борща со
    сметаной" inverted the other way — rejected at 2.72 catalog-wide, accepted
    at 3.26 filtered. Normalising per filter made the verdict depend on how
    the query was scoped rather than on the match.
    """
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
        if z >= min_zscore:
            scored.append((z, cosine, doc))
    return sorted(scored, key=lambda x: x[0], reverse=True)


def _best_per_video(scored: list[tuple[float, float, dict]]) -> list[tuple[float, float, dict]]:
    """Keep each catalog file's strongest row (rows come best first)."""
    seen: set[str] = set()
    collapsed = []
    for row in scored:
        key = row[2].get("catalog_path", "")
        if key not in seen:
            seen.add(key)
            collapsed.append(row)
    return collapsed


def _hit(z: float, cosine: float, doc: dict) -> Hit:
    """A stored document as a search result."""
    return Hit(
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


def index_stats() -> dict:
    """Size, model and per-level/per-game document counts of the index on disk."""
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

def main(argv: list[str] | None = None) -> None:
    """CLI: build, search or describe the semantic index."""
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

    add_verbose_flag(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.command == "build":
        try:
            stats = build_index(game=args.game, force=args.force)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"[Index] {exc}") from exc
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
    main()

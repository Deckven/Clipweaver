"""embed_index.search on a hand-built index and a fake embed_texts (R9).

The vectors are placed so every background document sits at cosine ~0.75 to
the query: above the EMBED_MIN_COSINE floor, the way E5 packs unrelated text
into a narrow band. Only the z-score against the whole catalog can tell the
real match from that background, and these tests pin that it does.
"""
import math

import numpy as np
import pytest

from clipmaker import config, embed_index

DIM = 128
QUERY = "galleon broadside in a storm"


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def at_cosine(cos: float, axis: int) -> np.ndarray:
    """A unit vector at `cos` to unit(0), leaning into its own axis."""
    return (cos * unit(0) + math.sqrt(1 - cos * cos) * unit(axis)).astype(np.float32)


def doc(name: str, *, game: str = "g1", level: str = "scene", start: float | None = 10.0,
        end: float | None = 20.0, path: str | None = None) -> dict:
    return {
        "doc_id": f"{name}:{level}:{start}", "text": f"{name} text", "level": level,
        "game": game, "catalog_path": path or f"{game}_videos/{name}.json",
        "original_name": f"{name}.mp4", "source_path": f"/footage/{name}.mp4",
        "title": name.title(), "start_sec": start, "end_sec": end, "scene_type": "gameplay",
    }


def background(n: int = 20, game_of=lambda i: "g1") -> list[tuple[dict, np.ndarray]]:
    """n documents spread over cosine 0.72-0.78: plausible, but nothing stands out."""
    return [(doc(f"bg{i:02d}", game=game_of(i)), at_cosine(0.72 + 0.06 * i / (n - 1), axis=i + 2))
            for i in range(n)]


def save(rows: list[tuple[dict, np.ndarray]]) -> None:
    index = embed_index.Index(vectors=np.vstack([v for _, v in rows]),
                              docs=[d for d, _ in rows], manifest={})
    embed_index._save_index(index)
    embed_index._index_cache = None


@pytest.fixture
def fake_embed(monkeypatch):
    calls = []

    def embed(texts, kind):
        calls.append((list(texts), kind))
        return np.vstack([unit(0) for _ in texts])

    monkeypatch.setattr(embed_index, "embed_texts", embed)
    return calls


TARGET = (doc("galleon"), at_cosine(0.97, axis=1))


def test_the_real_match_passes_and_the_background_does_not(fake_embed):
    save(background() + [TARGET])
    hits = embed_index.search(QUERY)
    assert [h.original_name for h in hits] == ["galleon.mp4"]
    hit = hits[0]
    assert hit.score >= config.EMBED_MIN_ZSCORE and hit.cosine == pytest.approx(0.97, abs=1e-3)
    assert (hit.start_sec, hit.end_sec, hit.timecode) == (10.0, 20.0, "00:10-00:20")
    assert fake_embed == [([QUERY], "query")]


def test_a_query_matching_nothing_is_rejected_although_every_cosine_clears_the_floor(fake_embed):
    # The "borscht recipe" control: every document scores above EMBED_MIN_COSINE,
    # none stands out from the catalog's own distribution.
    save(background())
    assert embed_index.search(QUERY) == []
    assert embed_index.search(QUERY, min_zscore=-10)  # the floor alone would pass them


def test_min_zscore_argument_overrides_the_config(fake_embed):
    save(background() + [TARGET])
    assert embed_index.search(QUERY, min_zscore=50) == []


def test_cosine_floor_rejects_a_high_z_that_is_still_far_off(fake_embed):
    # Everything is unrelated (cos ~0.1) and one document a little less so: its
    # z is huge, but 0.6 is below the sanity floor.
    rows = [(doc(f"far{i}"), at_cosine(0.1 + 0.001 * i, axis=i + 2)) for i in range(20)]
    save(rows + [(doc("closer"), at_cosine(0.6, axis=1))])
    assert embed_index.search(QUERY) == []


def test_the_game_filter_selects_hits_but_does_not_rescale_them(fake_embed):
    save(background(game_of=lambda i: "g1" if i % 2 else "g2") + [TARGET])
    unfiltered = embed_index.search(QUERY)
    filtered = embed_index.search(QUERY, game="g1")
    assert [h.score for h in filtered] == [h.score for h in unfiltered]
    assert embed_index.search(QUERY, game="g2") == []


def test_level_weight_lowers_a_window_hit(fake_embed):
    window = (doc("galleon", level="window"), TARGET[1])
    save(background() + [window])
    hits = embed_index.search(QUERY, min_zscore=0)
    window_hit = next(h for h in hits if h.original_name == "galleon.mp4")
    save(background() + [TARGET])
    scene_score = next(h for h in embed_index.search(QUERY, min_zscore=0)
                            if h.original_name == "galleon.mp4").score
    ratio = window_hit.score / scene_score
    assert ratio == pytest.approx(config.EMBED_LEVEL_WEIGHTS["window"], abs=1e-3)


def test_fewer_than_eight_candidates_returns_nothing_without_embedding(fake_embed):
    # z-scores mean nothing on a handful of documents.
    rows = background(game_of=lambda i: "g1" if i < 5 else "g2") + [TARGET]
    save(rows)
    assert embed_index.search(QUERY, game="g1") == []  # 5 background + target = 6
    assert fake_embed == []
    assert embed_index.search(QUERY, game="g2") == []  # 15 docs, none of them the match


def test_one_video_gives_one_hit_unless_all_hits_are_asked_for(fake_embed):
    second = (doc("galleon", start=40.0, end=50.0, path=TARGET[0]["catalog_path"]),
              at_cosine(0.96, axis=100))
    save(background(60) + [TARGET, second])
    assert len(embed_index.search(QUERY)) == 1
    both = embed_index.search(QUERY, per_video_best=False)
    assert [(h.start_sec, h.end_sec) for h in both] == [(10.0, 20.0), (40.0, 50.0)]


def test_k_caps_the_number_of_hits(fake_embed):
    strong = [(doc(f"ship{i}"), at_cosine(0.97, axis=100 + i)) for i in range(4)]
    save(background(60) + strong)
    assert len(embed_index.search(QUERY, k=2)) == 2


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_is_no_search(fake_embed, query):
    save(background() + [TARGET])
    assert embed_index.search(query) == []
    assert fake_embed == []


def test_no_index_is_no_hits(fake_embed):
    assert embed_index.search(QUERY) == []


def test_a_truncated_npz_is_reported_not_read_as_no_index(fake_embed, caplog):
    save(background() + [TARGET])
    vectors = embed_index.index_files().vectors
    vectors.write_bytes(vectors.read_bytes()[:40])
    embed_index._index_cache = None
    assert embed_index.search(QUERY) == []
    assert "unreadable" in caplog.text and "build --force" in caplog.text


def test_vectors_out_of_step_with_docs_are_reported(fake_embed, caplog):
    save(background() + [TARGET])
    rows = background()[:3]
    np.savez_compressed(embed_index.index_files().vectors,
                        vectors=np.vstack([v for _, v in rows]))
    embed_index._index_cache = None
    assert embed_index.search(QUERY) == []
    assert "out of step" in caplog.text


def test_a_backwards_scene_keeps_its_start_only():
    hit = embed_index.Hit(score=4.0, cosine=0.9, level="scene", game="g1", catalog_path="p",
                          original_name="v.mp4", source_path="", title="", text="",
                          start_sec=30.0, end_sec=12.0, scene_type=None)
    assert not hit.timing_ok and hit.timecode == "00:30~"

"""The storyboard search: report.generate_report end to end on a real index built
with a fake embedder, plus the two rules it rests on - one slot per shot query
(_best_hits_across_queries) and corroboration by name (_corroborates) (R9).

No Ollama (written shot queries only), no YouTube, no model download.
"""
import json
import math

import numpy as np
import pytest

from clipmaker import config, embed_index, report, shot_query
from clipmaker.embed_index import Hit
from clipmaker.shot_query import ShotQuery

GAME = "sb_game"
DIM = 96


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _at(cos: float, axis: int) -> np.ndarray:
    return (cos * _unit(0) + math.sqrt(1 - cos * cos) * _unit(axis)).astype(np.float32)


def fake_embed_texts(texts, kind):
    """A galleon query points along axis 0, any other query along axis 5. A
    passage naming the galleon sits at 0.97 to axis 0, everything else in the
    0.72-0.78 band every E5 passage lands in."""
    rows = []
    for text in texts:
        if kind == "query":
            rows.append(_unit(0) if "galleon" in text.lower() else _unit(5))
        elif "galleon" in text.lower():
            rows.append(_at(0.97, axis=1))
        else:
            slot = sum(map(ord, text)) % 80
            rows.append(_at(0.72 + 0.06 * slot / 79, axis=slot + 10))
    return np.vstack(rows)


def entry(name: str, title: str, scenes: list[tuple[str, str, str]], tags=()) -> dict:
    return {
        "game": GAME, "original_name": f"{name}.mp4", "source_path": f"/footage/{name}.mp4",
        "title": title, "summary": f"{title} in the open world", "tags": list(tags),
        "duration_sec": 120,
        "scenes": [{"start": s, "end": e, "type": "gameplay", "description": d}
                   for s, e, d in scenes],
    }


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setitem(config.GAMES, GAME, {"video_dirs": [], "catalog_subdir": f"{GAME}_videos"})
    folder = config.CATALOG_DIR / f"{GAME}_videos"
    folder.mkdir(parents=True)
    videos = [entry("naval", "Naval fight off the coast",
                    [("00:00", "00:20", "the ship sails out of the harbour at dawn"),
                     ("00:20", "00:30", "a galleon fires a broadside in the storm")],
                    tags=["havana", "naval_battle"])]
    videos += [entry(f"walk{i:02d}", f"Town walk number {i}",
                     [("00:00", "00:30", f"the player walks along street number {i}"),
                      ("00:30", "01:00", f"the player climbs roof number {i}")])
               for i in range(12)]
    for v in videos:
        (folder / v["original_name"].replace(".mp4", ".json")).write_text(
            json.dumps(v), encoding="utf-8")
    monkeypatch.setattr(embed_index, "embed_texts", fake_embed_texts)
    embed_index.build_index()

    def no_ollama(*_a, **_k):
        raise AssertionError("written shot queries must not ask Ollama")

    monkeypatch.setattr(shot_query, "build_shot_queries", no_ollama)
    monkeypatch.setattr(report, "search_youtube", lambda *a, **k: [])
    return folder


TEXT = "The galleon fights in a storm off Havana.\n\nNothing of this was ever recorded."
WRITTEN = {1: [ShotQuery("galleon in a storm", "written")],
           2: [ShotQuery("a cathedral interior", "written")]}


def test_storyboard_places_the_moment_and_records_it(catalog):
    md, moments = report.generate_report(
        TEXT, "proj", game=GAME, min_words=1, yt_per_fragment=0, written_shots=WRITTEN,
        query_overrides={1: "Galleon Havana storm", 2: "Cathedral"})

    assert md.count("## Fragment") == 2
    assert "semantic index:" in md.splitlines()[2]
    assert "Moment: **00:20-00:30**" in md
    assert "Found via shot query: `galleon in a storm`" in md
    assert "**Shot queries (shots.json):** `galleon in a storm`" in md
    assert "Unconfirmed" not in md  # 'havana' is in the tags

    first, second = moments["fragments"]
    best = first["candidates"][0]
    assert best["original_name"] == "naval.mp4" and best["match"] == "semantic"
    assert (best["start_sec"], best["end_sec"], best["timing_ok"]) == (20.0, 30.0, True)
    assert best["zscore"] >= config.EMBED_MIN_ZSCORE and best["level"] == "scene"
    assert best["shot_query"] == "galleon in a storm"
    assert second["candidates"] == []
    assert "*No matching local videos*" in md.split("## Fragment 2")[1]


def test_a_hit_that_names_none_of_the_query_entities_is_flagged(catalog):
    md, _ = report.generate_report(
        TEXT.split("\n\n")[0], "proj", game=GAME, min_words=1, yt_per_fragment=0,
        written_shots={1: WRITTEN[1]}, query_overrides={1: "Kingston storm"})
    assert "**Unconfirmed** — nothing here mentions Kingston" in md


def test_without_an_index_the_keyword_match_names_the_video_but_no_moment(catalog):
    for f in config.EMBED_INDEX_DIR.iterdir():
        f.unlink()
    embed_index._index_cache = None
    text = "Naval fight off the coast near havana."
    md, moments = report.generate_report(text, "proj", game=GAME, min_words=1,
                                         yt_per_fragment=0, written_shots={1: WRITTEN[1]})
    candidate = moments["fragments"][0]["candidates"][0]
    assert candidate["original_name"] == "naval.mp4" and candidate["match"] == "keyword"
    assert candidate["start_sec"] is None and candidate["end_sec"] is None
    assert "keyword match — video only" in md
    assert "semantic index: none (keyword fallback)" in md


def test_empty_text_is_no_storyboard(catalog):
    assert report.generate_report("   ", "proj", game=GAME) == (
        "No fragments found in text.", {"fragments": []})


# --- one slot per shot query ----------------------------------------------------

def hit(path: str, score: float, game: str = "g1") -> Hit:
    return Hit(score=score, cosine=0.8, level="scene", game=game, catalog_path=path,
               original_name=path, source_path="", title="", text="", start_sec=0.0,
               end_sec=5.0, scene_type=None)


@pytest.fixture
def results(monkeypatch):
    table: dict[tuple[str, str | None], list[Hit]] = {}
    calls = []

    def search(text, k, game):
        calls.append((text, game))
        return table.get((text, game), [])[:k]

    monkeypatch.setattr(embed_index, "search", search)
    return table, calls


def test_every_shot_query_places_a_hit_before_any_takes_a_second(results):
    table, _ = results
    table[("havana scenes", None)] = [hit("a1", 5.0), hit("a2", 4.9), hit("a3", 4.8)]
    table[("execution crowd", None)] = [hit("b1", 3.5)]
    chosen = report._best_hits_across_queries(
        [ShotQuery("havana scenes", "llm"), ShotQuery("execution crowd", "llm")], 3, None)
    assert [h.catalog_path for h in chosen] == ["a1", "a2", "b1"]
    assert {h.query for h in chosen} == {"havana scenes", "execution crowd"}


def test_a_video_found_by_two_queries_takes_one_slot(results):
    table, _ = results
    table[("q1", None)] = [hit("same", 5.0), hit("x", 4.0)]
    table[("q2", None)] = [hit("same", 4.5), hit("y", 3.9)]
    chosen = report._best_hits_across_queries(
        [ShotQuery("q1", "llm"), ShotQuery("q2", "llm")], 5, None)
    assert sorted(h.catalog_path for h in chosen) == ["same", "x", "y"]


def test_per_game_results_merge_on_one_scale_keeping_the_best_score(results):
    table, calls = results
    table[("q", "g1")] = [hit("v", 3.2, "g1")]
    table[("q", "g2")] = [hit("v", 4.1, "g2"), hit("w", 3.1, "g2")]
    chosen = report._best_hits_across_queries([ShotQuery("q", "llm")], 3, ["g1", "g2"])
    assert [(h.catalog_path, h.score) for h in chosen] == [("v", 4.1), ("w", 3.1)]
    assert calls == [("q", "g1"), ("q", "g2")]


def test_a_written_query_naming_its_game_searches_only_that_catalog(results):
    _, calls = results
    report._best_hits_across_queries([ShotQuery("ship", "written", "g2")], 3, ["g1", "g2"])
    assert calls == [("ship", "g2")]


def test_the_raw_fragment_fallback_is_not_reported_as_a_shot_query(results):
    table, _ = results
    table[("the paragraph itself", None)] = [hit("v", 3.3)]
    (chosen,) = report._best_hits_across_queries(
        [ShotQuery("the paragraph itself", "fragment")], 3, None)
    assert chosen.query == ""


def test_no_queries_no_hits(results):
    assert report._best_hits_across_queries([], 3, None) == []


# --- corroboration --------------------------------------------------------------

def test_query_entities_drop_franchise_words():
    assert report._query_entities("Assassin's Creed Unity Louis XVI execution") == {"Louis", "XVI"}


@pytest.mark.parametrize("where", ["title", "original_name", "tags", "hit"])
def test_an_entity_anywhere_in_the_entry_or_the_hit_corroborates(where):
    e = {"title": "", "original_name": "", "tags": []}
    h = hit("v", 3.5)
    if where == "tags":
        e["tags"] = ["kingston_rooftops"]
    elif where == "hit":
        h = Hit(**{**h.__dict__, "text": "rooftops of KINGSTON at night"})
    else:
        e[where] = "Kingston chase"
    assert report._corroborates({"Kingston"}, e, h)


def test_no_entity_named_means_nothing_to_corroborate():
    assert report._corroborates(set(), {"title": "Anything"}, None)


def test_a_generic_scene_does_not_corroborate_a_named_episode():
    e = {"title": "Havana harbour", "original_name": "h.mp4", "tags": ["havana"]}
    assert not report._corroborates({"Alexander"}, e, hit("h", 3.4))

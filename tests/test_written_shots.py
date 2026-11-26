"""shots.json: picture queries written before searching replace Ollama for their
fragments; the game named in the file narrows the search to that catalog."""
import json
from pathlib import Path

import pytest

from clipmaker import config, embed_index, report, shot_query
from clipmaker.shot_query import ShotQuery, load_written_shots


@pytest.fixture
def games(monkeypatch):
    for g in ("g1", "g2"):
        monkeypatch.setitem(config.GAMES, g, {"video_dirs": [], "catalog_subdir": f"{g}_videos"})


def write(tmp_path, data):
    path = tmp_path / "shots.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_missing_file_is_empty(tmp_path):
    assert load_written_shots(tmp_path / "shots.json") == {}


def test_object_shape_with_file_game_and_notes(tmp_path, games):
    path = write(tmp_path, {"_note": "hand-written", "game": "g1",
                            "1": ["ship at sea", {"text": "menu", "game": "g2"}], "2": []})
    shots = load_written_shots(path)
    assert shots[1] == [ShotQuery("ship at sea", "written", "g1"), ShotQuery("menu", "written", "g2")]
    assert shots[2] == []  # written as "nothing filmable"


def test_list_shape_as_in_the_sample(tmp_path, games):
    path = write(tmp_path, [{"fragment": 3, "game": "g2", "shots": ["swimming near a reef"],
                             "expected": "x.mp4"}])
    assert load_written_shots(path) == {3: [ShotQuery("swimming near a reef", "written", "g2")]}


def test_the_repository_sample_loads():
    assert set(load_written_shots(Path(__file__).resolve().parent.parent / "examples" / "sample" / "shots.json")) == {1, 2, 3, 4}


@pytest.mark.parametrize("data, match", [
    ({"one": ["x"]}, "not a number"),
    ({"0": ["x"]}, "start at 1"),
    ({"1": "x"}, "list of queries"),
    ({"1": [42]}, "string or"),
    ({"game": "nope", "1": ["x"]}, "unknown game"),
    ([{"shots": ["x"]}], "fragment"),
    ("just text", "object or a list"),
])
def test_malformed_files_name_the_problem(tmp_path, games, data, match):
    with pytest.raises(ValueError, match=match):
        load_written_shots(write(tmp_path, data))


def test_storyboard_uses_written_queries_and_their_game(monkeypatch, games):
    text = ("First paragraph about the ship and the long voyage across the sea " * 5 + "\n\n"
            + "Second paragraph about the harbour and the upgrades of the hull " * 5)
    asked_ollama, searched = [], []
    monkeypatch.setattr(shot_query, "build_shot_queries",
                        lambda frag, **k: asked_ollama.append(frag) or [ShotQuery("llm q", "llm")])
    monkeypatch.setattr(embed_index, "load_index", lambda: object())
    monkeypatch.setattr(embed_index, "index_stats", lambda: {"exists": False})
    monkeypatch.setattr(embed_index, "search",
                        lambda q, k, game: searched.append((q, game)) or [])
    monkeypatch.setattr(report, "_load_catalog", lambda games: [{"original_name": "v.mp4"}])
    monkeypatch.setattr(report, "search_youtube", lambda *a, **k: [], raising=False)

    md, _ = report.generate_report(
        text, "p", game="g1,g2", min_words=10, yt_per_fragment=0,
        written_shots={1: [ShotQuery("ship at sea", "written", "g2")]})

    assert len(asked_ollama) == 1 and asked_ollama[0].startswith("Second")
    assert ("ship at sea", "g2") in searched
    assert ("ship at sea", "g1") not in searched
    assert "**Shot queries (shots.json):** `ship at sea`" in md


def test_queries_numbered_for_another_split_are_refused(monkeypatch):
    monkeypatch.setattr(report, "_load_catalog", lambda games: [])
    text = "\n\n".join(f"Short paragraph number {n} about the ship." for n in range(1, 5))
    written = {4: [ShotQuery("ship", "written")]}
    with pytest.raises(ValueError, match="splits into 1"):
        report.generate_report(text, "p", min_words=40, yt_per_fragment=0, written_shots=written)
    monkeypatch.setattr(report, "search_youtube", lambda *a, **k: [], raising=False)
    md, _ = report.generate_report(text, "p", min_words=1, yt_per_fragment=0, written_shots=written)
    assert md.count("## Fragment") == 4

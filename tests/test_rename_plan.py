"""Sequence numbering in the catalog rename plan.

The regression covered here surfaced on the Black Flag Resynced round of
2026-09-04, when batch 8 wanted to renumber nine already-settled clips
(2 -> 12, 3 -> 13, ... 9 -> 19). The group had just passed ten members, and
the plan derived every number from string order, where "10" sorts before "2".
"""

import json
from pathlib import Path

import pytest

from clipmaker import analyzer


GAME = "test_game"
BASE = "Black Flag Exploring Great Inagua.mp4"


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """A catalog + video dir pair wired into config, returning an `add` helper."""
    videos = tmp_path / "videos"
    videos.mkdir()
    cat_root = tmp_path / "catalog"
    (cat_root / f"{GAME}_videos").mkdir(parents=True)

    monkeypatch.setattr(analyzer.config, "CATALOG_DIR", cat_root)
    monkeypatch.setitem(
        analyzer.config.GAMES, GAME, {"catalog_subdir": f"{GAME}_videos"}
    )

    def add(video_name: str, base: str = BASE) -> Path:
        video = videos / video_name
        video.write_bytes(b"")
        rec = {"source_path": str(video), "suggested_filename": base}
        jp = cat_root / f"{GAME}_videos" / f"{Path(video_name).stem}.json"
        jp.write_text(json.dumps(rec), encoding="utf-8")
        return video

    return add


def _plan(add_calls):
    return {
        (src.name, dst.name) for src, dst, _, _ in analyzer._rename_plan(GAME)
    }


def test_settled_group_of_ten_is_left_alone(catalog):
    for n in range(1, 11):
        catalog(f"Black Flag Exploring Great Inagua {n}.mp4")
    assert analyzer._rename_plan(GAME) == []


def test_new_entry_joins_a_group_of_ten_as_eleven(catalog):
    for n in range(1, 11):
        catalog(f"Black Flag Exploring Great Inagua {n}.mp4")
    catalog("Raw Capture_20260826173743.mp4")

    assert _plan(catalog) == {
        (
            "Raw Capture_20260826173743.mp4",
            "Black Flag Exploring Great Inagua 11.mp4",
        )
    }


def test_fresh_entries_take_the_lowest_free_numbers(catalog):
    catalog("Black Flag Exploring Great Inagua 1.mp4")
    catalog("Black Flag Exploring Great Inagua 3.mp4")
    catalog("Raw Capture_20260826173743.mp4")
    catalog("Raw Capture_20260826174653.mp4")

    assert _plan(catalog) == {
        ("Raw Capture_20260826173743.mp4", "Black Flag Exploring Great Inagua 2.mp4"),
        ("Raw Capture_20260826174653.mp4", "Black Flag Exploring Great Inagua 4.mp4"),
    }


def test_fresh_entries_are_numbered_in_recording_order(catalog):
    catalog("Raw Capture_20260826174653.mp4")
    catalog("Raw Capture_20260826173743.mp4")

    assert _plan(catalog) == {
        ("Raw Capture_20260826173743.mp4", "Black Flag Exploring Great Inagua 1.mp4"),
        ("Raw Capture_20260826174653.mp4", "Black Flag Exploring Great Inagua 2.mp4"),
    }


def test_numbering_is_per_base_name(catalog):
    other = "Black Flag Exploring Nassau.mp4"
    catalog("Black Flag Exploring Great Inagua 1.mp4")
    catalog("Raw Capture_20260826173743.mp4")
    catalog("Raw Capture_20260826174653.mp4", base=other)

    assert _plan(catalog) == {
        ("Raw Capture_20260826173743.mp4", "Black Flag Exploring Great Inagua 2.mp4"),
        ("Raw Capture_20260826174653.mp4", "Black Flag Exploring Nassau 1.mp4"),
    }


def test_plan_is_idempotent_after_being_applied(catalog):
    for n in range(1, 11):
        catalog(f"Black Flag Exploring Great Inagua {n}.mp4")
    catalog("Raw Capture_20260826173743.mp4")

    for src, dst, jp, new_jp in analyzer._rename_plan(GAME):
        rec = json.loads(jp.read_text(encoding="utf-8"))
        src.rename(dst)
        rec["source_path"] = str(dst)
        new_jp.write_text(json.dumps(rec), encoding="utf-8")
        jp.unlink()

    assert analyzer._rename_plan(GAME) == []


def test_shadowplay_colon_underscore_does_not_repeat_game_name(monkeypatch):
    # ShadowPlay writes "Onimusha: Way of the Sword" as "Onimusha_ Way of the
    # Sword"; the folder-derived location must fold into the display name.
    # The game entry is set here: the public config ships only the sample game.
    from clipmaker import config
    monkeypatch.setitem(config.GAMES, "onimusha", {
        "video_dirs": [], "catalog_subdir": "onimusha_videos",
        "display_name": "Onimusha: Way of the Sword"})
    rec = {
        "game": "onimusha",
        "title": "Initial Settings Menu",
        "path_hints": {"location": "Onimusha_ Way of the Sword"},
    }
    assert analyzer.suggest_filename(rec, Path("x.MP4")) == (
        "Onimusha Way of the Sword Initial Settings Menu.mp4"
    )

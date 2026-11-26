"""Batch renaming never loses or overwrites a recording (review by Vitaly Chashchin).

The standalone renamer planned every destination before moving anything and
checked only the disk, so two recordings with the same suggested name got the
same destination: on Windows the second rename failed after the first had run,
on Linux it replaced the first recording. Both entry points now share one
planner that reserves video and catalog destinations across the batch, and the
move itself refuses to overwrite.
"""
import json
from pathlib import Path

import pytest

from clipmaker import config, naming, rename_videos

GAME = "rename_game"
BASE = "Kingston Chase.mp4"


@pytest.fixture
def library(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    cat = tmp_path / "catalog" / f"{GAME}_videos"
    cat.mkdir(parents=True)
    monkeypatch.setattr(config, "CATALOG_DIR", tmp_path / "catalog")
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setitem(config.GAMES, GAME, {"catalog_subdir": f"{GAME}_videos", "video_dirs": [videos]})

    def add(rel: str, base: str = BASE, json_name: str | None = None) -> Path:
        video = videos / rel
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(rel.encode())  # distinct content per recording
        jp = cat / (json_name or f"{video.stem}.json")
        jp.write_text(json.dumps({"source_path": str(video), "suggested_filename": base}),
                      encoding="utf-8")
        return video

    return videos, cat, add


def _contents(folder: Path) -> set[bytes]:
    return {p.read_bytes() for p in folder.rglob("*.mp4")}


def test_standalone_renamer_keeps_both_recordings_with_the_same_name(library):
    videos, cat, add = library
    add("Raw Capture_1.mp4")
    add("Raw Capture_2.mp4")
    before = _contents(videos)

    rename_videos.main(["--game", GAME, "--apply"])

    assert _contents(videos) == before
    names = sorted(p.name for p in videos.glob("*.mp4"))
    assert names == ["Kingston Chase 1.mp4", "Kingston Chase 2.mp4"]
    entries = [json.loads(p.read_text(encoding="utf-8")) for p in cat.glob("*.json")]
    assert sorted(Path(e["source_path"]).name for e in entries) == names


def test_standalone_renamer_is_a_dry_run_by_default(library):
    videos, _, add = library
    add("Raw Capture_1.mp4")
    rename_videos.main(["--game", GAME])
    assert [p.name for p in videos.glob("*.mp4")] == ["Raw Capture_1.mp4"]


def test_catalog_destinations_are_reserved_across_the_batch(library):
    # Two folders, same suggested name: the videos may both become "Kingston
    # Chase 1.mp4" (different folders), but their entries share one catalog
    # folder and must not both become "Kingston Chase 1.json".
    _, _, add = library
    add("a/Raw Capture_1.mp4", json_name="a1.json")
    add("b/Raw Capture_1.mp4", json_name="b1.json")
    plan = naming._rename_plan(GAME)
    assert len(plan) == 2
    assert len({new_json for _, _, _, new_json in plan}) == 2


def test_an_existing_catalog_file_is_never_a_destination(library):
    _, cat, add = library
    add("Raw Capture_1.mp4")
    (cat / "Kingston Chase 1.json").write_text("{}", encoding="utf-8")  # someone else's entry
    ((_, new_video, _, new_json),) = naming._rename_plan(GAME)
    assert new_json.name != "Kingston Chase 1.json"
    assert new_video.stem == new_json.stem


def test_apply_refuses_a_destination_that_appeared_after_planning(library, monkeypatch, caplog):
    videos, _, add = library
    src = add("Raw Capture_1.mp4")
    plan = naming._rename_plan(GAME)
    taken = plan[0][1]
    taken.write_bytes(b"another recording")
    monkeypatch.setattr(naming, "_rename_plan", lambda game: plan)

    assert naming.rename_from_catalog(GAME, dry_run=False) == 0
    assert taken.read_bytes() == b"another recording"
    assert src.read_bytes() == b"Raw Capture_1.mp4"
    assert "FAILED video rename" in caplog.text

"""clipmaker.grid: a labelled frame grid of one capture, found by name."""

import shutil
import subprocess

import pytest

from clipmaker import config, grid

PREFIX = "Assassin's Creed IV Black Flag Resynced "


@pytest.fixture
def footage(tmp_path):
    names = [PREFIX + "Ship Upgrade Trading 1", PREFIX + "Ship Upgrade Trading 11",
             PREFIX + "Character Inventory Management 1", "Kill Keepstar 1A8-6G"]
    for n in names:
        (tmp_path / f"{n}.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


def test_suffix_rule_picks_the_numbered_clip_not_its_tens(footage):
    files = grid.footage_files([footage])
    assert grid.find_clip("Ship Upgrade Trading 1", files).stem == PREFIX + "Ship Upgrade Trading 1"


def test_exact_stem_and_unique_substring(footage):
    files = grid.footage_files([footage])
    assert grid.find_clip(PREFIX + "Ship Upgrade Trading 11", files).stem.endswith("11")
    assert grid.find_clip("keepstar", files).name == "Kill Keepstar 1A8-6G.mp4"
    assert grid.find_clip("Kill Keepstar 1A8-6G.mp4", files).name == "Kill Keepstar 1A8-6G.mp4"


def test_ambiguous_and_missing_names_raise(footage):
    files = grid.footage_files([footage])
    with pytest.raises(LookupError, match="2 matches"):
        grid.find_clip("Upgrade Trading", files)
    with pytest.raises(LookupError, match="no footage"):
        grid.find_clip("Havana", files)


def test_same_stem_in_two_folders_is_ambiguous(footage, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (other / "Kill Keepstar 1A8-6G.mp4").write_bytes(b"")
    with pytest.raises(LookupError):
        grid.find_clip("Kill Keepstar 1A8-6G", grid.footage_files([footage, other]))


def test_range_includes_both_ends():
    assert grid.range_times(10, 21, 12) == [float(t) for t in range(10, 22)]
    assert grid.range_times(5, 5, 1) == [5.0]
    with pytest.raises(ValueError):
        grid.range_times(9, 3, 4)


def test_game_dirs_come_from_config(monkeypatch, footage):
    monkeypatch.setitem(config.GAMES, "grid_game", {"video_dirs": [footage]})
    assert grid.search_dirs("grid_game", []) == [footage]
    with pytest.raises(ValueError):
        grid.search_dirs("no_such_game", [])


def test_name_without_dirs_is_a_usage_error():
    with pytest.raises(SystemExit):
        grid.main(["some clip", "--at", "1", "--out", "x.jpg"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_end_to_end_sheet(tmp_path):
    clip = tmp_path / "Test Pattern 1.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=duration=3:size=320x180:rate=10", "-pix_fmt", "yuv420p",
                    str(clip)], check=True)
    out = tmp_path / "grid.jpg"
    grid.main(["Pattern 1", "--dir", str(tmp_path), "--from", "0.5", "--to", "2.5",
               "--count", "3", "--out", str(out)])
    assert out.stat().st_size > 0

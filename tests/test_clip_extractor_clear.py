"""Re-cutting a fragment first removes its earlier cuts (any source)."""
from clipmaker.clip_extractor import clear_fragment_clips


def test_old_cuts_of_the_fragment_go_others_stay(tmp_path):
    for name in ("03_old.mp4", "03_new.mp4", "13_other.mp4", "04_x.mp4", "_cut_list.md"):
        (tmp_path / name).write_bytes(b"")
    assert clear_fragment_clips(tmp_path, 3) == ["03_new.mp4", "03_old.mp4"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["04_x.mp4", "13_other.mp4", "_cut_list.md"]


def test_missing_clips_folder_is_fine(tmp_path):
    assert clear_fragment_clips(tmp_path / "clips", 1) == []

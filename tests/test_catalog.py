"""One catalog reader: BOM files load everywhere, bad files and name clashes are
reported instead of disappearing."""
import json

import pytest

from clipmaker import build_storyboard, catalog, clip_extractor, config, layout, report


@pytest.fixture
def games(monkeypatch):
    monkeypatch.setattr(catalog, "_reported", set())
    for g in ("g1", "g2"):
        monkeypatch.setitem(config.GAMES, g, {"video_dirs": [], "catalog_subdir": f"{g}_videos"})
        (config.CATALOG_DIR / f"{g}_videos").mkdir(parents=True)
    return config.CATALOG_DIR


def write(root, rel, data, bom=False):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=4)
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return path


def entry(name, **extra):
    return {"original_name": name, "title": name, "source_path": f"/footage/{name}", **extra}


def test_file_with_bom_is_loaded_by_every_reader(games):
    write(games, "g1_videos/sub/a.json", entry("a.mp4"), bom=True)
    write(games, "g1_videos/b.json", entry("b.mp4"))
    assert sorted(e["original_name"] for e in report._load_catalog("g1")) == ["a.mp4", "b.mp4"]
    for index in (build_storyboard.load_catalog_index(["g1"]),
                  clip_extractor.load_catalog_index("g1"),
                  layout.load_catalog_index(["g1"])):
        assert set(index) == {"a.mp4", "b.mp4"}


def test_entries_carry_game_and_posix_catalog_path(games):
    write(games, "g1_videos/sub/a.json", entry("a.mp4"))
    (e,) = catalog.load_entries(["g1"])
    assert e["_game"] == "g1" and e["_catalog_path"] == "g1_videos/sub/a.json"


def test_unreadable_file_is_reported_once(games, capsys):
    (games / "g1_videos" / "broken.json").write_text("{not json", encoding="utf-8")
    write(games, "g1_videos/ok.json", entry("ok.mp4"))
    assert [e["original_name"] for e in catalog.load_entries(["g1"])] == ["ok.mp4"]
    catalog.load_entries(["g1"])
    assert capsys.readouterr().out.count("broken.json") == 1


def test_name_clash_keeps_first_game_and_says_so(games, capsys):
    write(games, "g1_videos/k.json", entry("keynote.mp4", source_path="/footage/one/keynote.mp4"))
    write(games, "g2_videos/k.json", entry("keynote.mp4", source_path="/footage/two/keynote.mp4"))
    index = catalog.load_index(["g1", "g2"])
    assert index["keynote.mp4"]["source_path"] == "/footage/one/keynote.mp4"
    out = capsys.readouterr().out
    assert "g1_videos/k.json (used)" in out and "g2_videos/k.json (ignored)" in out


def test_unknown_game_and_missing_folder_give_nothing(games):
    assert catalog.load_entries(["no_such_game"]) == []
    config.GAMES["g2"]["catalog_subdir"] = "missing_videos"
    assert catalog.load_entries(["g2"]) == []


def test_resolve_source_order(tmp_path):
    footage = tmp_path / "footage"
    footage.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "nice name.mp4").write_bytes(b"")
    cat = {"nice name.mp4": {"source_path": str(raw_dir / "old_raw.mp4")}}
    assert catalog.resolve_source("nice name.mp4", footage, cat) == raw_dir / "nice name.mp4"
    (footage / "nice name.mp4").write_bytes(b"")
    assert catalog.resolve_source("nice name.mp4", footage, cat) == footage / "nice name.mp4"
    assert catalog.resolve_source("absent.mp4", footage, cat) is None

"""CLIPMAKER_PATH_MAP: footage that moved to another drive or folder is found
without rewriting the catalog."""
import pytest

from clipmaker import catalog, config, resolve_export


def test_parse_orders_longest_first_and_normalises_slashes():
    pairs = config.parse_path_map(r"old\raw => new\ ; old\raw\Game 4 => new/Game IV; junk; => x")
    assert pairs == [("old/raw/Game 4", "new/Game IV"), ("old/raw", "new")]


def test_empty_or_unset_map():
    assert config.parse_path_map(None) == [] and config.parse_path_map("") == []


@pytest.fixture
def moved(tmp_path, monkeypatch):
    old, archive = tmp_path / "raw", tmp_path / "archive"
    (archive / "Diablo IV" / "season").mkdir(parents=True)
    (archive / "AC4").mkdir()
    (archive / "Diablo IV" / "season" / "fight 1.mp4").write_bytes(b"")
    (archive / "AC4" / "ship.mp4").write_bytes(b"")
    monkeypatch.setattr(config, "PATH_MAP", config.parse_path_map(
        f"{old} => {archive}; {old / 'Diablo 4'} => {archive / 'Diablo IV'}"))
    return old, archive


def test_longest_prefix_wins_and_case_does_not_matter(moved):
    old, archive = moved
    assert catalog.map_path(old / "Diablo 4" / "season" / "fight 1.mp4") == \
        archive / "Diablo IV" / "season" / "fight 1.mp4"
    upper = str(old / "AC4" / "ship.mp4").upper()
    assert catalog.map_path(upper).name == "SHIP.MP4"
    assert catalog.map_path("/elsewhere/x.mp4") is None


def test_resolve_source_finds_moved_and_renamed_files(moved, tmp_path):
    old, archive = moved
    cat = {"ship.mp4": {"source_path": str(old / "AC4" / "ship.mp4")},
           # renamed after indexing: source_path still has the raw name
           "fight 1.mp4": {"source_path": str(old / "Diablo 4" / "season" / "raw_0001.mp4")}}
    footage = tmp_path / "footage"
    assert catalog.resolve_source("ship.mp4", footage, cat) == archive / "AC4" / "ship.mp4"
    assert catalog.resolve_source("fight 1.mp4", footage, cat) == \
        archive / "Diablo IV" / "season" / "fight 1.mp4"


def test_resolve_export_reads_an_old_layout_through_the_map(moved, tmp_path):
    old, archive = moved
    layout = tmp_path / "layout.txt"
    layout.write_text(f"fps=60\n---\n0|1000|0|fr 01|{old / 'AC4' / 'ship.mp4'}\n", encoding="utf-8")
    (event,) = resolve_export.parse_layout(layout).events
    assert event.path == archive / "AC4" / "ship.mp4"

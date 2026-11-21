"""clipmaker.ocrgrep — moved from projects/*/tools/ocrgrep.py (AC4 and EVE copies)."""

import json
import re

import pytest

from clipmaker import config, ocrgrep

GAME = "grep_game"


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    cat = tmp_path / f"{GAME}_videos"
    (cat / "sub").mkdir(parents=True)
    monkeypatch.setattr(config, "CATALOG_DIR", tmp_path)
    monkeypatch.setitem(config.GAMES, GAME, {"catalog_subdir": f"{GAME}_videos"})

    def add(rel, entry):
        (cat / rel).write_text(json.dumps(entry), encoding="utf-8")

    add("a.json", {"original_name": "Fight Y4B-BQ.mp4", "duration_sec": 90, "timeline": [
        {"start": "00:10", "caption": "Local: 12; Y4B-BQ"},
        {"start": "00:40", "caption": "warp"},
        {"start": "01:05", "caption": "", "ocr": ["Y4B-BQ < 0.0 >"]},
    ], "scenes": [{"start": "00:50", "description": "Titans land in Y4B-BQ"}]})
    add("sub/b.json", {"original_name": "Mining.mp4", "duration_sec": 30,
                       "timeline": [{"start": "00:05", "caption": "Veldspar"}]})
    add("sub/broken.json", "not an entry")
    (cat / "sub" / "bad.json").write_text("{oops", encoding="utf-8")
    return cat


def test_hits_cover_captions_ocr_and_scenes(catalog):
    [m] = ocrgrep.search(catalog, re.compile("y4b-bq", re.I))
    assert m.name == "Fight Y4B-BQ.mp4"
    assert m.stamps == ("00:10", "01:05", "00:50")
    assert m.preview == "Local: 12; Y4B-BQ"


def test_unreadable_and_non_object_json_is_skipped(catalog):
    names = [m.name for m in ocrgrep.search(catalog, re.compile("."))]
    assert names == ["Fight Y4B-BQ.mp4", "Mining.mp4"]


def test_file_filter(catalog):
    assert [m.name for m in ocrgrep.search(catalog, re.compile("."), "mining")] == ["Mining.mp4"]


def test_format_truncates_stamps(catalog):
    [m] = ocrgrep.search(catalog, re.compile("y4b", re.I))
    assert ocrgrep.format_match(m, 2).splitlines()[0] == \
        "Fight Y4B-BQ.mp4 [90s] @ 00:10, 01:05 (+1)"


def test_cli(catalog, capsys):
    ocrgrep.main(["veldspar", "--game", GAME])
    out = capsys.readouterr().out
    assert "Mining.mp4 [30s] @ 00:05" in out
    assert out.rstrip().endswith("-- 1 files")


def test_unknown_game_is_an_argparse_error(catalog):
    with pytest.raises(SystemExit):
        ocrgrep.main(["x", "--game", "no_such_game"])

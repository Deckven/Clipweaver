"""vegas.json -> layout.txt take lines. These tests pin the Python side;
tests/test_vegas_loader.py runs the C# loader itself."""

import json

import pytest

from clipmaker import config, layout, vegas_takes
from clipmaker.vegas_takes import TakeSegment



# --- unit ------------------------------------------------------------------

def test_seconds_become_exact_milliseconds():
    t = TakeSegment.from_seconds("Recording 1.mp3", 12.5, 50.46, -0.477)
    assert (t.from_ms, t.to_ms, t.lag_ms) == (12500, 50460, -477)
    assert t.line() == "take=Recording 1.mp3|12500|50460|-477"
    assert TakeSegment.from_dict(t.to_dict()) == t
    assert vegas_takes.parse_take_value(t.line().split("=", 1)[1]) == t


@pytest.mark.parametrize("args", [("", 0, 1, 0), ("a|b.mp3", 0, 1, 0),
                                  ("a.mp3", 5, 5, 0), ("a.mp3", -1, 3, 0)])
def test_bad_segments_are_refused(args):
    with pytest.raises(ValueError):
        TakeSegment.from_seconds(*args)


SEG = [TakeSegment("Recording 1.mp3", 0, 12500, -477),
       TakeSegment("Recording 1.mp3", 12500, 50460, 4221)]
REC = [TakeSegment("Intro.mp3", 0, 9770, -405),
       TakeSegment("Recording 1.mp3", 0, 64750, 14675)]


LAYOUT = """# clipmaker layout v1 — p
project=p
audio=narr.mp3
fps=59.94
track=B-Roll
fade_ms=400
---
1000|2000|0|fr 01 take=not a header|E:\\a.mp4
"""


def test_set_layout_takes_replaces_only_the_take_lines():
    takes = SEG
    once = vegas_takes.set_layout_takes(LAYOUT, takes)
    assert vegas_takes.set_layout_takes(once, takes) == once  # idempotent
    header, parsed = vegas_takes.parse_layout_header(once)
    assert parsed == takes and header["fps"] == "59.94"
    assert once.split("---\n")[1] == LAYOUT.split("---\n")[1]  # events untouched
    assert vegas_takes.set_layout_takes(once, []) == LAYOUT  # removable


def test_vegas_json_is_found_in_a_parent_folder(tmp_path):
    proj = tmp_path / "eve" / "drafts" / "x"
    (proj / "overlays").mkdir(parents=True)
    vegas_takes.write_vegas_config(proj / "vegas.json", 59.94, "x.mp3",
                                   REC, "note")
    cfg = vegas_takes.find_vegas_config(proj / "overlays", root=tmp_path)
    assert (cfg.fps, cfg.audio, len(cfg.takes)) == (59.94, "x.mp3", 2)
    assert vegas_takes.find_vegas_config(tmp_path / "eve", root=tmp_path) is None


def test_write_keeps_unknown_keys_and_one_take_per_line(tmp_path):
    vj = tmp_path / "vegas.json"
    vj.write_text(json.dumps({"fps": 60, "comment": "keep me"}), encoding="utf-8")
    vegas_takes.write_vegas_config(vj, None, None, REC)
    text = vj.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["fps"] == 60 and data["comment"] == "keep me" and len(data["takes"]) == 2
    assert '    {"file": "Intro.mp3", "from_sec": 0.0, "to_sec": 9.77, "lag_sec": -0.405}' in text


def test_bad_fps_in_vegas_json_is_refused(tmp_path):
    with pytest.raises(ValueError):
        vegas_takes.parse_vegas_config({"fps": 0}, tmp_path / "vegas.json")


def test_layout_fps_comes_from_vegas_json_unless_given(tmp_path, capsys):
    cfg = vegas_takes.VegasConfig(tmp_path / "vegas.json", 59.94, None, ())
    assert layout.resolve_fps(None, cfg) == 59.94
    assert layout.resolve_fps(60, cfg) == 60
    assert "overrides" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        layout.resolve_fps(None, None)


def test_rendered_layout_carries_the_takes():
    takes = tuple(SEG)
    text = layout.render_layout_txt(layout.LayoutResult(), "p", "narr.mp3", 60, "T", 400, takes)
    assert vegas_takes.parse_layout_header(text)[1] == list(takes)
    assert text.index("take=") < text.index("---")


def test_sync_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "layout.txt").write_text(LAYOUT, encoding="utf-8")
    vegas_takes.write_vegas_config(proj / "vegas.json", 59.94, None,
                                   SEG)
    vegas_takes.main(["sync", "--project", "p"])
    assert len(vegas_takes.parse_layout_header(
        (proj / "layout.txt").read_text(encoding="utf-8"))[1]) == 2

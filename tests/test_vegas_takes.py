"""vegas.json -> layout.txt take lines, and the migration of the old loaders.

Until 2026-10-27 each raw-take project had its own copy of the Vegas loader
with a hand-typed SEG_* / REC_* lag table. The tables now live in vegas.json;
layout.py and `vegas_takes sync` copy them into layout.txt. These tests pin
the Python side; tests/test_vegas_loader.py runs the C# loader itself.
"""

import json
import re
from pathlib import Path

import pytest

from clipmaker import config, layout, vegas_takes
from clipmaker.vegas_takes import TakeSegment

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "vegas_scripts"
PIN = re.compile(r'LAYOUT_FILE = @"[^"]*[\\/]projects[\\/](.*)[\\/]layout\.txt"')


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


SEG_CS = """
    // Where each piece sits.
    // Measured by hand.
    static readonly string[] SEG_FILE = {
        "Recording 1.mp3", "Recording 1.mp3" };
    static readonly double[] SEG_FROM = {
        0.00, 12.50 };
    static readonly double[] SEG_TO = {
        12.50, 50.46 };
    static readonly double[] SEG_LAG = {
        -0.477, 4.221 };
"""
REC_CS = """
    static readonly string[] REC_NAMES = { "Intro.mp3", "Recording 1.mp3" };
    static readonly double[] REC_LAG = { -0.405, 14.675 };
    static readonly double[] REC_USED = { 9.77, 64.75 };
"""


def test_cs_tables_in_both_old_forms():
    assert vegas_takes.parse_cs_tables(SEG_CS) == [
        TakeSegment("Recording 1.mp3", 0, 12500, -477),
        TakeSegment("Recording 1.mp3", 12500, 50460, 4221)]
    assert vegas_takes.parse_cs_tables(SEG_CS.replace("SEG_FILE", "SEG_NAME"))[1].lag_ms == 4221
    assert vegas_takes.parse_cs_tables(REC_CS) == [
        TakeSegment("Intro.mp3", 0, 9770, -405),
        TakeSegment("Recording 1.mp3", 0, 64750, 14675)]
    assert vegas_takes.cs_table_note(SEG_CS) == "Where each piece sits. Measured by hand."


def test_cs_table_with_ragged_columns_is_refused():
    with pytest.raises(ValueError):
        vegas_takes.parse_cs_tables(SEG_CS.replace("-0.477, 4.221", "-0.477"))
    with pytest.raises(ValueError):
        vegas_takes.parse_cs_tables("no table here")


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
    takes = vegas_takes.parse_cs_tables(SEG_CS)
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
                                   vegas_takes.parse_cs_tables(REC_CS), "note")
    cfg = vegas_takes.find_vegas_config(proj / "overlays", root=tmp_path)
    assert (cfg.fps, cfg.audio, len(cfg.takes)) == (59.94, "x.mp3", 2)
    assert vegas_takes.find_vegas_config(tmp_path / "eve", root=tmp_path) is None


def test_write_keeps_unknown_keys_and_one_take_per_line(tmp_path):
    vj = tmp_path / "vegas.json"
    vj.write_text(json.dumps({"fps": 60, "comment": "keep me"}), encoding="utf-8")
    vegas_takes.write_vegas_config(vj, None, None, vegas_takes.parse_cs_tables(REC_CS))
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
    takes = tuple(vegas_takes.parse_cs_tables(SEG_CS))
    text = layout.render_layout_txt(layout.LayoutResult(), "p", "narr.mp3", 60, "T", 400, takes)
    assert vegas_takes.parse_layout_header(text)[1] == list(takes)
    assert text.index("take=") < text.index("---")


def test_sync_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "layout.txt").write_text(LAYOUT, encoding="utf-8")
    vegas_takes.write_vegas_config(proj / "vegas.json", 59.94, None,
                                   vegas_takes.parse_cs_tables(SEG_CS))
    vegas_takes.main(["sync", "--project", "p"])
    assert len(vegas_takes.parse_layout_header(
        (proj / "layout.txt").read_text(encoding="utf-8"))[1]) == 2


# --- the migrated projects in this repo ---------------------------------------

def old_loaders() -> list[Path]:
    """Every per-project loader in the repo that carries a lag table."""
    found = []
    for cs in sorted(SCRIPTS.glob("*.cs")) + sorted((SCRIPTS / "_archive").glob("*.cs")):
        text = cs.read_text(encoding="utf-8-sig")
        if re.search(r"\b(SEG_FILE|SEG_NAME|REC_NAMES)\b", text) and PIN.search(text):
            found.append(cs)
    return found


def project_of(cs: Path) -> Path:
    rel = PIN.search(cs.read_text(encoding="utf-8-sig")).group(1).replace("\\", "/")
    return config.PROJECTS_DIR / rel


def migrated_projects() -> list[Path]:
    return sorted(p.parent for p in config.PROJECTS_DIR.rglob("vegas.json"))


def test_there_is_something_to_check():
    # The public export ships neither the per-project loaders (_archive/) nor
    # projects/; the guard against a vacuous pass only makes sense where they live.
    if not (SCRIPTS / "_archive").is_dir():
        pytest.skip("public export: per-project loaders and projects are not published")
    assert len(old_loaders()) >= 6
    assert len(migrated_projects()) >= 6


@pytest.mark.parametrize("cs", old_loaders(), ids=lambda p: p.stem[:40])
def test_old_lag_table_equals_the_projects_vegas_json(cs):
    cfg = vegas_takes.find_vegas_config(project_of(cs))
    assert cfg is not None, f"no vegas.json for {cs.name}: run vegas_takes from-cs"
    assert list(cfg.takes) == vegas_takes.parse_cs_tables(cs.read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("proj", migrated_projects(), ids=lambda p: p.name[:40])
def test_layout_txt_carries_the_vegas_json_takes(proj):
    lay = proj / "layout.txt"
    if not lay.exists():
        pytest.skip("no layout.txt on this branch")
    cfg = vegas_takes.find_vegas_config(proj)
    header, takes = vegas_takes.parse_layout_header(lay.read_text(encoding="utf-8"))
    assert takes == list(cfg.takes), "run: python -m clipmaker.vegas_takes sync --project ..."
    assert float(header["fps"]) == pytest.approx(cfg.fps)
    if cfg.audio:
        assert header["audio"] == cfg.audio

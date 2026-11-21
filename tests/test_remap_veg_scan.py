"""clipmaker.remap and clipmaker.veg_scan — moved from the Jackdaw Upgrades tools."""

import json

import numpy as np
import pytest

from clipmaker import config, remap, veg_scan


def transcript(words):
    return {"segments": [{"end": words[-1][1] + 0.5,
                          "words": [{"word": w, "start": t} for w, t in words]}]}


OLD = [("Галка", 1.0), ("ещё", 2.0), ("одна", 3.0), ("правка", 4.0), ("и", 5.0),
       ("всё", 6.0), ("тут", 7.0)]
# re-voiced: a new 2 s opening, the same words after it; "Ещё" with ё/е and case
NEW = [("Привет", 0.0), ("друзья", 1.0)] + [(w.upper() if w == "ещё" else w, t + 2.0)
                                              for w, t in OLD]


def test_words_are_normalized():
    assert remap.normalize_word("Ещё,") == "еще"
    assert remap.transcript_words(transcript([("—", 0.0), ("Да!", 1.0)])) == [("да", 1.0)]


def test_events_move_by_their_anchor_shift():
    pairs = remap.anchor_pairs(remap.transcript_words(transcript(OLD)),
                               remap.transcript_words(transcript(NEW)))
    events = [{"start_sec": 4.5, "end_sec": 6.0, "label": "b"},
              {"start_sec": 1.5, "end_sec": 4.5, "label": "a"}]
    rows = remap.remap_events(events, pairs, new_end=9.5)
    assert [(r["label"], r["new_start_sec"], r["new_len_sec"]) for r in rows] == \
        [("a", 3.5, 3.0), ("b", 6.5, 3.0)]
    assert rows[1]["old_len_sec"] == 1.5
    assert "LONGER+1.5" in remap.row_flags(rows[1])
    assert events[0] == {"start_sec": 4.5, "end_sec": 6.0, "label": "b"}  # not mutated


def test_event_before_the_first_anchor_is_kept_and_flagged():
    pairs = [(10.0, 12.0), (11.0, 13.0)]
    [row] = remap.remap_events([{"start_sec": 0.0, "end_sec": 5.0}], pairs, new_end=20)
    assert row["new_start_sec"] == 0.0 and row["anchor_dist_sec"] is None
    assert "NO-ANCHOR" in remap.row_flags(row)


def test_short_runs_are_not_anchors():
    old = [("а", 0.0), ("б", 1.0), ("x", 2.0)]
    new = [("а", 5.0), ("б", 6.0), ("y", 7.0)]
    assert remap.anchor_pairs(old, new, min_run=3) == []


def test_remap_cli_writes_into_the_project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "old.json").write_text(json.dumps(transcript(OLD)), encoding="utf-8")
    (proj / "new.json").write_text(json.dumps(transcript(NEW)), encoding="utf-8")
    (proj / "vegas_events.json").write_text(
        json.dumps([{"start_sec": 1.5, "end_sec": 4.5, "label": "a"}]), encoding="utf-8")
    remap.main(["--project", "p", "--old", "old.json", "--new", "new.json"])
    [row] = json.loads((proj / "remap.json").read_text(encoding="utf-8"))
    assert row["new_start_sec"] == 3.5 and row["new_len_sec"] == 6.0  # until NEW's end 9.5


# --- veg_scan --------------------------------------------------------------

def veg_bytes(positions_sec, seed=1):
    rng = np.random.default_rng(seed)
    blob = bytearray(rng.integers(0, 256, size=4000, dtype=np.uint8).tobytes())
    for k, pos in enumerate(positions_sec):
        off = 100 + k * 211  # odd strides: real .veg fields are not 8-byte aligned
        blob[off: off + 8] = int(round(pos * veg_scan.TICKS_PER_SEC)).to_bytes(8, "little", signed=True)
    return bytes(blob)


def test_vote_finds_the_narration_zero():
    takes = [{"file": "Recording 1.mp3", "from_sec": 0.0, "to_sec": 12.5, "lag_sec": -0.477},
             {"file": "Recording 1.mp3", "from_sec": 12.5, "to_sec": 50.46, "lag_sec": 4.221},
             {"file": "Recording 4.mp3", "from_sec": 0.0, "to_sec": 25.87, "lag_sec": 102.229}]
    expected = veg_scan.expected_from_takes(takes)
    assert expected == {"Recording 1.mp3@0": -0.477, "Recording 1.mp3@12.5": 16.721,
                        "Recording 4.mp3@0": 102.229}
    zero = -0.027
    # the first piece would sit before timeline zero; only the other two are stored
    data = veg_bytes([16.721 + zero, 102.229 + zero])
    positions = veg_scan.scan_positions(data, max(expected.values()) + 1)
    votes = veg_scan.vote_zero(positions, expected)
    assert votes.most_common(1) == [(zero, 2)]


def test_lag_argument_parsing():
    assert veg_scan.parse_lag_arg("Recording 4=102.229") == ("Recording 4", 102.229)
    assert veg_scan.parse_lag_arg("a=b=1") == ("a=b", 1.0)
    with pytest.raises(ValueError):
        veg_scan.parse_lag_arg("102.229")

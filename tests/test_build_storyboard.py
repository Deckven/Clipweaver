"""Clip selection: curation wins, semantic moments fill the rest.

Guards the subtask-3 fix: the moment found at storyboard time must become the
cut range, and nothing may fall back to cross-language token overlap (which
scored 0 on RU narration vs EN catalog and picked effectively random clips).
"""

import json

import pytest

from clipmaker.build_storyboard import (
    choose_clip,
    derive_clip_range,
    load_curation,
    pick_moment,
)


def cand(name="vid.mp4", start=100.0, end=110.0, **extra):
    c = {"original_name": name, "start_sec": start, "end_sec": end,
         "timing_ok": True, "match": "semantic", "duration_sec": 600}
    c.update(extra)
    return c


# --- pick_moment -----------------------------------------------------------

def test_keyword_match_without_timing_is_not_a_moment():
    assert pick_moment([cand(start=None, end=None, match="keyword")]) is None


def test_backwards_vlm_interval_is_skipped():
    good = cand(name="b.mp4")
    assert pick_moment([cand(name="a.mp4", timing_ok=False), good]) == good


def test_video_filter_attaches_moment_to_the_curated_file():
    assert pick_moment([cand(name="other.mp4"), cand(name="mine.mp4")],
                       video="mine.mp4")["original_name"] == "mine.mp4"


# --- derive_clip_range -----------------------------------------------------

def test_moment_grows_to_fragment_duration_plus_pad():
    start, end = derive_clip_range(100.0, 110.0, frag_dur=30.0, video_dur=600)
    assert (start, end) == (100.0, 135.0)  # 30s narration + 5s headroom


def test_range_clamps_to_video_end_and_shifts_start_back():
    start, end = derive_clip_range(590.0, 595.0, frag_dur=30.0, video_dur=600)
    assert end == 600.0
    assert start == pytest.approx(565.0)  # shifted back to keep 35s of footage


def test_start_never_goes_negative():
    start, end = derive_clip_range(2.0, 5.0, frag_dur=30.0, video_dur=20)
    assert start == 0.0
    assert end == 20.0


# --- choose_clip -----------------------------------------------------------

def test_curated_range_wins_over_any_moment():
    choice = choose_clip(1, 30.0,
                         curation={1: ("cur.mp4", 40.0, 80.0)},
                         moments={1: [cand(name="sem.mp4")]},
                         catalog={})
    assert choice == {"primary_video": "cur.mp4", "clip_start": 40.0,
                      "clip_end": 80.0, "clip_source": "curation"}


def test_curated_video_without_range_borrows_its_own_moment():
    choice = choose_clip(1, 30.0,
                         curation={1: ("cur.mp4", None, None)},
                         moments={1: [cand(name="other.mp4", start=10),
                                      cand(name="cur.mp4", start=200)]},
                         catalog={"cur.mp4": {"duration_sec": 600}})
    assert choice["clip_source"] == "curation+moment"
    assert choice["clip_start"] == 200.0


def test_uncurated_fragment_takes_the_top_semantic_moment():
    choice = choose_clip(2, 20.0, curation={},
                         moments={2: [cand(name="sem.mp4", start=135, end=145)]},
                         catalog={})
    assert choice["primary_video"] == "sem.mp4"
    assert choice["clip_source"] == "moment"
    assert choice["clip_start"] == 135.0
    assert choice["clip_end"] == 160.0  # 20s + 5s pad


def test_nothing_found_stays_honestly_uncovered():
    choice = choose_clip(3, 20.0, curation={}, moments={}, catalog={})
    assert choice == {"primary_video": None, "clip_source": None}


# --- curation.json ---------------------------------------------------------

def test_load_curation_accepts_both_forms(tmp_path):
    (tmp_path / "curation.json").write_text(json.dumps({
        "1": "plain.mp4",
        "2": ["ranged.mp4", 30, 55],
    }), encoding="utf-8")
    cur = load_curation(tmp_path)
    assert cur[1] == ("plain.mp4", None, None)
    assert cur[2] == ("ranged.mp4", 30.0, 55.0)


def test_load_curation_rejects_malformed_entries(tmp_path):
    (tmp_path / "curation.json").write_text(json.dumps({"1": ["x.mp4", 5]}),
                                            encoding="utf-8")
    with pytest.raises(ValueError, match="expected a file name"):
        load_curation(tmp_path)

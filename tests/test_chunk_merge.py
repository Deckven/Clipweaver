"""Merge rules for the chunked VLM path.

Both defects covered here were found on the Black Flag Resynced round of
2026-08-08: a 15-minute Kingston mission indexed as "Menu Screen Transition"
(its first chunk was the pause menu), and an entry titled "Exploring奖励系统"
with 221 CJK characters spread across summary, tags and scene text.
"""

from pathlib import Path

import pytest

from clipmaker import vlm


# --- title selection -------------------------------------------------------

def test_title_comes_from_the_chunk_with_most_gameplay():
    candidates = [
        {"index": 0, "title": "Menu Screen Transition", "gameplay": 0},
        {"index": 1, "title": "Kingston Rooftop Chase", "gameplay": 9},
        {"index": 2, "title": "Confronting the Bishop", "gameplay": 4},
    ]
    assert vlm._pick_title(candidates) == "Kingston Rooftop Chase"


def test_title_ties_break_towards_the_earlier_chunk():
    candidates = [
        {"index": 0, "title": "Exploring Havana", "gameplay": 5},
        {"index": 1, "title": "Exploring Havana Docks", "gameplay": 5},
    ]
    assert vlm._pick_title(candidates) == "Exploring Havana"


def test_title_avoids_cjk_when_a_clean_candidate_exists():
    candidates = [
        {"index": 0, "title": "Exploring奖励系统", "gameplay": 12},
        {"index": 1, "title": "Sailing to Great Inagua", "gameplay": 3},
    ]
    assert vlm._pick_title(candidates) == "Sailing to Great Inagua"


def test_title_keeps_a_cjk_candidate_when_it_is_the_only_one():
    candidates = [{"index": 0, "title": "Exploring奖励系统", "gameplay": 12}]
    assert vlm._pick_title(candidates) == "Exploring奖励系统"


def test_title_is_empty_when_no_chunk_offered_one():
    assert vlm._pick_title([{"index": 0, "title": "  ", "gameplay": 3}]) == ""


# --- CJK detection ---------------------------------------------------------

@pytest.mark.parametrize("field, value", [
    ("title", "Exploring奖励系统"),
    ("summary", "玩家在Kingston探索"),
])
def test_chunk_cjk_detected_in_scalar_fields(field, value):
    assert vlm._chunk_has_cjk({field: value}) is True


def test_chunk_cjk_detected_in_tags_and_scene_text():
    assert vlm._chunk_has_cjk({"tags": ["kingston", "NPC交谈"]}) is True
    assert vlm._chunk_has_cjk(
        {"scenes": [{"description": "玩家跟随一名角色"}]}) is True


def test_clean_english_chunk_is_not_flagged():
    chunk = {
        "title": "Exploring Havana",
        "summary": "The player walks the docks.",
        "tags": ["havana", "docks"],
        "scenes": [{"description": "Edward climbs a crane.", "type": "gameplay"}],
    }
    assert vlm._chunk_has_cjk(chunk) is False


# --- end-to-end merge over stubbed VLM calls -------------------------------

def _chunked(monkeypatch, responses):
    """Run _run_vlm_chunked over canned per-call responses."""
    calls = []

    def fake_run_vlm(frame_paths, chunk_dur, interval, **kw):
        calls.append(kw)
        return responses[len(calls) - 1]

    monkeypatch.setattr(vlm, "run_vlm", fake_run_vlm)
    frames = [Path(f"f{i}.jpg") for i in range(100)]
    result = vlm._run_vlm_chunked(
        frames, duration_sec=500, interval=5,
        game_hint="assassins_creed_4", vocab_hint="", name_hint="clip",
        path_hints={}, ocr_lines=[], device="cpu", chunk_size=50,
    )
    return result, calls


UI_CHUNK = """{"title": "Menu Screen Transition", "summary": "The player opens the menu.",
 "tags": ["menu_screen", "interface"],
 "scenes": [{"start": "00:00", "end": "00:30", "type": "ui", "description": "Pause menu."}]}"""

GAMEPLAY_CHUNK = """{"title": "Kingston Rooftop Chase", "summary": "Edward chases the target.",
 "tags": ["kingston", "chase"],
 "scenes": [{"start": "00:00", "end": "01:00", "type": "gameplay", "description": "Rooftop run."},
            {"start": "01:00", "end": "02:00", "type": "gameplay", "description": "Alley fight."}]}"""

CJK_CHUNK = """{"title": "Exploring奖励系统", "summary": "玩家在Kingston探索。",
 "tags": ["kingston", "NPC交谈"],
 "scenes": [{"start": "00:00", "end": "01:00", "type": "gameplay", "description": "玩家跟随一名角色。"}]}"""


def test_menu_opening_does_not_name_the_video(monkeypatch):
    result, _ = _chunked(monkeypatch, [UI_CHUNK, GAMEPLAY_CHUNK])
    assert result["title"] == "Kingston Rooftop Chase"


def test_cjk_chunk_is_retried_with_different_decoding(monkeypatch):
    result, calls = _chunked(monkeypatch, [CJK_CHUNK, GAMEPLAY_CHUNK, GAMEPLAY_CHUNK])
    # chunk 1: greedy answered in Chinese, the retry profile took over.
    assert calls[0]["do_sample"] is False
    assert calls[1]["do_sample"] is True
    assert result["title"] == "Kingston Rooftop Chase"
    assert not any(vlm._has_cjk(t) for t in result["tags"])


def test_unfixable_cjk_chunk_keeps_its_scenes_but_loses_its_tags(monkeypatch):
    result, calls = _chunked(monkeypatch, [CJK_CHUNK] * 3 + [GAMEPLAY_CHUNK])
    assert len(calls) == 4, "three attempts on chunk 1, then chunk 2"
    # Coverage matters more than purity: the Chinese scene stays on the timeline.
    assert len(result["scenes"]) == 3
    # Tags are search keys — the Chinese ones would never match a query.
    assert result["tags"] == ["kingston", "chase"]
    assert not vlm._has_cjk(result["summary"])

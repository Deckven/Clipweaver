"""A catalog file flagged scene_timecodes_unreliable is searched by its timeline
windows, never by its scenes (whose timecodes point at the wrong moment)."""
from clipmaker import embed_index


def entry(**extra):
    e = {
        "game": "g", "original_name": "v.mp4", "title": "Cinematic",
        "summary": "A long cinematic with several story scenes.", "tags": ["cinematic"],
        "scenes": [{"start": "00:10", "end": "00:20", "type": "cutscene",
                    "description": "The hero walks through a burning village at night."}],
        "timeline": [{"start": "02:00", "end": "02:10",
                      "caption": "MAP; Quests; Inventory; Village Gate",  # HUD, not speech
                      "tags": ["village"]}],
    }
    e.update(extra)
    return e


def levels(docs):
    return sorted({d.level for d in docs})


def test_scenes_are_used_when_timecodes_are_fine():
    assert levels(embed_index.iter_documents(entry(), "g/v.json")) == ["scene", "video"]


def test_flagged_file_falls_back_to_timeline_windows():
    docs = embed_index.iter_documents(entry(scene_timecodes_unreliable="chunk parser bug"), "g/v.json")
    assert levels(docs) == ["video", "window"]
    window = next(d for d in docs if d.level == "window")
    assert window.start_sec == 120.0

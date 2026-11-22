"""Videos with scenes also get the timeline windows whose OCR reads like speech
(cutscene subtitles carry the names scene descriptions leave out); HUD windows
stay out."""
import pytest

from clipmaker import embed_index

SUBTITLE = "You're not aiming high enough. Starrick has influence in every branch of society"
HUD = "FOLLOW; Green; Follow; Snea; ESCAPE on the train"


@pytest.mark.parametrize("text, speech", [
    (SUBTITLE, True),
    (HUD, False),
    ("Capsule; Capsul; Capsule; Approal; No signatures", False),
    ("Ты не целишься достаточно высоко, у него влияние во всех кругах общества", True),
])
def test_looks_like_speech(text, speech):
    assert embed_index.looks_like_speech(text) is speech


def entry(*captions, scenes=True):
    tl = [{"start": f"00:{10 * i:02d}", "end": f"00:{10 * i + 9:02d}", "caption": c, "tags": []}
          for i, c in enumerate(captions)]
    e = {"game": "g", "original_name": "film.mp4", "title": "Cutscenes",
         "summary": "All cutscenes of the game in one film.", "tags": ["cutscene"], "timeline": tl}
    if scenes:
        e["scenes"] = [{"start": "00:00", "end": "00:30", "type": "cutscene",
                        "description": "Two men argue in a dark office about the city."}]
    return e


def windows(e, monkeypatch):
    monkeypatch.setattr(embed_index.config, "EMBED_WINDOW_FRAMES", 1)
    return [d for d in embed_index.iter_documents(e, "g/film.json") if d.level == "window"]


def test_video_with_scenes_keeps_only_subtitle_windows(monkeypatch):
    wins = windows(entry(HUD, SUBTITLE), monkeypatch)
    assert len(wins) == 1 and "Starrick" in wins[0].text and wins[0].start_sec == 10.0


def test_video_without_scenes_keeps_every_window(monkeypatch):
    assert len(windows(entry(HUD, SUBTITLE, scenes=False), monkeypatch)) == 2

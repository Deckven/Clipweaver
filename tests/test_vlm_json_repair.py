"""parse_json_response on the replies the VLM really gives: fenced, with trailing
commas, and cut off at max_new_tokens in the middle of the scenes array.

Before the fix the repair looked for a "timeline" array the prompt no longer
asks for, so a reply cut inside "scenes" failed all three attempts and that
stretch of the video dropped out of the catalog (review of 42afaac, item 1).
"""
import pytest

from clipmaker import vlm

HEAD = '{"title": "Kingston Chase", "summary": "Edward runs.", "tags": ["kingston", "chase"], '
SCENE_1 = '{"start": "00:00", "end": "00:20", "type": "gameplay", "description": "Rooftop run."}'
SCENE_2 = '{"start": "00:20", "end": "00:40", "type": "ui", "description": "Map with {marker} and ]."}'


def test_a_complete_reply_parses():
    reply = vlm.parse_json_response(HEAD + f'"scenes": [{SCENE_1}]}}')
    assert reply["title"] == "Kingston Chase"
    assert len(reply["scenes"]) == 1


def test_fenced_reply_with_trailing_commas_parses():
    reply = vlm.parse_json_response(f'```json\n{HEAD}"scenes": [{SCENE_1},],}}\n```')
    assert reply["scenes"][0]["description"] == "Rooftop run."


def test_reply_cut_inside_a_scene_keeps_the_complete_scenes():
    cut = HEAD + f'"scenes": [{SCENE_1}, {SCENE_2}, {{"start": "00:40", "end": "01:0'
    reply = vlm.parse_json_response(cut)
    assert reply["title"] == "Kingston Chase"
    assert [s["start"] for s in reply["scenes"]] == ["00:00", "00:20"]


def test_brackets_inside_strings_do_not_confuse_the_repair():
    cut = HEAD + f'"scenes": [{SCENE_2}, {{"start": "00:40", "description": "a {{ b ['
    reply = vlm.parse_json_response(cut)
    assert reply["scenes"][0]["description"] == "Map with {marker} and ]."


def test_reply_cut_inside_the_tags_keeps_the_complete_tags():
    cut = '{"title": "Kingston Chase", "summary": "Edward runs.", "tags": ["kingston", "cha'
    reply = vlm.parse_json_response(cut)
    assert reply["tags"] == ["kingston"]


def test_the_old_timeline_shape_is_still_repaired():
    cut = '{"title": "t", "timeline": [{"start": "00:00", "caption": "a"}, {"start": "00:1'
    assert vlm.parse_json_response(cut)["timeline"] == [{"start": "00:00", "caption": "a"}]


@pytest.mark.parametrize("raw", [
    "no json here",
    '{"title": "Kingston Ch',
])
def test_unrepairable_reply_is_an_error(raw):
    with pytest.raises(ValueError):
        vlm.parse_json_response(raw)

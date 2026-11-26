"""Loaders refuse malformed input with a message that names the file and the
entry, and small edge cases behave (R10, review of 21.11)."""
import json

import pytest

from clipmaker import align, analyzer, config, layout, naming, shot_query
from clipmaker.build_storyboard import load_curation
from clipmaker.catalog_docs import parse_timestamp
from clipmaker.contact_sheet import frame_times
from clipmaker.vegas_export import parse_timestamp as section_timestamp

GAME = "val_game"


# --- curation.json ----------------------------------------------------------------

def write_curation(tmp_path, data):
    text = data if isinstance(data, str) else json.dumps(data)
    (tmp_path / "curation.json").write_text(text, encoding="utf-8")
    return tmp_path


def test_curation_accepts_both_forms(tmp_path):
    folder = write_curation(tmp_path, {"1": "a.mp4", "2": ["b.mp4", 5, 12.5]})
    assert load_curation(folder) == {1: ("a.mp4", None, None), 2: ("b.mp4", 5.0, 12.5)}


def test_no_curation_is_empty(tmp_path):
    assert load_curation(tmp_path) == {}


@pytest.mark.parametrize("data, message", [
    ("{1: 'a.mp4'}", "not valid JSON"),
    (["a.mp4"], "expected an object"),
    ({"one": "a.mp4"}, "key 'one' is not a fragment number"),
    ({"0": "a.mp4"}, "key '0' is not a fragment number"),
    ({"1": ""}, "fragment 1: empty file name"),
    ({"1": ["a.mp4", 5]}, "fragment 1: expected a file name or"),
    ({"1": [7, 0, 5]}, "first item must be the video file name"),
    ({"1": ["a.mp4", "0:05", 9]}, "start must be a number"),
    ({"1": ["a.mp4", True, 9]}, "start must be a number"),
    ({"2": ["a.mp4", 9, 9]}, "fragment 2: range 9-9 s is empty or negative"),
    ({"2": ["a.mp4", 9, 3]}, "the end must come after the start"),
    ({"2": ["a.mp4", -1, 3]}, "empty or negative"),
])
def test_malformed_curation_names_the_fragment(tmp_path, data, message):
    with pytest.raises(ValueError, match=message):
        load_curation(write_curation(tmp_path, data))


# --- transcript.json --------------------------------------------------------------

SEG = {"start": 0.0, "end": 2.0, "text": "hello there",
       "words": [{"word": "hello", "start": 0.0, "end": 0.5}]}


def write_transcript(tmp_path, data):
    path = tmp_path / "transcript.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("data", [[SEG], {"segments": [SEG], "language": "en"}])
def test_transcript_list_or_object(tmp_path, data):
    assert align.load_transcript(write_transcript(tmp_path, data)) == [SEG]


@pytest.mark.parametrize("data, message", [
    ("[{", "not valid JSON"),
    ({"language": "en"}, "expected a list of segments"),
    (["text"], "segment 0: expected an object"),
    ([{"start": 0, "text": "x"}], "segment 0: 'end' must be a number"),
    ([SEG, {"start": "0:01", "end": 2, "text": "x"}], "segment 1: 'start' must be a number"),
    ([{"start": 3, "end": 1, "text": "x"}], "ends \\(1\\) before it starts \\(3\\)"),
    ([{"start": 0, "end": 1}], "missing \"text\""),
    ([{"start": 0, "end": 1, "text": "x", "words": [{"word": "x"}]}], "bad word entry"),
])
def test_malformed_transcript_names_the_segment(tmp_path, data, message):
    with pytest.raises(ValueError, match=message):
        align.load_transcript(write_transcript(tmp_path, data))


def test_align_cli_reports_a_bad_transcript_and_exits(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    project = tmp_path / "p"
    project.mkdir()
    (project / "storyboard.md").write_text("## Fragment 1\n\n> Hello there.\n", encoding="utf-8")
    write_transcript(project, {"segments": "oops"})
    with pytest.raises(SystemExit):
        align.main(["p"])
    assert "transcript.json: expected a list of segments" in caplog.text


@pytest.mark.parametrize("text, preview", [
    ("Short fragment.", "Short fragment."),
    ("x" * 70, "x" * 60 + "..."),
])
def test_preview_marks_only_a_real_cut(text, preview):
    assert align._preview(text) == preview


# --- catalog entry names ----------------------------------------------------------

@pytest.fixture
def videos(tmp_path, monkeypatch):
    folder = tmp_path / "videos"
    folder.mkdir()
    monkeypatch.setitem(config.GAMES, GAME, {"video_dirs": [folder],
                                             "catalog_subdir": f"{GAME}_videos"})
    return folder


def test_two_videos_with_one_stem_are_refused_before_any_work(videos):
    for name in ("Clip.mp4", "Clip.mkv", "Other.mp4"):
        (videos / name).write_bytes(b"")
    with pytest.raises(ValueError, match="Clip.json: .*Clip.mkv.*Clip.mp4"):
        analyzer.run_batch(GAME, None, force=False, interval=None, dry_run=True)


def test_an_entry_describing_another_file_of_that_stem_is_refused(videos):
    (videos / "Clip.mp4").write_bytes(b"")
    entry = analyzer.catalog_json_path(videos / "Clip.mp4", GAME)
    entry.parent.mkdir(parents=True)
    entry.write_text(json.dumps({"original_name": "Clip.mkv"}), encoding="utf-8")
    assert analyzer.catalog_collisions(GAME, [videos / "Clip.mp4"]) == [
        "Clip.json already describes Clip.mkv, not " + str(videos / "Clip.mp4")]


def test_an_entry_of_the_same_video_is_no_collision(videos):
    video = videos / "Clip.mp4"
    video.write_bytes(b"")
    entry = analyzer.catalog_json_path(video, GAME)
    entry.parent.mkdir(parents=True)
    entry.write_text(json.dumps({"original_name": "clip.MP4"}), encoding="utf-8")
    assert analyzer.catalog_collisions(GAME, [video]) == []


# --- rename vs curation.json ------------------------------------------------------

def test_rename_warns_about_curation_files_that_pick_the_old_name(videos, monkeypatch, caplog):
    video = videos / "Raw Capture 0001.mp4"
    video.write_bytes(b"")
    entry = analyzer.catalog_json_path(video, GAME)
    entry.parent.mkdir(parents=True)
    entry.write_text(json.dumps({"source_path": str(video), "original_name": video.name,
                                 "suggested_filename": "Game Harbour Fight.mp4"}),
                     encoding="utf-8")
    project = config.PROJECTS_DIR / "ep1"
    project.mkdir(parents=True)
    (project / "curation.json").write_text(
        json.dumps({"3": [video.name, 1, 4], "4": "untouched.mp4"}), encoding="utf-8")

    assert naming.rename_from_catalog(GAME, dry_run=True) == 1
    assert "curation.json picks 1 video(s) this renames (Raw Capture 0001.mp4)" in caplog.text
    assert video.exists()  # dry run


# --- small edge cases -------------------------------------------------------------

@pytest.mark.parametrize("value, seconds", [
    ("01:02", 62.0), ("1:01:02", 3662.0), ("01:02.5", 62.5), ("12.25", 12.25),
    (" 00:07 ", 7.0), (30, 30.0), (None, None), ("soon", None), ("1:xx", None), ("", None),
])
def test_catalog_timestamps_keep_fractions(value, seconds):
    assert parse_timestamp(value) == seconds


@pytest.mark.parametrize("value, seconds", [("00:01:02.5", 62.5), ("01:02.25", 62.25), ("7", 7.0)])
def test_section_timestamps_keep_fractions(value, seconds):
    assert section_timestamp(value) == seconds


def test_one_frame_is_the_middle_of_the_range():
    assert frame_times(100.0, 130.0, 1) == [115.0]
    assert frame_times(100.0, 100.0, 3) == [100.0]
    with pytest.raises(ValueError):
        frame_times(0.0, 10.0, 0)


def test_layout_labels_cannot_break_a_line_or_a_field():
    assert layout.clean_label("fr 01 a | b\nc\r\n d") == "fr 01 a / b c d"
    assert len(layout.clean_label("x" * 200)) == 80


def test_a_media_path_with_a_separator_is_refused(tmp_path):
    res = layout.LayoutResult(events=[layout.LayoutEvent(1, 0, 1000, 0, tmp_path / "a|b.mp4", "x")])
    with pytest.raises(ValueError, match="contains '\\|'"):
        layout.render_layout_txt(res, "p", "n.mp3", 60, "T", 400)


# --- why a shot query came back empty ---------------------------------------------

def test_a_reply_without_any_usable_line_falls_back_and_is_not_cached(monkeypatch, caplog):
    monkeypatch.setattr(shot_query, "_call_ollama", lambda *a, **k: "- \n* \n\"\"")
    monkeypatch.setattr(shot_query, "_warned", set())
    queries = shot_query.build_shot_queries("A paragraph about ships.", use_cache=True)
    assert [q.source for q in queries] == ["fragment"]
    assert not config.SHOT_QUERY_CACHE.exists()
    assert "no usable shot description" in caplog.text


def test_nothing_filmable_is_said_and_cached(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="clipmaker")
    monkeypatch.setattr(shot_query, "_call_ollama", lambda *a, **k: shot_query._NO_VISUAL)
    assert shot_query.build_shot_queries("Thanks for watching.", use_cache=True) == []
    assert "nothing filmable" in caplog.text
    assert json.loads(config.SHOT_QUERY_CACHE.read_text(encoding="utf-8")) != {}


def test_a_changed_prompt_or_query_limit_misses_the_cache(monkeypatch):
    # Review of 42afaac, item 5: the key was model|game|fragment, so an edited
    # system prompt, vocab or SHOT_QUERY_MAX kept serving the old answers.
    asked = []

    def ollama(fragment, system, model):
        asked.append(system)
        return "a ship in a storm"

    monkeypatch.setattr(shot_query, "_call_ollama", ollama)
    shot_query.build_shot_queries("Ships.", max_queries=3)
    shot_query.build_shot_queries("Ships.", max_queries=3)
    assert len(asked) == 1
    shot_query.build_shot_queries("Ships.", max_queries=5)
    assert len(asked) == 2
    monkeypatch.setattr(shot_query, "_SYSTEM", shot_query._SYSTEM + "\nNew rule.")
    shot_query.build_shot_queries("Ships.", max_queries=5)
    assert len(asked) == 3

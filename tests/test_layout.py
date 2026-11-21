"""Layout building and validation.

The failure this module exists to prevent: events silently overlapping,
bleeding past their fragment, or pointing at missing media — all of which
used to surface only on the Vegas timeline, at the user's expense.
"""

from pathlib import Path


from clipmaker.layout import (
    LayoutEvent,
    build_layout,
    find_overlaps,
    fmt_tc,
    quantize_ms,
    render_layout_txt,
    LayoutResult,
)

FPS = 60


def timing(num, start, dur, video="src.mp4", **extra):
    t = {
        "fragment": num,
        "start_sec": start,
        "end_sec": start + dur,
        "duration_sec": dur,
        "primary_video": video,
        "text_preview": f"fragment {num} text...",
    }
    t.update(extra)
    return t


def catalog_for(video="src.mp4", duration=600, source=None):
    return {video: {"source_path": str(source or ""), "duration_sec": duration}}


# --- frame quantization ----------------------------------------------------

def test_quantize_snaps_to_whole_frames():
    # 1/60 s = 16.666… ms; 0.02 s is closer to frame 1 than frame 2
    assert quantize_ms(0.02, 60) == 17
    assert quantize_ms(1.0, 60) == 1000


def test_fmt_tc_renders_frames():
    assert fmt_tc(1000, 60) == "00:00:01:00"
    assert fmt_tc(1017, 60) == "00:00:01:01"


def test_fmt_tc_keeps_wall_clock_at_ntsc_rates():
    # 537 s at 59.94 fps is 8:57, not 8:56 — counting frames and dividing by
    # the nominal 60 drifts half a second by the ninth minute.
    assert fmt_tc(537004, 59.94).startswith("00:08:57:")


# --- overlap detection -----------------------------------------------------

def _event(num, start_ms, length_ms):
    return LayoutEvent(fragment=num, start_ms=start_ms, length_ms=length_ms,
                       take_offset_ms=0, path=Path("x.mp4"), label=f"fr {num}")


def test_overlap_reported_when_event_runs_into_the_next():
    problems = find_overlaps([_event(1, 0, 6000), _event(2, 5000, 5000)], FPS)
    assert len(problems) == 1
    assert "fr 01" in problems[0] and "fr 02" in problems[0]


def test_one_frame_of_slack_is_not_an_overlap():
    problems = find_overlaps([_event(1, 0, 5010), _event(2, 5000, 5000)], FPS)
    assert problems == []


# --- building from timings -------------------------------------------------

def test_source_event_uses_clip_range_as_take_offset(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"")
    res = build_layout(
        [timing(1, 10.0, 20.0, clip_start=30, clip_end=55)],
        FPS, tmp_path / "clips", tmp_path / "footage",
        catalog_for(source=src), probe=lambda p: None)
    assert res.errors == []
    e = res.events[0]
    assert e.take_offset_ms == 30000
    assert e.start_ms == 10000
    assert e.length_ms == 20000  # clip is 25s, fragment 20s -> fragment wins


def test_precut_clip_wins_over_source_and_has_zero_offset(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "01_src.mp4").write_bytes(b"")
    res = build_layout(
        [timing(1, 0.0, 20.0, clip_start=30, clip_end=55)],
        FPS, clips, tmp_path / "footage",
        catalog_for(), probe=lambda p: 20.0)
    assert res.errors == []
    assert res.events[0].path.name == "01_src.mp4"
    assert res.events[0].take_offset_ms == 0


def test_short_clip_flags_a_gap(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "01_src.mp4").write_bytes(b"")
    res = build_layout(
        [timing(1, 0.0, 30.0)],
        FPS, clips, tmp_path / "footage",
        catalog_for(), probe=lambda p: 12.0)
    assert res.events[0].length_ms == 12000
    assert res.events[0].gap_ms == 18000
    assert any("gap" in w for w in res.warnings)


def test_missing_media_is_an_error_not_a_silent_skip(tmp_path):
    res = build_layout(
        [timing(1, 0.0, 20.0, clip_start=0, clip_end=20)],
        FPS, tmp_path / "clips", tmp_path / "footage",
        catalog_for(source=tmp_path / "nope.mp4"), probe=lambda p: None)
    assert res.events == []
    assert any("not found" in e for e in res.errors)


def test_source_without_clip_range_is_an_error(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"")
    res = build_layout(
        [timing(1, 0.0, 20.0)],
        FPS, tmp_path / "clips", tmp_path / "footage",
        catalog_for(source=src), probe=lambda p: None)
    assert res.events == []
    assert any("clip_start" in e for e in res.errors)


def test_fragment_without_video_is_uncovered_not_an_error(tmp_path):
    res = build_layout(
        [timing(1, 0.0, 20.0, video=None)],
        FPS, tmp_path / "clips", tmp_path / "footage", {}, probe=lambda p: None)
    assert res.errors == []
    assert len(res.uncovered) == 1


# --- layout.txt rendering --------------------------------------------------

def test_layout_txt_rows_have_five_pipe_fields_and_no_pipe_in_label():
    res = LayoutResult(events=[
        LayoutEvent(fragment=1, start_ms=0, length_ms=1000, take_offset_ms=500,
                    path=Path(r"C:\clips\01.mp4"), label="fr 01 a/b"),
    ])
    text = render_layout_txt(res, "p", "narr.mp3", 60, "Track", 400)
    row = [ln for ln in text.splitlines() if "|" in ln][0]
    assert row.split("|") == ["0", "1000", "500", "fr 01 a/b", r"C:\clips\01.mp4"]
    assert "audio=narr.mp3" in text


def test_label_pipes_are_sanitized_at_build_time(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "01_src.mp4").write_bytes(b"")
    res = build_layout(
        [timing(1, 0.0, 10.0, text_preview="a | b")],
        FPS, clips, tmp_path / "footage", catalog_for(), probe=lambda p: 10.0)
    assert "|" not in res.events[0].label


# --- explicit placements (--events) ------------------------------------------
# Map clips and document stills are placed on the spoken second, across
# fragment boundaries and several per fragment — the one-event-per-fragment
# model cannot express them.

from clipmaker.layout import events_to_timings  # noqa: E402


def test_events_become_timings_with_their_own_start_and_range(tmp_path):
    media = tmp_path / "map.mp4"
    media.write_bytes(b"x")
    rows = events_to_timings([
        {"start_sec": 57.5, "end_sec": 117.5, "path": str(media),
         "clip_start": 0, "label": "map 01"},
    ])
    res = build_layout(rows, FPS, tmp_path / "clips", tmp_path / "footage", {},
                       probe=lambda p: 60.0)
    assert not res.errors
    (ev,) = res.events
    assert ev.start_ms == quantize_ms(57.5, FPS)
    assert ev.length_ms == quantize_ms(60.0, FPS)
    assert ev.take_offset_ms == 0
    assert ev.path == media
    assert ev.label == "map 01"


def test_events_keep_the_source_offset_and_can_share_a_fragment(tmp_path):
    media = tmp_path / "wn.mp4"
    media.write_bytes(b"x")
    rows = events_to_timings([
        {"start_sec": 437.0, "end_sec": 446.3, "path": str(media), "clip_start": 83.9},
        {"start_sec": 446.3, "end_sec": 455.18, "path": str(media), "clip_start": 13.5},
    ])
    res = build_layout(rows, FPS, tmp_path / "clips", tmp_path / "footage", {})
    assert not res.errors
    assert [e.take_offset_ms for e in res.events] == [quantize_ms(83.9, FPS),
                                                       quantize_ms(13.5, FPS)]


def test_overlapping_events_are_still_rejected(tmp_path):
    media = tmp_path / "a.mp4"
    media.write_bytes(b"x")
    rows = events_to_timings([
        {"start_sec": 10, "end_sec": 20, "path": str(media), "clip_start": 0},
        {"start_sec": 15, "end_sec": 25, "path": str(media), "clip_start": 0},
    ])
    res = build_layout(rows, FPS, tmp_path / "clips", tmp_path / "footage", {})
    assert any("overlap" in e.lower() for e in res.errors)


def test_event_with_missing_media_is_an_error(tmp_path):
    rows = events_to_timings([
        {"start_sec": 1, "end_sec": 2, "path": str(tmp_path / "gone.mp4"), "clip_start": 0},
    ])
    res = build_layout(rows, FPS, tmp_path / "clips", tmp_path / "footage", {})
    assert any("not found" in e for e in res.errors)

"""Contact sheet planning and rendering (no ffmpeg involved)."""


from clipmaker.contact_sheet import frame_times, render_html, sheet_row


def timing(num=1, video="src.mp4", **extra):
    t = {"fragment": num, "start_ts": "00:00:10", "end_ts": "00:00:40",
         "duration_sec": 30.0, "text_preview": "текст фрагмента",
         "transcript_preview": "сказанное", "clip_source": "moment",
         "primary_video": video, "video_title": "Title"}
    t.update(extra)
    return t


def test_frame_times_avoid_the_exact_edges():
    times = frame_times(100.0, 130.0, 4)
    assert len(times) == 4
    assert times[0] > 100.0 and times[-1] < 130.0


def test_row_samples_source_at_the_clip_range(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"")
    row = sheet_row(timing(clip_start=100.0, clip_end=130.0),
                    tmp_path / "clips", tmp_path / "footage",
                    {"src.mp4": {"source_path": str(src)}}, frames=3)
    assert row["problem"] is None
    assert row["media"] == src
    assert all(100.0 < t < 130.0 for t in row["times"])


def test_row_without_range_reports_no_range(tmp_path):
    row = sheet_row(timing(), tmp_path / "clips", tmp_path / "footage",
                    {}, frames=3)
    assert "NO RANGE" in row["problem"]


def test_row_without_video_reports_it(tmp_path):
    row = sheet_row(timing(video=None), tmp_path / "clips",
                    tmp_path / "footage", {}, frames=3)
    assert row["problem"] == "no video assigned"


def test_precut_clip_is_sampled_from_zero(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "01_src.mp4").write_bytes(b"")
    row = sheet_row(timing(clip_start=100.0, clip_end=130.0),
                    clips, tmp_path / "footage", {}, frames=3)
    assert row["media"] == clips / "01_src.mp4"
    assert all(0 < t < 30.0 for t in row["times"])


def test_precut_of_another_pick_is_not_shown(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "01_old.mp4").write_bytes(b"")
    row = sheet_row(timing(clip_start=100.0, clip_end=130.0),
                    clips, tmp_path / "footage", {}, frames=3)
    assert row["media"] != clips / "01_old.mp4"


def test_html_marks_problem_fragments():
    ok = {"fragment": 1, "start_ts": "0", "end_ts": "1", "duration": 20.0,
          "text": "ok", "transcript": "", "source_label": "moment",
          "video": "v.mp4", "title": "", "problem": None,
          "range_label": "source 1.0-2.0s",
          "images": [("fr01_1.jpg", 1.5)]}
    bad = dict(ok, fragment=2, problem="NO RANGE", images=None)
    page = render_html("proj", [ok, bad])
    assert "frag bad" in page and "NO RANGE" in page
    assert "fr01_1.jpg" in page

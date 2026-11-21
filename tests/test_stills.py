"""Stills for article illustrations: frame grabs, black-bar crop, contact sheets.

The defects these guard against:

* a cut-scene grabbed with its letterbox bars — 270 px of black on a 1080p
  frame that the blog then shows as a thin strip;
* a YouTube frame that needed the whole video downloaded first (a 40-minute
  keynote for one still);
* a contact sheet whose tiles cannot be told apart — no timestamp, no index —
  so the chosen frame cannot be cut again at full size.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from clipmaker.stills import (
    build_grab_cmd,
    build_resolve_cmd,
    crop_from_detect,
    fetch,
    fmt_time,
    grab,
    grid_shape,
    image_sheet,
    is_remote,
    parse_time,
    sample_times,
    video_sheet,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


# --- time parsing ------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("90", 90.0),
    ("1:30", 90.0),
    ("1:02:03", 3723.0),
    ("0:44.5", 44.5),
    (12.25, 12.25),
])
def test_parse_time_accepts_seconds_and_clock_forms(value, expected):
    assert parse_time(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "abc", "1:2:3:4", "-5", "nan", "inf",
                                   "-1:90", float("nan")])
def test_parse_time_rejects_garbage(value):
    with pytest.raises(ValueError):
        parse_time(value)


def test_fmt_time_is_the_inverse_the_catalog_uses():
    assert fmt_time(3723.4) == "1:02:03"
    assert fmt_time(44.5) == "0:44"


# --- source kind ---------------------------------------------------------------

def test_remote_sources_are_urls_not_paths():
    assert is_remote("https://www.youtube.com/watch?v=abc")
    assert is_remote("http://example.com/v.mp4")
    assert not is_remote("D:/footage/clip.mp4")
    assert not is_remote(Path("clip.mp4"))


# --- black bars ------------------------------------------------------------------

CROPDETECT_LETTERBOX = (
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:1919 y1:136 y2:944 w:1920 h:800 "
    "x:0 y:138 pts:1 t:0.03 crop=1920:800:0:138\n"
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:1919 y1:135 y2:944 w:1920 h:810 "
    "x:0 y:135 pts:2 t:0.07 crop=1920:810:0:135\n"
)


def test_letterbox_is_cropped_to_the_last_detection():
    assert crop_from_detect(CROPDETECT_LETTERBOX, 1920, 1080) == (1920, 810, 0, 135)


def test_full_frame_is_left_alone():
    full = "... crop=1920:1072:0:4\n"  # 4 px of noise is not a letterbox
    assert crop_from_detect(full, 1920, 1080) is None


def test_no_detection_means_no_crop():
    assert crop_from_detect("no cropdetect lines here", 1920, 1080) is None


def test_dark_top_of_a_night_scene_is_not_a_letterbox():
    # black sky above, image below: a real letterbox is centred, this is not
    night = "... crop=1920:700:0:380\n"
    assert crop_from_detect(night, 1920, 1080) is None


def test_pillarbox_is_cropped_too():
    side = "... crop=1440:1080:240:0\n"
    assert crop_from_detect(side, 1920, 1080) == (1440, 1080, 240, 0)


# --- grab command ----------------------------------------------------------------

def test_grab_seeks_input_side_and_scales_to_width():
    cmd = build_grab_cmd("src.mp4", 83.5, Path("out.jpg"), width=1920, crop=None)
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "83.500"
    assert cmd[cmd.index("-frames:v") + 1] == "1"
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "scale='min(1920,iw)':-2"


def test_grab_applies_crop_before_scale():
    cmd = build_grab_cmd("src.mp4", 1, Path("o.jpg"), width=1920,
                         crop=(1920, 810, 0, 135))
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("crop=1920:810:0:135,")
    assert vf.endswith("scale='min(1920,iw)':-2")


def test_png_output_has_no_jpeg_quality_flag():
    cmd = build_grab_cmd("src.mp4", 1, Path("o.png"), width=1280, crop=None)
    assert "-q:v" not in cmd


def test_percent_in_output_name_is_not_a_frame_pattern():
    # image2 reads '%' as a sequence pattern unless told to write one file
    cmd = build_grab_cmd("src.mp4", 1, Path("50%_boss.jpg"), width=1280, crop=None)
    assert cmd[cmd.index("-update") + 1] == "1"
    assert cmd[-1] == "50%_boss.jpg"


def test_resolve_takes_the_video_not_its_playlist():
    cmd = build_resolve_cmd("https://www.youtube.com/watch?v=X&list=Y")
    assert "--no-playlist" in cmd
    assert cmd[-1] == "https://www.youtube.com/watch?v=X&list=Y"


# --- sheet layout ----------------------------------------------------------------

def test_sample_times_stay_inside_the_range():
    times = sample_times(100.0, 130.0, 6)
    assert len(times) == 6
    assert all(100.0 < t < 130.0 for t in times)
    assert times == sorted(times)


def test_sample_times_by_step_cover_the_range():
    times = sample_times(0.0, 100.0, step=30.0)
    assert times == [0.0, 30.0, 60.0, 90.0]


@pytest.mark.parametrize("start, end, kwargs", [
    (0.0, 100.0, {"step": -5.0}),   # used to loop forever
    (0.0, 100.0, {"step": 0.0}),
    (0.0, 100.0, {"count": 0}),
    (0.0, 0.0, {"count": 9}),       # unknown stream duration probes as 0
    (300.0, 60.0, {"count": 9}),    # --from after --to
])
def test_sample_times_rejects_ranges_that_cannot_be_sampled(start, end, kwargs):
    with pytest.raises(ValueError):
        sample_times(start, end, **kwargs)


@pytest.mark.parametrize("n, cols, expected", [
    (9, 3, (3, 3)),
    (10, 3, (3, 4)),
    (2, 4, (2, 1)),
    (1, 3, (1, 1)),
])
def test_grid_shape(n, cols, expected):
    assert grid_shape(n, cols) == expected


# --- integration (real ffmpeg) -----------------------------------------------------

@pytest.fixture(scope="module")
def letterboxed_video(tmp_path_factory):
    """10 s 1920x1080 video: a 1920x810 test pattern padded with black bars."""
    out = tmp_path_factory.mktemp("stills") / "letterbox.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=1920x810:duration=10:rate=10",
         "-vf", "pad=1920:1080:0:135:black",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_grab_removes_letterbox(letterboxed_video, tmp_path):
    out = tmp_path / "still.jpg"
    result = grab(str(letterboxed_video), 5.0, out, width=1920, crop="auto")
    assert result == out
    with Image.open(out) as img:
        width, height = img.size
    assert width == 1920
    assert 800 <= height <= 816  # cropdetect rounds to 16 px steps


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_grab_without_crop_keeps_the_bars(letterboxed_video, tmp_path):
    out = tmp_path / "still.jpg"
    grab(str(letterboxed_video), 5.0, out, width=960, crop="none")
    with Image.open(out) as img:
        assert img.size == (960, 540)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_video_sheet_labels_every_tile(letterboxed_video, tmp_path):
    out = tmp_path / "sheet.jpg"
    tiles = video_sheet(str(letterboxed_video), [1.0, 4.0, 7.0, 9.0], out,
                        cols=2, tile_width=320)
    assert [t["label"] for t in tiles] == ["1 · 0:01", "2 · 0:04",
                                           "3 · 0:07", "4 · 0:09"]
    with Image.open(out) as img:
        assert img.size[0] == 2 * 320


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_video_sheet_fails_loudly_when_no_frame_came_out(tmp_path):
    with pytest.raises(RuntimeError):
        video_sheet(str(tmp_path / "missing.mp4"), [1.0, 2.0], tmp_path / "s.jpg")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_grab_does_not_report_a_stale_file_as_success(letterboxed_video, tmp_path):
    out = tmp_path / "still.jpg"
    Image.new("RGB", (10, 10)).save(out)
    with pytest.raises(RuntimeError):
        grab(str(letterboxed_video), 500.0, out, crop="none")  # past the end
    assert not out.exists()


class _FakeResponse:
    def __init__(self, content, content_type):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        pass


def test_fetch_converts_to_the_requested_format(tmp_path, monkeypatch):
    import io
    buf = io.BytesIO()
    Image.new("RGBA", (64, 36), "red").save(buf, format="PNG")
    monkeypatch.setattr("clipmaker.stills.requests.get",
                        lambda *a, **k: _FakeResponse(buf.getvalue(), "image/png"))
    out = fetch("https://example.com/a.png", tmp_path / "a.jpg")
    with Image.open(out) as img:
        assert img.format == "JPEG"


def test_fetch_rejects_vector_and_html(tmp_path, monkeypatch):
    monkeypatch.setattr("clipmaker.stills.requests.get",
                        lambda *a, **k: _FakeResponse(b"<svg/>", "image/svg+xml"))
    with pytest.raises(RuntimeError):
        fetch("https://example.com/a.svg", tmp_path / "a.jpg")


def test_image_sheet_indexes_files_in_order(tmp_path):
    paths = []
    for i, color in enumerate(["red", "green", "blue"]):
        p = tmp_path / f"shot_{i}.png"
        Image.new("RGB", (800, 450), color).save(p)
        paths.append(p)
    out = tmp_path / "sheet.jpg"
    tiles = image_sheet(paths, out, cols=2, tile_width=200)
    assert [t["path"] for t in tiles] == paths
    assert [t["label"] for t in tiles] == ["1 · shot_0.png", "2 · shot_1.png",
                                           "3 · shot_2.png"]
    with Image.open(out) as img:
        assert img.size[0] == 400

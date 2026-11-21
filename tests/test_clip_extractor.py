"""Frame-accurate clip cutting.

The defect this guards against: ``-ss`` before ``-i`` with ``-c copy`` snapped
the cut to the previous keyframe. Long-GOP game captures shifted clips by up
to ~10 s, so a correctly chosen moment still landed wrong on the timeline.
"""

import json
import subprocess
from pathlib import Path

import pytest

from clipmaker.clip_extractor import build_cut_cmd, extract_clip


# --- command construction --------------------------------------------------

def test_seek_is_input_side_with_reencode():
    cmd = build_cut_cmd(Path("src.mp4"), 3.5, 5.5, Path("out.mp4"), "libx264")
    # accurate seek: -ss comes before -i, duration via -t, and no stream copy
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "3.500"
    assert cmd[cmd.index("-t") + 1] == "2.000"
    assert "copy" not in cmd
    assert "libx264" in cmd


def test_nvenc_variant_uses_nvenc():
    cmd = build_cut_cmd(Path("s.mp4"), 0, 10, Path("o.mp4"), "h264_nvenc")
    assert "h264_nvenc" in cmd


# --- real cut accuracy -----------------------------------------------------

def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def long_gop_source(tmp_path_factory) -> Path:
    """10 s synthetic video with a single keyframe — worst case for copy cuts."""
    src = tmp_path_factory.mktemp("cut") / "long_gop.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=duration=10:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
         "-c:v", "libx264", "-g", "600", "-keyint_min", "600",
         "-c:a", "aac", str(src)],
        capture_output=True, timeout=120, check=True)
    return src


def test_mid_gop_cut_has_the_requested_duration(long_gop_source, tmp_path):
    out = tmp_path / "clip.mp4"
    assert extract_clip(long_gop_source, 3.5, 5.5, out)
    info = _probe(out)
    duration = float(info["format"]["duration"])
    # the old copy path could only start at the keyframe (0.0) -> ~5.5-10 s out
    assert duration == pytest.approx(2.0, abs=0.15)


def test_cut_keeps_the_audio_stream(long_gop_source, tmp_path):
    out = tmp_path / "clip_audio.mp4"
    assert extract_clip(long_gop_source, 1.0, 3.0, out)
    kinds = {s["codec_type"] for s in _probe(out)["streams"]}
    assert kinds == {"video", "audio"}

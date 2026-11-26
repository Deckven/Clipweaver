"""Errors are reported, never swallowed (review of 21.11, R2).

Each test pins one place that used to drop a failure without a word: an
``except Exception: continue`` around the VLM parser, OCR that returned empty
text for every frame when EasyOCR crashed, a batch that printed FAILED and
still exited 0.
"""
import json
import logging
import sys
import types

import pytest

from clipmaker import analyzer, ocr, vlm
from clipmaker.log import setup_logging

GAME = "errors_game"


# --- logging setup ---------------------------------------------------------------

def test_cli_logging_goes_to_stderr_and_leaves_stdout_clean(capsys):
    setup_logging(verbose=False)
    log = logging.getLogger("clipmaker.some_module")
    log.debug("hidden")
    log.info("progress line")
    log.warning("careful")
    out, err = capsys.readouterr()
    assert out == ""
    assert "progress line" in err and "WARNING: careful" in err
    assert "hidden" not in err


def test_verbose_shows_debug_and_the_module_name(capsys):
    setup_logging(verbose=True)
    logging.getLogger("clipmaker.some_module").debug("detail")
    assert "clipmaker.some_module: detail" in capsys.readouterr().err
    setup_logging(verbose=False)


def test_setup_twice_does_not_print_every_line_twice(capsys):
    setup_logging()
    setup_logging()
    logging.getLogger("clipmaker.x").info("once")
    assert capsys.readouterr().err.count("once") == 1


# --- VLM output parsing ---------------------------------------------------------

@pytest.mark.parametrize("raw", ["I cannot see any frames.", '{"title": "x", "tags": [}'])
def test_unparseable_output_is_a_value_error(raw):
    # The callers now catch ValueError only; anything else is a bug and surfaces.
    with pytest.raises(ValueError):
        vlm.parse_json_response(raw)


def test_truncated_object_is_still_repaired():
    raw = '{"title": "Fight", "tags": ["a"], "timeline": [{"start": "00:00"}, {"sta'
    assert vlm.parse_json_response(raw)["title"] == "Fight"


# --- OCR ------------------------------------------------------------------------

@pytest.fixture
def fake_cv2(monkeypatch):
    """EasyOCR's cv2 without installing the ocr extra."""
    cv2 = types.SimpleNamespace(error=type("error", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


class _Reader:
    def __init__(self, fail_on: set[str]):
        self.fail_on = fail_on

    def readtext(self, img):
        if img in self.fail_on:
            raise RuntimeError("CUDA out of memory")
        return [(None, f"text of {img}", 0.9)]


def _frames(tmp_path, n):
    return [tmp_path / f"frame_{i}.jpg" for i in range(n)]


def test_ocr_failure_on_one_frame_is_logged_and_the_rest_kept(tmp_path, monkeypatch, fake_cv2, caplog):
    frames = _frames(tmp_path, 3)
    monkeypatch.setattr(ocr, "_imread_unicode", lambda p: p.name)
    monkeypatch.setattr(ocr, "_get_reader", lambda *a: _Reader({"frame_1.jpg"}))
    result = ocr.extract_frame_text(frames)
    assert result == [["text of frame_0.jpg"], [], ["text of frame_2.jpg"]]
    assert "OCR failed on frame_1.jpg" in caplog.text


def test_ocr_failing_on_every_frame_raises(tmp_path, monkeypatch, fake_cv2):
    frames = _frames(tmp_path, 2)
    monkeypatch.setattr(ocr, "_imread_unicode", lambda p: p.name)
    monkeypatch.setattr(ocr, "_get_reader", lambda *a: _Reader({f.name for f in frames}))
    with pytest.raises(RuntimeError, match="OCR failed on all 2 frames"):
        ocr.extract_frame_text(frames)


# --- batch ----------------------------------------------------------------------

@pytest.fixture
def batch(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("good.mp4", "bad.mp4"):
        (videos / name).write_bytes(b"")
    monkeypatch.setitem(analyzer.config.GAMES, GAME, {
        "catalog_subdir": f"{GAME}_videos", "video_dirs": [videos],
    })
    monkeypatch.setattr(analyzer, "_cuda_available", lambda: False)

    def analyze(video, **_):
        if video.name == "bad.mp4":
            raise RuntimeError("ffprobe failed: moov atom not found")
        return {"suggested_filename": "Good 1.mp4"}

    monkeypatch.setattr(analyzer, "analyze_video", analyze)
    return videos


def test_batch_counts_failed_videos(batch, caplog):
    assert analyzer.run_batch(GAME, None, force=False, interval=None) == 1
    assert "FAILED bad.mp4: ffprobe failed: moov atom not found" in caplog.text
    good = analyzer.catalog_json_path(batch / "good.mp4", GAME)
    assert json.loads(good.read_text(encoding="utf-8"))["suggested_filename"] == "Good 1.mp4"


def test_cli_exits_non_zero_when_a_video_failed(batch):
    with pytest.raises(SystemExit) as exc:
        analyzer.main(["--game", GAME])
    assert exc.value.code == 1


def test_missing_explicit_path_stops_before_any_work(batch, tmp_path, caplog):
    with pytest.raises(SystemExit):
        analyzer.main(["--game", GAME, str(tmp_path / "typo.mp4")])
    assert "1 of 1 paths do not exist" in caplog.text


def test_unknown_game_is_a_value_error_for_library_callers():
    with pytest.raises(ValueError, match="Unknown game"):
        analyzer.run_batch("no_such_game", None, force=False, interval=None)

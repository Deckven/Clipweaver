"""analyze_video on a real 24-second test video with the VLM and OCR stubbed (R9).

These pin the record the analyzer writes - single call and chunked, the retry
ladder, OCR timeline - so analyzer.py can be split without changing it (R1);
and the two failure rules: frames are removed even when the VLM fails, and a
broken input reports ffprobe's own message.
"""
import json
import shutil
import subprocess

import pytest

from clipmaker import analyzer, config, ocr, vlm

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not on PATH")

GAME = "av_game"
DURATION = 24


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    folder = tmp_path_factory.mktemp("footage") / "Harbour"
    folder.mkdir()
    path = folder / "2026.04.11 12.59.23.08 capture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=duration={DURATION}:size=320x180:rate=10",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def setup(monkeypatch, video):
    monkeypatch.setitem(config.GAMES, GAME, {
        "video_dirs": [video.parent.parent], "catalog_subdir": f"{GAME}_videos",
        "display_name": "Test Game",
    })
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    monkeypatch.setattr(config, "VLM_CHUNK_SIZE", 50)
    monkeypatch.setattr(analyzer, "vocab_for", lambda game: {})
    return video


def answer(title="Galleon broadside in a storm", scenes=None, tags=None, summary="A sea fight."):
    return json.dumps({
        "title": title, "summary": summary,
        "tags": tags if tags is not None else ["naval_battle", "storm", "gameplay", "123"],
        "scenes": scenes if scenes is not None else [
            {"start": "00:00", "end": "00:10", "type": "gameplay", "description": "ships close in"},
            {"start": "00:10", "end": "01:30", "type": "gameplay", "description": "broadside"},
        ],
    })


class FakeVlm:
    """run_vlm stand-in: replies in order, records what each call was given."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, frames, duration, interval, **kwargs):
        self.calls.append({"frames": len(frames), "duration": duration, **kwargs})
        return self.replies.pop(0)


def frame_dir(video):
    return config.VLM_WORK_DIR / video.stem


def test_single_call_record(setup, monkeypatch):
    fake = FakeVlm(answer())
    monkeypatch.setattr(vlm, "run_vlm", fake)
    rec = analyzer.analyze_video(setup, GAME, interval=4, device="cpu")

    assert len(fake.calls) == 1 and fake.calls[0]["frames"] == DURATION // 4
    assert fake.calls[0]["do_sample"] is True and fake.calls[0]["temperature"] == 0.1
    assert rec["duration_sec"] == DURATION and rec["frame_interval_sec"] == 4
    assert rec["n_frames_analyzed"] == DURATION // 4
    assert rec["game"] == GAME and rec["original_name"] == setup.name
    assert rec["title"] == "Galleon broadside in a storm" and rec["summary"] == "A sea fight."
    # noise tags dropped, the folder name is injected as the location
    assert rec["tags"] == ["harbour", "naval_battle", "storm"]
    assert rec["path_hints"]["location"] == "Harbour"
    # the second scene ran past the 24 s video: clamped, not dropped
    assert rec["scenes"][1]["start"] == "00:10" and rec["scenes"][1]["end"] == "00:24"
    assert [s["start"] for s in rec["timeline"]] == ["00:00"]  # no OCR: one empty caption
    assert rec["suggested_filename"].startswith("Test Game Harbour Galleon")
    assert rec["suggested_filename"].endswith(".mp4")
    assert not frame_dir(setup).exists()


def test_adaptive_interval_respects_the_frame_cap(setup, monkeypatch):
    monkeypatch.setattr(vlm, "run_vlm", FakeVlm(answer()))
    rec = analyzer.analyze_video(setup, GAME, max_frames=3, device="cpu")
    assert rec["frame_interval_sec"] == 8 and rec["n_frames_analyzed"] == 3


def test_retry_ladder_changes_decoding_until_the_json_parses(setup, monkeypatch):
    fake = FakeVlm("!!!!!!!!", "Here you go: {broken", answer())
    monkeypatch.setattr(vlm, "run_vlm", fake)
    rec = analyzer.analyze_video(setup, GAME, interval=4, device="cpu")
    assert [c["temperature"] for c in fake.calls] == [0.1, 0.4, 0.7]
    assert rec["title"] == "Galleon broadside in a storm"


def test_all_attempts_failing_saves_the_raw_reply_and_still_removes_frames(setup, monkeypatch):
    monkeypatch.setattr(vlm, "run_vlm", FakeVlm("no", "still no", "never"))
    with pytest.raises(RuntimeError, match="All 3 attempts failed"):
        analyzer.analyze_video(setup, GAME, interval=4, device="cpu")
    debug = config.CATALOG_DIR / "_debug" / f"{setup.stem}_raw.txt"
    assert debug.read_text(encoding="utf-8") == "never"
    assert not frame_dir(setup).exists()


def test_a_crash_inside_the_vlm_still_removes_frames(setup, monkeypatch):
    def crash(*_a, **_k):
        raise MemoryError("CUDA out of memory")

    monkeypatch.setattr(vlm, "run_vlm", crash)
    with pytest.raises(MemoryError):
        analyzer.analyze_video(setup, GAME, interval=4, device="cpu")
    assert not frame_dir(setup).exists()


def test_keep_frames_keeps_them(setup, monkeypatch):
    monkeypatch.setattr(vlm, "run_vlm", FakeVlm(answer()))
    analyzer.analyze_video(setup, GAME, interval=4, device="cpu", keep_frames=True)
    assert len(list(frame_dir(setup).glob("frame_*.jpg"))) == DURATION // 4
    analyzer.remove_frames(frame_dir(setup))


def test_chunked_path_offsets_scenes_and_merges_chunks(setup, monkeypatch):
    monkeypatch.setattr(config, "VLM_CHUNK_SIZE", 3)  # 6 frames -> 2 chunks of 12 s
    first = answer(title="Menu screen", tags=["menu"],
                   scenes=[{"start": "00:00", "end": "00:12", "type": "ui", "description": "menu"}])
    second = answer(title="Galleon broadside", tags=["naval_battle"], summary="The fight.",
                    scenes=[{"start": "00:02", "end": "00:30", "type": "gameplay", "description": "guns"}])
    fake = FakeVlm(first, second)
    monkeypatch.setattr(vlm, "run_vlm", fake)
    rec = analyzer.analyze_video(setup, GAME, interval=4, device="cpu")

    assert [c["frames"] for c in fake.calls] == [3, 3]
    assert fake.calls[1]["name_hint"].endswith("(part 2/2)")
    assert rec["title"] == "Galleon broadside"  # the gameplay chunk names the video
    assert rec["summary"] == "A sea fight. The fight."
    assert rec["tags"] == ["harbour", "menu", "naval_battle"]
    # chunk 2 starts at 12 s; its scene end is clamped to the chunk
    assert [(s["start"], s["end"]) for s in rec["scenes"]] == [("00:00", "00:12"), ("00:14", "00:24")]


def test_ocr_lines_feed_the_prompt_and_the_timeline(setup, monkeypatch):
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    lines = [["Warp Drive Active"], ["Warp Drive Active"], [], ["Jita IV - Moon 4"], [], []]
    monkeypatch.setattr(ocr, "extract_frame_text", lambda frames, **k: lines[:len(frames)])
    monkeypatch.setattr(ocr, "release", lambda: None)
    fake = FakeVlm(answer())
    monkeypatch.setattr(vlm, "run_vlm", fake)
    rec = analyzer.analyze_video(setup, GAME, interval=4, device="cpu")

    assert rec["ocr_text"] == ["Warp Drive Active", "Jita IV - Moon 4"]
    assert fake.calls[0]["ocr_lines"] == rec["ocr_text"]
    captions = [(s["start"], s["end"], s["caption"]) for s in rec["timeline"]]
    assert captions[0] == ("00:00", "00:08", "Warp Drive Active")
    assert ("00:12", "00:16", "Jita IV - Moon 4") in captions


def test_a_file_that_is_not_a_video_reports_ffprobes_message(tmp_path, setup):
    fake = tmp_path / "notes.mp4"
    fake.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        analyzer.analyze_video(fake, GAME, device="cpu")
    message = str(exc.value)
    assert message.startswith("ffprobe failed to read the duration of notes.mp4 (exit 1)")
    assert "Invalid data" in message  # ffprobe's own words, not just the exit status


def test_missing_ffmpeg_is_named(monkeypatch, setup):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="ffprobe is not on PATH"):
        analyzer.get_video_duration(setup)

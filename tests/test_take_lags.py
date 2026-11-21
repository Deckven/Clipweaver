"""clipmaker.take_lags and clipmaker.render_offset on synthetic audio.

Moved from projects/*/tools/take_lags.py, v2_takes.py and render_offset.py.
The signal is speech-like: noise under a random syllable envelope with pauses,
so the 10 ms log-energy envelope has the structure real narration has.
"""

import numpy as np
import pytest

from clipmaker import render_offset, take_lags


def speech_like(seconds: float, sr: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    syll = int(0.12 * sr)
    levels = rng.choice([0.0005, 0.05, 0.2, 0.5], size=n // syll + 1, p=[0.25, 0.25, 0.25, 0.25])
    env = np.repeat(levels, syll)[:n]
    return (rng.standard_normal(n) * env).astype(np.float32)


@pytest.fixture(scope="module")
def mix():
    return speech_like(60, take_lags.SR)


def test_a_clean_take_has_one_segment_at_its_lag(mix):
    sr = take_lags.SR
    take = mix[int(12.5 * sr): int(30.5 * sr)]
    segs = take_lags.group_segments(take_lags.window_hits(take, mix))
    assert len(segs) == 1
    assert segs[0]["lag_sec"] == pytest.approx(12.5, abs=0.002)
    assert segs[0]["from_sec"] == 0.0


def test_a_cut_take_splits_into_segments_with_their_own_lags(mix):
    sr = take_lags.SR
    # the take holds 8 s that the mix places at 5 s, then 10 s it places at 40 s
    take = np.concatenate([mix[5 * sr: 13 * sr], mix[40 * sr: 50 * sr]])
    segs = take_lags.group_segments(take_lags.window_hits(take, mix))
    lags = [s["lag_sec"] for s in segs if s["windows"] >= 3]
    assert lags == [pytest.approx(5.0, abs=0.002), pytest.approx(32.0, abs=0.002)]


def test_v2_takes_sampling_is_a_parameter(mix):
    sr = take_lags.SR
    take = mix[20 * sr: 32 * sr]
    hits = take_lags.window_hits(take, mix, win_s=4.0, step_s=2.0)
    assert [h[0] for h in hits] == [0.0, 2.0, 4.0, 6.0]
    assert all(h[1] == pytest.approx(20.0, abs=0.002) for h in hits)


def test_silence_matches_nothing(mix):
    assert take_lags.window_hits(np.zeros(10 * take_lags.SR, np.float32), mix) == []


def test_globs_are_expanded_here(tmp_path):
    for n in ("Recording 1.mp3", "Recording 2.mp3", "Final.mp3"):
        (tmp_path / n).write_bytes(b"")
    got = take_lags.expand_paths([str(tmp_path / "Recording *.mp3"), "literal.mp3"])
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in got] == \
        ["Recording 1.mp3", "Recording 2.mp3", "literal.mp3"]


def test_render_offset_finds_a_constant_shift():
    sr = render_offset.SR
    nar = speech_like(80, sr, seed=3)
    ren = np.concatenate([np.zeros(int(1.75 * sr), np.float32), nar])
    for t in render_offset.probe_times(80, count=4):
        off, score = render_offset.offset_at(nar, ren, t)
        assert off == pytest.approx(1.75, abs=0.001)
        assert score > 0.99


def test_render_offset_skips_silence_and_edges():
    sr = render_offset.SR
    nar = np.zeros(20 * sr, np.float32)
    assert render_offset.offset_at(nar, nar, 5) is None
    assert render_offset.probe_times(3, 10) == []

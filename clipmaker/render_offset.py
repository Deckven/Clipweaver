"""How far a rendered video's soundtrack is shifted against the narration mp3.

YouTube chapters and description timecodes are computed on the narration; the
uploaded file is a render. A 4 s narration window at each probe time is found
in the render's audio (normalised cross-correlation, searched from 30 s before
to 40 s after); a constant offset means the timecodes just shift, a drifting
one means the render is not the narration that was aligned.

Usage:
    python -m clipmaker.render_offset NARRATION.mp3 RENDER.mp4 [--at T ...] [--count 10]
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

from .stills import parse_time
from .take_lags import decode_audio, ncc

SR = 8000
WIN_S = 4.0
SEARCH_BEFORE_S = 30.0
SEARCH_AFTER_S = 40.0
DEFAULT_COUNT = 10


def probe_times(duration: float, count: int = DEFAULT_COUNT, win_s: float = WIN_S) -> list[float]:
    """``count`` probe starts spread over the narration, every window inside it."""
    last = duration - win_s
    if last <= 0 or count <= 0:
        return []
    if count == 1:
        return [0.0]
    return [round(i * last / (count - 1), 1) for i in range(count)]


def offset_at(narration: np.ndarray, render: np.ndarray, t: float, sr: int = SR,
              win_s: float = WIN_S) -> tuple[float, float] | None:
    """(render time - narration time, score) for the window at t, or None when
    the window is silent or falls outside either signal."""
    w = narration[int(t * sr): int((t + win_s) * sr)]
    if len(w) < int(win_s * sr) or w.std() < 1e-6:
        return None
    a = max(0, int((t - SEARCH_BEFORE_S) * sr))
    b = min(len(render), int((t + SEARCH_AFTER_S) * sr))
    if b - a <= len(w):
        return None
    r = ncc(w, render[a:b])
    i = int(np.argmax(r))
    return (a + i) / sr - t, float(r[i])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clipmaker.render_offset", description=__doc__.split("\n")[0])
    ap.add_argument("narration")
    ap.add_argument("render")
    ap.add_argument("--at", nargs="+", help="probe times (default: spread over the narration)")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = ap.parse_args(argv)

    try:
        nar = decode_audio(args.narration, SR)
        ren = decode_audio(args.render, SR)
        times = [parse_time(t) for t in args.at] if args.at else probe_times(len(nar) / SR, args.count)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"error: {exc}")
    print(f"narration {len(nar) / SR:.2f}s  render {len(ren) / SR:.2f}s")
    for t in times:
        hit = offset_at(nar, ren, t)
        if hit is None:
            print(f"narration {t:7.1f}s -> (silent or out of range)")
            continue
        off, score = hit
        print(f"narration {t:7.1f}s -> render {t + off:8.3f}s  offset {off:+7.3f}  r={score:.2f}")


if __name__ == "__main__":
    main()

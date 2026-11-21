"""Where each raw take sits inside the merged narration mp3.

The narration is recorded as raw takes (Recording N.mp3, Changes N.mp3, ...)
that the author cuts and arranges on the Vegas timeline, then renders into
one mix; the timings are aligned to that mix. When the timeline holds the raw
takes instead of the mix, the Vegas loader anchors the B-roll on the takes —
it needs each take piece's lag, measured here.

Slides windows (default 3 s, 1 s step) over every take, finds each window in
the mix by normalised cross-correlation of 10 ms log-energy envelopes, refines
on the raw 16 kHz signal, and groups consecutive windows with the same lag:
    take  take_from  take_to  lag (mix = take + lag)  score
A take with internal cuts shows up as several segments with different lags.
This replaces the per-project v2_takes.py (4 s windows every 2 s, raw-signal
FFT): `--window 4 --step 2` reproduces its sampling.

Usage:
    python -m clipmaker.take_lags MIX.mp3 TAKE.mp3 [TAKE.mp3 ...] [--json takes.json]

Globs work on Windows too ("Recording *.mp3"). --json writes the segments in
the shape of the "takes" list of a project's vegas.json; trim spurious short
segments and set the piece boundaries before pasting them there.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 16000
HOP = 160              # 10 ms envelope frames
FRAMES_PER_SEC = SR // HOP
WIN_S, STEP_S = 3.0, 1.0
MIN_SCORE = 0.80
SAME_LAG = 0.03        # s: windows closer than this share a segment
SILENCE_STD = 0.5      # log-energy spread below this is room tone
REFINE_PAD_S = 0.05


def decode_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Mono float32 samples in [-1, 1] via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def envelope(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    fr = x[: n * HOP].reshape(n, HOP)
    return np.log(np.sqrt((fr ** 2).mean(axis=1)) + 1e-4)


def ncc(win: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of win against every position of sig."""
    # float64: running sums over minutes of float32 audio lose the window energy
    win = np.asarray(win, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    m = len(win)
    w = (win - win.mean()) / (win.std() + 1e-9)
    nfft = 1 << int(np.ceil(np.log2(len(sig) + m)))
    num = np.fft.irfft(np.fft.rfft(sig, nfft) * np.conj(np.fft.rfft(w, nfft)), nfft)[: len(sig) - m + 1]
    c1 = np.concatenate([[0], np.cumsum(sig)])
    c2 = np.concatenate([[0], np.cumsum(sig ** 2)])
    s = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    std = np.sqrt(np.maximum(s2 / m - (s / m) ** 2, 1e-12))
    return num / (m * std)


def refine(take: np.ndarray, mix: np.ndarray, t0: float, lag: float,
           win_s: float = WIN_S, sr: int = SR) -> float:
    """Sample-accurate lag near a coarse one, on the raw signal."""
    a = int(t0 * sr)
    seg = take[a: a + int(win_s * sr)]
    pad = int(REFINE_PAD_S * sr)
    b = int((t0 + lag) * sr) - pad
    if b < 0 or b + len(seg) + 2 * pad > len(mix):
        return lag
    r = ncc(seg, mix[b: b + len(seg) + 2 * pad])
    return (b + int(np.argmax(r))) / sr - t0


def window_hits(take: np.ndarray, mix: np.ndarray, win_s: float = WIN_S,
                step_s: float = STEP_S, min_score: float = MIN_SCORE,
                sr: int = SR) -> list[tuple[float, float, float]]:
    """(take_time, lag, score) for every take window found in the mix."""
    menv, tenv = envelope(mix), envelope(take)
    wn, st = int(win_s * FRAMES_PER_SEC), max(1, int(step_s * FRAMES_PER_SEC))
    hits = []
    for i in range(0, len(tenv) - wn, st):
        w = tenv[i: i + wn]
        if w.std() < SILENCE_STD:
            continue
        r = ncc(w, menv)
        j = int(np.argmax(r))
        if r[j] < min_score:
            continue
        t0 = i / FRAMES_PER_SEC
        lag = refine(take, mix, t0, j / FRAMES_PER_SEC - t0, win_s, sr)
        hits.append((t0, lag, float(r[j])))
    return hits


def group_segments(hits: list[tuple[float, float, float]], win_s: float = WIN_S,
                   same_lag: float = SAME_LAG) -> list[dict]:
    """Consecutive windows with the same lag -> one take segment."""
    segs: list[dict] = []
    for t0, lag, sc in hits:
        if segs and abs(segs[-1]["lags"][0] - lag) < same_lag:
            s = segs[-1]
            s["to"] = t0 + win_s
            s["n"] += 1
            s["score"] = min(s["score"], sc)
            s["lags"].append(lag)
        else:
            segs.append({"from": t0, "to": t0 + win_s, "n": 1, "score": sc, "lags": [lag]})
    return [{"from_sec": round(s["from"], 2), "to_sec": round(s["to"], 2),
             "lag_sec": round(float(np.median(s["lags"])), 3),
             "windows": s["n"], "min_score": round(s["score"], 3)} for s in segs]


def expand_paths(items: list[str]) -> list[str]:
    """Shell-style globs expanded here: PowerShell passes them through literally."""
    out = []
    for item in items:
        matched = sorted(glob.glob(item)) if any(c in item for c in "*?[") else []
        out.extend(matched or [item])
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clipmaker.take_lags", description=__doc__.split("\n")[0])
    ap.add_argument("mix", help="the merged narration the timings were aligned to")
    ap.add_argument("takes", nargs="+", help="raw takes (globs allowed)")
    ap.add_argument("--window", type=float, default=WIN_S, help="window, s (default %(default)s)")
    ap.add_argument("--step", type=float, default=STEP_S, help="window step, s (default %(default)s)")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the segments as a vegas.json-style takes list")
    args = ap.parse_args(argv)
    if args.window <= 0 or args.step <= 0:
        ap.error("--window and --step must be positive")

    try:
        mix = decode_audio(args.mix)
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"error: cannot decode {args.mix}: {exc}")
    print(f"mix {len(mix) / SR:.2f}s")
    out = []
    for path in expand_paths(args.takes):
        try:
            take = decode_audio(path)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"\n{path}: cannot decode ({exc})")
            continue
        hits = window_hits(take, mix, args.window, args.step, args.min_score)
        segs = group_segments(hits, args.window)
        name = Path(path).name
        print(f"\n{name}  ({len(take) / SR:.2f}s, {len(hits)} matched windows)")
        for s in segs:
            print(f"  take {s['from_sec']:7.2f}-{s['to_sec']:7.2f}s  lag {s['lag_sec']:9.3f}"
                  f"  (n={s['windows']}, min score {s['min_score']:.2f})")
            out.append({"file": name, **s})
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n{len(out)} segments -> {args.json}")


if __name__ == "__main__":
    main()

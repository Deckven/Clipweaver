"""Find where the narration zero sits in a .veg, from the take lags alone.

A .veg stores event positions as little-endian int64 in 100 ns units. Every
take piece that starts at its segment's beginning lies on the timeline at
narration_zero + lag + from. Scanning every 8-byte window of the file for
values near those expected positions and voting on the difference gives the
narration zero — a check of the lag table against the project itself,
without opening Vegas. In the projects measured so far the pieces sat at
exactly lag - 0.027 s, i.e. the winning delta was -0.027.

Usage:
    python -m clipmaker.veg_scan PROJECT.veg --project <p>         # lags from <p>/vegas.json
    python -m clipmaker.veg_scan PROJECT.veg --lag "Recording 4=102.229" [--lag ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

from . import config
from .log import add_verbose_flag, setup_logging


TICKS_PER_SEC = 10_000_000
TOLERANCE_SEC = 1.0
SHOWN_DELTAS = 8


def scan_positions(data: bytes, max_sec: float) -> np.ndarray:
    """Every int64 (any byte alignment) in [0, max_sec], as seconds."""
    chunks = []
    for k in range(8):
        n = (len(data) - k) // 8
        if n <= 0:
            continue
        vals = np.frombuffer(data, dtype="<i8", count=n, offset=k)
        chunks.append(vals[(vals >= 0) & (vals <= max_sec * TICKS_PER_SEC)])
    if not chunks:
        return np.zeros(0)
    return np.concatenate(chunks) / TICKS_PER_SEC


def deltas_near(positions: np.ndarray, expected: float,
                tolerance: float = TOLERANCE_SEC) -> list[float]:
    """Distinct (position - expected), ms-rounded, for positions within tolerance."""
    d = positions[np.abs(positions - expected) < tolerance] - expected
    return sorted({round(float(x), 3) for x in d})


def vote_zero(positions: np.ndarray, expected: dict[str, float],
              tolerance: float = TOLERANCE_SEC) -> collections.Counter:
    """How many take pieces support each candidate narration zero."""
    votes: collections.Counter = collections.Counter()
    for pos in expected.values():
        votes.update(deltas_near(positions, pos, tolerance))
    return votes


def expected_from_takes(takes: list[dict]) -> dict[str, float]:
    """vegas.json takes -> {label: lag + from}: where each piece starts on the
    timeline relative to the narration zero."""
    out = {}
    for t in takes:
        label = f"{t['file']}@{float(t['from_sec']):g}"
        out[label] = round(float(t["lag_sec"]) + float(t["from_sec"]), 3)
    return out


def parse_lag_arg(item: str) -> tuple[str, float]:
    """NAME=SECONDS from a --lag argument."""
    name, sep, value = item.rpartition("=")
    if not sep or not name:
        raise ValueError(f"--lag wants NAME=SECONDS, got {item!r}")
    return name, float(value)


def main(argv: list[str] | None = None) -> None:
    """CLI: check measured take lags against a .veg project."""
    ap = argparse.ArgumentParser(prog="clipmaker.veg_scan", description=__doc__.split("\n")[0])
    ap.add_argument("veg", type=Path)
    ap.add_argument("--project", default=None, help="read take lags from <project>/vegas.json")
    ap.add_argument("--lag", action="append", default=[], help="NAME=SECONDS (repeatable)")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_SEC,
                    help="search window around each expected position, s")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    try:
        expected = dict(parse_lag_arg(x) for x in args.lag)
    except ValueError as exc:
        ap.error(str(exc))
    if args.project:
        vegas_json = config.PROJECTS_DIR / args.project / "vegas.json"
        if not vegas_json.exists():
            sys.exit(f"No {vegas_json}")
        takes = json.loads(vegas_json.read_text(encoding="utf-8")).get("takes") or []
        expected.update(expected_from_takes(takes))
    if not expected:
        ap.error("no lags: pass --project or --lag")
    if not args.veg.is_file():
        sys.exit(f"No {args.veg}")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    positions = scan_positions(args.veg.read_bytes(), max(expected.values()) + args.tolerance)
    for name, pos in expected.items():
        print(f"{name:28s} at {pos:9.3f}: deltas "
              f"{deltas_near(positions, pos, args.tolerance)[:SHOWN_DELTAS]}")
    votes = vote_zero(positions, expected, args.tolerance)
    print("most common delta (narration zero on the timeline):", votes.most_common(5))


if __name__ == "__main__":
    main()

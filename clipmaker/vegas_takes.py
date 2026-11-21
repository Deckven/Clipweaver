"""Per-project Vegas settings and take-lag tables: vegas.json -> layout.txt.

Until 2026-10-27 every video whose timeline held the raw narration takes got
its own copy of the loader .cs with a hand-typed lag table (SEG_* / REC_*
arrays) — against the "no per-project .cs" rule. The table now lives in the
project's vegas.json, layout.py copies it into layout.txt, and the one generic
vegas_scripts/import_layout.cs reads it from there.

vegas.json sits in the project folder (layout.py also finds it in a parent
folder, so <project>/overlays inherits it):

    {
      "fps": 59.94,                        Vegas project frame rate; layout.py
                                           quantizes at it (no --fps by hand)
      "audio": "narration v2.mp3",         narration the timings were aligned to
      "takes": [                           raw takes inside that narration
        {"file": "Recording 1.mp3", "from_sec": 0.0, "to_sec": 22.0, "lag_sec": 0.027}
      ],
      "takes_note": "how and when the lags were measured"
    }

A take segment says: the part [from_sec, to_sec) of that take file sits in the
narration at take time + lag_sec (measure with clipmaker.take_lags, check with
clipmaker.veg_scan). In layout.txt each becomes a header line in integer
milliseconds, `take=Recording 1.mp3|0|22000|27`; lags are audio measurements
and are not frame-quantized.

CLI:
    python -m clipmaker.vegas_takes sync --project <p>
        vegas.json takes -> the take= lines of <p>/layout.txt (events untouched)
    python -m clipmaker.vegas_takes from-cs <loader.cs> --project <p> [--fps F]
        migrate an old per-project loader: its SEG_*/REC_* table -> vegas.json,
        fps/audio from the project's layout.txt header, then sync
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config

VEGAS_JSON = "vegas.json"
LAYOUT_TXT = "layout.txt"
TAKE_KEY = "take"
TAKES_COMMENT = ("# take=<file>|<from_ms>|<to_ms>|<lag_ms>: that part of a raw take sits "
                 "in the narration at take time + lag (import_layout.cs anchors on them "
                 "when the narration mp3 is not on the timeline)")
_FORBIDDEN_IN_FILE = ("|", "\n", "\r")


@dataclass(frozen=True)
class TakeSegment:
    file: str
    from_ms: int
    to_ms: int
    lag_ms: int

    def __post_init__(self):
        if not self.file.strip() or any(c in self.file for c in _FORBIDDEN_IN_FILE):
            raise ValueError(f"bad take file name {self.file!r}")
        if self.from_ms < 0 or self.to_ms <= self.from_ms:
            raise ValueError(f"take {self.file}: empty or negative range "
                             f"{self.from_ms}..{self.to_ms} ms")

    @classmethod
    def from_seconds(cls, file: str, from_sec: float, to_sec: float, lag_sec: float) -> "TakeSegment":
        return cls(file, round(float(from_sec) * 1000), round(float(to_sec) * 1000),
                   round(float(lag_sec) * 1000))

    @classmethod
    def from_dict(cls, d: dict) -> "TakeSegment":
        try:
            return cls.from_seconds(str(d["file"]), d["from_sec"], d["to_sec"], d["lag_sec"])
        except KeyError as exc:
            raise ValueError(f"take entry {d!r} lacks {exc}") from None

    def to_dict(self) -> dict:
        return {"file": self.file, "from_sec": self.from_ms / 1000,
                "to_sec": self.to_ms / 1000, "lag_sec": self.lag_ms / 1000}

    def line(self) -> str:
        return f"{TAKE_KEY}={self.file}|{self.from_ms}|{self.to_ms}|{self.lag_ms}"


@dataclass(frozen=True)
class VegasConfig:
    path: Path
    fps: float | None
    audio: str | None
    takes: tuple[TakeSegment, ...]


# --------------------------------------------------------------------------
# vegas.json
# --------------------------------------------------------------------------

def parse_vegas_config(data: dict, path: Path) -> VegasConfig:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    fps = data.get("fps")
    if fps is not None:
        fps = float(fps)
        if not 1 <= fps <= 240:
            raise ValueError(f"{path}: fps {fps} is not a video frame rate")
    audio = data.get("audio") or None
    takes = tuple(TakeSegment.from_dict(t) for t in data.get("takes") or [])
    return VegasConfig(path, fps, audio, takes)


def find_vegas_config(project_dir: Path, root: Path | None = None) -> VegasConfig | None:
    """vegas.json of the project, or of the nearest parent folder below root
    (projects/), so <project>/overlays shares the project's settings."""
    root = (root or config.PROJECTS_DIR).resolve()
    d = project_dir.resolve()
    while True:
        candidate = d / VEGAS_JSON
        if candidate.is_file():
            return parse_vegas_config(json.loads(candidate.read_text(encoding="utf-8")), candidate)
        if d == root or root not in d.parents:
            return None
        d = d.parent


def write_vegas_config(path: Path, fps: float | None, audio: str | None,
                       takes: list[TakeSegment], note: str | None = None) -> None:
    """Create or update vegas.json; keys not given here are kept."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = {**old,
            **({"fps": fps} if fps is not None else {}),
            **({"audio": audio} if audio else {}),
            "takes": [t.to_dict() for t in takes],
            **({"takes_note": note} if note else {})}
    path.write_text(dump_vegas_json(data), encoding="utf-8")


def dump_vegas_json(data: dict) -> str:
    """JSON with one take per line, so a lag table stays readable and diffable."""
    parts = []
    for key, value in data.items():
        if key == "takes" and value:
            rows = ",\n".join("    " + json.dumps(t, ensure_ascii=False) for t in value)
            parts.append(f'  "takes": [\n{rows}\n  ]')
        else:
            parts.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


# --------------------------------------------------------------------------
# layout.txt header (Python mirror of import_layout.cs ParseLayout)
# --------------------------------------------------------------------------

def parse_take_value(value: str) -> TakeSegment:
    parts = value.split("|")
    if len(parts) != 4:
        raise ValueError(f"bad take line: {value!r}")
    return TakeSegment(parts[0], int(parts[1]), int(parts[2]), int(parts[3]))


def parse_layout_header(text: str) -> tuple[dict[str, str], list[TakeSegment]]:
    """Header keys (last one wins) and take lines, the way the loader reads them."""
    header: dict[str, str] = {}
    takes: list[TakeSegment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "---":
            break
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key == TAKE_KEY:
            takes.append(parse_take_value(value))
        else:
            header[key] = value
    return header, takes


def set_layout_takes(text: str, takes: list[TakeSegment] | tuple[TakeSegment, ...]) -> str:
    """layout.txt text with its take= lines replaced; events are not touched."""
    lines = text.splitlines()
    sep = next((i for i, ln in enumerate(lines) if ln.strip() == "---"), None)
    if sep is None:
        raise ValueError("layout.txt has no '---' line between header and events")
    head = [ln for ln in lines[:sep]
            if not ln.strip().startswith(TAKE_KEY + "=") and ln.strip() != TAKES_COMMENT]
    block = ([TAKES_COMMENT] + [t.line() for t in takes]) if takes else []
    return "\n".join(head + block + lines[sep:]) + "\n"


def sync_layout(project_dir: Path, takes: tuple[TakeSegment, ...] | list[TakeSegment]) -> Path:
    layout = project_dir / LAYOUT_TXT
    if not layout.exists():
        raise FileNotFoundError(f"No {layout} — run clipmaker.layout first")
    layout.write_text(set_layout_takes(layout.read_text(encoding="utf-8"), takes),
                      encoding="utf-8")
    return layout


# --------------------------------------------------------------------------
# migration from the old per-project loaders
# --------------------------------------------------------------------------

def _array_body(text: str, name: str) -> str | None:
    """The text between the braces of `static readonly T[] NAME = { ... };`."""
    m = re.search(r"static\s+readonly\s+\w+\[\]\s+" + name + r"\s*=\s*\{(.*?)\};", text, re.S)
    return None if m is None else m.group(1)


def _strings(text: str, name: str) -> list[str] | None:
    body = _array_body(text, name)
    return None if body is None else re.findall(r'"([^"]*)"', body)


def _doubles(text: str, name: str) -> list[float] | None:
    body = _array_body(text, name)
    if body is None:
        return None
    return [float(x) for x in body.split(",") if x.strip()]


def parse_cs_tables(text: str) -> list[TakeSegment]:
    """Take segments from an old loader: SEG_FILE (or SEG_NAME)/FROM/TO/LAG
    (piecewise) or REC_NAMES/LAG/USED (one piece per take, from 0 to USED)."""
    files = _strings(text, "SEG_FILE")
    if files is None:
        files = _strings(text, "SEG_NAME")
    if files is not None:
        cols = [_doubles(text, n) for n in ("SEG_FROM", "SEG_TO", "SEG_LAG")]
    else:
        files = _strings(text, "REC_NAMES")
        if files is None:
            raise ValueError("no SEG_FILE or REC_NAMES table found")
        lag, used = _doubles(text, "REC_LAG"), _doubles(text, "REC_USED")
        cols = [[0.0] * len(files), used, lag]
    if any(c is None or len(c) != len(files) for c in cols):
        raise ValueError(f"table columns differ in length from the {len(files)} file names")
    return [TakeSegment.from_seconds(f, a, b, lag)
            for f, a, b, lag in zip(files, cols[0], cols[1], cols[2])]


def cs_table_note(text: str) -> str | None:
    """The comment block right above the table — how the lags were measured."""
    m = re.search(r"((?:[ \t]*//[^\n]*\n)+)[ \t]*static\s+readonly\s+string\[\]\s+(?:SEG_FILE|SEG_NAME|REC_NAMES)",
                  text)
    if not m:
        return None
    lines = [ln.strip()[2:].strip() for ln in m.group(1).splitlines()]
    return " ".join(ln for ln in lines if ln) or None


def migrate_cs(cs_path: Path, project_dir: Path, fps: float | None = None) -> VegasConfig:
    text = cs_path.read_text(encoding="utf-8-sig")
    takes = parse_cs_tables(text)
    layout = project_dir / LAYOUT_TXT
    header = parse_layout_header(layout.read_text(encoding="utf-8"))[0] if layout.exists() else {}
    if fps is None and header.get("fps"):
        fps = float(header["fps"])
    note = cs_table_note(text)
    note = f"From {cs_path.name}: {note}" if note else f"From {cs_path.name}"
    vj = project_dir / VEGAS_JSON
    write_vegas_config(vj, fps, header.get("audio"), takes, note)
    if layout.exists():
        sync_layout(project_dir, takes)
    return parse_vegas_config(json.loads(vj.read_text(encoding="utf-8")), vj)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _project_dir(name: str) -> Path:
    d = config.PROJECTS_DIR / name
    if not d.is_dir():
        sys.exit(f"No project folder {d}")
    return d


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clipmaker.vegas_takes", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sync", help="vegas.json takes -> layout.txt header")
    p.add_argument("--project", required=True)
    p = sub.add_parser("from-cs", help="old per-project loader table -> vegas.json + layout.txt")
    p.add_argument("cs", type=Path)
    p.add_argument("--project", required=True)
    p.add_argument("--fps", type=float, default=None,
                   help="default: fps= from the project's layout.txt")
    args = ap.parse_args(argv)

    project_dir = _project_dir(args.project)
    try:
        if args.command == "sync":
            cfg = find_vegas_config(project_dir)
            if cfg is None:
                sys.exit(f"No {VEGAS_JSON} in {project_dir} or its parents")
            layout = sync_layout(project_dir, cfg.takes)
            print(f"{len(cfg.takes)} take segments from {cfg.path} -> {layout}")
        else:
            cfg = migrate_cs(args.cs, project_dir, args.fps)
            print(f"{len(cfg.takes)} take segments, fps {cfg.fps}, audio {cfg.audio!r} -> {cfg.path}")
            if (project_dir / LAYOUT_TXT).exists():
                print(f"take lines written into {project_dir / LAYOUT_TXT}")
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()

"""Per-project Vegas settings and take-lag tables: vegas.json -> layout.txt.

When the timeline holds the raw narration takes instead of one mixed mp3, the
loader needs to know where each take sits in the narration. That table lives in
the project's vegas.json; layout.py copies it into layout.txt, and the one
generic vegas_scripts/import_layout.cs reads it from there.

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
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config
from .log import add_verbose_flag, setup_logging


VEGAS_JSON = "vegas.json"
LAYOUT_TXT = "layout.txt"
TAKE_KEY = "take"
TAKES_COMMENT = ("# take=<file>|<from_ms>|<to_ms>|<lag_ms>: that part of a raw take sits "
                 "in the narration at take time + lag (import_layout.cs anchors on them "
                 "when the narration mp3 is not on the timeline)")
_FORBIDDEN_IN_FILE = ("|", "\n", "\r")


@dataclass(frozen=True)
class TakeSegment:
    """A piece of a raw take: file, the narration span it covers, and its lag."""
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
    def from_seconds(cls, file: str, from_sec: float, to_sec: float, lag_sec: float) -> TakeSegment:
        """A segment from second values, rounded to milliseconds."""
        return cls(file, round(float(from_sec) * 1000), round(float(to_sec) * 1000),
                   round(float(lag_sec) * 1000))

    @classmethod
    def from_dict(cls, d: dict) -> TakeSegment:
        """A segment from its vegas.json form; ValueError when a key is missing."""
        try:
            return cls.from_seconds(str(d["file"]), d["from_sec"], d["to_sec"], d["lag_sec"])
        except KeyError as exc:
            raise ValueError(f"take entry {d!r} lacks {exc}") from None

    def to_dict(self) -> dict:
        """The vegas.json form of the segment, in seconds."""
        return {"file": self.file, "from_sec": self.from_ms / 1000,
                "to_sec": self.to_ms / 1000, "lag_sec": self.lag_ms / 1000}

    def line(self) -> str:
        """The take= line layout.txt carries for the loader."""
        return f"{TAKE_KEY}={self.file}|{self.from_ms}|{self.to_ms}|{self.lag_ms}"


@dataclass(frozen=True)
class VegasConfig:
    """A project's vegas.json: fps, narration file and raw-take segments."""
    path: Path
    fps: float | None
    audio: str | None
    takes: tuple[TakeSegment, ...]


# --------------------------------------------------------------------------
# vegas.json
# --------------------------------------------------------------------------

def parse_vegas_config(data: dict, path: Path) -> VegasConfig:
    """Validate vegas.json data; ValueError naming the file and the bad field."""
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
    """A segment from the value of a take= line."""
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
    """Rewrite the take lines of the project's layout.txt; returns its path."""
    layout = project_dir / LAYOUT_TXT
    if not layout.exists():
        raise FileNotFoundError(f"No {layout} — run clipmaker.layout first")
    layout.write_text(set_layout_takes(layout.read_text(encoding="utf-8"), takes),
                      encoding="utf-8")
    return layout


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _project_dir(name: str) -> Path:
    d = config.PROJECTS_DIR / name
    if not d.is_dir():
        sys.exit(f"No project folder {d}")
    return d


def main(argv: list[str] | None = None) -> None:
    """CLI: copy vegas.json takes into layout.txt."""
    ap = argparse.ArgumentParser(prog="clipmaker.vegas_takes", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sync", help="vegas.json takes -> layout.txt header")
    p.add_argument("--project", required=True)
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    project_dir = _project_dir(args.project)
    try:
        cfg = find_vegas_config(project_dir)
        if cfg is None:
            sys.exit(f"No {VEGAS_JSON} in {project_dir} or its parents")
        layout = sync_layout(project_dir, cfg.takes)
        print(f"{len(cfg.takes)} take segments from {cfg.path} -> {layout}")
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()

"""
Stills for article illustrations: frame grabs and contact sheets.

The storyboard pipeline cuts clips for Vegas; an article needs single frames.
This module takes one frame at an exact second — from a local capture or
straight from a YouTube/remote video without downloading it (yt-dlp resolves
the stream URL, ffmpeg seeks over HTTP) — strips letterbox bars and scales
to blog width.

Contact sheets are for choosing: a grid of labelled tiles (index + timecode,
or index + file name) small enough to look at in one read, so the chosen
tile can be cut again at full size with ``grab``.

CLI:
    python -m clipmaker.stills grab SRC --at 1:23.5 --out still.jpg [--width 1920] [--crop auto|none|W:H:X:Y]
    python -m clipmaker.stills sheet SRC (--at T ... | [--from A] [--to B] (--count N | --every S)) --out sheet.jpg
    python -m clipmaker.stills images PATH... --out sheet.jpg [--per-sheet 24]
    python -m clipmaker.stills fetch URL --out image.jpg

SRC is a local path or an http(s) URL (YouTube page, direct media link).
"""

from __future__ import annotations

import argparse
import io
import math
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from clipmaker.image_search import USER_AGENT

DEFAULT_WIDTH = 1920
SHEET_TILE_WIDTH = 640
IMAGE_TILE_WIDTH = 480
LABEL_HEIGHT = 28
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
RASTER_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp")
FFMPEG_TIMEOUT = 180
PROBE_TIMEOUT = 120

# A detected box must remove at least this share of the frame to count as bars:
# encoders leave a few px of black noise at the edges of full-frame video.
MIN_BAR_SHARE = 0.03
# Letterbox bars are centred; a box shifted further than this is a dark scene.
CENTRE_TOLERANCE_PX = 16

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
_FONT_CANDIDATES = ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf")
_SAVE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def parse_time(value: str | int | float) -> float:
    """Seconds from ``90``, ``"1:30"``, ``"1:02:03"`` or ``"0:44.5"``."""
    if isinstance(value, (int, float)):
        parts = [float(value)]
    else:
        pieces = value.strip().split(":")
        if not value.strip() or len(pieces) > 3:
            raise ValueError(f"bad time: {value!r}")
        try:
            parts = [float(p) for p in pieces]
        except ValueError as exc:
            raise ValueError(f"bad time: {value!r}") from exc
    if any(not math.isfinite(p) or p < 0 for p in parts):
        raise ValueError(f"bad time: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def fmt_time(seconds: float) -> str:
    """``M:SS`` or ``H:MM:SS`` — the form the catalog timeline uses."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def is_remote(src: str | Path) -> bool:
    return isinstance(src, str) and src.startswith(("http://", "https://"))


def crop_from_detect(stderr: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Crop box from ffmpeg ``cropdetect`` output, or None when there are no bars.

    Uses the last detection (with ``reset=0`` it is the union over the window,
    so one dark frame cannot shrink it). Rejects boxes that are not centred:
    a night sky above a lit horizon is black too, but it is picture.
    """
    matches = _CROP_RE.findall(stderr)
    if not matches:
        return None
    cw, ch, cx, cy = (int(v) for v in matches[-1])
    removes_rows = (height - ch) >= MIN_BAR_SHARE * height
    removes_cols = (width - cw) >= MIN_BAR_SHARE * width
    if not (removes_rows or removes_cols):
        return None
    if removes_rows and abs(cy - (height - ch) / 2) > CENTRE_TOLERANCE_PX:
        return None
    if removes_cols and abs(cx - (width - cw) / 2) > CENTRE_TOLERANCE_PX:
        return None
    return cw, ch, cx, cy


def build_resolve_cmd(url: str) -> list[str]:
    """yt-dlp call printing a direct stream URL (<=1080p). ``--no-playlist``:
    a ``watch?v=X&list=Y`` link would otherwise resolve to the playlist's
    first video, not X."""
    return ["yt-dlp", "--no-playlist", "-g",
            "-f", "bv*[height<=1080]/b[height<=1080]/b", url]


def build_grab_cmd(src: str, at_sec: float, out: Path, width: int,
                   crop: tuple[int, int, int, int] | None) -> list[str]:
    """ffmpeg command for one frame. Input-side seek: fast, and over HTTP it
    becomes a range request instead of reading the video from the start."""
    filters = []
    if crop:
        filters.append("crop={}:{}:{}:{}".format(*crop))
    filters.append(f"scale='min({width},iw)':-2")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{at_sec:.3f}", "-i", str(src),
           "-frames:v", "1", "-vf", ",".join(filters)]
    if out.suffix.lower() in (".jpg", ".jpeg"):
        cmd += ["-q:v", "2"]
    # single-image output: without it image2 treats '%' in the name as a pattern
    return cmd + ["-update", "1", str(out)]


def sample_times(start: float, end: float, count: int | None = None,
                 step: float | None = None) -> list[float]:
    """Evenly spread times: ``count`` centred samples, or every ``step`` s."""
    if end <= start:
        raise ValueError(f"empty range {fmt_time(start)}-{fmt_time(end)} "
                         "(unknown stream duration? pass --to)")
    if step is not None:
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        return [round(start + i * step, 3) for i in range(math.ceil((end - start) / step))]
    count = 9 if count is None else count
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    span = (end - start) / count
    return [round(start + (i + 0.5) * span, 3) for i in range(count)]


def grid_shape(n: int, cols: int) -> tuple[int, int]:
    """(columns, rows) for ``n`` tiles — never wider than the tile count."""
    columns = max(1, min(cols, n))
    return columns, math.ceil(n / columns)


# ---------------------------------------------------------------------------
# ffmpeg / yt-dlp
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    # ffmpeg and yt-dlp write UTF-8; the Windows default (cp1251) cannot decode
    # every Cyrillic path they echo back, and would raise mid-run.
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def _tail(text: str) -> str:
    return text.strip()[-300:]


@lru_cache(maxsize=16)
def resolve_source(src: str) -> str:
    """Local path as is; a remote page becomes a direct stream URL."""
    if not is_remote(src):
        return src
    result = _run(build_resolve_cmd(src), PROBE_TIMEOUT)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"yt-dlp could not resolve {src}: {_tail(result.stderr)}")
    return result.stdout.strip().splitlines()[0]


def probe(media: str) -> tuple[int, int, float]:
    """(width, height, duration) of the first video stream; duration 0.0 if unknown."""
    result = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height:format=duration",
                   "-of", "default=noprint_wrappers=1", media], PROBE_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media}: {_tail(result.stderr)}")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not fields.get("width", "").isdigit() or not fields.get("height", "").isdigit():
        raise RuntimeError(f"no video stream in {media}")
    try:
        duration = float(fields.get("duration", ""))
    except ValueError:
        duration = 0.0
    return int(fields["width"]), int(fields["height"]), duration


def detect_crop(media: str, at_sec: float) -> tuple[int, int, int, int] | None:
    width, height, _ = probe(media)
    result = _run(["ffmpeg", "-hide_banner", "-ss", f"{max(at_sec - 1, 0):.3f}", "-i", media,
                   "-t", "2", "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
                  FFMPEG_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"cropdetect failed at {fmt_time(at_sec)}: {_tail(result.stderr)}")
    return crop_from_detect(result.stderr, width, height)


def _parse_crop_arg(crop: str, media: str, at_sec: float):
    if crop == "none":
        return None
    if crop == "auto":
        return detect_crop(media, at_sec)
    parts = crop.split(":")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        raise ValueError(f"crop must be auto, none or W:H:X:Y, got {crop!r}")
    return tuple(int(p) for p in parts)


def grab(src: str, at_sec: float, out: Path, width: int = DEFAULT_WIDTH,
         crop: str = "auto") -> Path:
    """Write one frame of ``src`` at ``at_sec`` to ``out``; returns ``out``."""
    media = resolve_source(str(src))
    box = _parse_crop_arg(crop, media, at_sec)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)  # a leftover file must not pass for this run's output
    result = _run(build_grab_cmd(media, at_sec, out, width, box), FFMPEG_TIMEOUT)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"no frame at {fmt_time(at_sec)} "
                           f"(past the end?): {_tail(result.stderr)}")
    return out


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_label(draw: ImageDraw.ImageDraw, label: str, font, max_width: int) -> str:
    if draw.textlength(label, font=font) <= max_width:
        return label
    while label and draw.textlength(label + "…", font=font) > max_width:
        label = label[:-1]
    return label + "…"


def _compose(images: list[Image.Image | None], labels: list[str], out: Path,
             cols: int, tile_width: int) -> None:
    """Grid of 16:9 cells, each image fitted inside, label bar underneath.
    ``None`` images become grey placeholders so indexes never shift."""
    columns, rows = grid_shape(len(labels), cols)
    cell_h = round(tile_width * 9 / 16)
    sheet = Image.new("RGB", (columns * tile_width, rows * (cell_h + LABEL_HEIGHT)), "black")
    draw = ImageDraw.Draw(sheet)
    font = _font(LABEL_HEIGHT - 8)
    for i, (image, label) in enumerate(zip(images, labels)):
        x = (i % columns) * tile_width
        y = (i // columns) * (cell_h + LABEL_HEIGHT)
        if image is None:
            draw.rectangle([x, y, x + tile_width - 1, y + cell_h - 1], fill=(60, 60, 60))
        else:
            fitted = image.convert("RGB")
            fitted.thumbnail((tile_width, cell_h))
            sheet.paste(fitted, (x + (tile_width - fitted.width) // 2,
                                 y + (cell_h - fitted.height) // 2))
        draw.text((x + 6, y + cell_h + 3), _fit_label(draw, label, font, tile_width - 12),
                  fill="white", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def _load(path: Path) -> Image.Image | None:
    """Image fully read into memory (file handle closed), or None if unreadable."""
    try:
        with Image.open(path) as img:
            return img.copy()
    except (OSError, Image.DecompressionBombError):
        return None


def video_sheet(src: str, times: list[float], out: Path, cols: int = 3,
                tile_width: int = SHEET_TILE_WIDTH) -> list[dict]:
    """Contact sheet of ``src`` at ``times``; tile labels are ``N · M:SS``.
    Failed tiles stay in place (grey, marked FAIL); no tile at all is an error."""
    if not times:
        raise ValueError("no times to sample")
    media = resolve_source(str(src))
    tiles, images, last_error = [], [], ""
    with tempfile.TemporaryDirectory() as tmp:
        for index, at in enumerate(times, start=1):
            frame = Path(tmp) / f"{index:03d}.jpg"
            try:
                result = _run(build_grab_cmd(media, at, frame, tile_width, None), FFMPEG_TIMEOUT)
                last_error = _tail(result.stderr) or last_error
            except subprocess.TimeoutExpired:
                last_error = f"timeout at {fmt_time(at)}"
            image = _load(frame) if frame.exists() else None
            images.append(image)
            label = f"{index} · {fmt_time(at)}" + ("" if image else " FAIL")
            tiles.append({"index": index, "time": at, "label": label, "ok": image is not None})
    if not any(t["ok"] for t in tiles):
        raise RuntimeError(f"no frames extracted from {src}: {last_error}")
    _compose(images, [t["label"] for t in tiles], Path(out), cols, tile_width)
    return tiles


def expand_images(paths: list[Path]) -> list[Path]:
    """Files as given; directories expand to their images, sorted by name."""
    found = []
    for path in map(Path, paths):
        if path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path)
    return found


def image_sheet(paths: list[Path], out: Path, cols: int = 4,
                tile_width: int = IMAGE_TILE_WIDTH, start_index: int = 1) -> list[dict]:
    """Contact sheet of existing image files; tile labels are ``N · name``."""
    tiles, images = [], []
    for index, path in enumerate(paths, start=start_index):
        image = _load(path)
        images.append(image)
        tiles.append({"index": index, "path": path, "label": f"{index} · {path.name}",
                      "ok": image is not None})
    _compose(images, [t["label"] for t in tiles], Path(out), cols, tile_width)
    return tiles


def fetch(url: str, out: Path) -> Path:
    """Download a raster image and save it in the format ``out``'s suffix names
    (a PNG served for ``x.jpg`` is converted, not saved with the wrong extension).
    Descriptive User-Agent: Fandom rejects requests without one."""
    fmt = _SAVE_FORMATS.get(Path(out).suffix.lower())
    if fmt is None:
        raise ValueError(f"output must be one of {sorted(_SAVE_FORMATS)}, got {out}")
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
    if content_type not in RASTER_TYPES:
        raise RuntimeError(f"not a raster image: {url} ({content_type or 'no type'})")
    with Image.open(io.BytesIO(resp.content)) as img:
        image = img.convert("RGB") if fmt == "JPEG" else img.copy()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_size(path: Path) -> None:
    with Image.open(path) as img:
        print(f"{path}  {img.width}x{img.height}")


def _cmd_grab(args) -> None:
    _print_size(grab(args.src, parse_time(args.at), Path(args.out), args.width, args.crop))


def _cmd_sheet(args) -> None:
    if args.at:
        times = [parse_time(t) for t in args.at]
    else:
        start = parse_time(args.start) if args.start else 0.0
        end = parse_time(args.end) if args.end else probe(resolve_source(args.src))[2]
        times = sample_times(start, end, count=args.count, step=args.every)
    tiles = video_sheet(args.src, times, Path(args.out), args.cols, args.tile_width)
    print(args.out)
    for tile in tiles:
        print(f"  {tile['label']}  ({tile['time']:.1f}s)")


def _cmd_images(args) -> None:
    if args.per_sheet <= 0:
        raise ValueError("--per-sheet must be positive")
    paths = expand_images(args.paths)
    if not paths:
        raise ValueError("no images found")
    out = Path(args.out)
    for batch_no, first in enumerate(range(0, len(paths), args.per_sheet), start=1):
        batch = paths[first:first + args.per_sheet]
        target = out if len(paths) <= args.per_sheet else \
            out.with_name(f"{out.stem}_{batch_no:02d}{out.suffix}")
        tiles = image_sheet(batch, target, args.cols, args.tile_width, start_index=first + 1)
        print(target)
        for tile in tiles:
            print(f"  {tile['index']}: {tile['path']}" + ("" if tile["ok"] else "  (unreadable)"))


def _cmd_fetch(args) -> None:
    _print_size(fetch(args.url, Path(args.out)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clipmaker.stills", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("grab", help="one frame at full width")
    p.add_argument("src")
    p.add_argument("--at", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--crop", default="auto", help="auto | none | W:H:X:Y")
    p.set_defaults(func=_cmd_grab)

    p = sub.add_parser("sheet", help="contact sheet of a video")
    p.add_argument("src")
    p.add_argument("--at", nargs="+")
    p.add_argument("--from", dest="start")
    p.add_argument("--to", dest="end")
    p.add_argument("--count", type=int)
    p.add_argument("--every", type=float)
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--tile-width", type=int, default=SHEET_TILE_WIDTH)
    p.set_defaults(func=_cmd_sheet)

    p = sub.add_parser("images", help="contact sheet of image files / folders")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--tile-width", type=int, default=IMAGE_TILE_WIDTH)
    p.add_argument("--per-sheet", type=int, default=24)
    p.set_defaults(func=_cmd_images)

    p = sub.add_parser("fetch", help="download an image")
    p.add_argument("url")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_fetch)

    args = parser.parse_args(argv)
    # Windows consoles default to cp1251/cp866: tile labels and Cyrillic paths
    # would print as '?'.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        args.func(args)
    except FileNotFoundError as exc:
        sys.exit(f"error: {exc} (is ffmpeg/ffprobe/yt-dlp on PATH?)")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError,
            requests.RequestException) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()

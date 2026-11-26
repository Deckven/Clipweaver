"""Export a project's layout to DaVinci Resolve: OpenTimelineIO and FCPXML.

layout.txt is already editor-neutral — every event is a start, a length and a
source in-point in milliseconds relative to the narration, quantized to whole
frames. This module turns it, plus overlays/layout.txt when present, into two
files Resolve Free imports (File > Import > Timeline):

  resolve/<name>.otio    OpenTimelineIO; fades become dissolves to and from
                         empty track space
  resolve/<name>.fcpxml  FCPXML 1.9; fades become opacity keyframes

Tracks: V1 B-roll, V2 overlays (PNG plaques composite over the B-roll), A1 the
narration mix from 0 — the raw-take lag tables the Vegas loader needs are not
used here, the timeline starts with the narration. Two fade encodings are
written on purpose: the import test shows which one Resolve keeps.

Source frame rates and durations come from ffprobe (captures are 120 or 60
fps); in-points are snapped to the source's own frames. Stills have no rate
and no duration. Two B-roll events that touch get a hard cut (Vegas dips
through black there); a fade is written only next to empty track space.

Usage:
    python -m clipmaker.resolve_export <project>
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import catalog, config
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

STILL_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OVERLAY_SUBPATH = Path("overlays") / "layout.txt"
_NTSC = {23.976: Fraction(24000, 1001), 29.97: Fraction(30000, 1001), 59.94: Fraction(60000, 1001)}


# --------------------------------------------------------------------------
# layout.txt
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """One layout.txt event, times in milliseconds."""
    start_ms: int
    length_ms: int
    offset_ms: int
    label: str
    path: Path


@dataclass(frozen=True)
class TrackLayout:
    """A parsed layout.txt: track name, fps, fade length, narration and events."""
    name: str
    fps: float
    fade_ms: int
    audio: str
    events: tuple[Event, ...]


def parse_layout(path: Path) -> TrackLayout:
    """Read a layout.txt written by clipmaker.layout (any version)."""
    header: dict[str, str] = {}
    events: list[Event] = []
    in_body = False
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not in_body:
            if line.strip() == "---":
                in_body = True
            elif not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key != "take":
                    header[key.strip()] = value.strip()
            continue
        if not line.strip():
            continue
        parts = line.split("|", 4)
        if len(parts) != 5:
            raise ValueError(f"{path}:{n}: expected start|length|offset|label|path")
        try:
            start, length, offset = (int(p) for p in parts[:3])
        except ValueError:
            raise ValueError(f"{path}:{n}: start, length and offset must be integers") from None
        if length <= 0:
            raise ValueError(f"{path}:{n}: length must be positive")
        # Footage moved since the layout was written: config.PATH_MAP finds it.
        events.append(Event(start, length, offset, parts[3], catalog.existing_path(Path(parts[4]))))
    if "fps" not in header:
        raise ValueError(f"{path}: no fps= line in the header")
    return TrackLayout(
        name=header.get("track", path.parent.name),
        fps=float(header["fps"]),
        fade_ms=int(float(header.get("fade_ms", "0"))),
        audio=header.get("audio", ""),
        events=tuple(sorted(events, key=lambda e: e.start_ms)),
    )


def timeline_rate(fps: float) -> Fraction:
    """Exact timeline rate for an fps (59.94 -> 60000/1001)."""
    for nominal, exact in _NTSC.items():
        if abs(fps - nominal) < 0.01:
            return exact
    return Fraction(fps).limit_denominator(1001)


# --------------------------------------------------------------------------
# media
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Media:
    """What ffprobe says about a media file: rate, duration, video and audio streams."""
    path: Path
    rate: Fraction | None        # None for stills
    duration: Fraction | None    # seconds; None for stills
    has_video: bool
    has_audio: bool
    width: int | None = None     # pixels; None for stills and audio
    height: int | None = None

    @property
    def still(self) -> bool:
        """True for an image: video without a frame rate."""
        return self.rate is None and self.has_video


STANDARD_RATES = (Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                  Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60),
                  Fraction(120000, 1001), Fraction(120), Fraction(144), Fraction(240))


def _rate(text: str | None) -> Fraction | None:
    """A frame rate from ffprobe, snapped to the standard rate it means.

    Variable-rate captures (ShadowPlay) report averages like 120.0015 or
    59.94005; written as-is they become frame durations like
    5083249/609997500 s that no editor expects."""
    try:
        rate = Fraction(text) if text else None
    except (ValueError, ZeroDivisionError):
        return None
    if not rate:
        return None
    nearest = min(STANDARD_RATES, key=lambda s: abs(s - rate))
    return nearest if abs(nearest - rate) / nearest < Fraction(2, 1000) else rate


def probe_media(path: Path) -> Media:
    """Frame rate, duration and streams via ffprobe; a still needs no probe."""
    if path.suffix.lower() in STILL_EXTS:
        return Media(path, None, None, True, False)
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=codec_type,avg_frame_rate,r_frame_rate,width,height:format=duration",
           "-of", "json", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        info = json.loads(result.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe failed on {path}: {exc}") from exc
    if result.returncode != 0 or "format" not in info:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()[-300:]}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    rate = None
    if video:
        rate = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    duration = Fraction(info["format"].get("duration", "0")).limit_denominator(1_000_000)
    return Media(path, rate, duration, video is not None,
                 any(s.get("codec_type") == "audio" for s in streams),
                 _dimension(video, "width"), _dimension(video, "height"))


def _dimension(stream: dict | None, key: str) -> int | None:
    """A stream's width or height from ffprobe, None when absent or not a number."""
    try:
        return int(stream[key]) if stream and stream.get(key) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# placement in frames
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Placed:
    """An event converted to timeline frames, with its source in-point and fades."""
    label: str
    media: Media
    start_f: int           # timeline frames
    dur_f: int
    in_point: Fraction     # seconds into the source, on a source frame
    fade_in_f: int
    fade_out_f: int


def _frames(ms: int, rate: Fraction) -> int:
    return round(Fraction(ms, 1000) * rate)


def place_track(layout: TrackLayout, rate: Fraction, media: dict[Path, Media]) -> list[Placed]:
    """Events as timeline frames; fades only where the neighbour is empty space."""
    raw = []
    for e in layout.events:
        m = media[e.path]
        start_f, dur_f = _frames(e.start_ms, rate), _frames(e.length_ms, rate)
        if m.rate:
            in_point = Fraction(round(Fraction(e.offset_ms, 1000) * m.rate)) / m.rate
        else:
            in_point = Fraction(e.offset_ms, 1000)
        raw.append((e.label, m, start_f, dur_f, in_point))
    placed: list[Placed] = []
    for i, (label, m, start_f, dur_f, in_point) in enumerate(raw):
        if placed and start_f < placed[-1].start_f + placed[-1].dur_f:
            raise ValueError(f"{layout.name}: {label!r} overlaps the event before it")
        fade = min(_frames(layout.fade_ms, rate), dur_f // 2)
        touches_prev = bool(placed) and start_f == placed[-1].start_f + placed[-1].dur_f
        touches_next = i + 1 < len(raw) and raw[i + 1][2] == start_f + dur_f
        placed.append(Placed(label, m, start_f, dur_f, in_point,
                             0 if touches_prev or start_f == 0 else fade,
                             0 if touches_next else fade))
    return placed


# --------------------------------------------------------------------------
# OpenTimelineIO
# --------------------------------------------------------------------------

def build_otio(name: str, rate: Fraction, video: list[tuple[str, list[Placed]]],
               narration: Media | None):
    """The timeline as an OpenTimelineIO object: video tracks plus the narration."""
    import opentimelineio as otio

    r = float(rate)

    def rt(frames: int):
        return otio.opentime.RationalTime(frames, r)

    def reference(m: Media):
        ref = otio.schema.ExternalReference(target_url=m.path.resolve().as_uri())
        if m.rate and m.duration:
            src = float(m.rate)
            ref.available_range = otio.opentime.TimeRange(
                otio.opentime.RationalTime(0, src),
                otio.opentime.RationalTime(round(m.duration * m.rate), src))
        return ref

    def dissolve(in_f: int, out_f: int):
        return otio.schema.Transition(transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
                                      in_offset=rt(in_f), out_offset=rt(out_f))

    timeline = otio.schema.Timeline(name=name, global_start_time=rt(0))
    for track_name, placed in video:
        track = otio.schema.Track(name=track_name, kind=otio.schema.TrackKind.Video)
        cursor = 0
        for p in placed:
            if p.start_f > cursor:
                track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(rt(0), rt(p.start_f - cursor))))
            if p.fade_in_f:
                track.append(dissolve(0, p.fade_in_f))
            src_rate = float(p.media.rate) if p.media.rate else r
            start = otio.opentime.RationalTime(float(p.in_point * Fraction(src_rate)), src_rate)
            track.append(otio.schema.Clip(
                name=p.label, media_reference=reference(p.media),
                source_range=otio.opentime.TimeRange(start, rt(p.dur_f).rescaled_to(src_rate))))
            if p.fade_out_f:
                # Fades into the empty space after the clip; the gap to the next
                # clip (or the closing gap below) is the transition's neighbour.
                track.append(dissolve(p.fade_out_f, 0))
            cursor = p.start_f + p.dur_f
        if placed and placed[-1].fade_out_f:
            track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(rt(0), rt(placed[-1].fade_out_f))))
        timeline.tracks.append(track)
    if narration is not None and narration.duration:
        audio = otio.schema.Track(name="Narration", kind=otio.schema.TrackKind.Audio)
        audio.append(otio.schema.Clip(
            name=narration.path.name, media_reference=reference(narration),
            source_range=otio.opentime.TimeRange(rt(0), rt(round(narration.duration * rate)))))
        timeline.tracks.append(audio)
    return timeline


# --------------------------------------------------------------------------
# FCPXML 1.9
# --------------------------------------------------------------------------

def _t(seconds: Fraction) -> str:
    seconds = Fraction(seconds)
    if seconds == 0:
        return "0s"
    if seconds.denominator == 1:
        return f"{seconds.numerator}s"
    return f"{seconds.numerator}/{seconds.denominator}s"


DEFAULT_FRAME_SIZE = (1920, 1080)


def frame_size(video: list[tuple[str, list[Placed]]]) -> tuple[int, int]:
    """The sequence size: the size most of the placed footage has.

    Ties go to the larger frame; stills and unprobed media do not vote, and with
    no sized footage at all the sequence stays 1920x1080."""
    votes: dict[tuple[int, int], int] = {}
    for _, placed in video:
        for p in placed:
            if p.media.width and p.media.height and not p.media.still:
                size = (p.media.width, p.media.height)
                votes[size] = votes.get(size, 0) + 1
    if not votes:
        return DEFAULT_FRAME_SIZE
    return max(votes, key=lambda s: (votes[s], s[0] * s[1]))


def build_fcpxml(name: str, rate: Fraction, video: list[tuple[str, list[Placed]]],
                 narration: Media | None) -> str:
    """The timeline as FCPXML 1.9 text for DaVinci Resolve."""
    frame = 1 / rate
    root = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(root, "resources")
    width, height = frame_size(video)
    ET.SubElement(resources, "format", id="r1", frameDuration=_t(frame),
                  width=str(width), height=str(height))
    formats: dict[Fraction | None, str] = {}
    assets: dict[Path, str] = {}

    def format_id(m: Media) -> str:
        key = m.rate
        if key not in formats:
            formats[key] = f"r{len(formats) + 2}"
            attrs = {"id": formats[key]}
            if key:
                attrs["frameDuration"] = _t(1 / key)
            else:
                attrs["name"] = "FFVideoFormatRateUndefined"
            ET.SubElement(resources, "format", attrs)
        return formats[key]

    def asset_id(m: Media) -> str:
        if m.path not in assets:
            assets[m.path] = f"a{len(assets) + 1}"
            attrs = {"id": assets[m.path], "name": m.path.stem, "start": "0s",
                     "duration": _t(m.duration or Fraction(0)),
                     "hasVideo": "1" if m.has_video else "0",
                     "hasAudio": "1" if m.has_audio else "0"}
            if m.has_video:
                attrs["format"] = format_id(m)
            if m.has_audio:
                attrs.update(audioSources="1", audioChannels="2", audioRate="48000")
            asset = ET.SubElement(resources, "asset", attrs)
            ET.SubElement(asset, "media-rep", kind="original-media", src=m.path.resolve().as_uri())
        return assets[m.path]

    ends = [p.start_f + p.dur_f for _, placed in video for p in placed]
    total = Fraction(max(ends, default=0)) * frame
    if narration is not None and narration.duration:
        total = max(total, Fraction(round(narration.duration * rate)) * frame)

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="Clipmaker")
    project = ET.SubElement(event, "project", name=name)
    sequence = ET.SubElement(project, "sequence", format="r1", duration=_t(total),
                             tcStart="0s", tcFormat="NDF", audioLayout="stereo", audioRate="48k")
    spine = ET.SubElement(sequence, "spine")
    gap = ET.SubElement(spine, "gap", name="Gap", offset="0s", start="0s", duration=_t(total))

    for lane, (_, placed) in enumerate(video, 1):
        for p in placed:
            # <video> uses only the picture of the asset: B-roll game audio stays out.
            clip = ET.SubElement(gap, "video", ref=asset_id(p.media), name=p.label, lane=str(lane),
                                 offset=_t(p.start_f * frame), start=_t(p.in_point),
                                 duration=_t(p.dur_f * frame))
            if p.fade_in_f or p.fade_out_f:
                keys = []
                if p.fade_in_f:
                    keys += [(0, "0"), (p.fade_in_f, "1")]
                if p.fade_out_f:
                    keys += [(p.dur_f - p.fade_out_f, "1"), (p.dur_f, "0")]
                blend = ET.SubElement(clip, "adjust-blend", amount="1")
                param = ET.SubElement(blend, "param", name="amount")
                anim = ET.SubElement(param, "keyframeAnimation")
                for f, value in keys:  # keyframe times are in the clip's source time
                    ET.SubElement(anim, "keyframe", time=_t(p.in_point + f * frame), value=value)
    if narration is not None and narration.duration:
        ET.SubElement(gap, "asset-clip", ref=asset_id(narration), name=narration.path.name,
                      lane="-1", offset="0s", start="0s",
                      duration=_t(Fraction(round(narration.duration * rate)) * frame))

    ET.indent(root)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
            + ET.tostring(root, encoding="unicode") + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def export_project(project: str, probe=probe_media) -> tuple[Path, Path]:
    """Write resolve/<name>.otio and .fcpxml from the project's layout files; returns both paths."""
    project_dir = config.project_path(project)
    main = project_dir / "layout.txt"
    if not main.is_file():
        raise FileNotFoundError(f"{main} not found — run `python -m clipmaker.layout {project}` first")
    layouts = [parse_layout(main)]
    if (project_dir / OVERLAY_SUBPATH).is_file():
        layouts.append(parse_layout(project_dir / OVERLAY_SUBPATH))
    rate = timeline_rate(layouts[0].fps)
    for extra in layouts[1:]:
        if timeline_rate(extra.fps) != rate:
            raise ValueError(f"{extra.name}: fps {extra.fps:g} differs from the B-roll's {layouts[0].fps:g}")

    paths = {e.path for lay in layouts for e in lay.events}
    missing = sorted(str(p) for p in paths if not p.is_file())
    if missing:
        raise FileNotFoundError("media not found:\n  " + "\n  ".join(missing))
    media = {p: probe(p) for p in sorted(paths)}
    video = [(lay.name, place_track(lay, rate, media)) for lay in layouts]

    narration = None
    audio_path = project_dir / layouts[0].audio if layouts[0].audio else None
    if audio_path and audio_path.is_file():
        narration = probe(audio_path)
    else:
        log.warning("[resolve] narration %r not in %s; no audio track", layouts[0].audio, project_dir)

    name = Path(project).name
    out_dir = project_dir / "resolve"
    out_dir.mkdir(exist_ok=True)
    otio_path, fcpxml_path = out_dir / f"{name}.otio", out_dir / f"{name}.fcpxml"
    import opentimelineio as otio
    otio.adapters.write_to_file(build_otio(name, rate, video, narration), str(otio_path))
    fcpxml_path.write_text(build_fcpxml(name, rate, video, narration), encoding="utf-8")
    return otio_path, fcpxml_path


def main(argv: list[str] | None = None) -> None:
    """CLI: export a project's layout to DaVinci Resolve."""
    ap = argparse.ArgumentParser(
        description="Export projects/<project>/layout.txt (+ overlays/layout.txt) to "
                    "DaVinci Resolve: resolve/<name>.otio and resolve/<name>.fcpxml")
    ap.add_argument("project", help="project folder under projects/")
    add_verbose_flag(ap)
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    try:
        import opentimelineio  # noqa: F401
    except ImportError:
        sys.exit('OpenTimelineIO is not installed: pip install -e ".[resolve]"')
    try:
        otio_path, fcpxml_path = export_project(args.project)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"[resolve] {exc}")
    print(f"[resolve] {otio_path}\n[resolve] {fcpxml_path}")
    print("[resolve] In Resolve: File > Import > Timeline, pick one of the two files.")


if __name__ == "__main__":
    main()

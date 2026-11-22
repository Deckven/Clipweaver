"""Resolve export: layout.txt (+ overlays) -> OTIO and FCPXML, on fixtures with
a fake ffprobe. Positions are checked in timeline frames."""
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest

from clipmaker import config, resolve_export as rx
from clipmaker.resolve_export import Media

otio = pytest.importorskip("opentimelineio")


def layout_text(track, fade_ms, events, audio="mix.mp3"):
    head = ["# clipmaker layout v2 — p", "project=p", f"audio={audio}", "fps=60",
            f"track={track}", f"fade_ms={fade_ms}", "take=Recording 1.mp3|0|1000|5", "---"]
    return "\n".join(head + ["|".join(str(x) for x in e) for e in events]) + "\n"


@pytest.fixture
def project(tmp_path):
    pdir = config.PROJECTS_DIR / "game" / "ep1"
    (pdir / "overlays" / "png").mkdir(parents=True)
    media = tmp_path / "media"
    media.mkdir()
    a, b = media / "a.mp4", media / "b.mp4"
    plaque = pdir / "overlays" / "png" / "K01.png"
    for f in (a, b, plaque, pdir / "mix.mp3"):
        f.write_bytes(b"")
    # B-roll: a at 1 s for 2 s from 4.167 s in; b touches it; then a gap; a again.
    (pdir / "layout.txt").write_text(layout_text("B-roll", 400, [
        (1000, 2000, 4167, "fr 01 one", a),
        (3000, 1500, 0, "fr 02 two", b),
        (6000, 1000, 500, "fr 03 three", a),
    ]), encoding="utf-8")
    (pdir / "overlays" / "layout.txt").write_text(layout_text("Plaques", 250, [
        (1500, 2000, 0, "K01 plaque", plaque),
    ]), encoding="utf-8")
    return "game/ep1", {a: Media(a, Fraction(120), Fraction(600), True, True),
                        b: Media(b, Fraction(30), Fraction(40), True, False),
                        plaque: Media(plaque, None, None, True, False),
                        pdir / "mix.mp3": Media(pdir / "mix.mp3", None, Fraction(20), False, True)}


def fake_probe(table):
    return lambda path: table[Path(path)]


def test_parse_layout_reads_header_and_events(project):
    lay = rx.parse_layout(config.PROJECTS_DIR / "game/ep1/layout.txt")
    assert (lay.name, lay.fps, lay.fade_ms, lay.audio) == ("B-roll", 60.0, 400, "mix.mp3")
    assert [e.start_ms for e in lay.events] == [1000, 3000, 6000]
    assert lay.events[0].label == "fr 01 one"


@pytest.mark.parametrize("body, match", [
    ("1000|2000|0|only four", "start|length"),
    ("1000|x|0|label|a.mp4", "integers"),
    ("1000|0|0|label|a.mp4", "positive"),
])
def test_parse_layout_rejects_bad_lines(tmp_path, body, match):
    path = tmp_path / "layout.txt"
    path.write_text(f"fps=60\n---\n{body}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        rx.parse_layout(path)


def test_ntsc_rates_are_exact():
    assert rx.timeline_rate(59.94) == Fraction(60000, 1001)
    assert rx.timeline_rate(60) == 60


def test_placement_frames_fades_and_source_frames(project):
    name, table = project
    lay = rx.parse_layout(config.PROJECTS_DIR / name / "layout.txt")
    placed = rx.place_track(lay, Fraction(60), table)
    assert [(p.start_f, p.dur_f) for p in placed] == [(60, 120), (180, 90), (360, 60)]
    # 4.167 s on a 120 fps source snaps to frame 500 = 25/6 s
    assert placed[0].in_point == Fraction(500, 120)
    # touching clips cut; fades only next to empty space (400 ms = 24 frames)
    assert [(p.fade_in_f, p.fade_out_f) for p in placed] == [(24, 0), (0, 24), (24, 24)]


def test_overlapping_events_are_refused(project):
    name, table = project
    lay = rx.parse_layout(config.PROJECTS_DIR / name / "layout.txt")
    bad = rx.TrackLayout(lay.name, 60, 0, "", (lay.events[0], lay.events[0]))
    with pytest.raises(ValueError, match="overlaps"):
        rx.place_track(bad, Fraction(60), table)


def test_otio_tracks_positions_and_round_trip(project):
    name, table = project
    otio_path, _ = rx.export_project(name, probe=fake_probe(table))
    tl = otio.adapters.read_from_file(str(otio_path))
    v1, v2, a1 = tl.tracks
    assert [t.kind for t in tl.tracks] == ["Video", "Video", "Audio"]
    clips = [c for c in v1 if isinstance(c, otio.schema.Clip)]
    starts = [c.range_in_parent().start_time.rescaled_to(60).value for c in clips]
    assert starts == [60, 180, 360]
    assert clips[0].source_range.start_time.rate == 120
    assert clips[0].source_range.start_time.value == 500
    assert len([t for t in v1 if isinstance(t, otio.schema.Transition)]) == 4
    assert [c.name for c in v2 if isinstance(c, otio.schema.Clip)] == ["K01 plaque"]
    assert a1[0].media_reference.target_url.endswith("mix.mp3")
    assert a1[0].range_in_parent().start_time.value == 0


def test_fcpxml_lanes_offsets_fades_and_audio(project):
    name, table = project
    _, fcpxml_path = rx.export_project(name, probe=fake_probe(table))
    root = ET.parse(fcpxml_path).getroot()
    assert root.get("version") == "1.9"
    gap = root.find("library/event/project/sequence/spine/gap")
    vids = gap.findall("video")
    assert [(v.get("lane"), v.get("offset")) for v in vids] == [
        ("1", "1s"), ("1", "3s"), ("1", "6s"), ("2", "3/2s")]
    assert vids[0].get("start") == "25/6s" and vids[0].get("duration") == "2s"
    keys = vids[0].findall("adjust-blend/param/keyframeAnimation/keyframe")
    # fade-in only (the next clip touches): 0 -> 1 over 24 frames from the in-point
    assert [(k.get("time"), k.get("value")) for k in keys] == [("25/6s", "0"), ("137/30s", "1")]
    assert vids[1].find("adjust-blend/param/keyframeAnimation/keyframe[1]").get("value") == "1"
    audio = gap.find("asset-clip")
    assert (audio.get("lane"), audio.get("offset"), audio.get("duration")) == ("-1", "0s", "20s")
    assets = {a.get("id"): a for a in root.findall("resources/asset")}
    plaque = assets[vids[3].get("ref")]
    assert plaque.get("duration") == "0s" and plaque.get("hasAudio") == "0"
    assert plaque.find("media-rep").get("src").startswith("file:///")


def test_missing_media_fails_before_anything_is_written(project):
    name, table = project
    (config.PROJECTS_DIR / name / "overlays" / "png" / "K01.png").unlink()
    with pytest.raises(FileNotFoundError, match="K01.png"):
        rx.export_project(name, probe=fake_probe(table))
    assert not (config.PROJECTS_DIR / name / "resolve").exists()


@pytest.mark.parametrize("text, rate", [
    ("120/1", Fraction(120)),
    ("1200015/10000", Fraction(120)),            # ShadowPlay average
    ("5994005/100000", Fraction(60000, 1001)),
    ("60000/1001", Fraction(60000, 1001)),
    ("15/1", Fraction(15)),                      # not near a standard rate: kept
    ("0/0", None),
    (None, None),
])
def test_probe_rates_snap_to_the_standard_rate(text, rate):
    assert rx._rate(text) == rate

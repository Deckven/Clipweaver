"""The generic Vegas loader, compiled and run outside Vegas.

vegas_scripts/import_layout.cs keeps its parsing and take-anchor logic in
static methods without Vegas types; tests/vegas_harness/LayoutHarness.cs calls
them. This proves, without Vegas:
  * every migrated layout.txt parses in C# to exactly the take segments of the
    project's vegas.json (= the old per-project SEG_*/REC_* table) and to the
    same events Python wrote;
  * on a timeline laid out by those tables, the new loader finds the same
    narration zero the old per-project script would have found.
Skipped when the .NET Framework compiler or the Vegas DLL is not installed.
"""

import re
import subprocess
from pathlib import Path

import pytest

from clipmaker import config, vegas_takes

ROOT = Path(__file__).resolve().parent.parent
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
VEGAS_DLL = Path(r"C:\Program Files\VEGAS\Vegas Pro 22\ScriptPortal.Vegas.dll")
LOADER = ROOT / "vegas_scripts" / "import_layout.cs"
HARNESS = ROOT / "tests" / "vegas_harness" / "LayoutHarness.cs"
PIN = re.compile(r'LAYOUT_FILE = @"[^"]*[\\/]projects[\\/](.*)[\\/]layout\.txt"')
NARRATION_ZERO_MS = -27.0  # where the measured projects really had it

pytestmark = pytest.mark.skipif(not (CSC.exists() and VEGAS_DLL.exists()),
                                reason="needs csc.exe and the Vegas Pro 22 DLL")


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    out = tmp_path_factory.mktemp("cs")
    for cmd in (
        [str(CSC), "-nologo", "-target:library", f"-out:{out / 'import_layout.dll'}",
         f"-reference:{VEGAS_DLL}", "-reference:System.Windows.Forms.dll", str(LOADER)],
        [str(CSC), "-nologo", "-target:exe", f"-out:{out / 'LayoutHarness.exe'}",
         f"-reference:{out / 'import_layout.dll'}", str(HARNESS)],
    ):
        res = subprocess.run(cmd, capture_output=True)
        assert res.returncode == 0, res.stdout.decode("cp866", "replace")
    exe = out / "LayoutHarness.exe"

    def run(*args) -> list[str]:
        res = subprocess.run([str(exe), *map(str, args)], capture_output=True)
        assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
        return res.stdout.decode("utf-8").splitlines()

    return run


def migrated_layouts() -> list[Path]:
    return sorted(p.parent / "layout.txt" for p in config.PROJECTS_DIR.rglob("vegas.json")
                  if (p.parent / "layout.txt").exists())


def python_events(text: str) -> list[list[str]]:
    rows = text.split("\n---\n", 1)[1].splitlines()
    return [r.split("|", 4) for r in rows if r.strip() and not r.startswith("#")]


@pytest.mark.parametrize("lay", migrated_layouts(), ids=lambda p: p.parent.name[:40])
def test_csharp_parses_the_same_takes_and_events(harness, lay):
    out = harness("parse", lay)
    cfg = vegas_takes.find_vegas_config(lay.parent)
    takes = [line.split("|")[1:] for line in out if line.startswith("take|")]
    assert [vegas_takes.TakeSegment(f, int(a), int(b), int(c)) for f, a, b, c in takes] \
        == list(cfg.takes)
    header = next(line for line in out if line.startswith("header|")).split("|")
    assert float(header[4]) == pytest.approx(cfg.fps)
    events = [line.split("|", 5)[1:] for line in out if line.startswith("ev|")]
    assert events == python_events(lay.read_text(encoding="utf-8"))


# --- the anchor ----------------------------------------------------------------

def timeline_from(takes, zero_ms: float) -> list[tuple[str, float, float, float]]:
    """Audio events (file, start, offset, length) as an editor lays the takes
    out: each segment once, the longer ones cut in two, plus unrelated audio."""
    pieces = []
    for t in takes:
        start = zero_ms + t.lag_ms + t.from_ms
        length = t.to_ms - t.from_ms
        if length > 20_000:
            half = length / 2
            pieces.append((t.file, start, t.from_ms, half))
            pieces.append((t.file, start + half, t.from_ms + half, length - half))
        else:
            pieces.append((t.file, start, t.from_ms, length))
    pieces.append(("Music bed.mp3", 0.0, 0.0, 600_000.0))
    return pieces


def old_anchor(cs_text: str, pieces) -> float:
    """What the old per-project FindStartFromRecordings computed (three variants)."""
    from clipmaker import vegas_migrate  # private module; only the author's old loaders reach here
    segs = vegas_migrate.parse_cs_tables(cs_text)
    middle_rule = "SEG_NAME" in cs_text  # Gold and Mats: the piece's middle picks the segment
    rec_rule = "REC_NAMES" in cs_text    # REC: any piece starting before USED
    zeros = []
    for file, start, offset, length in pieces:
        best, best_overlap = None, 0.0
        for s in segs:
            if s.file.lower() != file.lower():
                continue
            if rec_rule:
                if offset < s.to_ms and offset + length > 0:
                    best = s
            elif middle_rule:
                if s.from_ms <= offset + length / 2 < s.to_ms:
                    best = s
            else:
                overlap = min(offset + length, s.to_ms) - max(offset, s.from_ms)
                if overlap > best_overlap:
                    best, best_overlap = s, overlap
        if best is not None:
            zeros.append(start - offset - best.lag_ms)
    zeros.sort()
    return zeros[len(zeros) // 2]


def write_pieces(path: Path, pieces) -> Path:
    path.write_text("\n".join(f"{f}|{s!r}|{o!r}|{n!r}" for f, s, o, n in pieces),
                    encoding="utf-8")
    return path


def old_loaders() -> list[Path]:
    scripts = ROOT / "vegas_scripts"
    found = []
    for cs in sorted(scripts.glob("*.cs")) + sorted((scripts / "_archive").glob("*.cs")):
        text = cs.read_text(encoding="utf-8-sig")
        if re.search(r"\b(SEG_FILE|SEG_NAME|REC_NAMES)\b", text) and PIN.search(text):
            found.append(cs)
    return found


@pytest.mark.parametrize("cs", old_loaders(), ids=lambda p: p.stem[:40])
def test_new_loader_finds_the_same_narration_zero_as_the_old_script(harness, cs, tmp_path):
    text = cs.read_text(encoding="utf-8-sig")
    project = config.PROJECTS_DIR / PIN.search(text).group(1).replace("\\", "/")
    lay = project / "layout.txt"
    if not lay.exists():
        pytest.skip("no layout.txt on this branch")
    takes = vegas_takes.find_vegas_config(project).takes
    pieces = timeline_from(takes, NARRATION_ZERO_MS)

    out = harness("anchor", lay, write_pieces(tmp_path / "pieces.txt", pieces))

    zero = float(out[0].split("|")[1])
    assert zero == pytest.approx(NARRATION_ZERO_MS, abs=1e-6)
    assert zero == pytest.approx(old_anchor(text, pieces), abs=1e-6)
    assert out[-1].strip() == f"all {len(pieces) - 1} take pieces agree"


def test_moved_piece_is_outvoted_and_reported(harness, tmp_path):
    lay = tmp_path / "layout.txt"
    lay.write_text("audio=n.mp3\nfps=60\n"
                   "take=Recording 1.mp3|0|10000|500\n"
                   "take=Recording 2.mp3|0|10000|11000\n"
                   "take=Recording 3.mp3|0|10000|22000\n---\n", encoding="utf-8")
    pieces = [("Recording 1.mp3", 1500, 0, 10000),   # zero at 1000
              ("recording 2.MP3", 12000, 0, 10000),  # file names match case-insensitively
              ("Recording 3.mp3", 25000, 0, 10000)]  # moved by 2 s after the render
    out = harness("anchor", lay, write_pieces(tmp_path / "p.txt", pieces))
    assert out[0] == "zero|1000"
    assert "WARNING: 1 of 3 take pieces were moved" in out[-1]


def test_no_listed_take_on_the_timeline_gives_no_anchor(harness, tmp_path):
    lay = tmp_path / "layout.txt"
    lay.write_text("take=Recording 1.mp3|0|10000|500\n---\n", encoding="utf-8")
    out = harness("anchor", lay, write_pieces(tmp_path / "p.txt", [("Final.mp3", 0, 0, 5000)]))
    assert out[0] == "zero|NaN"


def test_bad_take_line_fails_loudly(harness, tmp_path):
    lay = tmp_path / "layout.txt"
    lay.write_text("take=Recording 1.mp3|0|10000\n---\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        harness("parse", lay)

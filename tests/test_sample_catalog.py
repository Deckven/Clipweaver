"""examples/sample — the sanitized catalog a fresh clone can search without video."""
import json
import re
from pathlib import Path

import pytest

from clipmaker import config, embed_index, ocrgrep
from clipmaker.youtube_search import split_into_paragraphs

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample"
CATALOG = SAMPLE / "catalog" / config.GAMES["assassins_creed_4"]["catalog_subdir"]
ENTRIES = sorted(CATALOG.glob("*.json"))
DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_sample_has_three_to_five_clips():
    assert 3 <= len(ENTRIES) <= 5


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem[40:])
def test_entry_is_complete_and_machine_neutral(path):
    entry = load(path)
    assert entry["game"] == "assassins_creed_4"
    assert entry["source_path"].startswith("footage/raw/")
    assert entry["source_path"].endswith(entry["original_name"])
    assert not DRIVE.search(path.read_text(encoding="utf-8"))
    assert entry["scenes"] and entry["timeline"] and entry["title"]


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem[40:])
def test_every_clip_yields_scene_documents_for_the_index(path):
    docs = embed_index.iter_documents(load(path), path.name)
    levels = {d.level for d in docs}
    assert {"video", "scene"} <= levels
    assert all(d.start_sec is not None for d in docs if d.level == "scene")


def test_ocr_finds_on_screen_text_as_moments():
    hits = ocrgrep.search(CATALOG, re.compile("harbou?rmaster", re.I))
    assert [h.name for h in hits] == [
        "Assassin's Creed IV Black Flag Resynced Ship Upgrade Trading 1.mp4"]
    assert hits[0].stamps[0] == "00:10"


def test_shots_cover_every_narration_paragraph_and_name_a_sample_clip():
    shots = json.loads((SAMPLE / "shots.json").read_text(encoding="utf-8"))
    narration = (SAMPLE / "narration.md").read_text(encoding="utf-8")
    paragraphs = [p for p in narration.split("\n\n") if p.strip()]
    assert [s["fragment"] for s in shots] == list(range(1, len(paragraphs) + 1))
    names = {load(p)["original_name"] for p in ENTRIES}
    assert {s["expected"] for s in shots} == names
    assert all(s["shots"] for s in shots)
    # report --min-words 1 must see the same fragments shots.json is numbered for
    assert len(split_into_paragraphs(narration, min_words=1)) == len(paragraphs)

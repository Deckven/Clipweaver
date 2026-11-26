"""production_plan.generate_plan with the catalog, YouTube and stock searches
stubbed: per-fragment recommendation, coverage, gaps and the shooting list."""
import copy

import pytest

from clipmaker import production_plan as pp

WORDS = " ".join(["word"] * 26)
TEXT = "\n\n".join([
    f"Kingston harbour galleon battle storm {WORDS}",          # catalog covers it
    f"Я сижу перед камерой и рассказываю о себе {WORDS}",       # talking head -> shoot
    f"Nothing anywhere matches this unusual paragraph {WORDS}",  # nothing -> shoot, low
])

CATALOG = [{"original_name": "naval.mp4", "title": "Kingston harbour galleon battle",
            "summary": "storm battle", "tags": ["kingston", "harbour"], "duration_sec": 90}]


@pytest.fixture
def searches(monkeypatch):
    catalog = copy.deepcopy(CATALOG)
    monkeypatch.setattr(pp, "_load_catalog", lambda game: catalog)
    monkeypatch.setattr(pp, "search_catalog", lambda text, catalog, max_results:
                        [e for e in catalog if "Kingston" in text])
    monkeypatch.setattr(pp, "search_youtube", lambda q, max_results: [
        {"title": "Unrelated vlog", "url": "https://y/1", "view_count": 10, "duration_sec": 60}])
    monkeypatch.setattr(pp, "search_stock", lambda q, max_results: [])


def test_plan_recommends_a_source_per_fragment(searches):
    plan = pp.generate_plan(TEXT, "proj", game="g")
    assert plan["total_fragments"] == 3
    recs = [(f["recommendation"], f["confidence"]) for f in plan["fragments"]]
    assert recs[0] == ("local", "high")
    assert recs[2] == ("shoot", "low")
    first = plan["fragments"][0]
    assert first["sources"]["local"][0]["file"] == "naval.mp4"
    assert first["duration_sec"] == round(len(first["text"].split()) / 150 * 60, 1)
    assert sum(plan["coverage"].values()) == pytest.approx(1.0, abs=0.02)


def test_gaps_and_shooting_list(searches):
    plan = pp.generate_plan(TEXT, "proj", game="g")
    shoot = [s["fragment_index"] for s in plan["shooting_list"]]
    assert 3 in shoot
    gap = next(g for g in plan["gaps"] if g["fragment_index"] == 3)
    assert gap["priority"] == "high" and gap["recommendation"] == "shoot"


def test_the_catalog_entries_are_not_written_into(searches):
    catalog = pp._load_catalog("g")
    pp.generate_plan(TEXT, "proj", game="g")
    assert catalog == CATALOG  # no "_score" left behind in the shared entries


def test_empty_text():
    assert pp.generate_plan("  ", "proj") == {"error": "No fragments found"}


def test_markdown_renders_every_fragment(searches):
    md = pp.render_plan_markdown(pp.generate_plan(TEXT, "proj", game="g"))
    assert md.count("naval.mp4") >= 1
    for i in (1, 2, 3):
        assert f"{i}" in md

"""Reading a project never creates folders; Ollama failures are reported and not cached."""
import pytest

from clipmaker import config, report, shot_query


def test_typo_in_project_name_creates_no_folder():
    with pytest.raises(FileNotFoundError):
        report.read_project_text("tpyo")
    assert not (config.PROJECTS_DIR / "tpyo").exists()
    assert report.load_query_overrides("tpyo") == {}
    assert not config.PROJECTS_DIR.exists()


def test_list_projects_without_a_projects_folder():
    assert report.list_projects() == []
    assert not config.PROJECTS_DIR.exists()


def test_empty_ollama_reply_is_not_cached(monkeypatch):
    monkeypatch.setattr(shot_query, "_call_ollama", lambda *a, **k: "   ")
    out = shot_query.build_shot_queries("a fragment", game=None)
    assert [q.source for q in out] == ["fragment"]
    assert not config.SHOT_QUERY_CACHE.exists()


def test_unreachable_ollama_is_said_once(monkeypatch, caplog):
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:9/api/chat")
    monkeypatch.setattr(config, "SHOT_QUERY_TIMEOUT", 2)
    monkeypatch.setattr(shot_query, "_warned", set())
    for frag in ("first fragment", "second fragment"):
        assert shot_query.build_shot_queries(frag, use_cache=False)[0].source == "fragment"
    assert caplog.text.count("did not answer") == 1

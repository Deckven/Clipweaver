"""Project names from the command line stay inside PROJECTS_DIR."""
import pytest

from clipmaker import config


def test_nested_name_is_fine_and_not_created(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    p = config.project_path("my_game/drafts/ep1")
    assert p == tmp_path / "my_game" / "drafts" / "ep1" and not p.exists()


@pytest.mark.parametrize("name", ["../outside", "a/../../outside", "--help", ""])
def test_escaping_or_flag_like_names_are_refused(tmp_path, monkeypatch, name):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    with pytest.raises(ValueError):
        config.project_path(name)


def test_absolute_path_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    with pytest.raises(ValueError):
        config.project_path(str(tmp_path / "elsewhere"))

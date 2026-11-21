"""align takes the project positionally or as --project, and never creates
folders named after flags.

The /storyboard skill documented `align --project <p>` while align read
sys.argv[1]: the run resolved to projects/--project (created on the spot by
get_project_dir) and failed with "No storyboard.md". `align --help` left
projects/--help the same way.
"""

import pytest

from clipmaker import align, config


def test_positional_project():
    assert align.parse_args(["eve_online/drafts/x"]) == "eve_online/drafts/x"


def test_project_flag_is_accepted_too():
    assert align.parse_args(["--project", "eve_online/drafts/x"]) == "eve_online/drafts/x"


def test_both_forms_must_agree():
    with pytest.raises(SystemExit):
        align.parse_args(["a", "--project", "b"])
    assert align.parse_args(["a", "--project", "a"]) == "a"


def test_missing_project_is_an_error_not_a_default():
    with pytest.raises(SystemExit):
        align.parse_args([])


def test_help_creates_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        align.main(["--help"])
    assert list(tmp_path.iterdir()) == []


def test_unknown_project_creates_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        align.main(["--project", "nope"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["--project", "--help", "-h", ""])
def test_get_project_dir_refuses_flag_like_names(tmp_path, monkeypatch, name):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    with pytest.raises(ValueError):
        config.get_project_dir(name)
    assert list(tmp_path.iterdir()) == []

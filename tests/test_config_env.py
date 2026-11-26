"""clipmaker.config — machine-specific paths and keys come from .env / the environment."""
import importlib
import os
from pathlib import Path

import pytest

from clipmaker import config


@pytest.fixture
def reload_config(monkeypatch, tmp_path):
    """Reload config under a patched environment; restore the real one after."""
    def _reload(**env):
        for key in ("CLIPMAKER_RAW_FOOTAGE_DIR", "CLIPMAKER_EDIT_PROJECTS_DIR",
                    "CLIPMAKER_CATALOG_DIR", "CLIPMAKER_PROJECTS_DIR", "PEXELS_API_KEY",
                    "CLIPMAKER_HOME"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("CLIPMAKER_DOTENV", str(tmp_path / "no.env"))
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)


def test_dotenv_fills_only_unset_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nCM_A=from file\nexport CM_B='quoted'\nCM_C=\"x=y\"\n"
                   "CM_SET=file\nnot a pair\n", encoding="utf-8")
    for key in ("CM_A", "CM_B", "CM_C"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CM_SET", "env")
    config._load_dotenv(env)
    assert (os.environ["CM_A"], os.environ["CM_B"], os.environ["CM_C"]) == \
        ("from file", "quoted", "x=y")
    assert os.environ["CM_SET"] == "env"
    for key in ("CM_A", "CM_B", "CM_C"):
        monkeypatch.delenv(key)


def test_missing_dotenv_is_not_an_error(tmp_path):
    config._load_dotenv(tmp_path / "absent.env")


def test_inline_comment_is_cut_from_unquoted_values_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CM_P=footage/raw  # captures\nCM_Q='a # b'\n", encoding="utf-8")
    for key in ("CM_P", "CM_Q"):
        monkeypatch.delenv(key, raising=False)
    config._load_dotenv(env)
    assert (os.environ["CM_P"], os.environ["CM_Q"]) == ("footage/raw", "a # b")
    for key in ("CM_P", "CM_Q"):
        monkeypatch.delenv(key)


def test_utf16_dotenv_fails_with_a_clear_message(tmp_path):
    env = tmp_path / ".env"
    env.write_text("CM_X=1\n", encoding="utf-16")
    with pytest.raises(RuntimeError, match="not UTF-8"):
        config._load_dotenv(env)


def test_relative_env_path_is_taken_from_the_repo_root(reload_config):
    cfg = reload_config(CLIPMAKER_CATALOG_DIR="examples/sample/catalog")
    assert cfg.CATALOG_DIR == cfg.ROOT_DIR / "examples" / "sample" / "catalog"


def test_data_root_follows_where_the_package_runs(reload_config):
    cfg = reload_config()
    if (cfg.PACKAGE_DIR.parent / "pyproject.toml").is_file():  # a checkout / pip install -e
        assert cfg.ROOT_DIR == cfg.PACKAGE_DIR.parent
    else:  # pip install .: never site-packages
        assert cfg.ROOT_DIR == Path.cwd().resolve()


def test_clipmaker_home_wins(reload_config, tmp_path):
    cfg = reload_config(CLIPMAKER_HOME=str(tmp_path))
    assert cfg.ROOT_DIR == tmp_path.resolve()
    assert cfg.CATALOG_DIR == tmp_path.resolve() / "catalog"


def test_an_installed_package_uses_the_current_folder(monkeypatch, tmp_path):
    # A regular `pip install .`: no pyproject.toml next to the package.
    monkeypatch.delenv("CLIPMAKER_HOME", raising=False)
    monkeypatch.setattr(config, "PACKAGE_DIR", tmp_path / "site-packages" / "clipmaker")
    monkeypatch.chdir(tmp_path)
    assert config._data_root() == tmp_path.resolve()


def test_games_file_in_the_data_folder_is_found(reload_config, tmp_path):
    (tmp_path / "games_local.py").write_text(
        "DEFAULT_GAME = 'mine'\n"
        "def games(raw, edit):\n"
        "    return {'mine': {'video_dirs': [raw / 'Mine'], 'catalog_subdir': 'mine_videos'}}\n",
        encoding="utf-8")
    cfg = reload_config(CLIPMAKER_HOME=str(tmp_path))
    if (cfg.PACKAGE_DIR / "games_local.py").is_file():
        pytest.skip("this checkout has its own clipmaker/games_local.py, which comes first")
    assert cfg.DEFAULT_GAME == "mine" and "mine" in cfg.GAMES


def test_defaults_stay_inside_the_repo(reload_config):
    cfg = reload_config()
    assert cfg.RAW_FOOTAGE_DIR == cfg.ROOT_DIR / "footage" / "raw"
    assert cfg.EDIT_PROJECTS_DIR == cfg.ROOT_DIR / "footage" / "edit"
    assert cfg.CATALOG_DIR == cfg.ROOT_DIR / "catalog"
    assert cfg.PEXELS_API_KEY is None


def test_every_game_folder_hangs_off_a_configured_root(reload_config, tmp_path):
    raw, edit = tmp_path / "raw", tmp_path / "edit"
    cfg = reload_config(CLIPMAKER_RAW_FOOTAGE_DIR=str(raw),
                        CLIPMAKER_EDIT_PROJECTS_DIR=str(edit),
                        CLIPMAKER_CATALOG_DIR=str(tmp_path / "cat"),
                        PEXELS_API_KEY="k")
    for game in cfg.GAMES.values():
        for folder in game["video_dirs"]:
            assert folder.parent in (raw, edit), folder
    assert cfg.EMBED_INDEX_DIR == tmp_path / "cat" / "_index"
    assert cfg.PEXELS_API_KEY == "k"

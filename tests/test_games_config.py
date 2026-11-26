"""GAMES comes from clipmaker/games_local.py when it exists, else the sample game."""
from pathlib import Path
from types import SimpleNamespace

from clipmaker import config


def test_sample_game_matches_the_bundled_sample_catalog():
    sample = config.SAMPLE_GAMES[config.SAMPLE_DEFAULT_GAME]
    assert config.SAMPLE_DEFAULT_GAME == "assassins_creed_4"
    assert (Path(__file__).resolve().parent.parent / "examples" / "sample" / "catalog" / sample["catalog_subdir"]).is_dir()


def test_local_games_replace_the_sample(tmp_path):
    seen = {}

    def games(raw, edit):
        seen["dirs"] = (raw, edit)
        return {"my_game": {"video_dirs": [raw / "My Game"], "catalog_subdir": "my_game_videos"}}

    local = SimpleNamespace(games=games, DEFAULT_GAME="my_game")
    loaded, default = config._load_games(local)
    assert list(loaded) == ["my_game"] and default == "my_game"
    assert seen["dirs"] == (config.RAW_FOOTAGE_DIR, config.EDIT_PROJECTS_DIR)


def test_without_local_module_only_the_sample_is_loaded(tmp_path):
    loaded, default = config._load_games(path=tmp_path / "games_local.py")
    assert loaded == config.SAMPLE_GAMES and loaded is not config.SAMPLE_GAMES
    assert default == config.SAMPLE_DEFAULT_GAME


def test_config_loaded_standalone_still_sees_games_local(tmp_path):
    """Newsroom's footage gate loads config.py by path, without the package."""
    import importlib.util
    import shutil
    pkg = tmp_path / "clipmaker"
    pkg.mkdir()
    shutil.copy(config.__file__, pkg / "config.py")
    (pkg / "games_local.py").write_text(
        "DEFAULT_GAME = 'mine'\n"
        "def games(raw, edit):\n"
        "    return {'mine': {'video_dirs': [raw / 'Mine'], 'catalog_subdir': 'mine_videos'}}\n",
        encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_standalone_config", pkg / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(module.GAMES) == ["mine"] and module.DEFAULT_GAME == "mine"

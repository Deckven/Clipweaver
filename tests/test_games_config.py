"""GAMES comes from clipmaker/games_local.py when it exists, else the sample game."""
from types import SimpleNamespace

from clipmaker import config


def test_sample_game_matches_the_bundled_sample_catalog():
    sample = config.SAMPLE_GAMES[config.SAMPLE_DEFAULT_GAME]
    assert config.SAMPLE_DEFAULT_GAME == "assassins_creed_4"
    assert (config.ROOT_DIR / "examples" / "sample" / "catalog" / sample["catalog_subdir"]).is_dir()


def test_local_games_replace_the_sample(tmp_path):
    seen = {}

    def games(raw, edit):
        seen["dirs"] = (raw, edit)
        return {"my_game": {"video_dirs": [raw / "My Game"], "catalog_subdir": "my_game_videos"}}

    local = SimpleNamespace(games=games, DEFAULT_GAME="my_game")
    loaded, default = config._load_games(local)
    assert list(loaded) == ["my_game"] and default == "my_game"
    assert seen["dirs"] == (config.RAW_FOOTAGE_DIR, config.EDIT_PROJECTS_DIR)


def test_without_local_module_only_the_sample_is_loaded(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "clipmaker.games_local" else real(name, *a))
    loaded, default = config._load_games()
    assert loaded == config.SAMPLE_GAMES and loaded is not config.SAMPLE_GAMES
    assert default == config.SAMPLE_DEFAULT_GAME

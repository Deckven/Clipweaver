"""`analyzer --dry-run` never does real work.

Before 2026-10-27 `--dry-run` only applied together with `--rename`; on its own
it was ignored and a real indexing run started (model load, GPU, catalog
writes) — the opposite of what the flag promises.
"""

import json
from pathlib import Path

import pytest

from clipmaker import analyzer, naming, vlm


GAME = "dry_run_game"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    (videos / "sub").mkdir(parents=True)
    cat_root = tmp_path / "catalog"
    cat_dir = cat_root / f"{GAME}_videos"
    cat_dir.mkdir(parents=True)

    monkeypatch.setattr(analyzer.config, "CATALOG_DIR", cat_root)
    monkeypatch.setitem(analyzer.config.GAMES, GAME, {
        "catalog_subdir": f"{GAME}_videos", "video_dirs": [videos],
    })

    def boom(*_a, **_k):
        raise AssertionError("dry run did real work")

    monkeypatch.setattr(analyzer, "analyze_video", boom)
    monkeypatch.setattr(vlm, "load_vlm", boom)
    monkeypatch.setattr(analyzer, "_cuda_available", boom)

    (videos / "new clip.mp4").write_bytes(b"")
    (videos / "sub" / "nested clip.mp4").write_bytes(b"")
    (videos / "cached clip.mp4").write_bytes(b"")
    cached = cat_dir / "cached clip.json"
    cached.write_text(json.dumps({"suggested_filename": "old.mp4",
                                  "source_path": str(videos / "cached clip.mp4")}),
                      encoding="utf-8")
    return videos, cat_root


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_dry_run_without_rename_lists_the_batch_and_touches_nothing(setup, capsys):
    videos, cat_root = setup
    before = _tree(cat_root)

    analyzer.main(["--game", GAME, "--dry-run", str(videos)])

    out = capsys.readouterr().out
    assert "DRY-RUN: 2 of 3 video(s) would be analyzed" in out
    assert "skip     cached clip.mp4" in out
    assert _tree(cat_root) == before  # not even the nested catalog folder


def test_dry_run_with_force_counts_cached_videos_too(setup, capsys):
    videos, cat_root = setup
    analyzer.main(["--game", GAME, "--dry-run", "--force", str(videos)])
    assert "DRY-RUN: 3 of 3 video(s) would be analyzed" in capsys.readouterr().out


def test_dry_run_regenerate_suggestions_writes_nothing(setup, monkeypatch, capsys):
    _, cat_root = setup
    monkeypatch.setattr(naming, "suggest_filename", lambda rec, orig: "new.mp4")
    cached = cat_root / f"{GAME}_videos" / "cached clip.json"
    before = cached.read_text(encoding="utf-8")

    analyzer.main(["--game", GAME, "--regenerate-suggestions", "--dry-run"])

    assert cached.read_text(encoding="utf-8") == before
    assert "would regenerate suggestions for 1/1" in capsys.readouterr().out


def test_index_plan_is_pure(setup):
    videos, cat_root = setup
    nested = videos / "sub" / "nested clip.mp4"
    plan = analyzer.index_plan(GAME, [nested], force=False)
    assert plan == [(nested, cat_root / f"{GAME}_videos" / "sub" / "nested clip.json", True)]
    assert not (cat_root / f"{GAME}_videos" / "sub").exists()

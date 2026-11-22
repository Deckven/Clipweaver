"""build_index on a fake embedder: incremental reuse, and the refusals that keep
an unplugged drive or a typo from wiping the index."""
import json

import numpy as np
import pytest

from clipmaker import config, embed_index


def entry(name):
    return {"game": "g", "original_name": f"{name}.mp4", "title": name, "summary": f"{name} summary",
            "tags": [name], "duration_sec": 60,
            "scenes": [{"start": "00:00", "end": "00:10", "type": "gameplay", "description": f"{name} scene"}]}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    cat = tmp_path / "catalog"
    sub = cat / "g_videos"
    sub.mkdir(parents=True)
    for n in ("a", "b", "c", "d"):
        (sub / f"{n}.json").write_text(json.dumps(entry(n)), encoding="utf-8")
    monkeypatch.setattr(config, "CATALOG_DIR", cat)
    monkeypatch.setattr(config, "EMBED_INDEX_DIR", cat / "_index")
    monkeypatch.setitem(config.GAMES, "g", {"video_dirs": [], "catalog_subdir": "g_videos"})
    monkeypatch.setattr(embed_index, "_index_cache", None)
    calls = []

    def fake_embed(texts, kind):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(embed_index, "embed_texts", fake_embed)
    return sub, calls


def test_rebuild_reuses_unchanged_files(catalog):
    _, calls = catalog
    first = embed_index.build_index()
    assert first.files_embedded == 4
    second = embed_index.build_index()
    assert second.files_embedded == 0 and second.documents == first.documents
    assert len(calls) == 1


def test_missing_catalog_leaves_the_index_alone(catalog, monkeypatch, tmp_path):
    embed_index.build_index()
    before = (config.EMBED_INDEX_DIR / "docs.json").read_bytes()
    monkeypatch.setattr(config, "CATALOG_DIR", tmp_path / "unplugged")
    with pytest.raises(RuntimeError, match="left as it is"):
        embed_index.build_index()
    assert (config.EMBED_INDEX_DIR / "docs.json").read_bytes() == before


def test_mass_drop_needs_force(catalog):
    sub, _ = catalog
    embed_index.build_index()
    for n in ("a", "b", "c"):
        (sub / f"{n}.json").unlink()
    with pytest.raises(RuntimeError, match="would drop 3 of 4"):
        embed_index.build_index()
    assert embed_index.build_index(force=True).files_dropped == 3


def test_unknown_game_is_refused_before_anything(catalog):
    with pytest.raises(ValueError, match="unknown game"):
        embed_index.build_index(game="no_such_game")
    assert not config.EMBED_INDEX_DIR.exists()


def test_save_leaves_no_temp_files(catalog):
    embed_index.build_index()
    names = sorted(p.name for p in config.EMBED_INDEX_DIR.iterdir())
    assert names == ["docs.json", "manifest.json", "vectors.npz"]


def test_changed_document_rules_rebuild_every_file(catalog, monkeypatch):
    embed_index.build_index()
    assert embed_index.build_index().files_embedded == 0
    monkeypatch.setattr(embed_index, "DOC_RULES", embed_index.DOC_RULES + 1)
    with pytest.raises(ValueError, match="full build"):
        embed_index.build_index(game="g")
    assert embed_index.build_index().files_embedded == 4
    assert embed_index.build_index().files_embedded == 0

"""build_index on a fake embedder: incremental reuse, and the refusals that keep
an unplugged drive or a typo from wiping the index."""
import json
from pathlib import Path

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
    before = embed_index.index_files().docs.read_bytes()
    monkeypatch.setattr(config, "CATALOG_DIR", tmp_path / "unplugged")
    with pytest.raises(RuntimeError, match="left as it is"):
        embed_index.build_index()
    assert embed_index.index_files().docs.read_bytes() == before


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
    assert names[0] == "CURRENT" and len(names) == 2 and names[1].startswith("gen-")
    generation = config.EMBED_INDEX_DIR / names[1]
    assert sorted(p.name for p in generation.iterdir()) == ["docs.json", "manifest.json", "vectors.npz"]


def test_an_interrupted_save_leaves_the_published_index_whole(catalog, monkeypatch):
    # Review by Vitaly Chashchin: the three files were replaced one by one, so a
    # refresh stopped after the vectors could pair new vectors with old documents
    # when both had the same length.
    sub, _ = catalog
    embed_index.build_index()
    before = embed_index.load_index()
    for n in ("a", "b", "c", "d"):
        (sub / f"{n}.json").write_text(json.dumps({**entry(n), "title": n + " renamed"}),
                                       encoding="utf-8")
    real_replace = embed_index.os.replace

    def crash_on_publish(src, dst):
        if Path(dst).name == "CURRENT":
            raise OSError("power cut")
        real_replace(src, dst)

    monkeypatch.setattr(embed_index.os, "replace", crash_on_publish)
    with pytest.raises(OSError, match="power cut"):
        embed_index.build_index()
    monkeypatch.setattr(embed_index, "_index_cache", None)
    after = embed_index.load_index()
    assert after.docs == before.docs
    assert np.array_equal(after.vectors, before.vectors)


def test_old_generations_are_pruned_but_the_previous_one_is_kept(catalog):
    sub, _ = catalog
    for i in range(3):
        (sub / "a.json").write_text(json.dumps({**entry("a"), "title": f"a{i}"}), encoding="utf-8")
        embed_index.build_index()
    generations = [p for p in config.EMBED_INDEX_DIR.iterdir() if p.name.startswith("gen-")]
    assert len(generations) == 2


def test_an_index_in_the_old_flat_layout_is_still_read(catalog):
    embed_index.build_index()
    current = embed_index.index_files()
    for f in (current.vectors, current.docs, current.manifest):
        f.replace(config.EMBED_INDEX_DIR / f.name)
    (config.EMBED_INDEX_DIR / "CURRENT").unlink()
    embed_index._index_cache = None
    assert len(embed_index.load_index()) > 0
    embed_index.build_index()  # the next save publishes a generation and drops the flat files
    assert not (config.EMBED_INDEX_DIR / "vectors.npz").exists()
    assert embed_index.index_files().vectors.parent.name.startswith("gen-")


def test_changed_document_rules_rebuild_every_file(catalog, monkeypatch):
    embed_index.build_index()
    assert embed_index.build_index().files_embedded == 0
    monkeypatch.setattr(embed_index, "DOC_RULES", embed_index.DOC_RULES + 1)
    with pytest.raises(ValueError, match="full build"):
        embed_index.build_index(game="g")
    assert embed_index.build_index().files_embedded == 4
    assert embed_index.build_index().files_embedded == 0


def test_forced_build_of_one_game_keeps_the_other_games(catalog, tmp_path, monkeypatch):
    # Review of 42afaac, item 2: --game a --force dropped every other game, and the
    # manifest kept their fingerprints, so the next plain build never brought them back.
    sub, _ = catalog
    other = sub.parent / "h_videos"
    other.mkdir()
    (other / "x.json").write_text(json.dumps({**entry("x"), "game": "h"}), encoding="utf-8")
    monkeypatch.setitem(config.GAMES, "h", {"video_dirs": [], "catalog_subdir": "h_videos"})

    full = embed_index.build_index()
    forced = embed_index.build_index(game="g", force=True)
    assert forced.files_embedded == 4
    assert forced.documents == full.documents
    after = embed_index.build_index()
    assert after.files_embedded == 0 and after.documents == full.documents

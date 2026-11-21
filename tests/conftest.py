"""Tests never touch your own data.

- Your .env is ignored: clipmaker.config would otherwise pick up your catalog,
  projects and footage folders (set before any test imports the package).
- Every test gets its catalog, projects, index, cache and frame folders in its
  own tmp_path, so a test that forgets to point somewhere cannot read or write
  the repository's catalog/, projects/ or _cache/.
"""
import os

import pytest

os.environ["CLIPMAKER_DOTENV"] = os.devnull


@pytest.fixture(autouse=True)
def _isolated_paths(request, tmp_path_factory, monkeypatch):
    if request.node.get_closest_marker("real_data"):
        return  # checks the author's own projects on purpose
    from clipmaker import config

    root = tmp_path_factory.mktemp("clipmaker")
    monkeypatch.setattr(config, "CATALOG_DIR", root / "catalog")
    monkeypatch.setattr(config, "PROJECTS_DIR", root / "projects")
    monkeypatch.setattr(config, "EMBED_INDEX_DIR", root / "catalog" / "_index")
    for name in ("SHOT_QUERY_CACHE", "VLM_WORK_DIR"):
        if hasattr(config, name):
            monkeypatch.setattr(config, name, root / name.lower())
    try:
        from clipmaker import embed_index
        monkeypatch.setattr(embed_index, "_index_cache", None)
    except ImportError:
        pass

"""constraints.txt pins a known-good point inside the pyproject.toml ranges.

pyproject.toml said transformers>=5,<6 while every indexing run used 4.57.6, so
a fresh install got a version the catalog was never built with (review of
42afaac, 2026-11-28). These tests keep the two files from drifting apart again.
"""
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO = Path(__file__).resolve().parent.parent


def _pins() -> dict[str, Version]:
    pins = {}
    for line in (REPO / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            assert version, f"constraints.txt pins exact versions only: {line!r}"
            pins[canonicalize_name(name)] = Version(version)
    return pins


def _declared() -> list[Requirement]:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    deps = list(project["dependencies"])
    for name, group in extras.items():
        if name not in ("all", "lint"):  # all is a self-reference; lint pins itself
            deps += group
    return [Requirement(d) for d in deps]


@pytest.mark.parametrize("req", _declared(), ids=str)
def test_every_declared_dependency_is_pinned_inside_its_range(req):
    pins = _pins()
    name = canonicalize_name(req.name)
    assert name in pins, f"{req.name} is not pinned in constraints.txt"
    assert req.specifier.contains(pins[name], prereleases=True), \
        f"constraints.txt pins {req.name}=={pins[name]}, outside pyproject's {req.specifier}"


def test_transformers_range_includes_the_version_the_catalog_was_built_with():
    (req,) = [r for r in _declared() if r.name == "transformers"]
    assert req.specifier.contains("4.57.6")

"""Write requirements.txt from pyproject.toml: the core dependencies, every extra
that ``[all]`` pulls in, and ``[test]``. pyproject.toml is the one source;
tests/test_cli.py fails when the two drift apart.

    python tools/requirements_txt.py
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

HEADER = """\
# Generated from pyproject.toml by tools/requirements_txt.py - edit pyproject.toml,
# then re-run the script. Everything: core + every extra in [all] + [test].
# Install torch from the PyTorch index that matches your CUDA first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
# System tools, not pip packages: ffmpeg/ffprobe on PATH; Ollama for shot
# queries and structured transcripts; Vegas Pro 22 for vegas_scripts/.
"""


def render(pyproject: dict) -> str:
    """requirements.txt text for a parsed pyproject.toml."""
    project = pyproject["project"]
    extras = project["optional-dependencies"]
    name = project["name"]
    in_all = re.fullmatch(rf"{name}\[(.+)\]", extras["all"][0])
    if not in_all:
        raise ValueError(f"[all] should be '{name}[a,b,...]', got {extras['all']!r}")
    groups = [("core", project["dependencies"])]
    groups += [(extra, extras[extra]) for extra in in_all.group(1).split(",")]
    groups.append(("test", extras["test"]))

    seen: set[str] = set()
    blocks = []
    for label, deps in groups:
        fresh = [d for d in deps if d not in seen]
        seen.update(fresh)
        if fresh:
            blocks.append(f"# {label}\n" + "\n".join(fresh))
    return HEADER + "\n" + "\n\n".join(blocks) + "\n"


def main() -> None:
    """Rewrite requirements.txt next to pyproject.toml."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    (REPO / "requirements.txt").write_text(render(pyproject), encoding="utf-8")


if __name__ == "__main__":
    main()

"""The clipweaver entry point reaches every tool, and every tool answers --help (R11)."""
import importlib
import importlib.util
import inspect
import tomllib
from pathlib import Path

import pytest

from clipmaker import cli

REPO = Path(__file__).resolve().parent.parent


def test_help_lists_every_tool(capsys):
    cli.main(["--help"])
    out = capsys.readouterr().out
    assert all(name in out for name in cli.TOOLS)


def test_unknown_tool_is_an_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-such-tool"])
    assert exc.value.code == 2
    assert "unknown tool 'no-such-tool'" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(cli.TOOLS))
def test_every_tool_takes_argv_and_answers_help(name, capsys):
    module = importlib.import_module(f"clipmaker.{cli.TOOLS[name][0]}")
    assert "argv" in inspect.signature(module.main).parameters
    with pytest.raises(SystemExit) as exc:
        cli.main([name.replace("_", "-"), "--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_every_public_tool_module_is_listed():
    # A module with a main() but no entry would only be reachable as python -m.
    private = {"cli", "vegas_migrate", "games_local"}
    with_main = {p.stem for p in (REPO / "clipmaker").glob("*.py")
                 if p.stem not in private and "\ndef main(" in p.read_text(encoding="utf-8")}
    assert with_main == {module for module, _ in cli.TOOLS.values()}


def test_console_script_is_declared():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"]["clipweaver"] == "clipmaker.cli:main"


def test_requirements_txt_is_generated_from_pyproject():
    spec = importlib.util.spec_from_file_location("requirements_txt", REPO / "tools" / "requirements_txt.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert (REPO / "requirements.txt").read_text(encoding="utf-8") == gen.render(pyproject), \
        "requirements.txt is stale: run python tools/requirements_txt.py"

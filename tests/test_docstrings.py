"""Every public function and class of the package says in one line what it does (R3)."""
import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "clipmaker"
MODULES = sorted(p for p in PACKAGE.glob("*.py") if p.name != "games_local.py")


def _is_stub(fn: ast.FunctionDef) -> bool:
    """A Protocol method: `...` and nothing else."""
    return len(fn.body) == 1 and isinstance(fn.body[0], ast.Expr) and fn.body[0].value.value is Ellipsis


def undocumented(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name.startswith("_"):
            continue
        if ast.get_docstring(node) is None:
            missing.append(node.name)
        if isinstance(node, ast.ClassDef):
            missing += [f"{node.name}.{m.name}" for m in node.body
                        if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")
                        and not _is_stub(m) and ast.get_docstring(m) is None]
    return missing


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_public_names_have_a_docstring(path):
    assert undocumented(path) == []


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_has_a_docstring(path):
    assert ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))), path.name

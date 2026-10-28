# Copyright 2026 Vidip Gaur, Vidu Gaur
"""scripts/ is split into pipeline/, analysis/, verify/ and dev/. Scripts import each
other by bare module name, so every script that imports a sibling must be able to
resolve it from the folders it puts on sys.path. Needs no third-party package."""
import ast
from importlib.machinery import PathFinder

import pytest

from conftest import FOLDERS, SCRIPT_DIRS, SCRIPTS, dev_only, script_files

FILES = script_files()
STEMS = {p.stem for p in FILES}


def sibling_imports(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return (found & STEMS) - {path.stem}


def test_four_folders_and_nothing_loose():
    assert all(d.is_dir() for d in SCRIPT_DIRS)
    assert not list(SCRIPTS.glob("*.py")), "scripts must live in one of the four folders"
    assert {p.name for p in SCRIPTS.iterdir() if p.is_dir() and not p.name.startswith("__")} == set(FOLDERS)


def test_script_names_are_unique():
    names = [p.name for p in FILES]
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)


def test_unshipped_names_are_absent():
    assert not (dev_only() & {p.name for p in FILES})


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_sibling_imports_resolve(path):
    """Simulate the bootstrap: the script's own folder first, then the four folders."""
    search = [str(path.parent)] + [str(d) for d in SCRIPT_DIRS if d != path.parent]
    missing = [m for m in sibling_imports(path) if PathFinder.find_spec(m, search) is None]
    assert not missing, f"{path.name} imports {missing}, not found in {search}"
    if sibling_imports(path):
        src = path.read_text(encoding="utf-8")
        assert "for _d in (\"pipeline\", \"analysis\", \"verify\", \"dev\")" in src, \
            f"{path.name} imports siblings but does not add the four folders to sys.path"

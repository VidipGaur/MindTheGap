# Copyright 2026 Vidip Gaur, Vidu Gaur
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
FOLDERS = ("pipeline", "analysis", "verify", "dev")
SCRIPT_DIRS = [SCRIPTS / f for f in FOLDERS]
TOOLS = REPO / "tools"


def script_files():
    return sorted(p for d in SCRIPT_DIRS for p in d.glob("*.py"))


def script_path(name):
    found = [d / name for d in SCRIPT_DIRS if (d / name).exists()]
    assert len(found) == 1, f"{name}: expected one copy under scripts/, found {found}"
    return found[0]


def dev_only():
    lines = (TOOLS / "dev_only.txt").read_text().splitlines()
    return {s.strip() for s in lines if s.strip() and not s.startswith("#")}


def clean_env(**extra):
    """Environment for a child process: no dataset root, no display, no GPU."""
    env = {k: v for k, v in os.environ.items() if k not in ("UBC_ROOT", "UBC_RUN_DIR")}
    env.update(MPLBACKEND="Agg", PYTHONPATH=os.pathsep.join(str(d) for d in SCRIPT_DIRS), CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def run_py(args, cwd, timeout=300, **env):
    return subprocess.run([sys.executable, *args], cwd=cwd, env=clean_env(**env),
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def run(tmp_path):
    """Run python with the given args from an empty directory."""
    def _run(*args, timeout=300, **env):
        return run_py(list(args), cwd=tmp_path, timeout=timeout, **env)
    return _run

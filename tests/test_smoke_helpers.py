# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Pieces of the smoke harness that can be checked without slides or a GPU."""
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "smoke"))
import build_manifest as M  # noqa: E402

TRAIN = {
    "10077": dict(image_id="10077", label="EC", image_width="53143", image_height="39150", is_tma="False"),
    "7": dict(image_id="7", label="CC", image_width="3000", image_height="3000", is_tma="True"),
}


def test_manifest_row_uses_train_csv_dimensions():
    rows, problems = M.build_rows(TRAIN, ["10077"], Path("/t"), lambda p: (53143, 39150))
    assert problems == []
    assert rows == [dict(image_id="10077", label="EC", image_width="53143", image_height="39150",
                         is_tma="False", file_path=str(Path("/t") / "10077.tiff"))]
    assert list(rows[0]) == M.COLUMNS
    assert "mpp" not in M.COLUMNS


@pytest.mark.parametrize("ids, dims, needle", [
    (["10077"], (39150, 53143), "TIFF is 39150x53143"),
    (["7"], None, "is_tma=True"),
    (["404"], None, "not in train.csv"),
])
def test_manifest_problems_stop_the_row(ids, dims, needle):
    rows, problems = M.build_rows(TRAIN, ids, Path("/t"), (lambda p: dims) if dims else None)
    assert rows == []
    assert any(needle in p for p in problems)


@pytest.mark.parametrize("script", ["fetch_slides.sh", "run_cellvit.sh", "run_smoke.sh"])
def test_shell_scripts_parse(script):
    r = subprocess.run(["bash", "-n", str(REPO / "smoke" / script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

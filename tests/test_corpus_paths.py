# Copyright 2026 Vidip Gaur, Vidu Gaur
import csv
import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture
def load(monkeypatch):
    """Re-import corpus_paths under a controlled environment and cwd."""
    def _load(cwd, **env):
        for k in ("UBC_ROOT", "UBC_RUN_DIR"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.chdir(cwd)
        import corpus_paths
        return importlib.reload(corpus_paths)
    return _load


def test_default_root_is_cwd_relative(load, tmp_path):
    cp = load(tmp_path)
    assert cp.ROOT == tmp_path / "data" / "UBC-OCEAN"
    assert cp.RUN_DIR == cp.ROOT / "cellvit_run"
    assert cp.IS_EXTERNAL is False


def test_root_is_exported_for_child_processes(load, tmp_path):
    cp = load(tmp_path)
    assert os.environ["UBC_ROOT"] == str(cp.ROOT)


def test_env_root_and_run_dir(load, tmp_path):
    root = tmp_path / "somewhere"
    cp = load(tmp_path, UBC_ROOT=str(root), UBC_RUN_DIR="bentaeb_run")
    assert cp.ROOT == root
    assert cp.RUN_DIR == root / "bentaeb_run"
    assert cp.GRAPH_DIR == root / "bentaeb_run" / "graphs"
    assert cp.TIFF_DIR == root / "bentaeb_run" / "wsi_pyramid_tiff"
    assert cp.IS_EXTERNAL is True


def test_relative_env_root_is_made_absolute(load, tmp_path):
    cp = load(tmp_path, UBC_ROOT="rel/root")
    assert cp.ROOT == tmp_path / "rel" / "root"


@pytest.mark.parametrize("name, expected", [
    ("10_OOU198_EC_SC_graph.npz", "10_OOU198_EC_SC"),
    ("10077_graph.npz", "10077"),
    ("/a/b/10077_graph.npz", "10077"),
    ("10077_kde.npz", "10077_kde"),
])
def test_image_id_from_graph(load, tmp_path, name, expected):
    cp = load(tmp_path)
    assert cp.image_id_from_graph(name) == expected


def test_slide_path_without_manifest(load, tmp_path):
    cp = load(tmp_path, UBC_ROOT=str(tmp_path))
    assert cp.manifest() == {}
    assert cp.slide_path("10077") == tmp_path / "cellvit_run" / "wsi_pyramid_tiff" / "10077.tiff"
    assert cp.native_mpp("10077") is None
    assert cp.nominal_mpp("10077") is None


def test_slide_path_with_manifest_row(load, tmp_path):
    run = tmp_path / "cellvit_run"
    run.mkdir()
    with open(run / "cellvit_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "label", "slide_path", "mpp", "native_mpp"])
        w.writeheader()
        w.writerow(dict(image_id="A1", label="EC", slide_path="/x/A1.svs", mpp="0.2496", native_mpp="0.4992"))
        w.writerow(dict(image_id="B2", label="CC", slide_path="", mpp="", native_mpp=""))
    cp = load(tmp_path, UBC_ROOT=str(tmp_path))
    assert cp.slide_path("A1") == Path("/x/A1.svs")
    assert cp.native_mpp("A1") == pytest.approx(0.4992)
    assert cp.nominal_mpp("A1") == pytest.approx(0.2496)
    # a row with an empty slide_path falls back to the UBC-OCEAN default
    assert cp.slide_path("B2") == run / "wsi_pyramid_tiff" / "B2.tiff"
    assert cp.native_mpp("B2") is None
    assert "manifest_rows=2" in cp.banner("test")

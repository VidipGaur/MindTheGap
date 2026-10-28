# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Offline end-to-end check of the training-side loaders on five synthetic slides.

Graphs are produced by the real builder (build_streamline_graphs_bridge.process_one)
from synthetic density maps, the same way its --self-test does. The nuclear,
chromatin and eosin blocks are random but carry each extractor's real
FEATURE_NAMES and the graph's grid_rc_checksum. smoke/integration_check.py then
loads them through fm_subspace_split.named_features() and bags_for(), so this
pins the arithmetic behind "49 named columns" and "10 stain columns" without any
real data. nuc_have_cv2 is held at 1.0, as it was across the real corpus, so
drop_degenerate removes it the same way.
"""
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import REPO, run_py

import build_streamline_graphs_bridge as B
import extract_chromatin_features as XC
import extract_eosin_features as XE
import extract_nuclear_features as XN

LABELS = ["CC", "EC", "HGSC", "LGSC", "MC"]


def kde(path, label, seed):
    rng = np.random.default_rng(seed)
    ds, H, W = B.EXPECTED_DOWNSAMPLE, 200, 360
    rr = np.arange(H, dtype=np.float32)[:, None]
    bands = (1.0 + np.sin(2 * np.pi * rr / (40.0 + 5 * seed))) * np.ones((1, W), np.float32)
    dens = gaussian_filter(bands + 0.5 * rng.random((H, W)).astype(np.float32), 3.0).astype(np.float32)
    np.savez_compressed(
        path, schema_version=np.array([B.KDE_SCHEMA_VERSION]),
        density_all=dens, density_neoplastic=(dens * 0.6).astype(np.float32),
        density_epithelial=(dens * 0.15).astype(np.float32),
        density_connective=(dens * 0.2).astype(np.float32),
        density_inflammatory=(dens * 0.05).astype(np.float32),
        density_dead=np.zeros_like(dens),
        downsample=np.array([ds]), sigma_px=np.array([15.0]), sigma_um=np.array([60.0]),
        pixel_size_um=np.array([B.WSI_MPP]),
        wsi_shape=np.array([H * ds, W * ds]), label=np.array([label]),
        image_id=np.array([path.stem]),
        grid_x=(np.arange(W) + 0.5) * ds, grid_y=(np.arange(H) + 0.5) * ds,
        cell_counts_json=np.array([json.dumps(
            {"Neoplastic": 50000, "Epithelial": 20000, "Connective": 30000,
             "Inflammatory": 8000, "Dead": 10})]),
        coord_diag_json=np.array([json.dumps({"frac_oob": 0.0})]))
    return H * ds, W * ds


def block(rng, n, names, flags):
    F = rng.random((n, len(names))).astype(np.float32) * 5
    for j, name in enumerate(names):
        if name in flags:
            F[:, j] = rng.integers(0, 2, n)
    return F


def make_corpus(root, per_class):
    g = root / "cellvit_run" / "graphs"
    for sub in ("nuclear_feats", "chromatin_feats", "eosin_feats"):
        (g / sub).mkdir(parents=True)
    (root / "cellvit_run" / "kde_maps").mkdir()
    k = 0
    for label in LABELS:
        for _ in range(per_class):
            iid = str(9000 + k)
            kp = root / "cellvit_run" / "kde_maps" / f"{iid}.npz"
            h, w = kde(kp, label, k)
            gp = g / f"{iid}_graph.npz"
            B.process_one(iid, kp, gp, h, w, save_polylines=False)
            with np.load(gp, allow_pickle=True) as z:
                n = z["node_features"].shape[0]
                ck = int(np.asarray(z["node_grid_rc"]).sum())
            rng = np.random.default_rng(100 + k)
            nuc = block(rng, n, XN.FEATURE_NAMES, XN.FLAG_COLS)
            nuc[:, XN.FEATURE_NAMES.index("nuc_have_cv2")] = 1.0
            np.savez_compressed(g / "nuclear_feats" / f"{iid}_nuclei.npz",
                                nuc_features=nuc,
                                nuc_feature_names=np.array(XN.FEATURE_NAMES, dtype=object),
                                grid_rc_checksum=np.array([ck]))
            np.savez_compressed(g / "chromatin_feats" / f"{iid}_chromatin.npz",
                                chrom_features=block(rng, n, XC.FEATURE_NAMES, XC.FLAG_COLS),
                                chrom_feature_names=np.array(XC.FEATURE_NAMES, dtype=object),
                                grid_rc_checksum=np.array([ck]))
            np.savez_compressed(g / "eosin_feats" / f"{iid}_eosin.npz",
                                eos_features=block(rng, n, XE.FEATURE_NAMES, XE.FLAG_COLS),
                                eos_feature_names=np.array(XE.FEATURE_NAMES, dtype=object),
                                grid_rc_checksum=np.array([ck]))
            k += 1
    return root


def integration(root, *extra):
    out = root / "integration.json"
    r = run_py([str(REPO / "smoke" / "integration_check.py"), "--json", str(out), *extra],
               cwd=root, timeout=900, UBC_ROOT=str(root))
    print(r.stdout[-4000:])
    return r, json.loads(out.read_text())


def counts(report):
    rows = {row["check"]: row for row in report["rows"]}
    return (rows["named_features('morph') columns"]["observed"],
            rows["named_features('stain') columns"]["observed"])


def test_five_per_class_gives_49_named_and_10_stain_columns(tmp_path):
    r, rep = integration(make_corpus(tmp_path, per_class=5))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert counts(rep) == ("49", "10")


def test_one_per_class_is_refused_by_the_flag_leak_report(tmp_path):
    root = make_corpus(tmp_path, per_class=1)
    r, rep = integration(root)
    assert r.returncode == 2
    assert "n_splits=5 cannot be greater than the number of members in each class" in rep["limit"]
    r, rep = integration(root, "--skip-flag-leak-report")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert counts(rep) == ("49", "10")

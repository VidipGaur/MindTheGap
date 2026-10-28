# Copyright 2026 Vidip Gaur, Vidu Gaur
"""The smoke checker must pass a well-formed synthetic run and fail each kind of
corruption with a named check. Synthetic files use the real keys; sizes are
scaled down (32 px tiles) so the TIFFs stay small."""
import copy
import csv
import json
import sys

import numpy as np
import pytest
import tifffile

from conftest import REPO

sys.path.insert(0, str(REPO / "smoke"))
import verify_smoke_outputs as V  # noqa: E402

TILE = 32
GRID = (20, 25)
WSI = (GRID[0] * TILE - 7, GRID[1] * TILE - 3)
N_NODES = 400
WIDTHS = {"graph_node": 39, "graph_edge": 23, "nuclear": 39, "chromatin": 39, "eosin": 35,
          "fm": {"phikonv2": 1024}}
SLIDES = [("101", "CC"), ("202", "EC")]


def build(root, seed=0):
    rng = np.random.default_rng(seed)
    run = root / "cellvit_run"
    g = run / "graphs"
    for d in (run / "wsi_pyramid_tiff", g / "nuclear_feats", g / "chromatin_feats",
              g / "eosin_feats", g / "fm_feats"):
        d.mkdir(parents=True, exist_ok=True)
    with open(root / "train.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "image_width", "image_height", "is_tma"])
        for iid, lab in SLIDES:
            w.writerow([iid, lab, WSI[1], WSI[0], "False"])

    cells = np.array([(r, c) for r in range(GRID[0]) for c in range(GRID[1])])
    oracle, check_rows = {}, []
    for iid, lab in SLIDES:
        rc = cells[rng.choice(len(cells), N_NODES, replace=False)]
        rc = rc[np.lexsort((rc[:, 1], rc[:, 0]))]
        ck = int(rc.sum())
        dens = rng.random(N_NODES)
        nf = rng.random((N_NODES, 39)).astype(np.float32)
        names = [f"g{j}" for j in range(39)]
        names[0] = "dens_all_mean"
        nf[:, 0] = dens
        tifffile.imwrite(run / "wsi_pyramid_tiff" / f"{iid}.tiff",
                         np.full((*WSI, 3), 200, np.uint8))
        np.savez_compressed(
            g / f"{iid}_graph.npz", image_id=np.array([iid]), label=np.array([lab]),
            node_features=nf, node_feature_names=np.array(names, dtype=object),
            edge_index=rng.integers(0, N_NODES, size=(2, 50)),
            edge_attr=rng.random((50, 23)).astype(np.float32),
            edge_attr_names=np.array([f"e{j}" for j in range(23)], dtype=object),
            node_grid_rc=rc, node_grid_shape=np.array(GRID), wsi_shape=np.array(WSI),
            tile_size_wsi_px=np.array([TILE]),
            channel_names=np.array(["neoplastic", "epithelial", "connective", "inflammatory", "interface"]))
        cnt = np.round(dens * 1000 + rng.normal(0, 20, N_NODES)).clip(0)
        nuc = rng.random((N_NODES, 39)).astype(np.float32)
        nuc[:, 0] = cnt
        nnames = ["nuc_count"] + [f"n{j}" for j in range(1, 39)]
        np.savez_compressed(
            g / "nuclear_feats" / f"{iid}_nuclei.npz", image_id=np.array([iid]), label=np.array([lab]),
            nuc_features=nuc, nuc_feature_names=np.array(nnames, dtype=object),
            grid_rc_checksum=np.array([ck]), n_cells=np.array([int(cnt.sum())]),
            coverage=np.array([0.97]))
        for block, key, n, fname in (("chromatin", "chrom", 39, "chromatin"), ("eosin", "eos", 35, "eosin")):
            np.savez_compressed(
                g / f"{block}_feats" / f"{iid}_{fname}.npz", image_id=np.array([iid]),
                **{f"{key}_features": rng.random((N_NODES, n)).astype(np.float32),
                   f"{key}_feature_names": np.array([f"{key}{j}" for j in range(n)], dtype=object)},
                grid_rc_checksum=np.array([ck]))
        np.savez_compressed(
            g / "fm_feats" / f"{iid}_phikonv2.npz", image_id=np.array([iid]), label=np.array([lab]),
            fm_features=rng.normal(size=(N_NODES, 1024)).astype(np.float32),
            grid_rc_checksum=np.array([ck]))
        rho = V.rank_rho(cnt, dens)
        check_rows.append(dict(image_id=iid, spearman_count_vs_density=round(rho, 3),
                               median_nuc_area_um2=40.0))
        oracle[iid] = dict(n_cells=int(cnt.sum()), rho=round(rho, 3))
    with open(g / "nuclear_feats" / "_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(check_rows[0]))
        w.writeheader()
        w.writerows(check_rows)

    base = json.loads((REPO / "smoke" / "slides.json").read_text())
    cfg = dict(base, widths=WIDTHS, tile_size_wsi_px=TILE, exact={
        "101": dict(node_features=[N_NODES, 39], edge_attr=[50, 23], edge_index=[2, 50],
                    node_grid_shape=list(GRID), wsi_shape=list(WSI), tile_size_wsi_px=TILE,
                    n_channel_names=5, nuclear=[N_NODES, 39])})
    cfg["slides"] = [
        dict(image_id="101", label="CC", n_cells=oracle["101"]["n_cells"], n_tiles=N_NODES,
             coverage=0.97, spearman_count_density=oracle["101"]["rho"], spearman_definition="npz",
             shuffled_control=None, median_nuc_area_um2=41.0),
        dict(image_id="202", label="EC", n_cells=None, n_tiles=None, coverage=None,
             spearman_count_density=oracle["202"]["rho"], spearman_definition="check_csv",
             shuffled_control=None, median_nuc_area_um2=None),
    ]
    (root / "slides.json").write_text(json.dumps(cfg))
    return root


def run_checker(root):
    out = root / "report.json"
    rc = V.main(["--root", str(root), "--slides", str(root / "slides.json"), "--json", str(out)])
    return rc, json.loads(out.read_text())["rows"]


def failed(rows):
    return {(r["slide"], r["check"]) for r in rows if r["status"] == "FAIL"}


@pytest.fixture
def root(tmp_path):
    return build(tmp_path)


def test_clean_run_passes(root):
    rc, rows = run_checker(root)
    assert rc == 0, failed(rows)
    statuses = {r["status"] for r in rows}
    assert statuses <= {"PASS", "N/A"}, [r for r in rows if r["status"] not in ("PASS", "N/A")]
    assert any(r["check"] == "oracle edge_attr shape" and r["status"] == "PASS" for r in rows)
    assert any(r["check"] == "spearman(count, density)" and r["status"] == "PASS" for r in rows)


def rewrite(path, **changes):
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d.update(changes)
    np.savez_compressed(path, **d)


def test_wrong_width_fails(root):
    p = root / "cellvit_run/graphs/101_graph.npz"
    with np.load(p, allow_pickle=True) as z:
        nf, names = z["node_features"][:, :38], z["node_feature_names"][:38]
    rewrite(p, node_features=nf, node_feature_names=names)
    rc, rows = run_checker(root)
    assert rc == 1
    assert ("101", "graph node_features width") in failed(rows)


def test_nan_fails(root):
    p = root / "cellvit_run/graphs/nuclear_feats/202_nuclei.npz"
    with np.load(p, allow_pickle=True) as z:
        F = z["nuc_features"].copy()
    F[3, 5] = np.nan
    rewrite(p, nuc_features=F)
    rc, rows = run_checker(root)
    assert rc == 1
    assert ("202", "nuclear finite") in failed(rows)


def test_checksum_mismatch_fails(root):
    p = root / "cellvit_run/graphs/chromatin_feats/101_chromatin.npz"
    with np.load(p) as z:
        ck = int(z["grid_rc_checksum"][0])
    rewrite(p, grid_rc_checksum=np.array([ck + 1]))
    rc, rows = run_checker(root)
    assert rc == 1
    assert ("101", "chromatin grid_rc_checksum == graph") in failed(rows)


def test_swapped_label_fails(root):
    rewrite(root / "cellvit_run/graphs/101_graph.npz", label=np.array(["EC"]))
    rc, rows = run_checker(root)
    assert rc == 1
    assert ("101", "graph label == train.csv") in failed(rows)


def test_transposed_shape_fails(root):
    rewrite(root / "cellvit_run/graphs/202_graph.npz", wsi_shape=np.array(WSI[::-1]))
    rc, rows = run_checker(root)
    assert rc == 1
    assert ("202", "wsi_shape == train.csv (height, width)") in failed(rows)


def test_missing_block_fails(root):
    (root / "cellvit_run/graphs/eosin_feats/202_eosin.npz").unlink()
    rc, rows = run_checker(root)
    assert rc == 1
    assert ("202", "eosin file") in failed(rows)


def test_tolerance_miss_fails(root):
    cfg = json.loads((root / "slides.json").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["slides"][0]["n_cells"] = int(cfg["slides"][0]["n_cells"] * 1.2)
    (root / "slides.json").write_text(json.dumps(cfg))
    rc, rows = run_checker(root)
    assert rc == 1
    assert ("101", "n_cells") in failed(rows)

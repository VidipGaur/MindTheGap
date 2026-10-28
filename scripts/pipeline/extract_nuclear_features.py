# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
P4 -- nuclear morphometrics as per-tile features, aligned to the existing graphs.

    python scripts/pipeline/extract_nuclear_features.py --check          # verify coord alignment on 6 slides, write nothing
    python scripts/pipeline/extract_nuclear_features.py --smoke          # 3 slides, full pipeline, writes to nuclear_feats/
    python scripts/pipeline/extract_nuclear_features.py                  # all 456 slides (resumes)
    python scripts/pipeline/extract_nuclear_features.py --only 10077,1289

WHAT THIS PRODUCES
==================
For every slide with a `*_graph.npz`, one file:

    cellvit_run/graphs/nuclear_feats/<id>_nuclei.npz

holding `nuc_features` of shape (K, F) where K is the SAME node count and the
SAME node order as `node_features` in `<id>_graph.npz`. Row i here describes the
same 512um tile as row i there, so the two can be concatenated column-wise with
no reindexing. `nuc_feature_names`, `nuc_log1p_cols` (which columns are
count-like and want log1p before z-scoring, same convention as the graph
builder), and per-slide coverage QC are included.

WHY THESE FEATURES
==================
The v5 confusion matrix (Aug 2026) shows the dominant error is HGSC <-> EC, not
HGSC <-> LGSC as the queue text assumed -- EC recall is ~0.38 and its mass goes
to HGSC; HGSC's largest off-diagonal (~0.25) goes to EC. Both of those splits
are driven by nuclear grade: HGSC has marked pleomorphism and irregular nuclear
membranes, EC (grades 1-2) and LGSC have uniform, rounder nuclei. The graph's
39 features carry density and coherence but nothing about nuclear SHAPE at all,
so the model has no access to the actual diagnostic criterion. This adds it:
per-tile distributions of nuclear area, area CV (pleomorphism), eccentricity
spread, solidity, plus tumour-infiltrating-lymphocyte spatial features.

COORDINATE SPACE (read before trusting the output)
=================================================
CellViT `cells.json` contours are in WSI base pixels at mpp = base_mpp (0.25 for
this corpus, 40x). The graph's `node_xy_wsi` / `tile_size_wsi_px` are in the
same WSI base pixels (KDE grid_x = (arange(W)+0.5)*downsample). So a cell at
(x, y) belongs to grid tile (row = y // tile_px, col = x // tile_px), which is
exactly the builder's `np.minimum((arange / node_size_px), n-1)` tiling scaled
by `downsample`. `--check` verifies this empirically: it reports what fraction
of cells land on a tissue node and the rank correlation between per-tile nucleus
count and the graph's own `dens_all_mean`. If coverage is low or the correlation
is weak, the coordinate spaces disagree (a transpose or the PNG->TIFF dim fix)
and nothing downstream should be trusted -- fix that first.

This project has been bitten by silent coordinate bugs before. Do not skip
`--check`.

STREAMING
=========
`cells.json` is 100-500 MB per slide and this box has 16 GB RAM, so it is never
loaded whole. If `ijson` is installed it is used; otherwise a small hand-rolled
streaming scanner walks the top-level `cells` array object by object. Both paths
give identical numbers. `pip install ijson` makes it ~2x faster but is optional.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
import json
import math
import sys
import time
import codecs
from pathlib import Path

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

import corpus_paths

ROOT = corpus_paths.ROOT
GRAPH_DIR = corpus_paths.GRAPH_DIR
DET_DIR = corpus_paths.DET_DIR
OUT_DIR = GRAPH_DIR / "nuclear_feats"

CLASSES = ["CC", "EC", "HGSC", "LGSC", "MC"]

# CellViT type_map for this corpus (confirmed from cell_detection.json /
# cells.geojson on slide 10077). Integer code -> name.
TYPE_MAP = {1: "neoplastic", 2: "inflammatory", 3: "connective",
            4: "dead", 5: "epithelial"}
NAME2CODE = {v: k for k, v in TYPE_MAP.items()}

MIN_NUC = 8            # tiles with fewer total nuclei -> morph_measured = 0
MIN_NEO = 10           # tiles with fewer neoplastic nuclei -> neo_measured = 0
INFILTRATE_UM = 30.0   # a neoplastic nucleus is "infiltrated" if an inflammatory
                       # nucleus sits within this distance
MIN_TYPE_PROB = 0.0    # raise to e.g. 0.5 to drop low-confidence detections

# nuclear plausibility guard (um^2). Anything outside is a segmentation artefact
# and is dropped from morphometry (still counted for density). Deliberately wide
# because CellViT/SAM under-segments small nuclei at 20x.
AREA_UM2_MIN, AREA_UM2_MAX = 2.0, 900.0

# The graph builder hard-codes wsi_mpp = 0.25 for every slide ("matches
# --wsi_mpp used at CellViT inference"), which is an unverified constant.
# Empirically it is wrong for the WSI slides: on slide 10077 (is_tma=False) the median neoplastic
# nucleus is 180 px^2, i.e. 3.8 um diameter at 0.25 -- impossibly small -- vs
# 7.6 um at 0.50. UBC-OCEAN WSIs are 20x (~0.5 um/px); the TMA cores are 40x
# (~0.25). So physical-unit nuclear features need a per-slide mpp keyed on
# is_tma, NOT the stored 0.25. Pass --mpp-from-graph to force the old constant
# (keeps P4 units consistent with the graph's streamline lengths, at the cost of
# every area being 4x too small).
DEFAULT_MPP_WSI = 0.50
DEFAULT_MPP_TMA = 0.25


def iter_cells(path: Path):
    """Yield each object in the top-level `cells` array of a CellViT json.

    Uses ijson if available, else a byte-level scanner that tracks string /
    escape state and brace depth. CellViT cell objects contain no nested
    objects, only arrays of numbers, so depth tracking is enough.
    """
    try:
        import ijson
        with open(path, "rb") as fh:
            yield from ijson.items(fh, "cells.item")
        return
    except ImportError:
        pass

    dec = codecs.getincrementaldecoder("utf-8")()
    with open(path, "rb") as fh:
        # locate  "cells" : [
        buf = ""
        while '"cells"' not in buf:
            chunk = fh.read(1 << 20)
            if not chunk:
                return
            buf += dec.decode(chunk)
        buf = buf.split('"cells"', 1)[1]
        buf = buf.split("[", 1)[1] if "[" in buf else ""

        depth = 0
        in_str = False
        esc = False
        cur = []
        capturing = False
        while True:
            for ch in buf:
                if capturing:
                    cur.append(ch)
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    if depth == 0:
                        capturing = True
                        cur = ["{"]
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and capturing:
                        yield json.loads("".join(cur))
                        capturing = False
                        cur = []
                elif ch == "]" and depth == 0 and not capturing:
                    return
            chunk = fh.read(1 << 20)
            if not chunk:
                return
            buf = dec.decode(chunk)


def iter_detection_cells(path: Path):
    """cell_detection.json fast path: bbox + centroid + type, no contour."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    for c in obj.get("cells", []):
        yield c


def _moment_ellipse(pts):
    """Eccentricity + minor/major from the polygon's central second moments.
    pts: (n,2) float. Fallback when cv2.fitEllipse is unusable (<5 pts)."""
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return 0.0, 1.0
    cx = ((x + x1) * cross).sum() / (6 * a)
    cy = ((y + y1) * cross).sum() / (6 * a)
    mxx = ((x * x + x * x1 + x1 * x1) * cross).sum() / (12 * a) - cx * cx
    myy = ((y * y + y * y1 + y1 * y1) * cross).sum() / (12 * a) - cy * cy
    mxy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cross).sum() / (24 * a) - cx * cy
    tr, det = mxx + myy, mxx * myy - mxy * mxy
    disc = max(tr * tr / 4 - det, 0.0)
    l1, l2 = tr / 2 + math.sqrt(disc), tr / 2 - math.sqrt(disc)
    if l1 <= 0:
        return 0.0, 1.0
    ratio = math.sqrt(max(l2, 0.0) / l1)
    ecc = math.sqrt(max(1 - l2 / l1, 0.0)) if l1 > 0 else 0.0
    return ecc, ratio


def shape_metrics(contour):
    """contour: list of [x, y]. Returns dict of scalar shape descriptors in px."""
    pts = np.asarray(contour, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 3:
        return None
    if _HAVE_CV2:
        cnt = pts.astype(np.float32).reshape(-1, 1, 2)
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            return None
        perim = float(cv2.arcLength(cnt, True))
        hull = cv2.convexHull(cnt)
        harea = float(cv2.contourArea(hull))
        solidity = area / harea if harea > 0 else 1.0
        if len(pts) >= 5:
            try:
                (_, _), (ax1, ax2), _ = cv2.fitEllipse(cnt)
                mn, mx = sorted((ax1, ax2))
                ratio = mn / mx if mx > 0 else 1.0
                ecc = math.sqrt(max(1 - ratio * ratio, 0.0))
            except cv2.error:
                ecc, ratio = _moment_ellipse(pts)
        else:
            ecc, ratio = _moment_ellipse(pts)
    else:
        x, y = pts[:, 0], pts[:, 1]
        area = abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0
        if area <= 0:
            return None
        perim = float(np.hypot(*(pts - np.roll(pts, -1, axis=0)).T).sum())
        solidity = 1.0  # no hull without cv2/scipy; flagged by nuc_have_cv2=0
        ecc, ratio = _moment_ellipse(pts)
    circ = 4 * math.pi * area / (perim * perim) if perim > 0 else 0.0
    return {"area_px": area, "circ": circ, "solidity": min(solidity, 1.0),
            "ecc": ecc, "axisratio": ratio}


FEATURE_NAMES = [
    # counts / density
    "nuc_count", "nuc_per_mm2",
    "neo_count", "inflam_count", "connective_count", "epithelial_count", "dead_count",
    "neo_frac", "inflam_frac", "connective_frac", "epithelial_frac", "dead_frac",
    # neoplastic nuclear morphology  (the HGSC/EC/LGSC grade signal)
    "neo_area_um2_mean", "neo_area_um2_std", "neo_area_um2_cv",
    "neo_area_um2_p10", "neo_area_um2_p90", "neo_area_p90_p10_ratio",
    "neo_equivdiam_um_mean",
    "neo_ecc_mean", "neo_ecc_std",
    "neo_axisratio_mean", "neo_circularity_mean",
    "neo_solidity_mean", "neo_solidity_std",
    # all-nuclei morphology (fallback where neoplastic are sparse)
    "all_area_um2_mean", "all_area_um2_cv", "all_ecc_mean", "all_solidity_mean",
    # spatial / immune
    "nnd_all_um_mean", "nnd_all_um_cv", "nnd_neo_um_mean",
    "til_inflam_nn_neo_um_mean", "til_frac_neo_infiltrated", "til_inflam_per_neo",
    # presence flags -- NOT to be z-scored; consumed like channel_valid
    "morph_measured", "neo_measured", "til_measured", "nuc_have_cv2",
]
LOG1P_COLS = {"nuc_count", "nuc_per_mm2", "neo_count", "inflam_count",
              "connective_count", "epithelial_count", "dead_count",
              "neo_area_p90_p10_ratio", "til_inflam_per_neo"}
FLAG_COLS = {"morph_measured", "neo_measured", "til_measured", "nuc_have_cv2"}


def _nn_mean(xy):
    """mean and CV of nearest-neighbour distance within a point set."""
    if len(xy) < 2:
        return np.nan, np.nan
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)
    nn = d[:, 1]
    return float(nn.mean()), float(nn.std() / nn.mean()) if nn.mean() > 0 else np.nan


def aggregate_tile(idx_cells, cx, cy, ctype, area_um2, ecc, axisr, circ, sol,
                   mpp, tile_area_mm2, slide_has_inflam):
    """Build one FEATURE_NAMES row for the cells whose indices are idx_cells."""
    f = {k: 0.0 for k in FEATURE_NAMES}
    f["nuc_have_cv2"] = 1.0 if _HAVE_CV2 else 0.0
    n = len(idx_cells)
    f["nuc_count"] = float(n)
    f["nuc_per_mm2"] = n / tile_area_mm2 if tile_area_mm2 > 0 else 0.0
    if n == 0:
        return f
    t = ctype[idx_cells]
    for code, nm in TYPE_MAP.items():
        key = {"neoplastic": "neo", "inflammatory": "inflam"}.get(nm, nm)
        c = int((t == code).sum())
        f[f"{key}_count"] = float(c)
        f[f"{key}_frac"] = c / n

    neo = idx_cells[t == NAME2CODE["neoplastic"]]
    inflam = idx_cells[t == NAME2CODE["inflammatory"]]

    # morphometry uses only cells with a valid (non-artefact) area
    good = idx_cells[np.isfinite(area_um2[idx_cells])
                     & (area_um2[idx_cells] >= AREA_UM2_MIN)
                     & (area_um2[idx_cells] <= AREA_UM2_MAX)]
    if len(good) >= MIN_NUC:
        f["morph_measured"] = 1.0
        f["all_area_um2_mean"] = float(np.mean(area_um2[good]))
        m = np.mean(area_um2[good])
        f["all_area_um2_cv"] = float(np.std(area_um2[good]) / m) if m > 0 else 0.0
        f["all_ecc_mean"] = float(np.mean(ecc[good]))
        f["all_solidity_mean"] = float(np.mean(sol[good]))

    neo_good = neo[np.isfinite(area_um2[neo])
                   & (area_um2[neo] >= AREA_UM2_MIN)
                   & (area_um2[neo] <= AREA_UM2_MAX)]
    if len(neo_good) >= MIN_NEO:
        f["neo_measured"] = 1.0
        a = area_um2[neo_good]
        am = float(np.mean(a))
        f["neo_area_um2_mean"] = am
        f["neo_area_um2_std"] = float(np.std(a))
        f["neo_area_um2_cv"] = float(np.std(a) / am) if am > 0 else 0.0
        p10, p90 = np.percentile(a, [10, 90])
        f["neo_area_um2_p10"] = float(p10)
        f["neo_area_um2_p90"] = float(p90)
        f["neo_area_p90_p10_ratio"] = float(p90 / p10) if p10 > 0 else 0.0
        f["neo_equivdiam_um_mean"] = float(np.mean(2 * np.sqrt(a / math.pi)))
        f["neo_ecc_mean"] = float(np.mean(ecc[neo_good]))
        f["neo_ecc_std"] = float(np.std(ecc[neo_good]))
        f["neo_axisratio_mean"] = float(np.mean(axisr[neo_good]))
        f["neo_circularity_mean"] = float(np.mean(circ[neo_good]))
        f["neo_solidity_mean"] = float(np.mean(sol[neo_good]))
        f["neo_solidity_std"] = float(np.std(sol[neo_good]))

    # spacing
    allxy = np.stack([cx[idx_cells], cy[idx_cells]], axis=1)
    nnm, nncv = _nn_mean(allxy)
    if np.isfinite(nnm):
        f["nnd_all_um_mean"] = nnm * mpp
        f["nnd_all_um_cv"] = nncv if np.isfinite(nncv) else 0.0
    if len(neo) >= 2:
        neoxy = np.stack([cx[neo], cy[neo]], axis=1)
        nm, _ = _nn_mean(neoxy)
        if np.isfinite(nm):
            f["nnd_neo_um_mean"] = nm * mpp

    # tumour-infiltrating lymphocytes
    if slide_has_inflam and len(neo) > 0:
        f["til_measured"] = 1.0
        f["til_inflam_per_neo"] = len(inflam) / len(neo)
        if len(inflam) > 0:
            from scipy.spatial import cKDTree
            neoxy = np.stack([cx[neo], cy[neo]], axis=1)
            infxy = np.stack([cx[inflam], cy[inflam]], axis=1)
            tree = cKDTree(neoxy)
            d_inf, _ = tree.query(infxy, k=1)
            f["til_inflam_nn_neo_um_mean"] = float(np.mean(d_inf)) * mpp
            d_neo, _ = cKDTree(infxy).query(neoxy, k=1)
            f["til_frac_neo_infiltrated"] = float(
                np.mean(d_neo * mpp <= INFILTRATE_UM))
    return f


def load_is_tma():
    """image_id -> bool is_tma, from train.csv + test.csv."""
    import csv
    out = {}
    for name in ("train.csv", "test.csv"):
        f = ROOT / name
        if not f.exists():
            continue
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                if "is_tma" in row:
                    out[str(row["image_id"])] = str(row["is_tma"]).strip().lower() == "true"
    return out


def load_graph_geometry(gpath: Path):
    with np.load(gpath, allow_pickle=True) as z:
        prov = json.loads(str(z["provenance_json"][0]))
        g = {
            "image_id": str(z["image_id"][0]),
            "label": str(z["label"][0]),
            "n_nodes": int(z["node_features"].shape[0]),
            "grid_rc": z["node_grid_rc"].astype(np.int64),
            "grid_shape": tuple(int(v) for v in z["node_grid_shape"]),
            "tile_px": float(np.asarray(z["tile_size_wsi_px"]).ravel()[0]),
            "wsi_shape": tuple(int(v) for v in z["wsi_shape"]),
            "mpp": float(prov.get("wsi_mpp", 0.25)),
            "feat_names": [str(v) for v in z["node_feature_names"]],
            "node_features": z["node_features"].astype(np.float32),
        }
    return g


def resolve_mpp(iid, g, mpp_cfg):
    if mpp_cfg["mode"] == "graph":
        return g["mpp"], "graph_constant"
    # A corpus manifest carrying a measured per-slide native_mpp outranks the
    # is_tma-keyed default: is_tma is a UBC-OCEAN column and says nothing about
    # a slide from another cohort, which may be 20x or 40x.
    nat = corpus_paths.native_mpp(iid)
    if nat is not None:
        return nat, "manifest:native_mpp"
    is_tma = mpp_cfg["is_tma_map"].get(iid)
    if is_tma is None:
        return g["mpp"], "graph_constant(no is_tma row)"
    return (mpp_cfg["tma"], "is_tma:TMA") if is_tma else (mpp_cfg["wsi"], "is_tma:WSI")


def process_slide(gpath: Path, source: str, mpp_cfg, check=False):
    g = load_graph_geometry(gpath)
    iid = g["image_id"]
    det = DET_DIR / iid
    src = det / source
    if not src.exists():
        return None, f"missing {source}"

    n_rows, n_cols = g["grid_shape"]
    node_of_grid = np.full((n_rows, n_cols), -1, dtype=np.int64)
    node_of_grid[g["grid_rc"][:, 0], g["grid_rc"][:, 1]] = np.arange(g["n_nodes"])
    tile_px = g["tile_px"]
    mpp, mpp_source = resolve_mpp(iid, g, mpp_cfg)

    cx, cy, ct, area_px, ecc, axisr, circ, sol = ([] for _ in range(8))
    it = iter_detection_cells(src) if source.startswith("cell_detection") \
        else iter_cells(src)
    n_seen = 0
    for c in it:
        n_seen += 1
        cen = c.get("centroid")
        if cen is None and "bbox" in c:
            (y0, x0), (y1, x1) = c["bbox"]
            cen = [(x0 + x1) / 2, (y0 + y1) / 2]
        if cen is None:
            continue
        tp = c.get("type_prob", 1.0)
        if tp is not None and tp < MIN_TYPE_PROB:
            continue
        typ = c.get("type")
        cont = c.get("contour")
        if cont is not None:
            sm = shape_metrics(cont)
        elif "bbox" in c:
            (y0, x0), (y1, x1) = c["bbox"]
            w, h = abs(x1 - x0), abs(y1 - y0)
            a = w * h * 0.785  # ellipse-in-box approx
            r = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
            sm = {"area_px": a, "circ": math.pi * r, "solidity": 1.0,
                  "ecc": math.sqrt(max(1 - r * r, 0.0)), "axisratio": r}
        else:
            sm = None
        cx.append(float(cen[0]))
        cy.append(float(cen[1]))
        ct.append(int(typ) if typ is not None else 0)
        if sm is None:
            area_px.append(np.nan); ecc.append(np.nan)
            axisr.append(np.nan); circ.append(np.nan); sol.append(np.nan)
        else:
            area_px.append(sm["area_px"]); ecc.append(sm["ecc"])
            axisr.append(sm["axisratio"]); circ.append(sm["circ"])
            sol.append(sm["solidity"])

    cx = np.asarray(cx); cy = np.asarray(cy); ct = np.asarray(ct, dtype=np.int16)
    area_px = np.asarray(area_px, dtype=np.float64)
    area_um2 = area_px * (mpp * mpp)
    ecc = np.asarray(ecc); axisr = np.asarray(axisr)
    circ = np.asarray(circ); sol = np.asarray(sol)
    n_cells = len(cx)
    if n_cells == 0:
        return None, "zero cells parsed"

    gc = np.clip((cx // tile_px).astype(np.int64), 0, n_cols - 1)
    gr = np.clip((cy // tile_px).astype(np.int64), 0, n_rows - 1)
    oob = int(((cx < 0) | (cy < 0) | (cx >= n_cols * tile_px)
               | (cy >= n_rows * tile_px)).sum())
    node_id = node_of_grid[gr, gc]
    on_node = node_id >= 0
    cov = float(on_node.mean())

    qc = dict(image_id=iid, label=g["label"], n_cells=n_cells,
              n_on_node=int(on_node.sum()), coverage=round(cov, 4),
              n_oob=oob, n_tiles=g["n_nodes"],
              n_empty_tiles=0, source=source,
              mpp=mpp, mpp_source=mpp_source)

    if check:
        # alignment check: per-tile nucleus count vs the graph's own density
        cnt = np.bincount(node_id[on_node], minlength=g["n_nodes"]).astype(float)
        dens_col = next((i for i, nm in enumerate(g["feat_names"])
                         if nm == "dens_all_mean"), None)
        rho = np.nan
        if dens_col is not None and cnt.std() > 0:
            d = g["node_features"][:, dens_col]
            rs = lambda v: np.argsort(np.argsort(v))
            rho = float(np.corrcoef(rs(cnt), rs(d))[0, 1])
        qc["spearman_count_vs_density"] = round(rho, 3)
        med_area = float(np.nanmedian(area_um2)) if np.isfinite(area_um2).any() else np.nan
        qc["median_nuc_area_um2"] = round(med_area, 1)
        return None, qc

    slide_has_inflam = bool((ct == NAME2CODE["inflammatory"]).sum() > 0)
    tile_area_mm2 = (tile_px * mpp / 1000.0) ** 2

    order = np.argsort(node_id[on_node], kind="stable")
    ni = node_id[on_node][order]
    src_idx = np.flatnonzero(on_node)[order]
    bounds = np.searchsorted(ni, np.arange(g["n_nodes"] + 1))

    feats = np.zeros((g["n_nodes"], len(FEATURE_NAMES)), dtype=np.float32)
    n_empty = 0
    for node in range(g["n_nodes"]):
        cell_idx = src_idx[bounds[node]:bounds[node + 1]]
        if len(cell_idx) == 0:
            n_empty += 1
        row = aggregate_tile(cell_idx, cx, cy, ct, area_um2, ecc, axisr, circ,
                             sol, mpp, tile_area_mm2, slide_has_inflam)
        feats[node] = [row[k] for k in FEATURE_NAMES]
    qc["n_empty_tiles"] = n_empty

    payload = dict(
        image_id=np.array([iid]),
        label=np.array([g["label"]]),
        nuc_features=feats,
        nuc_feature_names=np.array(FEATURE_NAMES, dtype=object),
        nuc_log1p_cols=np.array([n in LOG1P_COLS for n in FEATURE_NAMES]),
        nuc_flag_cols=np.array([n in FLAG_COLS for n in FEATURE_NAMES]),
        grid_shape=np.array(g["grid_shape"]),
        grid_rc_checksum=np.array([int(g["grid_rc"].sum())]),
        mpp=np.array([mpp]),
        mpp_source=np.array([mpp_source]),
        coverage=np.array([cov]),
        n_cells=np.array([n_cells]),
        source=np.array([source]),
        params=np.array([json.dumps(dict(
            MIN_NUC=MIN_NUC, MIN_NEO=MIN_NEO, INFILTRATE_UM=INFILTRATE_UM,
            MIN_TYPE_PROB=MIN_TYPE_PROB,
            AREA_UM2_MIN=AREA_UM2_MIN, AREA_UM2_MAX=AREA_UM2_MAX))]),
    )
    return payload, qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify coordinate alignment on a few slides, write nothing")
    ap.add_argument("--smoke", action="store_true", help="3 slides, full pipeline")
    ap.add_argument("--only", type=str, default=None, help="comma-separated ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", default="cells.json",
                    choices=["cells.json", "cell_detection.json"],
                    help="cell_detection.json = fast, bbox-only, no real contours")
    ap.add_argument("--rebuild", action="store_true", help="ignore existing outputs")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--mpp-wsi", type=float, default=DEFAULT_MPP_WSI,
                    help="um/px for is_tma=False slides (default 0.50, 20x)")
    ap.add_argument("--mpp-tma", type=float, default=DEFAULT_MPP_TMA,
                    help="um/px for is_tma=True slides (default 0.25, 40x)")
    ap.add_argument("--mpp-from-graph", action="store_true",
                    help="ignore is_tma, use the graph's stored wsi_mpp (0.25) "
                         "for every slide -- keeps units consistent with the "
                         "graph but makes every nuclear area ~4x too small")
    args = ap.parse_args()

    mpp_cfg = {"mode": "graph" if args.mpp_from_graph else "is_tma",
               "wsi": args.mpp_wsi, "tma": args.mpp_tma,
               "is_tma_map": {} if args.mpp_from_graph else load_is_tma()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = sorted(GRAPH_DIR.glob("*_graph.npz"))
    if not graphs:
        sys.exit(f"no *_graph.npz in {GRAPH_DIR}")
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        graphs = [g for g in graphs if corpus_paths.image_id_from_graph(g) in want]
    if args.check:
        graphs = graphs[:: max(1, len(graphs) // 6)][:6]
    elif args.smoke:
        graphs = graphs[:3]
    if args.limit:
        graphs = graphs[: args.limit]

    print(f"python      : {sys.version.split()[0]}")
    print(f"cv2         : {'yes' if _HAVE_CV2 else 'NO -- solidity/hull disabled'}")
    try:
        import ijson  # noqa
        print("json stream : ijson")
    except ImportError:
        print("json stream : builtin scanner (pip install ijson for ~2x)")
    print(f"source      : {args.source}")
    if mpp_cfg["mode"] == "graph":
        print("mpp         : graph constant (0.25) for all slides "
              "[--mpp-from-graph]")
    else:
        print(f"mpp         : is_tma-keyed  WSI={mpp_cfg['wsi']}  "
              f"TMA={mpp_cfg['tma']}  ({len(mpp_cfg['is_tma_map'])} slides mapped)")
    print(corpus_paths.banner("nuclear"))
    print(f"slides      : {len(graphs)}   out -> {out_dir}\n")

    manifest = []
    t0 = time.time()
    for i, gp in enumerate(graphs, 1):
        iid = corpus_paths.image_id_from_graph(gp)
        op = out_dir / f"{iid}_nuclei.npz"
        if op.exists() and not args.rebuild and not args.check:
            continue
        ts = time.time()
        try:
            payload, qc = process_slide(gp, args.source, mpp_cfg, check=args.check)
        except Exception as e:
            import traceback
            print(f"[{i}/{len(graphs)}] {iid}  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            manifest.append(dict(image_id=iid, error=f"{type(e).__name__}: {e}"[:200]))
            continue
        if isinstance(qc, str):
            print(f"[{i}/{len(graphs)}] {iid}  skipped: {qc}")
            manifest.append(dict(image_id=iid, error=qc))
            continue
        manifest.append(qc)
        dt = time.time() - ts
        if args.check:
            print(f"[{i}/{len(graphs)}] {iid} {qc['label']:4s}  "
                  f"cells={qc['n_cells']:>7}  coverage={qc['coverage']:.3f}  "
                  f"oob={qc['n_oob']:>5}  "
                  f"rho(count,density)={qc.get('spearman_count_vs_density')}  "
                  f"mpp={qc['mpp']}({qc['mpp_source']})  "
                  f"med_area_um2={qc.get('median_nuc_area_um2')}  ({dt:.0f}s)")
        else:
            np.savez_compressed(op, **payload)
            print(f"[{i}/{len(graphs)}] {iid} {qc['label']:4s}  "
                  f"cells={qc['n_cells']:>7}  coverage={qc['coverage']:.3f}  "
                  f"empty_tiles={qc['n_empty_tiles']}/{qc['n_tiles']}  "
                  f"({dt:.0f}s, elapsed {(time.time()-t0)/60:.0f}m)")

    if manifest:
        import pandas as pd
        mf = out_dir / ("_check.csv" if args.check else "_manifest.csv")
        df = pd.DataFrame(manifest)
        if mf.exists() and not args.check:
            old = pd.read_csv(mf)
            df = pd.concat([old[~old.image_id.isin(df.image_id)], df],
                           ignore_index=True)
        df.to_csv(mf, index=False)
        print(f"\nmanifest -> {mf}")
        if args.check and "coverage" in df:
            print("\nALIGNMENT VERDICT")
            cov = df["coverage"].astype(float)
            rho = df.get("spearman_count_vs_density")
            print(f"  coverage: min {cov.min():.3f}  median {cov.median():.3f}")
            if rho is not None:
                rho = rho.astype(float)
                print(f"  rho(nucleus count, graph density): "
                      f"min {rho.min():.2f}  median {rho.median():.2f}")
            if cov.median() > 0.85 and (rho is None or rho.median() > 0.5):
                print("  -> PASS. Coordinate spaces agree. Safe to run the full pass.")
            else:
                print("  -> FAIL. Cells are not landing on the tissue tiles, or")
                print("     per-tile counts do not track the graph's own density.")
                print("     Check for a WSI transpose or the PNG->TIFF dim fix")
                print("     before running the full extraction.")


if __name__ == "__main__":
    main()


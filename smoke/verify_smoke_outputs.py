# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Check the files a smoke run produced against smoke/slides.json.

Usage:
    python smoke/verify_smoke_outputs.py [--root UBC_ROOT] [--slides smoke/slides.json]
                                         [--models phikonv2 ...] [--json OUT]

Reads, under <root>/cellvit_run/: wsi_pyramid_tiff/<id>.tiff, graphs/<id>_graph.npz,
graphs/nuclear_feats/<id>_nuclei.npz (and _check.csv from the --check run),
graphs/chromatin_feats/<id>_chromatin.npz, graphs/eosin_feats/<id>_eosin.npz and
graphs/fm_feats/<id>_<model>.npz, plus <root>/train.csv.

Every check gets one of PASS, FAIL, WARN or N/A. Exact and tolerance checks
FAIL; sanity checks, which have no logged oracle, WARN. A tolerance check is N/A
when the slide has no logged value. The exit status is 1 if anything FAILs.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

BLOCKS = {
    "nuclear": ("nuclear_feats", "{iid}_nuclei.npz", "nuc_features", "nuc_feature_names"),
    "chromatin": ("chromatin_feats", "{iid}_chromatin.npz", "chrom_features", "chrom_feature_names"),
    "eosin": ("eosin_feats", "{iid}_eosin.npz", "eos_features", "eos_feature_names"),
}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, slide, check, expected, observed, tol, status, note=""):
        self.rows.append(dict(slide=slide, check=check, expected=_fmt(expected),
                              observed=_fmt(observed), tolerance=tol, status=status, note=note))

    def exact(self, slide, check, expected, observed, note=""):
        self.add(slide, check, expected, observed, "exact",
                 "PASS" if _same(expected, observed) else "FAIL", note)

    def fail(self, slide, check, note):
        self.add(slide, check, "", "", "", "FAIL", note)

    @property
    def n_fail(self):
        return sum(r["status"] == "FAIL" for r in self.rows)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return "" if v is None else str(v)


def _same(a, b):
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(map(int, a)) == list(map(int, b))
    return a == b


def rank_rho(a, b):
    """Rank correlation as computed by extract_nuclear_features.py --check and
    check_nuc.py: Pearson correlation of argsort ranks, ties broken by order."""
    rs = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rs(a), rs(b))[0, 1])


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def scalar(arr):
    return np.asarray(arr).ravel()[0]


def tiff_shape(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        shp = t.pages[0].shape
    return [int(shp[0]), int(shp[1])]


def within_rel(expected, observed, rel):
    return abs(observed - expected) <= rel * abs(expected)


def check_slide(rep, spec, cfg, root, train, models):
    iid, label = spec["image_id"], spec["label"]
    run = root / "cellvit_run"
    gdir = run / "graphs"
    tol = cfg["tolerances"]

    row = train.get(iid)
    if row is None:
        rep.fail(iid, "train.csv row", "image_id not in train.csv")
        return
    rep.exact(iid, "train.csv label", label, row["label"])
    rep.exact(iid, "train.csv is_tma", "False", row.get("is_tma", ""))
    hw = [int(row["image_height"]), int(row["image_width"])]

    gp = gdir / f"{iid}_graph.npz"
    if not gp.exists():
        rep.fail(iid, "graph file", f"missing {gp}")
        return
    g = load_npz(gp)
    nf, ea = g["node_features"], g["edge_attr"]
    n_nodes = nf.shape[0]
    rc_sum = int(np.asarray(g["node_grid_rc"]).sum())
    nnames = [str(x) for x in g["node_feature_names"]]
    enames = [str(x) for x in g["edge_attr_names"]]
    w = cfg["widths"]
    rep.exact(iid, "graph node_features width", w["graph_node"], nf.shape[1])
    rep.exact(iid, "graph edge_attr width", w["graph_edge"], ea.shape[1])
    rep.exact(iid, "node_feature_names length", nf.shape[1], len(nnames))
    rep.exact(iid, "edge_attr_names length", ea.shape[1], len(enames))
    rep.exact(iid, "node_feature_names unique", len(nnames), len(set(nnames)))
    rep.exact(iid, "edge_attr_names unique", len(enames), len(set(enames)))
    tile = int(scalar(g["tile_size_wsi_px"]))
    rep.exact(iid, "tile_size_wsi_px", cfg["tile_size_wsi_px"], tile)
    wsi = [int(v) for v in np.asarray(g["wsi_shape"]).ravel()]
    rep.exact(iid, "wsi_shape == train.csv (height, width)", hw, wsi)
    grid = [int(v) for v in np.asarray(g["node_grid_shape"]).ravel()]
    rep.exact(iid, "node_grid_shape == ceil(wsi_shape / tile)",
              [math.ceil(wsi[0] / tile), math.ceil(wsi[1] / tile)], grid)
    tp = run / "wsi_pyramid_tiff" / f"{iid}.tiff"
    if tp.exists():
        rep.exact(iid, "wsi_shape == TIFF level 0", tiff_shape(tp), wsi)
    else:
        rep.fail(iid, "TIFF file", f"missing {tp}")
    rep.exact(iid, "graph label == train.csv", row["label"], str(scalar(g["label"])))
    rep.exact(iid, "graph node_features finite", True, bool(np.isfinite(nf).all()))
    rep.exact(iid, "graph edge_attr finite", True, bool(np.isfinite(ea).all()))

    blocks = {}
    for name, (sub, pattern, fkey, nkey) in BLOCKS.items():
        p = gdir / sub / pattern.format(iid=iid)
        if not p.exists():
            rep.fail(iid, f"{name} file", f"missing {p}")
            continue
        z = load_npz(p)
        F = z[fkey]
        names = [str(x) for x in z[nkey]]
        blocks[name] = (z, F, names)
        rep.exact(iid, f"{name} rows == graph nodes", n_nodes, F.shape[0])
        rep.exact(iid, f"{name} width", w[name], F.shape[1])
        rep.exact(iid, f"{name} width == len(names)", F.shape[1], len(names))
        rep.exact(iid, f"{name} grid_rc_checksum == graph", rc_sum, int(scalar(z["grid_rc_checksum"])))
        rep.exact(iid, f"{name} finite", True, bool(np.isfinite(F).all()))
        rep.exact(iid, f"{name} dtype", "float32", str(F.dtype))
        if "label" in z:
            rep.exact(iid, f"{name} label == train.csv", row["label"], str(scalar(z["label"])))

    for m in models:
        p = gdir / "fm_feats" / f"{iid}_{m}.npz"
        if not p.exists():
            rep.fail(iid, f"fm {m} file", f"missing {p}")
            continue
        z = load_npz(p)
        E = z["fm_features"]
        rep.exact(iid, f"fm {m} rows == graph nodes", n_nodes, E.shape[0])
        rep.exact(iid, f"fm {m} width", w["fm"][m], E.shape[1])
        rep.exact(iid, f"fm {m} grid_rc_checksum == graph", rc_sum, int(scalar(z["grid_rc_checksum"])))
        rep.exact(iid, f"fm {m} finite", True, bool(np.isfinite(E).all()))
        rep.exact(iid, f"fm {m} dtype", "float32", str(E.dtype))
        if "label" in z:
            rep.exact(iid, f"fm {m} label == train.csv", row["label"], str(scalar(z["label"])))

    ex = cfg.get("exact", {}).get(iid)
    if ex:
        rep.exact(iid, "oracle node_features shape", ex["node_features"], list(nf.shape))
        rep.exact(iid, "oracle edge_attr shape", ex["edge_attr"], list(ea.shape))
        rep.exact(iid, "oracle edge_index shape", ex["edge_index"], list(np.asarray(g["edge_index"]).shape))
        rep.exact(iid, "oracle node_grid_shape", ex["node_grid_shape"], grid)
        rep.exact(iid, "oracle wsi_shape", ex["wsi_shape"], wsi)
        rep.exact(iid, "oracle tile_size_wsi_px", ex["tile_size_wsi_px"], tile)
        rep.exact(iid, "oracle channel_names count", ex["n_channel_names"], len(g["channel_names"]))
        if "nuclear" in blocks:
            rep.exact(iid, "oracle nuclear shape", ex["nuclear"], list(blocks["nuclear"][1].shape))

    # tolerance checks against the logged values
    nuc = blocks.get("nuclear")
    if spec.get("n_tiles") is not None:
        rep.add(iid, "n_tiles", spec["n_tiles"], n_nodes, f"+/-{tol['n_tiles_rel']:.0%}",
                "PASS" if within_rel(spec["n_tiles"], n_nodes, tol["n_tiles_rel"]) else "FAIL")
    else:
        rep.add(iid, "n_tiles", None, n_nodes, "", "N/A", "no logged value")
    if nuc is None:
        return
    z, F, names = nuc
    n_cells = int(scalar(z["n_cells"]))
    cov = float(scalar(z["coverage"]))
    for key, obs, kind, t in (("n_cells", n_cells, "rel", tol["n_cells_rel"]),
                              ("coverage", cov, "abs", tol["coverage_abs"])):
        if spec.get(key) is None:
            rep.add(iid, key, None, obs, "", "N/A", "no logged value")
            continue
        ok = within_rel(spec[key], obs, t) if kind == "rel" else abs(obs - spec[key]) <= t
        rep.add(iid, key, spec[key], obs, f"+/-{t:.0%}" if kind == "rel" else f"+/-{t}",
                "PASS" if ok else "FAIL")

    cnt = F[:, names.index("nuc_count")]
    dens = nf[:, nnames.index("dens_all_mean")]
    rho_npz = rank_rho(cnt, dens)
    perm = np.random.default_rng(0).permutation(len(cnt))
    rho_sh = rank_rho(cnt[perm], dens)
    check_csv = {}
    cp = gdir / "nuclear_feats" / "_check.csv"
    if cp.exists():
        with open(cp, newline="") as fh:
            check_csv = {r["image_id"]: r for r in csv.DictReader(fh)}
    crow = check_csv.get(iid)

    definition = spec.get("spearman_definition")
    if definition == "check_csv":
        obs = float(crow["spearman_count_vs_density"]) if crow else None
        how = "from nuclear_feats/_check.csv"
    else:
        obs, how = rho_npz, "nuc_count vs dens_all_mean from the npz files"
    if spec.get("spearman_count_density") is None:
        rep.add(iid, "spearman(count, density)", None, rho_npz, "", "N/A",
                "no logged value; observed from the npz files")
    elif obs is None:
        rep.fail(iid, "spearman(count, density)", "no _check.csv row; run extract_nuclear_features.py --check")
    else:
        ok = abs(obs - spec["spearman_count_density"]) <= tol["spearman_abs"] and obs > tol["spearman_min"]
        rep.add(iid, "spearman(count, density)", spec["spearman_count_density"], obs,
                f"+/-{tol['spearman_abs']} and >{tol['spearman_min']}", "PASS" if ok else "FAIL", how)
    rep.add(iid, "shuffled-control spearman", spec.get("shuffled_control") or 0.0, rho_sh,
            f"|rho|<={tol['shuffled_abs_max']}",
            "PASS" if abs(rho_sh) <= tol["shuffled_abs_max"] else "FAIL")

    if spec.get("median_nuc_area_um2") is None:
        rep.add(iid, "median nucleus area um2", None,
                float(crow["median_nuc_area_um2"]) if crow else None, "", "N/A", "no logged value")
    elif crow is None:
        rep.fail(iid, "median nucleus area um2", "no _check.csv row; run extract_nuclear_features.py --check")
    else:
        obs = float(crow["median_nuc_area_um2"])
        rep.add(iid, "median nucleus area um2", spec["median_nuc_area_um2"], obs,
                f"+/-{tol['median_area_rel']:.0%}",
                "PASS" if within_rel(spec["median_nuc_area_um2"], obs, tol["median_area_rel"]) else "FAIL")

    # sanity checks: no oracle, so they warn
    rep.add(iid, "coverage above minimum", f">{tol['coverage_min']}", cov, "sanity",
            "PASS" if cov > tol["coverage_min"] else "WARN")
    cells = grid[0] * grid[1]
    rep.add(iid, "tiles within grid", f"1..{cells}", n_nodes, "sanity",
            "PASS" if 0 < n_nodes <= cells else "WARN")
    known = set(cfg.get("known_constant_columns", []))
    for name, (_, B, bnames) in blocks.items():
        const = [n for i, n in enumerate(bnames) if np.ptp(B[:, i]) == 0 and n not in known]
        rep.add(iid, f"{name} unexpected constant columns", "none", len(const), "sanity",
                "PASS" if not const else "WARN", ", ".join(const[:8]))


def read_train(path):
    with open(path, newline="") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.environ.get("UBC_ROOT"),
                    help="dataset root used for the smoke run (default: $UBC_ROOT)")
    ap.add_argument("--slides", default=str(HERE / "slides.json"))
    ap.add_argument("--models", nargs="+", default=["phikonv2"])
    ap.add_argument("--json", help="also write the table here")
    args = ap.parse_args(argv)
    if not args.root:
        ap.error("--root or UBC_ROOT is required")

    root = Path(args.root)
    cfg = json.loads(Path(args.slides).read_text())
    rep = Report()
    train_path = root / "train.csv"
    if not train_path.exists():
        rep.fail("-", "train.csv", f"missing {train_path}")
    else:
        train = read_train(train_path)
        for spec in cfg["slides"]:
            try:
                check_slide(rep, spec, cfg, root, train, args.models)
            except Exception as e:  # a crash on one slide is a failure, not an abort
                rep.fail(spec["image_id"], "checker", f"{type(e).__name__}: {e}")

    cols = ("slide", "check", "expected", "observed", "tolerance", "status", "note")
    widths = {c: max(len(c), *(len(str(r[c])) for r in rep.rows)) if rep.rows else len(c) for c in cols}
    widths["note"] = len("note")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rep.rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    counts = {s: sum(r["status"] == s for r in rep.rows) for s in ("PASS", "FAIL", "WARN", "N/A")}
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(dict(counts=counts, rows=rep.rows), indent=1))
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

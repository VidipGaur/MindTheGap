# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Build multi-channel KDE nuclear density maps from CellViT outputs (UBC-OCEAN).

Stage 2 of the pipeline:
    CellViT -> [THIS SCRIPT] -> build_streamline_graphs.py -> GATv2

One density field per PanNuke cell type, plus a combined `all` channel, on a
DOWNSAMPLE-reduced grid, written to kde_maps/<image_id>_kde.npz.

Run:
    python build_kde_density_maps.py              # incremental; rebuilds stale schemas
    python build_kde_density_maps.py --force      # rebuild everything
    python build_kde_density_maps.py --only 4     # single slide, verbose
    python build_kde_density_maps.py --limit 10   # smoke run


WHAT CHANGED FROM THE VERSION THAT RAN OVER THE WEEKEND
=======================================================

(1) SIGMA. The previous file computed KDE_SIGMA_UM = 15 * 0.25 = 3.75 um, on the
    reasoning that the old raw constant KDE_SIGMA=15 was in WSI pixels. It was
    not -- it was in DOWNSAMPLED GRID pixels, and one grid pixel is
    mpp * DOWNSAMPLE = 0.25 * 16 = 4 um. So the physical equivalent is 60 um,
    and the old value produced sigma_px = 0.94, i.e. effectively no smoothing at
    all: a binned point cloud rather than a density field. Fixed to 60.0 um,
    which reproduces the previously-eyeballed bandwidth exactly at the current
    DOWNSAMPLE while staying correct if DOWNSAMPLE or mpp ever change.
    Still not a validated constant -- see visualize_kde_maps.py.

(2) STALE FILES ARE REBUILT, NOT SKIPPED. The previous version treated any
    existing <id>_kde.npz as done. With 456 two-channel files already on disk
    that made the whole run a no-op. Every file now carries SCHEMA_VERSION and
    its channel list; a file that does not match is rebuilt automatically. No
    manual deletion step, and no way to silently half-upgrade the corpus.

(3) OUT-OF-BOUNDS CENTROIDS ARE COUNTED, NOT SILENTLY CLIPPED. Clipping
    centroids to slide bounds is exactly what let the original row/col axis bug
    hide: swapped axes on a wide slide pile cells onto the last row and the
    clip absorbs it without a word. Every slide now records what fraction of
    centroids landed outside bounds and what fraction of each axis the centroid
    cloud actually spans, warns past a threshold, and writes both into the npz
    and the CSV log. This is a permanent regression detector for that bug class,
    not a one-off diagnostic.

(4) GEOJSON FALLBACK NO LONGER COLLAPSES CELLS. The previous fallback averaged
    every polygon in a MultiPolygon into ONE centroid -- a slide parsed that way
    would report ~5 cells instead of ~500,000 and produce an almost empty
    density map without failing. Each polygon is now its own cell. The fallback
    also warns loudly, because cells.json (which carries type_map) is the file
    we actually want.

(5) WRITES ARE VERIFIED. Each output is reopened and checked for its channels
    and shapes before being counted as complete, and a failed slide's partial
    file is deleted rather than left on disk to be read as "cached" next run.
    Same lesson as the CellViT run that logged [OK] for a crashed process.


COORDINATES -- STATED ONCE
==========================
CellViT writes centroid as [x, y] = [col, row]. It is swapped to (row, col)
exactly once, in load_centroids_by_type, and nothing downstream re-derives it.
Density arrays are (row, col) = (height-axis, width-axis), shape
(ceil(H/DOWNSAMPLE), ceil(W/DOWNSAMPLE)).

grid_x[j] / grid_y[i] give the WSI-pixel centre of grid cell [i, j]; that is the
canonical mapping for the attention overlay. wsi_shape is also written as (H, W)
for convenience. Both are saved so no downstream stage has to redo the math.
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
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Resolved through corpus_paths so a second corpus (different run dir, different
# slide-folder naming) runs this stage unchanged. Defaults are the UBC-OCEAN
# values this script was written against.
import corpus_paths

ROOT = str(corpus_paths.ROOT)
RUN_DIR_NAME = corpus_paths.RUN_DIR_NAME
MANIFEST_FILENAME = "cellvit_manifest.csv"
DETECTIONS_SUBFOLDER = "detections"
OUT_SUBFOLDER = "kde_maps"

# Order in which a slide's detection file is looked for. cells.json is preferred
# on UBC-OCEAN because it carries type_map. On corpora where cell_detection.json
# has been VERIFIED to carry an identical cell list (same centroids, same types)
# it is the better read: same numbers, ~8x less I/O and RAM. Override with
# --detection-file; the KDE only ever uses centroid + type, never the contour.
DETECTION_PREFERENCE = ["cells.json", "cell_detection.json",
                        "cell_detection.geojson", "cells.geojson"]

# Bump this whenever the saved schema changes; existing files that don't match
# are rebuilt automatically instead of being skipped as "cached".
SCHEMA_VERSION = 3

DOWNSAMPLE = 16

# Physical smoothing bandwidth. 60 um == the old KDE_SIGMA=15 grid px at
# DOWNSAMPLE=16 and 0.25 um/px (15 * 0.25 * 16 = 60). Roughly 6 nuclear
# diameters: tissue-architecture scale, which is what the structure tensor
# downstream wants. STARTING POINT, not validated -- spot-check across a
# low/mid/high cell-count spread before trusting it for RK4.
KDE_SIGMA_UM = 60.0

MPP_COLUMN_CANDIDATES = ["mpp", "pixel_size_um", "resolution_um", "wsi_mpp"]
DEFAULT_PIXEL_SIZE_UM = 0.25   # matches --wsi_mpp used at CellViT inference

# Confirmed enum from the detection JSONs for this run.
CELL_TYPE_ENUM = {"1": "Neoplastic", "2": "Inflammatory", "3": "Connective",
                  "4": "Dead", "5": "Epithelial"}
CHANNEL_ORDER = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

# Dead is built and saved for completeness but should NOT be integrated into
# streamlines: CellViT-H scores PQ ~0.15 on it against ~0.58 for Neoplastic.
# The graph builder skips it by default.
NOT_FOR_INTEGRATION = ["Dead"]

GENERATE_DENSITY_ALL = True

# Channels below this are saved (schema stays identical across slides) but
# flagged sparse, so the graph builder can mask rather than integrate noise.
SPARSE_CELL_THRESHOLD = 50

# Warn if more than this fraction of centroids fell outside slide bounds.
# Anything above ~0 means a coordinate assumption is wrong somewhere.
OOB_WARN_FRACTION = 0.001

# Set True only if a coordinate-scale check shows centroids are patch-local.
ADD_OFFSET_GLOBAL = False


def setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("kde_build")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    fh = logging.FileHandler(out_dir / "kde_build.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def find_detection_file(image_id: str, detections_dir: Path):
    """cells.json first -- it carries type_map and per-cell type ids."""
    folder = detections_dir / str(image_id)
    if not folder.exists():
        return None
    for name in DETECTION_PREFERENCE:
        p = folder / name
        if p.exists():
            return p
    return None


def _geojson_type(feat: dict):
    props = feat.get("properties", {})
    cls = props.get("classification")
    if isinstance(cls, dict) and cls.get("name"):
        return cls["name"]
    for key in ("type", "cell_type", "class", "name"):
        if key in props:
            return props[key]
    return None


def load_centroids_by_type(detection_path: Path, wsi_width: int, wsi_height: int,
                           log: logging.Logger):
    """
    Returns (by_type, diag).

    by_type: {type_name: (N, 2) float32 array of (row, col)} in ORIGINAL,
             un-downsampled WSI pixel space, clipped to slide bounds.
    diag:    out-of-bounds counts and per-axis coverage, for the regression
             check described in the module docstring.
    """
    with open(detection_path, "r") as f:
        data = json.load(f)

    by_type = {}
    unknown_ids = set()
    used_fallback = False

    if isinstance(data, dict) and "cells" in data and "type_map" in data:
        file_map = {str(k): v for k, v in data["type_map"].items()}
        if file_map and file_map != CELL_TYPE_ENUM:
            log.warning(f"{detection_path.parent.name}: type_map {file_map} differs "
                        f"from expected {CELL_TYPE_ENUM} -- using the file's own map.")
        type_map = file_map or CELL_TYPE_ENUM

        cells = data["cells"]
        if not cells:
            raise ValueError(f"no cells in {detection_path}")

        for cell in cells:
            # CellViT centroid is [x, y] = [col, row]. Swapped here, once.
            x, y = cell["centroid"][0], cell["centroid"][1]
            row, col = y, x
            if ADD_OFFSET_GLOBAL:
                off = cell.get("offset_global", [0, 0])
                row, col = row + off[1], col + off[0]
            tid = str(cell.get("type"))
            name = type_map.get(tid)
            if name is None:
                name = f"unknown_{tid}"
                unknown_ids.add(tid)
            by_type.setdefault(name, []).append((row, col))

    else:
        used_fallback = True
        log.warning(f"{detection_path.name}: falling back to GeoJSON parsing "
                    f"(cells.json not found). Verify the cell count looks sane.")
        features = data if isinstance(data, list) else data.get("features", [])
        if not features:
            raise ValueError(f"no features in {detection_path}")

        for feat in features:
            geom = feat.get("geometry", {}) or {}
            gtype = geom.get("type", "")
            name = _geojson_type(feat) or "unknown_geojson"
            pts = []
            if gtype == "Point":
                x, y = geom["coordinates"][:2]
                pts.append((y, x))
            elif gtype == "MultiPoint":
                for x, y in geom["coordinates"]:
                    pts.append((y, x))
            elif gtype == "Polygon":
                c = np.asarray(geom["coordinates"][0], dtype=np.float64)
                pts.append((c[:, 1].mean(), c[:, 0].mean()))
            elif gtype == "MultiPolygon":
                # One centroid PER POLYGON. Averaging the whole MultiPolygon
                # into a single point silently deletes ~all cells on the slide.
                for poly in geom["coordinates"]:
                    c = np.asarray(poly[0], dtype=np.float64)
                    pts.append((c[:, 1].mean(), c[:, 0].mean()))
            by_type.setdefault(name, []).extend(pts)

    if unknown_ids:
        log.warning(f"{detection_path.parent.name}: unknown type id(s) "
                    f"{sorted(unknown_ids)} -- counted in density_all only.")

    n_total = n_oob = 0
    max_row = max_col = 0.0
    result = {}
    for name, pts in by_type.items():
        arr = np.asarray(pts, dtype=np.float32)
        if arr.size == 0:
            result[name] = np.empty((0, 2), np.float32)
            continue
        n_total += len(arr)
        n_oob += int(((arr[:, 0] < 0) | (arr[:, 0] > wsi_height - 1) |
                      (arr[:, 1] < 0) | (arr[:, 1] > wsi_width - 1)).sum())
        max_row = max(max_row, float(arr[:, 0].max()))
        max_col = max(max_col, float(arr[:, 1].max()))
        arr[:, 0] = np.clip(arr[:, 0], 0, wsi_height - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, wsi_width - 1)
        result[name] = arr

    diag = dict(
        n_total=n_total,
        frac_oob=(n_oob / n_total) if n_total else 0.0,
        coverage_row=(max_row / max(wsi_height - 1, 1)) if n_total else 0.0,
        coverage_col=(max_col / max(wsi_width - 1, 1)) if n_total else 0.0,
        used_geojson_fallback=used_fallback,
    )
    return result, diag


def resolve_pixel_size_um(row, warned: dict, log: logging.Logger) -> float:
    for col in MPP_COLUMN_CANDIDATES:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    if not warned.get("done"):
        log.warning(f"No mpp column in manifest (checked {MPP_COLUMN_CANDIDATES}) -- "
                    f"using DEFAULT_PIXEL_SIZE_UM={DEFAULT_PIXEL_SIZE_UM} for all slides. "
                    f"Verify against the --wsi_mpp used at CellViT inference.")
        warned["done"] = True
    return DEFAULT_PIXEL_SIZE_UM


def build_density_map(centroids, wsi_height, wsi_width, downsample, sigma_px):
    H_ds = int(np.ceil(wsi_height / downsample))
    W_ds = int(np.ceil(wsi_width / downsample))
    d = np.zeros((H_ds, W_ds), dtype=np.float32)
    if len(centroids) == 0:
        return d
    r = np.clip((centroids[:, 0] / downsample).astype(np.int32), 0, H_ds - 1)
    c = np.clip((centroids[:, 1] / downsample).astype(np.int32), 0, W_ds - 1)
    np.add.at(d, (r, c), 1.0)
    return gaussian_filter(d, sigma=sigma_px).astype(np.float32)


def build_grid_coordinates(wsi_height, wsi_width, downsample):
    """WSI-pixel centre of each grid cell. density[i, j] <-> (x=grid_x[j], y=grid_y[i])."""
    H_ds = int(np.ceil(wsi_height / downsample))
    W_ds = int(np.ceil(wsi_width / downsample))
    grid_x = np.clip((np.arange(W_ds, dtype=np.float64) + 0.5) * downsample, 0, wsi_width - 1)
    grid_y = np.clip((np.arange(H_ds, dtype=np.float64) + 0.5) * downsample, 0, wsi_height - 1)
    return grid_x, grid_y


def expected_channel_keys():
    keys = [f"density_{n.lower()}" for n in CHANNEL_ORDER]
    if GENERATE_DENSITY_ALL:
        keys.append("density_all")
    return keys


def output_is_current(path: Path) -> bool:
    """True only if the file exists AND matches the current schema and channels."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            if "schema_version" not in z.files:
                return False
            if int(z["schema_version"][0]) != SCHEMA_VERSION:
                return False
            return all(k in z.files for k in expected_channel_keys())
    except Exception:
        return False


def verify_written(path: Path, expected_shape):
    with np.load(path, allow_pickle=True) as z:
        for k in expected_channel_keys() + ["grid_x", "grid_y", "wsi_shape", "label"]:
            if k not in z.files:
                raise ValueError(f"written file missing {k}")
        for k in expected_channel_keys():
            if z[k].shape != expected_shape:
                raise ValueError(f"{k} shape {z[k].shape} != {expected_shape}")
        if len(z["grid_y"]) != expected_shape[0] or len(z["grid_x"]) != expected_shape[1]:
            raise ValueError("grid_x/grid_y length does not match density shape")


def process_one(image_id, label, wsi_width, wsi_height, pixel_size_um,
                detection_path, out_dir, log, force=False):
    out_path = out_dir / f"{image_id}_kde.npz"
    if not force and output_is_current(out_path):
        return "cached", None

    by_type, diag = load_centroids_by_type(detection_path, wsi_width, wsi_height, log)

    sigma_px = KDE_SIGMA_UM / (pixel_size_um * DOWNSAMPLE)
    H_ds = int(np.ceil(wsi_height / DOWNSAMPLE))
    W_ds = int(np.ceil(wsi_width / DOWNSAMPLE))

    if diag["frac_oob"] > OOB_WARN_FRACTION:
        log.warning(f"slide={image_id}: {diag['frac_oob']:.2%} of centroids fell OUTSIDE "
                    f"slide bounds before clipping. That is the signature of a coordinate "
                    f"convention mismatch -- do not build graphs on this until checked.")
    if diag["n_total"] and (diag["coverage_row"] < 0.5 or diag["coverage_col"] < 0.5):
        log.warning(f"slide={image_id}: centroids span only "
                    f"{diag['coverage_row']:.0%} of rows / {diag['coverage_col']:.0%} of cols. "
                    f"Expected near-full coverage unless the slide is mostly background.")

    density_maps, cell_counts, stats, sparse = {}, {}, {}, []

    for name in CHANNEL_ORDER:
        cen = by_type.get(name, np.empty((0, 2), np.float32))
        n = len(cen)
        cell_counts[name] = int(n)
        if n == 0:
            log.warning(f"slide={image_id}: no {name} cells -- channel is all zeros.")
            sparse.append(name)
        elif n < SPARSE_CELL_THRESHOLD:
            log.warning(f"slide={image_id}: sparse {name} channel (n={n}).")
            sparse.append(name)
        d = build_density_map(cen, wsi_height, wsi_width, DOWNSAMPLE, sigma_px)
        density_maps[name.lower()] = d
        stats[name] = {"n_cells": n, "max": float(d.max()), "mean": float(d.mean())}

    for t, arr in by_type.items():
        if t not in CHANNEL_ORDER:
            cell_counts[t] = int(len(arr))

    if GENERATE_DENSITY_ALL:
        arrs = [a for a in by_type.values() if len(a)]
        allc = np.concatenate(arrs, axis=0) if arrs else np.empty((0, 2), np.float32)
        d = build_density_map(allc, wsi_height, wsi_width, DOWNSAMPLE, sigma_px)
        density_maps["all"] = d
        stats["all"] = {"n_cells": len(allc), "max": float(d.max()), "mean": float(d.mean())}
        del allc

    grid_x, grid_y = build_grid_coordinates(wsi_height, wsi_width, DOWNSAMPLE)

    try:
        np.savez_compressed(
            out_path,
            schema_version=np.array([SCHEMA_VERSION]),
            image_id=np.array([str(image_id)]),
            label=np.array([str(label)]),
            downsample=np.array([DOWNSAMPLE]),
            sigma_um=np.array([KDE_SIGMA_UM]),
            sigma_px=np.array([sigma_px]),
            pixel_size_um=np.array([pixel_size_um]),
            wsi_shape=np.array([wsi_height, wsi_width]),
            wsi_height=np.array([wsi_height]),
            wsi_width=np.array([wsi_width]),
            x_min=np.array([0]), y_min=np.array([0]),
            x_max=np.array([wsi_width]), y_max=np.array([wsi_height]),
            grid_x=grid_x, grid_y=grid_y,
            channel_order=np.array(CHANNEL_ORDER, dtype=object),
            not_for_integration=np.array(NOT_FOR_INTEGRATION, dtype=object),
            cell_types_present=np.array(list(by_type.keys()), dtype=object),
            sparse_channels=np.array(sparse, dtype=object),
            cell_counts_json=np.array([json.dumps(cell_counts)]),
            coord_diag_json=np.array([json.dumps(diag)]),
            **{f"density_{k}": v for k, v in density_maps.items()},
        )
        verify_written(out_path, (H_ds, W_ds))
    except Exception:
        if out_path.exists():
            out_path.unlink()   # never leave a partial file to be read as cached
        raise

    stats["_diag"] = diag
    return "ok", stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild every slide")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--detection-file", default=None,
                    help="force one detection filename (e.g. cell_detection.json) "
                         "instead of the cells.json-first preference order")
    args = ap.parse_args()

    if args.detection_file:
        global DETECTION_PREFERENCE
        DETECTION_PREFERENCE = [args.detection_file]

    run_dir = Path(ROOT) / RUN_DIR_NAME
    out_dir = run_dir / OUT_SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir)

    log.info(corpus_paths.banner("kde"))
    log.info(f"detection file preference: {DETECTION_PREFERENCE}")
    log.info(f"schema v{SCHEMA_VERSION} | channels: {CHANNEL_ORDER}"
             f"{' + all' if GENERATE_DENSITY_ALL else ''}")
    log.info(f"DOWNSAMPLE={DOWNSAMPLE} KDE_SIGMA_UM={KDE_SIGMA_UM} "
             f"-> sigma_px={KDE_SIGMA_UM / (DEFAULT_PIXEL_SIZE_UM * DOWNSAMPLE):.2f} "
             f"at {DEFAULT_PIXEL_SIZE_UM} um/px | 1 grid px = "
             f"{DEFAULT_PIXEL_SIZE_UM * DOWNSAMPLE:.2f} um")

    df = pd.read_csv(run_dir / MANIFEST_FILENAME)
    log.info(f"Manifest: {len(df)} rows")

    results, n_done, warned = [], 0, {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building KDE maps"):
        image_id = str(row["image_id"])
        if args.only and image_id != args.only:
            continue
        label = row["label"]
        w, h = int(row["image_width"]), int(row["image_height"])
        mpp = resolve_pixel_size_um(row, warned, log)

        det = find_detection_file(image_id, Path(run_dir) / DETECTIONS_SUBFOLDER)
        if det is None:
            results.append(dict(image_id=image_id, label=label, status="no CellViT output yet"))
            continue

        try:
            status, stats = process_one(image_id, label, w, h, mpp, det, out_dir, log,
                                        force=args.force)
        except Exception as e:
            results.append(dict(image_id=image_id, label=label, status=f"failed: {e}"))
            log.error(f"slide={image_id}: FAILED - {e}")
            continue

        rec = dict(image_id=image_id, label=label, status=status)
        if status == "ok":
            n_done += 1
            d = stats.pop("_diag")
            rec.update(total_cells=d["n_total"], frac_oob=round(d["frac_oob"], 6),
                       coverage_row=round(d["coverage_row"], 3),
                       coverage_col=round(d["coverage_col"], 3),
                       geojson_fallback=d["used_geojson_fallback"])
            for ch in CHANNEL_ORDER:
                rec[f"n_{ch.lower()}"] = stats[ch]["n_cells"]
            tot = max(stats.get("all", {}).get("n_cells", 0), 1)
            rec["neo_frac"] = round(stats["Neoplastic"]["n_cells"] / tot, 4)
            tqdm.write("[KDE] " + f"{image_id} ({label}) " + "  ".join(
                f"{c[:4]}={stats[c]['n_cells']:,}" for c in CHANNEL_ORDER) +
                f"  all={stats.get('all', {}).get('n_cells', 0):,}"
                f"  neo_frac={rec['neo_frac']:.2f}  oob={d['frac_oob']:.2%}")
        results.append(rec)

        if args.limit and n_done >= args.limit:
            break

    log_df = pd.DataFrame(results)
    log_df.to_csv(out_dir / "kde_build_log.csv", index=False)

    def n(s):
        return int((log_df["status"] == s).sum()) if len(log_df) else 0

    ok = log_df[log_df["status"] == "ok"] if len(log_df) else log_df
    failed = log_df[~log_df["status"].isin(["ok", "cached", "no CellViT output yet"])] \
        if len(log_df) else log_df
    log.info(f"Done. ok={len(ok)} cached={n('cached')} "
             f"pending_cellvit={n('no CellViT output yet')} failed={len(failed)}")

    if len(ok):
        # Cell-type composition by label. The neoplastic column should broadly
        # track the previously validated ordering (EC/HGSC high, CC/MC lower);
        # a wild departure means something upstream changed, not biology.
        cols = [f"n_{c.lower()}" for c in CHANNEL_ORDER]
        comp = ok.groupby("label")[cols].sum()
        frac = comp.div(comp.sum(axis=1), axis=0).round(3)
        print("\nCell-type composition by label (fraction of detected cells):")
        print(frac.to_string())
        print("\nCoordinate health:")
        print(f"  max frac_oob      = {ok['frac_oob'].max():.4%} "
              f"(any nonzero is worth a look)")
        print(f"  min coverage_row  = {ok['coverage_row'].min():.2f}")
        print(f"  min coverage_col  = {ok['coverage_col'].min():.2f}")
        if ok["geojson_fallback"].any():
            bad = ok[ok["geojson_fallback"]]["image_id"].tolist()
            print(f"  GeoJSON fallback used on {len(bad)} slide(s): {bad[:10]}")

    if len(failed):
        print("\nFailures:")
        print(failed[["image_id", "label", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()


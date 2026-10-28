# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Per-tile CYTOPLASM features from the eosin channel.

WHY THIS BLOCK EXISTS
---------------------
Every feature in the project so far describes NUCLEI -- geometry from the
CellViT contours (extract_nuclear_features.py) and chromatin texture from the
haematoxylin channel (extract_chromatin_features.py). The eosin channel, which
is where cytoplasm lives, has been computed and thrown away at every stage.

That is a real gap, because two of the five subtypes are diagnostically defined
in the cytoplasm, not the nucleus:

  * CC  (clear cell carcinoma)  -- defined by CLEAR cytoplasm: glycogen washes
        out in processing and leaves optically empty space around the nucleus.
        A pathologist calls CC largely on that appearance.
  * MC  (mucinous carcinoma)    -- defined by intracellular/extracellular
        MUCIN: pools of pale blue-grey material that take up neither stain
        strongly.

Both read as "tissue that is present but pale in both channels", which is
exactly what a haematoxylin-only feature set cannot see: an empty region and a
pale-pink region have the same (low) haematoxylin. So the model currently has
no way to distinguish the single most characteristic feature of two classes.

MC is also the class that Task 3c found resting almost entirely on one spacing
feature (nnd_all_um_cv, top driver on 95.5% of correct MC calls) -- a fragile
one-feature class. Mucin content is the mechanism that feature is a proxy for;
measuring it directly is the principled fix.

WHAT IS MEASURED
----------------
Two views, both on the eosin concentration map normalised to the slide's own
tissue percentiles (same robustness argument as normalise_h -- see that
docstring; per-slide Macenko is NOT used, it fails on this corpus):

  1. Tile level, over all tissue pixels: eosin level and spread, H:E balance,
     eosin texture, and the two "emptiness" fractions -- `clear_frac` (low in
     BOTH channels = optically empty tissue) and `pale_frac` (low eosin only).

  2. Peri-nuclear, per nucleus: the RING outside the inscribed ellipse of each
     detected nucleus is the cytoplasm belonging to that cell. Measuring eosin
     there, rather than over the whole tile, is what separates "this tumour
     cell has clear cytoplasm" from "this tile happens to contain some empty
     space". `clear_ring_frac` -- the fraction of tumour nuclei sitting in
     clear cytoplasm -- is the direct CC feature.

Also emitted: `tile_nuc_area_frac`, nuclear area over tissue area, a tile-level
proxy for the nuclear-to-cytoplasmic ratio (a standard grading parameter that
nothing in the project currently captures).

Output: cellvit_run/graphs/eosin_feats/<id>_eosin.npz, in the SAME node order
as the graph, guarded by the same grid_rc_checksum as the other blocks.

    python scripts/pipeline/extract_eosin_features.py --check --limit 5    # ALWAYS run first
    python scripts/pipeline/extract_eosin_features.py --workers 3
"""
from __future__ import annotations

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
import json
import time
from pathlib import Path

import numpy as np

import corpus_paths
from extract_chromatin_features import (BLACK_RGB_THRESH, BG_RGB_THRESH,
                                        DET_DIR, GRAPH_DIR, MAX_BBOX_PX,
                                        MIN_ALL_CHROM, MIN_NEO_CHROM,
                                        MIN_TISSUE_FRAC_NUC, NEO_CODE, REF_HE,
                                        ROOT, SAMPLE_PER_TILE, STAIN_PIX_CAP,
                                        STAIN_TILES, TIFF_DIR, ellipse_masks,
                                        load_detections, load_graph_geometry,
                                        open_slide, read_tile, rgb_to_od,
                                        slide_path, tissue_mask)

OUT_DIR = GRAPH_DIR / "eosin_feats"

# CLEARING IS MEASURED AS ENCLOSED EMPTY SPACE, NOT AS LOW STAIN
# The obvious implementation -- "a tissue pixel with low H and low E" -- was
# tried first and returned identically 0.0000 on every check slide. The reason
# is a genuine trap: clear cytoplasm and mucin are OPTICALLY EMPTY, so they are
# bright, so tissue_mask() (which calls anything brighter than BG_RGB_THRESH
# glass) throws away precisely the pixels that define CC and MC before they can
# be measured. The same trap fires on the per-nucleus path, where
# MIN_TISSUE_FRAC_NUC would drop a clear-cell nucleus wholesale for sitting in
# a pale window.
#
# So clearing is defined morphologically instead: close the tissue mask to get
# the tissue REGION (tissue plus the gaps it encloses), and count the empty
# pixels inside it. That distinguishes "empty space surrounded by tissue"
# (clear cytoplasm, a mucin pool, a gland lumen) from "empty space off the edge
# of the section" (glass), which is the distinction the raw brightness cannot
# make and the one that carries the diagnosis.
CLOSE_PX = 24             # ~12 um at 0.5 um/px: spans a clear cell, not a lumen
PALE_E = 0.25             # low eosin among pixels that DO hold stain
E_RANGE_CLIP = (-0.5, 2.0)
TILE_SUB = 2              # stride for all TILE-level statistics (see measure_tile)
ENT_BINS = 16
MIN_REGION_FRAC_NUC = 0.80   # nucleus window must be inside the tissue region


def _build_e_lut(HE=REF_HE):
    """Lookup tables mapping RGB straight to eosin concentration.

    Identical construction to _build_h_lut but taking row 1 of pinv(HE); see
    that docstring for why this collapses to three 256-entry tables. The naive
    lstsq route was 66% of the chromatin extractor's runtime before it was
    replaced, and this script reads exactly the same tiles.
    """
    pinv = np.linalg.pinv(HE)[1]
    vals = -np.log10(np.clip(np.arange(256, dtype=np.float32), 1.0, 255.0) / 255.0)
    return np.stack([pinv[c] * vals for c in range(3)]).astype(np.float32)


_E_LUT = _build_e_lut()


def deconvolve_e(rgb):
    return (_E_LUT[0][rgb[..., 0]] + _E_LUT[1][rgb[..., 1]]
            + _E_LUT[2][rgb[..., 2]])


def normalise(raw, lo, hi):
    return np.clip((raw - lo) / max(hi - lo, 1e-3), *E_RANGE_CLIP)


def tissue_region(tissue, close_px=None):
    """Tissue plus the empty space it encloses, by morphological closing.

    `tissue` is the dense-stain mask. Closing it with a kernel a little wider
    than a cell bridges the pale gaps *inside* the section (clear cytoplasm,
    mucin, gland lumina) while leaving the glass beyond the section's edge
    outside, because that has no tissue on the far side to close against.

    region & ~tissue is therefore "optically empty, but surrounded by tissue",
    which is the quantity CC and MC are diagnosed on.
    """
    import cv2
    k_px = CLOSE_PX if close_px is None else close_px
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_px, k_px))
    closed = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_CLOSE, k)
    return closed.astype(bool)


def upsample_mask(mask, y0, y1, x0, x1, step):
    """Read a full-resolution window out of a mask computed at 1/step scale."""
    ry = np.clip(np.arange(y0, y1) // step, 0, mask.shape[0] - 1)
    rx = np.clip(np.arange(x0, x1) // step, 0, mask.shape[1] - 1)
    return mask[ry][:, rx]


def fit_slide_stain_he(sl, tile_origins, tile_px, rng):
    """Slide tissue percentiles for BOTH channels.

    extract_chromatin_features.fit_slide_stain returns only the haematoxylin
    range plus a qc dict; the eosin p1 needed to normalise the E channel is not
    in it. Rather than change that function (its output feeds the batch-effect
    covariates in resolve_dead_count.py and must stay stable), the same fit is
    repeated here for both channels.
    """
    pick = tile_origins
    if len(pick) > STAIN_TILES:
        pick = [pick[i] for i in rng.choice(len(pick), STAIN_TILES, replace=False)]
    rgbs = []
    bgs = []
    for (x, y) in pick:
        try:
            rgb = read_tile(sl, x, y, tile_px)
        except Exception:
            continue
        sub = rgb[::4, ::4]
        tis = tissue_mask(sub)
        bgs.append(float(1.0 - tis.mean()))
        if tis.any():
            rgbs.append(sub[tis])
    if not rgbs:
        return None
    tis_rgb = np.concatenate(rgbs, axis=0)
    if len(tis_rgb) > STAIN_PIX_CAP:
        tis_rgb = tis_rgb[rng.choice(len(tis_rgb), STAIN_PIX_CAP, replace=False)]
    od = rgb_to_od(tis_rgb)
    C = np.linalg.lstsq(REF_HE, od.T, rcond=None)[0]
    h_lo, h_hi = np.percentile(C[0], [1, 99])
    e_lo, e_hi = np.percentile(C[1], [1, 99])
    if h_hi - h_lo < 1e-3 or e_hi - e_lo < 1e-3:
        return None
    return dict(h_lo=float(h_lo), h_hi=float(h_hi),
                e_lo=float(e_lo), e_hi=float(e_hi),
                bg_frac=float(np.mean(bgs)) if bgs else 1.0,
                stain_tiles=len(rgbs))


RING_METRICS = ["e_ring_mean", "e_ring_std", "e_ring_p10", "e_ring_cv",
                "e_core_ring_diff", "h_core_minus_e_ring", "ring_empty_frac"]

TILE_NAMES = ["tile_e_mean", "tile_e_std", "tile_e_p10", "tile_e_p90",
              "tile_e_entropy", "tile_e_grad", "tile_he_ratio",
              "tile_clear_frac", "tile_pale_frac", "tile_nuc_area_frac",
              "tile_region_frac"]

CV_METRICS = ("e_ring_mean", "e_ring_std", "e_core_ring_diff")


def build_feature_names():
    names = list(TILE_NAMES)
    for grp in ("neo", "all"):
        for m in RING_METRICS:
            names.append(f"{grp}_{m}_mean")
        for m in CV_METRICS:
            names.append(f"{grp}_{m}_cv")
        names.append(f"eos_n_{grp}_measured")
    names += ["neo_eos_measured", "all_eos_measured"]
    return names


FEATURE_NAMES = build_feature_names()
FLAG_COLS = {"neo_eos_measured", "all_eos_measured"}
LOG1P_COLS = {"eos_n_neo_measured", "eos_n_all_measured"}


def _entropy(v, lo=0.0, hi=1.0, bins=ENT_BINS):
    if v.size < 8:
        return 0.0
    h, _ = np.histogram(v, bins=bins, range=(lo, hi))
    p = h.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def nucleus_ring_metrics(e_crop, h_crop, region, tissue):
    """Eosin in the ring around one nucleus, plus the nucleus/cytoplasm contrast.

    core = the eroded inscribed ellipse (the nucleus itself); ring = outside the
    full ellipse (its immediate cytoplasm).

    Validity is judged against the tissue REGION, not the dense-stain mask: a
    clear-cell nucleus sits in cytoplasm that holds almost no stain, so gating
    on stain density would discard exactly the cells the feature exists to
    find. Pixels off the edge of the section are still excluded, because the
    closing that builds the region does not reach across glass.

    `ring_empty_frac` is then the direct per-cell reading of clear cytoplasm:
    the share of this nucleus's surround that is inside the section but holds
    no stain.
    """
    h_, w_ = e_crop.shape
    if h_ < 3 or w_ < 3:
        return None
    core, ring = ellipse_masks(h_, w_)
    if region.mean() < MIN_REGION_FRAC_NUC:
        return None
    core_v = core & region
    ring_v = ring & region
    if core_v.sum() < 4 or ring_v.sum() < 6:
        return None
    er = e_crop[ring_v]
    ec = e_crop[core_v]
    hc = h_crop[core_v]
    m = float(er.mean())
    sd = float(er.std())
    return {
        "e_ring_mean": m,
        "e_ring_std": sd,
        "e_ring_p10": float(np.percentile(er, 10)),
        "e_ring_cv": float(sd / (abs(m) + 1e-3)),
        # within one channel, so this IS a physical invariant: a nucleus holds
        # less eosin than the cytoplasm around it, hence expected negative.
        "e_core_ring_diff": float(ec.mean() - m),
        # nuclear haematoxylin against cytoplasmic eosin -- a contrast index,
        # not an invariant (the two channels carry separate percentile
        # scalings), but it is high exactly when a dark nucleus sits in pale
        # cytoplasm, which is what "clear cell" describes.
        "h_core_minus_e_ring": float(hc.mean() - m),
        "ring_empty_frac": float((ring_v & ~tissue).sum() / max(ring_v.sum(), 1)),
    }


def aggregate(rows, prefix, out):
    n = len(rows)
    if n == 0:
        return 0
    for m in RING_METRICS:
        v = np.array([r[m] for r in rows], dtype=np.float64)
        out[f"{prefix}_{m}_mean"] = float(v.mean())
        if m in CV_METRICS:
            out[f"{prefix}_{m}_cv"] = float(v.std() / (abs(v.mean()) + 1e-3))
    return n


def measure_tile(sl, g, node, src, bounds, cx, cy, ct, bb, st, tile_px, rng):
    idx = src[bounds[node]:bounds[node + 1]]
    if len(idx) < MIN_ALL_CHROM:
        return None
    ox = int(g["grid_rc"][node, 1] * tile_px)
    oy = int(g["grid_rc"][node, 0] * tile_px)
    try:
        rgb = read_tile(sl, ox, oy, tile_px)
    except Exception:
        return None

    from extract_chromatin_features import deconvolve_h
    # Tile-level statistics are regional quantities -- fractions, percentiles,
    # a gradient magnitude -- so they are computed on a STRIDED copy. tile_px is
    # 1024 here, a megapixel per tile, and deconvolving all of it dominated the
    # profile at 232 ms/tile even though the per-nucleus loop only ever touches
    # a couple of percent of those pixels. The nuclei themselves are still
    # measured at full resolution below: a nucleus bbox is ~16 px across and
    # would not survive downsampling.
    s = TILE_SUB
    sub = rgb[::s, ::s]
    e_s = normalise(deconvolve_e(sub), st["e_lo"], st["e_hi"])
    h_s = normalise(deconvolve_h(sub), st["h_lo"], st["h_hi"])
    tis_s = tissue_mask(sub)
    reg_s = tissue_region(tis_s, close_px=max(3, CLOSE_PX // s))

    out = {k: 0.0 for k in FEATURE_NAMES}
    out["tile_region_frac"] = float(reg_s.mean())
    if tis_s.any():
        et = e_s[tis_s]
        ht = h_s[tis_s]
        p10, p90 = np.percentile(et, [10, 90])
        out["tile_e_mean"] = float(et.mean())
        out["tile_e_std"] = float(et.std())
        out["tile_e_p10"] = float(p10)
        out["tile_e_p90"] = float(p90)
        out["tile_e_entropy"] = _entropy(et)
        gy, gx = np.gradient(e_s.astype(np.float32))
        out["tile_e_grad"] = float(np.sqrt(gy ** 2 + gx ** 2).mean()) / s
        out["tile_he_ratio"] = float(ht.mean() / (et.mean() + 1e-3))
        # cytoplasm that holds stain but little of it (washed-out / pale pink)
        out["tile_pale_frac"] = float((et < PALE_E).mean())
    if reg_s.any():
        # the CC/MC signal: optically empty space ENCLOSED by tissue, as a
        # share of the tissue region. Glass beyond the section is not in
        # `region` and so cannot inflate this.
        out["tile_clear_frac"] = float((reg_s & ~tis_s).sum() / reg_s.sum())

    # nuclear area over tissue area: an N:C ratio proxy
    tis_px = float(tis_s.sum()) * (s * s)
    if tis_px > 0:
        area = 0.0
        for i in idx:
            x0, y0, x1, y1 = bb[i]
            bw, bh = x1 - x0, y1 - y0
            if 3 <= bw <= MAX_BBOX_PX and 3 <= bh <= MAX_BBOX_PX:
                area += np.pi * (bw / 2.0) * (bh / 2.0)
        out["tile_nuc_area_frac"] = float(min(area / tis_px, 1.0))

    neo_idx = idx[ct[idx] == NEO_CODE]
    for group, sel in (("neo", neo_idx), ("all", idx)):
        if len(sel) > SAMPLE_PER_TILE:
            sel = sel[rng.choice(len(sel), SAMPLE_PER_TILE, replace=False)]
        rows = []
        for i in sel:
            x0, y0, x1, y1 = bb[i]
            bw, bh = x1 - x0, y1 - y0
            if bw < 3 or bh < 3 or bw > MAX_BBOX_PX or bh > MAX_BBOX_PX:
                continue
            lx0, ly0 = max(x0 - ox, 0), max(y0 - oy, 0)
            lx1, ly1 = min(x1 - ox, tile_px), min(y1 - oy, tile_px)
            if lx1 - lx0 < 3 or ly1 - ly0 < 3:
                continue
            # deconvolve only this nucleus's window, at full resolution
            crop = rgb[ly0:ly1, lx0:lx1]
            m = nucleus_ring_metrics(
                normalise(deconvolve_e(crop), st["e_lo"], st["e_hi"]),
                normalise(deconvolve_h(crop), st["h_lo"], st["h_hi"]),
                upsample_mask(reg_s, ly0, ly1, lx0, lx1, s),
                tissue_mask(crop))
            if m is not None:
                rows.append(m)
        n = aggregate(rows, group, out)
        out[f"eos_n_{group}_measured"] = float(n)
    out["neo_eos_measured"] = 1.0 if out["eos_n_neo_measured"] >= MIN_NEO_CHROM else 0.0
    out["all_eos_measured"] = 1.0 if out["eos_n_all_measured"] >= MIN_ALL_CHROM else 0.0
    return out


def process_slide(gpath, check=False, seed=0):
    g = load_graph_geometry(gpath)
    iid = g["image_id"]
    det = DET_DIR / iid / "cell_detection.json"
    tif = slide_path(iid)
    if not det.exists():
        return None, "missing cell_detection.json"
    if not tif.exists():
        return None, f"missing slide {tif.name}"

    rng = np.random.RandomState(seed)
    cx, cy, ct, bb = load_detections(det)
    if len(cx) == 0:
        return None, "zero cells"

    n_rows, n_cols = g["grid_shape"]
    tile_px = g["tile_px"]
    node_of_grid = np.full((n_rows, n_cols), -1, np.int64)
    node_of_grid[g["grid_rc"][:, 0], g["grid_rc"][:, 1]] = np.arange(g["n_nodes"])
    gc = np.clip((cx // tile_px).astype(np.int64), 0, n_cols - 1)
    gr = np.clip((cy // tile_px).astype(np.int64), 0, n_rows - 1)
    node_id = node_of_grid[gr, gc]
    on_node = node_id >= 0

    sl = open_slide(tif)
    try:
        w0, h0 = sl.level_dimensions[0]
        if (h0, w0) != g["wsi_shape"]:
            return None, (f"dim mismatch: slide (h,w)=({h0},{w0}) vs graph "
                          f"{g['wsi_shape']}")
        order = np.argsort(node_id[on_node], kind="stable")
        ni = node_id[on_node][order]
        src = np.flatnonzero(on_node)[order]
        bounds = np.searchsorted(ni, np.arange(g["n_nodes"] + 1))
        populated = np.flatnonzero(np.diff(bounds) >= MIN_ALL_CHROM)
        origins = [(int(g["grid_rc"][n, 1] * tile_px),
                    int(g["grid_rc"][n, 0] * tile_px)) for n in populated]
        if not origins:
            return None, "no populated tiles"

        st = fit_slide_stain_he(sl, origins, tile_px, rng)
        if st is None:
            return None, "stain model failed"

        qc = dict(image_id=iid, label=g["label"], n_tiles=g["n_nodes"],
                  n_populated=len(populated), **st)

        if check:
            demo = []
            for n in populated[:: max(1, len(populated) // 8)][:8]:
                row = measure_tile(sl, g, n, src, bounds, cx, cy, ct, bb, st,
                                   tile_px, rng)
                if row is not None:
                    demo.append(row)
            if demo:
                for k in ("tile_e_mean", "tile_clear_frac", "tile_pale_frac",
                          "tile_he_ratio", "tile_nuc_area_frac",
                          "neo_e_ring_mean_mean", "neo_ring_empty_frac_mean",
                          "neo_e_core_ring_diff_mean",
                          "neo_h_core_minus_e_ring_mean"):
                    qc[k] = round(float(np.mean([d[k] for d in demo])), 4)
                qc["demo_tiles"] = len(demo)
            return None, qc

        feats = np.zeros((g["n_nodes"], len(FEATURE_NAMES)), np.float32)
        n_meas = 0
        for n in populated:
            row = measure_tile(sl, g, n, src, bounds, cx, cy, ct, bb, st,
                               tile_px, rng)
            if row is None:
                continue
            feats[n] = [row[k] for k in FEATURE_NAMES]
            n_meas += 1
    finally:
        sl.close()

    qc["n_measured"] = n_meas
    payload = dict(
        image_id=np.array([iid]), label=np.array([g["label"]]),
        eos_features=feats,
        eos_feature_names=np.array(FEATURE_NAMES, dtype=object),
        eos_log1p_cols=np.array([n in LOG1P_COLS for n in FEATURE_NAMES]),
        eos_flag_cols=np.array([n in FLAG_COLS for n in FEATURE_NAMES]),
        grid_shape=np.array(g["grid_shape"]),
        grid_rc_checksum=np.array([int(g["grid_rc"].sum())]),
        stain_HE=REF_HE,
        stain_range=np.array([st["h_lo"], st["h_hi"], st["e_lo"], st["e_hi"]],
                             np.float32),
        slide_qc=np.array([json.dumps(qc)]),
        params=np.array([json.dumps(dict(
            CLOSE_PX=CLOSE_PX, PALE_E=PALE_E,
            SAMPLE_PER_TILE=SAMPLE_PER_TILE, MIN_NEO=MIN_NEO_CHROM,
            MIN_ALL=MIN_ALL_CHROM, MIN_REGION_FRAC_NUC=MIN_REGION_FRAC_NUC,
            BG_RGB_THRESH=BG_RGB_THRESH, BLACK_RGB_THRESH=BLACK_RGB_THRESH,
            clearing="enclosed empty space via morphological closing",
            stain="fixed REF_HE + per-slide tissue p1/p99, H and E"))]),
    )
    return payload, qc


def _worker(job):
    gpath, check, seed = job
    iid = corpus_paths.image_id_from_graph(gpath)
    t0 = time.time()
    try:
        payload, qc = process_slide(Path(gpath), check, seed)
    except Exception as e:
        import traceback
        return dict(image_id=iid, error=f"{type(e).__name__}: {e}"[:300],
                    tb=traceback.format_exc()[-800:], secs=time.time() - t0)
    if isinstance(qc, str):
        return dict(image_id=iid, error=qc, secs=time.time() - t0)
    if payload is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT_DIR / f"{iid}_eosin.npz", **payload)
    qc["secs"] = round(time.time() - t0, 1)
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="measure a few tiles per slide and print the stain "
                         "sanity numbers WITHOUT writing anything")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    graphs = sorted(GRAPH_DIR.glob("*_graph.npz"))
    if not args.check and not args.force:
        graphs = [g for g in graphs
                  if not (OUT_DIR / f"{corpus_paths.image_id_from_graph(g)}"
                                    f"_eosin.npz").exists()]
    if args.limit:
        graphs = graphs[:args.limit]
    print(corpus_paths.banner("eosin"))
    print(f"eosin extraction: {len(graphs)} slide(s) to do "
          f"({'CHECK, nothing written' if args.check else str(OUT_DIR)})")
    print(f"  {len(FEATURE_NAMES)} features per tile")
    if not graphs:
        return

    jobs = [(str(g), args.check, args.seed) for g in graphs]
    t0 = time.time()
    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, qc in enumerate(pool.imap_unordered(_worker, jobs), 1):
                rows.append(qc)
                _log(i, len(jobs), qc, t0, args.check)
    else:
        for i, job in enumerate(jobs, 1):
            qc = _worker(job)
            rows.append(qc)
            _log(i, len(jobs), qc, t0, args.check)

    import pandas as pd
    df = pd.DataFrame(rows)
    bad = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    print(f"\ndone: {len(df) - len(bad)} ok, {len(bad)} failed, "
          f"{(time.time() - t0) / 60:.1f} min")
    if len(bad):
        print(bad[["image_id", "error"]].head(20).to_string(index=False))
    if args.check and "tile_clear_frac" in df:
        print("\n  CHECK -- these must be physically sensible before a full run:")
        for c in ["tile_e_mean", "tile_clear_frac", "tile_pale_frac",
                  "tile_nuc_area_frac", "neo_e_ring_mean_mean",
                  "neo_ring_empty_frac_mean", "neo_e_core_ring_diff_mean",
                  "neo_h_core_minus_e_ring_mean"]:
            if c in df:
                print(f"    {c:32s} {df[c].describe()[['min','50%','max']].round(4).to_dict()}")
        print("\n    neo_e_core_ring_diff_mean must be NEGATIVE -- within the "
              "eosin channel a\n    nucleus holds less stain than the cytoplasm "
              "around it. That one is a\n    physical invariant and a positive "
              "value means the channel is inverted.\n"
              "    tile_clear_frac must VARY across slides and be non-zero on "
              "some; if it is\n    identically 0 the enclosed-space "
              "construction is not firing.")
    if not args.check:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "eosin_manifest.csv", index=False)


def _log(i, n, qc, t0, check=False):
    el = (time.time() - t0) / 60
    if qc.get("error"):
        print(f"  [{i}/{n}] {qc['image_id']} ERROR {qc['error']}  ({el:.1f}m)")
        return
    extra = ""
    if check:
        extra = (f" e_mean={qc.get('tile_e_mean')} "
                 f"clear={qc.get('tile_clear_frac')} "
                 f"h-e={qc.get('neo_h_core_minus_e_ring_mean')}")
    print(f"  [{i}/{n}] {qc['image_id']} {qc.get('label','')} "
          f"tiles={qc.get('n_populated')} {qc.get('secs')}s{extra}  ({el:.1f}m)")


if __name__ == "__main__":
    main()

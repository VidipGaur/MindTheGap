# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Chromatin and texture features from WSI pixels, aligned to the existing graphs.

    python scripts/pipeline/extract_chromatin_features.py --check      # 5 slides, verify alignment + stain fit, write nothing
    python scripts/pipeline/extract_chromatin_features.py --smoke      # 3 slides, full pipeline, writes
    python scripts/pipeline/extract_chromatin_features.py --workers 3  # all 456 slides (resumes)

WHAT THIS ADDS AND WHY
======================
Every feature the model currently has is GEOMETRY -- area, eccentricity,
solidity, spacing -- because contours are all `extract_nuclear_features.py`
took from CellViT. But nuclear atypia grading, the actual diagnostic criterion
separating HGSC from LGSC and EC, is substantially about CHROMATIN: how dark
the nucleus is (hyperchromasia), how coarse and clumped the chromatin is, how
irregularly it is distributed. None of that exists in a contour. It only exists
in the pixels.

So this goes upstream: for every tile, read the actual image, and measure the
chromatin inside the nuclei that CellViT found.

    cell_detection.json -> per-nucleus bbox + type   (44 MB/slide; the 314 MB
                           cells.json contours are not needed, see MASKS)
    pyramid TIFF        -> the pixels
    colour deconvolution + per-slide tissue normalisation
    -> per-tile chromatin statistics, same node order as <id>_graph.npz

Output: `cellvit_run/graphs/chromatin_feats/<id>_chromatin.npz`, with
`chrom_features` of shape (K, F) where K and the row order match
`node_features` in `<id>_graph.npz` exactly, guarded by the same
`grid_rc_checksum` the nuclear extractor uses.

STAIN NORMALISATION IS NOT OPTIONAL HERE
========================================
H&E varies enormously between sites and scanners -- across five check slides
the 99th percentile of tissue haematoxylin ranges 0.29 to 0.80, a 2.8x spread
that is staining, not biology. Raw intensity features would substantially
encode that, and this project already has one feature (`dead_count`) under
suspicion for exactly this failure. So every intensity feature is computed on
deconvolved HAEMATOXYLIN CONCENTRATION, normalised to each slide's own tissue
range rather than used raw.

Two further defences:
  * `h_ring_diff` compares each nucleus to the tissue immediately around it, so
    it measures local contrast rather than absolute darkness. If it and the
    absolute features disagree downstream, the normalisation is the thing to
    distrust.
  * The per-slide staining descriptors are SAVED -- background fraction, tissue
    optical density, H and E levels, the H:E balance, and blur. Those are
    exactly the batch-effect covariates needed to test whether `dead_count`
    tracks fixation quality rather than biology, and `train.csv` carries no
    site or scanner column to do it with. See resolve_dead_count.py.

WHY NOT PER-SLIDE MACENKO
=========================
It was tried first and abandoned; see fit_slide_stain() for the full account.
Briefly: the per-slide SVD is near-degenerate on slides dominated by one
stain, it returned achromatic "haematoxylin" vectors, and the resulting
ill-conditioned solve made nuclei measure lighter than the tissue around them
on 2 of 5 check slides. Cosine similarity to the reference vectors cannot gate
it, because every vector in the positive octant scores above 0.9 against every
other. Fixed reference vectors plus per-slide tissue-percentile scaling is both
more robust and directly verified: haematoxylin is higher inside the nucleus
than in the surrounding ring on 97-100% of neoplastic nuclei.

MASKS: why bbox-inscribed ellipses rather than the CellViT contours
===================================================================
`cells.json` carries real contours but costs 139.6 GB across the corpus and
this box has no `ijson`, so it would be parsed by a pure-Python byte scanner.
`cell_detection.json` is 19.6 GB and loads with plain `json.loads`.

A CellViT bbox is tight around the nucleus, and a nucleus is close to
elliptical, so the inscribed ellipse recovers essentially the nucleus and
little else. It is then eroded to ERODE_FRAC of its axes for the chromatin
statistics, which pushes the sampled region away from the membrane where
partial-volume mixing with cytoplasm lives. The discarded annulus is not
wasted: the ring between the ellipse and the bbox is used as the local
background for `h_ring_diff`.

The convention was verified rather than assumed: median bbox is 16x16 px,
i.e. 8x8 um at 0.5 um/px, exactly a tumour nucleus, and the stored centroid
sits at the bbox centre to within half a pixel (the transposed reading is off
by ~15,000 px, so there is no ambiguity).

The alternative -- re-segmenting each nucleus by thresholding the image -- was
rejected deliberately: it would make the mask itself a function of staining
intensity, which is the confound this whole script is trying not to reproduce.

COST
====
Dominated by WSI reads: one 1024x1024 level-0 region per populated tile, ~640
tiles per slide. Nuclei are subsampled to SAMPLE_PER_TILE per class per tile,
which is ample for a mean/CV and caps the per-tile compute. Use --workers to
run several slides at once; each worker holds one tile (3 MB) at a time.
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
import os
import sys
import time
from pathlib import Path

import numpy as np

import corpus_paths

ROOT = corpus_paths.ROOT
GRAPH_DIR = corpus_paths.GRAPH_DIR
DET_DIR = corpus_paths.DET_DIR
TIFF_DIR = corpus_paths.TIFF_DIR
OUT_DIR = GRAPH_DIR / "chromatin_feats"

# Slide pixels. On UBC-OCEAN this is wsi_pyramid_tiff/<id>.tiff, unchanged; a
# corpus manifest carrying a slide_path column (e.g. .svs under a Training/Test
# split) overrides it, so the extractors never encode a naming convention.
slide_path = corpus_paths.slide_path

CLASSES = ["CC", "EC", "HGSC", "LGSC", "MC"]
TYPE_MAP = {1: "neoplastic", 2: "inflammatory", 3: "connective",
            4: "dead", 5: "epithelial"}
NEO_CODE = 1

SAMPLE_PER_TILE = 30      # nuclei measured per class per tile; a mean/CV over
                          # 30 is already tight and this caps per-tile cost
MIN_NEO_CHROM = 10        # matches MIN_NEO in extract_nuclear_features.py so
                          # neo_measured and neo_chrom_measured gate alike
MIN_ALL_CHROM = 8         # matches MIN_NUC
MIN_NUC_PX = 9            # a nucleus smaller than this many mask pixels has no
                          # meaningful texture -- skipped
MAX_BBOX_PX = 200         # segmentation artefact guard (100 um at 0.5 um/px)
ERODE_FRAC = 0.75         # ellipse axes shrink factor for the chromatin mask
GLCM_LEVELS = 16
# histogram / quantisation range for the tissue-normalised haematoxylin
# channel. Normalisation puts tissue p1..p99 on 0..1, and nuclei are the dark
# end of that, so nuclear values live roughly in 0.4-1.3.
H_RANGE = (0.0, 1.5)

STAIN_TILES = 20          # tiles sampled to fit the per-slide stain vectors
STAIN_PIX_CAP = 200_000
BG_RGB_THRESH = 220       # brighter than this is glass
# ...and darker than this is SCANNER PADDING, not tissue. These pyramid TIFFs
# are padded with pure black outside the scanned region. Black has maximal
# optical density, so leaving it in put the 99th percentile of tissue
# haematoxylin at 3.8654 -- the exact value of an RGB (0,0,0) pixel -- on 3 of
# 5 check slides, which destroyed the normalisation range. It also landed in
# the ring used for the nucleus/surround ratio, which is why nuclei came out
# measuring "lighter" than their surroundings (ratio 0.64) on slides where the
# direct probe showed the opposite on 100% of nuclei.
BLACK_RGB_THRESH = 15
MIN_TISSUE_FRAC_NUC = 0.80   # a nucleus window this contaminated by non-tissue
                             # is skipped rather than measured through padding

# Ruifrok/Macenko reference H&E optical-density vectors, column 0 = haematoxylin.
# Haematoxylin is blue: it absorbs red and green, transmits blue, so its OD is
# high in R/G and lowest in B. Eosin is pink: lowest OD in R.
REF_HE = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]], dtype=np.float32)

DEFAULT_MPP_WSI = 0.50    # see extract_nuclear_features.py -- the graph's
DEFAULT_MPP_TMA = 0.25    # stored 0.25 is wrong for the WSIs


def rgb_to_od(rgb):
    """Beer-Lambert optical density. rgb uint8 (..,3) -> float32 (..,3)."""
    return -np.log10(np.clip(rgb.astype(np.float32), 1.0, 255.0) / 255.0)


def tissue_mask(rgb):
    """True where the pixel is tissue: neither glass nor black scanner padding.

    Summed in uint16 rather than averaged in float -- same threshold, a third
    of the memory traffic, and this runs on every pixel of every tile.
    """
    s = rgb.astype(np.uint16).sum(axis=-1)
    return (s < BG_RGB_THRESH * 3) & (s > BLACK_RGB_THRESH * 3)


def deconvolve(rgb, HE=REF_HE):
    """RGB -> (haematoxylin, eosin) concentration maps, no per-slide scaling."""
    od = rgb_to_od(rgb).reshape(-1, 3)
    C = np.linalg.lstsq(HE, od.T, rcond=None)[0]      # (2, P)
    return (C[0].reshape(rgb.shape[:2]).astype(np.float32),
            C[1].reshape(rgb.shape[:2]).astype(np.float32))


def _build_h_lut(HE=REF_HE):
    """Per-channel lookup tables that turn an RGB tile straight into
    haematoxylin concentration with three uint8 lookups and two adds.

    The concentration is a fixed linear map of optical density,
    pinv(HE)[0] . OD, and OD is a fixed function of each uint8 channel value.
    So the whole thing collapses into 3 tables of 256 entries.

    This matters: the naive route -- rgb_to_od() then lstsq with a million
    right-hand sides -- measured 143 ms per tile and was 66% of total runtime,
    more than the WSI read itself. Same arithmetic, ~10x faster.
    """
    pinv = np.linalg.pinv(HE)[0]                      # (3,)
    vals = -np.log10(np.clip(np.arange(256, dtype=np.float32), 1.0, 255.0) / 255.0)
    return np.stack([pinv[c] * vals for c in range(3)]).astype(np.float32)


_H_LUT = _build_h_lut()


def deconvolve_h(rgb):
    """RGB tile -> raw haematoxylin concentration, via the lookup tables."""
    return (_H_LUT[0][rgb[..., 0]] + _H_LUT[1][rgb[..., 1]]
            + _H_LUT[2][rgb[..., 2]])


def normalise_h(h_raw, h_lo, h_hi):
    """Map raw haematoxylin concentration onto a common ~[0, 1] tissue range.

    Scaling by the slide's own tissue percentiles rather than by the 99th
    percentile of the fitted concentrations is deliberate: the latter is a
    by-product of the stain fit, so a marginal fit propagated straight into
    every intensity feature. On the check slides it produced normalised nuclear
    haematoxylin of 0.035 on one slide and 0.384 on another -- a 10x spread
    that is staining and fitting, not biology, and it collapsed the texture
    entropy to ~0 because every pixel fell in one histogram bin.

    Percentiles of the tissue distribution are what actually set the dynamic
    range, are robust to a mediocre fit, and put every slide on the same scale.
    """
    return np.clip((h_raw - h_lo) / max(h_hi - h_lo, 1e-3), -0.5, 2.0)


def _ellipse_masks(h, w):
    """(core, ring) boolean masks for an h x w bbox.

    core = ellipse inscribed in the bbox, shrunk by ERODE_FRAC -- chromatin.
    ring = outside the full inscribed ellipse -- the local tissue background
           used for the scanner-invariant ratio feature.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = max(h / 2.0, 0.5), max(w / 2.0, 0.5)
    r2 = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return r2 <= ERODE_FRAC ** 2, r2 > 1.0


_MASK_CACHE = {}


def ellipse_masks(h, w):
    key = (h, w)
    m = _MASK_CACHE.get(key)
    if m is None:
        if len(_MASK_CACHE) > 4096:
            _MASK_CACHE.clear()
        m = _ellipse_masks(h, w)
        _MASK_CACHE[key] = m
    return m


NUC_METRICS = ["h_mean", "h_std", "h_cv", "h_p90_p10", "h_entropy", "h_grad",
               "h_ring_diff", "glcm_contrast", "glcm_homogeneity",
               "glcm_asm", "glcm_corr"]


def nucleus_metrics(crop, crop_raw, valid):
    """Chromatin descriptors for one nucleus.

    crop     : (h, w) float32, haematoxylin normalised to the slide's tissue
               range -- comparable across slides, used for level and texture.
    crop_raw : the same window before normalisation -- used only for the
               nucleus/surround ratio, which must stay a pure ratio so that it
               is invariant to any per-slide scaling.
    valid    : tissue mask for the window. Glass and black scanner padding are
               excluded from every statistic; a window more than
               1-MIN_TISSUE_FRAC_NUC contaminated is rejected outright rather
               than measured through the padding.
    Returns a list aligned to NUC_METRICS, or None if the nucleus is too small
    or sits on non-tissue.
    """
    h, w = crop.shape
    if valid.mean() < MIN_TISSUE_FRAC_NUC:
        return None
    core, ring = ellipse_masks(h, w)
    core = core & valid
    ring = ring & valid
    vals = crop[core]
    if vals.size < MIN_NUC_PX:
        return None
    m = float(vals.mean())
    sd = float(vals.std())
    p10, p90 = np.percentile(vals, [10, 90])

    # chromatin coarseness: entropy of the intensity histogram inside the
    # nucleus. A smooth (vesicular) nucleus concentrates in few bins; a coarse,
    # clumped one spreads out.
    hist, _ = np.histogram(vals, bins=GLCM_LEVELS, range=H_RANGE)
    p = hist.astype(np.float64)
    tot = p.sum()
    ent = 0.0
    if tot > 0:
        p = p[p > 0] / tot
        ent = float(-(p * np.log2(p)).sum())

    # chromatin margination / clumping: mean gradient magnitude inside the core
    gy, gx = np.gradient(crop)
    grad = float(np.hypot(gy, gx)[core].mean())

    # relative hyperchromasia: how much darker the nucleus is than the tissue
    # immediately around it. A DIFFERENCE on the normalised channel, not a
    # ratio on the raw one -- raw haematoxylin concentration goes slightly
    # negative on eosin-rich pixels (tissue p1 reaches -0.11 on some slides),
    # so a ratio there divides by something that can be near zero or negative
    # and both explodes and flips sign. The difference is well behaved, and
    # since the channel is already normalised by the slide's own tissue range
    # it stays comparable across slides.
    rv = crop[ring]
    ring_diff = float(vals.mean() - rv.mean()) if rv.size >= 4 else np.nan

    # GLCM texture on the quantised crop. Computed on the whole bbox rather
    # than the masked core because a co-occurrence matrix needs adjacency, and
    # masking punches holes that would be read as texture.
    q = np.clip(((crop - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0])
                 * (GLCM_LEVELS - 1)).astype(np.int32), 0, GLCM_LEVELS - 1)
    contrast = homog = asm = corr = np.nan
    if q.shape[0] >= 3 and q.shape[1] >= 3:
        # bincount on the flattened (i,j) pair index, not np.add.at -- the
        # latter has a large fixed per-call cost and this runs once per
        # nucleus per direction
        L = GLCM_LEVELS
        acc = np.zeros(L * L, dtype=np.float64)
        for a, b in ((q[:, :-1], q[:, 1:]), (q[:-1, :], q[1:, :])):
            av, bv = a.ravel(), b.ravel()
            acc += np.bincount(av * L + bv, minlength=L * L)
            acc += np.bincount(bv * L + av, minlength=L * L)
        glcm = acc.reshape(L, L)
        s = glcm.sum()
        if s > 0:
            glcm /= s
            i = np.arange(GLCM_LEVELS)
            I, J = np.meshgrid(i, i, indexing="ij")
            d2 = (I - J) ** 2
            contrast = float((glcm * d2).sum())
            homog = float((glcm / (1.0 + d2)).sum())
            asm = float((glcm ** 2).sum())
            mi = float((glcm * I).sum())
            mj = float((glcm * J).sum())
            si = math.sqrt(max(float((glcm * (I - mi) ** 2).sum()), 0.0))
            sj = math.sqrt(max(float((glcm * (J - mj) ** 2).sum()), 0.0))
            if si > 1e-8 and sj > 1e-8:
                corr = float((glcm * (I - mi) * (J - mj)).sum() / (si * sj))
    return [m, sd, sd / m if m > 1e-6 else 0.0, float(p90 / (p10 + 1e-3)),
            ent, grad, ring_diff, contrast, homog, asm, corr]


def build_feature_names():
    names = []
    for grp in ("neo", "all"):
        for k in NUC_METRICS:
            names.append(f"{grp}_{k}_mean")
        # spread matters as much as level: pleomorphism is variability
        for k in ("h_mean", "h_std", "h_entropy", "h_grad", "glcm_contrast"):
            names.append(f"{grp}_{k}_cv")
    names += ["chrom_n_neo_measured", "chrom_n_all_measured",
              "tile_h_mean", "tile_h_std", "tile_tissue_frac",
              "neo_chrom_measured", "all_chrom_measured"]
    return names


FEATURE_NAMES = build_feature_names()
FLAG_COLS = {"neo_chrom_measured", "all_chrom_measured"}
LOG1P_COLS = {"chrom_n_neo_measured", "chrom_n_all_measured"}
CV_METRICS = ("h_mean", "h_std", "h_entropy", "h_grad", "glcm_contrast")


def aggregate(rows, prefix, out):
    """rows: (n, len(NUC_METRICS)) -> mean of each metric + CV of a few."""
    if not len(rows):
        return 0
    A = np.asarray(rows, dtype=np.float64)
    # a whole column can be NaN (e.g. glcm_corr on a flat nucleus, or
    # h_ring_ratio when every sampled nucleus fills its bbox); nanmean warns
    # and returns NaN, which the isfinite guards below turn into 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
    for k, v in zip(NUC_METRICS, mu):
        out[f"{prefix}_{k}_mean"] = float(v) if np.isfinite(v) else 0.0
    for k in CV_METRICS:
        j = NUC_METRICS.index(k)
        c = sd[j] / mu[j] if np.isfinite(mu[j]) and abs(mu[j]) > 1e-6 else 0.0
        out[f"{prefix}_{k}_cv"] = float(c) if np.isfinite(c) else 0.0
    return len(A)


def load_graph_geometry(gpath):
    with np.load(gpath, allow_pickle=True) as z:
        return {
            "image_id": str(z["image_id"][0]),
            "label": str(z["label"][0]),
            "n_nodes": int(z["node_features"].shape[0]),
            "grid_rc": z["node_grid_rc"].astype(np.int64),
            "grid_shape": tuple(int(v) for v in z["node_grid_shape"]),
            "tile_px": int(np.asarray(z["tile_size_wsi_px"]).ravel()[0]),
            "wsi_shape": tuple(int(v) for v in z["wsi_shape"]),
        }


def load_detections(path):
    """cell_detection.json -> centroid xy, type, bbox. Arrays, one row per cell."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = obj.get("cells", [])
    n = len(cells)
    cx = np.empty(n, np.float32); cy = np.empty(n, np.float32)
    ct = np.zeros(n, np.int16)
    bb = np.empty((n, 4), np.int32)          # x0, y0, x1, y1
    k = 0
    for c in cells:
        b = c.get("bbox")
        if b is None:
            continue
        (y0, x0), (y1, x1) = b
        cen = c.get("centroid") or [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        cx[k] = cen[0]; cy[k] = cen[1]
        ct[k] = int(c.get("type") or 0)
        bb[k] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        k += 1
    return cx[:k], cy[:k], ct[:k], bb[:k]


def open_slide(path):
    try:
        import openslide
        return openslide.OpenSlide(str(path))
    except Exception:
        import tiffslide
        return tiffslide.TiffSlide(str(path))


def read_tile(sl, x, y, size):
    return np.asarray(sl.read_region((int(x), int(y)), 0,
                                     (int(size), int(size))).convert("RGB"))


def fit_slide_stain(sl, tile_origins, tile_px, rng):
    """Per-slide stain model, from a spread of tiles.

    Returns (h_lo, h_hi, qc): the 1st and 99th percentiles of haematoxylin
    concentration ON TISSUE, which set the common normalisation range, plus the
    slide-level staining descriptors.

    WHY THE PER-SLIDE STAIN VECTORS ARE NOT FITTED
    ----------------------------------------------
    A per-slide Macenko fit was tried first and abandoned. It is fragile on
    this corpus: the SVD plane is nearly degenerate on slides with one dominant
    stain, and it returned "haematoxylin" vectors like [0.78, 0.51, 0.36] and
    the achromatic [0.58, 0.58, 0.57]. Those propagate into an ill-conditioned
    3x2 solve -- one slide's tissue haematoxylin came out with a 1st percentile
    of -7.6 -- and the resulting nuclei measured LIGHTER than the tissue around
    them on 2 of 5 check slides, which is physically impossible.

    Gating the fit on cosine similarity to the reference vectors does not
    rescue it: every vector in the positive octant scores above 0.9 against
    every other, so the achromatic fit passed a 0.90 threshold comfortably. The
    similarity test cannot separate good fits from bad ones here.

    Fixed reference vectors have none of these failure modes and were verified
    directly: on three slides, with REF_HE, the haematoxylin concentration is
    higher inside the nucleus than in the surrounding ring on 97-100% of
    neoplastic nuclei (and eosin on only 28-31%, confirming the channel is the
    right one). Per-slide staining intensity -- the thing the fit was supposed
    to absorb -- is handled instead by the tissue-percentile normalisation
    below, which is the robust way to do it.

    qc carries background fraction, tissue optical density, blur, H and E
    levels and the H:E balance. Those are not diagnostics for their own sake:
    they are the batch-effect covariates resolve_dead_count.py needs, since
    train.csv has no site or scanner column.
    """
    pick = tile_origins
    if len(pick) > STAIN_TILES:
        pick = [pick[i] for i in rng.choice(len(pick), STAIN_TILES, replace=False)]
    bgs, blurs, tissue_od, rgbs = [], [], [], []
    for (x, y) in pick:
        try:
            rgb = read_tile(sl, x, y, tile_px)
        except Exception:
            continue
        sub = rgb[::4, ::4]
        tis = tissue_mask(sub)
        bgs.append(float(1.0 - tis.mean()))
        if not tis.any():
            continue
        od = rgb_to_od(sub).reshape(-1, 3)
        rgbs.append(sub[tis])
        tissue_od.append(od[tis.reshape(-1)].mean(axis=1))
        g = sub.mean(axis=-1).astype(np.float32)
        lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
               - 4 * g[1:-1, 1:-1])
        blurs.append(float(lap.var()))
    tis = np.concatenate(tissue_od) if tissue_od else np.array([0.0])
    qc = dict(bg_frac=float(np.mean(bgs)) if bgs else 1.0,
              tissue_od_p50=round(float(np.percentile(tis, 50)), 4),
              tissue_od_p95=round(float(np.percentile(tis, 95)), 4),
              blur_lapvar=round(float(np.mean(blurs)), 1) if blurs else 0.0,
              stain_tiles=len(rgbs))
    if not rgbs:
        return None, None, qc

    tis_rgb = np.concatenate(rgbs, axis=0)
    if len(tis_rgb) > STAIN_PIX_CAP:
        tis_rgb = tis_rgb[rng.choice(len(tis_rgb), STAIN_PIX_CAP, replace=False)]
    for c, v in zip("rgb", tis_rgb.mean(axis=0)):
        qc[f"tissue_{c}_mean"] = round(float(v), 2)
    od = rgb_to_od(tis_rgb)
    C = np.linalg.lstsq(REF_HE, od.T, rcond=None)[0]
    h, e = C[0], C[1]
    h_lo, h_p50, h_hi = np.percentile(h, [1, 50, 99])
    e_p50, e_p99 = np.percentile(e, [50, 99])
    qc.update(h_tissue_p1=round(float(h_lo), 4),
              h_tissue_p50=round(float(h_p50), 4),
              h_tissue_p99=round(float(h_hi), 4),
              e_tissue_p50=round(float(e_p50), 4),
              e_tissue_p99=round(float(e_p99), 4),
              he_ratio=round(float(h_hi / (e_p99 + 1e-6)), 4))
    if h_hi - h_lo < 1e-3:
        return None, None, qc
    return float(h_lo), float(h_hi), qc


def process_slide(gpath, out_dir, check=False, seed=0):
    g = load_graph_geometry(gpath)
    iid = g["image_id"]
    det = DET_DIR / iid / "cell_detection.json"
    tif = slide_path(iid)
    if not det.exists():
        return None, f"missing cell_detection.json"
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
    coverage = float(on_node.mean())

    sl = open_slide(tif)
    try:
        w0, h0 = sl.level_dimensions[0]
        if (h0, w0) != g["wsi_shape"]:
            return None, (f"dim mismatch: slide (h,w)=({h0},{w0}) vs graph "
                          f"{g['wsi_shape']}")

        # group cells by node
        order = np.argsort(node_id[on_node], kind="stable")
        ni = node_id[on_node][order]
        src = np.flatnonzero(on_node)[order]
        bounds = np.searchsorted(ni, np.arange(g["n_nodes"] + 1))

        counts = np.diff(bounds)
        populated = np.flatnonzero(counts >= MIN_ALL_CHROM)
        origins = [(int(g["grid_rc"][n, 1] * tile_px),
                    int(g["grid_rc"][n, 0] * tile_px)) for n in populated]
        if not origins:
            return None, "no populated tiles"

        h_lo, h_hi, qc = fit_slide_stain(sl, origins, tile_px, rng)
        if h_lo is None:
            return None, f"stain model failed (bg_frac={qc['bg_frac']:.2f})"

        if check:
            qc.update(image_id=iid, label=g["label"], n_cells=len(cx),
                      coverage=round(coverage, 4), n_tiles=g["n_nodes"],
                      n_populated=len(populated))
            # measure a handful of tiles so the sanity numbers are real
            demo = []
            for n in populated[:: max(1, len(populated) // 8)][:8]:
                row = measure_tile(sl, g, n, src, bounds, cx, cy, ct, bb,
                                   h_lo, h_hi, tile_px, rng)
                if row is not None:
                    demo.append(row)
            if demo:
                for k, col in (("neo_h_mean", "neo_h_mean_mean"),
                               ("neo_h_std", "neo_h_std_mean"),
                               ("neo_h_entropy", "neo_h_entropy_mean"),
                               ("neo_glcm_contrast", "neo_glcm_contrast_mean"),
                               ("neo_ring_diff", "neo_h_ring_diff_mean")):
                    qc[k] = round(float(np.mean([d[col] for d in demo])), 4)
                qc["demo_tiles"] = len(demo)
            return None, qc

        feats = np.zeros((g["n_nodes"], len(FEATURE_NAMES)), np.float32)
        n_meas = 0
        for n in populated:
            row = measure_tile(sl, g, n, src, bounds, cx, cy, ct, bb,
                               h_lo, h_hi, tile_px, rng)
            if row is None:
                continue
            feats[n] = [row[k] for k in FEATURE_NAMES]
            n_meas += 1
    finally:
        sl.close()

    qc.update(image_id=iid, label=g["label"], n_cells=len(cx),
              coverage=round(coverage, 4), n_tiles=g["n_nodes"],
              n_populated=len(populated), n_measured=n_meas)
    payload = dict(
        image_id=np.array([iid]), label=np.array([g["label"]]),
        chrom_features=feats,
        chrom_feature_names=np.array(FEATURE_NAMES, dtype=object),
        chrom_log1p_cols=np.array([n in LOG1P_COLS for n in FEATURE_NAMES]),
        chrom_flag_cols=np.array([n in FLAG_COLS for n in FEATURE_NAMES]),
        grid_shape=np.array(g["grid_shape"]),
        grid_rc_checksum=np.array([int(g["grid_rc"].sum())]),
        stain_HE=REF_HE, stain_h_range=np.array([h_lo, h_hi], np.float32),
        slide_qc=np.array([json.dumps(qc)]),
        params=np.array([json.dumps(dict(
            SAMPLE_PER_TILE=SAMPLE_PER_TILE, MIN_NEO_CHROM=MIN_NEO_CHROM,
            MIN_ALL_CHROM=MIN_ALL_CHROM, ERODE_FRAC=ERODE_FRAC,
            GLCM_LEVELS=GLCM_LEVELS, H_RANGE=list(H_RANGE),
            BG_RGB_THRESH=BG_RGB_THRESH, BLACK_RGB_THRESH=BLACK_RGB_THRESH,
            MIN_TISSUE_FRAC_NUC=MIN_TISSUE_FRAC_NUC,
            stain="fixed REF_HE + per-slide tissue p1/p99"))]),
    )
    return payload, qc


def measure_tile(sl, g, node, src, bounds, cx, cy, ct, bb, h_lo, h_hi,
                 tile_px, rng):
    idx = src[bounds[node]:bounds[node + 1]]
    if len(idx) < MIN_ALL_CHROM:
        return None
    ox = int(g["grid_rc"][node, 1] * tile_px)
    oy = int(g["grid_rc"][node, 0] * tile_px)
    try:
        rgb = read_tile(sl, ox, oy, tile_px)
    except Exception:
        return None
    hraw = deconvolve_h(rgb)
    hmap = normalise_h(hraw, h_lo, h_hi)
    tissue = tissue_mask(rgb)

    out = {k: 0.0 for k in FEATURE_NAMES}
    out["tile_h_mean"] = float(hmap[tissue].mean()) if tissue.any() else 0.0
    out["tile_h_std"] = float(hmap[tissue].std()) if tissue.any() else 0.0
    out["tile_tissue_frac"] = float(tissue.mean())

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
            # bbox -> tile-local, clipped; a nucleus straddling the tile edge is
            # measured on the part inside it rather than dropped
            lx0, ly0 = max(x0 - ox, 0), max(y0 - oy, 0)
            lx1, ly1 = min(x1 - ox, tile_px), min(y1 - oy, tile_px)
            if lx1 - lx0 < 3 or ly1 - ly0 < 3:
                continue
            m = nucleus_metrics(hmap[ly0:ly1, lx0:lx1], hraw[ly0:ly1, lx0:lx1],
                                tissue[ly0:ly1, lx0:lx1])
            if m is not None:
                rows.append(m)
        n = aggregate(rows, group, out)
        out[f"chrom_n_{group}_measured"] = float(n)
    out["neo_chrom_measured"] = 1.0 if out["chrom_n_neo_measured"] >= MIN_NEO_CHROM else 0.0
    out["all_chrom_measured"] = 1.0 if out["chrom_n_all_measured"] >= MIN_ALL_CHROM else 0.0
    return out


def _worker(job):
    gpath, out_dir, check, seed = job
    iid = corpus_paths.image_id_from_graph(gpath)
    t0 = time.time()
    try:
        payload, qc = process_slide(Path(gpath), Path(out_dir), check, seed)
    except Exception as e:
        import traceback
        return dict(image_id=iid, error=f"{type(e).__name__}: {e}"[:300],
                    tb=traceback.format_exc()[-800:], secs=time.time() - t0)
    if isinstance(qc, str):
        return dict(image_id=iid, error=qc, secs=time.time() - t0)
    if payload is not None:
        np.savez_compressed(Path(out_dir) / f"{iid}_chromatin.npz", **payload)
    qc["secs"] = round(time.time() - t0, 1)
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="5 slides: verify dims, stain fit and plausibility, "
                         "write nothing")
    ap.add_argument("--smoke", action="store_true", help="3 slides, writes")
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs = sorted(GRAPH_DIR.glob("*_graph.npz"))
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        graphs = [g for g in graphs if corpus_paths.image_id_from_graph(g) in want]
    if args.check:
        graphs = graphs[:: max(1, len(graphs) // 5)][:5]
    elif args.smoke:
        graphs = graphs[:3]
    if args.limit:
        graphs = graphs[:args.limit]
    if not args.rebuild and not args.check:
        graphs = [g for g in graphs
                  if not (out_dir / f"{corpus_paths.image_id_from_graph(g)}"
                                    f"_chromatin.npz").exists()]

    print(corpus_paths.banner("chromatin"))
    print(f"python   : {sys.version.split()[0]}")
    print(f"features : {len(FEATURE_NAMES)} per tile")
    print(f"slides   : {len(graphs)}   workers={args.workers}   out -> {out_dir}")
    print(f"stain    : Macenko per slide, {STAIN_TILES} tiles, "
          f"{SAMPLE_PER_TILE} nuclei/class/tile\n")
    if not graphs:
        print("nothing to do")
        return

    jobs = [(str(g), str(out_dir), args.check, 0) for g in graphs]
    t0 = time.time()
    results = []
    if args.workers > 1 and not args.check:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, qc in enumerate(pool.imap_unordered(_worker, jobs), 1):
                results.append(qc)
                _log(i, len(jobs), qc, t0)
    else:
        for i, job in enumerate(jobs, 1):
            qc = _worker(job)
            results.append(qc)
            _log(i, len(jobs), qc, t0, check=args.check)

    import pandas as pd
    df = pd.DataFrame(results)
    mf = out_dir / ("_check.csv" if args.check else "_manifest.csv")
    if mf.exists() and not args.check and "image_id" in df:
        # image_id must be compared as a string on BOTH sides: pandas reads it
        # back from CSV as int64, so an untyped .isin() never matches the new
        # rows and stale entries survive -- which made a successful run report
        # "3 slides failed" from the previous run's error rows.
        old = pd.read_csv(mf, dtype={"image_id": str})
        df["image_id"] = df["image_id"].astype(str)
        df = pd.concat([old[~old.image_id.isin(set(df.image_id))], df],
                       ignore_index=True)
    df.to_csv(mf, index=False)
    print(f"\nmanifest -> {mf}")

    if args.check:
        print("\n" + "=" * 74)
        print("CHECK VERDICT")
        print("=" * 74)
        ok = df[df.get("error").isna()] if "error" in df else df
        if not len(ok):
            print("  FAIL -- every slide errored. Fix before the full run.")
            return
        print(f"  coverage (cells landing on a tissue tile): "
              f"min {ok.coverage.min():.3f} median {ok.coverage.median():.3f}")
        print(f"  background fraction: median {ok.bg_frac.median():.2f}")
        if "h_tissue_p99" in ok:
            print(f"  raw tissue haematoxylin p99 across slides: "
                  f"{ok.h_tissue_p99.min():.2f} - {ok.h_tissue_p99.max():.2f}  "
                  f"<- the staining spread the normalisation has to remove")
        if "neo_h_mean" in ok:
            print(f"  neoplastic haematoxylin (tissue-normalised): "
                  f"{ok.neo_h_mean.min():.3f} - {ok.neo_h_mean.max():.3f}")
            print(f"  neoplastic chromatin entropy (bits, max 4): "
                  f"{ok.neo_h_entropy.min():.2f} - {ok.neo_h_entropy.max():.2f}")
            print(f"  neoplastic nucleus-minus-surround haematoxylin: "
                  f"{ok.neo_ring_diff.min():+.3f} - {ok.neo_ring_diff.max():+.3f}")
            print("\n  Three things have to hold before the full run:")
            print("   * nucleus-minus-surround clearly POSITIVE on every slide --")
            print("     nuclei ARE darker than the tissue around them, so the")
            print("     boxes are on nuclei and haematoxylin is the right channel.")
            print("   * normalised haematoxylin in a similar range across slides;")
            print("     a 10x spread would mean staining, not biology.")
            print("   * entropy well away from 0 -- near 0 means every pixel fell")
            print("     in one histogram bin and the texture features are dead.")

    if not args.check and "error" in df:
        bad = df[df.error.notna()]
        if len(bad):
            print(f"\n{len(bad)} slide(s) failed:")
            print(bad[["image_id", "error"]].head(20).to_string(index=False))


def _log(i, n, qc, t0, check=False):
    el = (time.time() - t0) / 60
    if qc.get("error"):
        print(f"[{i}/{n}] {qc['image_id']}  FAILED: {qc['error']}")
        if qc.get("tb"):
            print("    " + qc["tb"].replace("\n", "\n    "))
        return
    if check:
        print(f"[{i}/{n}] {qc['image_id']} {qc.get('label','?'):4s} "
              f"cells={qc['n_cells']:>7} cov={qc['coverage']:.3f} "
              f"tiles={qc['n_populated']}/{qc['n_tiles']} "
              f"bg={qc['bg_frac']:.2f} blur={qc['blur_lapvar']:.0f}\n"
              f"        h_tissue=[{qc.get('h_tissue_p1')}, "
              f"{qc.get('h_tissue_p50')}, {qc.get('h_tissue_p99')}] "
              f"he_ratio={qc.get('he_ratio')}\n"
              f"        neo_h={qc.get('neo_h_mean')} sd={qc.get('neo_h_std')} "
              f"entropy={qc.get('neo_h_entropy')} "
              f"glcm_contrast={qc.get('neo_glcm_contrast')} "
              f"ring={qc.get('neo_ring_ratio')}  ({qc['secs']:.0f}s)")
    else:
        print(f"[{i}/{n}] {qc['image_id']} {qc.get('label','?'):4s} "
              f"tiles={qc.get('n_measured')}/{qc.get('n_tiles')} "
              f"cov={qc['coverage']:.3f} ({qc['secs']:.0f}s, {el:.0f}m elapsed)")


if __name__ == "__main__":
    main()

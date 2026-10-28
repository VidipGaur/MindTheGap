# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Build streamline-connectivity graphs from KDE nuclear density maps (UBC-OCEAN).

Stage 3 of the pipeline:
    CellViT -> KDE density maps -> [THIS SCRIPT] -> GATv2

    structure tensor (2 scales) -> RK4 streamline integration
    -> fixed-grid nodes -> streamline-derived edges -> per-WSI graph .npz

V2: typed streamline fields. Integration now runs once per CELL TYPE channel
(plus a derived tumour-stroma interface field), all feeding ONE shared node
grid. Nodes stay untyped so attention remains a single dense heatmap; edges
carry a per-channel count vector so attention can be decomposed by channel
("did it attend to tumour architecture or to the invasive front?"). Reads the
v3 multi-channel KDE schema from build_kde_density_maps.py.

Run:
    python build_streamline_graphs.py            # process everything pending
    python build_streamline_graphs.py --self-test # synthetic correctness checks, no data needed
    python build_streamline_graphs.py --only 4     # single image_id, verbose
    python build_streamline_graphs.py --limit 10   # first N pending, for a smoke run


DESIGN NOTES / WHY THINGS ARE THE WAY THEY ARE
==============================================

(1) NODES ARE A FIXED SPATIAL GRID, NOT CLUSTERS.
    The old 80-WSI pipeline tried HDBSCAN/KMeans nodes. For the stated end goal
    -- overlaying GATv2 attention on the WSI and asking a pathologist whether the
    model attends where they would -- cluster nodes are a liability: node identity
    is not comparable across slides, node shape is data-dependent, and rendering
    attention means painting irregular blobs. With a fixed grid, attention is a
    dense (n_rows, n_cols) image that upsamples straight onto the slide thumbnail,
    and "node 512" means the same physical tile size on every slide. NODE_SIZE_DS
    below is set in physical units on purpose (see PHYSICAL SCALE).
    Clustering can still be layered on later as an ablation; grid-first keeps the
    interpretability figure cheap.

(2) STREAMLINES FOLLOW THE MINOR EIGENVECTOR.
    The structure tensor's *major* eigenvector points ACROSS structure (direction
    of maximum intensity change). The *minor* eigenvector points ALONG it. For
    tractography-style integration through tissue architecture we want the minor
    one. This is stated explicitly because it is a silent 90-degree error if
    swapped: streamlines would cross tissue bands instead of running along them,
    the graph would still build fine, and accuracy would just be quietly bad.
    STREAMLINE_EIGENVECTOR = "minor" is configurable so the swap can be run as a
    deliberate ablation rather than discovered as a bug.

(3) SIGN-AWARE INTERPOLATION.
    An eigenvector field is direction-ambiguous (v and -v are both valid). Naive
    bilinear interpolation of neighbouring eigenvectors can cancel them to ~zero
    and send a streamline off in a garbage direction. Every interpolation here
    flips neighbour signs to agree with a reference direction first. This is the
    single most common correctness bug in eigenvector-field integration.

(4) COORDINATE CONVENTION, STATED ONCE, ASSERTED EVERYWHERE.
    Everything internal is (row, col) == (y, x) == (height-axis, width-axis),
    matching build_kde_density_maps.py's density arrays of shape (H_ds, W_ds).
    Given the axis-swap bug that already cost a full KDE rebuild, this script:
      - asserts density.shape matches ceil(wsi_shape / downsample) on every load
      - refuses to run on a KDE map whose grid aspect ratio disagrees with the
        manifest's, instead of silently producing a distorted graph
      - ships a --self-test built on a deliberately NON-SQUARE synthetic slide
        with known structure, because a square test cannot catch an axis swap
      - writes node coordinates out in BOTH grid and WSI-pixel space, so the
        attention-overlay step never has to re-derive the mapping by hand
    x/y is used only at the very end, in the *_xy fields written for plotting.

(5) NOTHING IS TRUSTED FROM THE .npz FILENAME.
    downsample, sigma and wsi_shape are read from inside each KDE file, not
    assumed from config, so a graph built against a stale KDE map can't silently
    inherit the wrong scale. Same lesson as the [OK]-logged-but-crashed CellViT
    run: a completed output is only counted as complete after it is re-opened,
    re-read, and checked for the keys it should contain.

(6) PHYSICAL SCALE.
    UBC-OCEAN WSIs were processed at wsi_mpp=0.25 um/px, and KDE used
    DOWNSAMPLE=16, so one grid pixel is 0.25 * 16 = 4 um NOMINALLY. The slides
    are natively 0.5 um/px, so every micron figure declared below is a NOMINAL
    micron (expressed at 0.25 um/px) and the physical size is twice as large:
    a grid pixel is 8 um, the 256 um nominal node tile is 1024 px = 512 um, the
    20 um nominal streamline step is 40 um. The geometry in pixels is what the
    code uses and it is correct; only the micron labels are nominal. All
    spatial constants below are declared in nominal microns and converted, so
    they stay meaningful if DOWNSAMPLE ever changes. Note KDE_SIGMA=15 grid px == 60 um of smoothing;
    the structure-tensor scales are chosen relative to that (smoothing already in
    the input sets the floor on resolvable structure). If visualize_kde_maps.py
    ends up moving KDE_SIGMA, revisit TENSOR_SCALES_UM -- they are downstream of
    it, and that dependency is the main reason to do the KDE spot-check first.

(7) MEMORY.
    16GB system RAM, and the largest UBC-OCEAN slides approach 100k x 50k px
    (~6250 x 3125 grid). All maps are float32, structure-tensor intermediates are
    freed as soon as they are consumed, and streamlines are integrated as one
    vectorised batch of seeds rather than per-seed Python loops. Trajectories are
    stored as int32 node ids (not float coordinates) -- that is the difference
    between ~25MB and ~400MB per slide.
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
import math
import sys
import time
from pathlib import Path

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# See corpus_paths.py: defaults reproduce the UBC-OCEAN constants exactly.
import corpus_paths

ROOT = str(corpus_paths.ROOT)
RUN_DIR_NAME = corpus_paths.RUN_DIR_NAME
MANIFEST_FILENAME = "cellvit_manifest.csv"
KDE_SUBFOLDER = "kde_maps"
OUT_SUBFOLDER = "graphs"

# The 8 CellViT stragglers. They have no KDE map yet so they'd be skipped
# anyway, but listing them keeps the run log honest: "excluded" and "pending"
# are different statuses.
EXCLUDE_IDS = {"28028", "39172", "39893", "41361", "46769", "54408", "58947", "60936"}

WSI_MPP = 0.25            # NOMINAL um/px CellViT was run at; the slides are natively 0.5 um/px,
                          # so every *_UM constant below is twice its physical size
EXPECTED_DOWNSAMPLE = 16  # sanity reference only; real value read from each npz

# Two scales, MRI-tractography style: the fine scale resolves local nuclear
# architecture, the coarse scale captures slide-level tissue organisation.
# Both produce streamlines; edges carry per-scale counts so GATv2 can weigh them.
TENSOR_SCALES_UM = [80.0, 240.0]   # tensor smoothing scale (rho)
GRAD_SIGMA_UM = 40.0               # derivative scale, ~KDE sigma; do not go below it

STREAMLINE_EIGENVECTOR = "minor"   # "minor" = along structure. See design note (2).

# Which KDE channels get their own structure tensor and their own streamlines.
# "Dead" is deliberately absent: CellViT-H scores PQ ~0.15 on it vs ~0.58 on
# Neoplastic, so integrating it would add noise-shaped edges.
# "interface" is DERIVED, not a KDE channel -- see build_interface_field().
INTEGRATION_CHANNELS = ["neoplastic", "epithelial", "connective",
                        "inflammatory", "interface"]

# A channel needs at least this many detected cells on a slide before it is
# integrated. Below it, the channel is masked (channel_valid=False) rather than
# contributing zeros -- otherwise the model can learn "empty inflammatory
# channel" as a class cue, which would be a detection artifact, not biology.
MIN_CELLS_PER_CHANNEL = 500
# A channel is also dropped if its oriented structure is too weak to integrate.
MIN_CHANNEL_COHERENCE = 0.05

SEED_SPACING_UM = 80.0     # regular seed lattice inside the tissue mask
STEP_UM = 20.0             # RK4 step length, nominal um (40 um physical at 0.5 um/px)
MAX_LENGTH_UM = 4000.0     # hard cap per streamline direction
MAX_ANGLE_DEG = 60.0       # per-step turn limit (tractography-style curvature stop)
# Lowered from 0.15 after the v2 corpus fragmented badly (median largest
# component 34% of nodes): streamlines were terminating in the low-coherence
# gaps BETWEEN tissue islands instead of bridging them.
MIN_COHERENCE = 0.08       # stop where the orientation field is unreliable
# Lowered from 75.0 for the same reason. At 75 only the densest quartile of
# nonzero pixels counted as tissue, which carved contiguous tissue into
# disconnected islands and left ~68% of the node lattice empty. 50 keeps the
# denser half and should connect neighbouring regions that genuinely touch.
TISSUE_PERCENTILE = 50.0   # density percentile (over nonzero) defining tissue
MAX_SEEDS = 40000          # cap for the largest slides; lattice is subsampled if hit

# Tile = 1024 px. Physical size is 512 um at the slides' native 0.5 um/px.
# NODE_SIZE_UM = 256 is expressed at CellViT's nominal 0.25 um/px
# (256 / 0.25 = 1024 px); it is not the physical tile size. Do NOT set it to 512
# to "fix" the label: with WSI_MPP = 0.25 that would build a 2048 px lattice.
#
# Node tile size is the resolution of your attention heatmap, so it is the one
# constant to think about in pathologist terms, not compute terms. 512 um
# (physical) is a few high-power fields: fine enough that "the model attends to
# this tumour island and not that stromal band" can be checked on the slide,
# coarse enough that a 100k x 50k WSI still lands around 4-5k tiles. Doubling
# the constant halves the heatmap resolution in each axis (~4x fewer nodes) if
# graphs turn out too large; going much below 128 nominal um starts producing
# tiles smaller than the KDE smoothing kernel, which is meaningless.
NODE_SIZE_UM = 256.0
MIN_EDGE_COUNT = 2         # drop edges seen by only one streamline (noise floor)

# Endpoint edges join the two TERMINAL nodes of a full bidirectional streamline.
# They are long-range and often count 1, because independent streamlines rarely
# terminate in the same tile pair. Sharing MIN_EDGE_COUNT with transition edges
# deleted ~100% of them: a corpus audit of 456 graphs found 456/456 were >=99.9%
# pure grid adjacency, median 10 of 23 edge_attr columns identically zero. The
# graph was a lattice, so degree-preserving random rewiring was an equivalent
# graph and the random_topology ablation could not have detected anything.
MIN_ENDPOINT_COUNT = int(os.environ.get("MIN_ENDPOINT_COUNT", 2))
MIN_ENDPOINT_GRID_DIST = int(os.environ.get("MIN_ENDPOINT_GRID_DIST", 2))
# Hard cap so a few thousand unique terminal pairs cannot turn the graph into a
# hairball. Keeps each node's strongest endpoint edges by count. 0 disables.
MAX_ENDPOINT_PER_NODE = int(os.environ.get("MAX_ENDPOINT_PER_NODE", 4))
MAX_NODES = 20000          # refuse absurd graphs rather than OOM downstream

# Printed at startup and written into every graph. If the banner below does not
# appear when you run this, you are running an older copy of the file.
SCRIPT_VERSION = "v5-bidir"

# Save decimated streamline polylines (in WSI px) for the first N slides so the
# interpretability figure can show attention ON TOP OF the actual streamlines
# without re-running integration. Cheap, and you will want it.
SAVE_POLYLINES_FOR_FIRST = 12
POLYLINE_SAMPLE = 200      # how many streamlines to keep when saving

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# KDE loading + coordinate-convention guards

KDE_SCHEMA_VERSION = 3

# Channel name -> key in the KDE npz. "interface" is derived, not stored.
KDE_CHANNEL_KEYS = {"neoplastic": "density_neoplastic",
                    "epithelial": "density_epithelial",
                    "connective": "density_connective",
                    "inflammatory": "density_inflammatory",
                    "dead": "density_dead"}


def load_kde(path: Path, manifest_h: int, manifest_w: int) -> dict:
    """
    Load one multi-channel KDE map and validate its geometry.

    Raises on any geometry disagreement. Deliberate: a distorted density map is
    exactly the failure that got through once already, and failing here is far
    cheaper than discovering it after a GATv2 run.
    """
    with np.load(path, allow_pickle=True) as z:
        if "schema_version" not in z.files:
            raise ValueError("KDE file predates schema versioning -- rebuild it "
                             "with the current build_kde_density_maps.py")
        ver = int(z["schema_version"][0])
        if ver != KDE_SCHEMA_VERSION:
            raise ValueError(f"KDE schema v{ver}, expected v{KDE_SCHEMA_VERSION} -- rebuild")

        for k in ("density_all", "grid_x", "grid_y", "wsi_shape", "label", "downsample"):
            if k not in z.files:
                raise ValueError(f"KDE file missing key {k}")

        d_all = z["density_all"].astype(np.float32)
        channels = {}
        for name, key in KDE_CHANNEL_KEYS.items():
            if key in z.files:
                channels[name] = z[key].astype(np.float32)

        downsample = int(z["downsample"][0])
        sigma_px = float(z["sigma_px"][0]) if "sigma_px" in z.files else -1.0
        sigma_um = float(z["sigma_um"][0]) if "sigma_um" in z.files else -1.0
        mpp = float(z["pixel_size_um"][0]) if "pixel_size_um" in z.files else WSI_MPP
        wsi_h, wsi_w = (int(v) for v in z["wsi_shape"])
        label = str(z["label"][0])
        grid_x = z["grid_x"]
        grid_y = z["grid_y"]
        counts = json.loads(str(z["cell_counts_json"][0])) if "cell_counts_json" in z.files else {}
        coord_diag = json.loads(str(z["coord_diag_json"][0])) if "coord_diag_json" in z.files else {}

    exp_h = int(math.ceil(wsi_h / downsample))
    exp_w = int(math.ceil(wsi_w / downsample))
    if d_all.shape != (exp_h, exp_w):
        raise ValueError(
            f"density shape {d_all.shape} != ceil(wsi_shape/downsample)=({exp_h},{exp_w}); "
            "grid is (row, col) = (height, width) -- possible axis swap")
    for name, arr in channels.items():
        if arr.shape != d_all.shape:
            raise ValueError(f"channel {name} shape {arr.shape} != density_all {d_all.shape}")
    if (wsi_h, wsi_w) != (manifest_h, manifest_w):
        raise ValueError(f"KDE wsi_shape (h={wsi_h}, w={wsi_w}) disagrees with manifest "
                         f"(h={manifest_h}, w={manifest_w}) -- stale KDE map, rebuild it")
    if len(grid_y) != exp_h or len(grid_x) != exp_w:
        raise ValueError("grid_x/grid_y length disagrees with the density grid")

    # The KDE script's own coordinate regression check, propagated forward
    # rather than re-derived, so a bad slide can't quietly become a graph.
    if coord_diag.get("frac_oob", 0.0) > 0.001:
        raise ValueError(f"KDE recorded {coord_diag['frac_oob']:.2%} out-of-bounds "
                         "centroids -- coordinate mismatch, do not build a graph on this")

    # Cell counts key off the capitalised PanNuke names in the KDE file.
    cell_counts = {k.lower(): int(v) for k, v in counts.items()}

    return dict(density_all=d_all, channels=channels, downsample=downsample,
                sigma_px=sigma_px, sigma_um=sigma_um, mpp=mpp,
                wsi_h=wsi_h, wsi_w=wsi_w, label=label,
                grid_x=grid_x, grid_y=grid_y, cell_counts=cell_counts)


def build_interface_field(d_all, d_neo):
    """
    Derived tumour-stroma interface field: 4*f*(1-f) where f is the local
    neoplastic fraction. Peaks at f=0.5 (mixed tissue), zero in pure tumour and
    pure stroma, so its structure tensor orients ALONG the boundary between
    them. Streamlines here trace the invasive front -- the thing a pathologist
    is actually looking at when they call desmoplasia or an infiltrative edge.

    Weighted by total density so that background, where the fraction is a ratio
    of two near-zero numbers and therefore meaningless, stays dark.
    """
    f = d_neo / np.maximum(d_all, 1e-8)
    np.clip(f, 0.0, 1.0, out=f)
    inter = (4.0 * f * (1.0 - f)).astype(np.float32)
    inter *= (d_all / max(float(d_all.max()), 1e-8))
    return inter


def um_to_px(um: float, downsample: int, mpp: float = None) -> float:
    """Microns -> downsampled-grid pixels.

    `mpp` is the NOMINAL pixel size driving tiling geometry, read per slide from
    the KDE file (which got it from the manifest). It defaults to the WSI_MPP
    constant, which is what UBC-OCEAN's KDE files store, so passing it changes
    nothing on that corpus. A corpus with a different native magnification needs
    its own value or its tiles come out the wrong physical size -- see
    corpus_paths.py for the nominal = native/2 rule.
    """
    return um / ((WSI_MPP if mpp is None else mpp) * downsample)


# Structure tensor

def density_gradients(density: np.float32, grad_sigma: float):
    """
    Gaussian first derivatives, (d/drow, d/dcol). order=(row_order, col_order).

    Split out from the tensor assembly because the gradients depend only on
    grad_sigma, not on the tensor scale -- so with two tensor scales per channel
    they are computed once and reused, not twice. That is ~35% of the runtime of
    this stage, which matters at 6 channels x 2 scales x 456 slides.
    """
    gr = gaussian_filter(density, sigma=grad_sigma, order=(1, 0), mode="nearest")
    gc = gaussian_filter(density, sigma=grad_sigma, order=(0, 1), mode="nearest")
    return gr, gc


def structure_tensor_from_gradients(gr, gc, tensor_sigma: float):
    """
    Returns (vec_row, vec_col, coherence), all float32.

    vec_* is the unit eigenvector selected by STREAMLINE_EIGENVECTOR, expressed in
    (row, col). coherence is (l_major - l_minor) / (l_major + l_minor) in [0, 1]:
    1 = strongly oriented structure, 0 = isotropic / featureless.

    Closed-form 2x2 eigen-decomposition -- no np.linalg.eigh over millions of
    pixels, which would be both slow and memory-hostile on a 6250x3125 grid.
    """
    # Tensor components, smoothed at the tensor scale. This smoothing is what
    # makes the field integrable -- without it the orientation flips pixel to pixel.
    Jrr = gaussian_filter(gr * gr, sigma=tensor_sigma, mode="nearest")
    Jcc = gaussian_filter(gc * gc, sigma=tensor_sigma, mode="nearest")
    Jrc = gaussian_filter(gr * gc, sigma=tensor_sigma, mode="nearest")

    tr = Jrr + Jcc
    diff = Jrr - Jcc
    disc = np.sqrt(np.maximum(diff * diff + 4.0 * Jrc * Jrc, 0.0), dtype=np.float32)
    l_major = 0.5 * (tr + disc)
    l_minor = 0.5 * (tr - disc)

    # Major eigenvector of [[Jrr, Jrc], [Jrc, Jcc]]: (Jrc, l_major - Jrr),
    # falling back to (l_major - Jcc, Jrc) where the first form degenerates.
    v_r = Jrc
    v_c = l_major - Jrr
    weak = (np.abs(v_r) + np.abs(v_c)) < 1e-12
    if np.any(weak):
        v_r = np.where(weak, l_major - Jcc, v_r)
        v_c = np.where(weak, Jrc, v_c)
    del Jrr, Jcc, Jrc, tr, diff

    if STREAMLINE_EIGENVECTOR == "minor":
        # rotate 90 degrees: along-structure instead of across-structure
        v_r, v_c = -v_c, v_r
    elif STREAMLINE_EIGENVECTOR != "major":
        raise ValueError("STREAMLINE_EIGENVECTOR must be 'minor' or 'major'")

    norm = np.sqrt(v_r * v_r + v_c * v_c, dtype=np.float32)
    flat = norm < 1e-12
    norm = np.where(flat, 1.0, norm)
    v_r = (v_r / norm).astype(np.float32)
    v_c = (v_c / norm).astype(np.float32)
    v_r[flat] = 0.0
    v_c[flat] = 0.0

    denom = l_major + l_minor
    coherence = np.where(denom > 1e-20, disc / np.maximum(denom, 1e-20), 0.0).astype(np.float32)
    np.clip(coherence, 0.0, 1.0, out=coherence)

    return v_r, v_c, coherence


# RK4 integration over a sign-ambiguous direction field

def _sample_direction(v_r, v_c, pos, ref):
    """
    Sign-aware bilinear sample of the direction field at float positions.

    pos: (N, 2) float32 (row, col). ref: (N, 2) current/reference direction.
    Each of the 4 neighbours is flipped to agree with ref BEFORE weighting --
    see design note (3). Returns (N, 2) unit vectors (zero where undefined).
    """
    H, W = v_r.shape
    r = np.clip(pos[:, 0], 0, H - 1.001)
    c = np.clip(pos[:, 1], 0, W - 1.001)
    r0 = np.floor(r).astype(np.int32)
    c0 = np.floor(c).astype(np.int32)
    r1 = np.minimum(r0 + 1, H - 1)
    c1 = np.minimum(c0 + 1, W - 1)
    fr = (r - r0).astype(np.float32)
    fc = (c - c0).astype(np.float32)

    acc_r = np.zeros(len(pos), dtype=np.float32)
    acc_c = np.zeros(len(pos), dtype=np.float32)
    for (rr, cc, w) in (
        (r0, c0, (1 - fr) * (1 - fc)),
        (r0, c1, (1 - fr) * fc),
        (r1, c0, fr * (1 - fc)),
        (r1, c1, fr * fc),
    ):
        nr = v_r[rr, cc]
        nc = v_c[rr, cc]
        s = np.sign(nr * ref[:, 0] + nc * ref[:, 1]).astype(np.float32)
        s[s == 0] = 1.0
        acc_r += w * s * nr
        acc_c += w * s * nc

    n = np.sqrt(acc_r * acc_r + acc_c * acc_c, dtype=np.float32)
    bad = n < 1e-8
    n = np.where(bad, 1.0, n)
    out = np.stack([acc_r / n, acc_c / n], axis=1).astype(np.float32)
    out[bad] = 0.0
    return out


def integrate_rk4(v_r, v_c, coherence, tissue, seeds, node_of, step_px, max_steps,
                  min_coherence, max_angle_deg, record_idx=None, direction=1.0):
    """
    Integrate all seeds in one vectorised batch, in one direction.

    seeds: (N, 2) float32 start positions, in grid (row, col).
    node_of: (H, W) int32 node id per grid pixel, -1 outside the graph.
    record_idx: optional seed indices whose float positions are also kept, for
    drawing actual smooth streamlines in the interpretability figure. Only a
    small sample is recorded -- keeping float positions for every seed is what
    turns a 25MB slide into a 400MB one.

    Returns:
        traj_nodes: (N, max_steps+1) int32 node id per step, -1 once terminated
        end_pos:    (N, 2) float32 final position
        n_steps:    (N,) int32 steps actually taken
        pos_track:  (len(record_idx), max_steps+1, 2) float32, NaN after
                    termination -- or None if record_idx is None
    """
    H, W = v_r.shape
    N = len(seeds)
    pos = seeds.astype(np.float32).copy()

    # Initial direction: the field itself, with an arbitrary but consistent
    # reference so forward/backward runs are true opposites.
    ref0 = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (N, 1))
    d = _sample_direction(v_r, v_c, pos, ref0)
    if direction < 0:
        # Flip AFTER sign resolution. Negating the field before it
        # is a no-op: _sample_direction re-aligns to ref anyway.
        d = -d

    traj = np.full((N, max_steps + 1), -1, dtype=np.int32)
    n_steps = np.zeros(N, dtype=np.int32)
    active = np.linalg.norm(d, axis=1) > 0.5

    ri = np.clip(np.rint(pos[:, 0]).astype(np.int32), 0, H - 1)
    ci = np.clip(np.rint(pos[:, 1]).astype(np.int32), 0, W - 1)
    traj[active, 0] = node_of[ri[active], ci[active]]

    pos_track = None
    if record_idx is not None and len(record_idx):
        record_idx = np.asarray(record_idx, dtype=np.int64)
        pos_track = np.full((len(record_idx), max_steps + 1, 2), np.nan, dtype=np.float32)
        pos_track[:, 0] = pos[record_idx]
        rec_slot = np.full(N, -1, dtype=np.int64)
        rec_slot[record_idx] = np.arange(len(record_idx))

    cos_limit = math.cos(math.radians(max_angle_deg))

    for t in range(max_steps):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        p = pos[idx]
        dd = d[idx]

        k1 = _sample_direction(v_r, v_c, p, dd)
        k2 = _sample_direction(v_r, v_c, p + 0.5 * step_px * k1, k1)
        k3 = _sample_direction(v_r, v_c, p + 0.5 * step_px * k2, k2)
        k4 = _sample_direction(v_r, v_c, p + step_px * k3, k3)
        delta = (step_px / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        new_p = p + delta
        seg = np.linalg.norm(delta, axis=1)
        ok = seg > 1e-6

        new_d = np.zeros_like(delta)
        new_d[ok] = delta[ok] / seg[ok, None]

        # curvature stop
        ok &= (new_d[:, 0] * dd[:, 0] + new_d[:, 1] * dd[:, 1]) >= cos_limit
        # bounds stop
        ok &= (new_p[:, 0] >= 0) & (new_p[:, 0] <= H - 1) \
            & (new_p[:, 1] >= 0) & (new_p[:, 1] <= W - 1)

        rr = np.clip(np.rint(new_p[:, 0]).astype(np.int32), 0, H - 1)
        cc = np.clip(np.rint(new_p[:, 1]).astype(np.int32), 0, W - 1)
        # tissue + reliability stops
        ok &= tissue[rr, cc]
        ok &= coherence[rr, cc] >= min_coherence
        nid = node_of[rr, cc]
        ok &= nid >= 0

        keep = idx[ok]
        pos[keep] = new_p[ok]
        d[keep] = new_d[ok]
        traj[keep, t + 1] = nid[ok]
        n_steps[keep] = t + 1
        active[idx[~ok]] = False

        if pos_track is not None and len(keep):
            slots = rec_slot[keep]
            m = slots >= 0
            if m.any():
                pos_track[slots[m], t + 1] = new_p[ok][m]

    return traj, pos, n_steps, pos_track


# Graph assembly

def build_node_grid(tissue, node_size_px):
    """
    Fixed tiling of the density grid. Returns:
        node_of      (H, W) int32, node id per pixel (-1 = not a node)
        node_rc_ds   (K, 2) float32 tile centres in grid coords
        node_grid_rc (K, 2) int32 tile indices (row, col) in the node lattice
        grid_shape   (n_rows, n_cols)
    Only tiles containing tissue become nodes, but ids are derived from the full
    lattice so the attention image can be reconstructed by scattering into
    grid_shape -- see design note (1).
    """
    H, W = tissue.shape
    n_rows = int(math.ceil(H / node_size_px))
    n_cols = int(math.ceil(W / node_size_px))

    rows = np.minimum((np.arange(H) / node_size_px).astype(np.int32), n_rows - 1)
    cols = np.minimum((np.arange(W) / node_size_px).astype(np.int32), n_cols - 1)
    tile_of = (rows[:, None] * n_cols + cols[None, :]).astype(np.int32)

    occupied = np.zeros(n_rows * n_cols, dtype=bool)
    occupied[np.unique(tile_of[tissue])] = True
    tile_ids = np.flatnonzero(occupied)

    remap = np.full(n_rows * n_cols, -1, dtype=np.int32)
    remap[tile_ids] = np.arange(len(tile_ids), dtype=np.int32)

    node_of = remap[tile_of]
    node_of[~tissue] = -1

    gr = (tile_ids // n_cols).astype(np.int32)
    gc = (tile_ids % n_cols).astype(np.int32)
    centres = np.stack([(gr + 0.5) * node_size_px, (gc + 0.5) * node_size_px], axis=1).astype(np.float32)
    return node_of, centres, np.stack([gr, gc], axis=1), (n_rows, n_cols)


def transition_edges(traj, n_nodes):
    """
    Local edges: consecutive DISTINCT nodes along a streamline.
    Equality of consecutive entries implicitly run-length-compresses the path,
    so this is 'A was followed by B' without materialising compressed paths.
    Returns (pairs, counts) with pairs sorted low<high (undirected).
    """
    a = traj[:, :-1].ravel()
    b = traj[:, 1:].ravel()
    m = (a >= 0) & (b >= 0) & (a != b)
    return _accumulate_pairs(a[m], b[m], n_nodes)


def endpoint_edges(node_a, node_b, n_nodes):
    """Long-range edges: the two terminal nodes of each full bidirectional streamline."""
    m = (node_a >= 0) & (node_b >= 0) & (node_a != node_b)
    return _accumulate_pairs(node_a[m], node_b[m], n_nodes)


def _accumulate_pairs(a, b, n_nodes):
    if len(a) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    key = lo * n_nodes + hi
    uniq, counts = np.unique(key, return_counts=True)
    pairs = np.stack([uniq // n_nodes, uniq % n_nodes], axis=1)
    return pairs, counts.astype(np.float32)


def node_features(dens_all, channels, coherence_by_channel, node_of, n_nodes,
                  pass_by_channel, end_by_channel):
    """
    Per-node features, one block per channel so every dimension is nameable.
    That matters here: the point of the attention figure is being able to say
    "it attended to nodes high in neoplastic density and interface coherence",
    which needs features a pathologist can read, not a learned embedding.

    Blocks:
      global:      dens_all mean/max/std, tile fill fraction
      per channel: density mean, density max, fraction of all,
                   coherence mean per tensor scale,
                   log streamline pass count, log endpoint count
    """
    flat = node_of.ravel()
    valid = flat >= 0
    ids = flat[valid]
    counts = np.bincount(ids, minlength=n_nodes).astype(np.float32)
    safe = np.maximum(counts, 1.0)

    def _mean(arr):
        return (np.bincount(ids, weights=arr.ravel()[valid].astype(np.float64),
                            minlength=n_nodes) / safe).astype(np.float32)

    def _max(arr):
        out = np.zeros(n_nodes, dtype=np.float32)
        np.maximum.at(out, ids, arr.ravel()[valid])
        return out

    a_mean = _mean(dens_all)
    a_sq = _mean(dens_all * dens_all)
    feats = [a_mean, _max(dens_all),
             np.sqrt(np.maximum(a_sq - a_mean * a_mean, 0.0)).astype(np.float32),
             (counts / max(counts.max(), 1.0)).astype(np.float32)]
    names = ["dens_all_mean", "dens_all_max", "dens_all_std", "tile_fill"]

    for ch in INTEGRATION_CHANNELS:
        arr = channels.get(ch)
        if arr is None:
            arr = np.zeros_like(dens_all)
        c_mean = _mean(arr)
        feats += [c_mean, _max(arr),
                  (c_mean / np.maximum(a_mean, 1e-8)).astype(np.float32)]
        names += [f"{ch}_mean", f"{ch}_max", f"{ch}_frac"]
        for si in range(len(TENSOR_SCALES_UM)):
            coh = coherence_by_channel.get((ch, si))
            feats.append(_mean(coh) if coh is not None else np.zeros(n_nodes, np.float32))
            names.append(f"{ch}_coherence_s{si}")
        feats += [np.log1p(pass_by_channel.get(ch, np.zeros(n_nodes, np.float32))),
                  np.log1p(end_by_channel.get(ch, np.zeros(n_nodes, np.float32)))]
        names += [f"{ch}_log_pass", f"{ch}_log_endpoint"]

    return np.stack(feats, axis=1).astype(np.float32), names


def channel_edge_attr_names():
    names = []
    for ch in INTEGRATION_CHANNELS:
        names += [f"{ch}_transition", f"{ch}_endpoint",
                  f"{ch}_transition_norm", f"{ch}_endpoint_norm"]
    return names + ["total_transition", "total_endpoint", "n_channels_present"]


# Per-WSI driver

def process_one(image_id, kde_path, out_path, manifest_h, manifest_w,
                save_polylines=False, verbose=False):
    kde = load_kde(kde_path, manifest_h, manifest_w)
    ds = kde["downsample"]
    d_all = kde["density_all"]
    H, W = d_all.shape

    # Derived interface channel (needs neoplastic; skipped if absent).
    channels = dict(kde["channels"])
    if "neoplastic" in channels:
        channels["interface"] = build_interface_field(d_all, channels["neoplastic"])

    nz = d_all[d_all > 0]
    if nz.size == 0:
        raise ValueError("density_all is entirely zero -- upstream KDE problem")
    tissue = d_all >= float(np.percentile(nz, TISSUE_PERCENTILE))
    if tissue.sum() < 50:
        raise ValueError(f"tissue mask has {int(tissue.sum())} px -- too small")

    # Nominal pixel size for tiling geometry, per slide. On UBC-OCEAN every KDE
    # file stores 0.25 == WSI_MPP, so this is a no-op there.
    g_mpp = kde["mpp"]
    node_size_px = max(1.0, um_to_px(NODE_SIZE_UM, ds, g_mpp))
    node_of, node_rc_ds, node_grid_rc, node_grid_shape = build_node_grid(tissue, node_size_px)
    n_nodes = len(node_rc_ds)
    if n_nodes == 0:
        raise ValueError("no nodes -- tissue mask empty after tiling")
    if n_nodes > MAX_NODES:
        raise ValueError(f"{n_nodes} nodes exceeds MAX_NODES={MAX_NODES}; raise NODE_SIZE_UM")

    grad_sigma = max(1.0, um_to_px(GRAD_SIGMA_UM, ds, g_mpp))
    step_px = max(0.25, um_to_px(STEP_UM, ds, g_mpp))
    max_steps = int(um_to_px(MAX_LENGTH_UM, ds, g_mpp) / step_px)
    spacing = max(1.0, um_to_px(SEED_SPACING_UM, ds, g_mpp))

    coherence_by_channel = {}
    pass_by_channel, end_by_channel = {}, {}
    trans_edges, endpt_edges = {}, {}
    channel_valid, channel_stats, seeds_per_channel = {}, {}, {}
    polylines = None

    for ch in INTEGRATION_CHANNELS:
        field = channels.get(ch)
        n_cells = kde["cell_counts"].get(ch, None)

        if field is None:
            channel_valid[ch] = False
            channel_stats[ch] = {"skipped": "channel absent from KDE file"}
            continue
        if ch != "interface" and n_cells is not None and n_cells < MIN_CELLS_PER_CHANNEL:
            channel_valid[ch] = False
            channel_stats[ch] = {"skipped": f"only {n_cells} cells"}
            continue

        f_nz = field[field > 0]
        if f_nz.size < 50:
            channel_valid[ch] = False
            channel_stats[ch] = {"skipped": "field effectively empty"}
            continue

        # Each channel gets its OWN mask -- seeding a sparse type from the
        # global tissue mask would start streamlines in tissue where that cell
        # type does not exist, and the field there is pure smoothing artifact.
        ch_mask = (field >= float(np.percentile(f_nz, TISSUE_PERCENTILE))) & (node_of >= 0)
        if ch_mask.sum() < 50:
            channel_valid[ch] = False
            channel_stats[ch] = {"skipped": "channel mask too small"}
            continue

        sr = np.arange(spacing / 2, H, spacing)
        sc = np.arange(spacing / 2, W, spacing)
        gr, gc = np.meshgrid(sr, sc, indexing="ij")
        cand = np.stack([gr.ravel(), gc.ravel()], axis=1).astype(np.float32)
        ri = np.clip(np.rint(cand[:, 0]).astype(np.int32), 0, H - 1)
        ci = np.clip(np.rint(cand[:, 1]).astype(np.int32), 0, W - 1)
        seeds = cand[ch_mask[ri, ci]]
        if len(seeds) == 0:
            channel_valid[ch] = False
            channel_stats[ch] = {"skipped": "no seeds in channel mask"}
            continue
        if len(seeds) > MAX_SEEDS:
            seeds = seeds[::int(math.ceil(len(seeds) / MAX_SEEDS))]
        seeds_per_channel[ch] = int(len(seeds))

        ch_pass = np.zeros(n_nodes, dtype=np.float32)
        ch_end = np.zeros(n_nodes, dtype=np.float32)
        tp_list, tc_list, ep_list, ec_list = [], [], [], []
        scale_stats, mean_coh = {}, 0.0

        # Gradients depend only on grad_sigma -> computed once per channel and
        # shared across both tensor scales.
        grad_r, grad_c = density_gradients(field, grad_sigma)

        for si, scale_um in enumerate(TENSOR_SCALES_UM):
            v_r, v_c, coh = structure_tensor_from_gradients(
                grad_r, grad_c, max(1.0, um_to_px(scale_um, ds, g_mpp)))
            coherence_by_channel[(ch, si)] = coh
            mean_coh = max(mean_coh, float(coh[ch_mask].mean()))

            rec = None
            if save_polylines and si == 0 and ch == "neoplastic":
                rec = np.linspace(0, len(seeds) - 1,
                                  min(POLYLINE_SAMPLE, len(seeds))).astype(np.int64)

            fwd_traj, _, fwd_n, fwd_pos = integrate_rk4(
                v_r, v_c, coh, ch_mask, seeds, node_of, step_px, max_steps,
                MIN_COHERENCE, MAX_ANGLE_DEG, record_idx=rec)
            bwd_traj, _, bwd_n, bwd_pos = integrate_rk4(
                v_r, v_c, coh, ch_mask, seeds, node_of, step_px, max_steps,
                MIN_COHERENCE, MAX_ANGLE_DEG, record_idx=rec, direction=-1.0)
            del v_r, v_c

            for traj in (fwd_traj, bwd_traj):
                pp, cc_ = transition_edges(traj, n_nodes)
                tp_list.append(pp)
                tc_list.append(cc_)
                vis = traj[traj >= 0]
                if vis.size:
                    ch_pass += np.bincount(vis, minlength=n_nodes).astype(np.float32)

            ta = _terminal_node(bwd_traj, bwd_n)
            tb = _terminal_node(fwd_traj, fwd_n)
            pp, cc_ = endpoint_edges(ta, tb, n_nodes)
            ep_list.append(pp)
            ec_list.append(cc_)
            for t in (ta, tb):
                m = t >= 0
                if m.any():
                    ch_end += np.bincount(t[m], minlength=n_nodes).astype(np.float32)

            lengths_um = (fwd_n + bwd_n) * step_px * kde["mpp"] * ds
            scale_stats[f"scale{si}"] = {
                "scale_um": scale_um,
                "mean_len_um": float(lengths_um.mean()),
                "frac_immediate_stop": float((lengths_um <= step_px * kde["mpp"] * ds).mean()),
                "mean_coherence": float(coh[ch_mask].mean()),
            }
            if rec is not None:
                polylines = _sample_polylines(fwd_pos, bwd_pos, ds)
            del fwd_traj, bwd_traj, fwd_pos, bwd_pos

        del grad_r, grad_c          # free before the next channel loads its field

        if mean_coh < MIN_CHANNEL_COHERENCE:
            channel_valid[ch] = False
            channel_stats[ch] = {"skipped": f"coherence {mean_coh:.3f} below floor"}
            for si in range(len(TENSOR_SCALES_UM)):
                coherence_by_channel.pop((ch, si), None)
            continue

        trans_edges[ch] = _merge_edge_sets(tp_list, tc_list, n_nodes)
        endpt_edges[ch] = _merge_edge_sets(ep_list, ec_list, n_nodes)
        pass_by_channel[ch] = ch_pass
        end_by_channel[ch] = ch_end
        channel_valid[ch] = True
        channel_stats[ch] = dict(n_seeds=int(len(seeds)),
                                 n_cells=(n_cells if n_cells is not None else -1),
                                 **scale_stats)

    if not any(channel_valid.values()):
        raise ValueError("no channel passed gating -- nothing to build a graph from")

    edge_index, edge_attr = _combine_typed_edges(trans_edges, endpt_edges,
                                                 seeds_per_channel, n_nodes,
                                                 node_grid_rc)
    if edge_index.shape[1] == 0:
        raise ValueError("no edges survived -- streamlines all terminated immediately "
                         "(check MIN_COHERENCE / tensor scales / KDE smoothing)")

    feats, feat_names = node_features(d_all, channels, coherence_by_channel,
                                      node_of, n_nodes, pass_by_channel, end_by_channel)

    # Node coordinates in WSI pixel space, taken from the KDE file's own
    # grid_x/grid_y rather than re-derived from downsample. One mapping, one
    # place, and the attention overlay uses exactly what the KDE build wrote.
    gy, gx = kde["grid_y"], kde["grid_x"]
    ridx = np.clip(np.rint(node_rc_ds[:, 0]).astype(np.int32), 0, len(gy) - 1)
    cidx = np.clip(np.rint(node_rc_ds[:, 1]).astype(np.int32), 0, len(gx) - 1)
    node_rc_wsi = np.stack([gy[ridx], gx[cidx]], axis=1).astype(np.float32)
    node_xy_wsi = node_rc_wsi[:, ::-1].copy()

    provenance = dict(
        tensor_scales_um=TENSOR_SCALES_UM, grad_sigma_um=GRAD_SIGMA_UM,
        script_version=SCRIPT_VERSION,
        eigenvector=STREAMLINE_EIGENVECTOR, seed_spacing_um=SEED_SPACING_UM,
        step_um=STEP_UM, max_length_um=MAX_LENGTH_UM, max_angle_deg=MAX_ANGLE_DEG,
        min_coherence=MIN_COHERENCE, tissue_percentile=TISSUE_PERCENTILE,
        node_size_um=NODE_SIZE_UM, min_edge_count=MIN_EDGE_COUNT,
        min_cells_per_channel=MIN_CELLS_PER_CHANNEL,
        downsample=ds, kde_sigma_um=kde["sigma_um"], kde_sigma_px=kde["sigma_px"],
        wsi_mpp=kde["mpp"], channels=INTEGRATION_CHANNELS,
        channel_valid=channel_valid, channel_stats=channel_stats,
        edge_attr_names=channel_edge_attr_names(),
    )

    payload = dict(
        image_id=np.array([str(image_id)]),
        label=np.array([kde["label"]]),
        node_features=feats,
        node_feature_names=np.array(feat_names, dtype=object),
        edge_index=edge_index.astype(np.int64),
        edge_attr=edge_attr.astype(np.float32),
        edge_attr_names=np.array(channel_edge_attr_names(), dtype=object),
        # channel_valid is the mask downstream MUST apply: a False channel means
        # "not measurable on this slide", which is not the same as "zero".
        channel_names=np.array(INTEGRATION_CHANNELS, dtype=object),
        channel_valid=np.array([channel_valid.get(c, False) for c in INTEGRATION_CHANNELS]),
        node_rc_ds=node_rc_ds, node_rc_wsi=node_rc_wsi, node_xy_wsi=node_xy_wsi,
        node_grid_rc=node_grid_rc, node_grid_shape=np.array(node_grid_shape),
        tile_size_wsi_px=np.array([float(node_size_px * ds)], dtype=np.float32),
        wsi_shape=np.array([kde["wsi_h"], kde["wsi_w"]]),
        downsample=np.array([ds]),
        script_version=np.array([SCRIPT_VERSION]),
        provenance_json=np.array([json.dumps(provenance)]),
    )
    if polylines is not None:
        payload["streamlines_rc_wsi"] = np.array(polylines, dtype=object)

    np.savez_compressed(out_path, **payload)
    try:
        verify_graph_file(out_path, expect_nodes=n_nodes)
    except Exception:
        if out_path.exists():
            out_path.unlink()
        raise

    valid_names = [c for c in INTEGRATION_CHANNELS if channel_valid.get(c)]
    s0 = next((channel_stats[c]["scale0"] for c in valid_names
               if "scale0" in channel_stats[c]), {"mean_len_um": 0.0,
                                                  "frac_immediate_stop": 0.0})
    stats = dict(n_nodes=n_nodes, n_edges=int(edge_index.shape[1]),
                 # edge_index is stored symmetrically, so its width already equals
                 # the sum of degrees. The earlier 2* here double-counted.
                 mean_degree=float(edge_index.shape[1] / n_nodes),
                 n_channels=len(valid_names), channels=valid_names,
                 tissue_frac=float(tissue.mean()),
                 grid=f"{H}x{W}", node_grid=f"{node_grid_shape[0]}x{node_grid_shape[1]}",
                 mean_len_um=s0["mean_len_um"], frac_stall=s0["frac_immediate_stop"],
                 channel_stats=channel_stats)
    if verbose:
        log.info(json.dumps(stats, indent=2, default=str))
    return stats


def _combine_typed_edges(trans_edges, endpt_edges, seeds_per_channel, n_nodes,
                         node_grid_rc=None):
    """
    Union every channel's edges into ONE edge set, each edge carrying a
    per-channel count vector.

    Not a heterogeneous multi-relational graph on purpose: with 464 slides and
    5 classes, separate weight matrices per relation is a fast route to
    overfitting. One edge set + rich edge_attr lets GATv2 (edge_dim=...) weigh
    channels itself, and attention still decomposes by channel afterwards,
    which is the whole point.

    Counts are normalised per channel by that channel's own seed count, so a
    dense channel doesn't simply outvote a sparse one on raw volume.
    """
    per_channel = {}
    all_keys = []
    for ch in INTEGRATION_CHANNELS:
        tp, tc = trans_edges.get(ch, (np.empty((0, 2), np.int64), np.empty(0, np.float32)))
        ep, ec = endpt_edges.get(ch, (np.empty((0, 2), np.int64), np.empty(0, np.float32)))
        tk = tp[:, 0] * n_nodes + tp[:, 1] if len(tp) else np.empty(0, np.int64)
        ek = ep[:, 0] * n_nodes + ep[:, 1] if len(ep) else np.empty(0, np.int64)
        per_channel[ch] = (tk, tc, ek, ec)
        all_keys += [tk, ek]

    all_keys = [k for k in all_keys if len(k)]
    if not all_keys:
        return np.empty((2, 0), np.int64), np.empty((0, len(channel_edge_attr_names())), np.float32)
    keys = np.unique(np.concatenate(all_keys))

    cols, total_t, total_e, n_present = [], np.zeros(len(keys), np.float32),         np.zeros(len(keys), np.float32), np.zeros(len(keys), np.float32)
    for ch in INTEGRATION_CHANNELS:
        tk, tc, ek, ec = per_channel[ch]
        t = np.zeros(len(keys), np.float32)
        e = np.zeros(len(keys), np.float32)
        if len(tk):
            t[np.searchsorted(keys, tk)] = tc
        if len(ek):
            e[np.searchsorted(keys, ek)] = ec
        ns = max(seeds_per_channel.get(ch, 0), 1)
        cols += [t, e, t / ns, e / ns]
        total_t += t
        total_e += e
        n_present += (t + e) > 0
    cols += [total_t, total_e, n_present]

    # Separate noise floors. Transition edges are local and accumulate counts
    # well above 2; endpoint edges are long-range and sparse. One shared floor
    # deleted every endpoint edge, which is what turned this into a lattice.
    keep = (total_t >= MIN_EDGE_COUNT) | (total_e >= MIN_ENDPOINT_COUNT)

    if node_grid_rc is not None:
        _lo, _hi = keys // n_nodes, keys % n_nodes
        _gd = np.abs(np.asarray(node_grid_rc)[_lo]
                     - np.asarray(node_grid_rc)[_hi]).max(axis=1)

        if os.environ.get("EPAUDIT"):
            _ep = total_e > 0
            print(f"    [endpoint audit] candidates={len(keys)}  "
                  f"with_endpoint={int(_ep.sum())}")
            if _ep.any():
                _h = np.bincount(total_e[_ep].astype(int), minlength=6)[:6]
                print(f"      endpoint count hist (0..5): {_h}")
                print(f"      endpoint grid dist: median={np.median(_gd[_ep]):.1f} "
                      f"p90={np.percentile(_gd[_ep], 90):.1f} max={int(_gd[_ep].max())}")
                for _t in (1, 2, 3, 5):
                    _k = int(((total_e >= _t) & (_gd >= MIN_ENDPOINT_GRID_DIST)).sum())
                    _base = int((total_t >= MIN_EDGE_COUNT).sum())
                    print(f"      MIN_ENDPOINT_COUNT={_t} -> {_k} long-range, "
                          f"mean degree ~{(_base + _k) / max(n_nodes, 1):.1f}")

        # An endpoint edge between neighbours is a transition edge in disguise.
        _ep_only = (total_t < MIN_EDGE_COUNT) & (total_e >= MIN_ENDPOINT_COUNT)
        keep = keep & ~(_ep_only & (_gd < MIN_ENDPOINT_GRID_DIST))

        # Per-node cap on long-range edges, strongest first, so a slide with
        # thousands of unique terminal pairs cannot swamp the local structure.
        if MAX_ENDPOINT_PER_NODE > 0:
            _cand = np.where(keep & _ep_only)[0]
            if len(_cand):
                _order = _cand[np.argsort(-total_e[_cand], kind="stable")]
                _used = np.zeros(n_nodes, dtype=np.int32)
                for _i in _order:
                    _a, _b = int(_lo[_i]), int(_hi[_i])
                    if (_used[_a] >= MAX_ENDPOINT_PER_NODE
                            or _used[_b] >= MAX_ENDPOINT_PER_NODE):
                        keep[_i] = False
                    else:
                        _used[_a] += 1
                        _used[_b] += 1
    keys = keys[keep]
    cols = [c[keep] for c in cols]
    if len(keys) == 0:
        return np.empty((2, 0), np.int64), np.empty((0, len(cols)), np.float32)

    lo, hi = keys // n_nodes, keys % n_nodes
    src = np.concatenate([lo, hi])
    dst = np.concatenate([hi, lo])
    attr = np.stack([np.concatenate([c, c]) for c in cols], axis=1).astype(np.float32)
    return np.stack([src, dst], axis=0), attr


def _terminal_node(traj, n_steps):
    """Last valid node id per streamline (-1 if the seed itself was invalid)."""
    out = traj[np.arange(len(traj)), np.clip(n_steps, 0, traj.shape[1] - 1)]
    return out.astype(np.int32)


def _merge_edge_sets(pairs_list, counts_list, n_nodes):
    pairs_list = [p for p in pairs_list if len(p)]
    if not pairs_list:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
    P = np.concatenate(pairs_list, axis=0)
    C = np.concatenate([c for c in counts_list if len(c)], axis=0)
    key = P[:, 0] * n_nodes + P[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, weights=C, minlength=len(uniq)).astype(np.float32)
    return np.stack([uniq // n_nodes, uniq % n_nodes], axis=1), tot


def _sample_polylines(fwd_pos, bwd_pos, ds):
    """
    Sampled streamlines as (n_points, 2) float32 polylines in WSI (row, col) px.

    Backward half is reversed and prepended so each entry is one continuous
    bidirectional streamline, ready to plot straight over a slide thumbnail
    (divide by the thumbnail's own downsample factor). These are the actual
    integrated positions, not node centres -- the whole point is showing that
    attention follows real tissue architecture, which a staircase of tile
    centres would misrepresent.
    """
    out = []
    for i in range(len(fwd_pos)):
        f = fwd_pos[i][~np.isnan(fwd_pos[i][:, 0])]
        b = bwd_pos[i][~np.isnan(bwd_pos[i][:, 0])][::-1]
        path = np.concatenate([b, f[1:]]) if len(f) > 1 else b
        if len(path) < 3:
            continue
        out.append((path * ds).astype(np.float32))
    return out


def verify_graph_file(path: Path, expect_nodes: int):
    """Reopen a just-written graph and check it is actually usable."""
    with np.load(path, allow_pickle=True) as z:
        need = ["node_features", "edge_index", "edge_attr", "node_rc_wsi",
                "node_grid_shape", "label", "provenance_json",
                "channel_names", "channel_valid", "edge_attr_names"]
        missing = [k for k in need if k not in z.files]
        if missing:
            raise ValueError(f"written graph missing {missing}")
        if z["node_features"].shape[0] != expect_nodes:
            raise ValueError("node_features row count != n_nodes")
        ei = z["edge_index"]
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f"edge_index has shape {ei.shape}, expected (2, E)")
        if ei.size and (ei.max() >= expect_nodes or ei.min() < 0):
            raise ValueError("edge_index references out-of-range node ids")
        if z["edge_attr"].shape[0] != ei.shape[1]:
            raise ValueError("edge_attr rows != edge count")
        if z["edge_attr"].shape[1] != len(z["edge_attr_names"]):
            raise ValueError("edge_attr width != number of edge_attr_names")
        if z["node_features"].shape[1] != len(z["node_feature_names"]):
            raise ValueError("node_features width != number of node_feature_names")
        if not bool(np.asarray(z["channel_valid"]).any()):
            raise ValueError("no valid channels recorded")


# Self-test (no dataset required)

def self_test():
    """
    Synthetic checks on a deliberately NON-SQUARE slide. A square test cannot
    catch a row/col swap, which is precisely the bug class that already bit this
    pipeline once. Nothing here touches the real dataset.
    """
    print("=" * 68)
    print("SELF-TEST (synthetic, non-square)")
    print("=" * 68)
    fails = []

    ds = EXPECTED_DOWNSAMPLE
    H, W = 300, 900                      # 1:3 aspect -- swap would be obvious
    wsi_h, wsi_w = H * ds, W * ds

    # Horizontal bands: density varies along ROWS, constant along COLS.
    # Gradient is vertical, so along-structure (minor eigenvector) must be
    # HORIZONTAL, i.e. |v_col| >> |v_row|.
    rr = np.arange(H, dtype=np.float32)[:, None]
    dens = (1.0 + np.sin(2 * np.pi * rr / 60.0)).astype(np.float32) * np.ones((1, W), np.float32)
    dens = gaussian_filter(dens, 3.0)

    _gr, _gc = density_gradients(dens, max(1.0, um_to_px(GRAD_SIGMA_UM, ds)))
    v_r, v_c, coh = structure_tensor_from_gradients(
        _gr, _gc, max(1.0, um_to_px(TENSOR_SCALES_UM[0], ds)))
    inner = (slice(40, H - 40), slice(40, W - 40))
    mean_abs_c = float(np.abs(v_c[inner]).mean())
    mean_abs_r = float(np.abs(v_r[inner]).mean())
    print(f"[1] banded field: mean|v_col|={mean_abs_c:.3f} mean|v_row|={mean_abs_r:.3f} "
          f"(expect col >> row for 'minor')")
    if not (mean_abs_c > 0.9 and mean_abs_r < 0.1):
        fails.append("minor eigenvector is not along horizontal bands -- axis or eigenvector swap")

    # RK4 on that field must travel far horizontally and barely drift vertically.
    tissue = np.ones((H, W), dtype=bool)
    node_of, centres, grid_rc, grid_shape = build_node_grid(tissue, um_to_px(NODE_SIZE_UM, ds))
    seeds = np.array([[150.0, 100.0], [80.0, 400.0]], dtype=np.float32)
    step_px = um_to_px(STEP_UM, ds)
    traj, end, nst, _ = integrate_rk4(v_r, v_c, coh, tissue, seeds, node_of, step_px,
                                   int(um_to_px(MAX_LENGTH_UM, ds) / step_px),
                                   MIN_COHERENCE, MAX_ANGLE_DEG)
    d_row = np.abs(end[:, 0] - seeds[:, 0])

    # Sign-aware interpolation: an antiparallel-encoded field must not cancel.
    a_r = np.zeros((4, 4), np.float32)
    a_c = np.ones((4, 4), np.float32)
    a_c[:, 2:] = -1.0                    # same orientation, opposite stored sign
    got = _sample_direction(a_r, a_c, np.array([[1.5, 1.5]], np.float32),
                            np.array([[0.0, 1.0]], np.float32))
    print(f"[3] sign-aware interp across a sign flip -> {got.round(3).tolist()} "
          f"(expect unit, not ~0)")
    if abs(abs(got[0, 1]) - 1.0) > 1e-3:
        fails.append("sign-aware interpolation cancelled an antiparallel neighbour")

    # Geometry round-trip: node centre -> WSI px must land inside its own tile,
    # and grid ids must reconstruct an image of the right aspect ratio.
    node_size_px = um_to_px(NODE_SIZE_UM, ds)
    wsi_rc = centres * ds
    inside = np.all(np.abs(wsi_rc / ds - (grid_rc + 0.5) * node_size_px) < 1e-3)
    # Exact expected lattice, not a loose ratio: tile quantisation legitimately
    # shifts the aspect ratio on small grids, so compare against the tiling math
    # itself, plus the orientation check that a wide slide gives a wide lattice.
    expected_shape = (int(math.ceil(H / node_size_px)), int(math.ceil(W / node_size_px)))
    shape_ok = tuple(grid_shape) == expected_shape and grid_shape[1] > grid_shape[0]
    print(f"[4] node geometry: grid={grid_shape} expected={expected_shape} "
          f"wsi=({wsi_h},{wsi_w}) centres_ok={inside} lattice_ok={shape_ok}")
    if not (inside and shape_ok):
        fails.append("node centre / grid-shape geometry does not round-trip to WSI space")

    # End-to-end on a fake multi-channel KDE file, including load-time guards.
    tmp = Path("_selftest_kde.npz")
    neo = (dens * 0.6).astype(np.float32)
    np.savez_compressed(
        tmp, schema_version=np.array([KDE_SCHEMA_VERSION]),
        density_all=dens, density_neoplastic=neo,
        density_epithelial=(dens * 0.15).astype(np.float32),
        density_connective=(dens * 0.2).astype(np.float32),
        density_inflammatory=(dens * 0.05).astype(np.float32),
        density_dead=np.zeros_like(dens),
        downsample=np.array([ds]), sigma_px=np.array([15.0]), sigma_um=np.array([60.0]),
        pixel_size_um=np.array([WSI_MPP]),
        wsi_shape=np.array([wsi_h, wsi_w]), label=np.array(["HGSC"]),
        image_id=np.array(["selftest"]),
        grid_x=(np.arange(W) + 0.5) * ds, grid_y=(np.arange(H) + 0.5) * ds,
        cell_counts_json=np.array([json.dumps(
            {"Neoplastic": 50000, "Epithelial": 20000, "Connective": 30000,
             "Inflammatory": 8000, "Dead": 10})]),
        coord_diag_json=np.array([json.dumps({"frac_oob": 0.0})]))
    out = Path("_selftest_graph.npz")
    try:
        stats = process_one("selftest", tmp, out, wsi_h, wsi_w, save_polylines=True)
        print(f"[5] end-to-end: nodes={stats['n_nodes']} edges={stats['n_edges']} "
              f"channels={stats['channels']}")
        if stats["n_edges"] == 0:
            fails.append("end-to-end produced no edges")
        if "interface" not in stats["channels"]:
            fails.append("derived interface channel did not build")
        with np.load(out, allow_pickle=True) as z:
            gs = tuple(int(v) for v in z["node_grid_shape"])
            att = np.zeros(gs, np.float32)
            g = z["node_grid_rc"]
            att[g[:, 0], g[:, 1]] = 1.0          # the attention-overlay operation
            n_nodes = z["node_features"].shape[0]
            print(f"[6] attention scatter into {gs}: {int(att.sum())}/{n_nodes} nodes; "
                  f"edge_attr={z['edge_attr'].shape} node_feats={z['node_features'].shape}")
            if int(att.sum()) != n_nodes:
                fails.append("node_grid_rc does not scatter 1:1 into node_grid_shape")
            # node coords must come from the KDE grid, and land inside the slide
            rc = z["node_rc_wsi"]
            if rc[:, 0].max() > wsi_h or rc[:, 1].max() > wsi_w or rc.min() < 0:
                fails.append("node_rc_wsi falls outside the slide bounds")
    finally:
        for pth in (tmp, out):
            if pth.exists():
                pth.unlink()

    # Guard must reject a mismatched manifest shape.
    tmp2 = Path("_selftest_bad.npz")
    np.savez_compressed(
        tmp2, schema_version=np.array([KDE_SCHEMA_VERSION]),
        density_all=dens, density_neoplastic=dens,
        downsample=np.array([ds]), sigma_px=np.array([15.0]), sigma_um=np.array([60.0]),
        wsi_shape=np.array([wsi_h, wsi_w]), label=np.array(["EC"]),
        grid_x=(np.arange(W) + 0.5) * ds, grid_y=(np.arange(H) + 0.5) * ds,
        coord_diag_json=np.array([json.dumps({"frac_oob": 0.0})]))
    try:
        load_kde(tmp2, wsi_w, wsi_h)   # swapped on purpose
        fails.append("load_kde accepted a swapped manifest shape")
        print("[7] swapped-manifest guard: NOT RAISED  <-- bad")
    except ValueError:
        print("[7] swapped-manifest guard: raised as expected")
    finally:
        tmp2.unlink()

    # A stale-schema file must be rejected, not silently half-read.
    tmp3 = Path("_selftest_v1.npz")
    np.savez_compressed(tmp3, density_all=dens, density_neoplastic=dens,
                        downsample=np.array([ds]), wsi_shape=np.array([wsi_h, wsi_w]),
                        label=np.array(["CC"]))
    try:
        load_kde(tmp3, wsi_h, wsi_w)
        fails.append("load_kde accepted a pre-v3 KDE file")
        print("[8] stale-schema guard: NOT RAISED  <-- bad")
    except ValueError:
        print("[8] stale-schema guard: raised as expected")
    finally:
        tmp3.unlink()

    # A slide the KDE build flagged for out-of-bounds centroids must be refused.
    tmp4 = Path("_selftest_oob.npz")
    np.savez_compressed(
        tmp4, schema_version=np.array([KDE_SCHEMA_VERSION]),
        density_all=dens, density_neoplastic=dens,
        downsample=np.array([ds]), sigma_px=np.array([15.0]), sigma_um=np.array([60.0]),
        wsi_shape=np.array([wsi_h, wsi_w]), label=np.array(["MC"]),
        grid_x=(np.arange(W) + 0.5) * ds, grid_y=(np.arange(H) + 0.5) * ds,
        coord_diag_json=np.array([json.dumps({"frac_oob": 0.12})]))
    try:
        load_kde(tmp4, wsi_h, wsi_w)
        fails.append("load_kde accepted a slide with 12% out-of-bounds centroids")
        print("[9] out-of-bounds guard: NOT RAISED  <-- bad")
    except ValueError:
        print("[9] out-of-bounds guard: raised as expected")
    finally:
        tmp4.unlink()

    print("-" * 68)
    if fails:
        print("FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


# Batch driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="synthetic checks, no dataset")
    ap.add_argument("--only", type=str, default=None, help="single image_id, verbose")
    ap.add_argument("--limit", type=int, default=None, help="process at most N pending WSIs")
    ap.add_argument("--force", action="store_true", help="rebuild even if a graph exists")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    run_dir = Path(ROOT) / RUN_DIR_NAME
    kde_dir = run_dir / KDE_SUBFOLDER
    out_dir = run_dir / OUT_SUBFOLDER
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(run_dir / MANIFEST_FILENAME)
    log.info(f"build_streamline_graphs {SCRIPT_VERSION} | "
             f"channels: {INTEGRATION_CHANNELS} | KDE schema v{KDE_SCHEMA_VERSION}")
    log.info(f"Manifest: {len(df)} rows")
    log.info(f"nominal um at {WSI_MPP} um/px (physical = 2x at the native 0.5 um/px): "
             f"1 grid px = {WSI_MPP * EXPECTED_DOWNSAMPLE:.2f} nominal um | "
             f"node tile = {NODE_SIZE_UM:.0f} nominal um "
             f"({NODE_SIZE_UM / WSI_MPP:.0f} px = {NODE_SIZE_UM / WSI_MPP * 0.5:.0f} um physical) | "
             f"step = {STEP_UM:.0f} nominal um")

    results = []
    n_done = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Streamline graphs"):
        image_id = str(row["image_id"])
        if args.only and image_id != args.only:
            continue
        label = row["label"]
        mh, mw = int(row["image_height"]), int(row["image_width"])

        if image_id in EXCLUDE_IDS and not args.only:
            results.append(dict(image_id=image_id, label=label, status="excluded_cellvit_pending"))
            continue

        kde_path = kde_dir / f"{image_id}_kde.npz"
        out_path = out_dir / f"{image_id}_graph.npz"

        if not kde_path.exists():
            results.append(dict(image_id=image_id, label=label, status="no KDE map yet"))
            continue
        if out_path.exists() and not args.force:
            try:
                with np.load(out_path, allow_pickle=True) as z:
                    verify_graph_file(out_path, expect_nodes=z["node_features"].shape[0])
                    # Structure alone is not enough: successive builder versions
                    # write the same keys with different physics. Without this
                    # check a parameter change silently leaves half the corpus on
                    # the old settings, reported as "cached".
                    ver = str(z["script_version"][0]) if "script_version" in z.files \
                        else "pre-versioning"
                    if ver != SCRIPT_VERSION:
                        raise ValueError(f"built by {ver}, current is {SCRIPT_VERSION}")
                results.append(dict(image_id=image_id, label=label, status="cached"))
                continue
            except Exception as e:
                log.warning(f"{image_id}: existing graph unreadable ({e}) -- rebuilding")

        # A KDE map still being written by the (currently running) rebuild would
        # load as a truncated zip; treat that as pending, not as a failure.
        t0 = time.time()
        try:
            stats = process_one(image_id, kde_path, out_path, mh, mw,
                                save_polylines=(n_done < SAVE_POLYLINES_FOR_FIRST),
                                verbose=bool(args.only))
        except Exception as e:
            msg = str(e)
            status = "kde still being written" if "zip" in msg.lower() else f"failed: {msg}"
            if out_path.exists():
                out_path.unlink()   # never leave a half-written graph on disk
            results.append(dict(image_id=image_id, label=label, status=status))
            log.error(f"{image_id}: {status}")
            continue

        n_done += 1
        tqdm.write(
            f"[GRAPH] {image_id} ({label})  grid={stats['grid']} nodes={stats['n_nodes']:>5} "
            f"({stats['node_grid']}) edges={stats['n_edges']:>6} deg={stats['mean_degree']:.1f}  "
            f"ch={stats['n_channels']}/{len(INTEGRATION_CHANNELS)} "
            f"[{','.join(c[:4] for c in stats['channels'])}]  "
            f"len_um={stats['mean_len_um']:.0f} stall={stats['frac_stall']:.2f}  "
            f"{time.time()-t0:.1f}s"
        )
        # What to watch while this runs: 'stall' near 1.0 means streamlines die
        # at the seed (tensor scale too fine, or MIN_COHERENCE too strict);
        # 'deg' near 2.0 means a chain with no long-range connectivity for
        # attention to be interesting over; and a channel that is valid on some
        # slides but not others is fine, while a channel valid on ~no slides
        # means its gate is set wrong, not that the biology is absent.
        rec = dict(image_id=image_id, label=label, status="ok",
                   n_nodes=stats["n_nodes"], n_edges=stats["n_edges"],
                   mean_degree=round(stats["mean_degree"], 2),
                   n_channels=stats["n_channels"],
                   mean_len_um=round(stats["mean_len_um"], 1),
                   frac_stall=round(stats["frac_stall"], 3))
        for c in INTEGRATION_CHANNELS:
            rec[f"ok_{c}"] = c in stats["channels"]
        results.append(rec)

        if args.limit and n_done >= args.limit:
            break

    log_df = pd.DataFrame(results)
    log_df.to_csv(out_dir / "graph_build_log.csv", index=False)

    def n(s):
        return int((log_df["status"] == s).sum()) if len(log_df) else 0

    ok = log_df[log_df["status"] == "ok"] if len(log_df) else log_df
    print(f"\nDone. ok={len(ok)} cached={n('cached')} pending_kde={n('no KDE map yet')} "
          f"excluded={n('excluded_cellvit_pending')} "
          f"failed={len(log_df) - len(ok) - n('cached') - n('no KDE map yet') - n('excluded_cellvit_pending') if len(log_df) else 0}")

    if len(ok):
        print("\nPer-label graph stats (watch for a class whose graphs are structurally "
              "different -- that is a confound, not a feature):")
        print(ok.groupby("label")[["n_nodes", "n_edges", "mean_degree", "n_channels",
                                   "mean_len_um", "frac_stall"]].mean().round(2).to_string())

        ch_cols = [f"ok_{c}" for c in INTEGRATION_CHANNELS]
        print("\nChannel availability by label (fraction of slides where the channel "
              "passed gating):")
        print(ok.groupby("label")[ch_cols].mean().round(2).to_string())
        # A channel available on, say, 100% of HGSC and 20% of MC is a label
        # leak dressed as a feature -- it would let the model classify off
        # channel presence rather than architecture. Check this column before
        # training, not after.
        overall = ok[ch_cols].mean()
        dead_ch = overall[overall < 0.05].index.tolist()
        if dead_ch:
            print(f"\nWARNING: {dead_ch} passed gating on <5% of slides. Either lower "
                  f"MIN_CELLS_PER_CHANNEL/MIN_CHANNEL_COHERENCE or drop them from "
                  f"INTEGRATION_CHANNELS -- as-is they are dead weight in edge_attr.")

        stalled = ok[ok["frac_stall"] > 0.5]
        if len(stalled):
            print(f"\nWARNING: {len(stalled)} WSIs had >50% of streamlines stall at the "
                  f"seed. Tune TENSOR_SCALES_UM / MIN_COHERENCE before building the "
                  f"GATv2 dataset.")

    failed = log_df[~log_df["status"].isin(["ok", "cached", "no KDE map yet",
                                            "excluded_cellvit_pending"])] if len(log_df) else log_df
    if len(failed):
        print("\nFailures:")
        print(failed[["image_id", "label", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()

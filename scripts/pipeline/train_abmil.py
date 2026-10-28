# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
ABMIL sweep: what are the tile features worth, with the graph removed entirely?

    python scripts/pipeline/train_abmil.py --smoke              # ~2 min, checks it runs
    python scripts/pipeline/train_abmil.py                      # full sweep
    python scripts/pipeline/train_abmil.py --only abmil_h128    # one config

Reads node_features out of the existing *_graph.npz files and IGNORES
edge_index / edge_attr completely. No torch_geometric, no graph, no
streamline topology in the architecture.


WHY
===
The v5 sweep found every config within 0.024 balanced accuracy of every other,
under the measured 0.03 noise floor. Critically, attnpool_noedges (0.5448) tied
mean_pool (0.5598) -- and attnpool_noedges replaces the whole graph with self
loops, so it is a per-node MLP wearing a GNN's clothes. Message passing over
512um tile nodes contributes nothing.

But that model is still a GATv2Conv with add_self_loops=True and nothing else:
each node attends to exactly one neighbour, itself, and a softmax over one
element is 1.0. So it is a linear layer with dead attention machinery bolted
on, running hyperparameters (layers=3, hidden=64, heads=4) that were chosen for
a graph that no longer exists.

This script is the honest version of that model: a small MLP encoder per tile,
gated attention pooling (Ilse et al., ABMIL), a classifier. Swept properly.

It answers ONE question: is 0.55 a limit of the architecture, or of the 39
features? If a properly-fitted ABMIL still lands at 0.55, the features are the
ceiling and the next move is better features (nuclear morphometrics), not a
better model.

WHAT THIS IS NOT
================
This is NOT the positive control. It cannot tell you whether 0.55 is the
ceiling for this DATA -- it only compares your pipeline against itself. For
that you need tile embeddings from a pretrained pathology encoder through the
same MIL head and the same folds. Do that before writing anything up.

A SIDE BENEFIT
==============
The 0.03 noise floor was measured on PyG and attributed to CUDA scatter
nondeterminism. There is no scatter here -- attention pooling is a masked
softmax over a padded tensor. Run --noise-floor to re-measure it. If it drops,
effects too small to see in the v5 sweep become visible.

FOLDS MATCH THE v5 SWEEP EXACTLY
================================
StratifiedKFold(5, shuffle=True, random_state=repeat), same as
train_gatv2_sweep2.py. So results here are paired per (repeat, fold) with the
GAT results and can be compared directly.

LOAD-TIME FIXES TO THE NUCLEAR FEATURES
=======================================
An audit of extract_nuclear_features.py output (alignment, numerics, internal
consistency and physical plausibility all passed) surfaced three issues. All
three are fixed HERE rather than in the extractor, because the columns needed
to fix them are already in the npz -- no re-extraction required:

1. Availability tracks subtype. neo_measured is 0.97 on EC but 0.42 on MC
   (spread 0.55, Kruskal p=1.2e-3), so the presence flags leak the label.
   `flag_leak_report()` quantifies it at load time and `--drop-flags` runs the
   ablation arm. Keeping the flags is still the default -- dropping them makes
   0 ambiguous between "not measured" and a real zero.
2. The spacing features (nnd_*) ship no presence flag even though they are
   gated at >=2 nuclei while neo_measured gates at >=10. NUC_DERIVED_FLAGS
   reconstructs the missing gate exactly from the stored count columns.
3. nuc_have_cv2 is constant 1.0 corpus-wide. `drop_degenerate()` removes it
   and any other zero-variance column.

Also inert but NOT dropped automatically: neo_ecc_std (class means 0.145-0.149,
Kruskal H=1.02 p=0.80 at slide level). Slide-level flatness is weak evidence
against a per-tile feature inside a MIL model, so it stays in by default; drop
it with `--exclude-prefix neo_ecc_std` if you want it gone.

Not fixable here, for the record: tile_size_wsi_px=1024 at the true 0.5um/px
means the tile is 512um (the builder's NODE_SIZE_UM = 256 is a nominal figure at
0.25um/px), and the streamline step is 40um not 20um. Pure documentation --
nothing computational depends on it.
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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import corpus_paths

ROOT = str(corpus_paths.ROOT)
RUN_DIR_NAME = "cellvit_run"
GRAPH_SUBFOLDER = "graphs"
OUT_SUBFOLDER = "graphs/gatv2_run"

CLASSES = ["CC", "EC", "HGSC", "LGSC", "MC"]

REPEATS, FOLDS, EPOCHS, PATIENCE, BATCH = 3, 5, 150, 30, 8

# The 3 seeds agreed on the predicted class on only 63.4% of slides (Task 3b),
# and a large share of that is not the model being unstable -- it is the
# SELECTION RULE being unstable. `argmax` picks the single best epoch off an
# inner-validation curve computed on ~73 slides across 5 imbalanced classes.
# Two seeds can land on genuinely different points of the same trajectory just
# from which slide fell in which inner split.
#
# So: train the full trajectory (no early stop) and vary only how the final
# weights are chosen from it. Every mode below sees an IDENTICAL training run
# for a given seed, so the paired delta isolates the selection rule.
#   argmax  -- best inner-val epoch (the incumbent, minus early stopping)
#   smooth  -- best epoch of the val curve after a centred moving average;
#              same rule, denoised, so a single lucky epoch cannot win
#   fixed   -- last epoch, no selection at all (cosine has annealed to ~0 lr)
#   swa     -- average the WEIGHTS over the tail of training
#   swatop  -- average the weights of the top-k epochs by val score
# Weight averaging is safe here only because the model has no BatchNorm (just
# Linear/ReLU/Dropout), so there are no running statistics to recompute.
SELECT_MODES = ("argmax", "smooth", "fixed", "swa", "swatop")
SMOOTH_WINDOW = 7
SWA_START_FRAC = 0.60
SWA_TOP_K = 10
SCHEDULES = ("cosine", "cosine_floor", "const")
# cosine anneals the lr to 0, so the last epochs sit in one spot and averaging
# them is nearly a no-op. cosine_floor stops at lr*LR_FLOOR_FRAC and holds, so
# the tail actually explores and SWA has something to average over.
LR_FLOOR_FRAC = 0.10

# TransMIL bags reach ~970 tokens; full N^2 self-attention at BATCH=8 is a
# ~120 MB attention matrix per layer per direction. Halve the batch for it.
ARCH_BATCH = {"transmil": 4}

LOG1P_PREFIXES = ("dens_", "neoplastic_", "epithelial_", "connective_",
                  "inflammatory_", "interface_")
LOG1P_EXCLUDE = ("_frac", "_coherence", "_log_")

# Loaded and column-concatenated onto node_features when --nuclear-features is
# set. The npz itself carries `nuc_log1p_cols` / `nuc_flag_cols` masks; these
# name lists are the fallback and keep the normaliser honest:
#   log1p  -> count-like, heavy-tailed, want log before z-score
#   nonorm -> 0/1 presence flags, must pass through the normaliser untouched
NUC_LOG1P_NAMES = {
    "nuc_count", "nuc_per_mm2", "neo_count", "inflam_count", "connective_count",
    "epithelial_count", "dead_count", "neo_area_p90_p10_ratio", "til_inflam_per_neo",
}
NUC_FLAG_NAMES = {"morph_measured", "neo_measured", "til_measured", "nuc_have_cv2",
                  "nnd_all_measured", "nnd_neo_measured"}

# Loaded and column-concatenated after the nuclear block when
# --chromatin-features is set. These are the first features in the project
# derived from WSI PIXELS rather than from CellViT's output geometry.
CHROM_ANCHOR = "neo_h_mean_mean"          # first column of the chromatin block
CHROM_LOG1P_NAMES = {"chrom_n_neo_measured", "chrom_n_all_measured"}
CHROM_FLAG_NAMES = {"neo_chrom_measured", "all_chrom_measured"}
# suffix keys that split the chromatin block into interpretable groups, so the
# same drop-one-group-at-a-time ablation works on it
CHROM_LEVEL_KEYS = ("h_mean", "h_ring_diff")
CHROM_TEXTURE_KEYS = ("h_std", "h_cv", "h_p90_p10", "h_entropy", "h_grad",
                      "glcm_contrast", "glcm_homogeneity", "glcm_asm",
                      "glcm_corr")
EOS_ANCHOR = "tile_e_mean"                # first column of the eosin block
EOS_LOG1P_NAMES = {"eos_n_neo_measured", "eos_n_all_measured"}
EOS_FLAG_NAMES = {"neo_eos_measured", "all_eos_measured"}
# the CC/MC evidence proper (enclosed empty space) vs everything else eosin
EOS_CLEAR_KEYS = ("clear_frac", "pale_frac", "ring_empty_frac", "region_frac")

FM_PREFIX = "fm_"                         # synthetic names for encoder outputs

ALL_FLAG_NAMES = NUC_FLAG_NAMES | CHROM_FLAG_NAMES | EOS_FLAG_NAMES

# extract_nuclear_features.py gates the spacing features internally at >=2
# points (lines 340-349) but ships no flag for them, while neo_measured gates
# at MIN_NEO=10. So nnd_neo_um_mean is 0 on ~10% of tiles ("fewer than 2
# neoplastic nuclei") while neo_measured is 0 on ~15% -- and in the ~5% gap the
# value IS present but estimated from 2-9 nuclei, which is very noisy (one
# audited CC slide reaches 522um, legal inside a 512um tile but a two-cell
# estimate). Without a flag, 0 means "unmeasured" on some rows and a genuine
# small spacing on others, with nothing to separate them.
#
# The gate is exactly recoverable from columns the npz already stores, so this
# is a load-time fix and needs no re-extraction:
#   name -> (source column, threshold)
NUC_DERIVED_FLAGS = {
    "nnd_all_measured": ("nuc_count", 2.0),
    "nnd_neo_measured": ("neo_count", 2.0),
}

# Bump when the loader changes what goes into a bag, so an existing
# bag_cache_<tag>.pt built by an older loader is rebuilt instead of silently
# reused with the wrong columns.
#
# SCHEMA_VERSION only detects loader-code changes. It cannot know that new
# <id>_nuclei.npz files appeared on disk since the cache was built, so a sweep
# can silently train on a cache left from a partial extraction. After any change
# to the on-disk feature corpus, pass --rebuild-cache explicitly.
SCHEMA_VERSION = 4       # v4: eosin/cytoplasm and foundation-model blocks

# Which biological axis is actually earning its keep? Drop one group at a time
# and compare. Selection is POSITIONAL, not by name, because the graph block
# and the nuclear block both contain `epithelial_frac` and `connective_frac`
# -- name-based selection would silently hit the wrong column. The nuclear
# block begins at `nuc_count`, which is unique to it.
NUC_SHAPE_NAMES = [
    "neo_area_um2_mean", "neo_area_um2_std", "neo_area_um2_cv",
    "neo_area_um2_p10", "neo_area_um2_p90", "neo_area_p90_p10_ratio",
    "neo_equivdiam_um_mean", "neo_ecc_mean", "neo_ecc_std",
    "neo_axisratio_mean", "neo_circularity_mean", "neo_solidity_mean",
    "neo_solidity_std", "all_area_um2_mean", "all_area_um2_cv",
    "all_ecc_mean", "all_solidity_mean",
]
NUC_SPATIAL_NAMES = [
    "nnd_all_um_mean", "nnd_all_um_cv", "nnd_neo_um_mean",
    "til_inflam_nn_neo_um_mean", "til_frac_neo_infiltrated", "til_inflam_per_neo",
]
NUC_DENSITY_NAMES = [
    "nuc_count", "nuc_per_mm2", "neo_count", "inflam_count", "connective_count",
    "epithelial_count", "dead_count", "neo_frac", "inflam_frac",
    "connective_frac", "epithelial_frac", "dead_frac",
]
FEATURE_GROUPS = {
    "graph": None,                      # special-cased: everything before nuc_count
    "nuc_shape": NUC_SHAPE_NAMES,       # pleomorphism / morphometry
    "nuc_spatial": NUC_SPATIAL_NAMES,   # spacing + TIL / immune
    "nuc_density": NUC_DENSITY_NAMES,   # counts + composition fractions
    "nuc_flags": sorted(NUC_FLAG_NAMES),
    # chromatin block, resolved by suffix within the block rather than by an
    # explicit name list -- the names are generated (group x metric x
    # statistic) so enumerating them here would drift from the extractor
    "chrom_level": None,                # hyperchromasia: how dark the nuclei are
    "chrom_texture": None,              # coarseness / clumping / GLCM
    # NOT just presence flags: this also holds tile_h_mean / tile_h_std /
    # tile_tissue_frac, which describe the STAINING of the tile rather than any
    # nucleus in it. That makes them a confound of exactly the dead_count kind
    # -- a model could read slide provenance off them -- so they are grouped
    # here precisely so `--drop-groups chrom_context` can test whether the
    # chromatin gain survives without them.
    "chrom_context": None,
    "chrom": None,                      # the whole block, for the headline ablation

    # `eos_clear` is the CC/MC evidence proper: enclosed empty space, per tile
    # and per nucleus. Separated from the rest of the eosin block so the
    # ablation can ask whether the gain is the clearing measurement
    # specifically or just more eosin-channel intensity columns.
    "eos_clear": None,
    "eos_other": None,
    "eos": None,

    "fm": None,
}
CHROM_GROUPS = ("chrom_level", "chrom_texture", "chrom_context", "chrom")
EOS_GROUPS = ("eos_clear", "eos_other", "eos")

# name, hidden, layers, attn_hidden, dropout, lr, wd, pool, heads, arch
CONFIGS = [
    ("abmil_h64",       64, 1, 128, 0.30, 3e-4, 1e-4, "attn", 1, "abmil"),
    ("abmil_h128",     128, 1, 128, 0.30, 3e-4, 1e-4, "attn", 1, "abmil"),
    ("abmil_h256",     256, 1, 128, 0.30, 3e-4, 1e-4, "attn", 1, "abmil"),
    ("abmil_h128_L2",  128, 2, 128, 0.30, 3e-4, 1e-4, "attn", 1, "abmil"),
    ("abmil_h128_do5", 128, 1, 128, 0.50, 3e-4, 1e-4, "attn", 1, "abmil"),
    ("abmil_h128_lr1", 128, 1, 128, 0.30, 1e-3, 1e-4, "attn", 1, "abmil"),
    ("abmil_h128_wd3", 128, 1, 128, 0.30, 3e-4, 1e-3, "attn", 1, "abmil"),
    # baselines: does the attention pooling itself earn its place?
    ("meanpool_h128",  128, 1, 128, 0.30, 3e-4, 1e-4, "mean", 1, "abmil"),
    ("maxpool_h128",   128, 1, 128, 0.30, 3e-4, 1e-4, "max", 1, "abmil"),
    # The sweep above is one-factor-at-a-time off an h128 baseline, so the two
    # winners (lr=1e-3 and dropout=0.5) were never tried TOGETHER. Both point
    # the same way -- the train/val gap is wide (train 0.78-0.91 vs val
    # 0.58-0.67), so more regularisation plus a faster schedule should help.
    ("abmil_lr1_do5",    128, 1, 128, 0.50, 1e-3, 1e-4, "attn", 1, "abmil"),
    ("abmil_lr1_do5_L2", 128, 2, 128, 0.50, 1e-3, 1e-4, "attn", 1, "abmil"),
    # attn_hidden was pinned at 128 in every config, including abmil_h256 --
    # so the 256-d encoder output was squeezed through a 128-d attention
    # bottleneck. Unpin it.
    ("abmil_h256_a256",     256, 1, 256, 0.30, 3e-4, 1e-4, "attn", 1, "abmil"),
    ("abmil_h256_a256_lr1", 256, 1, 256, 0.30, 1e-3, 1e-4, "attn", 1, "abmil"),
    # Multi-head gated attention: parallel (V,U,w) triplets, each pooling the
    # bag independently, outputs concatenated. Lets different heads specialise
    # (e.g. one on density, one on nuclear grade) and gives a per-head weight
    # map to interpret instead of a single pooled one.
    ("abmil_mh4",     128, 1, 128, 0.30, 1e-3, 1e-4, "attn", 4, "abmil"),
    ("abmil_mh8",     128, 1, 128, 0.30, 1e-3, 1e-4, "attn", 8, "abmil"),
    ("abmil_mh4_do5", 128, 1, 128, 0.50, 1e-3, 1e-4, "attn", 4, "abmil"),
    # All hold hidden/dropout/lr at the abmil_h128_lr1 reference so the only
    # thing that changes is the aggregation formulation itself.
    ("addmil_h128",     128, 1, 128, 0.30, 1e-3, 1e-4, "attn", 1, "additive"),
    ("addmil_h128_do5", 128, 1, 128, 0.50, 1e-3, 1e-4, "attn", 1, "additive"),
    ("addmil_mh4",      128, 1, 128, 0.30, 1e-3, 1e-4, "attn", 4, "additive"),
    ("clam_h128",       128, 1, 128, 0.30, 1e-3, 1e-4, "attn", 1, "clam"),
    ("clam_h128_do5",   128, 1, 128, 0.50, 1e-3, 1e-4, "attn", 1, "clam"),
    ("transmil_L1",     128, 1, 128, 0.30, 1e-3, 1e-4, "attn", 4, "transmil"),
    ("transmil_L2",     128, 2, 128, 0.30, 1e-3, 1e-4, "attn", 4, "transmil"),
    ("transmil_L2_do5", 128, 2, 128, 0.50, 1e-3, 1e-4, "attn", 4, "transmil"),
]
COLS = ["name", "hidden", "layers", "attn_hidden", "dropout", "lr", "wd",
        "pool", "heads", "arch"]


def cfg_field(cfg, field, default=None):
    """Read one field out of a config tuple by name, tolerating short tuples.

    Configs grew from 8 to 9 to 10 elements over the project; checkpoints on
    disk carry whichever length was current when they were written."""
    i = COLS.index(field)
    return cfg[i] if len(cfg) > i else default


def append_derived_flags(NF, names):
    """Append the NUC_DERIVED_FLAGS presence columns to one slide's nuclear
    feature block. Returns (NF, names) unchanged if the source columns are
    absent, so an older nuclei npz still loads."""
    col = {n: i for i, n in enumerate(names)}
    extra, extra_names = [], []
    for flag, (src, thresh) in NUC_DERIVED_FLAGS.items():
        if src not in col or flag in col:
            continue
        extra.append((NF[:, col[src]] >= thresh).astype(np.float32))
        extra_names.append(flag)
    if not extra:
        return NF, names
    return np.concatenate([NF, np.stack(extra, axis=1)], axis=1), names + extra_names


def drop_degenerate(X, feat_names, label=""):
    """Remove columns with zero variance across the whole corpus.

    Catches nuc_have_cv2 (constant 1.0 on every slide once cv2 is installed)
    and anything else that turns out inert. The Normaliser's std.clamp(1e-6)
    already stops these from producing NaN, but they are dead input dimensions
    and a constant column that is constant only in the TRAIN split is a genuine
    amplification hazard, so it is cleaner to drop them up front.
    """
    if not len(X):
        return X, feat_names
    A = torch.cat(X, 0)
    keep = (A.std(0) > 1e-8).nonzero(as_tuple=True)[0].tolist()
    dropped = [feat_names[i] for i in range(len(feat_names)) if i not in set(keep)]
    if dropped:
        print(f"  dropping {len(dropped)} zero-variance column(s){label}: {dropped}")
        X = [b[:, keep] for b in X]
        feat_names = [feat_names[i] for i in keep]
    return X, feat_names


def resolve_groups(feat_names):
    """Map each feature group to the COLUMN INDICES it occupies.

    Positional, deliberately. The graph block and the nuclear block both carry
    a column literally named `epithelial_frac` and one named `connective_frac`
    (KDE-derived composition vs. raw nucleus-count composition -- different
    quantities, same name). Selecting by name alone would hit both and
    silently corrupt every group ablation, so each block is located by its
    unique first column -- `nuc_count` for nuclear, CHROM_ANCHOR for chromatin
    -- and names are only resolved WITHIN the block they belong to.
    """
    n = len(feat_names)

    def find(name):
        return feat_names.index(name) if name in feat_names else n

    nuc_start = find("nuc_count")
    chrom_start = find(CHROM_ANCHOR)
    eos_start = find(EOS_ANCHOR)

    # Both anchors are ordinary feature columns, so a group drop can DELETE the
    # anchor and leave the block unlocatable. EOS_ANCHOR (`tile_e_mean`) is an
    # eos_other column, so `--drop-groups eos_other` removes it; CHROM_ANCHOR
    # (`neo_h_mean_mean`) is a chrom_level column, likewise. When that happens
    # the block start falls through to n, the PRECEDING block's end expands to
    # swallow the survivors, and they are relabelled as the wrong group.
    #
    # This is analysis-time only -- apply_group_filter() runs resolve_groups on
    # the UNFILTERED names, so every training-time ablation was correct. But the
    # analysis scripts re-resolve groups on the already-filtered vector, and on
    # the final model (which drops eos_other) that silently reported the three
    # surviving eosin clearing columns as `chrom_context` -- a group the same
    # model is documented as having REMOVED. Any driver table built on that
    # would have attributed CC's clearing evidence to chromatin context.
    #
    # So: fall back to a name-set search that does not depend on one column.
    def first_matching(pred, lo=0):
        return next((i for i in range(lo, n) if pred(feat_names[i])), n)

    if eos_start == n:                      # eos-clear names are unique to eosin
        eos_start = first_matching(
            lambda f: any(k in f for k in EOS_CLEAR_KEYS),
            lo=0 if chrom_start == n else chrom_start)
    if chrom_start == n:                    # chromatin is the only H-channel block
        chrom_start = first_matching(
            lambda f: (f in CHROM_FLAG_NAMES or f in CHROM_LOG1P_NAMES
                       or any(f"_{k}_" in f for k in
                              CHROM_LEVEL_KEYS + CHROM_TEXTURE_KEYS)))
        chrom_start = min(chrom_start, eos_start)
    fm_idx = [i for i, f in enumerate(feat_names) if f.startswith(FM_PREFIX)]
    fm_start = fm_idx[0] if fm_idx else n

    # blocks are concatenated graph -> nuclear -> chromatin -> eosin -> fm, but
    # any of them can be absent, so each block ends at whichever starts next.
    # The chromatin block also carries `tile_*` columns, and the eosin block
    # carries its own `tile_*` columns, so the two must be bounded positionally
    # and never selected by that shared prefix across the whole vector.
    later = sorted(s for s in (nuc_start, chrom_start, eos_start, fm_start))
    graph_end = later[0] if later else n

    def block_end(start):
        after = [s for s in (nuc_start, chrom_start, eos_start, fm_start)
                 if s > start]
        return min(after) if after else n

    nuc_end = block_end(nuc_start)
    chrom_end = block_end(chrom_start)
    eos_end = block_end(eos_start)

    out = {}
    for gname, names in FEATURE_GROUPS.items():
        if gname == "graph":
            out[gname] = list(range(graph_end))
        elif gname == "fm":
            out[gname] = fm_idx
        elif gname in EOS_GROUPS:
            rng = range(eos_start, eos_end)
            if gname == "eos":
                out[gname] = list(rng)
            else:
                is_clear = [i for i in rng
                            if any(k in feat_names[i] for k in EOS_CLEAR_KEYS)]
                out[gname] = (is_clear if gname == "eos_clear"
                              else [i for i in rng if i not in set(is_clear)])
        elif gname in CHROM_GROUPS:
            rng = range(chrom_start, chrom_end)
            if gname == "chrom":
                out[gname] = list(rng)
            elif gname == "chrom_context":
                out[gname] = [i for i in rng
                              if feat_names[i] in CHROM_FLAG_NAMES
                              or feat_names[i] in CHROM_LOG1P_NAMES
                              or feat_names[i].startswith("tile_")]
            else:
                keys = (CHROM_LEVEL_KEYS if gname == "chrom_level"
                        else CHROM_TEXTURE_KEYS)
                out[gname] = [i for i in rng
                              if any(f"_{k}_" in feat_names[i] for k in keys)]
        else:
            want = set(names)
            out[gname] = [i for i in range(graph_end, nuc_end)
                          if feat_names[i] in want]
    return out


def apply_group_filter(X, feat_names, drop_groups=None, keep_groups=None):
    """Drop or keep whole feature groups. Runs at the TAIL of load_bags, after
    the cache is built, so every ablation arm shares one bag cache and only the
    outputs split."""
    if not drop_groups and not keep_groups:
        return X, feat_names
    groups = resolve_groups(feat_names)
    unknown = [g for g in list(drop_groups or []) + list(keep_groups or [])
               if g not in groups]
    if unknown:
        raise SystemExit(f"Unknown feature group(s) {unknown}. "
                         f"Known: {sorted(groups)}")
    if keep_groups:
        keep_set = set()
        for g in keep_groups:
            keep_set |= set(groups[g])
    else:
        keep_set = set(range(len(feat_names)))
    for g in (drop_groups or []):
        keep_set -= set(groups[g])
    keep = sorted(keep_set)
    if not keep:
        raise SystemExit("Feature-group filter removed every column.")
    gone = {g: len([i for i in groups[g] if i not in keep_set])
            for g in groups if any(i not in keep_set for i in groups[g])}
    print(f"  feature groups: keep={keep_groups or 'all'} drop={drop_groups or 'none'}")
    print(f"    removed {len(feat_names) - len(keep)} column(s) by group: {gone}")
    print(f"    surviving groups: "
          f"{ {g: len([i for i in idx if i in keep_set]) for g, idx in groups.items()} }")
    X = [b[:, keep] for b in X]
    feat_names = [feat_names[i] for i in keep]
    return X, feat_names


def flag_leak_report(X, y, feat_names):
    """How much label information do the presence flags carry ON THEIR OWN?

    neo_measured availability tracks class hard -- audited over the first 80
    extracted slides: EC 0.97, HGSC 0.86, CC 0.80, LGSC 0.65, MC 0.42, spread
    0.55, Kruskal-Wallis p=1.2e-3. That is the same shape of confound already
    logged for epithelial channel availability, but stronger. Since the flags
    are model inputs, the classifier can read subtype off *availability* without
    ever looking at a nucleus.

    This fits a logistic regression on the per-slide MEAN of the flag columns
    alone, mirroring graph_size_leak() in analyse_sweep2.py. It is a diagnostic
    printed at load time, not a gate -- the decisive test is running the sweep
    with and without --drop-flags and comparing.
    """
    idx = [i for i, f in enumerate(feat_names) if f in ALL_FLAG_NAMES]
    if not idx or len(set(y.tolist())) < 2:
        return
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    Z = np.stack([b[:, idx].mean(0).numpy() for b in X])
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    pred = cross_val_predict(pipe, Z, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))
    bal = balanced_accuracy_score(y, pred)
    print(f"  FLAG LEAK: slide-mean presence flags alone predict subtype at "
          f"balanced_acc {bal:.3f} (chance {1/len(CLASSES):.3f})")
    if bal > 0.30:
        print("    ^ substantial. Run the sweep a second time with --drop-flags;")
        print("      if accuracy barely moves, the morphometrics are carrying it.")
        print("      If it collapses, the result is an availability shortcut.")


def load_extra_block(path, feat_key, name_key, n_nodes, grid_rc_sum):
    """Load one aligned per-tile feature block (<id>_nuclei.npz or
    <id>_chromatin.npz) and verify it lines up with the graph's node order.

    Returns (features, names) or (None, reason). Alignment is checked two ways:
    row count, and the grid_rc_checksum both extractors store. A slide that
    fails either is DROPPED by the caller, never zero-filled -- a partial
    extraction must not be able to quietly bias the corpus.
    """
    if not path.exists():
        return None, "missing"
    try:
        with np.load(path, allow_pickle=True) as z:
            F_ = z[feat_key].astype(np.float32)
            names = [str(v) for v in z[name_key]]
            ck = int(z["grid_rc_checksum"][0]) \
                if "grid_rc_checksum" in z.files else grid_rc_sum
    except Exception as e:
        # the extractors write to the final path directly, so a slide being
        # written right now reads as a truncated zip
        return None, f"unreadable ({type(e).__name__})"
    if F_.shape[0] != n_nodes or ck != grid_rc_sum:
        return None, "misaligned"
    return (F_, names), None


def load_fm_block(path, n_nodes, grid_rc_sum):
    """Load one slide's foundation-model embeddings (<id>_<model>.npz).

    Separate from load_extra_block because the encoders emit an unnamed dense
    vector, not named quantities: the columns get synthetic names fm_0000.. so
    that every downstream mechanism keyed on feat_names (the normaliser, the
    group filter, the degenerate-column sweep) keeps working unchanged. The
    same row-count and grid_rc_checksum alignment guard applies.
    """
    if not path.exists():
        return None, "missing"
    try:
        with np.load(path, allow_pickle=True) as z:
            F_ = z["fm_features"].astype(np.float32)
            ck = int(z["grid_rc_checksum"][0])
    except Exception as e:
        return None, f"unreadable ({type(e).__name__})"
    if F_.shape[0] != n_nodes or ck != grid_rc_sum:
        return None, "misaligned"
    names = [f"{FM_PREFIX}{i:04d}" for i in range(F_.shape[1])]
    return (F_, names), None


def load_bags(gdir, cache, rebuild=False, exclude_prefix=None, nuclear_dir=None,
              drop_flags=False, drop_groups=None, keep_groups=None,
              chromatin_dir=None, eosin_dir=None, fm_dir=None, fm_model=None,
              fm_only=False):
    """
    Each slide becomes a BAG of instances: (n_tiles, n_features). Nothing else
    from the graph npz is read -- no edge_index, no edge_attr, no topology.

    If `nuclear_dir` is given, the per-tile nuclear morphometric features from
    extract_nuclear_features.py (`<id>_nuclei.npz`) are column-concatenated onto
    node_features. They are aligned by construction -- same tiling, same node
    order -- and the alignment is re-checked here via grid_rc_checksum. A slide
    whose nuclei file is missing or misaligned is DROPPED from the sweep (and
    reported), never zero-filled, so a partial extraction can't silently bias
    the result.
    """
    gdir, cache = Path(gdir), Path(cache)
    nuclear_dir = Path(nuclear_dir) if nuclear_dir else None
    chromatin_dir = Path(chromatin_dir) if chromatin_dir else None
    eosin_dir = Path(eosin_dir) if eosin_dir else None
    fm_dir = Path(fm_dir) if fm_dir else None
    stale = (cache.exists()
             and torch.load(cache, weights_only=False).get("schema") != SCHEMA_VERSION)
    if stale:
        print(f"  cache {cache.name} predates loader schema v{SCHEMA_VERSION} "
              f"-- rebuilding")
    if cache.exists() and not rebuild and not stale:
        blob = torch.load(cache, weights_only=False)
        print(f"Loaded {len(blob['X'])} bags from cache {cache.name}")
        print(f"  builder version(s): {blob.get('versions')}")
        if blob.get("nuclear"):
            print(f"  nuclear features: {blob['nuclear']}")
        if blob.get("chromatin"):
            print(f"  chromatin features: {blob['chromatin']}")
        if blob.get("eosin"):
            print(f"  eosin features: {blob['eosin']}")
        if blob.get("fm"):
            print(f"  fm features: {blob['fm']}")
    else:
        files = sorted(gdir.glob("*_graph.npz"))
        if not files:
            raise SystemExit(f"No graphs in {gdir}")
        X, y, ids, feat_names, versions = [], [], [], None, set()
        nuc_names, n_missing, n_misaligned = None, 0, 0
        chrom_names, n_chrom_missing, n_chrom_misaligned = None, 0, 0
        eos_names, n_eos_missing, n_eos_misaligned = None, 0, 0
        fm_names, n_fm_missing, n_fm_misaligned = None, 0, 0
        for p in files:
            with np.load(p, allow_pickle=True) as z:
                label = str(z["label"][0])
                if label not in CLASSES:
                    continue
                iid = str(z["image_id"][0])
                F_ = z["node_features"].astype(np.float32)
                base_names = [str(v) for v in z["node_feature_names"]]
                # channel_valid is a mask the pipeline MUST apply: a False
                # channel means "not measurable here", not "zero".
                for cn, ok in zip([str(c) for c in z["channel_names"]],
                                  np.asarray(z["channel_valid"]).astype(bool)):
                    if not ok:
                        cols = [i for i, f in enumerate(base_names)
                                if f.startswith(cn + "_")]
                        F_[:, cols] = 0.0
                grid_rc_sum = int(np.asarray(z["node_grid_rc"]).sum())
                ver = str(z["script_version"][0]) if "script_version" \
                    in z.files else "?"

            if nuclear_dir is not None:
                got, why = load_extra_block(
                    nuclear_dir / f"{iid}_nuclei.npz", "nuc_features",
                    "nuc_feature_names", F_.shape[0], grid_rc_sum)
                if got is None:
                    if why == "misaligned":
                        n_misaligned += 1
                    else:
                        if why != "missing":
                            print(f"  {iid} nuclei {why} -- dropped")
                        n_missing += 1
                    continue
                NF, this_nuc_names = got
                # spacing-feature presence flags the extractor does not ship
                NF, this_nuc_names = append_derived_flags(NF, this_nuc_names)
                F_ = np.concatenate([F_, NF], axis=1)
                base_names = base_names + this_nuc_names
                nuc_names = this_nuc_names

            if chromatin_dir is not None:
                got, why = load_extra_block(
                    chromatin_dir / f"{iid}_chromatin.npz", "chrom_features",
                    "chrom_feature_names", F_.shape[0], grid_rc_sum)
                if got is None:
                    if why == "misaligned":
                        n_chrom_misaligned += 1
                    else:
                        if why != "missing":
                            print(f"  {iid} chromatin {why} -- dropped")
                        n_chrom_missing += 1
                    continue
                CF, this_chrom_names = got
                F_ = np.concatenate([F_, CF], axis=1)
                base_names = base_names + this_chrom_names
                chrom_names = this_chrom_names

            if eosin_dir is not None:
                got, why = load_extra_block(
                    eosin_dir / f"{iid}_eosin.npz", "eos_features",
                    "eos_feature_names", F_.shape[0], grid_rc_sum)
                if got is None:
                    if why == "misaligned":
                        n_eos_misaligned += 1
                    else:
                        if why != "missing":
                            print(f"  {iid} eosin {why} -- dropped")
                        n_eos_missing += 1
                    continue
                EF, this_eos_names = got
                F_ = np.concatenate([F_, EF], axis=1)
                base_names = base_names + this_eos_names
                eos_names = this_eos_names

            if fm_dir is not None:
                got, why = load_fm_block(
                    fm_dir / f"{iid}_{fm_model}.npz", F_.shape[0], grid_rc_sum)
                if got is None:
                    if why == "misaligned":
                        n_fm_misaligned += 1
                    else:
                        if why != "missing":
                            print(f"  {iid} fm {why} -- dropped")
                        n_fm_missing += 1
                    continue
                MF, this_fm_names = got
                if fm_only:
                    # the positive control proper: the encoder's view and
                    # nothing else, so any gap is the representation alone
                    F_, base_names = MF, list(this_fm_names)
                else:
                    F_ = np.concatenate([F_, MF], axis=1)
                    base_names = base_names + this_fm_names
                fm_names = this_fm_names

            if feat_names is None:
                feat_names = base_names
            versions.add(ver)
            X.append(torch.from_numpy(F_))
            y.append(CLASSES.index(label))
            ids.append(iid)

        blob = {"X": X, "y": np.array(y), "ids": ids, "schema": SCHEMA_VERSION,
                "feat_names": feat_names, "versions": sorted(versions)}
        if nuclear_dir is not None:
            blob["nuclear"] = (f"{len(nuc_names or [])} cols concatenated; "
                               f"{n_missing} slides missing nuclei npz, "
                               f"{n_misaligned} misaligned -- all dropped")
        if chromatin_dir is not None:
            blob["chromatin"] = (f"{len(chrom_names or [])} cols concatenated; "
                                 f"{n_chrom_missing} slides missing chromatin "
                                 f"npz, {n_chrom_misaligned} misaligned -- "
                                 f"all dropped")
        if eosin_dir is not None:
            blob["eosin"] = (f"{len(eos_names or [])} cols concatenated; "
                             f"{n_eos_missing} slides missing eosin npz, "
                             f"{n_eos_misaligned} misaligned -- all dropped")
        if fm_dir is not None:
            blob["fm"] = (f"{len(fm_names or [])} cols from {fm_model} "
                          f"({'REPLACING' if fm_only else 'concatenated with'} "
                          f"the hand-built features); {n_fm_missing} slides "
                          f"missing fm npz, {n_fm_misaligned} misaligned -- "
                          f"all dropped")
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, cache)
        print(f"Built {len(X)} bags -> {cache.name}")
        print(f"  builder version(s): {blob['versions']}")
        if nuclear_dir is not None:
            print(f"  nuclear features: {blob['nuclear']}")
        if chromatin_dir is not None:
            print(f"  chromatin features: {blob['chromatin']}")
        if eosin_dir is not None:
            print(f"  eosin features: {blob['eosin']}")
        if fm_dir is not None:
            print(f"  fm features: {blob['fm']}")

    X, y, ids, feat_names = blob["X"], blob["y"], blob["ids"], blob["feat_names"]
    if exclude_prefix:
        keep = [i for i, f in enumerate(feat_names)
                if not f.startswith(tuple(exclude_prefix))]
        dropped = len(feat_names) - len(keep)
        print(f"  excluding {dropped} features matching {exclude_prefix}")
        X = [b[:, keep] for b in X]
        feat_names = [feat_names[i] for i in keep]

    # report the availability leak BEFORE any flag dropping, so the number is
    # comparable between the two arms of the ablation
    flag_leak_report(X, y, feat_names)

    if drop_flags:
        keep = [i for i, f in enumerate(feat_names) if f not in ALL_FLAG_NAMES]
        gone = [f for f in feat_names if f in ALL_FLAG_NAMES]
        print(f"  --drop-flags: removing {len(gone)} presence flag(s): {gone}")
        print("    NOTE: 0 in a gated morphometric column now means BOTH 'not")
        print("    measured' and a genuine zero. That ambiguity is the price of")
        print("    the ablation -- this arm is the control, not the headline run.")
        X = [b[:, keep] for b in X]
        feat_names = [feat_names[i] for i in keep]

    X, feat_names = apply_group_filter(X, feat_names, drop_groups, keep_groups)
    X, feat_names = drop_degenerate(X, feat_names)

    sizes = np.array([b.shape[0] for b in X])
    print(f"  bags={len(X)}  features={len(feat_names)}  "
          f"instances/bag median={np.median(sizes):.0f} "
          f"min={sizes.min()} max={sizes.max()}")
    return X, y, ids, feat_names


class Normaliser:
    """log1p the count-like columns, then z-score. Fitted on the train split.
    Nuclear 0/1 presence flags (NUC_FLAG_NAMES) are passed through untouched."""

    def __init__(self, feat_names):
        self.log_idx = [i for i, f in enumerate(feat_names)
                        if (f.startswith(LOG1P_PREFIXES)
                            and not any(x in f for x in LOG1P_EXCLUDE))
                        or f in NUC_LOG1P_NAMES or f in CHROM_LOG1P_NAMES
                        or f in EOS_LOG1P_NAMES]
        self.flag_idx = [i for i, f in enumerate(feat_names)
                         if f in ALL_FLAG_NAMES]
        self.mean = self.std = None

    def fit(self, bags):
        A = torch.cat([b for b in bags], 0).clone()
        A[:, self.log_idx] = torch.log1p(A[:, self.log_idx].clamp(min=0))
        self.mean, self.std = A.mean(0), A.std(0).clamp(min=1e-6)
        if self.flag_idx:                       # identity on presence flags
            self.mean[self.flag_idx] = 0.0
            self.std[self.flag_idx] = 1.0
        return self

    def __call__(self, x):
        x = x.clone()
        x[..., self.log_idx] = torch.log1p(x[..., self.log_idx].clamp(min=0))
        return (x - self.mean) / self.std

    def state_dict(self):
        return {"log_idx": self.log_idx,
                "mean": self.mean.cpu(), "std": self.std.cpu()}

    def to(self, dev):
        self.mean, self.std = self.mean.to(dev), self.std.to(dev)
        return self


def pad_collate(batch):
    """Variable-size bags -> (B, N_max, F) plus a validity mask."""
    xs, ys = zip(*batch)
    n_max = max(x.shape[0] for x in xs)
    B, Fdim = len(xs), xs[0].shape[1]
    X = torch.zeros(B, n_max, Fdim)
    M = torch.zeros(B, n_max, dtype=torch.bool)
    for i, x in enumerate(xs):
        X[i, :x.shape[0]] = x
        M[i, :x.shape[0]] = True
    return X, M, torch.tensor(ys, dtype=torch.long)


MIN_SUBSAMPLE_TILES = 32


class BagDS(torch.utils.data.Dataset):
    """`subsample` (0,1] keeps a random fraction of each bag's tiles, redrawn
    every epoch. TRAIN SPLIT ONLY -- val/test must see the whole bag or the
    metric moves around for the wrong reason.

    Why: the train/val gap is wide (train ~0.88 vs val ~0.67) and there is
    otherwise no bag-level regularisation at all -- every epoch sees the exact
    same tile set per slide. Randomly dropping tiles is the standard MIL
    augmentation (CLAM et al.) and is the natural one here: attention pooling
    already normalises over whatever tiles are present, so a subsampled bag is
    still a valid bag, just a smaller sample of the same slide.
    """

    def __init__(self, bags, labels, subsample=None):
        self.bags, self.labels = bags, labels
        self.subsample = subsample

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        b = self.bags[i]
        if self.subsample and 0 < self.subsample < 1.0:
            n = b.shape[0]
            k = max(MIN_SUBSAMPLE_TILES, int(round(n * self.subsample)))
            if k < n:
                b = b[torch.randperm(n)[:k]]
        return b, int(self.labels[i])


class ABMIL(nn.Module):
    """
    Instance encoder -> gated attention pooling -> classifier.

    The attention weight sums to 1 over the VALID instances of each bag, so the
    pooled vector is a weighted mean and stays invariant to bag size. That is
    what keeps the graph-size-by-subtype leak (MC median 966 tiles vs LGSC 500)
    from entering through the pooling step.

    That weight is also the interpretable quantity: "this tile contributed 4% of
    the slide representation" is literally true of it. Same mechanism the v5
    deletion test validated at 14-27x over a density-matched null.
    """

    def __init__(self, in_dim, n_classes, hidden=128, layers=1,
                 attn_hidden=128, dropout=0.3, pool="attn", heads=1):
        super().__init__()
        self.pool_kind = pool
        self.heads = heads
        enc, d = [], in_dim
        for _ in range(layers):
            enc += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.encoder = nn.Sequential(*enc)
        self.attn_hidden = attn_hidden
        if heads == 1:
            # kept structurally identical to the pre-Aug-31 model so existing
            # checkpoints (abmil_ckpt/v5, /v5nuc, /v5nucnoflag) still load
            self.V = nn.Linear(d, attn_hidden)
            self.U = nn.Linear(d, attn_hidden)
            self.w = nn.Linear(attn_hidden, 1)
            pooled = d
        else:
            self.Vh = nn.Linear(d, attn_hidden * heads)
            self.Uh = nn.Linear(d, attn_hidden * heads)
            self.wh = nn.Parameter(torch.randn(heads, attn_hidden) * 0.02)
            pooled = d * heads
        self.head = nn.Sequential(nn.Linear(pooled, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x, mask, return_attn=False, y=None):
        # `y` is accepted and ignored -- CLAM needs the label inside forward to
        # build its instance loss, and run_epoch calls every arch identically.
        h = self.encoder(x)                                  # (B, N, d)
        if self.pool_kind == "attn" and self.heads > 1:
            B, N, d = h.shape
            H, A = self.heads, self.attn_hidden
            v = torch.tanh(self.Vh(h)).view(B, N, H, A)
            u = torch.sigmoid(self.Uh(h)).view(B, N, H, A)
            a = torch.einsum("bnha,ha->bnh", v * u, self.wh)   # (B, N, H)
            a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            a = torch.softmax(a, dim=1)                        # over instances
            a = torch.nan_to_num(a)
            g = torch.einsum("bnh,bnd->bhd", a, h).reshape(B, H * d)
        elif self.pool_kind == "attn":
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            a = a.masked_fill(~mask, float("-inf"))
            a = torch.softmax(a, dim=1)
            a = torch.nan_to_num(a)                          # empty bag guard
            g = (h * a.unsqueeze(-1)).sum(1)
        elif self.pool_kind == "mean":
            m = mask.unsqueeze(-1).float()
            g = (h * m).sum(1) / m.sum(1).clamp(min=1)
            a = None
        elif self.pool_kind == "max":
            g = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
            g = torch.nan_to_num(g, neginf=0.0)
            a = None
        else:
            raise ValueError(self.pool_kind)
        out = self.head(g)
        return (out, a) if return_attn else out


class GatedAttention(nn.Module):
    """Ilse et al. gated attention, softmaxed over the valid instances.

    Structurally the same mechanism as ABMIL's inline version, factored out so
    the additive / CLAM / TransMIL heads can share it. Kept SEPARATE from
    ABMIL's own V/U/w so that the pre-Sep-2 checkpoints still load by name.
    Returns (B, N, heads); heads==1 still returns a trailing axis of 1.
    """

    def __init__(self, d, attn_hidden, heads=1):
        super().__init__()
        self.heads, self.attn_hidden = heads, attn_hidden
        self.Va = nn.Linear(d, attn_hidden * heads)
        self.Ua = nn.Linear(d, attn_hidden * heads)
        self.wa = nn.Parameter(torch.randn(heads, attn_hidden) * 0.02)

    def forward(self, h, mask):
        B, N, _ = h.shape
        H, A = self.heads, self.attn_hidden
        v = torch.tanh(self.Va(h)).view(B, N, H, A)
        u = torch.sigmoid(self.Ua(h)).view(B, N, H, A)
        a = torch.einsum("bnha,ha->bnh", v * u, self.wa)
        a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return torch.nan_to_num(torch.softmax(a, dim=1))


class AdditiveMIL(nn.Module):
    """Additive MIL (Javed et al. 2022): per-instance class scores, summed.

        logit_c = sum_n  a_n * psi(h_n)_c

    NOTHING runs after the pooling step -- no MLP on the pooled vector -- so
    tile n's contribution to class c is EXACTLY a_n * psi(h_n)_c, and those
    contributions add up to the logit. That is a strictly stronger
    interpretability claim than ABMIL's: attention alone says where the model
    looked, this says what it concluded there and how much it mattered, per
    class, with an exact accounting identity rather than a post-hoc
    attribution. It is also directly checkable -- `contributions.sum(1)` must
    equal the returned logits.

    With heads>1 the per-head additive decompositions are averaged, which
    preserves additivity (a mean of additive terms is additive).
    """

    def __init__(self, in_dim, n_classes, hidden=128, layers=1, attn_hidden=128,
                 dropout=0.3, heads=1):
        super().__init__()
        enc, d = [], in_dim
        for _ in range(layers):
            enc += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.encoder = nn.Sequential(*enc)
        self.attn = GatedAttention(d, attn_hidden, heads)
        self.inst = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        self.heads = heads

    def forward(self, x, mask, return_attn=False, return_contrib=False, y=None):
        h = self.encoder(x)
        a = self.attn(h, mask)                              # (B, N, H)
        psi = self.inst(h)                                  # (B, N, C)
        contrib = torch.einsum("bnh,bnc->bnhc", a, psi).mean(2)   # (B, N, C)
        contrib = contrib.masked_fill(~mask.unsqueeze(-1), 0.0)
        out = contrib.sum(1)
        amap = a.mean(-1)                                   # (B, N) for analysis
        if return_contrib:
            return out, amap, contrib
        return (out, amap) if return_attn else out


class CLAM(nn.Module):
    """CLAM-SB (Lu et al. 2021): ABMIL plus an instance-level clustering loss.

    The bag path is identical to ABMIL. The addition is auxiliary supervision
    ON THE ATTENTION ITSELF: for each bag, the top-k attended instances are
    pushed to be classifiable as evidence FOR the true class and the bottom-k
    as evidence against, through per-class linear instance classifiers.

    Why this is the relevant experiment here rather than an accuracy play:
    nothing currently constrains the attention map beyond "whatever pooled
    vector minimises bag loss", and many different maps give the same pooled
    vector. That degeneracy is a plausible driver of the seed-to-seed
    disagreement (Task 3b). The clustering loss removes some of it by giving
    the attention its own gradient signal.

    `clam_k` is clamped to n_valid//2 per bag so top-k and bottom-k can never
    overlap on a small bag.
    """

    def __init__(self, in_dim, n_classes, hidden=128, layers=1, attn_hidden=128,
                 dropout=0.3, heads=1, clam_k=8, clam_weight=0.3):
        super().__init__()
        enc, d = [], in_dim
        for _ in range(layers):
            enc += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.encoder = nn.Sequential(*enc)
        self.attn = GatedAttention(d, attn_hidden, heads)
        self.head = nn.Sequential(nn.Linear(d * heads, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        self.inst_cls = nn.ModuleList([nn.Linear(d, 2) for _ in range(n_classes)])
        self.n_classes, self.heads = n_classes, heads
        self.clam_k, self.clam_weight = clam_k, clam_weight
        self.last_aux = None

    def _instance_loss(self, h, a, mask, y):
        """In-class + out-of-class instance CE over the top/bottom-k attended."""
        losses = []
        for b in range(h.shape[0]):
            n = int(mask[b].sum())
            k = min(self.clam_k, n // 2)
            if k < 1:
                continue
            av = a[b, :n]
            order = torch.argsort(av, descending=True)
            top, bot = order[:k], order[-k:]
            hb = h[b, :n]
            c = int(y[b])
            # in-class: the most attended instances ARE this class's evidence,
            # the least attended are not
            logit = self.inst_cls[c](torch.cat([hb[top], hb[bot]], 0))
            tgt = torch.cat([torch.ones(k, dtype=torch.long, device=h.device),
                             torch.zeros(k, dtype=torch.long, device=h.device)])
            losses.append(F.cross_entropy(logit, tgt))
            # out-of-class: the same top instances are NOT evidence for the
            # other classes. One randomly chosen other class per bag keeps the
            # cost O(1) instead of O(C).
            others = [j for j in range(self.n_classes) if j != c]
            oc = others[int(torch.randint(len(others), (1,)).item())]
            lo = self.inst_cls[oc](hb[top])
            losses.append(F.cross_entropy(
                lo, torch.zeros(k, dtype=torch.long, device=h.device)))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def forward(self, x, mask, return_attn=False, y=None):
        h = self.encoder(x)
        a = self.attn(h, mask)                              # (B, N, H)
        g = torch.einsum("bnh,bnd->bhd", a, h).reshape(h.shape[0], -1)
        out = self.head(g)
        amap = a.mean(-1)
        self.last_aux = None
        if y is not None and self.training:
            il = self._instance_loss(h, amap, mask, y)
            if il is not None:
                self.last_aux = self.clam_weight * il
        return (out, amap) if return_attn else out


class TransMIL(nn.Module):
    """TransMIL-style: self-attention BETWEEN tiles, then the same pooling.

    Deliberately not the literal paper architecture (no PPEG convolutional
    position encoding, no Nystrom approximation). The point here is a clean
    A/B: everything is held identical to ABMIL except that the instances are
    contextualised by a transformer encoder before pooling. So if this wins,
    the win is attributable to tile-tile interaction and nothing else -- and
    the pooling weight stays the same interpretable quantity, so every
    downstream analysis script works on these checkpoints unchanged.

    Prior expectation is low: the v5 GAT sweep already found message passing
    over these tiles worthless. But that used FIXED streamline edges, and this
    LEARNS which tiles should talk, which is a genuinely different test.
    """

    def __init__(self, in_dim, n_classes, hidden=128, layers=2, attn_hidden=128,
                 dropout=0.3, heads=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.attn = GatedAttention(hidden, attn_hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x, mask, return_attn=False, y=None):
        h = self.proj(x)
        # src_key_padding_mask marks positions to IGNORE -> invert the validity
        h = self.tr(h, src_key_padding_mask=~mask)
        h = torch.nan_to_num(self.norm(h))   # fully-padded rows can emit NaN
        a = self.attn(h, mask)
        g = torch.einsum("bnh,bnd->bhd", a, h).reshape(h.shape[0], -1)
        out = self.head(g)
        return (out, a.mean(-1)) if return_attn else out


def build_model(cfg, in_dim, n_classes, clam_k=8, clam_weight=0.3):
    """Config tuple -> model. `arch` defaults to abmil for pre-Sep-2 tuples."""
    _, hidden, layers, attn_hidden, dropout, _, _, pool = cfg[:8]
    heads = cfg_field(cfg, "heads", 1)
    arch = cfg_field(cfg, "arch", "abmil")
    if arch == "abmil":
        return ABMIL(in_dim, n_classes, hidden, layers, attn_hidden, dropout,
                     pool, heads)
    if arch == "additive":
        return AdditiveMIL(in_dim, n_classes, hidden, layers, attn_hidden,
                           dropout, heads)
    if arch == "clam":
        return CLAM(in_dim, n_classes, hidden, layers, attn_hidden, dropout,
                    heads, clam_k=clam_k, clam_weight=clam_weight)
    if arch == "transmil":
        return TransMIL(in_dim, n_classes, hidden, layers, attn_hidden,
                        dropout, heads)
    raise ValueError(f"unknown arch {arch!r}")


def bag_mixup(X, M, alpha, gen=None):
    """Build mixed bags by drawing instances from two slides.

    WHY MIXUP AT THE BAG LEVEL AND NOT THE TILE LEVEL
    -------------------------------------------------
    Tile subsampling (--bag-subsample, Task 3c(a)) was a clean null: dropping
    half of a ~640-tile bag barely moved the train/val gap, because attention
    pooling averages over whatever is present and the slide-level statistic
    survives. The diagnosis was that the overfitting is at the SLIDE level --
    456 slides, train balanced accuracy 0.99 -- so no amount of resampling
    within a slide can help.

    This operates at exactly that level. Each mixed bag is a new slide that
    does not exist in the corpus, carrying a soft label, so the effective
    number of distinct training slides is no longer 456.

    Instances are mixed rather than embeddings because it works unchanged for
    every architecture (the model never learns it happened) and because a bag
    of tiles from two slides is at least a coherent object -- MIL assumes the
    bag is a set, and this respects that.

    The label weight is the ACHIEVED instance fraction, not the drawn lambda:
    bags differ in size by 2x, so a nominal 50/50 draw from a 966-tile MC slide
    and a 500-tile LGSC slide is not a 50/50 bag, and labelling it as one would
    inject systematic noise correlated with class (graph size tracks class hard
    -- Section 4).
    """
    B, N, _ = X.shape
    dev = X.device
    # A random permutation has fixed points -- about one bag per batch of 8 --
    # and a bag mixed with ITSELF is not augmented at all: same label on both
    # sides, instances drawn twice from one slide. That silently lowers the
    # effective mixup rate below the nominal one. A cyclic shift by a non-zero
    # offset is a derangement by construction, so every bag gets a genuine
    # partner.
    off = int(torch.randint(1, max(B, 2), (1,), generator=gen).item())
    perm = (torch.arange(B, device=dev) + off) % B
    lam = torch.from_numpy(
        np.random.beta(alpha, alpha, size=B).astype(np.float32)).to(dev)

    keep_a = (torch.rand(B, N, device=dev, generator=gen) < lam[:, None]) & M
    keep_b = (torch.rand(B, N, device=dev, generator=gen)
              < (1.0 - lam)[:, None]) & M[perm]

    na, nb = keep_a.sum(1), keep_b.sum(1)
    # a bag that lost every instance from one side is not a mix; fall back to
    # that side's original bag rather than emitting an empty or degenerate bag
    dead = (na == 0) | (nb == 0)
    if dead.any():
        keep_a[dead] = M[dead]
        keep_b[dead] = False
        na, nb = keep_a.sum(1), keep_b.sum(1)

    X_mix = torch.cat([X, X[perm]], dim=1)
    M_mix = torch.cat([keep_a, keep_b], dim=1)
    lam_eff = na.float() / (na + nb).float().clamp(min=1)
    return X_mix, M_mix, perm, lam_eff


def run_epoch(model, loader, norm, device, opt=None, w=None, mixup=0.0):
    train = opt is not None
    model.train(train)
    ys, ps = [], []
    for X, M, y in loader:
        X, M, y = X.to(device), M.to(device), y.to(device)
        use_mix = train and mixup > 0 and X.shape[0] > 1
        with torch.set_grad_enabled(train):
            if use_mix:
                Xm, Mm, perm, lam = bag_mixup(norm(X), M, mixup)
                # y=None disables CLAM's instance loss, which is undefined on a
                # bag with two labels
                out = model(Xm, Mm, y=None)
                loss = (lam * F.cross_entropy(out, y, weight=w, reduction="none")
                        + (1 - lam) * F.cross_entropy(out, y[perm], weight=w,
                                                      reduction="none")).mean()
            else:
                out = model(norm(X), M, y=y)
                loss = F.cross_entropy(out, y, weight=w)
            aux = getattr(model, "last_aux", None)   # CLAM instance loss
            if train and aux is not None:
                loss = loss + aux
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        ys.append(y.cpu())
        ps.append(out.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    # under mixup the reported "train accuracy" is scored against the FIRST
    # bag's label on a two-label bag, so it understates; it is a trajectory
    # diagnostic here, not a number to compare across mixup settings
    return balanced_accuracy_score(y, p), y, p


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _average_states(states):
    """Element-wise mean of a list of state dicts (SWA).

    Safe only because the models here carry no BatchNorm running statistics --
    every buffer is a parameter. Integer/bool entries would be corrupted by
    averaging, so they are taken from the first state instead."""
    if len(states) == 1:
        return states[0]
    out = {}
    for k in states[0]:
        v0 = states[0][k]
        if v0.is_floating_point():
            acc = torch.zeros_like(v0, dtype=torch.float64)
            for s in states:
                acc += s[k].double()
            out[k] = (acc / len(states)).to(v0.dtype)
        else:
            out[k] = v0.clone()
    return out


def _smooth(v, window):
    """Centred moving average, edge-padded so length is preserved."""
    v = np.asarray(v, dtype=float)
    if window <= 1 or len(v) < 2:
        return v
    w = min(window, len(v))
    pad = w // 2
    padded = np.pad(v, (pad, pad), mode="edge")
    kern = np.ones(w) / w
    return np.convolve(padded, kern, mode="valid")[:len(v)]


def select_state(mode, states, val_curve, smooth_window=SMOOTH_WINDOW,
                 swa_start=SWA_START_FRAC, swa_top_k=SWA_TOP_K):
    """Pick (or build) the final weights from the recorded trajectory.

    Returns (state_dict, description) where description records what was
    actually selected, so the results CSV can show e.g. which epoch argmax
    landed on vs where smooth landed."""
    E = len(states)
    if mode == "argmax":
        ep = int(np.argmax(val_curve))
        return states[ep], f"epoch {ep}/{E - 1}"
    if mode == "smooth":
        ep = int(np.argmax(_smooth(val_curve, smooth_window)))
        return states[ep], f"epoch {ep}/{E - 1} (smoothed w={smooth_window})"
    if mode == "fixed":
        return states[-1], f"epoch {E - 1}/{E - 1} (last)"
    if mode == "swa":
        i0 = min(int(round(swa_start * E)), E - 1)
        return _average_states(states[i0:]), f"SWA over epochs {i0}-{E - 1}"
    if mode == "swatop":
        k = min(swa_top_k, E)
        idx = np.argsort(val_curve)[::-1][:k]
        return _average_states([states[i] for i in idx]), \
            f"SWA over top-{k} val epochs"
    raise ValueError(f"unknown select mode {mode!r}")


def one_run(cfg, X, y_all, tr_idx, te_idx, feat_names, seed, device, epochs,
            ckpt_path=None, subsample=None, select="argmax", patience=PATIENCE,
            sched_kind="cosine", smooth_window=SMOOTH_WINDOW,
            swa_start=SWA_START_FRAC, swa_top_k=SWA_TOP_K,
            clam_k=8, clam_weight=0.3, mixup=0.0):
    name, hidden, layers, attn_hidden, dropout, lr, wd, pool = cfg[:8]
    arch = cfg_field(cfg, "arch", "abmil")

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tr2, va = next(inner.split(np.zeros(len(tr_idx)), y_all[tr_idx]))
    tr_i, va_i = tr_idx[tr2], tr_idx[va]

    batch = ARCH_BATCH.get(arch, BATCH)
    norm = Normaliser(feat_names).fit([X[i] for i in tr_i]).to(device)
    mk = lambda idx, sh, ss=None: torch.utils.data.DataLoader(
        BagDS([X[i] for i in idx], y_all[idx], subsample=ss), batch_size=batch,
        shuffle=sh, collate_fn=pad_collate)
    # augmentation on the train split only
    tl, vl, el = mk(tr_i, True, subsample), mk(va_i, False), mk(te_idx, False)

    w = torch.tensor(len(tr_i) / (len(CLASSES) *
                     np.bincount(y_all[tr_i], minlength=len(CLASSES)).clip(min=1)),
                     dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(cfg, X[0].shape[1], len(CLASSES),
                        clam_k=clam_k, clam_weight=clam_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if sched_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif sched_kind == "cosine_floor":
        # anneal to lr/10 and hold, so the tail of training keeps moving and
        # SWA has genuinely different points to average
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * LR_FLOOR_FRAC)
    elif sched_kind == "const":
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
    else:
        raise ValueError(f"unknown schedule {sched_kind!r}")

    # argmax with a patience is the incumbent rule and only ever needs the
    # running best, so it keeps the cheap path. Every other mode needs the
    # whole trajectory in memory (these models are 50k-400k params, so 150
    # CPU-side snapshots is tens to a couple hundred MB).
    keep_all = select != "argmax" or patience <= 0
    states, val_curve, train_curve = [], [], []
    best, best_state, bad, best_tra = -1.0, None, 0, np.nan
    for _ in range(epochs):
        tra, _, _ = run_epoch(model, tl, norm, device, opt, w, mixup=mixup)
        vla, _, _ = run_epoch(model, vl, norm, device)
        sched.step()
        val_curve.append(float(vla))
        train_curve.append(float(tra))
        if keep_all:
            states.append(_cpu_state(model))
            continue
        if vla > best:
            best, bad, best_tra = vla, 0, tra
            best_state = _cpu_state(model)
        else:
            bad += 1
        if bad >= patience:
            break

    if keep_all:
        best_state, sel_desc = select_state(select, states, val_curve,
                                            smooth_window, swa_start, swa_top_k)
        states.clear()
    else:
        sel_desc = f"epoch {int(np.argmax(val_curve))}/{len(val_curve) - 1} " \
                   f"(early stop, patience {patience})"

    model.load_state_dict(best_state)
    if keep_all:
        # the selected weights may never have been evaluated (SWA builds a
        # point that was never visited), so val has to be recomputed for the
        # reported number to mean anything
        best, _, _ = run_epoch(model, vl, norm, device)
        best_tra = float(np.mean(train_curve[-10:]))
    _, ty, tp = run_epoch(model, el, norm, device)

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "normaliser": norm.state_dict(),
                    "feat_names": feat_names, "classes": CLASSES,
                    "in_dim": X[0].shape[1], "seed": seed,
                    "select": select, "select_detail": sel_desc,
                    "val_curve": val_curve,
                    "config": dict(zip(COLS, cfg))}, ckpt_path)
    return ty, tp, best, best_tra, sel_desc, len(val_curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--arch", nargs="+", default=None,
                    metavar="A",
                    help="run only configs with these architectures: "
                         "abmil, additive, clam, transmil")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--select", default="argmax", choices=list(SELECT_MODES),
                    help="how the final weights are chosen from the training "
                         "trajectory. Anything other than plain argmax+patience "
                         "trains the full --epochs with NO early stop, so the "
                         "modes are paired on identical trajectories and the "
                         "delta isolates the selection rule.")
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="early-stopping patience for --select argmax. Set 0 to "
                         "disable, which gives the clean argmax control for the "
                         "selection-rule comparison.")
    ap.add_argument("--sched", default="cosine", choices=list(SCHEDULES),
                    help="lr schedule. cosine_floor stops at lr/10 instead of 0 "
                         "so the tail of training keeps exploring -- pair it "
                         "with --select swa, which has nothing to average when "
                         "the lr has already annealed to zero.")
    ap.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    ap.add_argument("--swa-start", type=float, default=SWA_START_FRAC,
                    help="fraction of training after which SWA starts averaging")
    ap.add_argument("--swa-top-k", type=int, default=SWA_TOP_K)
    ap.add_argument("--clam-k", type=int, default=8,
                    help="instances per bag used by the CLAM clustering loss")
    ap.add_argument("--clam-weight", type=float, default=0.3)
    ap.add_argument("--exclude-prefix", nargs="+", default=None,
                    help="drop node features with these prefixes, e.g. "
                         "--exclude-prefix neoplastic_ connective_")
    ap.add_argument("--noise-floor", action="store_true",
                    help="re-run the best config a second time under a "
                         "different torch seed to measure run-to-run spread "
                         "without PyG scatter in the loop")
    ap.add_argument("--nuclear-features", action="store_true",
                    help="concatenate the per-tile nuclear morphometric features "
                         "(extract_nuclear_features.py) onto node_features")
    ap.add_argument("--nuclear-dir", default=None,
                    help="dir of <id>_nuclei.npz (default: graphs/nuclear_feats)")
    ap.add_argument("--chromatin-features", action="store_true",
                    help="also concatenate the per-tile chromatin / texture "
                         "features (extract_chromatin_features.py). These are "
                         "the first features taken from WSI PIXELS rather than "
                         "from CellViT geometry -- hyperchromasia, chromatin "
                         "coarseness, GLCM texture. Implies --nuclear-features.")
    ap.add_argument("--chromatin-dir", default=None,
                    help="dir of <id>_chromatin.npz "
                         "(default: graphs/chromatin_feats)")
    ap.add_argument("--eosin-features", action="store_true",
                    help="also concatenate the per-tile eosin / cytoplasm "
                         "features (extract_eosin_features.py): clearing, "
                         "peri-nuclear eosin, N:C proxy. These target CC and "
                         "MC, whose diagnostic criteria live in the cytoplasm "
                         "and are invisible to every other block. Implies "
                         "--nuclear-features.")
    ap.add_argument("--eosin-dir", default=None,
                    help="dir of <id>_eosin.npz (default: graphs/eosin_feats)")
    ap.add_argument("--fm-features", default=None, metavar="MODEL",
                    help="load foundation-model tile embeddings from "
                         "embed_tiles_fm.py (e.g. phikonv2, phikon). By "
                         "default these REPLACE the hand-built features -- "
                         "that is the Task 4 positive control. Use "
                         "--fm-concat to append them instead.")
    ap.add_argument("--fm-concat", action="store_true",
                    help="with --fm-features, keep the hand-built features and "
                         "append the embeddings rather than replacing them "
                         "(the hybrid arm: does an interpretable block still "
                         "add anything on top of a foundation model?)")
    ap.add_argument("--fm-dir", default=None,
                    help="dir of <id>_<model>.npz (default: graphs/fm_feats)")
    ap.add_argument("--drop-groups", nargs="+", default=None,
                    metavar="G",
                    help="ablation: remove whole feature groups. Choices: "
                         "graph, nuc_shape, nuc_spatial, nuc_density, "
                         "nuc_flags. e.g. --drop-groups graph gives the "
                         "nuclear-only arm.")
    ap.add_argument("--keep-groups", nargs="+", default=None, metavar="G",
                    help="ablation: keep ONLY these feature groups (applied "
                         "before --drop-groups). Same choices.")
    ap.add_argument("--bag-subsample", type=float, default=None, metavar="F",
                    help="MIL augmentation: keep a random F fraction of each "
                         "bag's tiles per epoch, train split only (e.g. 0.8). "
                         "Targets the wide train/val gap -- there is otherwise "
                         "no bag-level regularisation. Auto-tags the outputs.")
    ap.add_argument("--mixup", type=float, default=0.0, metavar="ALPHA",
                    help="MIL augmentation: build training bags by drawing "
                         "instances from two slides with a Beta(a,a) ratio and "
                         "a correspondingly soft label (e.g. 0.4). Unlike "
                         "--bag-subsample this acts at the SLIDE level, which "
                         "is where the overfitting is (train bal ~0.99 on 456 "
                         "slides). Train split only. Auto-tags the outputs.")
    ap.add_argument("--arm", default=None,
                    help="suffix appended to the output tag, so an ablation "
                         "arm gets its own results CSV and checkpoint dir "
                         "instead of colliding with the headline run")
    ap.add_argument("--drop-flags", action="store_true",
                    help="ablation: remove the nuclear presence-flag columns "
                         "(morph/neo/til/nnd_*_measured). Their availability "
                         "tracks subtype hard (EC 0.97 vs MC 0.42), so the model "
                         "can read class off availability alone. Run the sweep "
                         "with and without this and compare.")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()

    repeats = 1 if args.smoke else args.repeats
    folds = 2 if args.smoke else FOLDS
    epochs = 8 if args.smoke else args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gdir = Path(args.root) / RUN_DIR_NAME / GRAPH_SUBFOLDER
    odir = Path(args.root) / RUN_DIR_NAME / OUT_SUBFOLDER
    odir.mkdir(parents=True, exist_ok=True)

    nuclear_dir = chromatin_dir = eosin_dir = fm_dir = None
    tag = args.tag
    if args.chromatin_features or args.eosin_features:
        # the pixel blocks are additions to the nuclear block, not alternatives
        # to it -- they describe the same tiles
        args.nuclear_features = True
    fm_only = bool(args.fm_features) and not args.fm_concat
    if fm_only:
        # the control replaces the features outright, so loading the hand-built
        # blocks would only waste time building columns that get discarded
        args.nuclear_features = args.chromatin_features = False
        args.eosin_features = False
    if args.nuclear_features:
        nuclear_dir = Path(args.nuclear_dir) if args.nuclear_dir \
            else gdir / "nuclear_feats"
        if "nuc" not in tag:
            tag = f"{tag}nuc"        # keep outputs separate from the 39-feat run
    if args.chromatin_features:
        chromatin_dir = Path(args.chromatin_dir) if args.chromatin_dir \
            else gdir / "chromatin_feats"
        if "chrom" not in tag:
            tag = f"{tag}chrom"
    if args.eosin_features:
        eosin_dir = Path(args.eosin_dir) if args.eosin_dir \
            else gdir / "eosin_feats"
        if "eos" not in tag:
            tag = f"{tag}eos"
    if args.fm_features:
        fm_dir = Path(args.fm_dir) if args.fm_dir else gdir / "fm_feats"
        tag = f"{tag}fm{args.fm_features}" + ("cat" if args.fm_concat else "")
    # The bag cache is built BEFORE flag dropping (which happens at the tail of
    # load_bags), so both ablation arms share one cache. Only the outputs split.
    cache_tag = tag
    if args.drop_flags:
        tag = f"{tag}noflag"         # and separate again from the flagged arm
    if args.bag_subsample:
        tag = f"{tag}_ss{int(round(args.bag_subsample * 100))}"
    if args.mixup:
        tag = f"{tag}_mx{int(round(args.mixup * 100))}"
    if args.arm:
        tag = f"{tag}_{args.arm}"    # ablation arms get their own outputs

    prefix = ("smoke_" if args.smoke else "") + f"abmil_{tag}"
    res_path = odir / f"{prefix}_results.csv"
    ckpt_dir = odir / "abmil_ckpt" / tag

    X, y_all, ids, feat_names = load_bags(
        gdir, odir / f"bag_cache_{cache_tag}.pt", args.rebuild_cache,
        args.exclude_prefix, nuclear_dir=nuclear_dir, drop_flags=args.drop_flags,
        drop_groups=args.drop_groups, keep_groups=args.keep_groups,
        chromatin_dir=chromatin_dir, eosin_dir=eosin_dir, fm_dir=fm_dir,
        fm_model=args.fm_features, fm_only=fm_only)

    configs = CONFIGS
    if args.arch:
        want_a = set(args.arch)
        configs = [c for c in configs if cfg_field(c, "arch", "abmil") in want_a]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        configs = [c for c in configs if c[0] in want]
        missing = want - {c[0] for c in configs}
        if missing:
            raise SystemExit(f"--only names no such config: {sorted(missing)}")
    if not configs:
        raise SystemExit("No configs selected.")
    if args.noise_floor:
        configs = [(c[0] + "_rerun",) + c[1:] for c in configs]

    keep_all = args.select != "argmax" or args.patience <= 0
    print(f"selection   : {args.select}"
          + (f" + early stop (patience {args.patience})" if not keep_all
             else f" over the full {epochs} epochs, no early stop"))
    print(f"lr schedule : {args.sched}")
    if args.select == "swa" and args.sched == "cosine":
        print("  NOTE: cosine anneals lr to 0, so the epochs SWA averages sit")
        print("  almost on top of each other and this is close to a no-op.")
        print("  --sched cosine_floor is the arm that actually tests SWA.")

    done, rows = set(), []
    if res_path.exists():
        prev = pd.read_csv(res_path)
        rows = prev.to_dict("records")
        done = {(r["config"], r["repeat"], r["fold"]) for r in rows}
        print(f"Resuming: {len(done)} runs already complete")

    print(f"device={device}  bags={len(X)}  configs={len(configs)}  "
          f"-> {len(configs) * repeats * folds} runs")
    print(f"checkpoints -> {ckpt_dir}\n")

    oof = {}            # config -> (true, pred) accumulated across folds, rep 0
    t0all = time.time()
    for cfg in configs:
        name = cfg[0]
        for rep in range(repeats):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
            for fold, (tr_idx, te_idx) in enumerate(
                    skf.split(np.zeros(len(y_all)), y_all)):
                if fold >= folds:
                    break
                if (name, rep, fold) in done:
                    continue
                t0 = time.time()
                try:
                    ty, tp, bval, btra, sel_desc, n_ep = one_run(
                        cfg, X, y_all, tr_idx, te_idx, feat_names,
                        seed=rep * 100 + fold, device=device, epochs=epochs,
                        ckpt_path=ckpt_dir / f"{name}_r{rep}f{fold}.pt",
                        subsample=args.bag_subsample,
                        select=args.select, patience=args.patience,
                        sched_kind=args.sched, smooth_window=args.smooth_window,
                        swa_start=args.swa_start, swa_top_k=args.swa_top_k,
                        clam_k=args.clam_k, clam_weight=args.clam_weight,
                        mixup=args.mixup)
                except Exception as e:
                    print(f"  {name} r{rep}f{fold} FAILED: {type(e).__name__}: {e}")
                    rows.append(dict(config=name, repeat=rep, fold=fold,
                                     balanced_acc=np.nan, error=str(e)[:200]))
                    pd.DataFrame(rows).to_csv(res_path, index=False)
                    continue
                bal = balanced_accuracy_score(ty, tp)
                if rep == 0:
                    a, b = oof.get(name, ([], []))
                    oof[name] = (a + list(ty), b + list(tp))
                row = dict(config=name, repeat=rep, fold=fold,
                           balanced_acc=round(float(bal), 4),
                           val_bal=round(float(bval), 4),
                           train_bal=round(float(btra), 4),
                           select=args.select, sched=args.sched,
                           sel_detail=sel_desc, n_epochs=n_ep,
                           secs=round(time.time() - t0, 1))
                row.update(dict(zip(COLS[1:], cfg[1:])))
                rows.append(row)
                pd.DataFrame(rows).to_csv(res_path, index=False)
                print(f"  {name:20s} r{rep} f{fold}  bal={bal:.3f}  "
                      f"train={btra:.3f}  [{sel_desc}]  ({row['secs']:.0f}s, "
                      f"elapsed {(time.time()-t0all)/60:.0f}m)")

    summarise(pd.DataFrame(rows), oof, odir, prefix)


def summarise(df, oof, odir, prefix):
    df = df[df["balanced_acc"].notna()]
    if not len(df):
        print("No completed runs.")
        return
    print("\n" + "=" * 78)
    print(f"ABMIL SWEEP -- {prefix}")
    print("=" * 78)
    g = df.groupby("config")[["balanced_acc", "train_bal"]].agg(
        ["mean", "std", "count"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.sort_values("balanced_acc_mean", ascending=False).round(4)
    print(g.to_string())

    best = g.index[0]
    spread = float(g.balanced_acc_mean.max() - g.balanced_acc_mean.min())
    print(f"\nbest: {best} = {g.balanced_acc_mean.iloc[0]:.4f} "
          f"(sd {g.balanced_acc_std.iloc[0]:.4f})")
    print(f"spread across configs: {spread:.4f}")

    print("\n--- vs the v5 GAT sweep ---")
    print("  The GAT corpus best was mean_randomtopo 0.5622, and every config")
    print("  there sat between 0.538 and 0.562.")
    b = float(g.balanced_acc_mean.iloc[0])
    if b >= 0.60:
        print(f"  This ABMIL reaches {b:.3f}. The GAT architecture WAS costing")
        print("  accuracy -- the 39 features are worth more than the v5 sweep")
        print("  suggested. Re-read the v5 negative in that light: it compared")
        print("  ablations of an architecture that was itself underfitting.")
    elif b >= 0.57:
        print(f"  This ABMIL reaches {b:.3f}, modestly above the GAT best. Some")
        print("  architecture cost, but the features are still the main limit.")
    else:
        print(f"  This ABMIL reaches {b:.3f}, level with the GAT sweep. The")
        print("  architecture was NOT the bottleneck: 39 tile features cap out")
        print("  here. Better features (nuclear morphometrics) is the next move,")
        print("  not a better model.")

    if "meanpool_h128" in set(df.config):
        for att in [c for c in set(df.config) if c.startswith("abmil")]:
            sa = df[df.config == att].set_index(["repeat", "fold"])["balanced_acc"]
            sb = df[df.config == "meanpool_h128"].set_index(
                ["repeat", "fold"])["balanced_acc"]
            common = sa.index.intersection(sb.index)
            if len(common) < 2 or att != g.index[0]:
                continue
            d = sa.loc[common] - sb.loc[common]
            print(f"\n--- does attention pooling beat mean pooling? ---")
            print(f"  {att} - meanpool_h128 = {d.mean():+.4f} "
                  f"(sd {d.std(ddof=1):.4f}, n={len(common)})")
            print("  Note: even at zero, attention still earns its place -- it")
            print("  is the only pooling here that yields a per-tile weight to")
            print("  interpret. Accuracy parity is not an argument against it.")

    if best in oof and len(oof[best][0]):
        ty, tp = np.array(oof[best][0]), np.array(oof[best][1])
        cm = confusion_matrix(ty, tp, labels=range(len(CLASSES)))
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        print(f"\n--- CONFUSION MATRIX, {best}, repeat 0 out-of-fold ---")
        print("  rows = true, cols = predicted, row-normalised\n")
        print("  " + pd.DataFrame(cmn.round(3), index=CLASSES,
                                  columns=CLASSES).to_string().replace("\n", "\n  "))
        print("\n  Off-diagonal mass is the actionable part: a class pair that")
        print("  collapses into each other tells you which feature is missing.")
        print("  On v5 the dominant error is HGSC<->EC (each ~25%), with EC the")
        print("  weakest class; HGSC<->LGSC is secondary. Both splits turn on")
        print("  nuclear grade (HGSC pleomorphic, EC/LGSC uniform), which is")
        print("  exactly what the morphometric features add.")

    txt = odir / f"{prefix}_summary.txt"
    txt.write_text(g.to_string(), encoding="utf-8")
    print(f"\nPer-run: {odir / (prefix + '_results.csv')}\nSummary: {txt}")


if __name__ == "__main__":
    main()


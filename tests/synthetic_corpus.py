# Copyright 2026 Vidip Gaur, Vidu Gaur
"""A small synthetic dataset root for the bag loaders, written straight to npz.

Every block carries the real column names: the graph block takes them from the
builder's own node_features(), and the nuclear, chromatin and eosin blocks from
each extractor's FEATURE_NAMES. The values are random. Each block stores the
graph's grid_rc_checksum, so the loaders' alignment guard passes.

Five slides per class: train_abmil.flag_leak_report() cross-validates with
StratifiedKFold(5) on every load and refuses fewer. nuc_have_cv2 is held at 1.0,
as it was across the real corpus, so drop_degenerate() removes it the same way.
"""
import numpy as np

import build_streamline_graphs_bridge as B
import extract_chromatin_features as XC
import extract_eosin_features as XE
import extract_nuclear_features as XN
import train_abmil as T

FM_MODEL = "phikonv2"
FM_DIM = 8
CACHE_TAG = "v5nucchromeos"


def graph_feature_names():
    _, names = B.node_features(np.zeros((2, 2), np.float32), {}, {},
                               np.zeros((2, 2), np.int64), 1, {}, {})
    return names


def random_block(rng, n, names, flags):
    F = rng.random((n, len(names))).astype(np.float32) * 5
    for j, name in enumerate(names):
        if name in flags:
            F[:, j] = rng.integers(0, 2, n)
    return F


def make_root(root, per_class=5, seed=0):
    """Write graphs plus nuclear, chromatin, eosin and embedding blocks."""
    g = root / T.RUN_DIR_NAME / T.GRAPH_SUBFOLDER
    for sub in ("nuclear_feats", "chromatin_feats", "eosin_feats", "fm_feats"):
        (g / sub).mkdir(parents=True)
    gnames = graph_feature_names()
    channels = list(B.INTEGRATION_CHANNELS)
    rng = np.random.default_rng(seed)
    k = 0
    for label in T.CLASSES:
        for _ in range(per_class):
            iid = str(9000 + k)
            n = int(rng.integers(6, 15))
            rc = np.stack([rng.integers(0, 40, n), rng.integers(0, 60, n)], axis=1)
            ck = np.array([int(rc.sum())])
            valid = np.ones(len(channels), bool)
            valid[channels.index("epithelial")] = k % 3 != 0
            np.savez_compressed(
                g / f"{iid}_graph.npz",
                node_features=random_block(rng, n, gnames, ()),
                node_feature_names=np.array(gnames, dtype=object),
                channel_names=np.array(channels, dtype=object),
                channel_valid=valid, node_grid_rc=rc,
                label=np.array([label]), image_id=np.array([iid]),
                script_version=np.array([B.SCRIPT_VERSION]))
            nuc = random_block(rng, n, XN.FEATURE_NAMES, XN.FLAG_COLS)
            nuc[:, XN.FEATURE_NAMES.index("nuc_have_cv2")] = 1.0
            np.savez_compressed(g / "nuclear_feats" / f"{iid}_nuclei.npz",
                                nuc_features=nuc,
                                nuc_feature_names=np.array(XN.FEATURE_NAMES, dtype=object),
                                grid_rc_checksum=ck)
            np.savez_compressed(g / "chromatin_feats" / f"{iid}_chromatin.npz",
                                chrom_features=random_block(rng, n, XC.FEATURE_NAMES, XC.FLAG_COLS),
                                chrom_feature_names=np.array(XC.FEATURE_NAMES, dtype=object),
                                grid_rc_checksum=ck)
            np.savez_compressed(g / "eosin_feats" / f"{iid}_eosin.npz",
                                eos_features=random_block(rng, n, XE.FEATURE_NAMES, XE.FLAG_COLS),
                                eos_feature_names=np.array(XE.FEATURE_NAMES, dtype=object),
                                grid_rc_checksum=ck)
            np.savez_compressed(g / "fm_feats" / f"{iid}_{FM_MODEL}.npz",
                                fm_features=rng.normal(size=(n, FM_DIM)).astype(np.float32),
                                grid_rc_checksum=ck)
            k += 1
    return root


def build_cache(root, blocks, tag=CACHE_TAG):
    """Build bag_cache_<tag>.pt from the given blocks, through the same
    load_bags() call train_abmil.main() makes."""
    g = root / T.RUN_DIR_NAME / T.GRAPH_SUBFOLDER
    o = root / T.RUN_DIR_NAME / T.OUT_SUBFOLDER
    T.load_bags(g, o / f"bag_cache_{tag}.pt", False, None,
                nuclear_dir=g / "nuclear_feats" if "nuclear" in blocks else None,
                chromatin_dir=g / "chromatin_feats" if "chromatin" in blocks else None,
                eosin_dir=g / "eosin_feats" if "eosin" in blocks else None,
                fm_dir=g / "fm_feats" if "fm" in blocks else None,
                fm_model=FM_MODEL if "fm" in blocks else None)
    return o / f"bag_cache_{tag}.pt"

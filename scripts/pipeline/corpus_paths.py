# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Corpus path + per-slide scale resolution, shared by every extraction stage.

WHY THIS EXISTS
===============
The pipeline was written against one corpus (UBC-OCEAN) whose slides are named
by a bare numeric image_id and live at a fixed path
(`cellvit_run/wsi_pyramid_tiff/<image_id>.tiff`).  Running the identical
pipeline on a second corpus -- the Ben Taeb et al. 80-slide set, whose CellViT
output folders are named `10_OOU198_EC-SC` and whose slides are `.svs` under a
Training/Test split -- needs three things the old constants cannot express:
a different run directory, a per-slide slide path, and a per-slide pixel size.

Rather than fork the extractors, every stage resolves those three through this
module.  With no environment set the dataset root is `./data/UBC-OCEAN`,
relative to the working directory the first script was started from; the
layout below the root is unchanged, so UBC-OCEAN results are unaffected.

USAGE
=====
    export UBC_ROOT=/data/UBC-OCEAN               # bash: where train.csv and cellvit_run/ live
    export UBC_RUN_DIR=bentaeb_run                # bash: switches the whole chain

    $env:UBC_ROOT = "D:\data\UBC-OCEAN"           # PowerShell
    $env:UBC_RUN_DIR = "bentaeb_run"              # PowerShell

The training-side scripts (train_abmil.py, fm_subspace_split.py,
bridge_fm_interpretable.py) build `<root>/cellvit_run/...` themselves and do
not read UBC_RUN_DIR.

The parallel corpus supplies a `cellvit_manifest.csv` carrying, in addition to
the columns the UBC-OCEAN manifest already has (image_id, label, image_width,
image_height):

    slide_path   absolute path to the WSI, any format openslide can read
    mpp          NOMINAL pixel size -- drives TILING GEOMETRY only
    native_mpp   TRUE pixel size    -- drives PHYSICAL UNITS (areas, distances)

TWO PIXEL SIZES, AND WHY
========================
UBC-OCEAN slides are natively 0.50 um/px, but the graph builder computes its
tile lattice against a hard-coded WSI_MPP = 0.25, which yields
tile_size_wsi_px = 1024 and therefore a 512 um physical tile.  Meanwhile
`extract_nuclear_features.py` deliberately uses the true 0.50 for areas, because
0.25 implies a 3.8 um nucleus (impossible).  Both are correct for their purpose
and both must be preserved.

Generalised to any corpus, the rule that reproduces UBC-OCEAN's geometry is

    nominal_mpp = native_mpp / 2

so a 20x slide (0.4992) gets 0.2496 -> 1024 px tiles, and a 40x slide (0.2520)
gets 0.1260 -> 2032 px tiles.  Both are ~512 um, i.e. every slide in every
corpus lands on the same physical grid.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

DEFAULT_ROOT = "./data/UBC-OCEAN"

ROOT = Path(os.path.abspath(os.environ.get("UBC_ROOT", DEFAULT_ROOT)))
# The orchestrators start every stage with cwd=scripts/, where a relative
# default would name a different directory. Exporting the absolute root keeps
# parent and child processes on the same corpus.
os.environ["UBC_ROOT"] = str(ROOT)
RUN_DIR_NAME = os.environ.get("UBC_RUN_DIR", "cellvit_run")
RUN_DIR = ROOT / RUN_DIR_NAME

MANIFEST = RUN_DIR / "cellvit_manifest.csv"

GRAPH_DIR = RUN_DIR / "graphs"
DET_DIR = RUN_DIR / "detections"
KDE_DIR = RUN_DIR / "kde_maps"
TIFF_DIR = RUN_DIR / "wsi_pyramid_tiff"

# True when a corpus other than the default UBC-OCEAN one is selected. Stages
# print this so a run can never be mistaken for the primary corpus in a log.
IS_EXTERNAL = RUN_DIR_NAME != "cellvit_run"

_MANIFEST_CACHE: dict | None = None


def manifest() -> dict:
    """image_id -> row dict, read once."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        out = {}
        if MANIFEST.exists():
            with open(MANIFEST, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    out[str(row["image_id"])] = row
        _MANIFEST_CACHE = out
    return _MANIFEST_CACHE


def slide_path(image_id: str) -> Path:
    """Path to the slide pixels.

    Manifest `slide_path` wins when present; otherwise the UBC-OCEAN default
    `wsi_pyramid_tiff/<image_id>.tiff`, unchanged.
    """
    row = manifest().get(str(image_id))
    if row:
        p = (row.get("slide_path") or "").strip()
        if p:
            return Path(p)
    return TIFF_DIR / f"{image_id}.tiff"


def native_mpp(image_id: str):
    """True um/px for physical units, or None to let the caller fall back to
    its own is_tma-keyed default (the UBC-OCEAN behaviour)."""
    row = manifest().get(str(image_id))
    if row:
        v = (row.get("native_mpp") or "").strip()
        if v:
            return float(v)
    return None


def nominal_mpp(image_id: str):
    """um/px used for TILING GEOMETRY, or None for the caller's own constant."""
    row = manifest().get(str(image_id))
    if row:
        v = (row.get("mpp") or "").strip()
        if v:
            return float(v)
    return None


GRAPH_SUFFIX = "_graph.npz"


def image_id_from_graph(path) -> str:
    """image_id for a `<id>_graph.npz` path.

    Everything before the `_graph` suffix, NOT the first underscore-delimited
    token. UBC-OCEAN ids are bare numbers so `split("_")[0]` happened to agree
    there, but an id like `10_OOU198_EC_SC` is truncated by a split to `10` --
    which silently collapses a whole corpus onto a handful of ids and makes
    every already-extracted slide look pending. Use this everywhere instead.
    """
    n = Path(path).name
    if n.endswith(GRAPH_SUFFIX):
        return n[:-len(GRAPH_SUFFIX)]
    return Path(n).stem


def banner(stage: str) -> str:
    return (f"[corpus] stage={stage} run_dir={RUN_DIR} "
            f"external={IS_EXTERNAL} manifest_rows={len(manifest())}")

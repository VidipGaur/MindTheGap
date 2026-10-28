# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Verify the corpus_paths edits: UBC-OCEAN defaults unchanged, Ben Taeb routed.

Run twice by the caller, once with UBC_RUN_DIR unset and once set, because the
path constants are resolved at import time.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_paths as cp

print(f"UBC_RUN_DIR env = {os.environ.get('UBC_RUN_DIR', '<unset>')}")
print(f"  RUN_DIR      = {cp.RUN_DIR}")
print(f"  IS_EXTERNAL  = {cp.IS_EXTERNAL}")
print(f"  manifest rows= {len(cp.manifest())}")

import build_kde_density_maps_new as kde
import build_streamline_graphs_bridge as gb
import extract_nuclear_features as nuc
import extract_chromatin_features as chrom
import extract_eosin_features as eos
import embed_tiles_fm as fm

print(f"  kde   ROOT/{kde.RUN_DIR_NAME}  pref={kde.DETECTION_PREFERENCE[:2]}")
print(f"  graph ROOT/{gb.RUN_DIR_NAME}   WSI_MPP={gb.WSI_MPP} "
      f"NODE_SIZE_UM={gb.NODE_SIZE_UM}")
print(f"  nuc   {nuc.GRAPH_DIR}")
print(f"  chrom {chrom.GRAPH_DIR}")
print(f"  eos   {eos.OUT_DIR}")
print(f"  fm    {fm.OUT_DIR}")

# um_to_px must be unchanged when no mpp is passed
# 256.0 is the builder's NODE_SIZE_UM, a nominal um figure at 0.25 um/px.
a = gb.um_to_px(256.0, 16)
b = gb.um_to_px(256.0, 16, 0.25)
c = gb.um_to_px(256.0, 16, 0.126)
print(f"  um_to_px(256,16)       = {a}        (node_size_px; x16 -> "
      f"{a * 16:.0f} px tile)")
print(f"  um_to_px(256,16,0.25)  = {b}        -> {b * 16:.0f} px tile")
print(f"  um_to_px(256,16,0.126) = {c:.4f}   -> {c * 16:.0f} px tile")
assert a == b, "default mpp path changed -- UBC-OCEAN geometry would move"

# image_id derivation
for n in ("10077_graph.npz", "10_OOU198_EC_SC_graph.npz",
          "17_test_VOA734B_graph.npz"):
    print(f"  image_id_from_graph({n:<32s}) = {cp.image_id_from_graph(n)}")
assert cp.image_id_from_graph("10077_graph.npz") == "10077"
assert cp.image_id_from_graph("10_OOU198_EC_SC_graph.npz") == "10_OOU198_EC_SC"

# slide path + mpp resolution
if cp.IS_EXTERNAL:
    iid = sorted(cp.manifest())[0]
    sp = cp.slide_path(iid)
    print(f"  slide_path({iid}) -> {sp}")
    print(f"     exists={sp.exists()}  native_mpp={cp.native_mpp(iid)}  "
          f"nominal_mpp={cp.nominal_mpp(iid)}")
    assert sp.exists(), "manifest slide_path does not resolve"
    missing = [i for i in cp.manifest() if not cp.slide_path(i).exists()]
    print(f"  slides missing on disk: {len(missing)}")
    assert not missing, missing[:5]
else:
    print(f"  slide_path(10077) -> {cp.slide_path('10077')}  "
          f"(exists={cp.slide_path('10077').exists()})")
    assert cp.slide_path("10077") == cp.TIFF_DIR / "10077.tiff"
    assert cp.native_mpp("10077") is None, "UBC-OCEAN must fall back to is_tma"

print("  OK\n")

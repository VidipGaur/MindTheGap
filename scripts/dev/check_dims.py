# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Confirm graph wsi_shape == train.csv dims (no PNG->TIFF transpose), and
quantify how much the mpp choice actually changes downstream."""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import csv, glob
from pathlib import Path
import numpy as np
import corpus_paths

R = corpus_paths.ROOT
rows = {r["image_id"]: r for r in csv.DictReader(open(R / "train.csv", newline=""))}

bad = same = swapped = 0
for p in sorted(glob.glob(str(R / "cellvit_run/graphs/*_graph.npz")))[:60]:
    iid = Path(p).name.split("_")[0]
    with np.load(p, allow_pickle=True) as z:
        h, w = (int(v) for v in z["wsi_shape"])
    r = rows.get(iid)
    if not r:
        continue
    cw, ch = int(r["image_width"]), int(r["image_height"])
    if (h, w) == (ch, cw):
        same += 1
    elif (h, w) == (cw, ch):
        swapped += 1
    else:
        bad += 1
        if bad <= 5:
            print(f"  MISMATCH {iid}: graph(h,w)=({h},{w}) csv(h,w)=({ch},{cw})")
print(f"wsi_shape vs train.csv over 60 graphs: match={same} transposed={swapped} other={bad}")

# how binding is the [2,900] um^2 artefact guard at mpp=0.5 vs 0.25?
print("\nartefact guard [2, 900] um^2 in pixel terms:")
for mpp in (0.25, 0.50):
    print(f"  mpp={mpp}: keeps nuclei of {2/mpp**2:8.0f} .. {900/mpp**2:8.0f} px^2 "
          f"(equiv diam {2*np.sqrt(2/np.pi)/1:.1f} .. {2*np.sqrt(900/np.pi):.0f} um)")

print("\nwhat the mpp choice actually changes:")
print("  INVARIANT to mpp (pure px ratios or scale-free): ecc, axisratio, circularity,")
print("    solidity, *_cv, p90/p10 ratio, all counts and fracs")
print("  GLOBAL RESCALE only (absorbed by z-scoring): areas (x4), equivdiam (x2),")
print("    nnd_* (x2), nuc_per_mm2 (/4)")
print("  GENUINELY DIFFERENT (a threshold in um is applied):")
print("    til_frac_neo_infiltrated  (30 um cut -> 120 px at 0.25 vs 60 px at 0.5)")
print("    which nuclei the [2,900] um^2 guard drops")

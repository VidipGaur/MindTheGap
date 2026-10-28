# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Eosin check on slides chosen ACROSS classes.

The default --check takes the first N graphs by id, which happened to be 3 EC
and 2 HGSC -- neither of the two subtypes these features exist to detect. A
clearing feature that reads zero on EC and HGSC is doing the right thing, so
that run could not distinguish "working" from "broken".

CC (clear cytoplasm) and MC (mucin) must read HIGHER on tile_clear_frac than
HGSC/EC or the block is not measuring what it claims to.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_eosin_features import GRAPH_DIR, process_slide

want = {"CC": 3, "MC": 3, "HGSC": 2, "EC": 2, "LGSC": 2}
rows = []
by_class = {}
for g in sorted(GRAPH_DIR.glob("*_graph.npz")):
    with np.load(g, allow_pickle=True) as z:
        lab = str(z["label"][0])
    if by_class.get(lab, 0) >= want.get(lab, 0):
        continue
    by_class[lab] = by_class.get(lab, 0) + 1
    _, qc = process_slide(g, check=True)
    if isinstance(qc, str):
        print(f"  {g.name} FAILED: {qc}")
        continue
    rows.append(qc)
    print(f"  {qc['image_id']:>7s} {qc['label']:5s} "
          f"clear={qc.get('tile_clear_frac'):.4f} "
          f"pale={qc.get('tile_pale_frac'):.4f} "
          f"e_ring={qc.get('neo_e_ring_mean_mean'):.4f} "
          f"empty_ring={qc.get('neo_ring_empty_frac_mean'):.4f} "
          f"core-ring={qc.get('neo_e_core_ring_diff_mean'):+.4f}")
    if sum(by_class.values()) >= sum(want.values()):
        break

df = pd.DataFrame(rows)
print("\n=== per class ===")
cols = ["tile_clear_frac", "tile_pale_frac", "neo_ring_empty_frac_mean",
        "neo_e_ring_mean_mean", "neo_e_core_ring_diff_mean",
        "tile_nuc_area_frac"]
print(df.groupby("label")[cols].mean().round(4).to_string())

print("\n=== sanity ===")
bad = (df["neo_e_core_ring_diff_mean"] >= 0).sum()
print(f"  slides with core-ring >= 0 (should be 0): {bad}/{len(df)}")
print(f"  tile_clear_frac range: {df.tile_clear_frac.min():.4f} .. "
      f"{df.tile_clear_frac.max():.4f}")
cc_mc = df[df.label.isin(["CC", "MC"])].tile_clear_frac.mean()
other = df[~df.label.isin(["CC", "MC"])].tile_clear_frac.mean()
print(f"  clear_frac  CC+MC {cc_mc:.4f}  vs  other {other:.4f}  "
      f"-> {'OK' if cc_mc > other else 'WRONG DIRECTION'}")

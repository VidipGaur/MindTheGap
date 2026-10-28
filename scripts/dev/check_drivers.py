# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Are nnd_all_um_cv and dead_count real signal or extraction artefacts?"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_abmil import CLASSES, load_bags
from pathlib import Path
import corpus_paths

R = corpus_paths.ROOT
gdir = R / "cellvit_run/graphs"
odir = R / "cellvit_run/graphs/gatv2_run"
X, y, ids, fn = load_bags(gdir, odir / "bag_cache_v5nuc.pt", False, None,
                          nuclear_dir=gdir / "nuclear_feats")

for feat in ["nnd_all_um_cv", "dead_count", "dead_frac", "neo_area_um2_std"]:
    if feat not in fn:
        print(f"{feat}: absent")
        continue
    j = fn.index(feat)
    # slide-level mean of the per-tile value
    v = np.array([float(b[:, j].mean()) for b in X])
    lab = np.array([CLASSES[t] for t in y])
    print("\n" + "=" * 70)
    print(f"{feat}   (slide-level mean of per-tile value)")
    print("=" * 70)
    t = pd.DataFrame({"v": v, "label": lab}).groupby("label").v.agg(
        ["mean", "median", "std", "min", "max", "count"]).round(3)
    print(t.to_string())
    kw = stats.kruskal(*[v[lab == c] for c in CLASSES if (lab == c).sum() > 2])
    print(f"  Kruskal-Wallis across classes: H={kw.statistic:.1f} p={kw.pvalue:.2e}")
    z = (v - v.mean()) / v.std()
    print(f"  corpus: skew={stats.skew(v):+.2f}  kurtosis={stats.kurtosis(v):+.2f}")
    print(f"  slides beyond +3 SD: {(z > 3).sum()}   beyond +5 SD: {(z > 5).sum()}")
    ext = np.argsort(-z)[:8]
    print("  most extreme slides: " +
          ", ".join(f"{ids[i]}({lab[i]},z={z[i]:+.1f})" for i in ext))

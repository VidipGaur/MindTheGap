# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Seed stability across the three configurations, at MATCHED seed counts.

"All N seeds agree" gets mechanically harder as N grows, so the historical
63.4% (3 seeds) cannot be compared against a 10-seed number. Everything here
is reported at 3 seeds AND at 10 where available, plus mean pairwise agreement
which is seed-count invariant.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import sys

import numpy as np
import torch
from scipy import stats
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_abmil_nuclear import (as_single_map, bags_for, collect,
                                   partial_spearman, pick_density_col)
from train_abmil import ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARMS = [
    ("79 feat (graph+nuclear), argmax+ES", "v5nuc", None, 3),
    ("40 feat (nuclear only),  argmax+ES", "v5nuc_sel_argmax_es", ["graph"], 10),
    ("40 feat (nuclear only),  swatop",    "v5nuc_sel_swatop", ["graph"], 10),
]

print(f"{'configuration':38s} {'seeds':>5s} {'unanim':>7s} {'pairwise':>9s} "
      f"{'map med':>8s} {'map q25':>8s}")
print("-" * 82)
rows = []
for label, tag, dg, reps in ARMS:
    X, y, ids, fn = bags_for(ROOT, tag, True, drop_groups=dg)
    per = collect(ROOT, tag, "abmil_h128_lr1", X, y, reps, device, want_attn=True)
    good = [r for r in per if per[r]]
    didx, _ = pick_density_col(fn)
    common = sorted(set.intersection(*[set(per[r]) for r in good]))

    for k in sorted({3, len(good)}):
        sub = good[:k]
        P = np.array([[per[r][i][0] for i in common] for r in sub])
        unan = float(np.mean([len(set(P[:, j])) == 1 for j in range(P.shape[1])]))
        pair = float(np.mean([(P[a] == P[b]).mean()
                              for a in range(k) for b in range(a + 1, k)]))
        res = []
        for i in common:
            maps = [as_single_map(per[r][i][2]) for r in sub]
            if any(m is None for m in maps):
                continue
            dens = X[i].numpy()[:, didx]
            for a in range(k):
                for b in range(a + 1, k):
                    v = partial_spearman(maps[a], maps[b], dens)
                    if np.isfinite(v):
                        res.append(v)
        res = np.array(res)
        print(f"{label:38s} {k:5d} {unan:7.3f} {pair:9.3f} "
              f"{np.median(res):+8.3f} {np.percentile(res,25):+8.3f}")
        rows.append((label, k, unan, pair, np.median(res),
                     np.percentile(res, 25)))
    print()

print("\nunanim   = fraction of slides where ALL k seeds predict the same class")
print("pairwise = mean agreement over seed PAIRS (invariant to k)")
print("map      = density-residualised Spearman between seed pairs' attention")
print("           maps; q25 >= 0.40 was fixed in advance as the bar for figures")

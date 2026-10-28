# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Is the 0.6875 -> 0.6603 drop from cleaning the features real or noise?

Paired on the shared (repeat, fold) cells, which is the only fair comparison at
this per-cell sd. Reads results CSVs already on disk -- no training.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
from pathlib import Path

import pandas as pd
from scipy import stats
import corpus_paths

R = corpus_paths.ROOT / "cellvit_run" / "graphs" / "gatv2_run"

PAIRS = [
    ("dirty  v5nucchrom_final_additive", "abmil_v5nucchrom_final_additive_results.csv",
     "addmil_h128_do5",
     "clean  v5nucchromeos_clean_final", "abmil_v5nucchromeos_clean_final_results.csv",
     "addmil_h128_do5"),
    ("dirty  v5nucchrom_final_additive", "abmil_v5nucchrom_final_additive_results.csv",
     "addmil_h128_do5",
     "nodead v5nuc_nodead10", "abmil_v5nuc_nodead10_results.csv",
     "abmil_h128_lr1"),
]


def load(fn, cfg):
    d = pd.read_csv(R / fn)
    d = d[d.config == cfg]
    return d.set_index(["repeat", "fold"])["balanced_acc"]


for na, fa, ca, nb, fb, cb in PAIRS:
    a, b = load(fa, ca), load(fb, cb)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    d = a - b
    t, p = stats.ttest_rel(a, b)
    _, pw = stats.wilcoxon(a, b)
    print(f"\n{na}  {a.mean():.4f}")
    print(f"{nb}  {b.mean():.4f}")
    print(f"  delta={d.mean():+.4f}  sd_of_diff={d.std():.4f}  "
          f"per-cell sd={a.std():.4f}")
    print(f"  paired t p={p:.4f}   wilcoxon p={pw:.4f}   "
          f"win={(d > 0).mean():.2f}   n={len(idx)}")

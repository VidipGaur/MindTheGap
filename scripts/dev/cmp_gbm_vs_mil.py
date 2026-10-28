# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Paired: does ABMIL beat the best slide-level model on the SAME folds?

Both ran 10 repeats x 5 folds with StratifiedKFold(random_state=repeat), so
every (repeat, fold) cell is the same split for both and the comparison can be
paired -- which removes the fold-difficulty variance that otherwise dominates.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import corpus_paths

OUT = corpus_paths.ROOT / "cellvit_run" / "graphs" / "gatv2_run"


def load(tag, config):
    d = pd.read_csv(OUT / f"abmil_{tag}_results.csv")
    d = d[(d.balanced_acc.notna()) & (d.config == config)]
    return d.set_index(["repeat", "fold"])["balanced_acc"]


mil = load("v5nuc_sel_argmax_es", "abmil_h128_lr1")
print(f"ABMIL (nuclear-only, 10 seeds)  n={len(mil)}  mean={mil.mean():.4f}")
for name in ["logreg", "gbm", "rf"]:
    sl = load("v5nuc_gbm", name)
    common = mil.index.intersection(sl.index)
    d = (mil.loc[common] - sl.loc[common]).values
    t = stats.ttest_rel(mil.loc[common].values, sl.loc[common].values)
    try:
        w = stats.wilcoxon(d).pvalue
    except Exception:
        w = np.nan
    print(f"\n  ABMIL - {name:7s}  n={len(common)}  "
          f"slide-level mean={sl.loc[common].mean():.4f}")
    print(f"    delta {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
          f"t p={t.pvalue:.2e}  wilcoxon p={w:.2e}  "
          f"MIL wins {100*(d > 0).mean():.0f}% of folds")

print("\n--- overfitting, train minus validation ---")
for tag, cfg in [("v5nuc_sel_argmax_es", "abmil_h128_lr1"),
                 ("v5nuc_gbm", "logreg"), ("v5nuc_gbm", "gbm"),
                 ("v5nuc_gbm", "rf")]:
    d = pd.read_csv(OUT / f"abmil_{tag}_results.csv")
    d = d[(d.balanced_acc.notna()) & (d.config == cfg)]
    print(f"  {cfg:16s} train {d.train_bal.mean():.3f}  "
          f"test {d.balanced_acc.mean():.3f}  gap {d.train_bal.mean()-d.balanced_acc.mean():+.3f}")

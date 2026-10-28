# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Stage C paired: does any alternative MIL formulation beat ABMIL?

All arms ran under the same selection rule (swatop) on the same
StratifiedKFold(random_state=repeat) splits, so every (repeat, fold) cell is
the same split for every config and the comparison is paired. That matters:
per-config sd is ~0.06 while the gaps between configs are ~0.01, so config
means alone are not evidence.
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

c = pd.read_csv(OUT / "abmil_v5nuc_alg_sel_swatop_results.csv")
a = pd.read_csv(OUT / "abmil_v5nuc_sel_swatop_results.csv")
d = pd.concat([c[c.balanced_acc.notna()],
               a[(a.balanced_acc.notna()) & (a.repeat < 5)]])

base = "abmil_h128_lr1"
sb = d[d.config == base].set_index(["repeat", "fold"])["balanced_acc"]
print(f"BASE {base} (ABMIL) mean = {sb.mean():.4f}  n={len(sb)}\n")

rows = []
for cfg in sorted(set(d.config) - {base}):
    sa = d[d.config == cfg].set_index(["repeat", "fold"])["balanced_acc"]
    common = sa.index.intersection(sb.index)
    if len(common) < 5:
        continue
    delta = (sa.loc[common] - sb.loc[common]).values
    t = stats.ttest_rel(sa.loc[common].values, sb.loc[common].values)
    try:
        w = stats.wilcoxon(delta).pvalue
    except Exception:
        w = np.nan
    arch = d[d.config == cfg]["arch"].iloc[0]
    rows.append(dict(config=cfg, arch=arch, mean=sa.loc[common].mean(),
                     delta=delta.mean(), sd=delta.std(ddof=1),
                     t_p=t.pvalue, wil_p=w, win=(delta > 0).mean(),
                     n=len(common)))
t = pd.DataFrame(rows).sort_values("delta", ascending=False)
print(t.round(4).to_string(index=False))
print("\n  A config is only meaningfully better if delta>0 AND p is small AND")
print("  win is well above 0.5. At this sd, +0.01 with p=0.4 is noise.")

# additive as a family vs abmil, pooling the three additive configs per cell
print("\n--- ARCHITECTURE FAMILIES, pooled per (repeat, fold) ---")
for arch in ["additive", "clam", "transmil"]:
    sub = d[d.arch == arch]
    if not len(sub):
        continue
    fam = sub.groupby(["repeat", "fold"])["balanced_acc"].mean()
    common = fam.index.intersection(sb.index)
    delta = (fam.loc[common] - sb.loc[common]).values
    t2 = stats.ttest_rel(fam.loc[common].values, sb.loc[common].values)
    print(f"  {arch:9s} mean {fam.loc[common].mean():.4f}  "
          f"delta {delta.mean():+.4f}  p={t2.pvalue:.3f}  "
          f"win {(delta > 0).mean():.2f}  ({sub.config.nunique()} configs pooled)")

print("\n--- OVERFITTING (train minus test) ---")
g = d.groupby(["config", "arch"])[["train_bal", "balanced_acc"]].mean()
g["gap"] = g.train_bal - g.balanced_acc
print(g.round(3).sort_values("gap").to_string())

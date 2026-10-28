# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Stage D paired: do the chromatin/texture features add anything?

chrom_full ran under the SAME selection rule (swatop), same config, same
StratifiedKFold(random_state=repeat) splits as the nuclear-only baseline, so
every (repeat, fold) cell is the same split in both arms. Unpaired means would
be meaningless here -- per-cell sd is ~0.06, the effect we care about is ~0.02.
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
CFG = "abmil_h128_lr1"

base = pd.read_csv(OUT / "abmil_v5nuc_sel_swatop_results.csv")
base = base[(base.config == CFG) & base.balanced_acc.notna()]
sb = base.set_index(["repeat", "fold"])["balanced_acc"]

print(f"BASELINE  nuclear-only (40 feat), swatop   mean {sb.mean():.4f}  n={len(sb)}")

for f in sorted(OUT.glob("abmil_v5nucchrom_*_results.csv")):
    arm = f.stem.replace("abmil_v5nucchrom_", "").replace("_results", "")
    d = pd.read_csv(f)
    d = d[(d.config == CFG) & d.balanced_acc.notna()]
    if not len(d):
        continue
    sa = d.set_index(["repeat", "fold"])["balanced_acc"]
    common = sa.index.intersection(sb.index)
    delta = (sa.loc[common] - sb.loc[common]).values
    t = stats.ttest_rel(sa.loc[common].values, sb.loc[common].values)
    try:
        w = stats.wilcoxon(delta).pvalue
    except Exception:
        w = np.nan
    print(f"\n  {arm:16s} mean {sa.loc[common].mean():.4f}  n={len(common)}")
    print(f"  {'':16s} delta {delta.mean():+.4f} +/- {delta.std(ddof=1):.4f}  "
          f"t_p={t.pvalue:.4f}  wilcoxon_p={w:.4f}  win={(delta>0).mean():.2f}")
    print(f"  {'':16s} train_bal {d.train_bal.mean():.3f} -> gap "
          f"{d.train_bal.mean()-sa.loc[common].mean():.3f}")

print("\n  A feature block only earns its place if delta>0 AND p is small AND")
print("  win is well above 0.5. At n=50 cells, +0.02 with p<0.05 is real.")

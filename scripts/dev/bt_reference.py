# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""The UBC-OCEAN reference column, under the SAME statistic the external run
reports: the mean over 50 (repeat, fold) cells of per-cell balanced accuracy.

Read straight off the per-fold result CSVs, which is what paper_numbers.py does
to produce Table 1 -- so these are the paper's own numbers, not a re-derivation.
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
import corpus_paths

R = corpus_paths.ROOT / "cellvit_run" / "graphs" / "gatv2_run"
ADD = "addmil_h128_do5"

ARMS = [
    ("clean", "Named morphometry (49 cols)", "abmil_v5nucchromeos_clean_final_results.csv"),
    ("uni2h", "UNI2-h (ViT-H/14, 1536-d)", "abmil_v5fmuni2h_fm_addmil_results.csv"),
    ("hoptimus0", "H-optimus-0 (ViT-g/14, 1536-d)", "abmil_v5fmhoptimus0_fm_addmil_results.csv"),
    ("phikonv2", "Phikon-v2 (ViT-L/16, 1024-d)", "abmil_v5fmphikonv2_fm_addmil_results.csv"),
    ("hibouL", "Hibou-L (ViT-L/14, 1024-d)", "abmil_v5fmhibouL_fm_addmil_results.csv"),
]

rows = {}
for key, nice, fn in ARMS:
    df = pd.read_csv(R / fn)
    df = df[df["config"] == ADD].set_index(["repeat", "fold"]).sort_index()
    rows[key] = (nice, df["balanced_acc"], float(df["train_bal"].mean()))

base_nice, base, _ = rows["clean"]
print(f"UBC-OCEAN reference -- mean of 50 (repeat, fold) cells, {ADD}\n")
print(f"{'arm':<32s} {'mean':>8s} {'sd':>7s} {'n':>4s} {'train_bal':>10s} "
      f"{'delta':>8s} {'paired p':>10s} {'win':>6s}")
print("-" * 92)
from scipy import stats
for key, nice, _ in ARMS:
    nice, s, tb = rows[key]
    line = f"{nice:<32s} {s.mean():8.4f} {s.std(ddof=1):7.4f} {len(s):4d} {tb:10.4f}"
    if key != "clean":
        j = s.index.intersection(base.index)
        d = (s.loc[j] - base.loc[j]).values
        t, p = stats.ttest_rel(s.loc[j], base.loc[j])
        line += f" {d.mean():8.4f} {p:10.2e} {float((d>0).mean()):6.2f}"
    print(line)

print("\nPaper draft's stated headline numbers, for cross-check:")
print("  clean 0.6603 | Phikon-v2 0.7848 (+0.1244) | H-optimus-0 0.7917 (+0.1313)")
print("  Hibou-L 0.8131 (+0.1528) | UNI2-h 0.8305 (+0.1701)")

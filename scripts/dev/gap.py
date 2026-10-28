# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
from pathlib import Path

import pandas as pd
import corpus_paths

O = corpus_paths.ROOT / "cellvit_run" / "graphs" / "gatv2_run"
cfgs = ["abmil_h128_lr1", "abmil_lr1_do5", "abmil_mh8"]
rows = []
for tag, lab in [("v5nuc", "none"), ("v5nuc_ss80", "0.8"), ("v5nuc_ss50", "0.5")]:
    f = O / f"abmil_{tag}_results.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    d = d[d.balanced_acc.notna() & d.config.isin(cfgs)]
    for c, g in d.groupby("config"):
        rows.append(dict(subsample=lab, config=c,
                         val=round(g.balanced_acc.mean(), 4),
                         train=round(g.train_bal.mean(), 4),
                         gap=round(g.train_bal.mean() - g.balanced_acc.mean(), 4)))
t = pd.DataFrame(rows).pivot(index="config", columns="subsample",
                             values=["val", "train", "gap"])
print(t.to_string())
print("\ngap = train_bal - balanced_acc (overfitting proxy)")

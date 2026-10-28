# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
import csv, glob, os
from collections import Counter
import numpy as np
from scipy.stats import chi2_contingency
import corpus_paths

ROOT = str(corpus_paths.ROOT)
ids = set(os.path.basename(p).replace("_graph.npz", "")
          for p in glob.glob(os.path.join(ROOT, "cellvit_run", "graphs", "*_graph.npz")))
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "train.csv")))
        if r["is_tma"] == "False"]
inc = [r for r in rows if r["image_id"] in ids]
exc = [r for r in rows if r["image_id"] not in ids]
CL = ["CC", "EC", "HGSC", "LGSC", "MC"]
ci = Counter(r["label"] for r in inc)
ce = Counter(r["label"] for r in exc)
print("WSI rows (is_tma False):", len(rows), " included:", len(inc), " excluded:", len(exc))
print("%-6s %6s %6s %8s" % ("class", "incl", "excl", "excl%"))
for c in CL:
    tot = ci[c] + ce[c]
    print("%-6s %6d %6d %7.1f%%" % (c, ci[c], ce[c], 100.0 * ce[c] / tot))
tab = np.array([[ci[c] for c in CL], [ce[c] for c in CL]])
chi2, p, dof, _ = chi2_contingency(tab)
print("chi2 = %.2f  dof=%d  p = %.3f" % (chi2, dof, p))

mpi = np.array([int(r["image_width"]) * int(r["image_height"]) / 1e6 for r in inc])
mpe = np.array([int(r["image_width"]) * int(r["image_height"]) / 1e6 for r in exc])
print("megapixels  incl median %.0f   excl median %.0f" % (np.median(mpi), np.median(mpe)))
from scipy.stats import mannwhitneyu
print("Mann-Whitney p = %.3g" % mannwhitneyu(mpi, mpe).pvalue)

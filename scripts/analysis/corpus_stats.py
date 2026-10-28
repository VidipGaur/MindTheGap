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
import corpus_paths

ROOT = str(corpus_paths.ROOT)
ids = set(os.path.basename(p).replace("_graph.npz", "")
          for p in glob.glob(os.path.join(ROOT, "cellvit_run", "graphs", "*_graph.npz")))
rows = list(csv.DictReader(open(os.path.join(ROOT, "train.csv"))))
print("train.csv rows:", len(rows), "| cols:", list(rows[0].keys()))
have = [r for r in rows if r["image_id"] in ids]
print("rows with graph:", len(have))
print("labels all      :", dict(Counter(r["label"] for r in rows)))
print("labels w/ graph :", dict(Counter(r["label"] for r in have)))
print("is_tma all      :", dict(Counter(r.get("is_tma") for r in rows)))
print("is_tma w/ graph :", dict(Counter(r.get("is_tma") for r in have)))

w = np.array([int(r["image_width"]) for r in have], float)
h = np.array([int(r["image_height"]) for r in have], float)
mp = w * h / 1e6
print("width  med %d  range %d-%d" % (np.median(w), w.min(), w.max()))
print("height med %d  range %d-%d" % (np.median(h), h.min(), h.max()))
print("megapixels med %.1f  range %.1f-%.1f" % (np.median(mp), mp.min(), mp.max()))

# tiles per slide from the graphs
n = []
for i in sorted(ids):
    with np.load(os.path.join(ROOT, "cellvit_run", "graphs", i + "_graph.npz"),
                 allow_pickle=True) as z:
        n.append(int(z["node_grid_rc"].shape[0]))
n = np.array(n)
print("tiles/slide: median %d  IQR %d-%d  min %d  max %d  total %d"
      % (np.median(n), np.percentile(n, 25), np.percentile(n, 75),
         n.min(), n.max(), n.sum()))

lab = {r["image_id"]: r["label"] for r in have}
order = sorted(ids)
for c in ["CC", "EC", "HGSC", "LGSC", "MC"]:
    sub = np.array([n[k] for k, i in enumerate(order) if lab.get(i) == c])
    print("  %-5s n=%3d  tiles median %5d  total %7d"
          % (c, len(sub), np.median(sub), sub.sum()))

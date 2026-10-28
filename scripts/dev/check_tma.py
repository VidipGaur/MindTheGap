# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
import csv, glob
from collections import Counter
from pathlib import Path
import corpus_paths

R = corpus_paths.ROOT
g = {Path(p).name.split("_")[0] for p in glob.glob(str(R / "cellvit_run/graphs/*_graph.npz"))}
print(f"graphs on disk: {len(g)}")
allids = set()
for n in ("train.csv", "test.csv"):
    f = R / n
    print(f"\n{n}: exists={f.exists()}")
    if not f.exists():
        continue
    rows = list(csv.DictReader(open(f, newline="")))
    print("   cols:", list(rows[0].keys()), " n=", len(rows))
    allids |= {r["image_id"] for r in rows}
    if "is_tma" in rows[0]:
        print("   is_tma:", dict(Counter(str(r["is_tma"]).strip().lower() for r in rows)))
        tma = [r["image_id"] for r in rows if str(r["is_tma"]).strip().lower() == "true"]
        print(f"   TMA ids: {len(tma)}  -> with a graph: {len(set(tma) & g)}")
        print("   TMA ids w/ graph (first 15):", sorted(set(tma) & g)[:15])
        if "label" in rows[0]:
            lab = {r["image_id"]: r["label"] for r in rows}
            print("   TMA-with-graph by class:",
                  dict(Counter(lab[i] for i in (set(tma) & g) if i in lab)))
miss = g - allids
print(f"\ngraphs with NO csv row (silently fall back to graph mpp=0.25): {len(miss)}")
if miss:
    print("   ->", sorted(miss)[:15])

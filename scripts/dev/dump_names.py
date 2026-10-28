# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
import glob
from pathlib import Path

import numpy as np
import corpus_paths

GD = str(corpus_paths.ROOT / "cellvit_run" / "graphs")
g = sorted(glob.glob(str(Path(GD) / "*_graph.npz")))[0]
with np.load(g, allow_pickle=True) as z:
    gn = [str(v) for v in z["node_feature_names"]]
iid = Path(g).name.split("_")[0]
with np.load(str(Path(GD) / "nuclear_feats" / f"{iid}_nuclei.npz"), allow_pickle=True) as z:
    nn_ = [str(v) for v in z["nuc_feature_names"]]

print(f"GRAPH node features: {len(gn)}")
for i, f in enumerate(gn):
    print(f"  {i:2d} {f}")
print(f"\nNUCLEAR features: {len(nn_)}")
for i, f in enumerate(nn_):
    print(f"  {i:2d} {f}")

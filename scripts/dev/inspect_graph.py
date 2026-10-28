# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
# Copyright 2026 Vidip Gaur, Vidu Gaur
import numpy as np, glob, json
import os
import corpus_paths
G = str(corpus_paths.ROOT / "cellvit_run" / "graphs")
p = sorted(glob.glob(os.path.join(G, "*_graph.npz")))
print("n graphs:", len(p))
z = np.load(p[0], allow_pickle=True)
print("file:", os.path.basename(p[0]))
print("keys:", sorted(z.files))
for k in ("image_id", "label", "wsi_shape", "node_grid_shape",
          "tile_size_wsi_px", "node_features", "node_grid_rc"):
    if k in z.files:
        v = z[k]
        print("  %-20s shape=%s  %s" % (k, v.shape, v.ravel()[:6]))
if "params" in z.files:
    try:
        d = json.loads(str(z["params"].ravel()[0]))
        for k in sorted(d):
            print("   param %-24s %s" % (k, d[k]))
    except Exception as e:
        print("params raw:", str(z["params"])[:900])

print("\n--- tile_size_wsi_px across 12 slides ---")
for f in p[:12]:
    with np.load(f, allow_pickle=True) as y:
        print("  %-8s tile_px=%s  grid=%s  wsi=%s  n=%d" % (
            str(y["image_id"][0]),
            np.asarray(y["tile_size_wsi_px"]).ravel()[0],
            tuple(int(v) for v in y["node_grid_shape"]),
            tuple(int(v) for v in y["wsi_shape"]),
            y["node_features"].shape[0]))

# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Where does chromatin extraction actually spend its 250 ms per tile?"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_chromatin_features as X
from pathlib import Path
import corpus_paths

ROOT = corpus_paths.ROOT
iid = "1020"
gp = next((ROOT / "cellvit_run/graphs").glob(f"{iid}_graph.npz"))
g = X.load_graph_geometry(gp)

t = time.time()
cx, cy, ct, bb = X.load_detections(ROOT / f"cellvit_run/detections/{iid}/cell_detection.json")
print(f"load_detections      {time.time()-t:6.2f}s  ({len(cx)} cells)")

sl = X.open_slide(ROOT / f"cellvit_run/wsi_pyramid_tiff/{iid}.tiff")
tile_px = g["tile_px"]
n_rows, n_cols = g["grid_shape"]
node_of = np.full((n_rows, n_cols), -1, np.int64)
node_of[g["grid_rc"][:, 0], g["grid_rc"][:, 1]] = np.arange(g["n_nodes"])
gc = np.clip((cx // tile_px).astype(int), 0, n_cols - 1)
gr = np.clip((cy // tile_px).astype(int), 0, n_rows - 1)
nid = node_of[gr, gc]
ok = nid >= 0
order = np.argsort(nid[ok], kind="stable")
ni = nid[ok][order]
src = np.flatnonzero(ok)[order]
bounds = np.searchsorted(ni, np.arange(g["n_nodes"] + 1))
counts = np.diff(bounds)
pop = np.flatnonzero(counts >= X.MIN_ALL_CHROM)
print(f"populated tiles      {len(pop)}")

rng = np.random.RandomState(0)
origins = [(int(g["grid_rc"][n, 1] * tile_px), int(g["grid_rc"][n, 0] * tile_px))
           for n in pop]
t = time.time()
h_lo, h_hi, qc = X.fit_slide_stain(sl, origins, tile_px, rng)
print(f"fit_slide_stain      {time.time()-t:6.2f}s")

sample = pop[:: max(1, len(pop) // 40)][:40]
tread = tdec = tnuc = 0.0
n_nuc = 0
for n in sample:
    ox = int(g["grid_rc"][n, 1] * tile_px); oy = int(g["grid_rc"][n, 0] * tile_px)
    t = time.time(); rgb = X.read_tile(sl, ox, oy, tile_px); tread += time.time() - t
    t = time.time()
    hraw = X.deconvolve_h(rgb)
    hmap = X.normalise_h(hraw, h_lo, h_hi)
    tis = X.tissue_mask(rgb)
    tdec += time.time() - t
    idx = src[bounds[n]:bounds[n + 1]]
    sel = idx[ct[idx] == 1]
    if len(sel) > X.SAMPLE_PER_TILE:
        sel = sel[rng.choice(len(sel), X.SAMPLE_PER_TILE, replace=False)]
    t = time.time()
    for i in sel:
        x0, y0, x1, y1 = bb[i]
        lx0, ly0 = max(x0 - ox, 0), max(y0 - oy, 0)
        lx1, ly1 = min(x1 - ox, tile_px), min(y1 - oy, tile_px)
        if lx1 - lx0 < 3 or ly1 - ly0 < 3:
            continue
        X.nucleus_metrics(hmap[ly0:ly1, lx0:lx1], hraw[ly0:ly1, lx0:lx1],
                          tis[ly0:ly1, lx0:lx1])
        n_nuc += 1
    tnuc += time.time() - t
k = len(sample)
print(f"\nper tile over {k} tiles:")
print(f"  read_region        {tread/k*1000:7.1f} ms")
print(f"  deconv+mask        {tdec/k*1000:7.1f} ms")
print(f"  {n_nuc/k:4.0f} nuclei        {tnuc/k*1000:7.1f} ms"
      f"   ({tnuc/max(n_nuc,1)*1000:.2f} ms each)")
print(f"  TOTAL              {(tread+tdec+tnuc)/k*1000:7.1f} ms"
      f"  -> {(tread+tdec+tnuc)/k*len(pop):.0f}s/slide for neo only")
print("  (the real run also measures the 'all' group, ~2x the nuclei)")
sl.close()

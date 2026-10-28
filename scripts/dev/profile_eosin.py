# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Where does extract_eosin_features spend its 232 ms per tile?

Guessing wrong here costs hours of wall clock, so measure before optimising --
the same mistake was avoided in the chromatin extractor, where the intuitive
suspect (the WSI read) turned out to be second to a lstsq nobody suspected.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import sys
import time

import numpy as np
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_eosin_features as E
from extract_chromatin_features import (deconvolve_h, load_detections,
                                        load_graph_geometry, open_slide,
                                        read_tile, tissue_mask)

g = load_graph_geometry(sorted(E.GRAPH_DIR.glob("*_graph.npz"))[0])
iid = g["image_id"]
sl = open_slide(E.TIFF_DIR / f"{iid}.tiff")
tile_px = g["tile_px"]
print(f"slide {iid}  tile_px={tile_px}  nodes={g['n_nodes']}")

origins = [(int(g["grid_rc"][n, 1] * tile_px), int(g["grid_rc"][n, 0] * tile_px))
           for n in range(0, g["n_nodes"], max(1, g["n_nodes"] // 25))][:25]

T = {k: 0.0 for k in ["read", "deconv_e", "deconv_h", "tissue", "close",
                      "percentile", "entropy", "gradient", "clear"]}
N = 0
for (ox, oy) in origins:
    t = time.perf_counter(); rgb = read_tile(sl, ox, oy, tile_px); T["read"] += time.perf_counter() - t
    t = time.perf_counter(); e = E.deconvolve_e(rgb); T["deconv_e"] += time.perf_counter() - t
    t = time.perf_counter(); h = deconvolve_h(rgb); T["deconv_h"] += time.perf_counter() - t
    t = time.perf_counter(); tis = tissue_mask(rgb); T["tissue"] += time.perf_counter() - t
    t = time.perf_counter(); reg = E.tissue_region(tis); T["close"] += time.perf_counter() - t
    em = E.normalise(e, 0.0, 1.0)
    et = em[tis]
    t = time.perf_counter(); np.percentile(et, 10); np.percentile(et, 90); T["percentile"] += time.perf_counter() - t
    t = time.perf_counter(); E._entropy(et); T["entropy"] += time.perf_counter() - t
    t = time.perf_counter()
    sub = em[::2, ::2]; gy, gx = np.gradient(sub.astype(np.float32))
    float(np.sqrt(gy ** 2 + gx ** 2).mean()); T["gradient"] += time.perf_counter() - t
    t = time.perf_counter(); float((reg & ~tis).sum() / max(reg.sum(), 1)); T["clear"] += time.perf_counter() - t
    N += 1
sl.close()

tot = sum(T.values())
print(f"\n{N} tiles, {tot/N*1000:.1f} ms/tile measured (excludes the nucleus loop)")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"  {k:12s} {v/N*1000:7.2f} ms/tile  {100*v/tot:5.1f}%")

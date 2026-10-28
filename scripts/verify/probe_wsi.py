# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Do the pyramid TIFFs share a coordinate space with the graphs and cells.json?

Before extracting a single pixel-level feature: level-0 dims must equal the
graph's wsi_shape, and a nucleus contour's bounding box must land on tissue,
not background. This project has been bitten by silent coordinate bugs twice.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import json
import sys
import time
from pathlib import Path

import numpy as np
import corpus_paths

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = corpus_paths.ROOT
GDIR = ROOT / "cellvit_run" / "graphs"
TIFF = ROOT / "cellvit_run" / "wsi_pyramid_tiff"
DET = ROOT / "cellvit_run" / "detections"

graphs = sorted(GDIR.glob("*_graph.npz"))
print(f"{len(graphs)} graphs, {len(list(TIFF.glob('*.tiff')))} tiffs\n")

import openslide
import tiffslide

for gp in graphs[:: max(1, len(graphs) // 5)][:5]:
    iid = gp.name.split("_")[0]
    with np.load(gp, allow_pickle=True) as z:
        wsi_shape = tuple(int(v) for v in z["wsi_shape"])
        tile_px = float(np.asarray(z["tile_size_wsi_px"]).ravel()[0])
        grid_shape = tuple(int(v) for v in z["node_grid_shape"])
        label = str(z["label"][0])
        prov = json.loads(str(z["provenance_json"][0]))
    tp = TIFF / f"{iid}.tiff"
    if not tp.exists():
        print(f"{iid}: NO TIFF")
        continue
    t0 = time.time()
    try:
        sl = openslide.OpenSlide(str(tp))
        backend = "openslide"
    except Exception as e:
        sl = tiffslide.TiffSlide(str(tp))
        backend = f"tiffslide (openslide failed: {type(e).__name__})"
    print(f"--- {iid} ({label}) via {backend} ---")
    print(f"  graph wsi_shape      : {wsi_shape}   tile_px={tile_px}  grid={grid_shape}")
    print(f"  slide level0 (w,h)   : {sl.level_dimensions[0]}")
    print(f"  levels               : {sl.level_dimensions}")
    print(f"  downsamples          : {[round(d,2) for d in sl.level_downsamples]}")
    print(f"  graph wsi_mpp (prov) : {prov.get('wsi_mpp')}")
    props = {k: sl.properties.get(k) for k in
             ("openslide.mpp-x", "tiffslide.mpp-x", "tiff.ResolutionUnit",
              "tiff.XResolution")}
    print(f"  props                : {props}")

    w, h = sl.level_dimensions[0]
    # graph wsi_shape is (rows, cols) = (h, w)?  check both orders
    print(f"  shape match (h,w)={wsi_shape == (h, w)}  (w,h)={wsi_shape == (w, h)}")
    print(f"  grid covers          : {grid_shape[0]*tile_px:.0f} x "
          f"{grid_shape[1]*tile_px:.0f} (rows x cols px)")

    # read one region and check it is tissue-like
    reg = np.array(sl.read_region((w // 2 - 512, h // 2 - 512), 0,
                                  (1024, 1024)).convert("RGB"))
    print(f"  centre 1024 patch    : mean RGB {reg.reshape(-1,3).mean(0).round(1)}  "
          f"({time.time()-t0:.1f}s incl open)")
    sl.close()
    print()

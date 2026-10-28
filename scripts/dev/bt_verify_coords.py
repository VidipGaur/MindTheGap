# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Verify the Ben Taeb CellViT coordinate convention before any KDE is built.

Three checks, all of which have burned this project before:

 1. COORDINATE SPACE.  CellViT was run with target_patch_mpp = base_mpp/2, so the
    question is whether centroids are in base (native) or target (2x) pixel space.
    If native, the centroid cloud spans ~the full slide; if 2x, it spans ~2x the
    slide and clips.  This is exactly the axis/scale bug class the KDE builder's
    frac_oob detector exists for -- checked here first so nothing is built on it.

 2. cell_detection.json vs cells.json.  The KDE only needs centroid+type. If the
    two files carry identical cell lists, the 3.9 GB file can be used instead of
    the 29.6 GB one with no numerical difference at all.

 3. NUCLEUS SIZE.  Median bbox diagonal in um under the native mpp. UBC-OCEAN's
    verified value is ~8 um (16 px at 0.5). A 2x-space reading would halve it.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import csv
import json
import sys
from pathlib import Path

import numpy as np
import corpus_paths

ROOT = corpus_paths.ROOT
RUN = ROOT / "bentaeb_run"
N_CHECK = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def load_cells_detection(p):
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj.get("cells", []), obj.get("wsi_metadata", {}), obj.get("type_map", {})


def stream_cells_json_head(p, n):
    """First n cells of the big cells.json, using the same byte scanner the
    nuclear extractor uses -- no full load."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_nuclear_features import iter_cells
    out = []
    for c in iter_cells(p):
        out.append(c)
        if len(out) >= n:
            break
    return out


def main():
    rows = list(csv.DictReader(open(RUN / "cellvit_manifest.csv", newline="",
                                    encoding="utf-8")))
    # spread the check across sizes + include the one 40x slide
    rows.sort(key=lambda r: int(r["image_width"]) * int(r["image_height"]))
    pick = [rows[0], rows[len(rows) // 4], rows[len(rows) // 2],
            rows[3 * len(rows) // 4], rows[-1]]
    forty = [r for r in rows if float(r["native_mpp"]) < 0.35]
    for r in forty:
        if r not in pick:
            pick.append(r)
    pick = pick[:N_CHECK]

    print(f"{'image_id':<34s} {'label':5s} {'native':>7s} {'W x H':>17s} "
          f"{'n_cells':>9s} {'covX':>6s} {'covY':>6s} {'oob':>7s} {'bbox_um':>8s}")
    print("-" * 112)

    bad = []
    for r in pick:
        iid = r["image_id"]
        det = RUN / "detections" / iid / "cell_detection.json"
        cells, meta, tmap = load_cells_detection(det)
        W, H = int(r["image_width"]), int(r["image_height"])
        nat = float(r["native_mpp"])

        cen = np.array([c["centroid"] for c in cells], dtype=np.float64)  # [x, y]
        bb = np.array([c["bbox"] for c in cells], dtype=np.float64)       # [[y0,x0],[y1,x1]]
        diag = np.hypot(bb[:, 1, 0] - bb[:, 0, 0], bb[:, 1, 1] - bb[:, 0, 1])

        covx = cen[:, 0].max() / (W - 1)
        covy = cen[:, 1].max() / (H - 1)
        oob = float(((cen[:, 0] < 0) | (cen[:, 0] > W - 1) |
                     (cen[:, 1] < 0) | (cen[:, 1] > H - 1)).mean())
        bbum = float(np.median(diag)) * nat

        flag = ""
        if oob > 0.001:
            flag = "  <-- OOB"
            bad.append((iid, f"frac_oob={oob:.3%}"))
        if not (0.30 <= covx <= 1.02) or not (0.30 <= covy <= 1.02):
            flag = "  <-- COVERAGE"
            bad.append((iid, f"coverage {covx:.2f}/{covy:.2f}"))

        print(f"{iid:<34s} {r['label']:5s} {nat:7.4f} {W:8d} x {H:<6d} "
              f"{len(cells):9,d} {covx:6.3f} {covy:6.3f} {oob:7.3%} {bbum:8.2f}{flag}")

        if meta:
            print(f"    cellvit: base_mpp={meta.get('base_mpp')} "
                  f"target_patch_mpp={meta.get('target_patch_mpp')} "
                  f"base_mag={meta.get('base_magnification')} "
                  f"patch={meta.get('patch_size')} overlap={meta.get('patch_overlap')}")
        if tmap and {str(k): v for k, v in tmap.items()} != {
                "1": "Neoplastic", "2": "Inflammatory", "3": "Connective",
                "4": "Dead", "5": "Epithelial"}:
            print(f"    !! type_map differs: {tmap}")
            bad.append((iid, "type_map differs"))

    r = pick[0]
    iid = r["image_id"]
    print(f"\ncells.json vs cell_detection.json on {iid} (first 5000 cells)")
    det_cells, _, _ = load_cells_detection(RUN / "detections" / iid / "cell_detection.json")
    big = stream_cells_json_head(RUN / "detections" / iid / "cells.json", 5000)
    a = np.array([c["centroid"] for c in det_cells[:5000]], dtype=np.float64)
    b = np.array([c["centroid"] for c in big], dtype=np.float64)
    ta = np.array([c["type"] for c in det_cells[:5000]])
    tb = np.array([c["type"] for c in big])
    print(f"    n detection={len(det_cells):,}   centroid max|diff|="
          f"{np.abs(a - b).max():.6g}   type mismatches={int((ta != tb).sum())}")
    print(f"    cells.json keys: {sorted(big[0].keys())}")
    if np.abs(a - b).max() > 1e-9 or (ta != tb).sum():
        bad.append((iid, "cells.json != cell_detection.json"))

    print("\n" + ("FAILED: " + "; ".join(f"{i}:{w}" for i, w in bad)
                  if bad else "ALL CHECKS PASSED"))


if __name__ == "__main__":
    main()

# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Audit the Ben Taeb tile lattice against the UBC-OCEAN one.

The single thing that must hold for Instrument 1 to be a fair external check is
that a Ben Taeb tile is the SAME PHYSICAL SQUARE as a UBC-OCEAN tile -- the head
was trained on 512 um instances and pools over whatever is in the bag, so a
different tile size is a different instance distribution, not a different corpus.
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
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import corpus_paths

ROOT = corpus_paths.ROOT
BT = ROOT / "bentaeb_run" / "graphs"
UBC = ROOT / "cellvit_run" / "graphs"

man = {r["image_id"]: r for r in csv.DictReader(
    open(ROOT / "bentaeb_run" / "cellvit_manifest.csv", newline="", encoding="utf-8"))}


def scan(d, native_mpp_of):
    rows = []
    for p in sorted(d.glob("*_graph.npz")):
        with np.load(p, allow_pickle=True) as z:
            iid = str(z["image_id"][0])
            tpx = float(np.asarray(z["tile_size_wsi_px"]).ravel()[0])
            rows.append(dict(
                iid=iid, label=str(z["label"][0]),
                n=int(z["node_features"].shape[0]),
                tpx=tpx,
                gh=int(z["node_grid_shape"][0]), gw=int(z["node_grid_shape"][1]),
                ver=str(z["script_version"][0]),
                um=tpx * native_mpp_of(iid),
                deg=float(z["edge_index"].shape[1]) / max(int(z["node_features"].shape[0]), 1),
                nch=int(np.asarray(z["channel_valid"]).sum()),
            ))
    return rows


def q(v, p):
    v = sorted(v)
    return v[min(int(len(v) * p), len(v) - 1)]


def report(name, rows):
    n = [r["n"] for r in rows]
    print(f"\n{name}  ({len(rows)} graphs)")
    print(f"  builder versions : {sorted(set(r['ver'] for r in rows))}")
    print(f"  tile_size_wsi_px : {sorted(set(r['tpx'] for r in rows))}")
    um = [r["um"] for r in rows]
    print(f"  PHYSICAL TILE um : min {min(um):.1f}  median {st.median(um):.1f}  "
          f"max {max(um):.1f}")
    print(f"  nodes/slide      : min {min(n)}  p25 {q(n,.25)}  median "
          f"{int(st.median(n))}  p75 {q(n,.75)}  max {max(n)}")
    print(f"  mean degree      : median {st.median([r['deg'] for r in rows]):.2f}")
    print(f"  channels valid   : {Counter(r['nch'] for r in rows).most_common()}")
    byl = {}
    for r in rows:
        byl.setdefault(r["label"], []).append(r["n"])
    print("  nodes by class   : " + "  ".join(
        f"{k} n={len(v)} med={int(st.median(v))}" for k, v in sorted(byl.items())))


bt = scan(BT, lambda i: float(man[i]["native_mpp"]))
report("BEN TAEB", bt)

ubc = scan(UBC, lambda i: 0.50)      # UBC-OCEAN WSIs are natively 0.50 um/px
report("UBC-OCEAN (reference)", ubc)

print("\nVERDICT")
uset = set(round(r["um"]) for r in bt) | set(round(r["um"]) for r in ubc)
print(f"  distinct physical tile sizes across BOTH corpora: "
      f"{sorted(uset)} um")
print("  -> identical instance scale" if len(uset) == 1
      else "  -> MISMATCH, instrument 1 would not be controlled")

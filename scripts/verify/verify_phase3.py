# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Unit-verify the new train_abmil.py machinery before an overnight run.

Accuracy from a smoke run only proves the code executes. These check that it
does the RIGHT thing: that mixup's soft label matches the instances actually
present, and that the four-block group resolution selects the right columns
when two blocks share a `tile_*` prefix -- the exact footgun that already cost
this project once, when `epithelial_frac` existed in both the graph and the
nuclear block and name-based selection silently hit both.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import sys

import numpy as np
import torch
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_abmil import (CHROM_ANCHOR, EOS_ANCHOR, FM_PREFIX, bag_mixup,
                         resolve_groups)

print("=" * 74)
print("1. bag_mixup -- is the soft label the ACHIEVED instance fraction?")
print("=" * 74)
torch.manual_seed(0)
np.random.seed(0)
B, N, D = 6, 200, 8
X = torch.randn(B, N, D)
# deliberately ragged, and with a 2x size spread like the real corpus
sizes = torch.tensor([200, 180, 100, 60, 150, 20])
M = torch.arange(N)[None, :] < sizes[:, None]
# tag each bag's rows with its own id so provenance is checkable
for b in range(B):
    X[b, :, 0] = b

Xm, Mm, perm, lam = bag_mixup(X, M, alpha=0.4)
print(f"  shapes: X {tuple(X.shape)} -> {tuple(Xm.shape)},  mask -> {tuple(Mm.shape)}")
fixed = int((perm == torch.arange(B)).sum())
print(f"  fixed points in the pairing (a bag mixed with itself): {fixed} "
      f"{'OK' if fixed == 0 else '-- these get NO augmentation'}")
ok = fixed == 0
for b in range(B):
    tags = Xm[b, :, 0][Mm[b]]
    a_cnt = int((tags == b).sum())
    b_cnt = int((tags == perm[b].item()).sum())
    other = int(Mm[b].sum()) - a_cnt - b_cnt
    exp = a_cnt / max(a_cnt + b_cnt, 1)
    match = abs(exp - lam[b].item()) < 1e-5
    ok &= match and other == 0 and Mm[b].sum() > 0
    print(f"  bag {b}: n={int(sizes[b])} partner={perm[b].item()} "
          f"kept {a_cnt}+{b_cnt}  lam={lam[b]:.4f} expected={exp:.4f} "
          f"{'OK' if match else 'MISMATCH'}"
          + ("" if other == 0 else f"  STRAY {other}"))
print(f"  -> label weight matches instance provenance: {ok}")

# a mixed bag must never be empty, and must never exceed the two sources
assert (Mm.sum(1) > 0).all(), "empty mixed bag"
assert (Mm.sum(1) <= M.sum(1) + M[perm].sum(1)).all(), "invented instances"
print("  -> no empty bags, no invented instances")

# and with alpha -> 0 the draw should collapse toward one source or the other
_, _, _, lam0 = bag_mixup(X, M, alpha=0.05)
print(f"  alpha=0.05 lam values (should hug 0/1): "
      f"{np.round(lam0.numpy(), 3)}")

print("\n" + "=" * 74)
print("2. resolve_groups -- four blocks, two of which share a `tile_` prefix")
print("=" * 74)
names = (
    ["dens_all_mean", "neoplastic_frac", "total_transitions"]      # graph  0-2
    + ["nuc_count", "neo_area_um2_mean", "nnd_all_um_mean",
       "neo_measured"]                                            # nuclear 3-6
    + [CHROM_ANCHOR, "neo_h_std_mean", "tile_h_mean",
       "neo_chrom_measured"]                                      # chrom   7-10
    + [EOS_ANCHOR, "tile_clear_frac", "neo_e_ring_mean_mean",
       "neo_ring_empty_frac_mean", "neo_eos_measured"]            # eosin  11-15
    + [f"{FM_PREFIX}{i:04d}" for i in range(4)]                    # fm     16-19
)
g = resolve_groups(names)
expect = {
    "graph": [0, 1, 2],
    "chrom": [7, 8, 9, 10],
    "eos": [11, 12, 13, 14, 15],
    "fm": [16, 17, 18, 19],
}
for k, v in expect.items():
    got = g.get(k, [])
    flag = "OK" if got == v else f"WRONG (got {got})"
    print(f"  {k:12s} {v}  {flag}")

print(f"  eos_clear   {g['eos_clear']}  "
      f"(-> {[names[i] for i in g['eos_clear']]})")
print(f"  eos_other   {g['eos_other']}  "
      f"(-> {[names[i] for i in g['eos_other']]})")
# the critical one: chromatin's tile_h_mean must NOT land in the eosin group
bad = [i for i in g["eos"] if names[i] == "tile_h_mean"]
print(f"  chromatin's tile_h_mean leaked into the eosin group: "
      f"{'YES -- BUG' if bad else 'no'}")
overlap = set(g["chrom"]) & set(g["eos"])
print(f"  chrom/eos index overlap: {overlap or 'none'}")

print("\n" + "=" * 74)
print("3. blocks present in isolation (each must still resolve)")
print("=" * 74)
for label, sub in [("graph+nuclear only", names[:7]),
                   ("nuclear+eosin, no chromatin", names[3:7] + names[11:16]),
                   ("fm alone", names[16:])]:
    gg = resolve_groups(sub)
    live = {k: len(v) for k, v in gg.items() if v}
    print(f"  {label:30s} {live}")

# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Sanity checks on the new architectures and selection modes.

1. AdditiveMIL: per-tile per-class contributions must SUM EXACTLY to the bag
   logits. That identity is the entire interpretability claim -- if it does not
   hold to float precision the architecture is mislabelled.
2. Masking: padded instances must contribute nothing, so a bag's logits must be
   identical whether it is batched alone or padded alongside a bigger bag.
3. select_state: each mode picks what it says it picks.
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
from train_abmil import (AdditiveMIL, CLAM, TransMIL, ABMIL, build_model,
                         select_state, _smooth, pad_collate, CONFIGS, cfg_field)

torch.manual_seed(0)
D, C = 41, 5

print("=" * 70)
print("1. ADDITIVE MIL -- contributions must sum to the logits")
print("=" * 70)
m = AdditiveMIL(D, C, hidden=128, layers=1, attn_hidden=128, dropout=0.0, heads=1).eval()
x = torch.randn(3, 40, D)
mask = torch.zeros(3, 40, dtype=torch.bool)
mask[0, :40] = True; mask[1, :17] = True; mask[2, :5] = True
out, a, contrib = m(x, mask, return_contrib=True)
err = (contrib.sum(1) - out).abs().max().item()
print(f"  max |sum(contributions) - logits| = {err:.3e}   "
      f"{'PASS' if err < 1e-4 else 'FAIL'}")
pad_mass = contrib[~mask].abs().sum().item()
print(f"  |contribution| on padded instances = {pad_mass:.3e}   "
      f"{'PASS' if pad_mass < 1e-9 else 'FAIL'}")
print(f"  attention sums per bag: {a.sum(1).tolist()}  (want 1.0 each)")

m4 = AdditiveMIL(D, C, 128, 1, 128, 0.0, heads=4).eval()
o4, _, c4 = m4(x, mask, return_contrib=True)
print(f"  heads=4 max |sum - logits| = {(c4.sum(1) - o4).abs().max().item():.3e}")

print()
print("=" * 70)
print("2. PADDING INVARIANCE -- a bag's logits must not depend on its neighbours")
print("=" * 70)
for name, mod in [("abmil", ABMIL(D, C, 128, 1, 128, 0.0, "attn", 1)),
                  ("abmil_mh4", ABMIL(D, C, 128, 1, 128, 0.0, "attn", 4)),
                  ("additive", AdditiveMIL(D, C, 128, 1, 128, 0.0, 1)),
                  ("clam", CLAM(D, C, 128, 1, 128, 0.0, 1)),
                  ("transmil", TransMIL(D, C, 128, 2, 128, 0.0, 4))]:
    mod = mod.eval()
    small = x[2:3, :5]
    m_small = torch.ones(1, 5, dtype=torch.bool)
    alone = mod(small, m_small)
    together = mod(x, mask)[2:3]
    d = (alone - together).abs().max().item()
    print(f"  {name:10s} max |alone - padded| = {d:.3e}   "
          f"{'PASS' if d < 1e-3 else 'FAIL <-- padding leaks'}")

print()
print("=" * 70)
print("3. SELECTION MODES")
print("=" * 70)
E = 30
states = [{"p": torch.full((2,), float(i))} for i in range(E)]
val = [0.0] * E
val[7] = 0.9            # a single lucky spike
val[20:25] = [0.8, 0.82, 0.85, 0.83, 0.81]   # a real, broad plateau
for mode in ("argmax", "smooth", "fixed", "swa", "swatop"):
    s, desc = select_state(mode, states, val, smooth_window=7,
                           swa_start=0.6, swa_top_k=5)
    print(f"  {mode:8s} -> mean epoch {s['p'][0].item():6.2f}   {desc}")
print("\n  argmax should land on the lucky spike (7); smooth should reject it")
print("  and land in the broad plateau (~22). That is the whole hypothesis.")

print()
print("=" * 70)
print("4. CONFIG TABLE")
print("=" * 70)
from collections import Counter
print("  ", Counter(cfg_field(c, "arch", "abmil") for c in CONFIGS))
for c in CONFIGS:
    if cfg_field(c, "arch", "abmil") != "abmil":
        mm = build_model(c, D, C)
        n = sum(p.numel() for p in mm.parameters())
        print(f"  {c[0]:18s} {cfg_field(c,'arch'):9s} {n/1000:7.1f}k params")

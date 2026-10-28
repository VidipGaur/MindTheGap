# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Pull Stage F/G/H numbers and repair the one paired comparison run_phase3 missed."""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import os
import pandas as pd
from scipy import stats
import corpus_paths

O = str(corpus_paths.ROOT / "cellvit_run" / "graphs" / "gatv2_run")


def cells(f, cfg=None):
    p = os.path.join(O, f)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if cfg:
        d = d[d.config == cfg]
    if not len(d):
        return None
    return d.set_index(["repeat", "fold"])["balanced_acc"]


def paired(fa, ca, fb, cb, label):
    a, b = cells(fa, ca), cells(fb, cb)
    if a is None or b is None:
        print(f"{label:38s} MISSING  a={'ok' if a is not None else 'none'} "
              f"b={'ok' if b is not None else 'none'}")
        return
    i = a.index.intersection(b.index)
    a, b = a.loc[i], b.loc[i]
    d = a - b
    print(f"{label:38s} A={a.mean():.4f} B={b.mean():.4f} delta={d.mean():+.4f} "
          f"p={stats.ttest_rel(a, b).pvalue:.4f} win={(d > 0).mean():.2f} n={len(i)}")


print("=" * 88)
print("LEVELS (10 repeats x 5 folds, swatop, abmil_h128_lr1 unless noted)")
print("=" * 88)
rows = [
    ("nuclear-only (reference)", "abmil_v5nuc_sel_swatop_results.csv", "abmil_h128_lr1"),
    ("  ...without dead_", "abmil_v5nuc_nodead10_results.csv", "abmil_h128_lr1"),
    ("+ chromatin (no texture)", "abmil_v5nucchrom_chrom_notexture_results.csv", "abmil_h128_lr1"),
    ("  ...without dead_", "abmil_v5nuc_chrom_notexture_nodead_results.csv", "abmil_h128_lr1"),
    ("+ eosin clearing only", "abmil_v5nuc_eos_onlyclear_results.csv", "abmil_h128_lr1"),
    ("mixup a=0.4", "abmil_v5nuc_mixup40_results.csv", "abmil_h128_lr1"),
    ("FINAL additive MIL", "abmil_v5nucchrom_final_additive_results.csv", "addmil_h128_do5"),
    ("Phikon-v2 (positive control)", "abmil_v5fmphikonv2_fm_phikonv2_results.csv", "abmil_h128_lr1"),
    ("Phikon v1", "abmil_v5fmphikon_fm_phikon_results.csv", "abmil_h128_lr1"),
    ("Phikon-v2 + nuclear (hybrid)", "abmil_v5fmphikonv2_fm_hybrid_results.csv", "abmil_h128_lr1"),
]
for lab, f, c in rows:
    s = cells(f, c)
    if s is None:
        # try without config filter, and report what configs exist
        p = os.path.join(O, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            print(f"{lab:32s}   -- config {c!r} not in {f}; has {sorted(d.config.unique())}")
        else:
            print(f"{lab:32s}   -- missing file {f}")
        continue
    print(f"{lab:32s} {s.mean():.4f}  sd {s.std():.4f}  n={len(s)}")

print()
print("=" * 88)
print("PAIRED COMPARISONS THE DRIVER DID NOT COMPLETE")
print("=" * 88)
paired("abmil_v5nucchrom_final_additive_results.csv", "addmil_h128_do5",
       "abmil_v5nucchrom_chrom_notexture_results.csv", "abmil_h128_lr1",
       "additive vs ABMIL (same features)")
paired("abmil_v5nucchrom_final_additive_results.csv", "addmil_h128_do5",
       "abmil_v5nuc_sel_swatop_results.csv", "abmil_h128_lr1",
       "additive vs nuclear-only ABMIL")
paired("abmil_v5fmphikonv2_fm_hybrid_results.csv", "abmil_h128_lr1",
       "abmil_v5fmphikonv2_fm_phikonv2_results.csv", "abmil_h128_lr1",
       "hybrid vs phikonv2 alone")
paired("abmil_v5fmphikonv2_fm_phikonv2_results.csv", "abmil_h128_lr1",
       "abmil_v5fmphikon_fm_phikon_results.csv", "abmil_h128_lr1",
       "phikonv2 vs phikon v1")
paired("abmil_v5nuc_eos_onlyclear_results.csv", "abmil_h128_lr1",
       "abmil_v5nucchrom_chrom_notexture_results.csv", "abmil_h128_lr1",
       "eosin-clear vs chromatin arm")

print()
print("=" * 88)
print("ALL RESULT FILES ON DISK")
print("=" * 88)
for f in sorted(os.listdir(O)):
    if f.endswith("_results.csv"):
        d = pd.read_csv(os.path.join(O, f))
        reps = d.repeat.nunique() if "repeat" in d else "?"
        print(f"  {f:56s} rows={len(d):4d} cfgs={d.config.nunique()} reps={reps} "
              f"best={d.groupby('config').balanced_acc.mean().max():.4f}")

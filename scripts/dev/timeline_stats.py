# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Pull every completed result cell into one place for the timeline doc."""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import os

import numpy as np
import pandas as pd
from scipy import stats
import corpus_paths

O = str(corpus_paths.ROOT / "cellvit_run" / "graphs" / "gatv2_run")


def load(f):
    p = os.path.join(O, f)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def summary(f, label):
    d = load(f)
    if d is None:
        print(f"-- {label}: MISSING {f}")
        return
    print(f"\n===== {label}  ({f})")
    g = d.groupby("config")["balanced_acc"].agg(["mean", "std", "count"])
    if "train_acc" in d.columns:
        g["train"] = d.groupby("config")["train_acc"].mean()
    print(g.sort_values("mean", ascending=False).round(4).to_string())


def cells(f, config):
    d = load(f)
    if d is None:
        return None
    d = d[d.config == config]
    if len(d) == 0:
        return None
    return d.set_index(["repeat", "fold"])["balanced_acc"]


def paired(fa, ca, fb, cb, label):
    a, b = cells(fa, ca), cells(fb, cb)
    if a is None or b is None:
        print(f"-- {label}: missing")
        return
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    d = a - b
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except Exception:
        w = float("nan")
    print(f"{label:44s} n={len(idx):3d}  A={a.mean():.4f} B={b.mean():.4f} "
          f"delta={d.mean():+.4f} p={t.pvalue:.4f} wil={w:.4f} win={(d>0).mean():.2f}")


print("#" * 78)
print("STAGE A -- selection rules")
for arm in ["argmax_es", "argmax", "smooth", "swa", "swaflr", "swatop", "fixed"]:
    summary(f"abmil_v5nuc_sel_{arm}_results.csv", f"sel_{arm}")

print("\n" + "#" * 78)
print("STAGE C -- algorithms")
summary("abmil_v5nuc_alg_sel_swatop_results.csv", "alg under swatop")
for c in ["addmil_h128", "addmil_h128_do5", "addmil_mh4", "clam_h128",
          "clam_h128_do5", "transmil_L1", "transmil_L2"]:
    paired("abmil_v5nuc_alg_sel_swatop_results.csv", c,
           "abmil_v5nuc_sel_swatop_results.csv", "abmil_h128_lr1",
           f"{c} vs abmil_h128_lr1/swatop")

print("\n" + "#" * 78)
print("GBM slide-level control")
summary("abmil_v5nuc_gbm_results.csv", "gbm")

print("\n" + "#" * 78)
print("STAGE D -- chromatin")
for arm in ["chrom_full", "chrom_notexture", "chrom_nolevel", "chrom_nocontext"]:
    summary(f"abmil_v5nucchrom_{arm}_results.csv", arm)
    paired(f"abmil_v5nucchrom_{arm}_results.csv", "abmil_h128_lr1",
           "abmil_v5nuc_sel_swatop_results.csv", "abmil_h128_lr1",
           f"{arm} vs nuclear-only/swatop")
paired("abmil_v5nucchrom_chrom_notexture_results.csv", "abmil_h128_lr1",
       "abmil_v5nucchrom_chrom_full_results.csv", "abmil_h128_lr1",
       "chrom_notexture vs chrom_full")

print("\n" + "#" * 78)
print("EARLIER arms still relevant")
for f, lab in [("abmil_v5nuc_nuconly_nodead_results.csv", "nuconly_nodead"),
               ("abmil_v5nuc_nuconly_results.csv", "nuconly (3 rep)")]:
    summary(f, lab)

print("\n" + "#" * 78)
print("ALL result csvs present")
for f in sorted(os.listdir(O)):
    if f.endswith("_results.csv"):
        d = pd.read_csv(os.path.join(O, f))
        print(f"  {f:52s} rows={len(d):4d} configs={d.config.nunique()} "
              f"reps={d.repeat.nunique() if 'repeat' in d else '?'}")

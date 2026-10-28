# Copyright 2026 Vidip Gaur, Vidu Gaur
"""The paired comparisons that decide what goes in the paper and what runs next."""

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

NUC = ("abmil_v5nuc_sel_swatop_results.csv", "abmil_h128_lr1")
CHR = ("abmil_v5nucchrom_chrom_notexture_results.csv", "abmil_h128_lr1")
EOS = ("abmil_v5nuceos_eos_onlyclear_results.csv", "abmil_h128_lr1")
MIX = ("abmil_v5nucchrom_mx40_mixup40_results.csv", "abmil_h128_lr1")
ADD = ("abmil_v5nucchrom_final_additive_results.csv", "addmil_h128_do5")
PV2 = ("abmil_v5fmphikonv2_fm_phikonv2_results.csv", "abmil_h128_lr1")
PV1 = ("abmil_v5fmphikon_fm_phikon_results.csv", "abmil_h128_lr1")
HYB = ("abmil_v5nucfmphikonv2cat_fm_hybrid_results.csv", "abmil_h128_lr1")
GBM = ("abmil_v5nuc_gbm_results.csv", None)


def cells(spec):
    f, cfg = spec
    p = os.path.join(O, f)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if cfg:
        d = d[d.config == cfg]
    return d.set_index(["repeat", "fold"])["balanced_acc"] if len(d) else None


def paired(A, B, label):
    a, b = cells(A), cells(B)
    if a is None or b is None:
        print(f"{label:44s} MISSING")
        return
    i = a.index.intersection(b.index)
    a, b = a.loc[i], b.loc[i]
    d = a - b
    try:
        w = stats.wilcoxon(a, b).pvalue
    except Exception:
        w = float("nan")
    star = "*" if stats.ttest_rel(a, b).pvalue < 0.10 else " "
    print(f"{label:44s} {a.mean():.4f} vs {b.mean():.4f}  d={d.mean():+.4f} "
          f"p={stats.ttest_rel(a, b).pvalue:.4f}{star} wil={w:.4f} win={(d > 0).mean():.2f} n={len(i)}")


print("=" * 96)
print("A. DOES THE INTERPRETABLE MODEL ADD ANYTHING ON TOP OF PHIKON?  (decides the paper's claim)")
print("=" * 96)
paired(HYB, PV2, "hybrid (phikonv2+nuclear) vs phikonv2")
paired(HYB, PV1, "hybrid vs phikon v1")
paired(PV2, PV1, "phikonv2 vs phikon v1")

print()
print("=" * 96)
print("B. THE INTERPRETABLE STACK -- what actually accumulates")
print("=" * 96)
paired(CHR, NUC, "+chromatin(notex) vs nuclear-only")
paired(EOS, NUC, "+eosin(clear only) vs nuclear-only")
paired(EOS, CHR, "eosin arm vs chromatin arm")
paired(MIX, CHR, "mixup a=0.4 vs chromatin arm")
paired(ADD, CHR, "additive MIL vs ABMIL (same feats)")
paired(ADD, NUC, "additive MIL vs nuclear-only ABMIL")

print()
print("=" * 96)
print("C. GAP TO THE CONTROLS")
print("=" * 96)
paired(PV2, ADD, "phikonv2 vs FINAL additive")
paired(ADD, GBM, "FINAL additive vs slide-level GBM/logreg")
g = pd.read_csv(os.path.join(O, GBM[0]))
print("\n  slide-level control, by model:")
print(g.groupby("config").balanced_acc.agg(["mean", "std", "count"]).round(4).to_string())

print()
print("=" * 96)
print("D. NOT YET RUN -- the obvious gaps")
print("=" * 96)
for name, f in [
    ("nuclear + chromatin + eosin (all three)", "abmil_*_chromeos_*"),
    ("eosin arm without dead_", "abmil_*_eos_onlyclear_nodead_*"),
    ("chrom_context dropped (staining confound)", "abmil_v5nucchrom_chrom_nocontext (EXISTS)"),
    ("hybrid without dead_", "abmil_*_fm_hybrid_nodead_*"),
    ("additive MIL on phikonv2", "abmil_*_fm_phikonv2 with addmil config"),
]:
    print(f"  - {name:44s} {f}")

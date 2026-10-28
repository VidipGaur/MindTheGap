# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Every number the summary doc needs, computed once, paired where it matters."""

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
    return d.set_index(["repeat", "fold"])["balanced_acc"] if len(d) else None


def paired(A, B, label):
    a, b = cells(*A), cells(*B)
    if a is None or b is None:
        print(f"{label:46s} MISSING")
        return
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    i = a.index.intersection(b.index)
    a, b = a.loc[i], b.loc[i]
    d = a - b
    print(f"{label:46s} {a.mean():.4f} vs {b.mean():.4f}  d={d.mean():+.4f} "
          f"p={stats.ttest_rel(a, b).pvalue:.4f} win={(d > 0).mean():.2f} n={len(i)}")


CLEANADD = ("abmil_v5nucchromeos_clean_additive_results.csv", "addmil_h128_do5")
CLEANALL = ("abmil_v5nucchromeos_clean_all_results.csv", "abmil_h128_lr1")
CLEANNOCTX = ("abmil_v5nucchromeos_clean_noctx_results.csv", "abmil_h128_lr1")
DIRTYADD = ("abmil_v5nucchrom_final_additive_results.csv", "addmil_h128_do5")
LOGREG = ("abmil_v5nuc_gbm_results.csv", "logreg")
GBMC = ("abmil_v5nuc_gbm_results.csv", "gbm")
PV2 = ("abmil_v5fmphikonv2_fm_phikonv2_results.csv", "abmil_h128_lr1")
FMADD = ("abmil_v5fmphikonv2_fm_addmil_results.csv", "addmil_h128_do5")
MIX40 = ("abmil_v5fmphikonv2_fm_mixup40_results.csv", "abmil_h128_lr1")
NUCONLY = ("abmil_v5nuc_sel_swatop_results.csv", "abmil_h128_lr1")

print("=" * 92)
print("THE UNCOMFORTABLE CONTROL -- does the bag structure earn its place?")
print("=" * 92)
paired(CLEANADD, LOGREG, "clean interpretable MIL vs slide-level logreg")
paired(CLEANALL, LOGREG, "clean ABMIL vs slide-level logreg")
paired(NUCONLY, LOGREG, "nuclear-only (dirty) vs slide-level logreg")
paired(DIRTYADD, LOGREG, "dirty additive vs slide-level logreg")
paired(PV2, LOGREG, "Phikon-v2 vs slide-level logreg")

print()
print("=" * 92)
print("STAGE K + the rest")
print("=" * 92)
paired(MIX40, PV2, "mixup a=0.4 on Phikon vs Phikon")
paired(FMADD, PV2, "additive on Phikon vs ABMIL on Phikon")
paired(CLEANADD, PV2, "clean interpretable vs Phikon-v2")
paired(CLEANNOCTX, CLEANALL, "clean_noctx vs clean_all (staining cost)")
paired(CLEANADD, DIRTYADD, "clean additive vs dirty additive")

print()
print("=" * 92)
print("LEVELS")
print("=" * 92)
for lab, spec in [
        ("slide-level RF", ("abmil_v5nuc_gbm_results.csv", "rf")),
        ("slide-level GBM", GBMC),
        ("slide-level logreg", LOGREG),
        ("clean interpretable ABMIL", CLEANALL),
        ("clean interpretable, no chrom_context", CLEANNOCTX),
        ("clean interpretable Additive", CLEANADD),
        ("dirty interpretable Additive", DIRTYADD),
        ("Phikon-v2 ABMIL", PV2),
        ("Phikon-v2 Additive", FMADD),
        ("Phikon-v2 + mixup0.4", MIX40)]:
    s = cells(*spec)
    if s is None:
        print(f"  {lab:42s} MISSING")
        continue
    s = s[~s.index.duplicated()]
    print(f"  {lab:42s} {s.mean():.4f}  sd {s.std():.4f}  n={len(s)}")

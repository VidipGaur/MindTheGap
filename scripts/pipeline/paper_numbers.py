# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Every number the paper quotes, computed from the results CSVs.

WHY THIS EXISTS
---------------
A draft that carries numbers typed in by hand drifts from the experiments the
moment an arm is re-run. This script is the single source of truth: it reads the
per-fold CSVs, does every paired comparison the paper makes, and writes
paper/numbers.json plus a readable dump. The figure scripts read the same JSON,
so a table and the figure beside it cannot disagree.

Every comparison is PAIRED on shared (repeat, fold) cells. Per-cell sd is ~0.055
against effects of ~0.02, so unpaired means are not evidence at this n.

USAGE
    python scripts/pipeline/paper_numbers.py            # print + write numbers.json
    python scripts/pipeline/paper_numbers.py --quiet    # write only
"""
from __future__ import annotations

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import corpus_paths

ROOT = corpus_paths.ROOT
ODIR = ROOT / "cellvit_run" / "graphs" / "gatv2_run"
OUT = ROOT / "paper" / "numbers.json"

CFG = "abmil_h128_lr1"
ADD = "addmil_h128_do5"

ENCODERS = [
    ("uni2h", "UNI2-h", "ViT-H/14", 1536),
    ("hoptimus0", "H-optimus-0", "ViT-g/14", 1536),
    ("phikonv2", "Phikon-v2", "ViT-L/16", 1024),
    ("hibouL", "Hibou-L", "ViT-L/14", 1024),
]

CLEAN = ("abmil_v5nucchromeos_clean_final_results.csv", ADD)


def cells(fname, cfg=None):
    p = ODIR / fname
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if cfg:
        d = d[d.config == cfg]
    d = d.dropna(subset=["balanced_acc"])
    if not len(d):
        return None
    return d.set_index(["repeat", "fold"])["balanced_acc"].sort_index()


def paired(a_spec, b_spec):
    """Paired comparison over shared cells. Returns a dict or None."""
    a = cells(*a_spec) if isinstance(a_spec, tuple) else a_spec
    b = cells(*b_spec) if isinstance(b_spec, tuple) else b_spec
    if a is None or b is None:
        return None
    i = a.index.intersection(b.index)
    if len(i) < 3:
        return None
    a, b = a.loc[i], b.loc[i]
    d = a - b
    try:
        w = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        w = float("nan")
    return dict(
        mean_a=float(a.mean()), mean_b=float(b.mean()), delta=float(d.mean()),
        sd_a=float(a.std()), sd_diff=float(d.std()),
        p=float(stats.ttest_rel(a, b).pvalue), wilcoxon=w,
        win=float((d > 0).mean()), n=int(len(i)))


def show(label, r):
    if r is None:
        print(f"  {label:46s} MISSING")
        return
    pstr = f"{r['p']:.2e}" if r["p"] < 1e-4 else f"{r['p']:.4f}"
    print(f"  {label:46s} {r['mean_a']:.4f} vs {r['mean_b']:.4f}  "
          f"d={r['delta']:+.4f}  p={pstr}  win={r['win']:.2f}  n={r['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    N = {}

    N["corpus"] = dict(n_slides=456, classes=["CC", "EC", "HGSC", "LGSC", "MC"],
                       counts=dict(HGSC=201, EC=105, CC=81, LGSC=36, MC=33),
                       tile_um=256, mpp=0.5, repeats=10, folds=5)

    print("\n=== ENCODER LEDGER (Additive MIL, vs clean interpretable) ===")
    clean = cells(*CLEAN)
    N["clean"] = dict(mean=float(clean.mean()), sd=float(clean.std()),
                      n=int(len(clean)))
    print(f"  clean interpretable (Additive MIL) = {clean.mean():.4f} "
          f"(sd {clean.std():.4f}, n={len(clean)})")

    N["encoders"] = {}
    for key, nice, arch, dim in ENCODERS:
        e = {}
        full = (f"abmil_v5fm{key}_fm_addmil_results.csv", ADD)
        abmil = (f"abmil_v5fm{key}_fm_{key}_results.csv", CFG)
        e["arch"], e["dim"], e["name"] = arch, dim, nice
        c = cells(*full)
        e["additive"] = float(c.mean()) if c is not None else None
        c2 = cells(*abmil)
        e["abmil"] = float(c2.mean()) if c2 is not None else None
        e["gap_vs_clean"] = paired(full, CLEAN)
        # the hybrid arms were run under the FEATURE reference head (CFG), not
        # the additive head, so they pair against the encoder's own ABMIL arm.
        e["hybrid"] = paired(
            (f"abmil_v5nucfm{key}cat_fm_hybrid_results.csv", CFG), abmil)
        hc = cells(f"abmil_v5nucfm{key}cat_fm_hybrid_results.csv", CFG)
        e["hybrid_mean"] = float(hc.mean()) if hc is not None else None
        # subspace deletion
        R = {arm: (f"abmil_v5fm{key}{arm}_fm_{arm}_results.csv", ADD)
             for arm in ("resid", "residnull", "residctl")}
        e["resid_vs_null"] = paired(R["resid"], R["residnull"])
        e["resid_vs_ctl"] = paired(R["resid"], R["residctl"])
        e["resid_vs_full"] = paired(R["resid"], full)
        e["null_vs_full"] = paired(R["residnull"], full)
        e["resid_vs_clean"] = paired(R["resid"], CLEAN)
        for arm in ("resid", "residnull", "residctl"):
            cc = cells(*R[arm])
            e[arm] = float(cc.mean()) if cc is not None else None
        # stain audit
        S = {arm: (f"abmil_v5fm{key}stain{arm}_fm_stain{arm}_results.csv", ADD)
             for arm in ("resid", "residnull", "residctl")}
        e["stain_resid_vs_null"] = paired(S["resid"], S["residnull"])
        e["stain_resid_vs_full"] = paired(S["resid"], full)
        for arm in ("resid", "residnull", "residctl"):
            cc = cells(*S[arm])
            e[f"stain_{arm}"] = float(cc.mean()) if cc is not None else None
        N["encoders"][key] = e

        print(f"\n  {nice} ({arch}, {dim}-d)  additive={e['additive']}")
        show("gap vs clean interpretable", e["gap_vs_clean"])
        show("hybrid (concat named) vs encoder", e["hybrid"])
        show("resid vs residnull  (THE TEST)", e["resid_vs_null"])
        show("resid vs residctl   (variance-matched)", e["resid_vs_ctl"])
        show("residnull vs full   (sanity ~0)", e["null_vs_full"])
        show("resid vs clean interpretable", e["resid_vs_clean"])
        show("STAIN resid vs residnull", e["stain_resid_vs_null"])

    print("\n=== SUBSPACE SHARES (delta / encoder advantage) ===")
    for key, nice, _, _ in ENCODERS:
        e = N["encoders"][key]
        adv = e["gap_vs_clean"]
        for name, k in (("rank-matched", "resid_vs_null"),
                        ("variance-matched", "resid_vs_ctl")):
            r = e[k]
            if r and adv:
                e[f"share_{k}"] = abs(r["delta"]) / abs(adv["delta"]) * 100
                print(f"  {nice:14s} {name:18s} "
                      f"{abs(r['delta']):.4f} / {abs(adv['delta']):.4f} = "
                      f"{e[f'share_{k}']:5.1f}%")

    print("\n=== SLIDE-LEVEL CONTROLS (no bag structure) ===")
    g = ODIR / "abmil_v5nuc_gbm_results.csv"
    if g.exists():
        d = pd.read_csv(g)
        N["slidelevel"] = {}
        for cfg, sub in d.groupby("config"):
            m = float(sub.balanced_acc.mean())
            N["slidelevel"][cfg] = m
            print(f"  {cfg:34s} {m:.4f}  (n={len(sub)})")

    print("\n=== INTERPRETABLE ARM ===")
    pairs = [
        ("dirty (with dead_+chrom_context)",
         ("abmil_v5nucchrom_final_additive_results.csv", ADD), CLEAN),
        ("nuclear-only vs clean-final",
         ("abmil_v5nuc_sel_swatop_results.csv", CFG), CLEAN),
    ]
    N["interpretable"] = {}
    for label, A, B in pairs:
        r = paired(A, B)
        N["interpretable"][label] = r
        show(label, r)

    # The refit runs the bridge regression once per repeat. pct_ceiling is a
    # RATIO of two noisy R^2 values, and the denominator (the interpretable
    # model's own self-ceiling) collapses on some repeats -- repeat 3 spikes on
    # every encoder simultaneously, which is a shared-denominator artefact, not
    # four independent findings. Values above 100% are impossible as a
    # "fraction of ceiling" and are the tell. So we report the RAW encoder R^2,
    # which has no denominator, and give the ratio only as a median with its
    # spread and an explicit note.
    print("\n=== BRIDGE (raw R^2 has no denominator; ratio is unstable) ===")
    N["bridge"] = {}
    bdir = ODIR
    for key, nice, _, _ in ENCODERS:
        rows = []
        for p in sorted(bdir.glob(f"bridge_repeats_v5fm{key}/r2_by_class_r*.csv")):
            rows.append(pd.read_csv(p))
        if not rows:
            continue
        d = pd.concat(rows, ignore_index=True).dropna(subset=["pct_ceiling"])
        per_class = {}
        for cls, sub in d.groupby("cls"):
            per_class[cls] = dict(
                r2_med=float(sub["phikon"].median()),
                r2_q25=float(sub["phikon"].quantile(0.25)),
                r2_q75=float(sub["phikon"].quantile(0.75)),
                ceil_med=float(sub["self"].median()),
                pct_med=float(sub["pct_ceiling"].median()),
                pct_q25=float(sub["pct_ceiling"].quantile(0.25)),
                pct_q75=float(sub["pct_ceiling"].quantile(0.75)),
                n=int(len(sub)))
        # per-repeat median across classes, the quantity the sweep reported
        per_rep = d.groupby("repeat")["pct_ceiling"].median()
        N["bridge"][key] = dict(
            name=nice, per_class=per_class,
            pct_med=float(per_rep.median()),
            pct_q25=float(per_rep.quantile(0.25)),
            pct_q75=float(per_rep.quantile(0.75)),
            pct_max=float(per_rep.max()),
            n_repeats=int(per_rep.shape[0]),
            n_over_100=int((per_rep > 100).sum()))
        b = N["bridge"][key]
        print(f"  {nice:14s} median {b['pct_med']:6.1f}%  "
              f"IQR [{b['pct_q25']:.1f}, {b['pct_q75']:.1f}]  "
              f"max {b['pct_max']:.0f}%  "
              f"repeats over 100%: {b['n_over_100']}/{b['n_repeats']}")
        for cls in ("CC", "EC", "HGSC", "LGSC", "MC"):
            if cls in per_class:
                c = per_class[cls]
                print(f"      {cls:5s} R2 {c['r2_med']:.3f} "
                      f"[{c['r2_q25']:.3f},{c['r2_q75']:.3f}]   "
                      f"ceiling {c['ceil_med']:.3f}   "
                      f"pct {c['pct_med']:5.1f}%")

    print("\n=== SELECTION RULES (10 repeats, nuclear-only) ===")
    N["selection"] = {}
    for rule in ("argmax_es", "argmax", "smooth", "fixed", "swa", "swaflr",
                 "swatop"):
        c = cells(f"abmil_v5nuc_sel_{rule}_results.csv", CFG)
        if c is not None:
            N["selection"][rule] = float(c.mean())
            print(f"  {rule:12s} {c.mean():.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(N, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

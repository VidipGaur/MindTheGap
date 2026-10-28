# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Paired config comparison for the ABMIL sweeps.

    python scripts/analysis/compare_abmil_configs.py --tag v5nuc --base abmil_h128_lr1

Config means alone are misleading here: the per-config sd is ~0.06 while the
gaps between the top configs are ~0.01. But every config is evaluated on the
SAME (repeat, fold) splits, so the comparison can be paired, which removes the
fold-difficulty variance that dominates that sd. This prints the paired delta,
its own sd, and a paired t-test / Wilcoxon per config against a baseline.

Also prints the arm-vs-arm comparison (e.g. flags vs --drop-flags, or a
feature-group ablation vs the full feature set) when --other is given.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import corpus_paths

ROOT = str(corpus_paths.ROOT)
OUT = "cellvit_run/graphs/gatv2_run"


def load(root, tag):
    p = Path(root) / OUT / f"abmil_{tag}_results.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    df = pd.read_csv(p)
    return df[df.balanced_acc.notna()]


def paired(df, a, b):
    """Delta a-b over the (repeat, fold) cells both ran."""
    sa = df[df.config == a].set_index(["repeat", "fold"])["balanced_acc"]
    sb = df[df.config == b].set_index(["repeat", "fold"])["balanced_acc"]
    common = sa.index.intersection(sb.index)
    if len(common) < 3:
        return None
    d = (sa.loc[common] - sb.loc[common]).values
    t = stats.ttest_rel(sa.loc[common].values, sb.loc[common].values)
    try:
        w = stats.wilcoxon(d).pvalue
    except Exception:
        w = np.nan
    return dict(n=len(d), delta=d.mean(), sd=d.std(ddof=1),
                t_p=t.pvalue, wilcoxon_p=w,
                win=float((d > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag", default="v5nuc")
    ap.add_argument("--base", default="abmil_h128_lr1")
    ap.add_argument("--other", default=None,
                    help="second results tag to compare arm-vs-arm "
                         "(e.g. v5nucnoflag, v5nuc_nuconly)")
    args = ap.parse_args()

    df = load(args.root, args.tag)
    g = df.groupby("config")["balanced_acc"].agg(["mean", "std", "count"])
    g = g.sort_values("mean", ascending=False)

    print("=" * 84)
    print(f"PAIRED vs {args.base}   [{args.tag}]")
    print("=" * 84)
    rows = []
    for cfg in g.index:
        if cfg == args.base:
            continue
        r = paired(df, cfg, args.base)
        if r:
            rows.append(dict(config=cfg, mean=g.loc[cfg, "mean"], **r))
    if rows:
        t = pd.DataFrame(rows).sort_values("delta", ascending=False)
        print(t.round(4).to_string(index=False))
        print("  delta = config minus base, paired on (repeat, fold).")
        print("  win = fraction of the 15 folds where the config beat the base.")
        print("\n  A config is only meaningfully better if the paired delta is")
        print("  positive AND the p-values are small AND win is well above 0.5.")
        print("  Config-mean differences alone are not evidence at this sd.")
    else:
        print(f"  (only one config in this arm -- nothing to pair against "
              f"within it; see the arm-vs-arm section below)")
    if args.base in g.index:
        print(f"\n  base {args.base} mean = {g.loc[args.base, 'mean']:.4f}")

    if args.other:
        d2 = load(args.root, args.other)
        print("\n" + "=" * 84)
        print(f"ARM vs ARM: {args.tag} minus {args.other}")
        print("=" * 84)
        rows = []
        for cfg in sorted(set(df.config) & set(d2.config)):
            sa = df[df.config == cfg].set_index(["repeat", "fold"])["balanced_acc"]
            sb = d2[d2.config == cfg].set_index(["repeat", "fold"])["balanced_acc"]
            common = sa.index.intersection(sb.index)
            if len(common) < 3:
                continue
            d = (sa.loc[common] - sb.loc[common]).values
            rows.append(dict(config=cfg, n=len(d),
                             a_mean=sa.loc[common].mean(),
                             b_mean=sb.loc[common].mean(),
                             delta=d.mean(), sd=d.std(ddof=1),
                             t_p=stats.ttest_rel(sa.loc[common].values,
                                                 sb.loc[common].values).pvalue))
        if rows:
            tt = pd.DataFrame(rows).sort_values("delta", ascending=False)
            print(tt.round(4).to_string(index=False))
            print(f"\n  positive delta = {args.tag} is better than {args.other}")


if __name__ == "__main__":
    main()

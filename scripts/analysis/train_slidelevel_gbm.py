# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Slide-level control: how much of the 0.67 needs MIL at all?

    python scripts/analysis/train_slidelevel_gbm.py --nuclear-features --drop-groups graph
    python scripts/analysis/train_slidelevel_gbm.py --nuclear-features --drop-groups graph --repeats 10

WHAT THIS ASKS
==============
The ABMIL model treats a slide as a BAG of tiles and learns which tiles matter.
That machinery is the paper's interpretability story. But it is only worth
anything if it beats the obvious alternative: throw away the bag structure
entirely, average every feature over the whole slide, and hand the resulting
fixed-length vector to a strong tabular classifier.

If gradient boosting on slide-level aggregates matches ABMIL, then the signal
is essentially slide-level composition, the attention mechanism is decoration,
and the honest framing of the paper changes. That is an uncomfortable question,
which is exactly why it should be answered here.

It is also a genuine positive control of a different kind from Task 4: Task 4
asks whether better FEATURES (pretrained embeddings) beat ours, this asks
whether a simpler MODEL matches ours on the same features.

HOW SLIDE VECTORS ARE BUILT
===========================
Each per-tile feature becomes five slide-level columns -- mean, sd, p10, p50,
p90 across the slide's tiles. The percentiles matter: pleomorphism is a
distributional property, and a slide whose tiles are uniformly moderate has the
same mean as one with a mix of bland and wildly atypical tiles. Giving the
tabular model the spread is what makes this a fair fight rather than a
strawman.

Folds are StratifiedKFold(5, shuffle=True, random_state=repeat), identical to
train_abmil.py, so results pair per (repeat, fold) and can go straight into
compare_abmil_configs.py.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_abmil import (CLASSES, ROOT, RUN_DIR_NAME, GRAPH_SUBFOLDER,
                         OUT_SUBFOLDER, load_bags, Normaliser)

STATS = ("mean", "sd", "p10", "p50", "p90")


def slide_matrix(X, feat_names):
    """Bags -> (n_slides, n_features * len(STATS)) plus column names."""
    rows = []
    for b in X:
        a = b.numpy()
        p10, p50, p90 = np.percentile(a, [10, 50, 90], axis=0)
        rows.append(np.concatenate([a.mean(0), a.std(0), p10, p50, p90]))
    cols = [f"{f}_{s}" for s in STATS for f in feat_names]
    return np.nan_to_num(np.stack(rows).astype(np.float64)), cols


def make_model(name, seed):
    if name == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=10, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=2, random_state=seed)
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced",
                               random_state=seed))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag", default="v5nuc")
    ap.add_argument("--arm", default="gbm")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=["gbm", "rf", "logreg"])
    ap.add_argument("--nuclear-features", action="store_true")
    ap.add_argument("--drop-groups", nargs="+", default=None)
    ap.add_argument("--keep-groups", nargs="+", default=None)
    ap.add_argument("--drop-flags", action="store_true")
    ap.add_argument("--exclude-prefix", nargs="+", default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--importance", action="store_true",
                    help="permutation importance on the best model, repeat 0")
    args = ap.parse_args()

    gdir = Path(args.root) / RUN_DIR_NAME / GRAPH_SUBFOLDER
    odir = Path(args.root) / RUN_DIR_NAME / OUT_SUBFOLDER
    nuclear_dir = gdir / "nuclear_feats" if args.nuclear_features else None
    cache_tag = args.tag if not args.nuclear_features or "nuc" in args.tag \
        else f"{args.tag}nuc"

    X, y, ids, feat_names = load_bags(
        gdir, odir / f"bag_cache_{cache_tag}.pt", args.rebuild_cache,
        args.exclude_prefix, nuclear_dir=nuclear_dir,
        drop_flags=args.drop_flags, drop_groups=args.drop_groups,
        keep_groups=args.keep_groups)

    Z, cols = slide_matrix(X, feat_names)
    print(f"\nslide matrix: {Z.shape[0]} slides x {Z.shape[1]} features "
          f"({len(feat_names)} per-tile features x {len(STATS)} statistics)")
    print(f"class counts: {dict(zip(CLASSES, np.bincount(y, minlength=5)))}\n")

    tag = f"{cache_tag}_{args.arm}"
    res_path = odir / f"abmil_{tag}_results.csv"
    rows = []
    t0 = time.time()
    oof = {}
    for name in args.models:
        for rep in range(args.repeats):
            skf = StratifiedKFold(5, shuffle=True, random_state=rep)
            for fold, (tr, te) in enumerate(skf.split(Z, y)):
                ts = time.time()
                m = make_model(name, seed=rep * 100 + fold)
                m.fit(Z[tr], y[tr])
                pte = m.predict(Z[te])
                ptr = m.predict(Z[tr])
                bal = balanced_accuracy_score(y[te], pte)
                if rep == 0:
                    a, b = oof.get(name, ([], []))
                    oof[name] = (a + list(y[te]), b + list(pte))
                rows.append(dict(config=name, repeat=rep, fold=fold,
                                 balanced_acc=round(float(bal), 4),
                                 val_bal=np.nan,
                                 train_bal=round(float(
                                     balanced_accuracy_score(y[tr], ptr)), 4),
                                 arch="slidelevel", select="none",
                                 secs=round(time.time() - ts, 1)))
                pd.DataFrame(rows).to_csv(res_path, index=False)
            print(f"  {name:8s} repeat {rep}  "
                  f"bal={np.mean([r['balanced_acc'] for r in rows if r['config']==name and r['repeat']==rep]):.4f}"
                  f"  ({(time.time()-t0)/60:.1f}m)")

    df = pd.DataFrame(rows)
    g = df.groupby("config")[["balanced_acc", "train_bal"]].agg(["mean", "std", "count"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.sort_values("balanced_acc_mean", ascending=False).round(4)
    print("\n" + "=" * 78)
    print("SLIDE-LEVEL CONTROL -- no bag structure, no attention")
    print("=" * 78)
    print(g.to_string())

    best = g.index[0]
    b = float(g.balanced_acc_mean.iloc[0])
    print(f"\nbest slide-level model: {best} = {b:.4f}")
    print("\n--- what this means for the MIL result ---")
    print("  ABMIL on the same features (abmil_h128_lr1, nuclear-only) = 0.6732.")
    if b >= 0.66:
        print(f"  This reaches {b:.3f}, i.e. LEVEL WITH the MIL model. The bag")
        print("  structure and the attention mechanism are not buying accuracy:")
        print("  the signal is slide-level composition. Attention can still be")
        print("  defended as the thing that makes predictions explainable per")
        print("  tile, but it must NOT be claimed to improve classification, and")
        print("  the paper needs this baseline in it.")
    elif b >= 0.62:
        print(f"  This reaches {b:.3f}, a little below the MIL model. Some of the")
        print("  gain is genuinely from modelling the slide as a bag, but most of")
        print("  the signal survives simple aggregation. Report both.")
    else:
        print(f"  This reaches {b:.3f}, clearly below the MIL model. Treating the")
        print("  slide as a bag of tiles and learning which tiles matter is doing")
        print("  real work beyond averaging -- that is the result that justifies")
        print("  the architecture.")

    if best in oof and len(oof[best][0]):
        ty, tp = np.array(oof[best][0]), np.array(oof[best][1])
        cm = confusion_matrix(ty, tp, labels=range(len(CLASSES)))
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        print(f"\n--- CONFUSION MATRIX, {best}, repeat 0 out-of-fold ---")
        print("  " + pd.DataFrame(cmn.round(3), index=CLASSES,
                                  columns=CLASSES).to_string().replace("\n", "\n  "))

    if args.importance:
        from sklearn.inspection import permutation_importance
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        tr, te = next(iter(skf.split(Z, y)))
        m = make_model(best, seed=0).fit(Z[tr], y[tr])
        r = permutation_importance(m, Z[te], y[te], n_repeats=10,
                                   random_state=0, scoring="balanced_accuracy")
        imp = pd.DataFrame({"feature": cols, "imp": r.importances_mean}) \
            .sort_values("imp", ascending=False).head(25)
        print("\n--- top slide-level features (permutation importance) ---")
        print("  " + imp.round(4).to_string(index=False).replace("\n", "\n  "))
        print("\n  Compare these against the per-tile drivers explain_slide.py")
        print("  reports. Agreement means the two models are reading the same")
        print("  biology; disagreement means the MIL model is using something")
        print("  that only exists at tile level, which is the interesting case.")

    print(f"\nPer-run: {res_path}")
    print(f"Pair it against the MIL arms with:\n"
          f"  python scripts/analysis/compare_abmil_configs.py --tag {tag} --base {best}")


if __name__ == "__main__":
    main()

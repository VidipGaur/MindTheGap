# Copyright 2026 Vidip Gaur, Vidu Gaur
"""How much of what Phikon does per tile is expressible in named pathology terms?

THE PROBLEM THIS EXISTS TO SOLVE
================================
Stage J gave Additive MIL on Phikon-v2: per-tile per-class contributions that sum
EXACTLY to the bag logit. That is exact credit assignment -- where the model
decided, and how much, per class -- but it is not interpretability in any
pathological sense, because the instance vector is 1024 anonymous dimensions.
"Tile 412 contributed +0.83 toward HGSC" is true and unhelpful.

Separately, every one of those same tiles carries ~56 NAMED measurements on the
identical grid, alignment-verified by grid_rc_checksum at extraction: nuclear
area and its variability, eccentricity, solidity, chromatin level, nuclear
spacing, cytoplasmic clearing, mucin.

So the question is answerable directly rather than rhetorically:

    contribution_to_class_c(tile) ~ named_features(tile)

The out-of-fold R^2 of that regression is a NUMBER for how much of a foundation
model's per-tile decision is diagnostically legible. Its coefficients say which
named quantities, per class. Its residual isolates the tiles where the model is
confident and no named feature explains it -- which is the interesting set.

WHAT EACH OUTPUT MEANS, AND HOW IT COULD MISLEAD
================================================
R^2 alone is not the claim. Three controls make it one:

1. SELF-CEILING. The same regression is run against the INTERPRETABLE model's own
   contributions. That model is literally a function of these features, so its R^2
   is the ceiling this method can reach at all -- the encoder MLP is nonlinear, so
   even there it is well under 1.0. Phikon's R^2 must be read as a fraction of
   that ceiling, never against 1.0. Without this control a mediocre-looking 0.45
   would be reported as a failure when it might be 90% of what is achievable.

2. DENSITY RESIDUALISATION. Tile density predicts almost everything in this
   corpus, and "the model attends to dense tumour" is the null hypothesis that
   has been controlled for at every other step of this project (the deletion
   test's density-matched null, the attention correlations). Nucleus count is
   regressed out of both sides before the interesting number is computed.

3. SLIDE-GROUPED CV. Tiles within a slide are heavily correlated, so plain k-fold
   over tiles leaks and inflates R^2 badly. Folds are grouped by slide.

INTERPRETING THE RESULT (fixed in advance so it is not chosen after the fact)
============================================================================
  HIGH  (>60% of self-ceiling, coefficients matching WHO criteria per class)
        -> a foundation model rediscovers the diagnostic criteria without ever
           being supervised on them. The named features EXPLAIN the strong model
           even though they cannot MATCH it (they score 0.65 alone, and actively
           HURT when concatenated: -0.0094, p=0.046). That is the paper.
  LOW   (<30% of self-ceiling)
        -> foundation models use tile properties that named morphometry does not
           capture. A weaker headline but a real and reportable negative, and the
           residual tiles become a hypothesis generator for features pathologists
           do not currently measure.
  MIXED (class-inconsistent, no coherent coefficient story)
        -> no claim. Do not dress it up.

Usage:
    python scripts/pipeline/bridge_fm_interpretable.py                    # full run
    python scripts/pipeline/bridge_fm_interpretable.py --slides 60        # quick smoke
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from analyse_abmil_nuclear import (arm_paths, bags_for, check_feature_match,
                                   rebuild_model)
from train_abmil import CLASSES
import corpus_paths

FM_TAG, FM_CFG = "v5fmphikonv2_fm_addmil", "addmil_h128_do5"
INT_TAG, INT_CFG = "v5nucchromeos_clean_additive", "addmil_h128_do5"
INT_DROP = ["graph", "chrom_texture", "eos_other"]
INT_EXCL = ["dead_"]
# Residualising on nucleus count alone left connective_count, inflam_count and
# nuc_per_mm2 among the strongest Spearman terms in the first full run -- those
# are still density, just of a different cell type, so "the model found the
# cellular tumour" was only partly controlled. Everything that is a raw count or
# a per-area rate comes out of both sides now.
DENSITY = ("nuc_count", "dens_all_mean", "nuc_per_mm2", "neo_count",
           "connective_count", "inflam_count", "epithelial_count",
           "dead_count", "tile_tissue_frac")
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)

# Presence/availability flags dominated the CC, EC, HGSC and MC driver tables in
# the first run (`neo_measured` was HGSC's single largest coefficient at
# -0.0605). Phikon cannot see our flags -- they are a proxy for "this tile had
# >=10 neoplastic nuclei", i.e. tumour cellularity, and they are an artefact of
# OUR gating rather than a pathology quantity. They are the same confound family
# as dead_count. Keeping them in measures how well our pipeline's availability
# gating tracks Phikon, which is not the question.
FLAG_SUFFIX = ("_measured",)


@torch.no_grad()
def contributions(root, tag, config, X, y, repeats, device, repeat_only=None):
    """Per-slide (N, C) additive contributions, averaged over every repeat in
    which the slide was out-of-fold. Every repeat scores every slide OOF, so the
    average is over 10 honest estimates -- the same ensembling that lifted the
    attention-map q25 from 0.708 to 0.906, applied to contributions.

    `repeat_only` restricts the average to a single repeat. That is what makes
    the bridge R^2 a DISTRIBUTION rather than one number: run the whole
    measurement once per repeat and the spread across repeats is the error bar
    the headline number never had. Every repeat still scores every slide
    out-of-fold exactly once, so each single-repeat estimate is honest on its
    own -- it is simply noisier than the 10-repeat average, which is the point."""
    _, ckdir = arm_paths(root, tag)
    acc = {i: [] for i in range(len(X))}
    reps = [repeat_only] if repeat_only is not None else range(repeats)
    for rep in reps:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        for fold, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            p = ckdir / f"{config}_r{rep}f{fold}.pt"
            if not p.exists():
                continue
            ck = torch.load(p, map_location=device, weights_only=False)
            check_feature_match(ck, int(X[0].shape[1]), tag, config)
            model, _ = rebuild_model(ck, device)
            nst = ck["normaliser"]
            li, mu, sd = nst["log_idx"], nst["mean"].to(device), nst["std"].to(device)
            for i in te:
                x = X[i].to(device).clone()
                if li:
                    x[..., li] = torch.log1p(x[..., li].clamp(min=0))
                x = ((x - mu) / sd).unsqueeze(0)
                msk = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
                _, _, c = model(x, msk, return_contrib=True)
                acc[i].append(c[0].cpu().numpy())
    out, n_missing = {}, 0
    for i, lst in acc.items():
        if lst:
            out[i] = np.mean(lst, axis=0)          # (N, C)
        else:
            n_missing += 1
    if n_missing:
        print(f"  WARNING: {n_missing} slides had no usable checkpoint")
    return out


def grouped_r2(Xf, yv, groups, alphas=RIDGE_ALPHAS, nonlinear=False):
    """Out-of-fold R^2 with folds grouped by SLIDE, alpha chosen on an inner
    split. Tiles within a slide are correlated; ungrouped CV inflates this
    badly, which would be the single easiest way to fake this whole analysis.

    `nonlinear` swaps ridge for gradient boosting. This matters more than it
    looks: a contribution is a PRODUCT, attention_n * psi(h_n)_c, of two
    nonlinear functions of the same features. A linear model is close to the
    wrong instrument for it, and the first full run's self-ceiling of only
    0.12-0.15 -- i.e. named features linearly explaining ~13% of the variance in
    contributions of a model that takes exactly those features as its ONLY input
    -- is the symptom. Whether the ceiling rises under gradient boosting decides
    whether the low absolute R^2 is a fact about the features or about ridge."""
    gkf = GroupKFold(n_splits=5)
    pred = np.zeros_like(yv)
    for tr, te in gkf.split(Xf, yv, groups):
        sc = StandardScaler().fit(Xf[tr])
        A, B = sc.transform(Xf[tr]), sc.transform(Xf[te])
        if nonlinear:
            m = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                min_samples_leaf=50, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15,
                random_state=0).fit(A, yv[tr])
            pred[te] = m.predict(B)
            continue
        # inner split on groups to pick alpha, so alpha never sees the test fold
        g_tr = groups[tr]
        uniq = np.unique(g_tr)
        inner = np.isin(g_tr, uniq[: max(1, len(uniq) // 5)])
        best, best_s = alphas[0], -np.inf
        for a in alphas:
            m = Ridge(alpha=a).fit(A[~inner], yv[tr][~inner])
            s = m.score(A[inner], yv[tr][inner])
            if s > best_s:
                best, best_s = a, s
        pred[te] = Ridge(alpha=best).fit(A, yv[tr]).predict(B)
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot, pred


def residualise(v, dens):
    """Remove the part explained by tile density/cellularity. The null
    hypothesis everywhere else in this project is 'the model just found the
    dense tumour'; this keeps that consistent. `dens` is now a MATRIX of every
    count/rate column, not nucleus count alone."""
    D = np.column_stack([np.asarray(dens).reshape(len(v), -1), np.ones(len(v))])
    beta, *_ = np.linalg.lstsq(D, v, rcond=None)
    return v - D @ beta


def coef_table(Xf, yv, names, top=12):
    sc = StandardScaler().fit(Xf)
    m = Ridge(alpha=10.0).fit(sc.transform(Xf), yv)
    o = np.argsort(-np.abs(m.coef_))[:top]
    return [(names[i], float(m.coef_[i])) for i in o]


def run_arm(label, tag, cfg, nuclear, drop_groups, excl, args, device,
            names_out=None):
    """NOTE: the bag list is never truncated before contributions() runs.

    The fold assignment is StratifiedKFold(5, random_state=rep) over the FULL
    456-bag order, so truncating first silently renumbers the folds and slides
    get scored by checkpoints that trained on them. --slides therefore caps only
    how many slides enter the regression, after every contribution has been
    computed out-of-fold against the real folds."""
    print(f"\n  loading bags for {label} ...", flush=True)
    X, y, ids, fn = bags_for(args.root, tag, nuclear, drop_groups=drop_groups,
                             exclude_prefix=excl)
    print(f"  {len(X)} bags, {X[0].shape[1]} features", flush=True)
    C = contributions(args.root, tag, cfg, X, y, args.repeats, device,
                      repeat_only=args.repeat_only)
    if names_out is not None:
        names_out.extend(fn)
    return X, y, ids, fn, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(corpus_paths.ROOT))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--slides", type=int, default=0, help="cap for a smoke run")
    ap.add_argument("--nonlinear", action="store_true",
                    help="gradient boosting instead of ridge. A contribution is "
                         "a PRODUCT of two nonlinear functions of the features, "
                         "so ridge may simply be the wrong instrument -- compare "
                         "the SELF-CEILING between the two before reading "
                         "anything into the absolute R^2.")
    ap.add_argument("--keep-flags", action="store_true",
                    help="keep the *_measured presence flags in the named set. "
                         "Off by default: they are our own gating artefact, not "
                         "a pathology quantity, and they dominated four of five "
                         "class driver tables in the first run.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fm-tag", default=FM_TAG,
                    help="the additive arm tag of the encoder to bridge "
                         "(default the Phikon-v2 one). Parameterised so the "
                         "same measurement can be repeated per encoder -- one "
                         "R2 on one model is a fact about that model, not "
                         "about pathology foundation models.")
    ap.add_argument("--fm-name", default="Phikon-v2",
                    help="display name for the encoder being bridged")
    ap.add_argument("--repeat-only", type=int, default=None,
                    help="use ONE repeat's checkpoints instead of averaging all "
                         "of them. Run 0..9 in turn to get a distribution of the "
                         "bridge R^2 rather than a point estimate -- without it "
                         "there is no way to tell a real between-encoder "
                         "difference from re-seeding noise.")
    ap.add_argument("--brief", action="store_true",
                    help="skip the driver / unexplained-tile / Spearman blocks "
                         "and their CSVs. Only the per-class R^2 table and the "
                         "verdict are produced. For repeat sweeps, where those "
                         "blocks would be recomputed 10x and thrown away.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fm_tag = args.fm_tag
    print("=" * 78)
    print(f"BRIDGE -- is {args.fm_name}'s per-tile decision expressible in "
          f"named pathology terms?")
    print("=" * 78)

    Xi, yi, ids_i, names, Ci = run_arm(
        "interpretable (self-ceiling)", INT_TAG, INT_CFG, True, INT_DROP,
        INT_EXCL, args, device)

    Xf, yf, ids_f, _, Cf = run_arm(
        f"{args.fm_name} additive", fm_tag, FM_CFG, False, None, None, args,
        device)

    pos_i = {s: k for k, s in enumerate(ids_i)}
    pos_f = {s: k for k, s in enumerate(ids_f)}
    shared = [s for s in ids_i if s in pos_f]
    print(f"\n  {len(shared)} slides present in both arms")
    if args.slides:
        shared = shared[: args.slides]
        print(f"  --slides: regression restricted to {len(shared)} of them "
              f"(contributions were still computed out-of-fold on all slides)")

    rows_x, rows_cf, rows_ci, grp, lab = [], [], [], [], []
    bad = 0
    for s in shared:
        a, b = pos_i[s], pos_f[s]
        if a not in Ci or b not in Cf:
            continue
        if Xi[a].shape[0] != Cf[b].shape[0]:
            bad += 1
            continue                      # different tile count => not the same grid
        rows_x.append(Xi[a].numpy())
        rows_ci.append(Ci[a])
        rows_cf.append(Cf[b])
        grp.append(np.full(Xi[a].shape[0], a))
        lab.append(np.full(Xi[a].shape[0], yi[a]))
    if bad:
        print(f"  WARNING: {bad} slides dropped on tile-count mismatch")
    Xmat = np.concatenate(rows_x).astype(np.float64)
    CFm = np.concatenate(rows_cf).astype(np.float64)
    CIm = np.concatenate(rows_ci).astype(np.float64)
    groups = np.concatenate(grp)
    labels = np.concatenate(lab)
    print(f"  {Xmat.shape[0]:,} tiles x {Xmat.shape[1]} named features")

    di = [k for k, n in enumerate(names) if n in DENSITY]
    dens = Xmat[:, di] if di else np.zeros((len(Xmat), 1))
    print(f"  residualising on {len(di)} density/cellularity column(s): "
          f"{[names[k] for k in di]}")

    # presence flags are gating artefacts of our own pipeline, not pathology
    if not args.keep_flags:
        fi = [k for k, n in enumerate(names) if n.endswith(FLAG_SUFFIX)]
        if fi:
            print(f"  dropping {len(fi)} presence flag(s) from the named set: "
                  f"{[names[k] for k in fi]}")
            keep = [k for k in range(len(names)) if k not in set(fi)]
            Xmat = Xmat[:, keep]
            names = [names[k] for k in keep]
    print(f"  regressing on {Xmat.shape[1]} named pathology columns"
          f"  [{'gradient boosting' if args.nonlinear else 'ridge'}]")

    print("\n" + "=" * 78)
    print("R^2 OF NAMED FEATURES ON PER-TILE CLASS CONTRIBUTION")
    print("  self = the interpretable model's own contributions (the ceiling")
    print("         this method can reach at all -- its encoder is nonlinear)")
    print("  phikon = Phikon-v2's contributions.  resid = density removed.")
    print("=" * 78)
    print(f"  {'class':6s} {'phikon':>8s} {'phik/res':>9s} "
          f"{'self':>8s} {'self/res':>9s} {'% of ceiling':>13s}")
    summary = []
    for c, cname in enumerate(CLASSES):
        rf, _ = grouped_r2(Xmat, CFm[:, c], groups, nonlinear=args.nonlinear)
        ri, _ = grouped_r2(Xmat, CIm[:, c], groups, nonlinear=args.nonlinear)
        rfr, _ = grouped_r2(Xmat, residualise(CFm[:, c], dens), groups, nonlinear=args.nonlinear)
        rir, _ = grouped_r2(Xmat, residualise(CIm[:, c], dens), groups, nonlinear=args.nonlinear)
        frac = (rfr / rir * 100) if rir > 0.01 else float("nan")
        print(f"  {cname:6s} {rf:8.3f} {rfr:9.3f} {ri:8.3f} {rir:9.3f} "
              f"{frac:12.0f}%")
        summary.append(dict(cls=cname, phikon=rf, phikon_resid=rfr,
                            self=ri, self_resid=rir, pct_ceiling=frac))

    if args.brief:
        med = np.nanmedian([s["pct_ceiling"] for s in summary])
        out = Path(args.out) if args.out else (
            Path(args.root) / "cellvit_run" / "graphs" / "gatv2_run" /
            f"bridge_repeats_{fm_tag.split('_')[0]}")
        out.mkdir(parents=True, exist_ok=True)
        rep = args.repeat_only if args.repeat_only is not None else "all"
        pd.DataFrame(summary).assign(repeat=rep, encoder=args.fm_name).to_csv(
            out / f"r2_by_class_r{rep}.csv", index=False)
        print(f"\nwrote -> {out}")
        print(f"\nBRIDGE_REPEAT_RESULT encoder={args.fm_name} "
              f"repeat={rep} median_pct_ceiling={med:.4f}")
        return

    print("\n" + "=" * 78)
    print(f"TOP NAMED DRIVERS OF {args.fm_name.upper()}'S CONTRIBUTION "
          f"(standardised ridge coef)")
    print("  the test of the claim is whether these match WHO criteria:")
    print("  HGSC -> pleomorphism, CC -> clearing, MC -> mucin/spacing")
    print("=" * 78)
    for c, cname in enumerate(CLASSES):
        print(f"\n  --- {cname} ---")
        for n, v in coef_table(Xmat, residualise(CFm[:, c], dens), names):
            print(f"    {v:+.4f}  {n}")

    print("\n" + "=" * 78)
    print("UNEXPLAINED HIGH-CONTRIBUTION TILES")
    print("  tiles in the top decile of |contribution| whose value the named")
    print("  features predict worst. If the R^2 is low, THIS is the product:")
    print("  morphology the model uses that current features do not name.")
    print("=" * 78)
    unexplained = []
    for c, cname in enumerate(CLASSES):
        v = residualise(CFm[:, c], dens)
        _, pred = grouped_r2(Xmat, v, groups, nonlinear=args.nonlinear)
        hi = np.abs(v) >= np.quantile(np.abs(v), 0.90)
        err = np.abs(v - pred)
        err[~hi] = -np.inf
        top = np.argsort(-err)[:200]
        sl = pd.Series(groups[top]).value_counts().head(6)
        print(f"\n  {cname}: worst 200 of {hi.sum():,} high-contribution tiles "
              f"concentrate in slides (bag idx: n tiles)")
        print("    " + ", ".join(f"{ids_i[int(k)]}:{int(n)}"
                                 for k, n in sl.items()))
        for k, n in sl.items():
            unexplained.append(dict(cls=cname, image_id=ids_i[int(k)],
                                    n_tiles=int(n)))

    print("\n" + "=" * 78)
    print("PER-FEATURE SPEARMAN vs PHIKON CONTRIBUTION (density-residualised)")
    print("=" * 78)
    rows = []
    for c, cname in enumerate(CLASSES):
        v = residualise(CFm[:, c], dens)
        for k, n in enumerate(names):
            rho = stats.spearmanr(Xmat[:, k], v).statistic
            rows.append(dict(cls=cname, feature=n, rho=rho))
    sp = pd.DataFrame(rows)
    piv = sp.pivot(index="feature", columns="cls", values="rho")
    piv["absmax"] = piv.abs().max(1)
    print(piv.sort_values("absmax", ascending=False).head(20).round(3).to_string())

    # A non-default encoder must not silently overwrite the Phikon-v2 bridge
    # results that are already on disk and already written up.
    default_out = (Path(args.root) / "cellvit_run" / "graphs" / "gatv2_run" /
                   "bridge_fm_interpretable")
    if fm_tag != FM_TAG:
        default_out = default_out.with_name(
            f"{default_out.name}_{fm_tag.split('_')[0]}")
    out = Path(args.out or default_out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "r2_by_class.csv", index=False)
    sp.to_csv(out / "spearman_by_feature.csv", index=False)
    pd.DataFrame(unexplained).to_csv(out / "unexplained_slides.csv", index=False)
    print(f"\nwrote -> {out}")

    med = np.nanmedian([s["pct_ceiling"] for s in summary])
    print("\n" + "=" * 78)
    print(f"VERDICT: named features recover a median {med:.0f}% of the "
          f"interpretable\n  model's own explainable variance in "
          f"{args.fm_name}'s per-tile contributions.")
    if med > 60:
        print(f"  HIGH -- {args.fm_name}'s per-tile decisions are largely "
              "expressible in\n  named diagnostic quantities. Check the "
              "coefficient tables match\n  WHO criteria per class before "
              "claiming it.")
    elif med < 30:
        print(f"  LOW -- {args.fm_name} uses tile properties named morphometry "
              "misses.\n  Report as a negative and mine the unexplained tiles.")
    else:
        print("  MIXED -- do not force a story. Report the number and the\n"
              "  per-class coefficients, and let the residual analysis lead.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())

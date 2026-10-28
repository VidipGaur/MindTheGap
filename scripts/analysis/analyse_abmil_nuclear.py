# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Post-hoc analysis of the nuclear-feature ABMIL runs (Task 3 follow-ups).

    python scripts/analysis/analyse_abmil_nuclear.py --what confusion
    python scripts/analysis/analyse_abmil_nuclear.py --what attention
    python scripts/analysis/analyse_abmil_nuclear.py --what stability
    python scripts/analysis/analyse_abmil_nuclear.py --what all

Everything here works off the SAVED CHECKPOINTS in abmil_ckpt/<tag>/, not off
the results CSVs -- the CSVs only store per-run scalars, never per-slide
predictions or attention weights, so the folds are reproduced and re-inferred.
Fold reproduction is exact: StratifiedKFold(5, shuffle=True, random_state=rep)
over the same bag order load_bags() returns, which is the same protocol
train_abmil.py and train_gatv2_sweep2.py both use.

WHAT EACH MODE ANSWERS
======================
confusion  Which CLASSES did the nuclear morphometrics actually fix? The
           headline +0.13 balanced accuracy is an average; if it came from
           HGSC/LGSC/EC (the nuclear-grade splits) that is the mechanism we
           predicted, and if it came from CC/MC it is something else and the
           paper's story needs rewriting. Diffs two arms per-class.

attention  Does the pooling attention actually concentrate on the tiles with
           atypical nuclei? An accuracy number alone does NOT establish this
           -- the model could be reading nuclear features while attending
           somewhere else entirely. Correlates the per-tile attention weight
           against every raw feature, per class, and residualises out tile
           density so "it just found the dense tumour" is excluded (same
           control logic as the v5 deletion test).

stability  The v5 GAT models agreed on the predicted class across seeds only
           54.4% of the time, which is what made single-slide attention
           figures indefensible. That check has never been run on ABMIL. If
           this model is more stable, single-slide figures become viable; if
           not, the paper stays on aggregate evidence.

NOTE ON MULTI-HEAD CONFIGS
==========================
heads>1 checkpoints return attention of shape (N, heads). The per-head maps
are reported separately (that is the point of having heads) and a mean map is
used wherever a single map is needed.
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from train_abmil import (CLASSES, COLS, GRAPH_SUBFOLDER, OUT_SUBFOLDER, ROOT,
                         RUN_DIR_NAME, build_model, load_bags, resolve_groups)

DENSITY_CANDIDATES = ("dens_all_mean", "nuc_count", "nuc_per_mm2")
# random subsets drawn per ensemble size / per split-half, so the reported
# numbers are not an artefact of which seeds happened to land together
ENS_SUBSETS = 20

# used to rebuild a config tuple from a checkpoint's saved config dict.
# Older checkpoints have no `arch` key and are ABMIL.
CFG_DEFAULTS = {"name": "?", "hidden": 128, "layers": 1, "attn_hidden": 128,
                "dropout": 0.3, "lr": 1e-3, "wd": 1e-4, "pool": "attn",
                "heads": 1, "arch": "abmil"}


def arm_paths(root, tag):
    odir = Path(root) / RUN_DIR_NAME / OUT_SUBFOLDER
    return odir, odir / "abmil_ckpt" / tag


def bags_for(root, tag, nuclear, drop_flags=False, drop_groups=None,
             exclude_prefix=None, chromatin=False, eosin=False):
    """Rebuild the exact bag list an arm trained on. cache_tag strips the
    arm/noflag suffixes the same way train_abmil.main() composes them.

    `exclude_prefix` must mirror whatever the arm was TRAINED with -- the
    dead_-excluded arms drop columns by name, not by group, so without this the
    column count comes back 2 too high and check_feature_match() refuses the
    run. That refusal is correct (a mismatched column set would misattribute
    silently), but it made the dead_-free arms unanalysable until this was
    threaded through.

    `chromatin`/`eosin` are the same story one block further out. The final
    model (v5nucchromeos_clean_final) trains on nuclear + chromatin + eosin,
    so loading only the nuclear block returns far too few columns and
    check_feature_match() refuses -- which is why the per-subtype driver
    profiles were still being produced from the older nuclear-only model whose
    top EC driver is the dead_count staining artefact. Pass the same blocks
    the arm was trained with and the counts line up."""
    gdir = Path(root) / RUN_DIR_NAME / GRAPH_SUBFOLDER
    odir = Path(root) / RUN_DIR_NAME / OUT_SUBFOLDER
    cache_tag = tag.split("_")[0].replace("noflag", "")
    return load_bags(gdir, odir / f"bag_cache_{cache_tag}.pt", False,
                     exclude_prefix,
                     nuclear_dir=(gdir / "nuclear_feats") if nuclear else None,
                     chromatin_dir=(gdir / "chromatin_feats") if chromatin
                     else None,
                     eosin_dir=(gdir / "eosin_feats") if eosin else None,
                     drop_flags=drop_flags, drop_groups=drop_groups)


def check_feature_match(ck, n_loaded, tag, config):
    """A checkpoint trained on a different column set than the bags we just
    loaded would silently produce garbage attributions rather than an error.
    The nuclear-only arms drop 39 columns, so this mismatch is one forgotten
    --drop-groups away."""
    n = int(ck["in_dim"])
    if n != n_loaded:
        raise SystemExit(
            f"Feature mismatch for {tag}/{config}: the checkpoint expects "
            f"{n} columns, the loaded bags have {n_loaded}.\n"
            f"  The nuclear-only arms were trained with --drop-groups graph; "
            f"pass the same flag here (--drop-groups graph).")


def apply_norm(x, nst):
    x = x.clone()
    li = nst["log_idx"]
    if li:
        x[..., li] = torch.log1p(x[..., li].clamp(min=0))
    return (x - nst["mean"]) / nst["std"]


def rebuild_model(ck, device):
    """Reconstruct any architecture from its checkpoint.

    Goes through build_model() rather than constructing ABMIL directly, so the
    additive / CLAM / TransMIL checkpoints load here too. Pre-Sep-2
    checkpoints carry no `arch` key and default to abmil, which is what they
    are."""
    cfg = ck["config"]
    tup = tuple(cfg.get(k, CFG_DEFAULTS[k]) if cfg.get(k) is not None
                else CFG_DEFAULTS[k] for k in COLS)
    m = build_model(tup, int(ck["in_dim"]), len(CLASSES))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), int(tup[COLS.index("heads")])


@torch.no_grad()
def infer(model, nst, bag, device):
    """One slide -> (class probabilities, attention map or None).
    Attention is (N,) for single-head, (N, heads) for multi-head."""
    x = apply_norm(bag.to(device), nst).unsqueeze(0)
    msk = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    out, a = model(x, msk, return_attn=True)
    p = torch.softmax(out, 1)[0].cpu().numpy()
    return p, (None if a is None else a[0].cpu().numpy())


def collect(root, tag, config, X, y, repeats, device, want_attn=False):
    """Run every (repeat, fold) checkpoint of one config over its own test
    fold. Returns per-repeat dicts: slide index -> (pred, prob, attn)."""
    _, ckdir = arm_paths(root, tag)
    per_rep = {}
    missing = 0
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        got = {}
        for fold, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            p = ckdir / f"{config}_r{rep}f{fold}.pt"
            if not p.exists():
                missing += 1
                continue
            ck = torch.load(p, map_location=device, weights_only=False)
            check_feature_match(ck, int(X[0].shape[1]), tag, config)
            model, _ = rebuild_model(ck, device)
            nst = ck["normaliser"]
            nst = {"log_idx": nst["log_idx"],
                   "mean": nst["mean"].to(device), "std": nst["std"].to(device)}
            for i in te:
                prob, a = infer(model, nst, X[i], device)
                got[int(i)] = (int(prob.argmax()), prob,
                               a if want_attn else None)
        per_rep[rep] = got
    if missing:
        print(f"  ({missing} checkpoint(s) absent for {config} in {tag})")
    return per_rep


def pick_density_col(feat_names):
    for c in DENSITY_CANDIDATES:
        if c in feat_names:
            return feat_names.index(c), c
    return None, None


def partial_spearman(u, v, covar):
    """Spearman correlation of u and v after linearly removing covar, all on
    ranks. This is the number that matters for attention claims: raw agreement
    is inflated by both maps tracking tile density."""
    if len(u) < 8:
        return np.nan
    ru, rv, rc = (stats.rankdata(z).astype(float) for z in (u, v, covar))
    if np.std(rc) < 1e-9:
        return stats.spearmanr(u, v).statistic
    A = np.column_stack([np.ones_like(rc), rc])
    eu = ru - A @ np.linalg.lstsq(A, ru, rcond=None)[0]
    ev = rv - A @ np.linalg.lstsq(A, rv, rcond=None)[0]
    if np.std(eu) < 1e-9 or np.std(ev) < 1e-9:
        return np.nan
    return float(np.corrcoef(eu, ev)[0, 1])


def as_single_map(a):
    """(N,) stays; (N, heads) -> mean over heads."""
    return a if a is None or a.ndim == 1 else a.mean(1)


def mode_confusion(args, device):
    """Per-class recall, one arm vs another. This is the 'which classes did
    the morphometrics actually fix' question."""
    print("=" * 78)
    print("CONFUSION DIFF -- which classes did the nuclear features fix?")
    print("=" * 78)

    arms = [("graph only (39 feat)", args.base_tag, args.base_config, False),
            ("+ nuclear (79 feat)", args.tag, args.config, True)]
    recalls, mats = {}, {}
    for label, tag, cfg, nuc in arms:
        X, y, ids, fn = bags_for(args.root, tag, nuc)
        per_rep = collect(args.root, tag, cfg, X, y, args.repeats, device)
        ty, tp = [], []
        for rep, got in per_rep.items():
            for i, (pred, _, _) in got.items():
                ty.append(y[i])
                tp.append(pred)
        if not ty:
            print(f"  no checkpoints found for {label} ({tag}/{cfg}) -- skipped")
            continue
        ty, tp = np.array(ty), np.array(tp)
        cm = confusion_matrix(ty, tp, labels=range(len(CLASSES)))
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        mats[label] = cmn
        recalls[label] = np.diag(cmn)
        print(f"\n--- {label}  [{tag} / {cfg}]  "
              f"balanced_acc {balanced_accuracy_score(ty, tp):.4f}, "
              f"n={len(ty)} ({args.repeats} repeats pooled) ---")
        print("  " + pd.DataFrame(cmn.round(3), index=CLASSES, columns=CLASSES)
              .to_string().replace("\n", "\n  "))

    if len(recalls) == 2:
        a, b = [r for r in recalls]
        d = recalls[b] - recalls[a]
        print(f"\n--- PER-CLASS RECALL DELTA ({b} minus {a}) ---")
        t = pd.DataFrame({a: recalls[a].round(3), b: recalls[b].round(3),
                          "delta": d.round(3)}, index=CLASSES)
        print("  " + t.to_string().replace("\n", "\n  "))
        gain = [CLASSES[i] for i in np.argsort(-d)[:3] if d[i] > 0.02]
        print(f"\n  biggest gains: {gain}")
        nuclear_grade = {"HGSC", "LGSC", "EC"}
        got = set(gain) & nuclear_grade
        if got:
            print(f"  {sorted(got)} are the nuclear-grade classes -- gains here")
            print("  are the predicted mechanism (HGSC pleomorphic vs EC/LGSC")
            print("  uniform), so the accuracy story and the biology agree.")
        else:
            print("  the gains are NOT on the nuclear-grade classes. The")
            print("  morphometrics are helping via some other route -- do not")
            print("  claim the nuclear-atypia mechanism without more evidence.")


def mode_attention(args, device):
    """Does attention actually sit on the atypical-nucleus tiles?"""
    print("\n" + "=" * 78)
    print("ATTENTION vs FEATURES -- what does the model actually look at?")
    print("=" * 78)
    X, y, ids, fn = bags_for(args.root, args.tag, True)
    per_rep = collect(args.root, args.tag, args.config, X, y, 1, device,
                      want_attn=True)
    got = per_rep.get(0, {})
    if not got or all(v[2] is None for v in got.values()):
        print("  no attention maps (mean/max pool config?) -- nothing to do")
        return
    didx, dname = pick_density_col(fn)
    print(f"  residualising against '{dname}' (column {didx})")
    print(f"  slides: {len(got)}   config: {args.config}")

    groups = resolve_groups(fn)
    # `epithelial_frac` and `connective_frac` exist in BOTH the graph block
    # (KDE-derived) and the nuclear block (raw nucleus counts) -- different
    # quantities, identical names. Disambiguate by group before any groupby,
    # or the two get pooled into one meaningless row.
    col2grp = {k: g for g, idx in groups.items() for k in idx}
    dup = {n for n in fn if fn.count(n) > 1}
    disp = [f"{n} [{col2grp.get(j, 'other')}]" if n in dup else n
            for j, n in enumerate(fn)]
    if dup:
        print(f"  note: {len(dup)} feature name(s) appear in both blocks "
              f"{sorted(dup)} -- suffixed by group below")

    rows = []
    for i, (pred, _, a) in got.items():
        a1 = as_single_map(a)
        if a1 is None or len(a1) < 8:
            continue
        bag = X[i].numpy()
        dens = bag[:, didx] if didx is not None else np.zeros(len(a1))
        for j, name in enumerate(disp):
            col = bag[:, j]
            if np.std(col) < 1e-9:
                continue
            rows.append(dict(slide=ids[i], label=CLASSES[y[i]], feature=name,
                             group=col2grp.get(j, "other"),
                             raw=stats.spearmanr(a1, col).statistic,
                             resid=partial_spearman(a1, col, dens)))
    df = pd.DataFrame(rows)
    if df.empty:
        print("  nothing computable")
        return

    agg = (df.groupby(["group", "feature"])["resid"]
             .agg(["median", "count"]).reset_index()
             .sort_values("median", key=abs, ascending=False))
    print("\n--- top 20 features by |median density-residualised correlation")
    print("    with pooling attention| (across slides) ---")
    print("  " + agg.head(20).round(3).to_string(index=False)
          .replace("\n", "\n  "))

    print("\n--- by feature group (median over features x slides) ---")
    gg = df.groupby("group")["resid"].agg(["median", "count"]).round(3)
    print("  " + gg.to_string().replace("\n", "\n  "))

    key = [f for f in ("neo_area_um2_cv", "neo_area_p90_p10_ratio",
                       "neo_ecc_std", "neo_solidity_mean", "neo_area_um2_mean")
           if f in set(df.feature)]
    if key:
        print("\n--- pleomorphism features, per class (median resid corr) ---")
        piv = (df[df.feature.isin(key)]
               .pivot_table(index="feature", columns="label", values="resid",
                            aggfunc="median").round(3))
        print("  " + piv.to_string().replace("\n", "\n  "))
        print("\n  A positive value means attention concentrates on tiles with")
        print("  MORE variable / more atypical nuclei, beyond what tile density")
        print("  alone explains. HGSC is where that should be strongest if the")
        print("  nuclear-atypia story is real.")

    out = Path(args.root) / RUN_DIR_NAME / OUT_SUBFOLDER / \
        f"attention_feature_corr_{args.tag}_{args.config}.csv"
    df.to_csv(out, index=False)
    print(f"\n  per-slide correlations -> {out}")


def mode_stability(args, device):
    """Do independently-seeded models agree, on the class and on the map?"""
    print("\n" + "=" * 78)
    print("SEED STABILITY -- is this model stable enough for single-slide figures?")
    print("=" * 78)
    X, y, ids, fn = bags_for(args.root, args.tag, True)
    per_rep = collect(args.root, args.tag, args.config, X, y, args.repeats,
                      device, want_attn=True)
    reps = [r for r in per_rep if per_rep[r]]
    if len(reps) < 2:
        print("  need >=2 repeats of checkpoints -- skipped")
        return
    didx, dname = pick_density_col(fn)

    common = set(per_rep[reps[0]])
    for r in reps[1:]:
        common &= set(per_rep[r])
    common = sorted(common)
    print(f"  {len(common)} slides scored out-of-fold in all {len(reps)} repeats")

    agree, raw, resid = [], [], []
    for i in common:
        preds = [per_rep[r][i][0] for r in reps]
        agree.append(len(set(preds)) == 1)
        maps = [as_single_map(per_rep[r][i][2]) for r in reps]
        if any(m is None for m in maps):
            continue
        dens = X[i].numpy()[:, didx] if didx is not None else np.zeros(len(maps[0]))
        for k in range(len(maps)):
            for l in range(k + 1, len(maps)):
                raw.append(stats.spearmanr(maps[k], maps[l]).statistic)
                resid.append(partial_spearman(maps[k], maps[l], dens))
    raw = np.array([v for v in raw if np.isfinite(v)])
    resid = np.array([v for v in resid if np.isfinite(v)])

    acc = float(np.mean(agree))
    print(f"\n  all {len(reps)} seeds agree on the predicted CLASS: {acc*100:.1f}% "
          f"of slides")
    print(f"    (chance for 3 independent 5-way guesses is ~4-20%; the v5 GAT")
    print(f"     models managed 54.4%)")
    if len(raw):
        print(f"\n  attention-map agreement, seed pair to seed pair (Spearman):")
        print(f"    raw               median {np.median(raw):+.3f}  "
              f"q25 {np.percentile(raw, 25):+.3f}")
        print(f"    density-residual  median {np.median(resid):+.3f}  "
              f"q25 {np.percentile(resid, 25):+.3f}")
        print("\n  The residualised number is the one that matters -- raw")
        print("  agreement is inflated by both maps tracking tile density.")
        q25 = np.percentile(resid, 25)
        print("\n  VERDICT -- the two axes moved differently, report both:")
        if q25 >= 0.40:
            print(f"    MAP stability: q25 {q25:+.3f} clears the 0.40 bar. The v5")
            print("    GAT sat at q25 +0.274, which was judged too low to")
            print("    defend. Where the attention lands is now reproducible")
            print("    across seeds -- single-slide overlays are defensible IF")
            print("    the seed-agreement number is reported alongside them.")
        else:
            print(f"    MAP stability: q25 {q25:+.3f} is below 0.40 -- where the")
            print("    attention lands still moves too much between seeds.")
        if acc >= 0.70:
            print(f"    CLASS stability: {acc*100:.1f}% is solid.")
        else:
            print(f"    CLASS stability: {acc*100:.1f}% is better than the GAT's")
            print("    54.4% but still means ~1 slide in 3 changes its predicted")
            print("    subtype with the seed. Any SINGLE-SLIDE claim inherits")
            print("    that, regardless of how stable the map is -- so pick")
            print("    figure slides where the seeds agree, and say so.")
        print("    Aggregate evidence (deletion test, the correlations above)")
        print("    is unaffected either way -- it averages over this variance.")


def mode_ensemble(args, device):
    """Average the per-slide probabilities of the independently-seeded models.

    Each repeat scores every slide OUT OF FOLD (the model that predicts slide
    i in repeat r never saw slide i during training in repeat r), so averaging
    across repeats is a legitimate ensemble evaluation, not leakage.

    Two reasons to expect a gain here specifically: the seeds only agree on
    the predicted class 63.4% of the time, and disagreement of that size is
    exactly what averaging fixes; and it attacks the stability problem that
    currently blocks single-slide figures, rather than trading against it.
    """
    print("\n" + "=" * 78)
    print("SEED ENSEMBLE -- does averaging the 3 seeds beat any single seed?")
    print("=" * 78)
    X, y, ids, fn = bags_for(args.root, args.tag, args.nuclear)
    per_rep = collect(args.root, args.tag, args.config, X, y, args.repeats,
                      device)
    reps = [r for r in per_rep if per_rep[r]]
    if len(reps) < 2:
        print("  need >=2 repeats of checkpoints -- skipped")
        return
    common = sorted(set.intersection(*[set(per_rep[r]) for r in reps]))
    yt = np.array([y[i] for i in common])

    singles = []
    for r in reps:
        p = np.array([per_rep[r][i][0] for i in common])
        singles.append(balanced_accuracy_score(yt, p))
        print(f"  repeat {r} alone            : {singles[-1]:.4f}")
    print(f"  mean of single seeds       : {np.mean(singles):.4f} "
          f"(sd {np.std(singles, ddof=1):.4f})")
    print(f"  best single seed           : {max(singles):.4f}")

    P = np.stack([np.stack([per_rep[r][i][1] for i in common]) for r in reps])
    for how, pred in (("mean prob", P.mean(0).argmax(1)),
                      ("majority vote",
                       stats.mode(P.argmax(2), axis=0, keepdims=False).mode)):
        b = balanced_accuracy_score(yt, pred)
        print(f"  ENSEMBLE ({how:13s})  : {b:.4f}  "
              f"({b - np.mean(singles):+.4f} vs mean single seed)")

    pred = P.mean(0).argmax(1)
    cm = confusion_matrix(yt, pred, labels=range(len(CLASSES)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    print(f"\n--- ENSEMBLE CONFUSION (mean prob, n={len(common)}) ---")
    print("  " + pd.DataFrame(cmn.round(3), index=CLASSES, columns=CLASSES)
          .to_string().replace("\n", "\n  "))
    print("\n  Per-class recall: " + ", ".join(
        f"{c} {v:.3f}" for c, v in zip(CLASSES, np.diag(cmn))))

    # Averaging k seeds, over many random subsets of size k, so the curve is
    # not an artefact of which particular seeds happened to be picked.
    if len(reps) >= 3:
        print(f"\n--- ENSEMBLE SIZE CURVE (mean over {ENS_SUBSETS} random "
              f"subsets per k) ---")
        rs = np.random.RandomState(0)
        prev = None
        for k in range(1, len(reps) + 1):
            accs = []
            for _ in range(1 if k == len(reps) else ENS_SUBSETS):
                sel = rs.choice(len(reps), k, replace=False)
                accs.append(balanced_accuracy_score(
                    yt, P[sel].mean(0).argmax(1)))
            m, s = float(np.mean(accs)), float(np.std(accs))
            gain = "" if prev is None else f"   (+{m - prev:.4f} over k-1)"
            print(f"    k={k:2d} seeds : {m:.4f}  (sd {s:.4f}){gain}")
            prev = m
        print("\n  Read the marginal column, not the level: once the k-to-k+1")
        print("  gain drops below the fold-to-fold noise there is no reason to")
        print("  train more seeds, and each one costs a full 5-fold run.")

    print("\n  NOTE: every repeat scores every slide out-of-fold, so this is a")
    print("  fair number -- but it costs k-times inference and the paper must")
    print("  say the headline model is an ensemble if this is the one reported.")


def mode_attnensemble(args, device):
    """Does averaging the attention maps across seeds make them figure-grade?

    Task 3b measured seed-PAIR agreement of single-model attention maps
    (density-residualised median +0.688, q25 +0.562) and concluded single-slide
    overlays were defensible only on slides where the seeds happened to agree.
    That measured the wrong object if the figure is going to show an ENSEMBLE
    map -- averaging k maps should be markedly more reproducible than any one.

    The test is split-half reliability: average the maps from one disjoint half
    of the seeds, average the other half, and correlate the two ensembles.
    Reported against the single-seed pair agreement on exactly the same slides,
    which is the honest comparison -- and against ensemble size, so "how many
    seeds do I need for a defensible figure" has a number rather than a guess.
    """
    print("\n" + "=" * 78)
    print("ATTENTION-MAP ENSEMBLING -- are averaged maps figure-grade?")
    print("=" * 78)
    X, y, ids, fn = bags_for(args.root, args.tag, args.nuclear,
                             drop_groups=args.drop_groups)
    per_rep = collect(args.root, args.tag, args.config, X, y, args.repeats,
                      device, want_attn=True)
    reps = [r for r in per_rep if per_rep[r]]
    if len(reps) < 4:
        print(f"  need >=4 repeats to split into halves (have {len(reps)})")
        print("  -- run train_abmil.py with a larger --repeats first")
        return
    didx, dname = pick_density_col(fn)
    common = sorted(set.intersection(*[set(per_rep[r]) for r in reps]))
    print(f"  {len(common)} slides scored out-of-fold in all {len(reps)} repeats")
    print(f"  residualising against '{dname}'")

    rs = np.random.RandomState(0)
    single, half = [], []
    ens_maps, ens_rows = {}, []
    for i in common:
        maps = [as_single_map(per_rep[r][i][2]) for r in reps]
        if any(m is None for m in maps):
            continue
        M = np.stack(maps)
        dens = X[i].numpy()[:, didx] if didx is not None else np.zeros(M.shape[1])
        for k in range(len(reps)):
            for l in range(k + 1, len(reps)):
                v = partial_spearman(M[k], M[l], dens)
                if np.isfinite(v):
                    single.append(v)
        for _ in range(ENS_SUBSETS):
            perm = rs.permutation(len(reps))
            h = len(reps) // 2
            v = partial_spearman(M[perm[:h]].mean(0), M[perm[h:2 * h]].mean(0),
                                 dens)
            if np.isfinite(v):
                half.append(v)
        ens_maps[ids[i]] = M.mean(0).astype(np.float32)
        w = M.mean(0)
        srt = np.sort(w)[::-1]
        ens_rows.append(dict(slide=ids[i], label=CLASSES[y[i]], n_tiles=len(w),
                             top10_share=float(srt[:max(1, len(w) // 10)].sum()),
                             seed_sd=float(M.std(0).mean())))

    single, half = np.array(single), np.array(half)
    if not len(half):
        print("  nothing computable")
        return
    print(f"\n  density-residualised agreement (Spearman):")
    print(f"    single seed vs single seed   median {np.median(single):+.3f}  "
          f"q25 {np.percentile(single, 25):+.3f}  n={len(single)}")
    print(f"    half-ensemble vs half-ens.   median {np.median(half):+.3f}  "
          f"q25 {np.percentile(half, 25):+.3f}  n={len(half)}")
    print(f"    improvement in q25           "
          f"{np.percentile(half, 25) - np.percentile(single, 25):+.3f}")

    q25 = np.percentile(half, 25)
    print("\n  VERDICT")
    if q25 >= 0.40:
        print(f"    q25 {q25:+.3f} clears the pre-set 0.40 bar. An")
        print("    ENSEMBLED attention map is reproducible enough to put in a")
        print("    figure. Note this licenses the ensemble map specifically --")
        print("    a single-seed overlay is still the weaker claim, so figures")
        print("    should be built from the averaged map and said to be.")
    else:
        print(f"    q25 {q25:+.3f} is still below 0.40. Averaging helps but not")
        print("    enough; keep the paper on aggregate evidence.")
    print(f"\n    Each half-ensemble averages {len(reps)//2} seeds, so the full")
    print(f"    {len(reps)}-seed map is MORE stable than this number -- it is a")
    print("    lower bound on what the reported figure would show.")

    odir = Path(args.root) / RUN_DIR_NAME / OUT_SUBFOLDER
    out = odir / f"attn_ensemble_{args.tag}_{args.config}.npz"
    np.savez_compressed(out, **ens_maps)
    pd.DataFrame(ens_rows).to_csv(
        odir / f"attn_ensemble_{args.tag}_{args.config}.csv", index=False)
    print(f"\n  per-slide ensembled maps -> {out.name}")
    print(f"  (keyed by image_id, aligned to the graph's node order, so they")
    print(f"   drop straight into a spatial overlay)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["all", "confusion", "attention", "stability",
                             "ensemble", "attnensemble"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag", default="v5nuc", help="arm with nuclear features")
    ap.add_argument("--config", default="abmil_h128_lr1")
    ap.add_argument("--base-tag", default="v5", help="arm without them")
    ap.add_argument("--base-config", default="abmil_h128_L2")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--exclude-prefix", nargs="+", default=None,
                    help="must match what the arm was TRAINED with, e.g. "
                         "--exclude-prefix dead_ for the artefact-free arms. "
                         "Group drops alone cannot reproduce a by-name column "
                         "drop, so without this the checkpoint guard refuses "
                         "the run rather than misattributing silently.")
    ap.add_argument("--drop-groups", nargs="+", default=None,
                    help="must match what the arm was TRAINED with, e.g. "
                         "--drop-groups graph for the nuclear-only arms; a "
                         "mismatch is caught and refused, not silently wrong")
    ap.add_argument("--no-nuclear", dest="nuclear", action="store_false",
                    help="the arm under --tag has no nuclear features")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}\n")
    if args.what in ("all", "confusion"):
        mode_confusion(args, device)
    if args.what in ("all", "attention"):
        mode_attention(args, device)
    if args.what in ("all", "stability"):
        mode_stability(args, device)
    if args.what in ("all", "ensemble"):
        mode_ensemble(args, device)
    if args.what in ("all", "attnensemble"):
        mode_attnensemble(args, device)


if __name__ == "__main__":
    main()

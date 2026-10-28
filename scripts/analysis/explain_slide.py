# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Turn one slide's prediction into a sentence in diagnostic vocabulary.

    python scripts/analysis/explain_slide.py --slide 10077
    python scripts/analysis/explain_slide.py --n 8                 # 8 random slides
    python scripts/analysis/explain_slide.py --all --out explanations.csv

WHAT THIS PRODUCES
==================
    Slide 10077 -- predicted EC (confidence 0.71, all 3 seeds agree)
      Driven by:
        1. variability in tumour-nucleus size (pleomorphism)  [LOW, -1.4 SD]
           removing it costs 0.12 of the predicted probability
        2. ...
      Attention concentrated on 8.4% of tiles (top decile holds 41% of weight).

HOW THE ATTRIBUTION IS COMPUTED -- and what it is not
=====================================================
Two separate measurements, deliberately kept apart:

  WHICH TILES   the gated-attention pooling weight. This is the quantity the
                v5 deletion test validated causally (deleting the top 5% of
                tiles costs 5.3pp of predicted probability vs 0.37pp for a
                density-matched random control, 14x). It is a real importance
                measure, not a by-product.

  WHICH FEATURES  occlusion. Each feature is replaced, across every tile, by
                its corpus-wide median, and the drop in the predicted class
                probability is measured. That is a counterfactual on the
                model's actual decision -- the same logic as the deletion
                test, moved from the tile axis to the feature axis.

Gradient x input is deliberately NOT used: it was tested on this project and
confirmed broken (masking the top-k features by it *increases* confidence).

RUN THIS ON THE NUCLEAR-ONLY ARM. This matters:

    python scripts/analysis/explain_slide.py --tag v5nuc_nuconly --drop-groups graph --n 5

Occlusion measures what THIS TRAINED MODEL leans on. The retraining ablation
(Task 3b) measures whether the INFORMATION is necessary. They disagree here,
and the disagreement is instructive rather than a bug: on the full 79-feature
model the KDE/streamline group dominates every occlusion attribution (+0.57,
+0.68 on the slides first tested), yet dropping that entire group and
retraining costs nothing at all (0.6732 vs 0.6692, paired p=0.84). Redundant
correlated inputs get arbitrary shares of a trained model's reliance, so
occluding them looks devastating while the information survives elsewhere.

Explaining the full model would therefore tell a pathologist their CC call
"rests on epithelial streamline throughput" -- attributing the decision to a
feature block we have proven carries no unique information. The nuclear-only
arm has the same accuracy, no graph features to misattribute to, and every
column in it is a named pathology quantity. That is the model to explain.

THREE HONESTY CONSTRAINTS, enforced in the output
=================================================
1. This explains the MODEL, not the patient. "The model's EC call rests on
   low pleomorphism" is supportable. "This tumour is EC because pleomorphism
   is low" is not, and the wording never claims it.
2. Nuclear features are strongly correlated (neo_area_mean / _p90 / _std / _cv
   move together). Occluding one lets the others compensate, which UNDER-states
   its importance. Group-level occlusion is therefore reported alongside the
   per-feature numbers and should be trusted over them where they disagree.
3. The seeds disagree on the predicted class for ~1 slide in 3 (63.4% all-3
   agreement). Every explanation states the seed agreement, and slides where
   the seeds disagree are flagged -- an explanation of an unstable prediction
   is an explanation of noise.
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

from sklearn.model_selection import StratifiedKFold

from analyse_abmil_nuclear import (apply_norm, arm_paths, bags_for,
                                   rebuild_model)
from train_abmil import CLASSES, resolve_groups
import corpus_paths

# feature name -> how a pathologist would say it
GLOSS = {
    "neo_area_um2_cv": "variability in tumour-nucleus size (pleomorphism)",
    "neo_area_p90_p10_ratio": "spread between largest and smallest tumour nuclei "
                              "(pleomorphism)",
    "neo_area_um2_std": "spread of tumour-nucleus size",
    "neo_area_um2_mean": "average tumour-nucleus size",
    "neo_area_um2_p90": "size of the largest tumour nuclei",
    "neo_area_um2_p10": "size of the smallest tumour nuclei",
    "neo_equivdiam_um_mean": "average tumour-nucleus diameter",
    "neo_ecc_mean": "tumour-nucleus elongation",
    "neo_ecc_std": "variability in tumour-nucleus elongation",
    "neo_axisratio_mean": "tumour-nucleus axis ratio",
    "neo_circularity_mean": "tumour-nucleus circularity",
    "neo_solidity_mean": "tumour-nucleus contour smoothness (low = irregular)",
    "neo_solidity_std": "variability in tumour-nucleus contour irregularity",
    "all_area_um2_mean": "average nucleus size, all cell types",
    "all_area_um2_cv": "variability in nucleus size, all cell types",
    "all_ecc_mean": "nucleus elongation, all cell types",
    "all_solidity_mean": "nucleus contour smoothness, all cell types",
    "nuc_count": "number of nuclei per tile",
    "nuc_per_mm2": "nuclear density",
    "neo_count": "number of tumour nuclei",
    "neo_frac": "proportion of nuclei that are tumour",
    "inflam_count": "number of inflammatory cells",
    "inflam_frac": "proportion of inflammatory cells",
    "connective_count": "number of stromal nuclei",
    "connective_frac": "proportion of stromal nuclei",
    "epithelial_count": "number of benign epithelial nuclei",
    "epithelial_frac": "proportion of benign epithelial nuclei",
    "dead_count": "number of necrotic nuclei",
    "dead_frac": "proportion of necrotic nuclei",
    "nnd_all_um_mean": "spacing between neighbouring nuclei (crowding)",
    "nnd_all_um_cv": "variability in nuclear spacing",
    "nnd_neo_um_mean": "spacing between tumour nuclei (crowding)",
    "til_inflam_nn_neo_um_mean": "distance from lymphocytes to nearest tumour nucleus",
    "til_frac_neo_infiltrated": "fraction of tumour nuclei with lymphocytes "
                                "within 30um (TIL infiltration)",
    "til_inflam_per_neo": "lymphocytes per tumour nucleus",
    "morph_measured": "availability of morphometry in the tile",
    "neo_measured": "availability of tumour morphometry in the tile",
    "til_measured": "availability of TIL measurement in the tile",
    "nnd_all_measured": "availability of nuclear-spacing measurement",
    "nnd_neo_measured": "availability of tumour-spacing measurement",
    "dens_all_mean": "overall cell density (KDE)",
    "dens_all_max": "peak cell density (KDE)",
    "dens_all_std": "variation in cell density (KDE)",
    "tile_fill": "fraction of the tile containing tissue",
}
GROUP_GLOSS = {
    "nuc_shape": "nuclear morphometry (size, shape, pleomorphism)",
    "nuc_density": "cellular composition and density",
    "nuc_spatial": "nuclear spacing and immune infiltration",
    "nuc_flags": "measurement availability",
    "graph": "KDE channel / streamline features",
}


def gloss(name):
    base = name.split(" [")[0]
    if base in GLOSS:
        return GLOSS[base]
    for chan in ("neoplastic", "epithelial", "connective", "inflammatory",
                 "interface"):
        if base.startswith(chan + "_"):
            what = base[len(chan) + 1:]
            kind = {"mean": "average density", "max": "peak density",
                    "frac": "share of the tile", "coherence_s0": "flow coherence",
                    "coherence_s1": "flow coherence (fine scale)",
                    "log_pass": "streamline throughput",
                    "log_endpoint": "streamline endpoints"}.get(what, what)
            return f"{chan} {kind} (KDE/streamline)"
    return base


def models_for(root, tag, config, idx, y, repeats, device, feat_names=None):
    """Every checkpoint that scored this slide OUT OF FOLD."""
    _, ckdir = arm_paths(root, tag)
    out = []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        for fold, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            if idx in set(te.tolist()):
                p = ckdir / f"{config}_r{rep}f{fold}.pt"
                if p.exists():
                    ck = torch.load(p, map_location=device, weights_only=False)
                    if feat_names is not None and \
                            list(ck["feat_names"]) != list(feat_names):
                        raise SystemExit(
                            f"FEATURE MISMATCH: {p.name} was trained on "
                            f"{len(ck['feat_names'])} features but the bags "
                            f"loaded here have {len(feat_names)}. Pass the same "
                            f"--drop-groups / --drop-flags the arm was trained "
                            f"with, or the attribution is meaningless.")
                    m, _ = rebuild_model(ck, device)
                    n = ck["normaliser"]
                    out.append((m, {"log_idx": n["log_idx"],
                                    "mean": n["mean"].to(device),
                                    "std": n["std"].to(device)}))
                break
    return out


@torch.no_grad()
def probs(model, nst, bag, device):
    x = apply_norm(bag.to(device), nst).unsqueeze(0)
    msk = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    out, a = model(x, msk, return_attn=True)
    p = torch.softmax(out, 1)[0].cpu().numpy()
    if a is not None:
        a = a[0].cpu().numpy()
        if a.ndim > 1:
            a = a.mean(1)
    return p, a


@torch.no_grad()
def occlude(models, bag, cls, cols, med, device):
    """Mean drop in P(cls) when `cols` are replaced by the corpus median,
    averaged over the ensemble members."""
    drops = []
    for m, nst in models:
        base = probs(m, nst, bag, device)[0][cls]
        b2 = bag.clone()
        b2[:, cols] = med[cols]
        drops.append(base - probs(m, nst, b2, device)[0][cls])
    return float(np.mean(drops))


@torch.no_grad()
def additive_tiles(models, bag, cls, device, topk=5):
    """Exact per-tile contribution to the predicted class, for AdditiveMIL.

    Every other explanation in this file is an ATTRIBUTION -- occlusion asks
    what happens when something is removed, and the answer depends on what the
    rest of the model can compensate with. AdditiveMIL needs no attribution at
    all: its bag logit IS the sum of per-tile per-class terms, so a tile's
    contribution is a fact about the computation rather than an estimate of
    one. `contributions.sum(1) == logits` holds to float precision.

    Returns None for any other architecture, so callers can fall back.
    """
    from train_abmil import AdditiveMIL
    rows = []
    for m, nst in models:
        if not isinstance(m, AdditiveMIL):
            return None
        x = apply_norm(bag.to(device), nst).unsqueeze(0)
        msk = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
        out, _, contrib = m(x, msk, return_contrib=True)
        rows.append(contrib[0, :, cls].cpu().numpy())
    c = np.mean(rows, 0)
    tot = float(c.sum())
    order = np.argsort(-c)[:topk]
    return dict(logit=tot, n_tiles=len(c),
                top=[(int(j), float(c[j]),
                      float(c[j] / tot) if abs(tot) > 1e-8 else np.nan)
                     for j in order],
                share_top10=float(c[np.argsort(-c)[:max(1, len(c) // 10)]].sum()
                                  / tot) if abs(tot) > 1e-8 else np.nan)


def explain(idx, X, y, ids, fn, models, med, corpus_mean, corpus_sd, groups,
            device, topk=4):
    bag = X[idx]
    P, A = [], []
    for m, nst in models:
        p, a = probs(m, nst, bag, device)
        P.append(p)
        if a is not None:
            A.append(a)
    P = np.stack(P)
    pm = P.mean(0)
    cls = int(pm.argmax())
    seed_preds = P.argmax(1)
    agree = len(set(seed_preds.tolist())) == 1

    att = np.mean(A, 0) if A else None
    if att is not None and att.sum() > 0:
        att = att / att.sum()
        order = np.argsort(-att)
        top10 = max(1, int(round(0.10 * len(att))))
        conc = float(att[order[:top10]].sum())
    else:
        conc = np.nan

    # attention-weighted feature profile, in corpus SD units
    w = att if att is not None else np.full(len(bag), 1 / len(bag))
    prof = (bag.numpy() * w[:, None]).sum(0)
    z = (prof - corpus_mean) / corpus_sd

    grp_drop = {g: occlude(models, bag, cls, idx_, med, device)
                for g, idx_ in groups.items() if idx_}
    feat_drop = np.array([occlude(models, bag, cls, [j], med, device)
                          for j in range(len(fn))])

    top = np.argsort(-feat_drop)[:topk]
    return dict(slide=ids[idx], truth=CLASSES[y[idx]], pred=CLASSES[cls],
                conf=float(pm[cls]), seeds_agree=agree,
                n_seeds=len(models), attn_top10=conc,
                group_drop=grp_drop,
                additive=additive_tiles(models, bag, cls, device),
                top_feats=[(fn[j], float(feat_drop[j]), float(z[j]))
                           for j in top])


def render(e):
    L = []
    ok = "all %d seeds agree" % e["n_seeds"] if e["seeds_agree"] \
        else "SEEDS DISAGREE -- treat with caution"
    L.append(f"Slide {e['slide']}  (true {e['truth']})")
    L.append(f"  predicted {e['pred']}  confidence {e['conf']:.2f}   [{ok}]")
    if np.isfinite(e["attn_top10"]):
        L.append(f"  attention concentration: top 10% of tiles hold "
                 f"{e['attn_top10']*100:.0f}% of the pooling weight "
                 f"(uniform would be 10%)")
    L.append("  the call rests on:")
    for i, (name, drop, z) in enumerate(e["top_feats"], 1):
        if drop <= 0:
            continue
        lvl = "elevated" if z > 0.5 else ("reduced" if z < -0.5 else "typical")
        L.append(f"    {i}. {gloss(name)}")
        L.append(f"       {lvl} ({z:+.1f} SD vs corpus) -- removing it costs "
                 f"{drop:.3f} of the predicted probability")
    gd = sorted(e["group_drop"].items(), key=lambda kv: -kv[1])
    L.append("  by feature group (more reliable -- these features are "
             "correlated):")
    for g, d in gd:
        if abs(d) < 1e-4:
            continue
        L.append(f"    {GROUP_GLOSS.get(g, g):48s} {d:+.3f}")
    add = e.get("additive")
    if add:
        L.append(f"  which TILES carried the call (exact, not attributed --")
        L.append(f"  these contributions sum to the {e['pred']} logit "
                 f"{add['logit']:+.2f}):")
        for j, c, share in add["top"]:
            L.append(f"    tile {j:5d}  {c:+.3f}  ({share*100:4.1f}% of the logit)")
        L.append(f"    top 10% of tiles carry "
                 f"{add['share_top10']*100:.0f}% of it "
                 f"({add['n_tiles']} tiles total)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(corpus_paths.ROOT))
    ap.add_argument("--tag", default="v5nuc")
    ap.add_argument("--config", default="abmil_h128_lr1")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--slide", default=None, help="image_id to explain")
    ap.add_argument("--n", type=int, default=0, help="explain N random slides")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write a CSV of the numbers")
    ap.add_argument("--drop-groups", nargs="+", default=None,
                    help="must match what the arm under --tag was TRAINED "
                         "with, e.g. --tag v5nuc_nuconly --drop-groups graph")
    ap.add_argument("--chromatin-features", action="store_true",
                    help="load the chromatin block too. Required for the final "
                         "model (v5nucchromeos_clean_final); without it the "
                         "column count will not match the checkpoint and the "
                         "run aborts rather than misattributing.")
    ap.add_argument("--eosin-features", action="store_true",
                    help="load the eosin/cytoplasm block too. Same reason.")
    ap.add_argument("--exclude-prefix", nargs="+", default=None,
                    help="drop columns by NAME prefix, mirroring the arm's own "
                         "training flags. The clean arms were trained with "
                         "--exclude-prefix dead_ (the convicted staining "
                         "artefact), which is a by-name drop, not a group drop.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, ids, fn = bags_for(args.root, args.tag, True,
                             drop_groups=args.drop_groups,
                             exclude_prefix=args.exclude_prefix,
                             chromatin=args.chromatin_features,
                             eosin=args.eosin_features)
    groups = resolve_groups(fn)

    A = torch.cat(X, 0)
    med = A.median(0).values
    corpus_mean = A.mean(0).numpy()
    corpus_sd = A.std(0).clamp(min=1e-6).numpy()

    if args.slide:
        want = [ids.index(str(args.slide))] if str(args.slide) in ids else []
        if not want:
            raise SystemExit(f"slide {args.slide} not in the corpus")
    elif args.all:
        want = list(range(len(ids)))
    else:
        rng = np.random.default_rng(args.seed)
        want = sorted(rng.choice(len(ids), size=max(1, args.n or 5),
                                 replace=False).tolist())

    print(f"device={device}  explaining {len(want)} slide(s) "
          f"with {args.config} [{args.tag}]\n")
    print("NOTE: this describes what the MODEL used, not a diagnosis.\n")

    rows = []
    for k, i in enumerate(want, 1):
        models = models_for(args.root, args.tag, args.config, i, y,
                            args.repeats, device, feat_names=fn)
        if not models:
            continue
        e = explain(i, X, y, ids, fn, models, med, corpus_mean, corpus_sd,
                    groups, device)
        print(render(e))
        print()
        flat = {k2: v for k2, v in e.items()
                if k2 not in ("group_drop", "top_feats")}
        flat.update({f"grp_{g}": v for g, v in e["group_drop"].items()})
        for r, (nm, d, z) in enumerate(e["top_feats"], 1):
            flat[f"feat{r}"], flat[f"feat{r}_drop"], flat[f"feat{r}_z"] = nm, d, z
        rows.append(flat)
        if args.all and k % 25 == 0:
            print(f"  ... {k}/{len(want)}")

    if args.out and rows:
        p = Path(args.root) / "cellvit_run/graphs/gatv2_run" / args.out
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"-> {p}")


if __name__ == "__main__":
    main()

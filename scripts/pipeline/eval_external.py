# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Instrument 1 (representation substitution) on an EXTERNAL corpus, frozen.

WHAT THIS IS
============
The paper's headline is that a pathology foundation model beats explicit named
morphometry through an identical Additive MIL head on UBC-OCEAN. This script
asks whether that gap survives out of distribution, on a corpus the head has
never seen, WITHOUT retraining anything.

Every UBC-OCEAN checkpoint is used exactly as trained:

  * the model weights are loaded and run in eval mode -- no fine-tuning, no
    head re-fitting, no calibration;
  * the normalisation statistics come out of the CHECKPOINT, i.e. they are the
    mean/std fitted on that fold's UBC-OCEAN inner-training split. Nothing is
    re-fitted on the external corpus. Re-fitting z-scores on 80 slides is a
    quiet form of adaptation and is exactly what over-fit a previous attempt at
    this size;
  * feature columns are aligned to the checkpoint's own `feat_names`, in the
    checkpoint's order, and a mismatch is a hard failure rather than a silent
    reorder. This follows explain_slide.py's precedent -- a misaligned column
    set does not crash, it misattributes.

WHAT IS DELIBERATELY NOT HERE
=============================
Subspace deletion (Instrument 2) and the bridge regression (Instrument 3). Both
need a ridge map fitted on the corpus under test; at n=80 that is not a
measurement, it is an invitation to overfit. Out of scope by decision.

STATISTICS
==========
There are no (repeat, fold) cells to pair on here -- every checkpoint scores
every external slide -- so the paired t / Wilcoxon used on the primary corpus
does not apply. Instead:

  * point estimate = the mean over the 50 checkpoints of per-checkpoint
    balanced accuracy. This is the statistic the paper's headline numbers are
    (a mean over 50 models), so it is the like-for-like comparison.
  * the 50-checkpoint probability ENSEMBLE is also reported, which is the
    analogue of the paper's ensembled column.
  * confidence intervals come from resampling SLIDES with replacement
    (default 1000x). Slides are the sampling unit; the checkpoints are not, so
    each bootstrap replicate re-scores the same 50 models on the resampled
    slide set.
  * arm-vs-arm deltas use the SAME bootstrap resamples for both arms (common
    random numbers). The arms see an identical slide list, so the slides are
    paired even though the folds are not, and a paired interval is both valid
    and much tighter than differencing two independent ones.

A bootstrap replicate that loses a whole class is dropped from the balanced
accuracy interval and counted, because balanced accuracy is undefined for an
absent class. With a 10-slide class this is rare (~1e-5) but it is reported
rather than silently averaged over.

USAGE
=====
    $env:UBC_RUN_DIR = "bentaeb_run"
    python scripts/pipeline/eval_external.py --out ..\bentaeb_run\instrument1.json
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
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import corpus_paths
from train_abmil import (CLASSES, COLS, FM_PREFIX, append_derived_flags,
                         apply_group_filter, build_model, load_extra_block,
                         load_fm_block)

PRIMARY_ROOT = corpus_paths.ROOT
CKPT_ROOT = PRIMARY_ROOT / "cellvit_run" / "graphs" / "gatv2_run" / "abmil_ckpt"
CFGNAME = "addmil_h128_do5"

# The five arms of Instrument 1, exactly as trained on UBC-OCEAN.
#
# The clean interpretable recipe is taken from fm_subspace_split.py (INT_TAG /
# INT_DROP / INT_EXCL), which is the script that consumes this arm and is
# therefore the authority on how it was built. It yields the 49-column schema:
# post dead_ exclusion, post chrom_context exclusion, chromatin levels in,
# GLCM texture out.
ARMS = [
    dict(key="clean", nice="Named morphometry (49 cols)",
         ckpt_tag="v5nucchromeos_clean_final",
         nuclear=True, chromatin=True, eosin=True, fm=None,
         drop_groups=["graph", "chrom_texture", "eos_other", "chrom_context"],
         exclude_prefix=["dead_"]),
    dict(key="uni2h", nice="UNI2-h (ViT-H/14, 1536-d)",
         ckpt_tag="v5fmuni2h_fm_addmil", fm="uni2h"),
    dict(key="hoptimus0", nice="H-optimus-0 (ViT-g/14, 1536-d)",
         ckpt_tag="v5fmhoptimus0_fm_addmil", fm="hoptimus0"),
    dict(key="phikonv2", nice="Phikon-v2 (ViT-L/16, 1024-d)",
         ckpt_tag="v5fmphikonv2_fm_addmil", fm="phikonv2"),
    dict(key="hibouL", nice="Hibou-L (ViT-L/14, 1024-d)",
         ckpt_tag="v5fmhibouL_fm_addmil", fm="hibouL"),
]


def load_external_bags(arm, gdir):
    """Build one arm's bags for the external corpus.

    Mirrors train_abmil.load_bags but deliberately omits two of its steps:

      * no bag CACHE. The cache is keyed by tag, and a stale one silently
        scores the wrong bags. Inference over 80 slides is cheap; a wrong cache is not.
      * no drop_degenerate(). That sweep removes columns with zero variance ON
        THE CORPUS BEING LOADED. A column that happens to be constant across 80
        external slides but varied across 456 training slides would be dropped
        here and not in the checkpoint, so the two column sets would silently
        differ in length. Alignment to the checkpoint's feat_names does that job
        correctly instead.
    """
    nuc_dir = gdir / "nuclear_feats"
    chrom_dir = gdir / "chromatin_feats"
    eos_dir = gdir / "eosin_feats"
    fm_dir = gdir / "fm_feats"

    X, y, ids, feat_names = [], [], [], None
    skipped = Counter()
    for p in sorted(gdir.glob("*_graph.npz")):
        with np.load(p, allow_pickle=True) as z:
            label = str(z["label"][0])
            if label not in CLASSES:
                skipped["label not in CLASSES"] += 1
                continue
            iid = str(z["image_id"][0])
            F_ = z["node_features"].astype(np.float32)
            names = [str(v) for v in z["node_feature_names"]]
            # channel_valid is a MASK, not a value: a False channel means "not
            # measurable on this slide", which is not the same as zero.
            for cn, ok in zip([str(c) for c in z["channel_names"]],
                              np.asarray(z["channel_valid"]).astype(bool)):
                if not ok:
                    cols = [i for i, f in enumerate(names)
                            if f.startswith(cn + "_")]
                    F_[:, cols] = 0.0
            grid_rc_sum = int(np.asarray(z["node_grid_rc"]).sum())

        bad = None
        if arm.get("nuclear"):
            got, why = load_extra_block(nuc_dir / f"{iid}_nuclei.npz",
                                        "nuc_features", "nuc_feature_names",
                                        F_.shape[0], grid_rc_sum)
            if got is None:
                bad = f"nuclear {why}"
            else:
                NF, nn = append_derived_flags(*got)
                F_ = np.concatenate([F_, NF], axis=1)
                names = names + nn
        if bad is None and arm.get("chromatin"):
            got, why = load_extra_block(chrom_dir / f"{iid}_chromatin.npz",
                                        "chrom_features", "chrom_feature_names",
                                        F_.shape[0], grid_rc_sum)
            if got is None:
                bad = f"chromatin {why}"
            else:
                F_ = np.concatenate([F_, got[0]], axis=1)
                names = names + got[1]
        if bad is None and arm.get("eosin"):
            got, why = load_extra_block(eos_dir / f"{iid}_eosin.npz",
                                        "eos_features", "eos_feature_names",
                                        F_.shape[0], grid_rc_sum)
            if got is None:
                bad = f"eosin {why}"
            else:
                F_ = np.concatenate([F_, got[0]], axis=1)
                names = names + got[1]
        if bad is None and arm.get("fm"):
            got, why = load_fm_block(fm_dir / f"{iid}_{arm['fm']}.npz",
                                     F_.shape[0], grid_rc_sum)
            if got is None:
                bad = f"fm {why}"
            else:
                # fm_only: the encoder's view and nothing else, so any gap is
                # the representation alone
                F_, names = got[0], list(got[1])

        if bad is not None:
            skipped[bad] += 1
            continue

        if feat_names is None:
            feat_names = names
        elif names != feat_names:
            raise SystemExit(f"{iid}: feature names differ from the first slide")
        X.append(torch.from_numpy(F_))
        y.append(CLASSES.index(label))
        ids.append(iid)

    if not X:
        raise SystemExit(f"no usable bags for arm {arm['key']} in {gdir}")

    X, feat_names = apply_group_filter(X, feat_names,
                                       arm.get("drop_groups"), None)
    if arm.get("exclude_prefix"):
        keep = [i for i, f in enumerate(feat_names)
                if not f.startswith(tuple(arm["exclude_prefix"]))]
        print(f"  excluding {len(feat_names) - len(keep)} column(s) matching "
              f"{arm['exclude_prefix']}")
        X = [b[:, keep] for b in X]
        feat_names = [feat_names[i] for i in keep]

    sizes = np.array([b.shape[0] for b in X])
    print(f"  bags={len(X)}  features={len(feat_names)}  "
          f"tiles/bag median={np.median(sizes):.0f} min={sizes.min()} "
          f"max={sizes.max()}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    return X, np.asarray(y), ids, feat_names, dict(skipped)


def align_to_checkpoint(X, feat_names, ckpt_names):
    """Reorder/select external columns into the checkpoint's own order.

    Hard-fails on a missing name or an ambiguous duplicate. Both are silent
    misattribution risks, not recoverable conditions: the graph block and the
    nuclear block genuinely share names (`connective_frac`, `epithelial_frac`),
    which is why the duplicate check is not paranoia.
    """
    pos = {}
    dupes = set()
    for i, f in enumerate(feat_names):
        if f in pos:
            dupes.add(f)
        pos[f] = i
    need = [f for f in ckpt_names if f in dupes]
    if need:
        raise SystemExit(f"ambiguous duplicate feature name(s) {need} -- "
                         "cannot align to the checkpoint unambiguously")
    missing = [f for f in ckpt_names if f not in pos]
    if missing:
        raise SystemExit(
            f"{len(missing)} checkpoint feature(s) absent from the external "
            f"corpus, e.g. {missing[:8]}. Refusing to run: a partial column "
            f"set would be scored against weights trained on the full one.")
    idx = [pos[f] for f in ckpt_names]
    extra = len(feat_names) - len(idx)
    if extra:
        print(f"    dropping {extra} external column(s) not in the checkpoint")
    return [b[:, idx] for b in X]


def frozen_normalise(x, log_idx, mean, std):
    """Exactly train_abmil.Normaliser.__call__, with the checkpoint's stats.

    The presence-flag identity is already baked into mean=0/std=1 at fit time,
    so nothing extra is needed here -- and nothing is re-estimated.
    """
    x = x.clone()
    if log_idx:
        x[..., log_idx] = torch.log1p(x[..., log_idx].clamp(min=0))
    return (x - mean) / std


@torch.no_grad()
def score_checkpoint(ckpt_path, X, device):
    """One checkpoint -> (n_slides, n_classes) probabilities."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck["classes"] != CLASSES:
        raise SystemExit(f"{ckpt_path.name}: class order {ck['classes']} "
                         f"!= {CLASSES}")
    cfg = tuple(ck["config"][c] for c in COLS)
    model = build_model(cfg, ck["in_dim"], len(CLASSES)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    n = ck["normaliser"]
    log_idx = list(n["log_idx"])
    mean = n["mean"].to(device)
    std = n["std"].to(device)

    out = np.empty((len(X), len(CLASSES)), np.float64)
    for i, b in enumerate(X):
        if b.shape[1] != ck["in_dim"]:
            raise SystemExit(f"{ckpt_path.name}: bag has {b.shape[1]} columns, "
                             f"checkpoint expects {ck['in_dim']}")
        x = frozen_normalise(b.to(device).unsqueeze(0), log_idx, mean, std)
        m = torch.ones(1, b.shape[0], dtype=torch.bool, device=device)
        logits = model(x, m)
        out[i] = torch.softmax(logits.float(), -1).cpu().numpy()[0]
    return out, ck


def run_arm(arm, gdir, device, verbose=True):
    print(f"\n{'=' * 74}\nARM {arm['key']}  --  {arm['nice']}\n{'=' * 74}")
    X, y, ids, feat_names, skipped = load_external_bags(arm, gdir)

    cdir = CKPT_ROOT / arm["ckpt_tag"]
    ckpts = sorted(cdir.glob(f"{CFGNAME}_r*f*.pt"))
    if not ckpts:
        raise SystemExit(f"no {CFGNAME} checkpoints in {cdir}")
    print(f"  checkpoints: {len(ckpts)} from {arm['ckpt_tag']}")

    # align once against the first checkpoint, then assert every other agrees
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    Xa = align_to_checkpoint(X, feat_names, ck0["feat_names"])
    print(f"  aligned to {len(ck0['feat_names'])} checkpoint columns "
          f"(in_dim={ck0['in_dim']}, select={ck0.get('select')})")

    P = np.empty((len(ckpts), len(Xa), len(CLASSES)), np.float64)
    per_ckpt = []
    t0 = time.time()
    for k, cp in enumerate(ckpts):
        probs, ck = score_checkpoint(cp, Xa, device)
        if ck["feat_names"] != ck0["feat_names"]:
            raise SystemExit(f"{cp.name}: feat_names differ from {ckpts[0].name}")
        P[k] = probs
        bal = balanced_accuracy_score(y, probs.argmax(1))
        per_ckpt.append(dict(ckpt=cp.stem, bal=float(bal)))
        if verbose and (k + 1) % 10 == 0:
            print(f"    {k + 1}/{len(ckpts)} scored "
                  f"({time.time() - t0:.0f}s, running mean bal="
                  f"{np.mean([c['bal'] for c in per_ckpt]):.4f})")

    return dict(arm=arm, X=Xa, y=y, ids=ids, P=P, per_ckpt=per_ckpt,
                feat_names=ck0["feat_names"], skipped=skipped)


def bal_acc(y, pred, n_classes=len(CLASSES)):
    """Balanced accuracy, NaN when a class is absent from y (undefined)."""
    present = np.unique(y)
    if len(present) < n_classes:
        return np.nan
    return balanced_accuracy_score(y, pred)


def bootstrap(results, n_boot, seed):
    """Slide-resampling bootstrap, shared resamples across arms.

    Common random numbers matter: the arms score the SAME slides, so a delta
    computed within a replicate removes the slide-composition variance that
    dominates at n=80. Differencing two independently-bootstrapped intervals
    would badly overstate the uncertainty on the gap.
    """
    keys = [r["arm"]["key"] for r in results]
    n = len(results[0]["y"])
    for r in results:
        if len(r["y"]) != n or r["ids"] != results[0]["ids"]:
            raise SystemExit("arms do not share an identical slide list -- "
                             "a paired bootstrap would be invalid")
    y = results[0]["y"]

    rng = np.random.default_rng(seed)
    mean_boot = {k: [] for k in keys}
    ens_boot = {k: [] for k in keys}
    n_degenerate = 0

    # precompute per-checkpoint and ensemble predictions once
    pred_ck = {r["arm"]["key"]: r["P"].argmax(2) for r in results}    # (K, N)
    pred_en = {r["arm"]["key"]: r["P"].mean(0).argmax(1) for r in results}  # (N,)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < len(CLASSES):
            n_degenerate += 1
            continue
        for k in keys:
            pk = pred_ck[k][:, idx]
            mean_boot[k].append(float(np.mean(
                [balanced_accuracy_score(yb, pk[j]) for j in range(pk.shape[0])])))
            ens_boot[k].append(float(balanced_accuracy_score(yb, pred_en[k][idx])))

    return (mean_boot, ens_boot, n_degenerate,
            {k: np.asarray(v) for k, v in mean_boot.items()})


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def self_test(arm_key, device, tol=1e-4):
    """Replay the PRIMARY corpus folds and check this harness reproduces the
    balanced accuracy `train_abmil.py` recorded when it wrote each checkpoint.

    This is the whole safety argument for the external run. Loading a
    checkpoint, rebuilding the architecture from its stored config, restoring
    its frozen normaliser and running a forward pass are four places to be
    subtly wrong in a way that still produces plausible numbers. If checkpoint
    (r, f) scored on its OWN held-out UBC-OCEAN fold reproduces the number in
    the results CSV to 1e-4, none of those four is wrong.

    Run with UBC_RUN_DIR unset, i.e. against cellvit_run.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    arm = next(a for a in ARMS if a["key"] == arm_key)
    gdir = PRIMARY_ROOT / "cellvit_run" / "graphs"
    rdir = gdir / "gatv2_run"
    csv = rdir / f"abmil_{arm['ckpt_tag']}_results.csv"
    if not csv.exists():
        raise SystemExit(f"no results CSV at {csv}")
    df = pd.read_csv(csv)
    df = df[df["config"] == CFGNAME]

    print(f"\nSELF-TEST on the primary corpus, arm {arm_key}")
    X, y, ids, feat_names, _ = load_external_bags(arm, gdir)

    cdir = CKPT_ROOT / arm["ckpt_tag"]
    ck0 = torch.load(sorted(cdir.glob(f"{CFGNAME}_r*f*.pt"))[0],
                     map_location="cpu", weights_only=False)
    Xa = align_to_checkpoint(X, feat_names, ck0["feat_names"])

    worst, checked, rows = 0.0, 0, []
    for _, row in df.iterrows():
        r, f = int(row["repeat"]), int(row["fold"])
        cp = cdir / f"{CFGNAME}_r{r}f{f}.pt"
        if not cp.exists():
            continue
        # exactly the split train_abmil.py used: outer StratifiedKFold(5,
        # shuffle=True, random_state=repeat) over the SAME ordered bag list
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        te = list(skf.split(np.zeros(len(Xa)), y))[f][1]
        probs, _ = score_checkpoint(cp, [Xa[i] for i in te], device)
        got = balanced_accuracy_score(y[te], probs.argmax(1))
        want = float(row["balanced_acc"])
        d = abs(got - want)
        worst = max(worst, d)
        checked += 1
        rows.append((r, f, want, got, d))
        if d > tol:
            print(f"    MISMATCH r{r}f{f}: csv={want:.4f} harness={got:.4f} "
                  f"diff={d:.5f}")

    print(f"  slides loaded : {len(Xa)}  features {len(ck0['feat_names'])}")
    print(f"  cells checked : {checked}")
    print(f"  worst |diff|  : {worst:.6f}   (tolerance {tol})")
    for r, f, want, got, d in rows[:5]:
        print(f"    r{r}f{f}  csv {want:.4f}  harness {got:.4f}  diff {d:.6f}")
    ok = worst <= tol and checked > 0
    print(f"  {'PASS -- harness reproduces training-time scoring' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", default=None, metavar="ARM",
                    help="replay the primary corpus folds for one arm and "
                         "check against its results CSV (run with UBC_RUN_DIR "
                         "unset)")
    ap.add_argument("--arms", nargs="+", default=[a["key"] for a in ARMS])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test(args.self_test, args.device) else 1)

    gdir = corpus_paths.GRAPH_DIR
    print(corpus_paths.banner("eval_external"))
    if not corpus_paths.IS_EXTERNAL:
        print("  WARNING: UBC_RUN_DIR is the PRIMARY corpus -- the checkpoints "
              "were trained on these very slides, so nothing here is held out.")
    print(f"  graphs   : {gdir}")
    print(f"  ckpt root: {CKPT_ROOT}")
    print(f"  device   : {args.device}")

    arms = [a for a in ARMS if a["key"] in args.arms]
    results = [run_arm(a, gdir, args.device) for a in arms]

    y = results[0]["y"]
    counts = Counter(CLASSES[i] for i in y)
    print(f"\n{'=' * 74}\nEXTERNAL CORPUS: {len(y)} slides  "
          f"{dict(sorted(counts.items()))}\n{'=' * 74}")

    print(f"\nbootstrap: {args.boot} slide resamples, seed {args.seed}, "
          f"shared across arms")
    mean_boot, ens_boot, n_deg, _ = bootstrap(results, args.boot, args.seed)
    if n_deg:
        print(f"  {n_deg} replicate(s) dropped for losing a class entirely")

    print(f"\n{'arm':<32s} {'mean of 50':>10s} {'95% CI':>18s} "
          f"{'ensemble':>9s} {'95% CI':>18s}")
    print("-" * 94)
    summary = {}
    for r in results:
        k = r["arm"]["key"]
        m = float(np.mean([c["bal"] for c in r["per_ckpt"]]))
        s = float(np.std([c["bal"] for c in r["per_ckpt"]]))
        e = float(balanced_accuracy_score(r["y"], r["P"].mean(0).argmax(1)))
        ml, mh = ci(mean_boot[k])
        el, eh = ci(ens_boot[k])
        print(f"{r['arm']['nice']:<32s} {m:10.4f} [{ml:6.4f},{mh:6.4f}] "
              f"{e:9.4f} [{el:6.4f},{eh:6.4f}]")
        summary[k] = dict(nice=r["arm"]["nice"], mean_of_50=m, sd_across_ckpts=s,
                          mean_ci=[ml, mh], ensemble=e, ensemble_ci=[el, eh],
                          per_ckpt=r["per_ckpt"], n_features=len(r["feat_names"]))

    if "clean" in summary and len(results) > 1:
        print(f"\n{'delta vs named morphometry':<32s} {'mean':>10s} "
              f"{'95% CI':>18s} {'P(>0)':>7s}")
        print("-" * 70)
        base = np.asarray(mean_boot["clean"])
        for r in results:
            k = r["arm"]["key"]
            if k == "clean":
                continue
            d = np.asarray(mean_boot[k]) - base
            dl, dh = ci(d)
            pg = float((d > 0).mean())
            print(f"{r['arm']['nice']:<32s} "
                  f"{summary[k]['mean_of_50'] - summary['clean']['mean_of_50']:10.4f} "
                  f"[{dl:6.4f},{dh:6.4f}] {pg:7.3f}")
            summary[k]["delta_vs_clean"] = (summary[k]["mean_of_50"]
                                            - summary["clean"]["mean_of_50"])
            summary[k]["delta_ci"] = [dl, dh]
            summary[k]["delta_p_gt_0"] = pg

    # per-class recall, with the thin classes flagged
    print(f"\nper-class recall (ensemble of 50)")
    hdr = f"{'arm':<32s}" + "".join(f"{c:>9s}" for c in CLASSES)
    print(hdr)
    print(f"{'n slides':<32s}" + "".join(f"{counts[c]:>9d}" for c in CLASSES))
    print("-" * len(hdr))
    for r in results:
        pred = r["P"].mean(0).argmax(1)
        cm = confusion_matrix(r["y"], pred, labels=range(len(CLASSES)))
        rec = cm.diagonal() / cm.sum(1).clip(min=1)
        print(f"{r['arm']['nice']:<32s}" + "".join(f"{v:9.3f}" for v in rec))
        summary[r["arm"]["key"]]["per_class_recall"] = {
            c: float(v) for c, v in zip(CLASSES, rec)}
        summary[r["arm"]["key"]]["confusion"] = cm.tolist()

    thin = [c for c in CLASSES if counts[c] < 15]
    if thin:
        print(f"\n  NOTE: {', '.join(thin)} have <15 slides "
              f"({', '.join(f'{c}={counts[c]}' for c in thin)}). Recall on "
              f"those moves in steps of 1/n, so the balanced accuracy -- which "
              f"weights all five classes equally -- carries their noise at full "
              f"weight. Treat their per-class numbers as indicative only.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = dict(
            corpus=str(gdir), n_slides=int(len(y)),
            class_counts={c: int(counts[c]) for c in CLASSES},
            slide_ids=results[0]["ids"], classes=CLASSES,
            n_boot=args.boot, seed=args.seed,
            bootstrap_degenerate=n_deg, config=CFGNAME, arms=summary)
        out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

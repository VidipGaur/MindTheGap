# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Overnight driver: model selection -> more seeds -> alternative MIL algorithms.

    python scripts/analysis/run_overnight.py --stage all          # everything, in order
    python scripts/analysis/run_overnight.py --stage a            # selection rules only
    python scripts/analysis/run_overnight.py --stage c            # algorithms only
    python scripts/analysis/run_overnight.py --dry-run            # print the plan, run nothing

WHY A DRIVER AND NOT A LIST OF COMMANDS
=======================================
Stage C (which MIL algorithm) depends on Stage A (which model-selection rule),
because comparing algorithms under a selection rule that is itself the dominant
noise source measures the noise, not the algorithms. So this script runs Stage
A, READS ITS RESULTS, picks the winning rule by paired comparison against the
incumbent, and only then launches Stage C under that rule. Unattended, with the
dependency resolved automatically.

Every child run writes its own results CSV incrementally and resumes from
partial progress, so killing this script loses at most one run.

STAGE A -- the selection rule
=============================
The 3 seeds agreed on the predicted class on only 63.4% of slides. A large part
of that is suspected to be the SELECTION RULE, not the model: `argmax` picks the
single best epoch off an inner-validation curve computed on ~73 slides across 5
imbalanced classes, so one lucky epoch can win and two seeds can land on very
different points of the same trajectory.

Seven arms, all at abmil_h128_lr1 on the nuclear-only feature set, all 10
repeats x 5 folds, all on IDENTICAL training trajectories for a given seed --
only the rule that reads the trajectory changes:

    argmax_es   argmax + early stopping (patience 30)   <- the incumbent
    argmax      argmax over the full run, no early stop <- isolates the stop
    smooth      argmax of the smoothed val curve
    fixed       last epoch, no selection at all
    swa         weight-average the tail of training
    swaflr      swa, but with the lr floored at lr/10 so the tail moves
    swatop      weight-average the top-k epochs by val

STAGE C -- the algorithm
========================
Alternative MIL formulations at the winning selection rule, 5 repeats x 5 folds
(paired against Stage A's repeats 0-4):

    additive    per-instance class contributions that SUM to the logit
    clam        ABMIL + instance clustering loss supervising the attention
    transmil    self-attention between tiles, then the same pooling

Not included here: the slide-level gradient-boosting control, which is CPU-only
and runs separately via train_slidelevel_gbm.py.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import corpus_paths

ROOT = corpus_paths.ROOT
OUT = ROOT / "cellvit_run" / "graphs" / "gatv2_run"
SCRIPTS = Path(__file__).resolve().parent.parent
PY = sys.executable

# every arm runs on the nuclear-only feature set: Task 3b showed the 39
# graph/streamline columns are redundant (p=0.84), and dropping them keeps
# every surviving column a named pathology quantity
BASE = ["--nuclear-features", "--drop-groups", "graph"]
REF_CONFIG = "abmil_h128_lr1"

# arm -> extra flags. Ordered cheapest-first so a truncated night still yields
# the incumbent plus the most likely winners.
STAGE_A = [
    ("sel_argmax_es", ["--select", "argmax", "--patience", "30"]),
    ("sel_argmax",    ["--select", "argmax", "--patience", "0"]),
    ("sel_smooth",    ["--select", "smooth"]),
    ("sel_swatop",    ["--select", "swatop"]),
    ("sel_swaflr",    ["--select", "swa", "--sched", "cosine_floor"]),
    ("sel_swa",       ["--select", "swa"]),
    ("sel_fixed",     ["--select", "fixed"]),
]

STAGE_C_CONFIGS = ["addmil_h128", "addmil_h128_do5", "addmil_mh4",
                   "clam_h128", "clam_h128_do5",
                   "transmil_L1", "transmil_L2"]

# Stage D -- the chromatin/texture retrain. arm -> extra feature flags, all at
# REF_CONFIG so the only thing changing is what goes into the encoder.
# `chrom_full` is the headline; the rest are the ablations that say WHICH part
# of the pixel-derived block is doing the work, and whether it survives
# dropping the per-tile staining descriptors (which are a dead_count-shaped
# confound in their own right).
STAGE_D = [
    ("chrom_full",      []),
    ("chrom_notexture", ["--drop-groups", "graph", "chrom_texture"]),
    ("chrom_nolevel",   ["--drop-groups", "graph", "chrom_level"]),
    ("chrom_nocontext", ["--drop-groups", "graph", "chrom_context"]),
]

# map an arm name back to the flags that reproduce its selection rule, so
# Stage C can inherit Stage A's winner
ARM_FLAGS = dict(STAGE_A)


def run(cmd, log_path, dry=False):
    """Run a child process, streaming to a log file. Returns its exit code.

    stdout and stderr go to separate files deliberately: piping them together
    through PowerShell makes Python fully buffer stdout, which hides warnings
    such as a stale bag cache until the run ends."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    print(f"  log -> {log_path.name}")
    if dry:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as out, \
            open(log_path.with_suffix(".err.log"), "w", encoding="utf-8") as err:
        rc = subprocess.run(cmd, stdout=out, stderr=err, cwd=str(SCRIPTS)).returncode
    print(f"  exit {rc} after {(time.time() - t0) / 60:.1f} min")
    if rc != 0:
        tail = log_path.with_suffix(".err.log").read_text(encoding="utf-8",
                                                          errors="replace")
        print("  stderr tail:\n    " + "\n    ".join(tail.strip().splitlines()[-12:]))
    return rc


def train_cmd(arm, extra, repeats, epochs, configs, tag="v5nuc"):
    cmd = [PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"), "--tag", tag, *BASE,
           "--arm", arm, "--repeats", str(repeats), "--epochs", str(epochs)]
    if configs:
        cmd += ["--only", ",".join(configs)]
    return cmd + extra


def load_arm(arm, tag="v5nuc"):
    p = OUT / f"abmil_{tag}_{arm}_results.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[d.balanced_acc.notna()]


def paired(a, b, key="balanced_acc"):
    """Paired delta a-b over the (config, repeat, fold) cells both ran."""
    from scipy import stats
    ka = a.set_index(["config", "repeat", "fold"])[key]
    kb = b.set_index(["config", "repeat", "fold"])[key]
    common = ka.index.intersection(kb.index)
    if len(common) < 3:
        return None
    d = (ka.loc[common] - kb.loc[common]).values
    return dict(n=len(d), delta=float(d.mean()), sd=float(d.std(ddof=1)),
                p=float(stats.ttest_rel(ka.loc[common].values,
                                        kb.loc[common].values).pvalue),
                win=float((d > 0).mean()))


def seed_agreement(arm, repeats, min_repeats=4):
    """Fraction of slides on which every seed predicts the same subtype.

    This is the number Stage A actually exists to move. Accuracy was never the
    complaint -- the complaint is that the 3 seeds agreed on the predicted
    class only 63.4% of the time, which is what makes any single-slide claim
    fragile. A selection rule that ties on accuracy but agrees far more often
    is a win, and scoring arms on accuracy alone would miss it entirely.

    Runs the saved checkpoints out-of-fold, so it costs one forward pass per
    slide per repeat -- about a minute per arm.
    """
    import torch
    from analyse_abmil_nuclear import bags_for, collect
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, ids, fn = bags_for(ROOT, f"v5nuc_{arm}", True, drop_groups=["graph"])
    per_rep = collect(ROOT, f"v5nuc_{arm}", REF_CONFIG, X, y, repeats, device)
    reps = [r for r in per_rep if per_rep[r]]
    if len(reps) < min_repeats:
        return None
    common = sorted(set.intersection(*[set(per_rep[r]) for r in reps]))
    if not common:
        return None
    P = np.array([[per_rep[r][i][0] for i in common] for r in reps])
    unanimous = float(np.mean([len(set(P[:, j])) == 1 for j in range(P.shape[1])]))
    pair = float(np.mean([(P[k] == P[l]).mean()
                          for k in range(len(reps))
                          for l in range(k + 1, len(reps))]))
    return dict(n_seeds=len(reps), n_slides=len(common),
                unanimous=unanimous, pairwise=pair)


def pick_selection_winner(verbose=True, with_stability=True, repeats=10):
    """Which selection rule to carry into Stage C.

    Two criteria, in the order the project actually cares about:

      1. ACCURACY IS A CONSTRAINT, not the objective. An arm is eligible if it
         does not significantly LOSE accuracy against the incumbent (paired
         delta above -0.01, or a p-value that says the gap is noise). At this
         sd, chasing a nominal +0.01 with p=0.6 just moves noise around.
      2. SEED AGREEMENT IS THE OBJECTIVE. Among eligible arms, take the one
         whose independently-seeded models most often predict the same class.
         That is the number Stage A was built to move.

    The incumbent keeps the job on ties, and whenever stability cannot be
    measured (fewer than 4 seeds of checkpoints on disk).
    """
    base = load_arm("sel_argmax_es")
    if base is None or not len(base):
        return "sel_argmax_es", "incumbent (no Stage A results found)"
    rows = []
    for arm, _ in STAGE_A:
        d = load_arm(arm)
        if d is None or not len(d):
            continue
        r = paired(d, base) if arm != "sel_argmax_es" else dict(
            n=len(d), delta=0.0, sd=0.0, p=1.0, win=0.5)
        if not r:
            continue
        row = dict(arm=arm, mean=float(d.balanced_acc.mean()),
                   fold_sd=float(d.balanced_acc.std(ddof=1)), **r)
        if with_stability:
            try:
                s = seed_agreement(arm, repeats)
            except Exception as e:
                print(f"  (stability unavailable for {arm}: "
                      f"{type(e).__name__}: {e})")
                s = None
            if s:
                row.update(seeds=s["n_seeds"], unanimous=s["unanimous"],
                           pairwise=s["pairwise"])
        rows.append(row)

    if verbose:
        print("\n" + "=" * 78)
        print("STAGE A -- selection rules vs argmax + early stopping")
        print("=" * 78)
        print("  delta/p/win = paired ACCURACY vs the incumbent (a constraint)")
        print("  unanimous   = fraction of slides all seeds label the same "
              "(the objective)")
        if rows:
            key = "pairwise" if any("pairwise" in r for r in rows) else "delta"
            t = pd.DataFrame(rows).sort_values(key, ascending=False)
            print(t.round(4).to_string(index=False))
            print("  (unanimous falls as seeds are added and only compares arms "
                  "with equal seed counts; pairwise is the one being optimised)")

    if not rows:
        return "sel_argmax_es", "incumbent (nothing to compare)"
    if not any("unanimous" in r for r in rows):
        cand = [r for r in rows
                if r["arm"] != "sel_argmax_es" and r["delta"] > 0
                and r["p"] < 0.20 and r["win"] > 0.55]
        if not cand:
            return "sel_argmax_es", ("incumbent (no stability measurable; no arm "
                                     "cleared delta>0, p<0.20, win>0.55)")
        cand.sort(key=lambda r: -r["delta"])
        w = cand[0]
        return w["arm"], (f"accuracy only: delta {w['delta']:+.4f} "
                          f"p={w['p']:.3f} win={w['win']:.2f}")

    # accuracy as a constraint, stability as the objective.
    # PAIRWISE agreement is the objective, not unanimous: "all seeds agree"
    # gets mechanically harder as seeds are added (10 seeds unanimous on 44.3%
    # of slides vs 3 seeds on 63.4% for the same models), so it cannot compare
    # arms that have finished different numbers of repeats. Mean pairwise
    # agreement is invariant to seed count and measures the same thing.
    elig = [r for r in rows if "pairwise" in r
            and (r["delta"] > -0.01 or r["p"] > 0.10)]
    if not elig:
        return "sel_argmax_es", "incumbent (every arm lost accuracy)"
    base_u = next((r["pairwise"] for r in elig if r["arm"] == "sel_argmax_es"),
                  None)
    elig.sort(key=lambda r: (-r["pairwise"], -r["delta"]))
    w = elig[0]
    if w["arm"] == "sel_argmax_es" or base_u is None \
            or w["pairwise"] - base_u < 0.02:
        return "sel_argmax_es", (
            f"incumbent -- best challenger gains only "
            f"{(w['pairwise'] - (base_u or 0)):+.3f} pairwise seed agreement")
    return w["arm"], (f"pairwise seed agreement {w['pairwise']:.3f} vs "
                      f"incumbent {base_u:.3f} ({w['pairwise'] - base_u:+.3f}), "
                      f"accuracy delta {w['delta']:+.4f} (p={w['p']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "a", "c", "d", "pick"])
    ap.add_argument("--allow-partial", action="store_true",
                    help="Stage D only: run even though the chromatin "
                         "extraction has not finished for every slide")
    ap.add_argument("--repeats-a", type=int, default=10)
    ap.add_argument("--repeats-c", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-selection", default=None,
                    help="skip the Stage A read and use this arm's rule for "
                         "Stage C (e.g. sel_smooth)")
    ap.add_argument("--no-stability", dest="stability", action="store_false",
                    help="pick the Stage C rule on accuracy alone, without "
                         "running checkpoints to measure seed agreement")
    args = ap.parse_args()

    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.stage in ("all", "a"):
        print("=" * 78)
        print(f"STAGE A: {len(STAGE_A)} selection rules x {args.repeats_a} "
              f"repeats x 5 folds, {REF_CONFIG}, nuclear-only")
        print("=" * 78)
        for arm, extra in STAGE_A:
            run(train_cmd(arm, extra, args.repeats_a, args.epochs, [REF_CONFIG]),
                logs / f"stageA_{arm}.log", args.dry_run)

    if args.stage in ("all", "c", "pick"):
        if args.force_selection:
            win, why = args.force_selection, "forced on the command line"
        else:
            win, why = pick_selection_winner(with_stability=args.stability,
                                             repeats=args.repeats_a)
        print(f"\n>>> Stage C selection rule: {win}  ({why})")
        if args.stage == "pick":
            return
        extra = ARM_FLAGS[win]
        print("=" * 78)
        print(f"STAGE C: {len(STAGE_C_CONFIGS)} configs x {args.repeats_c} "
              f"repeats x 5 folds, under {win}")
        print("=" * 78)
        for cfg in STAGE_C_CONFIGS:
            # one child process per config so a crash in transmil cannot take
            # the additive/clam results down with it
            run(train_cmd(f"alg_{win}", extra, args.repeats_c, args.epochs, [cfg]),
                logs / f"stageC_{cfg}.log", args.dry_run)

    if args.stage in ("all", "d"):
        run_stage_d(args, logs)

    print(f"\nTOTAL {(time.time() - t0) / 3600:.2f} h")


def run_stage_d(args, logs):
    """Retrain on the chromatin/texture block, plus its ablations.

    Gated on a COMPLETE extraction. This project has already lost a sweep to a
    bag cache built from 106 of 456 slides, and the loader silently drops any
    slide whose chromatin npz is missing -- so a partial corpus here would not
    error, it would quietly train on a biased subset. Every run also passes
    --rebuild-cache, because SCHEMA_VERSION only detects loader-code changes
    and cannot know new npz files appeared on disk.
    """
    gdir = ROOT / "cellvit_run" / "graphs"
    n_graph = len(list(gdir.glob("*_graph.npz")))
    n_chrom = len(list((gdir / "chromatin_feats").glob("*_chromatin.npz")))
    print("\n" + "=" * 78)
    print(f"STAGE D: chromatin retrain -- {n_chrom}/{n_graph} slides extracted")
    print("=" * 78)
    if n_chrom < n_graph and not args.allow_partial:
        print(f"  REFUSING to run on a partial corpus ({n_graph - n_chrom} "
              f"slides still missing).")
        print("  The loader DROPS slides without a chromatin npz rather than")
        print("  erroring, so this would train on a biased subset and look fine.")
        print("  Wait for extract_chromatin_features.py, or pass --allow-partial")
        print("  if you deliberately want the subset.")
        return
    # Stage D must use the SAME selection rule as the nuclear-only baseline it
    # will be paired against, or the chromatin delta would be confounded with a
    # change of selection rule.
    if args.force_selection:
        sel_arm = args.force_selection
    else:
        sel_arm, why = pick_selection_winner(with_stability=args.stability,
                                             repeats=args.repeats_a)
        print(f"  selection rule: {sel_arm}  ({why})")
    sel_flags = ARM_FLAGS[sel_arm]
    print(f"  baseline to pair against: abmil_v5nuc_{sel_arm} / {REF_CONFIG}")

    base = ["--nuclear-features", "--chromatin-features", "--rebuild-cache"]
    for i, (arm, extra) in enumerate(STAGE_D):
        cmd = [PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"), "--tag", "v5nuc", *base,
               "--arm", arm, "--repeats", str(args.repeats_a),
               "--epochs", str(args.epochs), "--only", REF_CONFIG]
        cmd += extra if extra else ["--drop-groups", "graph"]
        cmd += sel_flags
        # only the first arm needs to rebuild; the rest share the cache
        if i:
            cmd = [c for c in cmd if c != "--rebuild-cache"]
        run(cmd, logs / f"stageD_{arm}.log", args.dry_run)


if __name__ == "__main__":
    main()

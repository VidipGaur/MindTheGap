# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Phase 3 driver: eosin, the positive control, mixup, and the final ensemble.

Runs unattended from wherever the corpus currently is to a finished set of
results, making the between-stage decisions in code rather than waiting for a
human. Sequence:

  wait   -- for the eosin and foundation-model extractions to finish, and for
            any training already on the GPU to exit
  E      -- eosin/cytoplasm features: does the block help, and is it the
            clearing measurement specifically or just more eosin columns?
  F      -- THE POSITIVE CONTROL. Phikon and Phikon-v2 tile embeddings through
            the identical bag/fold/pooling/selection machinery, plus a hybrid
            arm that concatenates them with the interpretable features.
  G      -- bag-level mixup on whichever feature set won, at three strengths
  H      -- the final model: best features + best augmentation under Additive
            MIL, 10 seeds, with the probability ensemble and the attention-map
            ensemble measured

DECISION RULES, stated up front so the choices are not post-hoc:

  * A feature block is ADOPTED if its paired delta against the current best is
    positive and p < 0.10 (paired t over shared (repeat, fold) cells). Per-cell
    sd is ~0.06 while real effects are ~0.02, so unpaired means are worthless
    here and a p-value on 50 paired cells is the minimum evidence.
  * Mixup is ADOPTED on the same rule.
  * The reference config for every FEATURE arm stays abmil_h128_lr1, so a delta
    is attributable to the features and not to a simultaneous architecture
    change. Additive MIL enters only at Stage H, once the feature set is fixed.
  * Selection rule is swatop throughout (Stage A's winner), for the same
    reason: one variable at a time.

    python scripts/analysis/run_phase3.py                 # full chain
    python scripts/analysis/run_phase3.py --skip-wait     # extractions already done
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
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import corpus_paths

ROOT = corpus_paths.ROOT
GDIR = ROOT / "cellvit_run" / "graphs"
ODIR = GDIR / "gatv2_run"
LOGS = ODIR / "logs"
PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent.parent

REF_CONFIG = "abmil_h128_lr1"        # feature arms are judged at this config
ADD_CONFIG = "addmil_h128_do5"       # Stage C's best additive config
SELECT = ["--select", "swatop"]
REPEATS = 10
EPOCHS = 120

# nuclear-only under the winning selection rule -- the incumbent to beat
BASE_TAG = "v5nuc_sel_swatop"

ALPHA = 0.10                          # adoption threshold, stated above


def run(cmd, log_name, timeout=None):
    """Run one job, stdout and stderr to SEPARATE files.

    Never combine them: PowerShell's 2>&1 fully buffers Python's stdout when it
    is not a TTY, which hides warnings (for example a stale bag cache) until
    the run ends.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    out_p, err_p = LOGS / f"{log_name}.log", LOGS / f"{log_name}.err.log"
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    print(f"  log -> {out_p.name}")
    t0 = time.time()
    with open(out_p, "w", encoding="utf-8") as o, \
            open(err_p, "w", encoding="utf-8") as e:
        p = subprocess.run(cmd, stdout=o, stderr=e, cwd=str(SCRIPTS),
                           timeout=timeout)
    print(f"  exit {p.returncode} after {(time.time()-t0)/60:.1f} min")
    return p.returncode


def train(tag, arm, extra, config=REF_CONFIG, repeats=REPEATS, rebuild=False):
    cmd = [PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"), "--tag", tag,
           "--arm", arm, "--repeats", str(repeats), "--epochs", str(EPOCHS),
           "--only", config, *SELECT, *extra]
    if rebuild:
        cmd.append("--rebuild-cache")
    return run(cmd, f"p3_{arm}")


def n_files(d, pat):
    return len(list(Path(d).glob(pat))) if Path(d).exists() else 0


def wait_for(label, check, target, timeout_h=14, poll=180):
    """Block until `check()` reaches `target`, reporting progress."""
    t0 = time.time()
    last = -1
    while True:
        n = check()
        if n != last:
            el = (time.time() - t0) / 60
            rate = n / max(el, 1e-6)
            eta = (target - n) / rate if rate > 0 else float("inf")
            print(f"  [{label}] {n}/{target}  ({el:.0f}m elapsed, "
                  f"eta {eta:.0f}m)", flush=True)
            last = n
        if n >= target:
            return True
        if (time.time() - t0) / 3600 > timeout_h:
            print(f"  [{label}] TIMED OUT at {n}/{target} after {timeout_h}h")
            return False
        time.sleep(poll)


def gpu_busy():
    """Any train_abmil.py still running? Two sweeps on one GPU just thrash."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-WmiObject Win32_Process -Filter \"Name='python.exe'\" | "
                 "Select-Object -ExpandProperty CommandLine"],
                capture_output=True, text=True, timeout=60).stdout
        else:
            out = "\n".join(
                a for a in subprocess.run(
                    ["ps", "-axo", "args="],
                    capture_output=True, text=True, timeout=60).stdout.splitlines()
                if "python" in a)
    except Exception:
        return False
    return "train_abmil.py" in (out or "")


def cells(tag, config=REF_CONFIG):
    """The (repeat, fold) -> balanced_acc series for one arm, or None."""
    p = ODIR / f"abmil_{tag}_results.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[(d.config == config) & d.balanced_acc.notna()]
    if not len(d):
        return None
    return d.set_index(["repeat", "fold"])["balanced_acc"]


def paired(a_tag, b_tag, config=REF_CONFIG, label=""):
    """Paired comparison of arm a against arm b on shared folds."""
    sa, sb = cells(a_tag, config), cells(b_tag, config)
    if sa is None or sb is None:
        print(f"  {label or a_tag}: MISSING results "
              f"(a={'ok' if sa is not None else 'none'}, "
              f"b={'ok' if sb is not None else 'none'})")
        return None
    common = sa.index.intersection(sb.index)
    if len(common) < 10:
        print(f"  {label or a_tag}: only {len(common)} shared cells -- skipped")
        return None
    delta = (sa.loc[common] - sb.loc[common]).values
    t = stats.ttest_rel(sa.loc[common].values, sb.loc[common].values)
    try:
        w = stats.wilcoxon(delta).pvalue
    except Exception:
        w = float("nan")
    r = dict(arm=a_tag, mean=float(sa.loc[common].mean()),
             base=float(sb.loc[common].mean()), delta=float(delta.mean()),
             p=float(t.pvalue), wil=float(w), win=float((delta > 0).mean()),
             n=len(common))
    print(f"  {label or a_tag:28s} {r['mean']:.4f}  delta {r['delta']:+.4f}  "
          f"p={r['p']:.4f} wil={r['wil']:.4f} win={r['win']:.2f} n={r['n']}")
    return r


def adopt(r):
    return r is not None and r["delta"] > 0 and r["p"] < ALPHA


def stage_e():
    """Eosin/cytoplasm. Does it help, and is it the clearing measurement?"""
    print("\n" + "=" * 78)
    print("STAGE E -- eosin / cytoplasm features (CC and MC evidence)")
    print("=" * 78)
    arms = [
        ("eos_full",   ["--nuclear-features", "--eosin-features",
                        "--drop-groups", "graph"], True),
        ("eos_noclear", ["--nuclear-features", "--eosin-features",
                         "--drop-groups", "graph", "eos_clear"], False),
        ("eos_onlyclear", ["--nuclear-features", "--eosin-features",
                           "--drop-groups", "graph", "eos_other"], False),
    ]
    for arm, extra, rebuild in arms:
        if (ODIR / f"abmil_v5nuceos_{arm}_results.csv").exists():
            print(f"  {arm}: results already present, skipping")
            continue
        train("v5nuc", arm, extra, rebuild=rebuild)

    print("\n  paired vs nuclear-only baseline:")
    res = {a: paired(f"v5nuceos_{a}", BASE_TAG, label=a) for a, _, _ in arms}
    return res


def stage_f():
    """The positive control: foundation-model embeddings, same everything else."""
    print("\n" + "=" * 78)
    print("STAGE F -- POSITIVE CONTROL: pathology foundation-model embeddings")
    print("=" * 78)
    print("  phikon is the encoder behind the Kaggle winner on this dataset")
    print("  (~0.66 balanced accuracy); phikon-v2 is its stronger successor.")
    arms = [
        ("fm_phikonv2", ["--fm-features", "phikonv2"], "v5fmphikonv2"),
        ("fm_phikon",   ["--fm-features", "phikon"], "v5fmphikon"),
        ("fm_hybrid",   ["--fm-features", "phikonv2", "--fm-concat",
                         "--nuclear-features", "--drop-groups", "graph"],
         "v5nucfmphikonv2cat"),
    ]
    for arm, extra, tag in arms:
        if (ODIR / f"abmil_{tag}_{arm}_results.csv").exists():
            print(f"  {arm}: results already present, skipping")
            continue
        train("v5", arm, extra, rebuild=True)

    print("\n  paired vs the interpretable nuclear-only model:")
    out = {}
    for arm, _, tag in arms:
        out[arm] = paired(f"{tag}_{arm}", BASE_TAG, label=arm)
    return out


def best_features(stage_e_res, stage_f_res):
    """Which feature set goes forward? Adoption rule stated in the docstring."""
    print("\n" + "-" * 78)
    print("FEATURE SET DECISION")
    print("-" * 78)
    # Chromatin was decided in Stage D of the previous driver. Both variants are
    # re-read, because dropping the GLCM texture columns beat keeping them
    # (+0.026 p=0.003 vs +0.018 p=0.062): the texture block dilutes the signal,
    # so the arm carried forward must be the better one, not the full block.
    chrom_full = paired("v5nucchrom_chrom_full", BASE_TAG,
                        label="chromatin, full")
    chrom_notex = paired("v5nucchrom_chrom_notexture", BASE_TAG,
                         label="chromatin, no texture")
    chrom_extra = ["--nuclear-features", "--chromatin-features",
                   "--drop-groups", "graph"]
    chrom, chrom_tag = chrom_full, "v5nucchrom_chrom_full"
    if chrom_notex and (chrom is None or chrom_notex["delta"] > chrom["delta"]):
        chrom, chrom_tag = chrom_notex, "v5nucchrom_chrom_notexture"
        chrom_extra = chrom_extra + ["chrom_texture"]

    eos_extra = ["--nuclear-features", "--eosin-features",
                 "--drop-groups", "graph"]
    eos, eos_tag = stage_e_res.get("eos_full"), "v5nuceos_eos_full"
    eos_noclear = stage_e_res.get("eos_noclear")
    if eos_noclear and (eos is None or eos_noclear["delta"] > eos["delta"]):
        eos, eos_tag = eos_noclear, "v5nuceos_eos_noclear"
        eos_extra = eos_extra + ["eos_clear"]

    cands = [
        ("nuclear only", BASE_TAG, ["--nuclear-features", "--drop-groups", "graph"],
         None),
        ("+ chromatin", chrom_tag, chrom_extra, chrom),
        ("+ eosin", eos_tag, eos_extra, eos),
    ]
    winner = cands[0]
    for name, tag, extra, r in cands[1:]:
        if adopt(r) and (winner[3] is None or r["delta"] > winner[3]["delta"]):
            winner = (name, tag, extra, r)
    print(f"\n  -> carrying forward: {winner[0]}  (tag {winner[1]})")
    if winner[3] is not None:
        print(f"     adopted on delta {winner[3]['delta']:+.4f}, "
              f"p={winner[3]['p']:.4f}")
    else:
        print("     nothing beat the nuclear-only baseline at p<0.10")

    # If BOTH pixel blocks were adopted individually the combination is worth
    # one more arm -- they read different stains (haematoxylin vs eosin) and
    # target different classes (nuclear grade vs CC/MC), so they may well be
    # additive rather than redundant. Whichever variant of each block won above
    # is the one combined, so the drop list is inherited, not rebuilt.
    if adopt(chrom) and adopt(eos):
        print("\n  both pixel blocks were adopted; testing them together")
        drops = ["graph"]
        for e in (chrom_extra, eos_extra):
            i = e.index("--drop-groups")
            drops += [g for g in e[i + 1:] if g != "graph"]
        combo = (["--nuclear-features", "--chromatin-features",
                  "--eosin-features", "--drop-groups"] + sorted(set(drops)))
        arm = "chrom_eos"
        if not any(ODIR.glob(f"abmil_*_{arm}_results.csv")):
            train("v5nuc", arm, combo, rebuild=True)
        hits = sorted(ODIR.glob(f"abmil_*_{arm}_results.csv"))
        if hits:
            tag = hits[0].stem.replace("abmil_", "").replace("_results", "")
            r = paired(tag, BASE_TAG, label="chromatin + eosin")
            if adopt(r) and r["delta"] > (winner[3]["delta"] if winner[3] else 0):
                winner = ("+ chromatin + eosin", tag, combo, r)
                print(f"  -> combination wins: {r['delta']:+.4f}")
    return winner


def stage_g(winner):
    """Bag-level mixup at three strengths on the winning feature set."""
    print("\n" + "=" * 78)
    print("STAGE G -- bag-level mixup (attacks SLIDE-level overfitting)")
    print("=" * 78)
    print(f"  feature set: {winner[0]}")
    name, base_tag, extra, _ = winner
    out = {}
    for a in (0.2, 0.4, 1.0):
        arm = f"mixup{int(a*100)}"
        tag_guess = None
        cmd_extra = extra + ["--mixup", str(a)]
        if not any(ODIR.glob(f"abmil_*_mx{int(a*100)}_{arm}_results.csv")):
            train("v5nuc", arm, cmd_extra, rebuild=False)
        hits = sorted(ODIR.glob(f"abmil_*_mx{int(a*100)}_{arm}_results.csv"))
        if hits:
            tag_guess = hits[0].stem.replace("abmil_", "").replace("_results", "")
            out[arm] = paired(tag_guess, base_tag, label=f"mixup a={a}")
    best = None
    for arm, r in out.items():
        if adopt(r) and (best is None or r["delta"] > out[best]["delta"]):
            best = arm
    print(f"\n  -> mixup: {best or 'not adopted (no arm beat p<0.10)'}")
    return best, out


def stage_h(winner, mixup_arm):
    """Final model: winning features under Additive MIL, 10 seeds, ensembled."""
    print("\n" + "=" * 78)
    print("STAGE H -- final model (Additive MIL) + ensembles")
    print("=" * 78)
    name, base_tag, extra, _ = winner
    cmd_extra = list(extra)
    if mixup_arm:
        cmd_extra += ["--mixup", str(int(mixup_arm.replace("mixup", "")) / 100)]
    arm = "final_additive"
    if not any(ODIR.glob(f"abmil_*_{arm}_results.csv")):
        train("v5nuc", arm, cmd_extra, config=ADD_CONFIG, rebuild=False)
    hits = sorted(ODIR.glob(f"abmil_*_{arm}_results.csv"))
    if not hits:
        print("  final arm produced no results -- stopping before the ensemble")
        return
    tag = hits[0].stem.replace("abmil_", "").replace("_results", "")
    paired(tag, base_tag, config=ADD_CONFIG, label="final additive")

    dg = []
    if "--drop-groups" in extra:
        i = extra.index("--drop-groups")
        dg = extra[i + 1:]
        dg = [g for g in dg if not g.startswith("--")]
    for what in ("ensemble", "attnensemble", "stability"):
        cmd = [PY, "-u", str(SCRIPTS / "analysis" / "analyse_abmil_nuclear.py"),
               "--what", what, "--tag", tag, "--config", ADD_CONFIG,
               "--repeats", str(REPEATS)]
        if dg:
            cmd += ["--drop-groups", *dg]
        run(cmd, f"p3_final_{what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--timeout-h", type=float, default=14)
    args = ap.parse_args()

    n_graphs = n_files(GDIR, "*_graph.npz")
    print("=" * 78)
    print(f"PHASE 3 DRIVER -- {n_graphs} graphs in the corpus")
    print("=" * 78)

    if not args.skip_wait:
        print("\nwaiting for upstream extractions")
        ok_e = wait_for("eosin", lambda: n_files(GDIR / "eosin_feats",
                                                 "*_eosin.npz"),
                        n_graphs, args.timeout_h)
        ok_f = wait_for("fm/phikonv2",
                        lambda: n_files(GDIR / "fm_feats", "*_phikonv2.npz"),
                        n_graphs, args.timeout_h)
        if not (ok_e and ok_f):
            print("\n  an extraction did not complete; running the stages that")
            print("  do not depend on it and skipping the rest")
        while gpu_busy():
            print("  waiting for the GPU (another train_abmil.py is running)",
                  flush=True)
            time.sleep(300)

    e_res = stage_e()
    f_res = stage_f()
    winner = best_features(e_res, f_res)
    mix, _ = stage_g(winner)
    stage_h(winner, mix)

    print("\n" + "=" * 78)
    print("PHASE 3 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()

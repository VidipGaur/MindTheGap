# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Phase 4 -- clean the books, then move interpretability onto the model that works.

Phase 3 settled the positive control and it changed the target. Phikon-v2 embeddings
through the identical head reach 0.7930 against 0.6875 for the best interpretable
model (paired +0.1055, p<1e-8, win rate 0.98), and concatenating the interpretable
features onto Phikon makes it WORSE (0.7836, -0.0094, p=0.046). So the interpretable
block carries no information Phikon lacks, and "our features add to a foundation
model" is dead as a claim. Two consequences drive this phase.

STAGE I -- clean the books. Every interpretable number still in play was trained
with `dead_count`, which resolve_dead_count.py convicted as a staining artefact, and
with `chrom_context`, which is tile-level staining descriptors and has never been put
through the same test even though dropping it takes the chromatin gain from +0.026
(p=0.003) to +0.006 (p=0.51). Nothing can be written up until the interpretable
headline is measured with both removed. Also runs the eosin and chromatin blocks
TOGETHER for the first time -- each beat nuclear-only alone (+0.026, +0.020) and they
are statistically indistinguishable from each other, so whether they stack is unknown
and cheap to find out.

STAGE J -- put the interpretability on the strong model instead of pushing the weak
one uphill. Additive MIL gives per-instance per-class contributions that sum exactly
to the bag logit; that identity does not care whether the instance vector is 40 named
morphometrics or 1024 opaque dims. Running it on Phikon-v2 buys the exact accounting
at 0.79 instead of 0.69. What it does not buy is named axes -- that is what the
bridge analysis is for, and it is the piece that makes the paper work: if Phikon's
high-contribution tiles are the same tiles the interpretable model flags, and if its
contributions correlate with pleomorphism, then the named features EXPLAIN the strong
model even though they cannot MATCH it.

STAGE K -- the two cheapest remaining accuracy levers on the strong model. Mixup was
directionally positive at all three strengths on the weak features (+0.009/+0.011/
+0.009) but never cleared p<0.10, which reads as underpowered rather than null; and
Phikon-v2 and Phikon v1 tie on accuracy (+0.0105, p=0.11) while being different
encoders, so their errors should decorrelate.

Adoption rule, fixed in advance and unchanged from phase 3: a block is kept only if
its PAIRED delta over shared (repeat, fold) cells is positive at p<0.10. Per-cell sd
is ~0.05 against ~0.02 effects, so unpaired means are worthless here.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from scipy import stats
import corpus_paths

ROOT = corpus_paths.ROOT
ODIR = ROOT / "cellvit_run" / "graphs" / "gatv2_run"
LOGS = ODIR / "logs"
SCRIPTS = Path(__file__).resolve().parent.parent
PY = sys.executable

CFG = "abmil_h128_lr1"      # reference head for every FEATURE arm
ADD = "addmil_h128_do5"     # additive MIL, enters only once features are fixed
REPEATS = "10"
EPOCHS = "120"

# baselines already on disk, as (results file, config)
NUC = ("abmil_v5nuc_sel_swatop_results.csv", CFG)                    # 0.6550
CHR = ("abmil_v5nucchrom_chrom_notexture_results.csv", CFG)          # 0.6810
PV2 = ("abmil_v5fmphikonv2_fm_phikonv2_results.csv", CFG)            # 0.7930


def sh(cmd, log_name):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    out, err = LOGS / f"{log_name}.log", LOGS / f"{log_name}.err.log"
    with open(out, "w") as fo, open(err, "w") as fe:
        rc = subprocess.call([str(c) for c in cmd], cwd=SCRIPTS, stdout=fo, stderr=fe)
    print(f"  log -> {out.name}   exit {rc} after {(time.time()-t0)/60:.1f} min",
          flush=True)
    return rc


def train(tag, arm, extra, config=CFG, log=None):
    return sh([PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"), "--tag", tag, "--arm", arm,
               "--repeats", REPEATS, "--epochs", EPOCHS, "--only", config,
               "--select", "swatop", *extra], log or f"p4_{arm}")


def cells(spec):
    f, cfg = spec
    p = ODIR / f
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if cfg:
        d = d[d.config == cfg]
    return d.set_index(["repeat", "fold"])["balanced_acc"] if len(d) else None


def paired(A, B, label):
    """Paired delta over shared (repeat, fold) cells. Returns (delta, p) or None."""
    a, b = cells(A), cells(B)
    if a is None or b is None:
        print(f"  {label:44s} MISSING "
              f"(a={'ok' if a is not None else 'none'}, "
              f"b={'ok' if b is not None else 'none'})", flush=True)
        return None
    i = a.index.intersection(b.index)
    a, b = a.loc[i], b.loc[i]
    d = (a - b).mean()
    p = stats.ttest_rel(a, b).pvalue
    try:
        w = stats.wilcoxon(a, b).pvalue
    except Exception:
        w = float("nan")
    print(f"  {label:44s} {a.mean():.4f} vs {b.mean():.4f}  delta {d:+.4f}  "
          f"p={p:.4f} wil={w:.4f} win={((a-b) > 0).mean():.2f} n={len(i)}", flush=True)
    return d, p


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# STAGE I -- the honest interpretable headline
def stage_i():
    banner("STAGE I -- clean the interpretable model (no dead_, chrom_context tested)")
    print("  every interpretable number so far contains dead_count (a convicted\n"
          "  staining artefact) and chrom_context (untested staining descriptors).\n"
          "  this stage measures the headline with them removed, and stacks the\n"
          "  eosin and chromatin blocks together for the first time.", flush=True)

    base = ["--nuclear-features", "--chromatin-features", "--eosin-features",
            "--exclude-prefix", "dead_"]

    # all three blocks, dead_ removed, chrom_context still in
    train("v5nuc", "clean_all", base +
          ["--drop-groups", "graph", "chrom_texture", "eos_other", "--rebuild-cache"])
    # same, but also without the tile staining descriptors
    train("v5nuc", "clean_noctx", base +
          ["--drop-groups", "graph", "chrom_texture", "eos_other", "chrom_context"])

    ALL = ("abmil_v5nucchromeos_clean_all_results.csv", CFG)
    NOCTX = ("abmil_v5nucchromeos_clean_noctx_results.csv", CFG)

    print("\n  paired:", flush=True)
    paired(ALL, NUC, "clean_all vs nuclear-only (dirty ref)")
    paired(ALL, CHR, "clean_all vs chromatin arm (dirty ref)")
    r = paired(NOCTX, ALL, "clean_noctx vs clean_all (staining cost)")

    # carry the arm that survives the confound controls, not the higher number
    winner = ALL
    wname, wdrop = "clean_all", ["graph", "chrom_texture", "eos_other"]
    if r is not None and r[1] < 0.10 and r[0] > 0:
        winner = NOCTX
        wname, wdrop = "clean_noctx", ["graph", "chrom_texture", "eos_other",
                                       "chrom_context"]
    print(f"\n  -> carrying forward: {wname}", flush=True)

    # additive MIL on the clean feature set == the reportable interpretable model
    train("v5nuc", "clean_additive",
          ["--nuclear-features", "--chromatin-features", "--eosin-features",
           "--exclude-prefix", "dead_", "--drop-groups", *wdrop],
          config=ADD)
    CLEAN_ADD = ("abmil_v5nucchromeos_clean_additive_results.csv", ADD)
    paired(CLEAN_ADD, winner, "clean additive vs clean ABMIL")
    paired(CLEAN_ADD, PV2, "clean additive vs Phikon-v2 (the real gap)")

    for what in ("ensemble", "stability"):
        sh([PY, "-u", str(SCRIPTS / "analysis" / "analyse_abmil_nuclear.py"), "--what", what,
            "--tag", "v5nucchromeos_clean_additive", "--config", ADD,
            "--repeats", REPEATS, "--drop-groups", *wdrop],
           f"p4_clean_{what}")
    return wdrop


# STAGE J -- exact per-tile accounting on the model that actually works
def stage_j():
    banner("STAGE J -- Additive MIL on Phikon-v2 (interpretability where the "
           "accuracy is)")
    print("  the additive identity (per-instance per-class contributions summing\n"
          "  exactly to the bag logit) is a property of the HEAD, not the instance\n"
          "  vector -- so it transfers to embeddings unchanged. this buys the exact\n"
          "  accounting at 0.79 instead of 0.69. the axes stay unnamed; naming them\n"
          "  is the bridge analysis, which is a separate build.", flush=True)

    train("v5", "fm_addmil",
          ["--fm-features", "phikonv2", "--rebuild-cache"], config=ADD)
    FMA = ("abmil_v5fmphikonv2_fm_addmil_results.csv", ADD)
    paired(FMA, PV2, "additive on Phikon-v2 vs ABMIL on Phikon-v2")

    for what in ("ensemble", "attnensemble", "stability"):
        sh([PY, "-u", str(SCRIPTS / "analysis" / "analyse_abmil_nuclear.py"), "--what", what,
            "--tag", "v5fmphikonv2_fm_addmil", "--config", ADD,
            "--repeats", REPEATS, "--no-nuclear"],
           f"p4_fmadd_{what}")


# STAGE K -- the two cheap accuracy levers left on the strong model
def stage_k():
    banner("STAGE K -- accuracy levers on Phikon-v2")
    print("  mixup was directionally positive at all three strengths on the weak\n"
          "  features but never cleared p<0.10 -- underpowered, not null. and the\n"
          "  two encoders tie on accuracy while being architecturally different, so\n"
          "  their errors should decorrelate.", flush=True)

    train("v5", "fm_mixup40", ["--fm-features", "phikonv2", "--mixup", "0.4"])
    paired(("abmil_v5fmphikonv2_fm_mixup40_results.csv", CFG), PV2,
           "mixup a=0.4 on Phikon-v2")

    train("v5", "fm_mixup20", ["--fm-features", "phikonv2", "--mixup", "0.2"])
    paired(("abmil_v5fmphikonv2_fm_mixup20_results.csv", CFG), PV2,
           "mixup a=0.2 on Phikon-v2")


def main():
    t0 = time.time()
    banner("PHASE 4 -- clean the books, then move the interpretability onto Phikon")
    stage_i()
    stage_j()
    stage_k()
    banner(f"PHASE 4 COMPLETE -- {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())

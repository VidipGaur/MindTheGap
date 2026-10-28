# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Replicate the Phikon-v2 audit programme across three more encoders.

WHY
---
Every audit number in this project is measured on ONE encoder. "The named
pathology vocabulary reconstructs a subspace worth ~17-52% of the encoder's
advantage" (record §19) and "the bridge recovers 38% of the self-ceiling" (§8)
are, as they stand, facts about owkin/phikon-v2. The paper's framing (A/C) is a
claim about pathology FOUNDATION MODELS. One model is an anecdote; four from
four different labs, pretraining corpora and objectives is a finding.

    phikonv2   ViT-L/16   Owkin       460k WSI   (already done -- the baseline)
    uni2h      ViT-H/14   Mahmood     ~200k WSI
    hoptimus0  ViT-g/14   Bioptimus   ~500k WSI
    hibouL     ViT-L/14   HistAI      ~1.2M WSI

Two outcomes and both are publishable. If the subspace share and the bridge R2
land in the same range on all four, the audit result generalises and that is the
paper's strongest sentence. If they scatter, then "how legible a foundation
model is" is a property of the individual encoder -- which is a more interesting
finding, and one nobody can state without having measured more than one.

STAGE ORDER IS PRIORITY ORDER
-----------------------------
The chain is long (~30 h of training after ~5.5 h of extraction) and is meant to
run unattended, so it is ordered so that a partial completion still yields the
results that matter most, and every stage SKIPS work already on disk:

  A  accuracy per encoder            -- the headline comparison
  B  additive MIL per encoder        -- exact per-tile accounting; bridge needs it
  C  paired comparisons              -- free, reads CSVs
  D  bridge per encoder              -- how much is expressible in named terms
  E  subspace split per encoder      -- IS the named part the part doing the work
  F  hybrid per encoder              -- does the vocabulary ADD (phikonv2: no)
  G  ensemble + stability per encoder

CONVENTIONS CARRIED FROM run_phase3/run_phase4 UNCHANGED, deliberately, so
every new number pairs directly against the existing ones:
  * reference head abmil_h128_lr1 for feature arms; addmil_h128_do5 for additive
  * selection swatop, 10 repeats x 5 folds, 120 epochs
  * adoption/report rule: PAIRED delta over shared (repeat, fold) cells. Per-cell
    sd is ~0.06 against ~0.02 effects, so unpaired means are worthless here.

    python scripts/pipeline/run_multiencoder.py               # full chain, waits for extraction
    python scripts/pipeline/run_multiencoder.py --stages A B  # a subset
    python scripts/pipeline/run_multiencoder.py --dry-run     # print the plan and exit
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
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from scipy import stats
import corpus_paths

ROOT = corpus_paths.ROOT
GDIR = ROOT / "cellvit_run" / "graphs"
FMDIR = GDIR / "fm_feats"
ODIR = GDIR / "gatv2_run"
LOGS = ODIR / "logs"
SCRIPTS = Path(__file__).resolve().parent.parent
PY = sys.executable

CFG = "abmil_h128_lr1"       # reference head for every FEATURE arm
ADD = "addmil_h128_do5"      # additive MIL, once the features are fixed
REPEATS = "10"
EPOCHS = "120"
N_GRAPHS = 456

ENCODERS = [
    ("uni2h", "UNI2-h"),
    ("hoptimus0", "H-optimus-0"),
    ("hibouL", "Hibou-L"),
]
SUB_ARMS = ("resid", "residnull", "residctl")

# already on disk, for pairing
PV2 = ("abmil_v5fmphikonv2_fm_phikonv2_results.csv", CFG)          # 0.7930
PV2_ADD = ("abmil_v5fmphikonv2_fm_addmil_results.csv", ADD)        # 0.7848
CLEAN = ("abmil_v5nucchromeos_clean_final_results.csv", ADD)       # 0.6603


# plumbing
def sh(cmd, log_name):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    out, err = LOGS / f"{log_name}.log", LOGS / f"{log_name}.err.log"
    with open(out, "w") as fo, open(err, "w") as fe:
        rc = subprocess.call([str(c) for c in cmd], cwd=str(SCRIPTS),
                             stdout=fo, stderr=fe)
    print(f"  log -> {out.name}   exit {rc} after "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return rc


def results_path(tag, arm):
    return ODIR / f"abmil_{tag}_{arm}_results.csv"


def cli_tag(tag, extra):
    """train_abmil.py rewrites --tag itself. The nuc/chrom/eos suffixes are
    guarded by `if "nuc" not in tag`, so they are idempotent -- but the fm one
    (line ~1549) is NOT: it appends `fm<name>` unconditionally whenever
    --fm-features is passed. Handing it a tag that already carries the suffix
    yields v5fmuni2h -> v5fmuni2hfmuni2h, every path computed here misses, and
    the run is scored as a failure despite exiting 0. That is exactly what
    stops the chain after Stage S.

    Callers pass the tag they want to see ON DISK; this returns the stem that
    produces it, so the two conventions can never drift apart again."""
    if "--fm-features" not in extra:
        return tag
    name = extra[extra.index("--fm-features") + 1]
    suffix = f"fm{name}" + ("cat" if "--fm-concat" in extra else "")
    if not tag.endswith(suffix):
        raise ValueError(
            f"tag {tag!r} must end with {suffix!r} -- train_abmil.py will "
            f"append it, so the on-disk name would be {tag + suffix!r}")
    return tag[: -len(suffix)]


def n_expected(config):
    """Cells a finished arm must have: one config x REPEATS x 5 folds."""
    return int(REPEATS) * 5


def train(tag, arm, extra, config=CFG, log=None, rebuild=False):
    """`tag` is the EFFECTIVE (on-disk) tag; see cli_tag(). Skip if this exact
    (tag, arm) is already COMPLETE -- the chain is restartable, and a re-run
    would burn 1.5 h to reproduce a CSV.

    Completeness is row-counted, not existence-checked. The original version
    skipped on `p.exists()` alone, which is only safe if every run that ever
    wrote a CSV also finished. It does not: train_abmil.py writes the CSV after
    EVERY fold, so a crash, an OOM kill or a Ctrl-C leaves a valid-looking file
    holding 6 of 50 cells -- and the next pass would skip it and report the arm
    as done. train_abmil.py resumes from a partial CSV on its own ("Resuming: N
    runs already complete"), so the correct move on a short file is to re-enter
    it and let it fill in the gaps, never to skip it.
    """
    p = results_path(tag, arm)
    if p.exists():
        try:
            have = len(pd.read_csv(p))
        except Exception:
            have = 0
        want = n_expected(config)
        if have >= want:
            print(f"  SKIP {p.name} (complete, {have}/{want})", flush=True)
            return 0
        print(f"  RESUME {p.name} ({have}/{want} cells -- partial, "
              f"re-entering rather than skipping)", flush=True)
    cmd = [PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"), "--tag", cli_tag(tag, extra), "--arm", arm,
           "--repeats", REPEATS, "--epochs", EPOCHS, "--only", config,
           "--select", "swatop", *extra]
    if rebuild:
        # mandatory whenever the on-disk feature corpus changed: SCHEMA_VERSION
        # guards loader-CODE changes only, never the appearance of new npz.
        cmd.append("--rebuild-cache")
    return sh(cmd, log or f"me_{tag}_{arm}")


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
    a, b = cells(A), cells(B)
    if a is None or b is None:
        print(f"  {label:52s} MISSING "
              f"(a={'ok' if a is not None else 'none'}, "
              f"b={'ok' if b is not None else 'none'})", flush=True)
        return None
    i = a.index.intersection(b.index)
    if not len(i):
        print(f"  {label:52s} NO SHARED CELLS", flush=True)
        return None
    a, b = a.loc[i], b.loc[i]
    d = (a - b).mean()
    p = stats.ttest_rel(a, b).pvalue
    try:
        w = stats.wilcoxon(a, b).pvalue
    except Exception:
        w = float("nan")
    print(f"  {label:52s} {a.mean():.4f} vs {b.mean():.4f}  delta {d:+.4f}  "
          f"p={p:.4f} wil={w:.4f} win={((a - b) > 0).mean():.2f} n={len(i)}",
          flush=True)
    return d, p


def banner(t):
    print("\n" + "=" * 82 + f"\n{t}\n" + "=" * 82, flush=True)


def n_files(pat):
    return len(list(FMDIR.glob(pat)))


def wait_for_embeddings(timeout_h):
    """Block until all three encoders have embedded every graph."""
    banner("STAGE 0 -- wait for the embedding extraction")
    t0 = time.time()
    while True:
        counts = {m: n_files(f"*_{m}.npz") for m, _ in ENCODERS}
        done = all(v >= N_GRAPHS for v in counts.values())
        el = (time.time() - t0) / 3600
        print(f"  [{el:5.2f}h] " + "  ".join(
            f"{m}={counts[m]}/{N_GRAPHS}" for m, _ in ENCODERS), flush=True)
        if done:
            print("  all encoders complete.", flush=True)
            return True
        if el > timeout_h:
            print(f"  TIMEOUT after {timeout_h}h -- proceeding with whatever "
                  f"is complete; incomplete encoders will be skipped.",
                  flush=True)
            return False
        time.sleep(600)


def complete(model):
    return n_files(f"*_{model}.npz") >= N_GRAPHS


# STAGE S -- smoke, and it is not optional
def stage_s(models):
    """One short arm per encoder before ~30 h of training is committed.

    Any new or edited training path is smoke-tested first. These encoders are new
    columns of a new width (1536 vs Phikon-v2's 1024), so the loader, the
    normaliser and the degenerate-column sweep are all seeing something they
    have not seen.

    It shares the tag -- and therefore the bag cache -- with Stage A, so the
    2.7 GB cache is built ONCE, here, after extraction is complete. That is
    also the safe ordering: a cache built while extraction was still running
    would silently hold a partial corpus.
    """
    banner("STAGE S -- smoke test each encoder before committing the chain")
    ok = []
    for m, nice in models:
        rc = sh([PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"),
                 "--tag", cli_tag(f"v5fm{m}", ["--fm-features", m]),
                 "--arm", "smoke", "--fm-features", m, "--repeats", "1",
                 "--epochs", "8", "--only", CFG, "--select", "swatop",
                 "--rebuild-cache"], f"me_smoke_{m}")
        p = results_path(f"v5fm{m}", "smoke")
        if rc == 0 and p.exists():
            d = pd.read_csv(p)
            print(f"  {nice:12s} OK -- {len(d)} folds, "
                  f"bal_acc {d.balanced_acc.mean():.3f}", flush=True)
            ok.append((m, nice))
        else:
            print(f"  {nice:12s} SMOKE FAILED (exit {rc}) -- this encoder is "
                  f"dropped from the rest of the chain", flush=True)
    return ok


# STAGE A -- accuracy per encoder (the headline comparison)
def stage_a(models):
    banner("STAGE A -- positive control accuracy, one arm per encoder")
    print("  same bags, same folds, same pooling, same selection rule as every\n"
          "  interpretable arm. The ONLY thing that differs is the instance\n"
          "  representation, so any gap is attributable to the features alone.",
          flush=True)
    for m, nice in models:
        # no --rebuild-cache: Stage S built this tag's cache after extraction
        # completed. If A is run alone, pass --rebuild-cache by hand.
        train(f"v5fm{m}", f"fm_{m}", ["--fm-features", m])


# STAGE B -- additive MIL per encoder
def stage_b(models):
    banner("STAGE B -- Additive MIL per encoder (exact per-tile accounting)")
    print("  per-instance per-class contributions that SUM to the bag logit, so\n"
          "  the explanation is an identity rather than an attribution. The\n"
          "  bridge (Stage D) regresses exactly these contributions onto the\n"
          "  named features, so it cannot run without this stage.", flush=True)
    for m, nice in models:
        train(f"v5fm{m}", "fm_addmil", ["--fm-features", m], config=ADD)


# STAGE C -- paired comparisons (free)
def stage_c(models):
    banner("STAGE C -- paired comparisons")
    print("\n  vs Phikon-v2 (ABMIL head, the established baseline):", flush=True)
    for m, nice in models:
        paired((f"abmil_v5fm{m}_fm_{m}_results.csv", CFG), PV2,
               f"{nice} vs Phikon-v2")
    print("\n  vs the clean interpretable model (the audit's gap):", flush=True)
    for m, nice in models:
        paired((f"abmil_v5fm{m}_fm_{m}_results.csv", CFG), CLEAN,
               f"{nice} vs clean interpretable")
    print("\n  additive vs ABMIL head, within encoder (should be ~0):", flush=True)
    for m, nice in models:
        paired((f"abmil_v5fm{m}_fm_addmil_results.csv", ADD),
               (f"abmil_v5fm{m}_fm_{m}_results.csv", CFG),
               f"{nice}: additive vs ABMIL")
    print("\n  encoder vs encoder:", flush=True)
    allm = models + [("phikonv2", "Phikon-v2")]
    for i in range(len(allm)):
        for j in range(i + 1, len(allm)):
            a, an = allm[i]
            b, bn = allm[j]
            paired((f"abmil_v5fm{a}_fm_{a}_results.csv", CFG),
                   (f"abmil_v5fm{b}_fm_{b}_results.csv", CFG),
                   f"{an} vs {bn}")


# STAGE D -- bridge per encoder
def stage_d(models):
    banner("STAGE D -- bridge: how much of each encoder is expressible in "
           "named terms")
    print("  regresses the encoder's exact per-tile per-class contributions onto\n"
          "  the ~49 named pathology columns on the IDENTICAL tile grid\n"
          "  (grid_rc_checksum, not approximate registration), reported against\n"
          "  the self-ceiling. Phikon-v2 scored 38%. Whether that number is a\n"
          "  property of Phikon or of foundation models is the whole question.",
          flush=True)
    for m, nice in models:
        if not results_path(f"v5fm{m}", "fm_addmil").exists():
            print(f"  SKIP {nice} -- Stage B did not produce an additive arm",
                  flush=True)
            continue
        outd = ODIR / f"bridge_fm_interpretable_v5fm{m}"
        if (outd / "r2_by_class.csv").exists():
            print(f"  SKIP {nice} bridge (already on disk)", flush=True)
            continue
        sh([PY, "-u", str(SCRIPTS / "pipeline" / "bridge_fm_interpretable.py"),
            "--fm-tag", f"v5fm{m}_fm_addmil", "--fm-name", nice,
            "--out", str(outd)], f"me_bridge_{m}")


# STAGE E -- subspace split per encoder (the strongest audit experiment)
def stage_e(models):
    banner("STAGE E -- subspace split per encoder")
    print("  splits the embedding into the morphometry-ALIGNED subspace and its\n"
          "  residual, then lets the classifier say which half carries the\n"
          "  accuracy. Two controls, both required: residnull is rank-matched\n"
          "  but aligned with nothing (fitted from MISMATCHED slides), residctl\n"
          "  is variance-matched random PCA. Neither can match rank AND variance\n"
          "  at once, so the honest answer is the BRACKET between them -- on\n"
          "  Phikon-v2 that was 17-52% of its advantage. Quote the bracket.",
          flush=True)
    for m, nice in models:
        have = min(n_files(f"*_{m}{a}.npz") for a in SUB_ARMS)
        if have < N_GRAPHS:
            if sh([PY, "-u", str(SCRIPTS / "pipeline" / "fm_subspace_split.py"), "--model", m],
                  f"me_split_{m}") != 0:
                print(f"  split FAILED for {nice}; skipping its arms",
                      flush=True)
                continue
        else:
            print(f"  {nice}: decomposition already on disk ({have} slides/arm)",
                  flush=True)
        for a in SUB_ARMS:
            train(f"v5fm{m}{a}", f"fm_{a}", ["--fm-features", f"{m}{a}"],
                  config=ADD, rebuild=True)

    banner("STAGE E RESULTS -- paired")
    for m, nice in models:
        print(f"\n  {nice}:", flush=True)
        R = {a: (f"abmil_v5fm{m}{a}_fm_{a}_results.csv", ADD) for a in SUB_ARMS}
        FMA = (f"abmil_v5fm{m}_fm_addmil_results.csv", ADD)
        paired(R["resid"], R["residnull"], "resid vs residnull (THE TEST)")
        paired(R["resid"], R["residctl"], "resid vs residctl (variance-matched)")
        paired(R["resid"], FMA, "resid vs full encoder")
        paired(R["residnull"], FMA, "residnull vs full encoder (sanity: ~0)")
        paired(R["resid"], CLEAN, "resid vs clean interpretable")
        paired(FMA, CLEAN, "full encoder vs clean interpretable")


# STAGE F -- hybrid per encoder
def stage_f(models):
    banner("STAGE F -- hybrid: does the named vocabulary ADD to each encoder?")
    print("  on Phikon-v2 it did NOT (0.7836 vs 0.7930, -0.0094, p=0.046), which\n"
          "  §19.4 later explained mechanically: the embedding already contains\n"
          "  that subspace, so concatenation appends redundant dimensions and\n"
          "  pays the variance cost of a wider input. If that explanation is\n"
          "  right, the same small negative should appear on all four encoders.",
          flush=True)
    for m, nice in models:
        train(f"v5nucfm{m}cat", "fm_hybrid",
              ["--fm-features", m, "--fm-concat", "--nuclear-features",
               "--drop-groups", "graph"], rebuild=True)
    for m, nice in models:
        paired((f"abmil_v5nucfm{m}cat_fm_hybrid_results.csv", CFG),
               (f"abmil_v5fm{m}_fm_{m}_results.csv", CFG),
               f"{nice}: hybrid vs encoder alone")


# STAGE G -- ensembles and seed stability per encoder
def stage_g(models):
    banner("STAGE G -- ensembles and seed stability per encoder")
    print("  seed disagreement is a difficulty detector, not just noise (§18),\n"
          "  and it funds the free abstention rule. Measuring it on four encoders\n"
          "  tests whether that is a property of the corpus or of the features.",
          flush=True)
    for m, nice in models:
        if not results_path(f"v5fm{m}", "fm_addmil").exists():
            print(f"  SKIP {nice} -- no additive arm", flush=True)
            continue
        for what in ("ensemble", "attnensemble", "stability"):
            sh([PY, "-u", str(SCRIPTS / "analysis" / "analyse_abmil_nuclear.py"), "--what", what,
                "--tag", f"v5fm{m}_fm_addmil", "--config", ADD,
                "--repeats", REPEATS, "--no-nuclear"],
               f"me_{m}_{what}")


STAGES = {"S": stage_s, "A": stage_a, "B": stage_b, "C": stage_c,
          "D": stage_d, "E": stage_e, "F": stage_f, "G": stage_g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=list("SABCDEFG"),
                    choices=sorted(STAGES))
    ap.add_argument("--models", nargs="+", default=[m for m, _ in ENCODERS])
    ap.add_argument("--timeout-h", type=float, default=12.0,
                    help="how long to wait for the embedding extraction")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = [(m, n) for m, n in ENCODERS if m in args.models]
    print("=" * 82)
    print("MULTI-ENCODER AUDIT -- replicating the Phikon-v2 programme")
    print("=" * 82)
    print(f"  encoders : {[n for _, n in models]}")
    print(f"  stages   : {args.stages}")
    print(f"  heads    : {CFG} (features) / {ADD} (additive), swatop, "
          f"{REPEATS} repeats x 5 folds", flush=True)
    if args.dry_run:
        return

    if not args.no_wait:
        wait_for_embeddings(args.timeout_h)
    ready = [(m, n) for m, n in models if complete(m)]
    if len(ready) < len(models):
        missing = [n for m, n in models if not complete(m)]
        print(f"\n  !! incomplete embeddings, skipping: {missing}", flush=True)
    if not ready:
        print("  nothing to do.", flush=True)
        return

    t0 = time.time()
    for s in args.stages:
        try:
            out = STAGES[s](ready)
            if s == "S":
                # an encoder that cannot even train one short fold must not
                # consume hours of the chain producing unusable checkpoints
                ready = out
                if not ready:
                    print("\n  !! every encoder failed its smoke test -- "
                          "stopping.", flush=True)
                    return
        except Exception as e:
            # one stage failing must not take the rest of an overnight chain
            # down with it
            print(f"\n  !! STAGE {s} raised {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            print(f"\n  !! STAGE {s} raised {type(e).__name__}: {e} -- "
                  f"continuing", flush=True)
    print(f"\nALL DONE in {(time.time() - t0) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()

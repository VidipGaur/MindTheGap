# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Does Phikon-v2's accuracy live in the part our vocabulary can describe?

Builds the decomposition (fm_subspace_split.py), then trains Additive MIL on
each residual corpus under the identical protocol as the plain Phikon-v2 arm --
10 repeats x 5 folds, swatop selection, addmil_h128_do5 -- so every comparison
is paired on shared (repeat, fold) cells.

  fm_resid      E - (morphometry-fitted rank-50 map)
  fm_residnull  E - (same map fitted from MISMATCHED slides)   <- primary ctl
  fm_residctl   E - (variance-matched random PCA subspace)     <- second ctl

Reads the pre-registered rule out of fm_subspace_split.py's docstring; the
verdict is printed rather than decided here, because the interesting outcome
(resid ~= null) is a null result and a driver that "picks a winner" would
misrepresent it.
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
ADD = "addmil_h128_do5"
ABM = "abmil_h128_lr1"
ARMS = ["resid", "residnull", "residctl"]


def sh(cmd, log):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    with open(LOGS / f"{log}.log", "w") as fo, open(LOGS / f"{log}.err.log", "w") as fe:
        rc = subprocess.call([str(c) for c in cmd], cwd=SCRIPTS, stdout=fo, stderr=fe)
    print(f"  log -> {log}.log   exit {rc} after {(time.time()-t0)/60:.1f} min",
          flush=True)
    return rc


def cells(f, cfg):
    p = ODIR / f
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.config == cfg]
    if not len(d):
        return None
    s = d.set_index(["repeat", "fold"])["balanced_acc"]
    return s[~s.index.duplicated()]


def paired(A, B, label):
    a, b = cells(*A), cells(*B)
    if a is None or b is None:
        print(f"  {label:46s} MISSING ({'A' if a is None else 'B'})")
        return None
    i = a.index.intersection(b.index)
    a, b = a.loc[i], b.loc[i]
    d = a - b
    p = stats.ttest_rel(a, b).pvalue
    print(f"  {label:46s} {a.mean():.4f} vs {b.mean():.4f}  d={d.mean():+.4f}  "
          f"p={p:.4f} win={(d > 0).mean():.2f} n={len(i)}", flush=True)
    return d.mean(), p


def main():
    print("=" * 82)
    print("SUBSPACE SPLIT -- is Phikon-v2's advantage inside our vocabulary?")
    print("=" * 82)

    fmdir = ROOT / "cellvit_run" / "graphs" / "fm_feats"
    have = min(len(list(fmdir.glob(f"*_phikonv2{a}.npz"))) for a in ARMS)
    if have >= 456:
        print(f"  decomposition already on disk ({have} slides per arm); "
              f"skipping the split.", flush=True)
    elif sh([PY, "-u", str(SCRIPTS / "pipeline" / "fm_subspace_split.py")], "p6_split") != 0:
        print("  decomposition failed; stopping.")
        return 1

    for arm in ARMS:
        # --rebuild-cache is mandatory: the on-disk feature corpus just changed,
        # and SCHEMA_VERSION only guards loader-code changes, not new npz files.
        sh([PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"), "--tag", "v5", "--arm", f"fm_{arm}",
            "--fm-features", f"phikonv2{arm}", "--rebuild-cache",
            "--repeats", "10", "--epochs", "120", "--only", ADD,
            "--select", "swatop"], f"p6_fm_{arm}")

    PV2A = ("abmil_v5fmphikonv2_fm_addmil_results.csv", ADD)
    PV2 = ("abmil_v5fmphikonv2_fm_phikonv2_results.csv", ABM)
    CLEAN = ("abmil_v5nucchromeos_clean_final_results.csv", ADD)
    R = {a: (f"abmil_v5fmphikonv2{a}_fm_{a}_results.csv", ADD) for a in ARMS}

    print("\n" + "=" * 82)
    print("PAIRED RESULTS")
    print("=" * 82)
    real = paired(R["resid"], R["residnull"], "resid vs residnull (THE TEST)")
    paired(R["resid"], R["residctl"], "resid vs residctl (variance-matched)")
    paired(R["resid"], PV2A, "resid vs full Phikon-v2 (additive)")
    paired(R["residnull"], PV2A, "residnull vs full Phikon-v2 (sanity: ~0)")
    paired(R["resid"], CLEAN, "resid vs clean interpretable")
    paired(PV2A, CLEAN, "full Phikon-v2 vs clean interpretable")
    paired(PV2A, PV2, "additive vs abmil on Phikon-v2 (context)")

    print("\n" + "=" * 82)
    print("VERDICT against the pre-registered rule")
    print("=" * 82)
    if real is None:
        print("  cannot read the primary comparison.")
    elif real[1] >= 0.10:
        print(f"  resid == residnull (d={real[0]:+.4f}, p={real[1]:.3f}).")
        print("  Removing the morphometry-aligned subspace costs no more than")
        print("  removing a matched uninformative map. Phikon's advantage does")
        print("  NOT live in what the named vocabulary describes -- the named")
        print("  features are a shadow of the embedding, not its engine.")
        print("  This is the honest form of the audit finding, and it makes the")
        print("  38%-of-ceiling bridge number a statement about correlation")
        print("  only, with no claim to sufficiency.")
    elif real[0] < 0:
        print(f"  resid < residnull (d={real[0]:+.4f}, p={real[1]:.3f}).")
        print("  Deleting morphometry costs real accuracy beyond the matched")
        print("  control. The gap is the share of Phikon's advantage our")
        print("  vocabulary accounts for; report it as that share, and pair it")
        print("  with the bridge R2 as two views of the same claim.")
    else:
        print(f"  resid > residnull (d={real[0]:+.4f}, p={real[1]:.3f}) --")
        print("  removing morphometry HELPED. Treat with suspicion: check the")
        print("  control's rank/variance match before reporting anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

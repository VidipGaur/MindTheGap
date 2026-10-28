# Copyright 2026 Vidip Gaur, Vidu Gaur
"""The genuinely clean interpretable headline -- the one number the paper needs.

run_phase4.py's stage_i() carried `clean_all` forward because its rule only
switched to `clean_noctx` if that arm SCORED HIGHER, which contradicts its own
docstring ("carry the arm that survives the confound controls, not the higher
number"). `clean_noctx` scores 0.0104 lower at p=0.055 -- that is the COST of
removing a suspected confound, not a reason to keep it. So the Additive MIL run
in Stage I inherited `chrom_context`, the tile-level staining descriptors that
have never been put through the test that convicted dead_count.

This runs Additive MIL on the fully controlled feature set:
  nuclear + chromatin(no texture) + eosin(clearing only)
  minus dead_          (convicted staining artefact)
  minus chrom_context  (untested staining descriptors, removed on suspicion)

and pairs it against both the ABMIL version of the same set and the partly-clean
Additive arm it replaces, so the cost of each control is separately visible.
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
DROP = ["graph", "chrom_texture", "eos_other", "chrom_context"]


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
    return d.set_index(["repeat", "fold"])["balanced_acc"] if len(d) else None


def paired(A, B, label):
    a, b = cells(*A), cells(*B)
    if a is None or b is None:
        print(f"  {label:44s} MISSING")
        return
    i = a.index.intersection(b.index)
    a, b = a.loc[i], b.loc[i]
    d = a - b
    print(f"  {label:44s} {a.mean():.4f} vs {b.mean():.4f}  d={d.mean():+.4f}  "
          f"p={stats.ttest_rel(a, b).pvalue:.4f} win={(d > 0).mean():.2f} n={len(i)}",
          flush=True)


def main():
    print("=" * 78)
    print("CLEAN FINAL -- Additive MIL, dead_ AND chrom_context both removed")
    print("=" * 78)
    sh([PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"), "--tag", "v5nuc", "--arm", "clean_final",
        "--nuclear-features", "--chromatin-features", "--eosin-features",
        "--exclude-prefix", "dead_", "--drop-groups", *DROP,
        "--repeats", "10", "--epochs", "120", "--only", ADD,
        "--select", "swatop"], "p5_clean_final")

    FIN = ("abmil_v5nucchromeos_clean_final_results.csv", ADD)
    print("\n  paired:", flush=True)
    paired(FIN, ("abmil_v5nucchromeos_clean_noctx_results.csv", "abmil_h128_lr1"),
           "clean_final Additive vs its ABMIL")
    paired(FIN, ("abmil_v5nucchromeos_clean_additive_results.csv", ADD),
           "clean_final vs clean_additive (ctx cost)")
    paired(FIN, ("abmil_v5nuc_gbm_results.csv", "logreg"),
           "clean_final vs slide-level logreg")
    paired(FIN, ("abmil_v5fmphikonv2_fm_phikonv2_results.csv", "abmil_h128_lr1"),
           "clean_final vs Phikon-v2 (the honest gap)")

    for what in ("ensemble", "stability"):
        sh([PY, "-u", str(SCRIPTS / "analysis" / "analyse_abmil_nuclear.py"), "--what", what,
            "--tag", "v5nucchromeos_clean_final", "--config", ADD,
            "--repeats", "10", "--drop-groups", *DROP,
            "--exclude-prefix", "dead_"], f"p5_clean_final_{what}")


if __name__ == "__main__":
    sys.exit(main())

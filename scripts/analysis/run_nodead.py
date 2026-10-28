# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Re-baseline the model with the convicted `dead_count` features removed.

resolve_dead_count.py established that `dead_count` is a staining/scanner
artefact (rho=-0.42 with tissue green channel; staining alone out-predicts it;
staining-only clusters reproduce both it and the label). Every accuracy number
in the project still contains it, so nothing can be written up until this is
measured at the current protocol.

Two arms, both 10 repeats x 5 folds under swatop at abmil_h128_lr1, each paired
against its own dead-containing counterpart on shared (repeat, fold) cells:

  nodead10                -> vs abmil_v5nuc_sel_swatop        (0.6550)
  chrom_notexture_nodead  -> vs abmil_v5nucchrom_chrom_notexture (0.6810)

Note what this does and does not remove. It drops the two explicit necrosis
columns (`dead_count`, `dead_frac`) from the model's inputs. The necrotic nuclei
themselves are still counted in `nuc_count` and still contribute to the `all_*`
morphometric aggregates, because those pool over every detection regardless of
type. So this is the cheap version of the fix and its result is an optimistic
floor -- if the drop here is materially smaller than the 3-repeat estimate of
0.051, the strict version (excluding `dead` detections from the `all_*`
aggregates, which needs a re-extraction) is worth running.
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
PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent.parent
CONFIG = "abmil_h128_lr1"

ARMS = [
    # (arm name, results-file tag, extra flags, baseline file, baseline label)
    ("nodead10", "v5nuc",
     ["--nuclear-features", "--drop-groups", "graph"],
     "abmil_v5nuc_sel_swatop_results.csv", "nuclear-only (0.6550)"),
    ("chrom_notexture_nodead", "v5nucchrom",
     ["--nuclear-features", "--chromatin-features",
      "--drop-groups", "graph", "chrom_texture"],
     "abmil_v5nucchrom_chrom_notexture_results.csv",
     "nuclear + chromatin, no texture (0.6810)"),
]


def run(arm, extra, log_name):
    cmd = [PY, "-u", str(SCRIPTS / "pipeline" / "train_abmil.py"),
           "--tag", "v5nuc", *extra,
           "--exclude-prefix", "dead_",
           "--arm", arm,
           "--repeats", "10", "--epochs", "120",
           "--only", CONFIG, "--select", "swatop"]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    out = LOGS / f"{log_name}.log"
    err = LOGS / f"{log_name}.err.log"
    with open(out, "w") as fo, open(err, "w") as fe:
        rc = subprocess.call(cmd, cwd=SCRIPTS, stdout=fo, stderr=fe)
    print(f"  log -> {out.name}   exit {rc} after {(time.time()-t0)/60:.1f} min",
          flush=True)
    return rc


def cells(fname):
    p = ODIR / fname
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.config == CONFIG]
    return d.set_index(["repeat", "fold"])["balanced_acc"] if len(d) else None


def compare(nodead_file, base_file, label):
    a, b = cells(nodead_file), cells(base_file)
    if a is None or b is None:
        print(f"  cannot compare {label}: missing results")
        return
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    d = a - b
    p = stats.ttest_rel(a, b).pvalue
    try:
        w = stats.wilcoxon(a, b).pvalue
    except Exception:
        w = float("nan")
    print(f"\n  {label}")
    print(f"    without dead_ : {a.mean():.4f}")
    print(f"    with dead_    : {b.mean():.4f}")
    print(f"    cost of removing it: {-d.mean():+.4f}  "
          f"p={p:.4f}  wilcoxon={w:.4f}  n={len(idx)}")
    print(f"    (3-repeat estimate was 0.051; smaller here means the cheap fix "
          f"understates and the strict re-extraction is worth running)")


def main():
    print("=" * 78)
    print("dead_count RE-BASELINE -- 2 arms x 10 repeats x 5 folds, swatop")
    print("=" * 78)
    t0 = time.time()
    for arm, tag, extra, base, label in ARMS:
        run(arm, extra, f"nodead_{arm}")
        compare(f"abmil_{tag}_{arm}_results.csv", base, label)
    print(f"\nTOTAL {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())

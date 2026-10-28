# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Regenerate the per-slide explanations on the FINAL model, queued behind
the bridge refit.

WHY THIS EXISTS
---------------
The per-subtype driver profiles currently in the record (Task 3c(d)) were
produced from `v5nuc_nuconly` -- a nuclear-only ABMIL whose top EC driver is
`dead_count`, the necrosis feature later convicted as a staining/batch
artefact (Task 8). Those profiles cannot go in the paper: the single most
striking claim in them ("EC rests on necrotic nuclei on 76.3% of correct
calls") is a statement about fixation quality, not biology.

The final model, `v5nucchromeos_clean_final`, is Additive MIL trained with
`--exclude-prefix dead_` and with the chrom_context tile-staining descriptors
dropped as well, so both convicted blocks are gone. Re-running the explanation
layer on it is a REGENERATION of an existing analysis on the correct
checkpoint, not a new experiment.

Two things had to be fixed before this could run at all, both recorded here
because the numbers are only trustworthy given them:

1. `explain_slide.py` could load only the nuclear block, so it could not
   assemble the final model's 49-column feature set and aborted on the
   feature-count guard. It now takes --chromatin-features / --eosin-features /
   --exclude-prefix, mirroring the arm's own training flags.

2. `resolve_groups()` located the eosin block by a single anchor column,
   `tile_e_mean`, which is itself an eos_other column -- so the final model's
   `--drop-groups eos_other` deleted the anchor, the chromatin block's range
   expanded to swallow the surviving eosin columns, and the three clearing
   features (tile_clear_frac / tile_pale_frac / tile_region_frac) were
   reported under `chrom_context`. That is the group this very model is
   documented as having REMOVED, and clearing is the CC evidence proper, so a
   driver table built on it would have credited CC's diagnostic signal to
   chromatin context. Fixed by falling back to a name-set search when an
   anchor is missing. Training was never affected -- the group filter runs on
   the unfiltered names -- only analysis-time relabelling was.

Upgrade worth noting in the writeup: the final model is Additive MIL, so the
per-tile numbers in these explanations are EXACT contributions summing to the
bag logit, not attributions. The occlusion-based feature ranking is still an
attribution and still carries the Task 3c(c) caveat (it measures what this
trained model leans on, not what information is necessary).

USAGE
    python scripts/dev/queue_explain_final.py                 # wait for the refit, then run
    python scripts/dev/queue_explain_final.py --wait-pid 0    # run now
    python scripts/dev/queue_explain_final.py --n 25          # a subset, for a spot check
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
import time
from pathlib import Path
import sys
import corpus_paths

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent.parent
ROOT = corpus_paths.ROOT
LOGS = ROOT / "cellvit_run" / "graphs" / "gatv2_run" / "logs"

TAG = "v5nucchromeos_clean_final"
CFG = "addmil_h128_do5"
# must mirror run_cleanfinal.py exactly; explain_slide.py hard-fails otherwise
DROP = ["graph", "chrom_texture", "eos_other", "chrom_context"]
CSV = "explanations_clean_final.csv"


def wait_for(pid: int, poll: int = 120) -> None:
    if not pid:
        return
    print(f"[queue] waiting for pid {pid} ...", flush=True)
    t0 = time.time()
    while True:
        if os.name == "nt":
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                               capture_output=True, text=True)
            gone = str(pid) not in r.stdout
        else:
            gone = subprocess.run(["ps", "-p", str(pid)],
                                  capture_output=True).returncode != 0
        if gone:
            print(f"[queue] pid {pid} gone after {(time.time()-t0)/60:.0f} min",
                  flush=True)
            return
        time.sleep(poll)


def sh(cmd, log_name: str) -> int:
    log = LOGS / f"{log_name}.log"
    print(f"\n$ {' '.join(cmd)}\n  log -> {log.name}", flush=True)
    t0 = time.time()
    # separate handles: merged output through PowerShell is fully buffered
    with open(log, "w", encoding="utf-8") as fo, \
         open(log.with_suffix(".err.log"), "w", encoding="utf-8") as fe:
        rc = subprocess.run(cmd, stdout=fo, stderr=fe, cwd=str(SCRIPTS)).returncode
    print(f"  exit {rc} after {(time.time()-t0)/60:.1f} min", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=28360,
                    help="pid of the bridge refit; 0 to start immediately")
    ap.add_argument("--n", type=int, default=0,
                    help="explain N slides instead of all 456 (spot check)")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    wait_for(args.wait_pid)

    cmd = [PY, "-u", str(SCRIPTS / "analysis" / "explain_slide.py"),
           "--tag", TAG, "--config", CFG, "--repeats", "10",
           "--chromatin-features", "--eosin-features",
           "--exclude-prefix", "dead_", "--drop-groups", *DROP,
           "--out", CSV]
    cmd += ["--n", str(args.n)] if args.n else ["--all"]

    if sh(cmd, "explain_clean_final") != 0:
        print("explain_slide FAILED -- not running the summary", flush=True)
        return

    sh([PY, "-u", str(SCRIPTS / "dev" / "summarise_explanations.py"), "--csv", CSV],
       "summarise_clean_final")
    print("\nDone. Per-subtype driver profiles are in "
          "summarise_clean_final.log; these supersede the Task 3c(d) table, "
          "which was built on the dead_count model.", flush=True)


if __name__ == "__main__":
    main()

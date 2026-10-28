# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Per-repeat refit of the bridge R^2, queued behind the running chain.

WHY THIS EXISTS
---------------
Stage D produced one bridge number per encoder: UNI2-h 59%, H-optimus-0 52%,
Hibou-L 41%, Phikon-v2 38% of the interpretable model's self-ceiling. That
looks like an ordering -- stronger encoders are MORE nameable -- and it is
tempting to report it as one. It cannot be reported as one, because each of
those numbers is a single fit over contributions averaged across all 10
repeats. There is no error bar, so there is no way to distinguish a real
between-encoder difference from the noise of re-seeding the Additive MIL model
that produced the contributions in the first place. Per-cell accuracy sd on
this corpus is ~0.05; assuming the bridge is tighter than that is an assumption
nobody has tested.

So: run the whole bridge measurement once per repeat, per encoder. Each repeat
scores every slide out-of-fold exactly once, so a single-repeat estimate is
honest, just noisier. Ten of them give a distribution, and the distribution is
what decides whether the ordering survives.

The comparison is PAIRED on repeat index. Repeat r uses the same fold
assignment (StratifiedKFold(5, random_state=r)) for every encoder, so the r-th
estimates share their train/test split and their slide-level luck. Pairing
removes that shared variance, which is the same reason every accuracy
comparison in this project is paired on (repeat, fold) -- per-cell sd dwarfs
the effects being measured.

WHAT IT DOES NOT SETTLE
-----------------------
There is a second explanation for the spread that error bars cannot rule out:
a more accurate model produces cleaner, more concentrated per-tile
contributions, and cleaner regression targets are easier to fit onto anything.
The self-ceiling normalisation was meant to absorb this, but the ceiling is
computed on the INTERPRETABLE model's contributions, not the encoder's, so it
does not control for target noise in the FM arm. If the ordering does survive,
that confound still has to be stated -- it is a limitation, not a bug.

USAGE
-----
    python scripts/pipeline/queue_bridge_refit.py                  # wait for the chain, then run
    python scripts/pipeline/queue_bridge_refit.py --wait-pid 0     # run immediately
    python scripts/pipeline/queue_bridge_refit.py --smoke          # 1 encoder x 2 repeats
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
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import corpus_paths

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent.parent
ROOT = corpus_paths.ROOT
RUN = ROOT / "cellvit_run" / "graphs" / "gatv2_run"
LOGS = RUN / "logs"

# (display name, additive-arm tag). Order is the Stage D reporting order.
ENCODERS = [
    ("UNI2-h",      "v5fmuni2h_fm_addmil"),
    ("H-optimus-0", "v5fmhoptimus0_fm_addmil"),
    ("Hibou-L",     "v5fmhibouL_fm_addmil"),
    ("Phikon-v2",   "v5fmphikonv2_fm_addmil"),
]

RESULT_RE = re.compile(
    r"BRIDGE_REPEAT_RESULT encoder=(\S+) repeat=(\S+) "
    r"median_pct_ceiling=([-\d.naN]+)")


def wait_for(pid: int, poll: int = 120) -> None:
    """Block until `pid` is gone. The chain holds the GPU; running the refit
    alongside it would not corrupt anything (this is read-only inference) but
    would slow both and make the timings unreadable."""
    if not pid:
        return
    print(f"[queue] waiting for pid {pid} to finish ...", flush=True)
    t0 = time.time()
    while True:
        if os.name == "nt":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
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


def one_run(name: str, tag: str, rep: int, slides: int = 0) -> float:
    """One bridge fit on one repeat. Returns the median %-of-ceiling, or nan."""
    out = RUN / f"bridge_repeats_{tag.split('_')[0]}"
    log = LOGS / f"bridgerep_{tag.split('_')[0]}_r{rep}.log"
    cmd = [PY, "-u", str(SCRIPTS / "pipeline" / "bridge_fm_interpretable.py"),
           "--fm-tag", tag, "--fm-name", name,
           "--repeat-only", str(rep), "--brief", "--out", str(out)]
    if slides:
        cmd += ["--slides", str(slides)]
    t0 = time.time()
    # Separate handles: merging through PowerShell fully buffers stdout and
    # hides warnings until the run ends.
    with open(log, "w", encoding="utf-8") as fo, \
         open(log.with_suffix(".err.log"), "w", encoding="utf-8") as fe:
        subprocess.run(cmd, stdout=fo, stderr=fe, cwd=str(SCRIPTS))
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = RESULT_RE.search(txt)
    val = float(m.group(3)) if m else float("nan")
    print(f"  {name:12s} r{rep}  median={val:6.1f}%   "
          f"({(time.time()-t0)/60:.1f} min)  log -> {log.name}", flush=True)
    return val


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("BRIDGE R^2 ACROSS REPEATS -- does the encoder ordering survive?")
    print("=" * 78)
    piv = df.pivot(index="repeat", columns="encoder", values="median_pct")
    piv = piv[[n for n, _ in ENCODERS if n in piv.columns]]
    print("\nper-repeat median %-of-ceiling")
    print(piv.round(1).to_string())

    print("\n" + "-" * 78)
    print(f"  {'encoder':13s} {'mean':>7s} {'sd':>7s} {'min':>7s} {'max':>7s} "
          f"{'n':>4s}")
    for c in piv.columns:
        v = piv[c].dropna()
        print(f"  {c:13s} {v.mean():7.1f} {v.std():7.1f} {v.min():7.1f} "
              f"{v.max():7.1f} {len(v):4d}")

    print("\n" + "-" * 78)
    print("PAIRED on repeat index (same fold assignment => shared split luck).")
    print("An unpaired comparison at this sd would be worthless.")
    cols = list(piv.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = piv[cols[i]], piv[cols[j]]
            ok = a.notna() & b.notna()
            if ok.sum() < 3:
                continue
            d = (a[ok] - b[ok])
            t, p = stats.ttest_rel(a[ok], b[ok])
            try:
                _, pw = stats.wilcoxon(a[ok], b[ok])
            except ValueError:
                pw = float("nan")
            flag = "SIGNIF" if p < 0.05 else ("weak" if p < 0.10 else "n.s.")
            print(f"  {cols[i]:13s} - {cols[j]:13s}  "
                  f"delta={d.mean():+6.1f}  sd={d.std():5.1f}  "
                  f"p={p:.3f}  wilcoxon={pw:.3f}  win={np.mean(d>0):.2f}  {flag}")

    print("\n" + "=" * 78)
    print("READING IT: if the pairwise tests are n.s., the 38-59% spread is")
    print("re-seeding noise and the paper reports a RANGE, not an ordering.")
    print("If they are significant, the ordering is real -- and the")
    print("target-noise confound in this file's docstring must then be stated")
    print("explicitly, because it predicts the same ordering for a reason that")
    print("has nothing to do with nameability.")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=8180,
                    help="pid of the running chain; 0 to start immediately")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="1 encoder x 2 repeats x 40 slides, to prove the "
                         "plumbing before committing ~2h of compute")
    ap.add_argument("--out", default=str(RUN / "bridge_repeat_sweep.csv"))
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    encoders = ENCODERS[:1] if args.smoke else ENCODERS
    reps = range(2) if args.smoke else range(args.repeats)
    slides = 40 if args.smoke else 0

    wait_for(args.wait_pid)

    print("=" * 78)
    print(f"BRIDGE REFIT -- {len(encoders)} encoders x {len(list(reps))} repeats")
    print("=" * 78, flush=True)

    rows = []
    for name, tag in encoders:
        for rep in reps:
            rows.append(dict(encoder=name, tag=tag, repeat=rep,
                             median_pct=one_run(name, tag, rep, slides)))
            pd.DataFrame(rows).to_csv(args.out, index=False)   # crash-safe

    df = pd.DataFrame(rows)
    print(f"\nwrote -> {args.out}")
    if df["median_pct"].notna().sum() == 0:
        print("ERROR: every run returned nan -- check the per-run logs")
        sys.exit(1)
    report(df)


if __name__ == "__main__":
    main()

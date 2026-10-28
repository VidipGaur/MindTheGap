# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Per-class recall and full confusion counts, on BOTH corpora, from frozen
checkpoints. Storage only -- nothing here is computed for a specific claim.

WHY THIS EXISTS
===============
The per-fold result CSVs record `balanced_acc` and nothing else, so the corpus
has never had a per-class breakdown on the INTERNAL folds. The external run
(eval_external.py) prints one, but only for the 50-checkpoint ensemble, and
only into a log and instrument1.json. This script produces the per-class layer
for both corpora in one reusable long-format pair of CSVs.

No model is retrained. Every number comes from a checkpoint that already
exists, scored exactly the way it was scored at training time.

WHAT IS SCORED
==============
The five Instrument-1 arms (named morphometry + the four encoders), which are
the only arms that exist on both corpora. The subspace-deletion arms are
internal-only -- they need a ridge map fitted on the corpus under test -- so
they are deliberately out of scope here, same reasoning as eval_external.py.

  internal  each checkpoint r{r}f{f} scores its OWN held-out UBC-OCEAN fold,
            reproducing StratifiedKFold(5, shuffle=True, random_state=r) over
            the same ordered bag list train_abmil.py used. Three levels are
            emitted: per (repeat, fold) cell; per repeat pooled out-of-fold
            over all 456 slides; and the 10-repeat out-of-fold probability
            ensemble.
  external  all 50 checkpoints score all 80 Ben Taeb slides, as in
            eval_external.py. Two levels: per checkpoint, and the
            50-checkpoint probability ensemble.

CORRECTNESS GUARDS
==================
Per-class numbers are only worth storing if the harness reproduces the scoring
that produced the paper's balanced accuracies. Two checks run automatically and
both are hard failures:

  * INTERNAL: every (repeat, fold) cell's balanced accuracy recomputed here
    must match the value in that arm's results CSV to 1e-4. This is
    eval_external.py's --self-test logic, applied to all five arms rather than
    one, and it is what proves the fold reconstruction is the right one.
  * EXTERNAL: the ensemble per-class recall must match the values already
    stored in bentaeb_run/instrument1.json.

USAGE
=====
    python scripts/pipeline/per_class_numbers.py                 # both corpora, all five arms
    python scripts/pipeline/per_class_numbers.py --only internal
    python scripts/pipeline/per_class_numbers.py --arms clean uni2h
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
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from eval_external import (ARMS, CFGNAME, CKPT_ROOT, align_to_checkpoint,
                           load_external_bags, score_checkpoint)
from train_abmil import CLASSES
import corpus_paths

ROOT = corpus_paths.ROOT
INTERNAL_GDIR = ROOT / "cellvit_run" / "graphs"
EXTERNAL_GDIR = ROOT / "bentaeb_run" / "graphs"
RESULTS_DIR = ROOT / "cellvit_run" / "graphs" / "gatv2_run"
INSTRUMENT1 = ROOT / "bentaeb_run" / "instrument1.json"
OUT_DIR = ROOT / "paper" / "per_class"

TOL = 1e-4


def ckpt_cells(cdir):
    """(repeat, fold, path) for every addmil checkpoint in a tag directory."""
    out = []
    for p in sorted(cdir.glob(f"{CFGNAME}_r*f*.pt")):
        stem = p.stem[len(CFGNAME) + 1:]          # "r3f2"
        r, f = stem[1:].split("f")
        out.append((int(r), int(f), p))
    return sorted(out)


def emit(rows_recall, rows_conf, dataset, arm, level, repeat, fold, y, pred):
    """Append one confusion matrix to both output tables."""
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    support = cm.sum(1)
    correct = cm.diagonal()
    for i, c in enumerate(CLASSES):
        rows_recall.append(dict(
            dataset=dataset, arm=arm["key"], arm_name=arm["nice"], level=level,
            repeat=repeat, fold=fold, cls=c, support=int(support[i]),
            correct=int(correct[i]),
            recall=(float(correct[i] / support[i]) if support[i] else ""),
        ))
    for i, tc in enumerate(CLASSES):
        for j, pc in enumerate(CLASSES):
            rows_conf.append(dict(
                dataset=dataset, arm=arm["key"], level=level, repeat=repeat,
                fold=fold, true_cls=tc, pred_cls=pc, count=int(cm[i, j])))
    return cm


def load_and_align(arm, gdir):
    """Bags for one arm, already in the checkpoint's column order."""
    X, y, ids, feat_names, _ = load_external_bags(arm, gdir)
    cdir = CKPT_ROOT / arm["ckpt_tag"]
    cells = ckpt_cells(cdir)
    if not cells:
        raise SystemExit(f"no {CFGNAME} checkpoints in {cdir}")
    ck0 = torch.load(cells[0][2], map_location="cpu", weights_only=False)
    Xa = align_to_checkpoint(X, feat_names, ck0["feat_names"])
    print(f"  checkpoints: {len(cells)}   aligned to {len(ck0['feat_names'])} "
          f"columns (in_dim={ck0['in_dim']})")
    return Xa, y, ids, cells


def run_internal(arm, device, rows_recall, rows_conf):
    print(f"\n{'=' * 74}\nINTERNAL  {arm['key']}  --  {arm['nice']}\n{'=' * 74}")
    Xa, y, ids, cells = load_and_align(arm, INTERNAL_GDIR)

    csv_path = RESULTS_DIR / f"abmil_{arm['ckpt_tag']}_results.csv"
    df = pd.read_csv(csv_path)
    df = df[df["config"] == CFGNAME].set_index(["repeat", "fold"])
    print(f"  cross-check against {csv_path.name} ({len(df)} cells)")

    # slide-level OOF probabilities, one column block per repeat
    repeats = sorted({r for r, _, _ in cells})
    oof = np.full((len(repeats), len(Xa), len(CLASSES)), np.nan)
    rmap = {r: i for i, r in enumerate(repeats)}

    worst, t0 = 0.0, time.time()
    for n, (r, f, cp) in enumerate(cells):
        # exactly the split train_abmil.py used for this repeat
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        te = list(skf.split(np.zeros(len(Xa)), y))[f][1]
        probs, _ = score_checkpoint(cp, [Xa[i] for i in te], device)
        oof[rmap[r], te] = probs

        pred = probs.argmax(1)
        emit(rows_recall, rows_conf, "internal", arm, "cell", r, f, y[te], pred)

        got = balanced_accuracy_score(y[te], pred)
        want = float(df.loc[(r, f), "balanced_acc"])
        worst = max(worst, abs(got - want))
        if (n + 1) % 10 == 0:
            print(f"    {n + 1}/{len(cells)} cells ({time.time() - t0:.0f}s, "
                  f"worst |diff| so far {worst:.2e})")

    if worst > TOL:
        raise SystemExit(f"SELF-TEST FAILED for {arm['key']}: worst balanced "
                         f"accuracy difference {worst:.6f} > {TOL}. The fold "
                         f"reconstruction does not reproduce training-time "
                         f"scoring, so the per-class numbers would be wrong.")
    print(f"  self-test PASS -- worst |diff| vs results CSV {worst:.2e}")

    # each repeat covers all slides exactly once -> a full OOF pass
    for r in repeats:
        P = oof[rmap[r]]
        if np.isnan(P).any():
            raise SystemExit(f"{arm['key']} repeat {r}: {int(np.isnan(P[:, 0]).sum())} "
                             f"slide(s) never scored -- folds do not tile the corpus")
        emit(rows_recall, rows_conf, "internal", arm, "repeat_oof", r, "",
             y, P.argmax(1))

    cm = emit(rows_recall, rows_conf, "internal", arm, "ensemble", "", "",
              y, oof.mean(0).argmax(1))
    rec = cm.diagonal() / cm.sum(1).clip(min=1)
    print(f"  {len(repeats)}-repeat OOF ensemble per-class recall: "
          + "  ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, rec)))
    del Xa


def run_external(arm, device, rows_recall, rows_conf, ref):
    print(f"\n{'=' * 74}\nEXTERNAL  {arm['key']}  --  {arm['nice']}\n{'=' * 74}")
    Xa, y, ids, cells = load_and_align(arm, EXTERNAL_GDIR)

    P = np.empty((len(cells), len(Xa), len(CLASSES)), np.float64)
    t0 = time.time()
    for n, (r, f, cp) in enumerate(cells):
        probs, _ = score_checkpoint(cp, Xa, device)
        P[n] = probs
        emit(rows_recall, rows_conf, "external", arm, "checkpoint", r, f,
             y, probs.argmax(1))
        if (n + 1) % 10 == 0:
            print(f"    {n + 1}/{len(cells)} checkpoints "
                  f"({time.time() - t0:.0f}s)")

    cm = emit(rows_recall, rows_conf, "external", arm, "ensemble", "", "",
              y, P.mean(0).argmax(1))
    rec = cm.diagonal() / cm.sum(1).clip(min=1)
    print("  50-checkpoint ensemble per-class recall: "
          + "  ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, rec)))

    # cross-check against the numbers instrument1.json already stores
    if ref and arm["key"] in ref.get("arms", {}):
        want = ref["arms"][arm["key"]].get("per_class_recall")
        if want:
            d = max(abs(float(want[c]) - v) for c, v in zip(CLASSES, rec))
            if d > TOL:
                raise SystemExit(
                    f"EXTERNAL CROSS-CHECK FAILED for {arm['key']}: ensemble "
                    f"per-class recall differs from instrument1.json by {d:.6f}")
            print(f"  cross-check PASS vs instrument1.json (max |diff| {d:.2e})")
    del Xa


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[a["key"] for a in ARMS])
    ap.add_argument("--only", choices=["internal", "external"], default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    arms = [x for x in ARMS if x["key"] in a.arms]
    ref = (json.loads(INSTRUMENT1.read_text(encoding="utf-8"))
           if INSTRUMENT1.exists() else None)

    rows_recall, rows_conf = [], []
    if a.only != "external":
        for arm in arms:
            run_internal(arm, a.device, rows_recall, rows_conf)
    if a.only != "internal":
        for arm in arms:
            run_external(arm, a.device, rows_recall, rows_conf, ref)

    write_csv(OUT_DIR / "per_class_recall.csv", rows_recall,
              ["dataset", "arm", "arm_name", "level", "repeat", "fold", "cls",
               "support", "correct", "recall"])
    write_csv(OUT_DIR / "per_class_confusion.csv", rows_conf,
              ["dataset", "arm", "level", "repeat", "fold", "true_cls",
               "pred_cls", "count"])


if __name__ == "__main__":
    main()

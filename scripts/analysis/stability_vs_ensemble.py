# Copyright 2026 Vidip Gaur, Vidu Gaur
"""What does the 50.4% unanimity number actually cost the ENSEMBLE?

The stability figure describes the 10 individual seeds -- the ensemble's INPUTS.
The ensemble itself has no seed, so it cannot disagree with itself; the question
is whether the slides the seeds fight over are also the slides the ensemble gets
wrong, and whether the ensemble's own confidence flags them.

Reports, per arm:
  - unanimity rate (the headline stability number)
  - ensemble accuracy on unanimous vs contested slides
  - what share of all ensemble errors live in the contested set
  - mean ensemble confidence in each set (is instability self-declaring?)
  - seed-vote margin as a rejection criterion
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_abmil_nuclear import bags_for, collect  # noqa: E402
from train_abmil import CLASSES  # noqa: E402
import corpus_paths  # noqa: E402

ROOT = str(corpus_paths.ROOT)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ADD = "addmil_h128_do5"
R = 10

ARMS = [
    ("clean interpretable (Additive, dead_+ctx removed)",
     dict(tag="v5nucchromeos_clean_final", nuclear=True,
          drop_groups=["graph", "chrom_texture", "eos_other", "chrom_context"],
          exclude_prefix=["dead_"])),
    ("Phikon-v2 (Additive)",
     dict(tag="v5fmphikonv2_fm_addmil", nuclear=False,
          drop_groups=None, exclude_prefix=None)),
]


def run(label, kw):
    X, y, _sl, _fn = bags_for(ROOT, kw["tag"], kw["nuclear"],
                              drop_groups=kw["drop_groups"],
                              exclude_prefix=kw["exclude_prefix"])[:4]
    y = np.asarray(y)
    per = collect(ROOT, kw["tag"], ADD, X, y, R, DEV)
    n, C = len(y), len(CLASSES)

    P = np.full((R, n, C), np.nan)
    for r, got in per.items():
        for i, (_pred, prob, _a) in got.items():
            P[r, i] = prob
    ok = ~np.isnan(P[:, :, 0]).any(0)          # slides scored in every repeat
    P, y = P[:, ok], y[ok]
    preds = P.argmax(2)                        # (R, n)

    ens = P.mean(0)
    ep = ens.argmax(1)
    conf = ens.max(1)

    unan = (preds == preds[0]).all(0)
    votes = np.array([np.bincount(preds[:, i], minlength=C) for i in range(len(y))])
    modal_share = votes.max(1) / R

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print(f"  slides {len(y)}   seeds {R}")
    print(f"  unanimous (all {R} seeds same class) : {unan.mean():.1%}"
          f"   ({unan.sum()} slides)")
    print(f"  median share of seeds on modal class : {np.median(modal_share):.2f}")
    print(f"  ensemble balanced acc (all)          : "
          f"{balanced_accuracy_score(y, ep):.4f}")
    print(f"  mean single-seed balanced acc        : "
          f"{np.mean([balanced_accuracy_score(y, preds[r]) for r in range(R)]):.4f}")

    for name, m in (("unanimous", unan), ("contested", ~unan)):
        if m.sum() == 0:
            continue
        acc = (ep[m] == y[m]).mean()
        print(f"  [{name:9s}] n={m.sum():3d}  ensemble plain acc {acc:.3f}   "
              f"mean conf {conf[m].mean():.3f}")

    err = ep != y
    print(f"  ensemble errors total {err.sum()}, of which "
          f"{(err & ~unan).sum()} ({(err & ~unan).sum()/max(err.sum(),1):.0%}) "
          f"are on contested slides")
    print(f"  contested slides are {(~unan).mean():.0%} of the corpus but carry "
          f"{(err & ~unan).sum()/max(err.sum(),1):.0%} of the errors")

    # does the seed vote work as an abstention rule?
    print("\n  abstention curve (reject lowest-agreement slides first):")
    order = np.argsort(-modal_share)
    for keep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = order[:int(round(keep * len(y)))]
        print(f"    keep {keep:4.0%} ({len(k):3d} slides): "
              f"bal acc {balanced_accuracy_score(y[k], ep[k]):.4f}   "
              f"plain {(ep[k] == y[k]).mean():.3f}")

    # per-class instability
    print("\n  per class: unanimity / ensemble recall")
    for c, name in enumerate(CLASSES):
        m = y == c
        print(f"    {name:5s} n={m.sum():3d}  unanimity {unan[m].mean():.1%}   "
              f"recall {(ep[m] == y[m]).mean():.3f}")
    return unan, ep, y, ok


if __name__ == "__main__":
    for lab, kw in ARMS:
        run(lab, kw)

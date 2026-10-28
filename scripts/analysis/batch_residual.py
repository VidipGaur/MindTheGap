# Copyright 2026 Vidip Gaur, Vidu Gaur
"""How much of the signal survives removing everything staining can explain?

resolve_dead_count.py found that slide-level STAINING descriptors alone predict
subtype at 0.358 vs 0.200 chance. That is a corpus-wide batch structure tracking
the label, and it threatens far more than dead_count -- it puts a question mark
over the whole result.

The test: regress each slide-level feature on the staining descriptors, keep
only the RESIDUAL, and re-run the classifier. Anything the batch variable could
linearly account for is gone. If accuracy holds up, the biology is real and the
batch effect rides alongside it; if it collapses, the result is provenance.

Slide-level (not MIL) so it is cheap and interpretable; the MIL model uses the
same underlying features, so the conclusion transfers.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path

from resolve_dead_count import STAIN_COLS, load_stain
from train_abmil import CLASSES, load_bags
from train_slidelevel_gbm import slide_matrix
import corpus_paths

R = corpus_paths.ROOT
gdir = R / "cellvit_run/graphs"
odir = R / "cellvit_run/graphs/gatv2_run"

X, y, ids, fn = load_bags(gdir, odir / "bag_cache_v5nuc.pt", False, None,
                          nuclear_dir=gdir / "nuclear_feats",
                          drop_groups=["graph"])
Z, cols = slide_matrix(X, fn)
stain = load_stain(R).set_index("image_id")
use = [c for c in STAIN_COLS if c in stain.columns]
S = stain.reindex([str(i) for i in ids])[use].values.astype(float)
ok = np.isfinite(S).all(1)
print(f"\n{ok.sum()}/{len(ids)} slides have staining descriptors")
Z, y2, S = Z[ok], np.array(y)[ok], S[ok]
Sz = StandardScaler().fit_transform(S)

cv = StratifiedKFold(5, shuffle=True, random_state=0)


def bal(M, seeds=5):
    out = []
    for s in range(seeds):
        cvs = StratifiedKFold(5, shuffle=True, random_state=s)
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=0.1,
                                                class_weight="balanced"))
        out.append(balanced_accuracy_score(
            y2, cross_val_predict(pipe, M, y2, cv=cvs)))
    return float(np.mean(out)), float(np.std(out))


# residualise every feature against the staining descriptors
A = np.column_stack([np.ones(len(Sz)), Sz])
beta = np.linalg.lstsq(A, Z, rcond=None)[0]
Zr = Z - A @ beta

print("\n" + "=" * 70)
print("DOES THE SIGNAL SURVIVE REMOVING THE BATCH EFFECT?")
print("=" * 70)
for label, M in [("staining descriptors alone", Sz),
                 ("nuclear features (raw)", Z),
                 ("nuclear features + staining", np.column_stack([Z, Sz])),
                 ("nuclear features RESIDUALISED on staining", Zr)]:
    m, s = bal(M)
    print(f"  {label:44s} {m:.4f}  (sd {s:.4f})")
print(f"  {'chance':44s} {1/len(CLASSES):.4f}")

# and without the necrosis columns at all
dead = [i for i, c in enumerate(cols) if c.startswith("dead_")]
keep = [i for i in range(len(cols)) if i not in set(dead)]
print(f"\n  dropping the {len(dead)} dead_* slide columns:")
m, s = bal(Z[:, keep])
print(f"  {'nuclear features, no dead_*':44s} {m:.4f}  (sd {s:.4f})")
m, s = bal(Zr[:, keep])
print(f"  {'residualised, no dead_*':44s} {m:.4f}  (sd {s:.4f})")

print("\n  Residualising removes everything the staining descriptors can")
print("  linearly explain. A small drop means the morphometrics carry")
print("  biology that is not reducible to slide provenance.")

# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Is the encoder's stain content LINEARLY accessible, or is there more of it
hiding where a ridge map cannot reach?

WHY THIS EXISTS
---------------
run_stainaudit.py deletes the stain-aligned subspace of an embedding using a
linear ridge map, then retrains. If that comes back null -- deleting stain
costs no more than deleting a matched random subspace -- the honest reading is
"no LINEARLY accessible batch shortcut", not "no batch shortcut". An encoder
could represent staining nonlinearly and keep using it by a route the map
never touched, and the deletion would miss it entirely.

That caveat is either fatal to the null or nearly irrelevant, and which one it
is happens to be measurable. Predict each stain descriptor FROM the embedding
twice: once with ridge (the same linear family the deletion uses) and once
with gradient boosting (which can use whatever nonlinear structure is there).

  GBM R2 >> ridge R2  -> the embedding holds stain information the deletion
      map cannot see. A null from the deletion is weak and must be reported
      as bounding linear signal only.
  GBM R2 ~= ridge R2  -> the linear map already sees essentially all the stain
      content in the embedding. The deletion's null is then close to a real
      answer, not a hedged one.

This does NOT ask whether the classifier uses stain -- that is the deletion
arms' job. It asks only whether the deletion arms were capable of finding it.

HONESTY CONSTRAINTS
-------------------
Held out by SLIDE, never by tile. Tiles within a slide share the slide's
staining almost by definition, so a tile-level split would let the model
memorise each slide's stain level and report a spectacular R2 that means
nothing. This is the same failure mode grouped CV guards against in the bridge
regression, and it is the single easiest way to fake this analysis.

Both models see the IDENTICAL rows and columns, so the comparison is about
model class and nothing else. Ridge alpha is chosen on the training slides
only.
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
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_abmil_nuclear import bags_for                  # noqa: E402
from fm_subspace_split import STAIN_COLS                    # noqa: E402
import corpus_paths  # noqa: E402

ROOT = corpus_paths.ROOT
FMDIR = ROOT / "cellvit_run" / "graphs" / "fm_feats"
INT_TAG = "v5nucchromeos_clean_final"
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="uni2h")
    ap.add_argument("--tiles-per-slide", type=int, default=60,
                    help="subsample; the question is about representational "
                         "content, which does not need all 318k tiles")
    ap.add_argument("--holdout", type=float, default=0.25)
    a = ap.parse_args()

    X, y, ids, names = bags_for(ROOT, INT_TAG, True, chromatin=True,
                                eosin=True)[:4]
    keep = [names.index(c) for c in STAIN_COLS]
    feats = {str(i): np.asarray(x, dtype=np.float64)[:, keep]
             for i, x in zip(ids, X)}

    rng = np.random.default_rng(SEED)
    sids = sorted(feats)
    held = set(rng.choice(sids, size=int(len(sids) * a.holdout), replace=False))

    rows = {"tr": ([], []), "te": ([], [])}
    for iid in sids:
        p = FMDIR / f"{iid}_{a.model}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            E = z["fm_features"].astype(np.float32)
        F = feats[iid]
        if E.shape[0] != F.shape[0]:
            continue
        n = min(a.tiles_per_slide, E.shape[0])
        sel = rng.choice(E.shape[0], size=n, replace=False)
        k = "te" if iid in held else "tr"
        rows[k][0].append(E[sel])
        rows[k][1].append(F[sel])

    Etr = np.vstack(rows["tr"][0]); Ftr = np.vstack(rows["tr"][1])
    Ete = np.vstack(rows["te"][0]); Fte = np.vstack(rows["te"][1])
    print(f"  encoder {a.model}: train {Etr.shape}  held-out {Ete.shape} "
          f"({len(sids)-len(held)} / {len(held)} slides)", flush=True)
    print("  held out by SLIDE -- a tile-level split would be self-fulfilling\n",
          flush=True)

    mu, sd = Etr.mean(0), Etr.std(0) + 1e-8
    Ztr, Zte = (Etr - mu) / sd, (Ete - mu) / sd

    print(f"  {'stain descriptor':22s} {'ridge R2':>9s} {'GBM R2':>9s} "
          f"{'gain':>8s}")
    print("  " + "-" * 52)
    gains = []
    for j, col in enumerate(STAIN_COLS):
        ytr, yte = Ftr[:, j], Fte[:, j]
        if ytr.std() < 1e-9:
            continue
        rid = RidgeCV(alphas=(1e0, 1e1, 1e2, 1e3, 1e4)).fit(Ztr, ytr)
        r_lin = r2_score(yte, rid.predict(Zte))
        gbm = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, random_state=SEED).fit(Ztr, ytr)
        r_gbm = r2_score(yte, gbm.predict(Zte))
        gains.append(r_gbm - r_lin)
        print(f"  {col:22s} {r_lin:9.3f} {r_gbm:9.3f} {r_gbm - r_lin:+8.3f}",
              flush=True)

    g = float(np.median(gains))
    print("\n  median nonlinear gain: {:+.3f}".format(g))
    if g > 0.10:
        print("  SUBSTANTIAL -- the embedding carries stain structure the "
              "linear\n  deletion map cannot reach. A null from the stain "
              "audit bounds\n  LINEAR batch signal only, and must be worded "
              "that way.")
    else:
        print("  SMALL -- the linear map already recovers essentially all the "
              "stain\n  content the embedding holds, so the stain audit's "
              "deletion was\n  capable of finding the shortcut if it was "
              "being used.")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"\n  ({(time.time()-t0)/60:.1f} min)")
    sys.exit(rc)

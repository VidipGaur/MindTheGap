# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Split Phikon-v2's embedding into the part our pathology vocabulary explains
and the part it does not -- then let the classifier tell us which half matters.

WHY THIS AND NOT ANOTHER BRIDGE REGRESSION
------------------------------------------
bridge_fm_interpretable.py answers "how much of Phikon's per-tile contribution
can the named features predict?" (median 38% of its own self-ceiling). That is a
statement about *explainability*. It is silent on the question a reader will
actually ask, which is about *sufficiency*: does the morphometry-aligned part of
the embedding carry the accuracy, or is Phikon winning on something outside the
vocabulary entirely?

The decomposition here answers that directly. For tile-level embeddings E
(1024-d) and named pathology features F (49 clean columns), fit the ridge map
  E ~= [1, F] W
and define
  E_fit   = [1, F] W        (rank <= 50; the morphometry-aligned subspace)
  E_resid = E - E_fit       (everything Phikon sees that F cannot express)

Note E_fit needs no experiment: it is a fixed linear function of F, so any MIL
model trained on E_fit can be reparameterised as one trained on F. Its accuracy
IS the interpretable model's accuracy, 0.6603, already measured. The only arm
that carries information is the RESIDUAL. Hence:

  fm_resid      Phikon-v2 minus its morphometry-predictable component
  fm_residnull  Phikon-v2 minus a RANK-MATCHED uninformative map (primary ctl)
  fm_residctl   Phikon-v2 minus a VARIANCE-MATCHED random PCA subspace (2nd ctl)

CONTROLS, AND WHY THEY ARE NOT OPTIONAL
---------------------------------------
Removing any rank-50 component removes capacity, so `fm_resid` scoring below
plain Phikon proves nothing on its own.

`fm_residnull` is the primary control because it is matched on the thing that
matters: it runs the IDENTICAL procedure -- fit a rank-50 ridge map from 49
pathology columns, subtract it -- but the features are drawn from a DIFFERENT
slide (cyclic shift, never a fixed point, so no slide is paired with itself),
resampled to the right row count. Same rank, same feature marginals, same
estimator, no true tile correspondence. Whatever the subtraction costs by
construction, it costs here too.

`fm_residctl` deletes randomly chosen principal components of E summing to the
same variance the morphometry map removed. It is variance-matched but NOT
rank-matched -- because Phikon's variance is concentrated in its leading
components, matching 28% of variance by random draw takes several hundred PCs
against the morphometry map's 50. That asymmetry runs AGAINST the null (the
control is handicapped), so it is reported as a bound, not as a clean twin.

PRE-REGISTERED READING (fixed before the run, paired on shared repeat/fold):
  fm_resid ~= fm_residnull (p > 0.10)  -> the morphometry-aligned subspace is
      not special. Phikon's accuracy does not live in what our vocabulary
      describes; the named features are a projection of it, not its engine.
  fm_resid <  fm_residnull (p < 0.10)  -> removing morphometry costs more than
      removing a matched uninformative map. The gap is the quantitative share
      of Phikon's advantage that our vocabulary accounts for.
  fm_resid ~= plain Phikon             -> Phikon has ample redundant routes to
      the same answer; explainability and sufficiency come apart.

LEAKAGE: W is fit with NO labels -- it is a regression of embeddings on
features. There is therefore no label leakage into the folds. It is
transductive (test slides' own tiles inform the projection), which is the same
standing as any corpus-level normalisation, and is stated in the writeup rather
than engineered away: making it per-fold would mean writing 50 copies of a
1.2 GB feature corpus for no change to the question being asked.

Writes <id>_phikonv2resid.npz / <id>_phikonv2residctl.npz next to the source
embeddings, carrying the same grid_rc_checksum so every alignment guard
downstream still fires.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_abmil_nuclear import bags_for  # noqa: E402
from train_abmil import OUT_SUBFOLDER, RUN_DIR_NAME, SCHEMA_VERSION  # noqa: E402
import corpus_paths  # noqa: E402

ROOT = corpus_paths.ROOT
GDIR = ROOT / "cellvit_run" / "graphs"
FMDIR = GDIR / "fm_feats"
SRC = "phikonv2"

INT_TAG = "v5nucchromeos_clean_final"
INT_DROP = ["graph", "chrom_texture", "eos_other", "chrom_context"]
INT_EXCL = ["dead_"]

# Per-tile descriptors of how the slide was STAINED and SCANNED, not of what
# the tissue is. The batch confound is established on this corpus: staining
# descriptors alone predict subtype at 0.358 balanced accuracy against 0.20
# chance, and an unsupervised clustering built on staining alone yields a
# cluster containing zero CC and zero MC slides. We removed the columns
# carrying that signal from the interpretable model (dead_*, chrom_context),
# which cost it 0.0272 balanced accuracy, p=0.0009 -- but we could not remove
# it from the encoders, because an embedding dimension has no name. So the
# reported gap is contaminated in the encoders' favour by an unknown amount.
#
# This makes the amount knowable. Point the SAME decomposition at the stain
# descriptors instead of at morphometry: delete the stain-aligned subspace of
# the embedding, retrain, and the accuracy the encoder loses is the share of
# its advantage that was batch rather than biology.
#
# Excluded deliberately, though they are also tile_* columns: tile_clear_frac,
# tile_pale_frac, tile_region_frac and tile_nuc_area_frac describe enclosed
# empty space and nuclear area -- clear cytoplasm and mucin, i.e. the CC/MC
# diagnostic evidence proper. They are morphology that happens to be measured
# photometrically, and the clean model keeps them on purpose. Putting them in
# the stain probe would delete real diagnostic signal and inflate the answer.
STAIN_COLS = [
    "tile_h_mean", "tile_h_std", "tile_tissue_frac",          # haematoxylin
    "tile_e_mean", "tile_e_std", "tile_e_p10", "tile_e_p90",  # eosin level
    "tile_e_entropy", "tile_e_grad",                          # focus / blur
    "tile_he_ratio",                                          # H:E balance
]

ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
SEED = 0


def check_bag_cache():
    """Refuse a bag cache under INT_TAG that was not built from exactly the
    nuclear, chromatin and eosin blocks.

    bags_for() keys its cache on the tag alone and reuses it whatever blocks
    the caller asks for, and INT_DROP removes only the groups it names. A cache
    missing the chromatin or eosin block, or carrying an embedding block, would
    therefore return a different column set without any error."""
    cache_tag = INT_TAG.split("_")[0].replace("noflag", "")
    cache = ROOT / RUN_DIR_NAME / OUT_SUBFOLDER / f"bag_cache_{cache_tag}.pt"
    if not cache.exists():
        return
    blob = torch.load(cache, weights_only=False)
    if blob.get("schema") != SCHEMA_VERSION:
        return      # load_bags() rebuilds it from the blocks bags_for() asks for
    built = [b for b in ("nuclear", "chromatin", "eosin", "fm") if b in blob]
    if built != ["nuclear", "chromatin", "eosin"]:
        raise SystemExit(
            f"{cache} was built from the feature blocks "
            f"[{', '.join(built) or 'graph only'}], but the named features need "
            f"nuclear, chromatin and eosin. Delete it and rerun; "
            f"named_features() rebuilds it from those three.")


def named_features(target="morph"):
    """The feature block that defines the subspace to delete, keyed by slide id
    so it can be paired with the embedding files.

    `morph` is the exact 49 clean columns the interpretable headline model
    trains on. `stain` is STAIN_COLS -- the batch probe. The stain block is
    selected from the UNFILTERED feature set, because the clean model's own
    --drop-groups removes precisely the columns the stain probe needs."""
    if target == "morph":
        check_bag_cache()
        X, y, ids, names = bags_for(ROOT, INT_TAG, True, drop_groups=INT_DROP,
                                    exclude_prefix=INT_EXCL,
                                    chromatin=True, eosin=True)[:4]
        return ({str(i): np.asarray(x, dtype=np.float64)
                 for i, x in zip(ids, X)}, names)

    X, y, ids, names = bags_for(ROOT, INT_TAG, True,
                                chromatin=True, eosin=True)[:4]
    keep = [names.index(c) for c in STAIN_COLS if c in names]
    missing = [c for c in STAIN_COLS if c not in names]
    if missing:
        raise SystemExit(f"stain probe columns absent: {missing}")
    return ({str(i): np.asarray(x, dtype=np.float64)[:, keep]
             for i, x in zip(ids, X)}, [names[i] for i in keep])


def load_fm(iid):
    p = FMDIR / f"{iid}_{SRC}.npz"
    if not p.exists():
        return None, None
    with np.load(p, allow_pickle=True) as z:
        return z["fm_features"].astype(np.float64), int(z["grid_rc_checksum"][0])


def shuffled_partner(ids):
    """Cyclic shift, not a random permutation: a permutation has fixed points,
    and a slide paired with itself would receive the REAL map, quietly
    weakening the null. Same reasoning as the mixup pairing fix (Task 10)."""
    return {i: ids[(k + 1) % len(ids)] for k, i in enumerate(ids)}


def resample_rows(F, n, rng):
    """Match a donor slide's feature block to the recipient's tile count."""
    if F.shape[0] == n:
        return F
    return F[rng.integers(0, F.shape[0], size=n)]


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fit and report the decomposition, write nothing")
    ap.add_argument("--model", default=SRC,
                    help="which encoder to decompose (default phikonv2). The "
                         "audit claim is about pathology foundation models in "
                         "general, so the split has to be repeatable across "
                         "encoders, not measured once on one of them.")
    ap.add_argument("--target", choices=("morph", "stain"), default="morph",
                    help="which subspace to delete. 'morph' = the 49 named "
                         "pathology columns (the audit). 'stain' = the "
                         "per-tile staining/scanning descriptors (the batch "
                         "confound), which answers how much of the encoder's "
                         "advantage is the shortcut we removed from the "
                         "interpretable arm and could not remove from it.")
    a = ap.parse_args()

    SRC = a.model
    # stain arms get their own npz suffix so the two decompositions of the same
    # encoder never overwrite each other
    out_pre = "stain" if a.target == "stain" else ""
    print(f"  encoder: {SRC}   target: {a.target}", flush=True)

    feats, names = named_features(a.target)
    print(f"  {a.target} features: {len(names)} columns, {len(feats)} slides",
          flush=True)
    if a.target == "stain":
        print(f"    {', '.join(names)}", flush=True)

    ids = sorted(feats)
    partner = shuffled_partner(ids)
    rng = np.random.default_rng(SEED)
    held = set(rng.choice(ids, size=max(1, len(ids) // 5), replace=False))
    d = len(names) + 1
    acc = {k: dict(n=0, FF=np.zeros((d, d)), FE=None, EE=None,
                   sF=np.zeros(d), sE=None) for k in ("tr", "te")}
    nullFF, nullFE = np.zeros((d, d)), None
    skipped = []
    for k, iid in enumerate(ids):
        E, ck = load_fm(iid)
        F = feats[iid]
        if E is None:
            skipped.append((iid, "no fm npz"))
            continue
        if E.shape[0] != F.shape[0]:
            skipped.append((iid, f"rows {E.shape[0]} vs {F.shape[0]}"))
            continue
        Fa = np.hstack([np.ones((F.shape[0], 1)), F])
        Fn = np.hstack([np.ones((F.shape[0], 1)),
                        resample_rows(feats[partner[iid]], F.shape[0],
                                      np.random.default_rng(SEED + 1000 + k))])
        s = acc["te" if iid in held else "tr"]
        if s["FE"] is None:
            s["FE"] = np.zeros((d, E.shape[1]))
            s["EE"] = np.zeros(E.shape[1])
            s["sE"] = np.zeros(E.shape[1])
        if nullFE is None:
            nullFE = np.zeros((d, E.shape[1]))
        s["n"] += E.shape[0]
        s["FF"] += Fa.T @ Fa
        s["FE"] += Fa.T @ E
        s["EE"] += (E * E).sum(0)
        s["sF"] += Fa.sum(0)
        s["sE"] += E.sum(0)
        nullFF += Fn.T @ Fn
        nullFE += Fn.T @ E
    if skipped:
        print(f"  SKIPPED {len(skipped)} slides: {skipped[:5]}", flush=True)
    print(f"  tiles: train {acc['tr']['n']}  held-out {acc['te']['n']}",
          flush=True)

    tr, te = acc["tr"], acc["te"]
    ridge = np.eye(d)
    ridge[0, 0] = 0.0                      # never penalise the intercept
    best = None
    for al in ALPHAS:
        W = np.linalg.solve(tr["FF"] + al * ridge, tr["FE"])
        # held-out residual SS = EE - 2 tr(W' FE) + tr(W' FF W)
        ss = (te["EE"].sum() - 2 * np.trace(W.T @ te["FE"])
              + np.trace(W.T @ te["FF"] @ W))
        tot = te["EE"].sum() - (te["sE"] ** 2).sum() / te["n"]
        r2 = 1.0 - ss / tot
        print(f"    alpha {al:9.0f}  held-out R2 {r2:+.4f}", flush=True)
        if best is None or r2 > best[0]:
            best = (r2, al, W)
    r2, alpha, _ = best
    ALL_FF = tr["FF"] + te["FF"]
    ALL_FE = tr["FE"] + te["FE"]
    ALL_EE = tr["EE"] + te["EE"]
    ALL_sE = tr["sE"] + te["sE"]
    N = tr["n"] + te["n"]
    W = np.linalg.solve(ALL_FF + alpha * ridge, ALL_FE)
    ss_res = (ALL_EE.sum() - 2 * np.trace(W.T @ ALL_FE)
              + np.trace(W.T @ ALL_FF @ W))
    ss_tot = ALL_EE.sum() - (ALL_sE ** 2).sum() / N
    frac = 1.0 - ss_res / ss_tot
    print(f"\n  chosen alpha {alpha:.0f} (held-out R2 {r2:+.4f})")
    print(f"  the {len(names)} {a.target} features explain {frac:.1%} of "
          f"{SRC}'s total tile-embedding variance", flush=True)

    # The null map gets the SAME alpha, deliberately. Re-selecting alpha for it
    # would shrink an uninformative map toward zero and hand the control an
    # easier job than the real one -- the point is an identical procedure.
    Wn = np.linalg.solve(nullFF + alpha * ridge, nullFE)
    ss_n = (ALL_EE.sum() - 2 * np.trace(Wn.T @ nullFE)
            + np.trace(Wn.T @ nullFF @ Wn))
    frac_n = 1.0 - ss_n / ss_tot
    print(f"  the same map from MISMATCHED slides explains {frac_n:.1%} "
          f"-- the floor the real number must beat", flush=True)

    mu = ALL_sE / N
    # E'E is not accumulated in full (1024x1024 = 8 MB would be fine, but the
    # per-dim variances are enough: PCA on a diagonal basis is not PCA, so do
    # a second pass for the full covariance).
    print("\n  pass 2: covariance for the matched random control", flush=True)
    C = np.zeros((len(mu), len(mu)))
    for iid in ids:
        E, _ = load_fm(iid)
        if E is None or E.shape[0] != feats[iid].shape[0]:
            continue
        Ec = E - mu
        C += Ec.T @ Ec
    C /= N
    ev, V = np.linalg.eigh(C)
    ev, V = ev[::-1], V[:, ::-1]
    share = ev / ev.sum()
    # pick a random component set whose cumulative variance ~= frac
    rng2 = np.random.default_rng(SEED + 1)
    order = rng2.permutation(len(share))
    pick, tot = [], 0.0
    for j in order:
        if tot >= frac:
            break
        pick.append(j)
        tot += share[j]
    P = V[:, pick]
    print(f"  control removes {len(pick)} random PCs carrying {tot:.1%} of "
          f"variance (target {frac:.1%}); morphometry map is rank {d}",
          flush=True)

    if a.check:
        print("\n  --check: nothing written.")
        return 0

    print("\n  pass 3: writing residual embeddings", flush=True)
    nw = 0
    for k, iid in enumerate(ids):
        E, ck = load_fm(iid)
        F = feats[iid]
        if E is None or E.shape[0] != F.shape[0]:
            continue
        Fa = np.hstack([np.ones((F.shape[0], 1)), F])
        # same per-slide seed as pass 1, so the null map is subtracted using the
        # very rows it was fitted on
        Fn = np.hstack([np.ones((F.shape[0], 1)),
                        resample_rows(feats[partner[iid]], F.shape[0],
                                      np.random.default_rng(SEED + 1000 + k))])
        Ec = E - mu
        resid = (E - Fa @ W).astype(np.float32)
        null = (E - Fn @ Wn).astype(np.float32)
        ctl = (Ec - (Ec @ P) @ P.T).astype(np.float32)
        for tag, arr in ((f"{out_pre}resid", resid),
                         (f"{out_pre}residnull", null),
                         (f"{out_pre}residctl", ctl)):
            np.savez_compressed(FMDIR / f"{iid}_{SRC}{tag}.npz",
                                fm_features=arr,
                                grid_rc_checksum=np.array([ck]))
        nw += 1
        if nw % 50 == 0:
            print(f"    {nw}/{len(ids)}", flush=True)
    print(f"  wrote {nw} slides x 3 arms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

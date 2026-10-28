# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Is `dead_count` biology, or is it a staining/fixation artefact?

    python scripts/analysis/resolve_dead_count.py                 # the statistical case
    python scripts/analysis/resolve_dead_count.py --montage 6     # + image panels to eyeball

WHY THIS BLOCKS EVERYTHING ELSE
===============================
`dead_count` (CellViT's "dead" nucleus class, per tile) is the top explanation
driver on 76.3% of correct EC calls, and removing the necrosis features costs
0.0508 balanced accuracy (p=0.0078). So roughly 5 points of the headline 0.67
rest on it.

It is also deeply suspect. Its median is ~0 in EVERY class, skew +9.6, kurtosis
+110, with 4 slides beyond +5 SD. Necrosis is not a criterion that separates
endometrioid carcinoma from the other ovarian subtypes. And "dead" is precisely
the CellViT class most likely to fire on fixation and staining quality --
pyknotic-looking debris, over-stained nuclei, crush artefact -- rather than on
biology. If it is an artefact, the headline number changes and the paper's EC
story is wrong.

The obstacle is that `train.csv` has no site or scanner column, so the usual
batch test is unavailable. This script builds the missing covariates from the
images themselves: extract_chromatin_features.py records, per slide, the
background fraction, tissue optical density, haematoxylin and eosin levels,
the H:E balance and a blur measure. Those are a proxy for "which lab and
scanner did this slide come from".

THE FOUR TESTS
==============
1. CORRELATION -- does dead_count track the staining descriptors at all?
   A strong association with H:E balance or blur, and no plausible biological
   route for it, is the artefact signature.

2. CONDITIONAL SIGNAL -- does dead_count still predict subtype once staining
   is controlled for? Compares three logistic models: dead_count alone,
   staining alone, and both. If dead_count adds nothing over staining, its
   apparent signal was the batch effect.

3. BATCH CLUSTERING -- cluster slides on staining alone (no labels, no
   dead_count). If the resulting clusters predict subtype AND carry the
   dead_count mass, there is a batch structure driving both.

4. THE PICTURES -- crop the highest-dead_count tiles from the most extreme
   slides, next to matched low-dead_count tiles from the SAME slides. This is
   the test that actually settles it, and it needs a human.

VERDICT LOGIC
=============
The tests can exonerate or convict, and are reported separately rather than
collapsed into a single score, because the honest outcome may well be "the
statistics are consistent with biology but the pictures show debris".
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_abmil import (CLASSES, GRAPH_SUBFOLDER, OUT_SUBFOLDER, ROOT,
                         RUN_DIR_NAME, load_bags)

DEAD_FEATS = ("dead_count", "dead_frac")
# slide-level staining / acquisition descriptors written by
# extract_chromatin_features.py -- the stand-in for the missing site column
STAIN_COLS = ["bg_frac", "tissue_od_p50", "tissue_od_p95", "blur_lapvar",
              "h_tissue_p50", "h_tissue_p99", "e_tissue_p50", "e_tissue_p99",
              "he_ratio", "tissue_r_mean", "tissue_g_mean", "tissue_b_mean"]


def slide_level(X, y, ids, feat_names):
    """Per-slide mean of each dead_* feature, plus a few shape statistics."""
    out = {"image_id": ids, "label": [CLASSES[t] for t in y]}
    for f in DEAD_FEATS:
        if f in feat_names:
            j = feat_names.index(f)
            out[f] = [float(b[:, j].mean()) for b in X]
            out[f + "_max"] = [float(b[:, j].max()) for b in X]
            out[f + "_tilefrac"] = [float((b[:, j] > 0).float().mean())
                                    for b in X]
    return pd.DataFrame(out)


def load_stain(root):
    """Slide-level staining descriptors from the chromatin extraction."""
    p = Path(root) / RUN_DIR_NAME / GRAPH_SUBFOLDER / "chromatin_feats" / \
        "_manifest.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"image_id": str})
    if "error" in d:
        d = d[d.error.isna()]
    keep = ["image_id"] + [c for c in STAIN_COLS if c in d.columns]
    return d[keep].drop_duplicates("image_id")


def test_correlation(df, cols):
    print("\n" + "=" * 78)
    print("TEST 1 -- does dead_count track slide staining?")
    print("=" * 78)
    rows = []
    for c in cols:
        v = df[c].values.astype(float)
        ok = np.isfinite(v) & np.isfinite(df.dead_count.values)
        if ok.sum() < 20 or np.std(v[ok]) < 1e-12:
            continue
        r = stats.spearmanr(df.dead_count.values[ok], v[ok])
        rows.append(dict(descriptor=c, rho=r.statistic, p=r.pvalue, n=int(ok.sum())))
    if not rows:
        print("  no staining descriptors available")
        return None
    t = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    print("  " + t.round(4).to_string(index=False).replace("\n", "\n  "))
    top = t.iloc[0]
    print(f"\n  strongest: {top.descriptor} rho={top.rho:+.3f} p={top.p:.2e}")
    if abs(top.rho) > 0.35 and top.p < 0.001:
        print("  -> dead_count is substantially a function of how the slide was")
        print("     stained or scanned. That is the artefact signature.")
    elif abs(top.rho) > 0.20:
        print("  -> a moderate association. Not damning on its own, but it means")
        print("     dead_count is not purely biological either.")
    else:
        print("  -> weak. dead_count does not simply track staining, which is")
        print("     evidence FOR it being real. Keep going to tests 2-4.")
    return t


def test_conditional(df, cols):
    print("\n" + "=" * 78)
    print("TEST 2 -- does dead_count predict subtype once staining is controlled?")
    print("=" * 78)
    y = np.array([CLASSES.index(c) for c in df.label])
    use = [c for c in cols if c in df and np.std(df[c].astype(float)) > 1e-12]
    dead = df[[c for c in df.columns if c.startswith("dead_")]].values.astype(float)
    stain = df[use].values.astype(float)
    if len(np.unique(y)) < 2:
        return
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def bal(Z):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000,
                                                class_weight="balanced"))
        return balanced_accuracy_score(y, cross_val_predict(pipe, Z, y, cv=cv))

    b_dead, b_stain = bal(dead), bal(stain)
    b_both = bal(np.column_stack([dead, stain]))
    chance = 1.0 / len(CLASSES)
    print(f"  dead_* features alone      : {b_dead:.4f}")
    print(f"  staining descriptors alone : {b_stain:.4f}")
    print(f"  both together              : {b_both:.4f}")
    print(f"  chance                     : {chance:.4f}")
    print(f"\n  dead_* adds {b_both - b_stain:+.4f} on top of staining")
    if b_stain > b_dead and (b_both - b_stain) < 0.02:
        print("  -> staining alone predicts subtype at least as well as")
        print("     dead_count does, and dead_count adds nothing on top. The")
        print("     apparent necrosis signal is the batch effect. CONVICTING.")
    elif (b_both - b_stain) > 0.05:
        print("  -> dead_count carries subtype information that staining does")
        print("     NOT explain. That is real signal, whatever its biological")
        print("     interpretation. EXONERATING.")
    else:
        print("  -> ambiguous; the two are partly entangled. Weight tests 3-4.")
    if b_stain > chance + 0.08:
        print(f"\n  SEPARATE WARNING: staining alone predicts subtype at "
              f"{b_stain:.3f} vs")
        print("  chance. Whatever happens to dead_count, this corpus has a batch")
        print("  structure that tracks the label, and that belongs in the paper's")
        print("  limitations regardless.")


def test_batch(df, cols, k=4):
    print("\n" + "=" * 78)
    print(f"TEST 3 -- unsupervised batch clusters ({k} clusters on staining only)")
    print("=" * 78)
    use = [c for c in cols if c in df and np.std(df[c].astype(float)) > 1e-12]
    if len(use) < 3:
        print("  too few descriptors")
        return
    Z = StandardScaler().fit_transform(df[use].values.astype(float))
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    df = df.assign(batch=lab)
    ct = pd.crosstab(df.batch, df.label, normalize="index").round(3)
    print("  subtype composition per cluster (rows sum to 1):")
    print("  " + ct.to_string().replace("\n", "\n  "))
    print("\n  dead_count by cluster:")
    print("  " + df.groupby("batch").dead_count.agg(
        ["mean", "median", "max", "count"]).round(3).to_string()
        .replace("\n", "\n  "))
    chi = stats.chi2_contingency(pd.crosstab(df.batch, df.label))
    kw = stats.kruskal(*[df.dead_count[df.batch == b].values for b in range(k)
                         if (df.batch == b).sum() > 2])
    print(f"\n  cluster vs subtype   chi2 p = {chi.pvalue:.2e}")
    print(f"  cluster vs dead_count Kruskal p = {kw.pvalue:.2e}")
    if chi.pvalue < 0.01 and kw.pvalue < 0.01:
        print("  -> the staining clusters predict BOTH subtype and dead_count.")
        print("     A batch variable driving both is exactly the confound.")
    else:
        print("  -> no single staining cluster explains both. Weaker case for")
        print("     a batch artefact.")


def test_distribution(df):
    print("\n" + "=" * 78)
    print("TEST 0 -- the distribution, restated")
    print("=" * 78)
    t = df.groupby("label").dead_count.agg(
        ["mean", "median", "std", "max", "count"]).round(4)
    print("  " + t.to_string().replace("\n", "\n  "))
    v = df.dead_count.values
    z = (v - v.mean()) / (v.std() + 1e-12)
    print(f"\n  corpus skew {stats.skew(v):+.2f}  kurtosis {stats.kurtosis(v):+.2f}")
    print(f"  slides beyond +3 SD: {(z > 3).sum()}   beyond +5 SD: {(z > 5).sum()}")
    kw = stats.kruskal(*[v[df.label == c] for c in CLASSES
                         if (df.label == c).sum() > 2])
    print(f"  Kruskal-Wallis across subtypes: p = {kw.pvalue:.2e}")
    ext = df.iloc[np.argsort(-z)[:10]]
    print("\n  most extreme slides (these are the ones to look at):")
    print("  " + ext[["image_id", "label", "dead_count", "dead_frac"]]
          .round(4).to_string(index=False).replace("\n", "\n  "))
    return list(ext.image_id)


def make_montage(root, slide_ids, out_dir, n_slides=6, per_slide=4):
    """Crop the highest-dead_count tiles next to low-dead_count tiles from the
    SAME slide. Same slide is the point: it holds staining constant, so a
    difference between the two rows is about the tissue, not the batch."""
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_chromatin_features import (load_detections, load_graph_geometry,
                                            open_slide, read_tile)
    gdir = Path(root) / RUN_DIR_NAME / GRAPH_SUBFOLDER
    tdir = Path(root) / RUN_DIR_NAME / "wsi_pyramid_tiff"
    ddir = Path(root) / RUN_DIR_NAME / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for iid in slide_ids[:n_slides]:
        gp = list(gdir.glob(f"{iid}_graph.npz"))
        if not gp or not (tdir / f"{iid}.tiff").exists():
            continue
        g = load_graph_geometry(gp[0])
        cx, cy, ct, bb = load_detections(ddir / iid / "cell_detection.json")
        tile_px = g["tile_px"]
        nr, nc = g["grid_shape"]
        node_of = np.full((nr, nc), -1, np.int64)
        node_of[g["grid_rc"][:, 0], g["grid_rc"][:, 1]] = np.arange(g["n_nodes"])
        gc = np.clip((cx // tile_px).astype(int), 0, nc - 1)
        gr = np.clip((cy // tile_px).astype(int), 0, nr - 1)
        nid = node_of[gr, gc]
        ok = nid >= 0
        n_dead = np.bincount(nid[ok & (ct == 4)], minlength=g["n_nodes"])
        n_all = np.bincount(nid[ok], minlength=g["n_nodes"])
        # low row must still be real tissue, else the comparison is with glass
        busy = n_all >= max(50, int(np.percentile(n_all, 60)))
        hi = np.argsort(-n_dead)[:per_slide]
        lo = [n for n in np.argsort(n_dead) if busy[n]][:per_slide]
        if not len(hi) or not len(lo):
            continue
        sl = open_slide(tdir / f"{iid}.tiff")
        S = 320
        im = Image.new("RGB", (S * per_slide, S * 2 + 26), "white")
        d = ImageDraw.Draw(im)
        for r, nodes in enumerate((hi, lo)):
            for c, node in enumerate(list(nodes)[:per_slide]):
                ox = int(g["grid_rc"][node, 1] * tile_px)
                oy = int(g["grid_rc"][node, 0] * tile_px)
                try:
                    rgb = read_tile(sl, ox, oy, tile_px)
                except Exception:
                    continue
                im.paste(Image.fromarray(rgb).resize((S, S), Image.BILINEAR),
                         (c * S, 26 + r * S))
                d.text((c * S + 4, 28 + r * S),
                       f"dead={n_dead[node]} / {n_all[node]} nuclei",
                       fill=(255, 0, 0) if r == 0 else (0, 100, 0))
        sl.close()
        d.text((4, 6), f"{iid}  {g['label']}   TOP ROW = highest dead_count, "
                       f"BOTTOM ROW = lowest (same slide, same staining)",
               fill=(0, 0, 0))
        p = out_dir / f"deadcount_{iid}_{g['label']}.png"
        im.save(p)
        made.append(p)
        print(f"  wrote {p.name}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag", default="v5nuc")
    ap.add_argument("--montage", type=int, default=0,
                    help="render image panels for the N most extreme slides")
    ap.add_argument("--clusters", type=int, default=4)
    args = ap.parse_args()

    gdir = Path(args.root) / RUN_DIR_NAME / GRAPH_SUBFOLDER
    odir = Path(args.root) / RUN_DIR_NAME / OUT_SUBFOLDER
    X, y, ids, fn = load_bags(gdir, odir / f"bag_cache_{args.tag}.pt", False,
                              None, nuclear_dir=gdir / "nuclear_feats")
    if "dead_count" not in fn:
        sys.exit("dead_count is not in the loaded feature set")

    df = slide_level(X, y, ids, fn)
    extreme = test_distribution(df)

    stain = load_stain(args.root)
    if stain is None or len(stain) < 50:
        n = 0 if stain is None else len(stain)
        print("\n" + "=" * 78)
        print(f"STAINING DESCRIPTORS: only {n} slides available")
        print("=" * 78)
        print("  Tests 1-3 need extract_chromatin_features.py to have finished")
        print("  (it writes the per-slide staining descriptors into")
        print("  chromatin_feats/_manifest.csv). Running the image panels only.")
    else:
        m = df.merge(stain, on="image_id", how="inner")
        print(f"\n  merged staining descriptors for {len(m)}/{len(df)} slides")
        cols = [c for c in STAIN_COLS if c in m.columns]
        test_correlation(m, cols)
        test_conditional(m, cols)
        test_batch(m, cols, k=args.clusters)
        m.to_csv(odir / "deadcount_slide_table.csv", index=False)
        print(f"\n  slide table -> {odir / 'deadcount_slide_table.csv'}")

    if args.montage:
        print("\n" + "=" * 78)
        print("TEST 4 -- the pictures. This is the one that settles it.")
        print("=" * 78)
        made = make_montage(args.root, extreme,
                            odir / "deadcount_montage", args.montage)
        if made:
            print(f"\n  {len(made)} panel(s) in {odir / 'deadcount_montage'}")
            print("\n  What to look for in the TOP row:")
            print("    * genuine necrosis: confluent eosinophilic debris, ghost")
            print("      outlines, karyorrhectic nuclear dust in sheets")
            print("    * artefact: crushed/smeared tissue at an edge, folds,")
            print("      out-of-focus haze, uniformly over-stained dark nuclei")
            print("      that look intact, ink, or bubbles")
            print("  The bottom row is the control -- same slide, same staining,")
            print("  so anything the two rows share is not what dead_count is")
            print("  keying on.")


if __name__ == "__main__":
    main()

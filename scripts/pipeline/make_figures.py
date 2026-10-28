# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Every figure in the paper, rendered from paper/numbers.json and the corpus.

WHY THIS EXISTS
---------------
The figures and the tables must not be able to disagree. paper_numbers.py reads
the per-fold result CSVs and writes numbers.json; this script reads only that
file (plus the corpus itself for the qualitative panel) and writes PDFs into
paper/figs/. Nothing here has a hand-typed result in it.

USAGE
    python scripts/pipeline/paper_numbers.py            # first -- writes numbers.json
    python scripts/pipeline/make_figures.py             # all figures
    python scripts/pipeline/make_figures.py --only 3    # one figure
    python scripts/pipeline/make_figures.py --no-slide  # skip fig 1's WSI panel (slow)

The slide panel reads a 2.4 GB pyramid PNG through pyvips (~40 s) and caches the
thumbnail as .npy, so re-running is instant.
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
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
import corpus_paths  # noqa: E402

ROOT = corpus_paths.ROOT
GDIR = ROOT / "cellvit_run" / "graphs"
FIGS = ROOT / "paper" / "figs"
CACHE = ROOT / "paper" / ".cache"
NUMBERS = ROOT / "paper" / "numbers.json"

# Okabe-Ito, colourblind-safe. Fixed roles so a colour means the same thing in
# every figure: the encoder is blue, the interpretable model is orange, the
# aligned deletion is vermillion, controls are grey.
C_ENC = "#0072B2"
C_INT = "#E69F00"
C_ALIGNED = "#D55E00"
C_NULL = "#999999"
C_CTL = "#CCCCCC"
C_ACCENT = "#009E73"
C_TEXT = "#222222"

ORDER = ["uni2h", "hibouL", "hoptimus0", "phikonv2"]
CLASSES = ["CC", "EC", "HGSC", "LGSC", "MC"]


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.labelcolor": C_TEXT,
        "text.color": C_TEXT,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figs/{name}.pdf")


# Figure 1 -- the aligned corpus and the three instruments
def fig1(N, want_slide=True):
    """Teaser: one real slide, its tile grid coloured by a named quantity, and
    the audit design. The point of panel (a) is that the SAME grid carries the
    embedding and the named columns, which is what makes the bridge and the
    subspace map well-posed."""
    # constrained_layout so the three panels' title baselines and the colourbar
    # are placed by the layout engine rather than by hand -- the previous
    # hand-tuned gridspec left panel (c) overflowing its axes.
    fig = plt.figure(figsize=(7.4, 2.55), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01, wspace=0.03)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 0.86, 1.6])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    sid, thumb, grid = load_slide(want_slide)

    ax_a.set_title(f"(a) WSI {sid}" + (f"  \u2014 {grid['label']}" if grid else ""),
                   loc="left")
    if thumb is not None:
        ax_a.imshow(crop_to_tissue(thumb))
    ax_a.set_xticks([]); ax_a.set_yticks([]); ax_a.grid(False)
    ax_a.set_anchor("N")     # both image panels hang from their title, so an
    for s in ax_a.spines.values():   # aspect-driven shrink cannot desync them
        s.set_visible(True)

    ax_b.set_title("(b) named feature per tile", loc="left")
    if grid is not None:
        img = grid["map"]
        m = np.ma.masked_invalid(img)
        im = ax_b.imshow(m, cmap="magma", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("nuclear area CV", fontsize=6, labelpad=1)
    ax_b.set_xticks([]); ax_b.set_yticks([]); ax_b.grid(False)
    ax_b.set_anchor("N")
    for s in ax_b.spines.values():
        s.set_visible(True)

    ax_c.set_title("(c) one grid, two representations", loc="left")
    ax_c.set_xlim(0, 10); ax_c.set_ylim(0, 10.4)
    ax_c.axis("off")
    ax_c.set_anchor("N")

    def box(x, y, w, h, txt, fc, ec, fs=6.6, weight="normal"):
        ax_c.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                 linewidth=0.9, zorder=2))
        ax_c.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                  fontsize=fs, zorder=3, weight=weight, linespacing=1.35)

    box(0.15, 8.3, 9.7, 1.6, "456 WSIs $\\rightarrow$ 256 $\\mu$m tile grid\n"
                             "(shared node order, checksum-verified)",
        "#F2F2F2", "#888888", 6.4)
    box(0.15, 5.3, 4.6, 2.0, "foundation-model\nembedding\n1024\u20131536-d",
        "#DCE9F5", C_ENC, 6.4)
    box(5.25, 5.3, 4.6, 2.0, "$\\approx$50 named\npathology\ncolumns",
        "#FBE9CF", C_INT, 6.4)
    box(0.15, 2.7, 9.7, 1.6, "identical Additive MIL head, folds,\n"
                             "selection rule and hyperparameters",
        "#F2F2F2", "#888888", 6.4)

    # The three instruments are three boxes, not one line of text inside one
    # box: a single long string overflowed its rectangle at this panel width.
    for i, name in enumerate(("substitution", "subspace\ndeletion", "bridge")):
        box(0.15 + i * 3.35, 0.25, 3.0, 1.5, name,
            "#E4F3EC", C_ACCENT, 6.4, "bold")

    def arrow(x, y0, y1):
        ax_c.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                       mutation_scale=7, color="#888888",
                                       linewidth=0.8, zorder=1))

    for x0 in (2.45, 7.55):
        arrow(x0, 8.25, 7.35)
        arrow(x0, 5.25, 4.35)
    for i in range(3):
        arrow(1.65 + i * 3.35, 2.65, 1.8)
    save(fig, "fig1_design")


def crop_to_tissue(thumb, pad=6):
    """Trim the pyramid TIFF's black padding off the thumbnail.

    These files are padded with pure BLACK outside the scanned region, not
    white, so an uncropped thumbnail is mostly dead frame; with imshow's equal
    aspect that also dictates the panel's shape and pushes its title away from
    the image. Keep the bounding box of anything that is not near-black.
    """
    a = np.asarray(thumb)
    if a.ndim != 3:
        return thumb
    keep = a[..., :3].max(axis=2) > 12
    if not keep.any():
        return thumb
    rs, cs = np.where(keep)
    r0, r1 = max(rs.min() - pad, 0), min(rs.max() + pad + 1, a.shape[0])
    c0, c1 = max(cs.min() - pad, 0), min(cs.max() + pad + 1, a.shape[1])
    return a[r0:r1, c0:c1]


def load_slide(want_slide=True):
    """A slide picked at RANDOM under a fixed seed, so the figure is not chosen
    to flatter the method. Returns (id, thumbnail, grid dict)."""
    import csv
    labels = {}
    with open(ROOT / "train.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            labels[r["image_id"]] = r["label"]

    rng = np.random.default_rng(7)
    ids = sorted(p.stem.replace("_graph", "")
                 for p in GDIR.glob("*_graph.npz"))
    ids = [i for i in ids if (ROOT / "train_images" / f"{i}.png").exists()
           and (GDIR / "nuclear_feats" / f"{i}_nuclei.npz").exists()]
    sid = str(rng.choice(ids))
    print(f"  fig1 slide (seeded random): {sid} [{labels.get(sid,'?')}]")

    with np.load(GDIR / f"{sid}_graph.npz", allow_pickle=True) as z:
        grc = z["node_grid_rc"]
        gshape = z["node_grid_shape"]
    with np.load(GDIR / "nuclear_feats" / f"{sid}_nuclei.npz",
                 allow_pickle=True) as z:
        names = [str(s) for s in z["nuc_feature_names"]]
        F = z["nuc_features"]
    col = names.index("neo_area_um2_cv") if "neo_area_um2_cv" in names else 0
    v = F[:, col].astype(float)
    v[v <= 0] = np.nan

    img = np.full((int(gshape[0]), int(gshape[1])), np.nan)
    img[grc[:, 0], grc[:, 1]] = v
    grid = dict(map=img, label=labels.get(sid, "?"))

    thumb = None
    if want_slide:
        CACHE.mkdir(parents=True, exist_ok=True)
        cp = CACHE / f"thumb_{sid}.npy"
        if cp.exists():
            thumb = np.load(cp)
        else:
            try:
                import pyvips
                im = pyvips.Image.thumbnail(
                    str(ROOT / "train_images" / f"{sid}.png"), 900)
                thumb = np.ndarray(buffer=im.write_to_memory(), dtype=np.uint8,
                                   shape=[im.height, im.width, im.bands])
                np.save(cp, thumb)
            except Exception as e:      # a missing thumbnail must not kill the run
                print(f"  !! thumbnail failed ({type(e).__name__}: {e})")
    return sid, thumb, grid


# Figure 2 -- the ledger
def fig2(N):
    """Where every arm lands. The message is the size and consistency of the
    gap, so the interpretable model gets a reference line rather than being one
    bar among many."""
    fig, ax = plt.subplots(figsize=(5.4, 2.9))

    rows = []
    sl = N.get("slidelevel", {})
    rows.append(("Slide-level RF", sl.get("rf"), C_CTL, "control"))
    rows.append(("Slide-level GBM", sl.get("gbm"), C_CTL, "control"))
    rows.append(("Slide-level logistic", sl.get("logreg"), C_CTL, "control"))
    rows.append(("Named features (MIL)", N["clean"]["mean"], C_INT, "interp"))
    for k in ORDER[::-1]:
        e = N["encoders"][k]
        rows.append((e["name"], e["additive"], C_ENC, "enc"))

    ys = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(ys, vals, color=[r[2] for r in rows], height=0.68,
            edgecolor="white", linewidth=0.6)
    ax.axvline(N["clean"]["mean"], color=C_INT, linestyle="--", linewidth=0.9,
               zorder=1)
    ax.axvline(0.20, color="#BBBBBB", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(0.205, len(rows) - 0.35, "chance", fontsize=6, color="#888888")

    for y, (_, v, _, kind) in zip(ys, rows):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=7)
        if kind == "enc":
            pass
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.15, 0.93)
    ax.set_xlabel("balanced accuracy (10 repeats $\\times$ 5 folds)")
    ax.set_title("Every encoder beats named morphometry by a wide margin",
                 loc="left")

    # the gap, annotated once
    top = len(rows) - 1
    e = N["encoders"]["uni2h"]
    ax.annotate("", xy=(e["additive"], top + 0.55),
                xytext=(N["clean"]["mean"], top + 0.55),
                arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=0.9))
    ax.text((e["additive"] + N["clean"]["mean"]) / 2, top + 0.78,
            f"+{e['gap_vs_clean']['delta']:.3f}", ha="center", fontsize=7,
            color=C_ACCENT, weight="bold")
    ax.set_ylim(-0.7, top + 1.15)
    save(fig, "fig2_ledger")


# Figure 3 -- subspace deletion, the central result
def fig3(N):
    """The paper's central measurement, on all four encoders.

    Ordering of the bars is deliberate: full, then the two CONTROLS, then the
    aligned deletion, so the eye reads 'nothing, nothing, then a drop'."""
    fig, ax = plt.subplots(figsize=(7.1, 3.2))

    labels = ["full encoder",
              "rank-matched null\n(mismatched slides)",
              "variance-matched\nrandom subspace",
              "morphometry-aligned\nsubspace deleted"]
    keys = ["additive", "residnull", "residctl", "resid"]
    cols = [C_ENC, C_NULL, C_CTL, C_ALIGNED]

    w = 0.2
    xs = np.arange(len(ORDER))
    for j, (k, c) in enumerate(zip(keys, cols)):
        vals = [N["encoders"][m][k] for m in ORDER]
        ax.bar(xs + (j - 1.5) * w, vals, width=w, color=c, edgecolor="white",
               linewidth=0.6, label=labels[j], zorder=3)

    ax.axhline(N["clean"]["mean"], color=C_INT, linestyle="--", linewidth=1.0,
               zorder=2)
    ax.text(-0.46, N["clean"]["mean"] + 0.005,
            f"named features, {N['clean']['mean']:.3f}", fontsize=6.5,
            color=C_INT, ha="left", va="bottom")

    # significance of THE TEST (aligned vs rank-matched null)
    for i, m in enumerate(ORDER):
        e = N["encoders"][m]
        r = e["resid_vs_null"]
        y = max(e["residnull"], e["resid"]) + 0.022
        x0, x1 = xs[i] - 0.5 * w, xs[i] + 1.5 * w
        ax.plot([x0, x0, x1, x1], [y, y + 0.008, y + 0.008, y], lw=0.8,
                color="#555555", zorder=4)
        ax.text(xs[i] + 0.5 * w, y + 0.012,
                f"{r['delta']:+.3f} {stars(r['p'])}", ha="center", fontsize=6.6,
                color="#333333")

    ax.set_xticks(xs)
    ax.set_xticklabels([N["encoders"][m]["name"] for m in ORDER])
    ax.set_ylim(0.60, 0.90)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Deleting the morphometry-aligned subspace costs accuracy;\n"
                 "deleting matched control subspaces does not", loc="left")
    ax.legend(loc="upper right", ncol=2, frameon=False, columnspacing=1.1,
              handlelength=1.2)
    save(fig, "fig3_deletion")


# Figure 4 -- the bridge
def fig4(N):
    """Raw R^2 per class, with the interpretable model's self-ceiling as a
    marker. We plot RAW R^2 rather than the percentage-of-ceiling because the
    ratio's denominator is unstable across repeats -- see paper_numbers.py."""
    fig, ax = plt.subplots(figsize=(6.2, 2.9))

    w = 0.19
    xs = np.arange(len(CLASSES))
    for j, m in enumerate(ORDER):
        b = N["bridge"].get(m)
        if not b:
            continue
        med = [b["per_class"][c]["r2_med"] for c in CLASSES]
        lo = [med[i] - b["per_class"][c]["r2_q25"] for i, c in enumerate(CLASSES)]
        hi = [b["per_class"][c]["r2_q75"] - med[i] for i, c in enumerate(CLASSES)]
        ax.bar(xs + (j - 1.5) * w, med, width=w, yerr=[lo, hi],
               error_kw=dict(lw=0.7, capsize=1.6, ecolor="#666666"),
               color=plt.cm.Blues(0.35 + 0.18 * j), edgecolor="white",
               linewidth=0.5, label=N["encoders"][m]["name"], zorder=3)

    ceil = [N["bridge"]["uni2h"]["per_class"][c]["ceil_med"] for c in CLASSES]
    ax.scatter(xs, ceil, marker="_", s=420, color=C_INT, linewidths=1.6,
               zorder=5, label="interpretable self-ceiling")

    ax.set_xticks(xs); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("$R^2$ of named features on\nper-tile decision contributions")
    ax.set_title("How much of each encoder's per-tile decision is expressible "
                 "in named terms", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.2)
    save(fig, "fig4_bridge")


# Figure 5 -- morphometry vs staining, the specificity check
def fig5(N):
    """The audit turned on itself. We removed a staining shortcut from the named
    features but could not remove it from an embedding, so we deleted the
    stain-aligned subspace by the same instrument. Morphometry deletion bites;
    stain deletion does not."""
    have = [m for m in ORDER if N["encoders"][m].get("stain_resid_vs_null")]
    if not have:
        print("  fig5 skipped -- no stain-audit arms complete yet")
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.7))

    xs = np.arange(len(have))
    w = 0.34
    morph = [abs(N["encoders"][m]["resid_vs_null"]["delta"]) for m in have]
    stain = [abs(N["encoders"][m]["stain_resid_vs_null"]["delta"]) for m in have]

    ax.bar(xs - w / 2, morph, width=w, color=C_ALIGNED, edgecolor="white",
           linewidth=0.6, label="morphometry-aligned", zorder=3)
    ax.bar(xs + w / 2, stain, width=w, color="#8C8C8C", edgecolor="white",
           linewidth=0.6, label="stain-aligned", zorder=3)

    for i, m in enumerate(have):
        for xoff, r in ((-w / 2, N["encoders"][m]["resid_vs_null"]),
                        (w / 2, N["encoders"][m]["stain_resid_vs_null"])):
            ax.text(xs[i] + xoff, abs(r["delta"]) + 0.0016, stars(r["p"]),
                    ha="center", fontsize=6.8, color="#333333")

    ax.set_xticks(xs)
    ax.set_xticklabels([N["encoders"][m]["name"] for m in have])
    ax.set_ylim(0, max(morph + stain) * 1.35)
    ax.set_ylabel("accuracy lost vs rank-matched null")
    ax.set_title("Deleting stain costs the encoder far less\nthan deleting "
                 "morphometry", loc="left")
    ax.legend(frameon=False, handlelength=1.2)
    save(fig, "fig5_stain")


# Figure 5 (grouped) -- the stain deletion in Figure 3's idiom
def fig5_grouped(N):
    """The specificity control, drawn exactly like Figure 3 so the two are read
    side by side: same four arms, same bar order, same axis, same bracket.

    Figure 3 deletes the morphometry-aligned subspace; this deletes the
    stain-aligned one by the identical instrument. Keeping the idiom identical
    is the point -- any difference the eye picks up is a difference in the
    result, not in the drawing."""
    fig, ax = plt.subplots(figsize=(7.1, 3.2))

    labels = ["full encoder",
              "rank-matched null\n(mismatched slides)",
              "variance-matched\nrandom subspace",
              "stain-aligned\nsubspace deleted"]
    keys = ["additive", "stain_residnull", "stain_residctl", "stain_resid"]
    cols = [C_ENC, C_NULL, C_CTL, C_ALIGNED]

    w = 0.2
    xs = np.arange(len(ORDER))
    for j, (k, c) in enumerate(zip(keys, cols)):
        vals = [N["encoders"][m][k] for m in ORDER]
        ax.bar(xs + (j - 1.5) * w, vals, width=w, color=c, edgecolor="white",
               linewidth=0.6, label=labels[j], zorder=3)

    ax.axhline(N["clean"]["mean"], color=C_INT, linestyle="--", linewidth=1.0,
               zorder=2)
    ax.text(-0.46, N["clean"]["mean"] + 0.005,
            f"named features, {N['clean']['mean']:.3f}", fontsize=6.5,
            color=C_INT, ha="left", va="bottom")

    # significance of THE TEST (stain-aligned vs rank-matched null)
    for i, m in enumerate(ORDER):
        e = N["encoders"][m]
        r = e["stain_resid_vs_null"]
        y = max(e["stain_residnull"], e["stain_resid"]) + 0.022
        x0, x1 = xs[i] - 0.5 * w, xs[i] + 1.5 * w
        ax.plot([x0, x0, x1, x1], [y, y + 0.008, y + 0.008, y], lw=0.8,
                color="#555555", zorder=4)
        ax.text(xs[i] + 0.5 * w, y + 0.012,
                f"{r['delta']:+.3f} {stars(r['p'])}", ha="center", fontsize=6.6,
                color="#333333")

    ax.set_xticks(xs)
    ax.set_xticklabels([N["encoders"][m]["name"] for m in ORDER])
    ax.set_ylim(0.60, 0.90)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Deleting the stain-aligned subspace, by the same instrument\n"
                 "and in the same layout as the morphometry deletion", loc="left")
    ax.legend(loc="upper right", ncol=2, frameon=False, columnspacing=1.1,
              handlelength=1.2)
    save(fig, "fig5_stain_grouped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--no-slide", action="store_true")
    a = ap.parse_args()

    style()
    N = json.loads(NUMBERS.read_text(encoding="utf-8"))
    todo = {1: lambda: fig1(N, not a.no_slide), 2: lambda: fig2(N),
            3: lambda: fig3(N), 4: lambda: fig4(N), 5: lambda: fig5(N),
            6: lambda: fig5_grouped(N)}
    for k, fn in todo.items():
        if a.only and k != a.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

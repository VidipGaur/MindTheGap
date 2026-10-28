# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Sanity check extract_nuclear_features.py output against the source graphs."""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import glob, json, sys
from pathlib import Path
import numpy as np
import corpus_paths

GD = corpus_paths.ROOT / "cellvit_run" / "graphs"
ND = GD / "nuclear_feats"

done = sorted(ND.glob("*_nuclei.npz"), key=lambda p: p.stat().st_mtime)
# skip the 3 newest -- a run is in flight, avoid a half-written file
done = done[:-3]
print(f"completed npz available for check: {len(done)}\n")

# pick 3 spread across labels
picked, seen = [], set()
for p in done:
    with np.load(p, allow_pickle=True) as z:
        lab = str(z["label"][0])
    if lab not in seen:
        seen.add(lab); picked.append((p, lab))
    if len(picked) == 3:
        break

BAD = []
def chk(cond, msg, slide=""):
    tag = "  ok  " if cond else " FAIL "
    if not cond:
        BAD.append(f"{slide}: {msg}")
    print(f"   [{tag}] {msg}")

for p, lab in picked:
    iid = p.name.split("_")[0]
    print("=" * 78)
    print(f"SLIDE {iid}  ({lab})")
    print("=" * 78)
    z = np.load(p, allow_pickle=True)
    g = np.load(GD / f"{iid}_graph.npz", allow_pickle=True)

    F = z["nuc_features"]
    names = [str(v) for v in z["nuc_feature_names"]]
    col = {n: i for i, n in enumerate(names)}
    flags = z["nuc_flag_cols"]
    gN = g["node_features"]
    gnames = [str(v) for v in g["node_feature_names"]]

    print(f"\n-- schema / alignment")
    print(f"   nuc_features {F.shape} {F.dtype}   graph node_features {gN.shape}")
    chk(F.shape[0] == gN.shape[0], f"node count matches graph ({F.shape[0]} == {gN.shape[0]})", iid)
    chk(F.shape[1] == len(names), f"n_cols == len(feature_names) ({F.shape[1]})", iid)
    chk(len(set(names)) == len(names), "feature names unique", iid)
    chk(int(z["grid_rc_checksum"][0]) == int(g["node_grid_rc"].sum()),
        "grid_rc checksum matches graph (same node ORDER)", iid)
    chk(tuple(z["grid_shape"]) == tuple(g["node_grid_shape"]), "grid_shape matches", iid)
    chk(str(z["label"][0]) == str(g["label"][0]), "label matches", iid)
    print(f"   mpp={float(z['mpp'][0])} ({str(z['mpp_source'][0])})  "
          f"coverage={float(z['coverage'][0]):.4f}  n_cells={int(z['n_cells'][0]):,}  "
          f"source={str(z['source'][0])}")

    print(f"\n-- numeric health")
    chk(np.isfinite(F).all(), f"all finite (nan={int(np.isnan(F).sum())}, inf={int(np.isinf(F).sum())})", iid)
    chk((F[:, ~flags] >= -1e-6).all(), "all non-flag features non-negative", iid)
    dead = [names[i] for i in range(F.shape[1]) if F[:, i].std() == 0]
    print(f"   constant columns ({len(dead)}): {dead}")

    print(f"\n-- row-order alignment vs graph density")
    cnt = F[:, col["nuc_count"]]
    d = gN[:, gnames.index("dens_all_mean")]
    rs = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rs(cnt), rs(d))[0, 1])
    # shuffled control
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(cnt))
    rho_sh = float(np.corrcoef(rs(cnt[perm]), rs(d))[0, 1])
    print(f"   spearman(nuc_count, dens_all_mean) = {rho:+.3f}   (shuffled control {rho_sh:+.3f})")
    chk(rho > 0.5, "row order aligned with graph (rho > 0.5)", iid)
    # also vs neoplastic_frac
    nf = F[:, col["neo_frac"]]
    gnf = gN[:, gnames.index("neoplastic_frac")]
    r2 = float(np.corrcoef(rs(nf), rs(gnf))[0, 1])
    print(f"   spearman(neo_frac, graph neoplastic_frac) = {r2:+.3f}")
    chk(r2 > 0.3, "neoplastic fraction agrees with graph's KDE-derived one", iid)

    print(f"\n-- internal consistency")
    n = F[:, col["nuc_count"]]
    nz = n > 0
    tsum = sum(F[:, col[f"{k}_count"]] for k in
               ["neo", "inflam", "connective", "epithelial", "dead"])
    chk(np.allclose(tsum[nz], n[nz], rtol=1e-4), "type counts sum to nuc_count", iid)
    untyped = int((n - tsum).sum())
    print(f"   cells with type outside TYPE_MAP: {untyped}")
    fsum = sum(F[:, col[f"{k}_frac"]] for k in
               ["neo", "inflam", "connective", "epithelial", "dead"])
    chk(np.allclose(fsum[nz], 1.0, atol=1e-3), "type fracs sum to 1 on non-empty tiles", iid)
    chk(np.allclose(fsum[~nz], 0.0), "type fracs are 0 on empty tiles", iid)
    chk(np.all(np.isin(F[:, flags], [0.0, 1.0])), "flag columns are strictly 0/1", iid)
    chk(np.allclose(n, np.round(n)), "nuc_count integral", iid)

    nm = F[:, col["neo_measured"]] > 0
    mm = F[:, col["morph_measured"]] > 0
    neo_cols = [c for c in names if c.startswith("neo_") and c != "neo_count" and c != "neo_frac" and c != "neo_measured"]
    chk(np.all(F[np.ix_(~nm, [col[c] for c in neo_cols])] == 0),
        "neo morph cols are 0 wherever neo_measured==0", iid)
    chk(np.all(F[:, col["neo_count"]][nm] >= 10), "neo_measured only where neo_count>=10", iid)
    chk(np.all(F[:, col["nuc_count"]][mm] >= 8), "morph_measured only where nuc_count>=8", iid)

    # cv == std/mean, p10<=p90, equivdiam vs area
    a_m = F[:, col["neo_area_um2_mean"]][nm]
    a_s = F[:, col["neo_area_um2_std"]][nm]
    a_cv = F[:, col["neo_area_um2_cv"]][nm]
    chk(np.allclose(a_cv, a_s / a_m, rtol=1e-3), "neo_area_cv == std/mean", iid)
    p10 = F[:, col["neo_area_um2_p10"]][nm]; p90 = F[:, col["neo_area_um2_p90"]][nm]
    chk(np.all(p10 <= p90), "p10 <= p90", iid)
    chk(np.all(p10 <= a_m) and np.all(a_m <= p90), "p10 <= mean <= p90", iid)
    ratio = F[:, col["neo_area_p90_p10_ratio"]][nm]
    chk(np.allclose(ratio, p90 / p10, rtol=1e-3), "p90_p10_ratio consistent", iid)
    ed = F[:, col["neo_equivdiam_um_mean"]][nm]
    ed_from_mean = 2 * np.sqrt(a_m / np.pi)
    # Jensen: mean(2*sqrt(a/pi)) <= 2*sqrt(mean(a)/pi)
    chk(np.all(ed <= ed_from_mean * 1.001), "equivdiam <= diam-of-mean-area (Jensen)", iid)
    print(f"   equivdiam/diam(mean area) ratio: {np.median(ed/ed_from_mean):.3f}")

    print(f"\n-- physical plausibility (mpp={float(z['mpp'][0])})")
    def rng_(c, lo, hi, mask=None):
        v = F[:, col[c]][mask if mask is not None else slice(None)]
        v = v[v != 0] if mask is None else v
        if len(v) == 0:
            print(f"   {c:32s} <all zero>"); return
        q = np.percentile(v, [5, 50, 95])
        ok = (v.min() >= lo - 1e-6) and (v.max() <= hi + 1e-6)
        print(f"   {c:32s} p5={q[0]:8.2f} med={q[1]:8.2f} p95={q[2]:8.2f} "
              f"min={v.min():8.2f} max={v.max():8.2f} {'' if ok else '<-- OUT OF RANGE'}")
        if not ok:
            BAD.append(f"{iid}: {c} outside [{lo},{hi}]")
    rng_("neo_area_um2_mean", 2, 900, nm)
    rng_("neo_area_um2_cv", 0, 5, nm)
    rng_("neo_equivdiam_um_mean", 1, 40, nm)
    rng_("neo_ecc_mean", 0, 1, nm)
    rng_("neo_axisratio_mean", 0, 1, nm)
    rng_("neo_circularity_mean", 0, 1.35, nm)
    rng_("neo_solidity_mean", 0, 1, nm)
    rng_("all_ecc_mean", 0, 1, mm)
    rng_("all_solidity_mean", 0, 1, mm)
    rng_("nnd_all_um_mean", 0, 200)
    rng_("nnd_neo_um_mean", 0, 200)
    rng_("til_frac_neo_infiltrated", 0, 1)
    rng_("til_inflam_per_neo", 0, 1e4)
    rng_("nuc_per_mm2", 0, 1e6)

    tile_px = float(np.asarray(g["tile_size_wsi_px"]).ravel()[0])
    mpp = float(z["mpp"][0])
    tile_um = tile_px * mpp
    tile_mm2 = (tile_um / 1000.0) ** 2
    print(f"\n   tile = {tile_px:.0f} px = {tile_um:.0f} um  ({tile_mm2:.4f} mm2)")
    implied = F[:, col["nuc_count"]] / tile_mm2
    chk(np.allclose(implied, F[:, col["nuc_per_mm2"]], rtol=1e-3),
        "nuc_per_mm2 == nuc_count / tile_area_mm2", iid)
    print(f"   max nuclei/tile = {int(n.max())}  -> {n.max()/tile_mm2:,.0f}/mm2; "
          f"packing limit at med area {np.median(F[:,col['all_area_um2_mean']][mm]):.0f}um2 "
          f"= {1e6/np.median(F[:,col['all_area_um2_mean']][mm]):,.0f}/mm2")

    print(f"\n-- coverage of the flags")
    for c in ["morph_measured", "neo_measured", "til_measured", "nuc_have_cv2"]:
        print(f"   {c:20s} {F[:,col[c]].mean()*100:5.1f}% of tiles")
    print(f"   empty tiles (nuc_count==0): {(n==0).mean()*100:.1f}%")
    z.close(); g.close()
    print()

print("=" * 78)
print(f"AGGREGATE over all {len(done)} completed slides")
print("=" * 78)
rows = []
for p in done:
    with np.load(p, allow_pickle=True) as z:
        F = z["nuc_features"]; names = [str(v) for v in z["nuc_feature_names"]]
        c = {n: i for i, n in enumerate(names)}
        nm = F[:, c["neo_measured"]] > 0
        w = F[:, c["neo_count"]][nm]
        def wm(k):
            v = F[:, c[k]][nm]
            return float(np.average(v, weights=w)) if len(v) and w.sum() > 0 else np.nan
        rows.append(dict(
            iid=str(z["image_id"][0]), label=str(z["label"][0]),
            mpp=float(z["mpp"][0]), cov=float(z["coverage"][0]),
            ncells=int(z["n_cells"][0]), ntiles=F.shape[0],
            finite=bool(np.isfinite(F).all()),
            neo_meas=float(nm.mean()),
            area=wm("neo_area_um2_mean"), cv=wm("neo_area_um2_cv"),
            ecc=wm("neo_ecc_mean"), sol=wm("neo_solidity_mean"),
            til=wm("til_frac_neo_infiltrated"),
        ))
import pandas as pd
df = pd.DataFrame(rows)
print(f"\nany non-finite file: {(~df.finite).sum()}")
print(f"coverage: min {df.cov.min():.3f} med {df.cov.median():.3f} "
      f"(<0.85 on {int((df.cov<0.85).sum())} slides)")
print(f"mpp values used: {df.mpp.value_counts().to_dict()}")
print(f"\nslides per class:\n{df.label.value_counts().to_string()}")
print("\nneoplastic-count-weighted per-slide means, by class:")
print(df.groupby("label")[["neo_meas", "area", "cv", "ecc", "sol", "til"]]
        .agg(["mean", "count"]).round(3).to_string())
print("\nHGSC vs others -- pleomorphism (area CV) is the whole point of P4:")
for k in ["area", "cv", "ecc", "sol"]:
    h = df[df.label == "HGSC"][k].dropna()
    o = df[df.label != "HGSC"][k].dropna()
    if len(h) > 1 and len(o) > 1:
        from scipy import stats
        u = stats.mannwhitneyu(h, o)
        d = (h.mean() - o.mean()) / np.sqrt((h.var() + o.var()) / 2)
        print(f"   {k:6s} HGSC {h.mean():7.3f} (n={len(h)})  other {o.mean():7.3f} "
              f"(n={len(o)})   cohen_d={d:+.2f}  p={u.pvalue:.3f}")

print("\n" + "=" * 78)
if BAD:
    print(f"{len(BAD)} FAILED CHECK(S):")
    for b in BAD:
        print("  -", b)
else:
    print("ALL HARD CHECKS PASSED")

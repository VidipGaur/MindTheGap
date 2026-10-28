# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Aggregate pass + the two issues the per-slide pass surfaced."""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import corpus_paths

GD = corpus_paths.ROOT / "cellvit_run" / "graphs"
ND = GD / "nuclear_feats"
done = sorted(ND.glob("*_nuclei.npz"), key=lambda p: p.stat().st_mtime)[:-3]

rows = []
zero_frac = {}
for p in done:
    with np.load(p, allow_pickle=True) as z:
        F = z["nuc_features"]; names = [str(v) for v in z["nuc_feature_names"]]
        c = {n: i for i, n in enumerate(names)}
        nm = F[:, c["neo_measured"]] > 0
        w = F[:, c["neo_count"]][nm]
        wm = lambda k: (float(np.average(F[:, c[k]][nm], weights=w))
                        if nm.any() and w.sum() > 0 else np.nan)
        for n_ in names:
            zero_frac.setdefault(n_, []).append(float((F[:, c[n_]] == 0).mean()))
        rows.append(dict(
            iid=str(z["image_id"][0]), label=str(z["label"][0]),
            mpp=float(z["mpp"][0]), mpp_src=str(z["mpp_source"][0]),
            coverage=float(z["coverage"][0]), ncells=int(z["n_cells"][0]),
            ntiles=F.shape[0], finite=bool(np.isfinite(F).all()),
            neo_meas=float(nm.mean()), morph_meas=float(F[:, c["morph_measured"]].mean()),
            til_meas=float(F[:, c["til_measured"]].mean()),
            empty_frac=float((F[:, c["nuc_count"]] == 0).mean()),
            nnd_neo_zero=float((F[:, c["nnd_neo_um_mean"]] == 0).mean()),
            area=wm("neo_area_um2_mean"), cv=wm("neo_area_um2_cv"),
            ecc=wm("neo_ecc_mean"), eccsd=wm("neo_ecc_std"),
            sol=wm("neo_solidity_mean"), p9010=wm("neo_area_p90_p10_ratio"),
            til=wm("til_frac_neo_infiltrated"),
        ))
df = pd.DataFrame(rows)

print("=" * 78)
print(f"AGGREGATE over {len(df)} completed slides")
print("=" * 78)
print(f"non-finite files      : {(~df.finite).sum()}")
print(f"coverage              : min {df.coverage.min():.3f}  med {df.coverage.median():.3f}"
      f"   (<0.85 on {int((df.coverage < 0.85).sum())} slides)")
print(f"mpp used              : {df.mpp.value_counts().to_dict()}  src {df.mpp_src.value_counts().to_dict()}")
print(f"slides w/ 0 empty tile: {int((df['empty_frac'] == 0).sum())}/{len(df)}")
print(f"\nclass counts: {df.label.value_counts().to_dict()}")

print("\n--- ISSUE 1: neo_measured coverage varies by class (shortcut risk) ---")
t = df.groupby("label")[["neo_meas", "morph_meas", "til_meas", "empty_frac"]].agg(["mean", "count"])
print(t.round(3).to_string())
sp = df.groupby("label").neo_meas.mean()
print(f"\n  between-class spread in neo_measured: {sp.max()-sp.min():.3f} "
      f"({sp.idxmax()} {sp.max():.3f} vs {sp.idxmin()} {sp.min():.3f})")
try:
    kw = stats.kruskal(*[g.neo_meas.values for _, g in df.groupby("label")])
    print(f"  Kruskal-Wallis across classes: H={kw.statistic:.1f}  p={kw.pvalue:.2e}")
except Exception as e:
    print("  kruskal failed:", e)

print("\n--- ISSUE 2: columns where 0 means 'not measured', with no flag ---")
zf = pd.Series({k: np.mean(v) for k, v in zero_frac.items()}).sort_values(ascending=False)
print("  mean fraction of tiles that are exactly 0, per column (top 14):")
print(zf.head(14).round(3).to_string())
print(f"\n  nnd_neo_um_mean is 0 on {df.nnd_neo_zero.mean()*100:.1f}% of tiles on average")
print(f"  but neo_measured is 0 on {(1-df.neo_meas.mean())*100:.1f}% -- the two do NOT coincide,")
print("  so nnd_neo is 'measured but on 2-9 nuclei' (very noisy) in the gap.")

print("\n--- discriminative check: do the morphometrics separate the classes at all? ---")
print("(neoplastic-count-weighted slide means)")
print(df.groupby("label")[["area", "cv", "p9010", "ecc", "eccsd", "sol", "til"]]
        .mean().round(3).to_string())
print("\nHGSC vs each other class, area CV (pleomorphism) and friends:")
for k in ["area", "cv", "p9010", "ecc", "eccsd", "sol"]:
    h = df[df.label == "HGSC"][k].dropna()
    line = f"  {k:6s} "
    for oc in ["CC", "EC", "LGSC", "MC"]:
        o = df[df.label == oc][k].dropna()
        if len(h) > 2 and len(o) > 2:
            d = (h.mean() - o.mean()) / np.sqrt((h.var() + o.var()) / 2)
            pv = stats.mannwhitneyu(h, o).pvalue
            line += f"| vs {oc:4s} d={d:+.2f} p={pv:.3f} "
        else:
            line += f"| vs {oc:4s} n/a "
    print(line)

print("\n--- univariate 5-way separability of each morphometric (slide level) ---")
res = []
for k in ["area", "cv", "p9010", "ecc", "eccsd", "sol", "til"]:
    sub = df[["label", k]].dropna()
    if sub[k].nunique() < 5:
        continue
    groups = [g[k].values for _, g in sub.groupby("label") if len(g) > 2]
    if len(groups) >= 3:
        kw = stats.kruskal(*groups)
        res.append((k, kw.statistic, kw.pvalue))
for k, h, pv in sorted(res, key=lambda r: -r[1]):
    print(f"  {k:6s}  H={h:6.2f}  p={pv:.4f}  {'*' if pv < 0.05 else ''}")
print("\n(n is small -- 76 slides, unbalanced -- treat as a smell test, not a result)")

# Copyright 2026 Vidip Gaur, Vidu Gaur
"""
Aggregate the per-slide explanations into a per-subtype picture.

    python scripts/dev/summarise_explanations.py --csv explanations_nuconly.csv

One slide's explanation is an anecdote. This asks whether the drivers are
CONSISTENT within a predicted subtype -- i.e. whether "the model calls HGSC
on nuclear pleomorphism" is a property of the model's HGSC behaviour or just
something that happened on one slide.

Restricted by default to slides where all seeds agreed AND the prediction was
correct: an explanation of a wrong or unstable prediction explains an error,
which is interesting separately but must not be pooled into the headline
claim.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)

import argparse
from pathlib import Path

import pandas as pd
import corpus_paths

GRP = {"grp_nuc_shape": "nuclear morphometry",
       "grp_nuc_spatial": "spacing / immune",
       "grp_nuc_density": "composition / density",
       "grp_nuc_flags": "availability",
       "grp_graph": "KDE / streamline"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(corpus_paths.ROOT))
    ap.add_argument("--csv", default="explanations_nuconly.csv")
    ap.add_argument("--include-wrong", action="store_true")
    args = ap.parse_args()

    p = Path(args.root) / "cellvit_run/graphs/gatv2_run" / args.csv
    df = pd.read_csv(p)
    print(f"{len(df)} slides in {p.name}")
    print(f"  seeds agree: {df.seeds_agree.sum()}  "
          f"correct: {(df.pred == df.truth).sum()}")

    d = df if args.include_wrong else \
        df[(df.seeds_agree) & (df.pred == df.truth)]
    print(f"  using {len(d)} slides"
          f"{'' if args.include_wrong else ' (confident + correct only)'}\n")

    gcols = [c for c in GRP if c in d.columns]
    print("=" * 74)
    print("MEAN GROUP ATTRIBUTION BY PREDICTED SUBTYPE")
    print("(probability lost when the whole group is replaced by corpus median)")
    print("=" * 74)
    t = d.groupby("pred")[gcols].mean().round(3)
    t.columns = [GRP[c] for c in t.columns]
    print(t.to_string())
    print(f"\n  n per subtype: {d.groupby('pred').size().to_dict()}")

    print("\n" + "=" * 74)
    print("MOST FREQUENT TOP-1 DRIVER BY PREDICTED SUBTYPE")
    print("=" * 74)
    for cls, g in d.groupby("pred"):
        vc = g.feat1.value_counts().head(3)
        print(f"\n  {cls}  (n={len(g)})")
        for name, cnt in vc.items():
            share = cnt / len(g)
            zmean = g.loc[g.feat1 == name, "feat1_z"].mean()
            print(f"    {name:34s} top driver on {cnt:3d} slides "
                  f"({share*100:4.1f}%), mean {zmean:+.1f} SD")

    print("\n" + "=" * 74)
    print("CONFIDENCE AND ATTENTION CONCENTRATION")
    print("=" * 74)
    s = df.groupby(["pred"]).agg(
        n=("slide", "size"), conf=("conf", "mean"),
        attn_top10=("attn_top10", "mean"),
        seed_agree=("seeds_agree", "mean")).round(3)
    print(s.to_string())

    if not args.include_wrong:
        w = df[(df.pred != df.truth)]
        if len(w):
            print("\n" + "=" * 74)
            print(f"ERRORS ({len(w)} slides) -- confidently wrong is the "
                  f"dangerous class")
            print("=" * 74)
            cw = w[w.seeds_agree & (w.conf > 0.7)]
            print(f"  all seeds agreed AND confidence>0.7 while WRONG: "
                  f"{len(cw)} slides")
            if len(cw):
                print("  " + cw[["slide", "truth", "pred", "conf", "feat1",
                                 "feat1_z"]].head(12).to_string(index=False)
                      .replace("\n", "\n  "))
                print("\n  A high |SD| on the top driver here means the call was")
                print("  made on an extreme feature value -- check those slides")
                print("  for extraction artefacts before trusting the feature.")


if __name__ == "__main__":
    main()

# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Match Ben Taeb SVS files to labels.csv and report per-class counts.

labels.csv uses wsi_id like OOU_198 / VOA_351 / VS_30682 and subtype in
{HGSC, LGSC, CC, MC, EN}.  EN is endometrioid == UBC-OCEAN's EC.
Filenames carry the same case id without the underscore (10_OOU198_EC-SC.svs).
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import csv
import re
import sys
from collections import Counter
from pathlib import Path
import corpus_paths

ROOT = corpus_paths.ROOT / "OCDatasetPaperVidu"
BT = ROOT / "total_data"
LAB = ROOT / "labels.csv"
PROBE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bentaeb_slides.csv")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bentaeb_manifest.csv")

# labels.csv subtype -> UBC-OCEAN CLASSES
MAP = {"HGSC": "HGSC", "LGSC": "LGSC", "CC": "CC", "MC": "MC", "EN": "EC"}


def norm_case(s):
    """OOU_198 -> OOU198 ; also uppercases (one file is 'OOu297')."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def case_from_fname(name):
    """10_OOU198_EC-SC.svs -> OOU198 ; 40_test_VS30682.svs -> VS30682."""
    stem = Path(name).stem
    for tok in stem.split("_"):
        m = re.fullmatch(r"(OOU|VOA|VS)(\d+)[A-Z]?", tok.upper())
        if m:
            return m.group(1) + m.group(2)
    return None


def main():
    labels = {}
    with open(LAB, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels[norm_case(r["wsi_id"])] = (r["Set"], r["subtype"].strip())
    print(f"labels.csv: {len(labels)} cases")

    slides = []
    with open(PROBE, newline="", encoding="utf-8") as fh:
        slides = list(csv.DictReader(fh))
    print(f"probe csv : {len(slides)} svs files\n")

    rows, unmatched, unused = [], [], set(labels)
    for s in slides:
        case = case_from_fname(s["fname"])
        hit = labels.get(case) if case else None
        if hit is None:
            unmatched.append((s["fname"], case))
            continue
        unused.discard(case)
        split, sub = hit
        rows.append(dict(
            case=case, fname=s["fname"], split=s["split"], label_set=split,
            subtype_raw=sub, label=MAP[sub],
            width=s["width"], height=s["height"], mpx=s["mpx"],
            mpp=s.get("mpp_x") or s.get("aperio_mpp"),
            objective=s.get("objective") or s.get("appmag"),
            levels=s["levels"]))

    print(f"matched   : {len(rows)}")
    print(f"unmatched files ({len(unmatched)}):")
    for f, c in unmatched:
        print(f"    {f}   (parsed case={c})")
    print(f"label rows with no file ({len(unused)}): {sorted(unused)}\n")

    print("=" * 62)
    print("PER-CLASS COUNTS (UBC-OCEAN class names)")
    print("=" * 62)
    tot = Counter(r["label"] for r in rows)
    bysplit = Counter((r["label"], r["split"]) for r in rows)
    print(f"  {'class':6s} {'total':>6s} {'Training':>9s} {'Test':>6s}   "
          f"{'UBC-OCEAN':>10s}")
    ubc = dict(HGSC=201, EC=105, CC=81, LGSC=36, MC=33)
    for c in ("CC", "EC", "HGSC", "LGSC", "MC"):
        print(f"  {c:6s} {tot[c]:6d} {bysplit[(c,'Training')]:9d} "
              f"{bysplit[(c,'Test')]:6d}   {ubc[c]:10d}")
    print(f"  {'TOTAL':6s} {sum(tot.values()):6d} "
          f"{sum(v for (c,s),v in bysplit.items() if s=='Training'):9d} "
          f"{sum(v for (c,s),v in bysplit.items() if s=='Test'):6d}   "
          f"{sum(ubc.values()):10d}")

    print("\n" + "=" * 62)
    print("RESOLUTION / GEOMETRY")
    print("=" * 62)
    mpps = Counter(r["mpp"] for r in rows)
    for m, n in mpps.most_common():
        print(f"  mpp={m:<22s} n={n}")
    objs = Counter(r["objective"] for r in rows)
    for o, n in objs.most_common():
        print(f"  objective={o:<16s} n={n}")
    # tile counts at the pipeline's ACTUAL grid: 1024 px tiles
    print("\n  tiles per slide at tile_size_wsi_px=1024 (full lattice, "
          "pre tissue-gating):")
    tl = sorted(int(r["width"]) // 1024 * (int(r["height"]) // 1024)
                for r in rows)
    import statistics
    print(f"    min {tl[0]}  p25 {tl[len(tl)//4]}  median "
          f"{statistics.median(tl):.0f}  p75 {tl[3*len(tl)//4]}  max {tl[-1]}")
    px = sorted(float(r["mpx"]) for r in rows)
    print(f"  megapixels: min {px[0]:.0f}  median {statistics.median(px):.0f}  "
          f"max {px[-1]:.0f}   (UBC-OCEAN median 1520)")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

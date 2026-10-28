# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Write <root>/cellvit_run/cellvit_manifest.csv for the given slides.

Columns follow the manifest (image_id, label, image_width,
image_height, is_tma, file_path). The KDE and graph builders read image_id,
label, image_height and image_width. No mpp column is written: the UBC-OCEAN
manifest had none, and corpus_paths and the KDE builder would pick one up.

Dimensions come from train.csv and are checked against the pyramid TIFF. A
mismatch stops the script: train.csv and the graphs agreed in the original run, so a mismatch here
needs a person.

Usage:
    python smoke/build_manifest.py [--root UBC_ROOT] ID [ID ...]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

COLUMNS = ["image_id", "label", "image_width", "image_height", "is_tma", "file_path"]


def build_rows(train_rows, ids, tiff_dir, tiff_dims=None):
    """Manifest rows for `ids`; `tiff_dims(path) -> (width, height)` or None to skip the check."""
    out, problems = [], []
    for iid in ids:
        r = train_rows.get(iid)
        if r is None:
            problems.append(f"{iid}: not in train.csv")
            continue
        if str(r["is_tma"]) != "False":
            problems.append(f"{iid}: is_tma={r['is_tma']}; the cohort is WSI-only")
            continue
        path = tiff_dir / f"{iid}.tiff"
        if tiff_dims is not None:
            w, h = tiff_dims(path)
            if (w, h) != (int(r["image_width"]), int(r["image_height"])):
                problems.append(f"{iid}: TIFF is {w}x{h}, train.csv says "
                                f"{r['image_width']}x{r['image_height']}")
                continue
        out.append(dict(image_id=iid, label=r["label"], image_width=r["image_width"],
                        image_height=r["image_height"], is_tma=r["is_tma"],
                        file_path=str(path)))
    return out, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--root", default=os.environ.get("UBC_ROOT"))
    args = ap.parse_args(argv)
    if not args.root:
        ap.error("--root or UBC_ROOT is required")

    import openslide

    root = Path(args.root)
    run = root / "cellvit_run"
    with open(root / "train.csv", newline="") as fh:
        train = {r["image_id"]: r for r in csv.DictReader(fh)}
    rows, problems = build_rows(train, args.ids, run / "wsi_pyramid_tiff",
                                lambda p: openslide.OpenSlide(str(p)).dimensions)
    for p in problems:
        print("PROBLEM", p)
    if problems:
        return 1
    with open(run / "cellvit_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {run / 'cellvit_manifest.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

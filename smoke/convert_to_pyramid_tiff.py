# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Convert UBC-OCEAN PNG slides to pyramidal BigTIFF with pyvips: tiled 256 px,
pyramid, BigTIFF, JPEG quality 90.

Usage:
    python smoke/convert_to_pyramid_tiff.py [--root UBC_ROOT] ID [ID ...]

Reads <root>/train_images/<id>.png and writes
<root>/cellvit_run/wsi_pyramid_tiff/<id>.tiff. Each output is reopened with
OpenSlide and its level-0 size compared with train.csv; the exit status is 1 if
any slide fails. A PNG that is not 3-band is refused, because how to
handle an alpha channel is not specified.
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--root", default=os.environ.get("UBC_ROOT"))
    args = ap.parse_args(argv)
    if not args.root:
        ap.error("--root or UBC_ROOT is required")

    import openslide
    import pyvips

    root = Path(args.root)
    with open(root / "train.csv", newline="") as fh:
        rows = {r["image_id"]: r for r in csv.DictReader(fh)}
    out_dir = root / "cellvit_run" / "wsi_pyramid_tiff"
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for iid in args.ids:
        src = root / "train_images" / f"{iid}.png"
        dst = out_dir / f"{iid}.tiff"
        if not dst.exists():
            img = pyvips.Image.new_from_file(str(src), access="sequential")
            if img.bands != 3:
                print(f"{iid}: {img.bands} bands, refusing to guess how to convert")
                failed += 1
                continue
            part = dst.with_name(dst.name + ".part")
            img.tiffsave(str(part), tile=True, tile_width=256, tile_height=256,
                         pyramid=True, bigtiff=True, compression="jpeg", Q=90)
            part.rename(dst)
        w, h = openslide.OpenSlide(str(dst)).dimensions
        want = (int(rows[iid]["image_width"]), int(rows[iid]["image_height"]))
        ok = (w, h) == want
        failed += not ok
        print(f"{iid}: {dst.name} {w}x{h} (train.csv {want[0]}x{want[1]}) "
              f"{'ok' if ok else 'DIMENSION MISMATCH'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

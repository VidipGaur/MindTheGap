# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Probe the Ben Taeb SVS slides: dimensions, native MPP, magnification,
level structure -- and compare against the UBC-OCEAN grid actually used
(tile_size_wsi_px = 1024 px, i.e. 512 um at 0.5 um/px).

Read-only. Writes a CSV to the scratchpad.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import csv
import os
import re
import sys
import traceback
from pathlib import Path
import corpus_paths

BT = corpus_paths.ROOT / "OCDatasetPaperVidu" / "total_data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bentaeb_slides.csv")


def open_slide(path):
    try:
        import openslide
        return openslide.OpenSlide(str(path)), "openslide"
    except Exception:
        import tiffslide
        return tiffslide.TiffSlide(str(path)), "tiffslide"


def probe(path, split):
    row = dict(split=split, fname=path.name, bytes=path.stat().st_size)
    try:
        sl, backend = open_slide(path)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:160]
        return row
    try:
        row["backend"] = backend
        p = dict(sl.properties)
        w, h = sl.level_dimensions[0]
        row["width"] = w
        row["height"] = h
        row["mpx"] = round(w * h / 1e6, 1)
        row["levels"] = sl.level_count
        row["downsamples"] = ";".join(f"{d:.2f}" for d in sl.level_downsamples)
        row["level_dims"] = ";".join(f"{a}x{b}" for a, b in sl.level_dimensions)
        for k, tgt in (("openslide.mpp-x", "mpp_x"), ("openslide.mpp-y", "mpp_y"),
                       ("openslide.objective-power", "objective"),
                       ("aperio.AppMag", "appmag"), ("aperio.MPP", "aperio_mpp"),
                       ("tiff.ResolutionUnit", "res_unit"),
                       ("openslide.vendor", "vendor")):
            if k in p:
                row[tgt] = p[k]
        row["n_props"] = len(p)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:160]
    finally:
        try:
            sl.close()
        except Exception:
            pass
    return row


def main():
    files = []
    for split in ("Training", "Test"):
        d = BT / split / "svs"
        if d.exists():
            for f in sorted(d.glob("*.svs")):
                files.append((f, split))
    print(f"{len(files)} svs files\n", flush=True)

    rows = []
    for i, (f, split) in enumerate(files, 1):
        r = probe(f, split)
        rows.append(r)
        print("[%3d/%d] %-34s %-8s %6sx%-6s mpp=%-8s obj=%-5s lv=%-2s %s" % (
            i, len(files), r["fname"][:34], split,
            r.get("width", "?"), r.get("height", "?"),
            r.get("mpp_x", r.get("aperio_mpp", "?")),
            r.get("objective", r.get("appmag", "?")),
            r.get("levels", "?"), r.get("error", "")), flush=True)

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

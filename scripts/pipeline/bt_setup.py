# Copyright 2026 Vidip Gaur, Vidu Gaur
r"""Stage 0 -- build the Ben Taeb parallel corpus scaffold.

Mirrors the UBC-OCEAN layout exactly:

    bentaeb_run/
        cellvit_manifest.csv      image_id,label,image_width,image_height,mpp,...
        detections/<image_id>/    -> junction to the real CellViT output folder
        kde_maps/  graphs/  nuclear_feats/  chromatin_feats/  eosin_feats/  fm_feats/

image_id is the sanitised slide stem, so the Ben Taeb folder naming
(`10_OOU198_EC-SC`) never has to match UBC-OCEAN's numeric ids -- the manifest
carries the mapping and nothing downstream re-derives it.

SCALE.  UBC-OCEAN slides are natively 0.50 um/px and the pipeline runs its
tiling geometry at a NOMINAL 0.25 um/px (build_streamline_graphs_bridge.WSI_MPP),
which yields tile_size_wsi_px = 1024 and a 512 um physical tile.  To put Ben Taeb
on the identical physical grid the same rule is applied per slide:

    nominal_mpp = native_mpp / 2

so a 20x slide (0.4992) gets 0.2496 -> 1024 px tiles, and the single 40x slide
(0.2520) gets 0.1260 -> 2048 px tiles.  Both are ~512 um.  Physical measurements
(nuclear area etc.) use the NATIVE mpp, matching extract_nuclear_features.py's
DEFAULT_MPP_WSI=0.50 choice for UBC-OCEAN.
"""

# scripts/ is split into four folders; keep bare-name imports working from any of them.
import sys as _sys
from pathlib import Path as _Path
for _d in ("pipeline", "analysis", "verify", "dev"):
    _p = str(_Path(__file__).resolve().parent.parent / _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
import corpus_paths

ROOT = corpus_paths.ROOT
BT = ROOT / "OCDatasetPaperVidu"
DET_SRC = BT / "cellvit_detections"
RUN = ROOT / "bentaeb_run"
SCRATCH = Path(__file__).parent
PROBE = SCRATCH / "bentaeb_slides.csv"

MAP = {"HGSC": "HGSC", "LGSC": "LGSC", "CC": "CC", "MC": "MC", "EN": "EC"}


def norm_case(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def case_from_stem(stem):
    for tok in stem.split("_"):
        m = re.fullmatch(r"(OOU|VOA|VS)(\d+)[A-Z]?", tok.upper())
        if m:
            return m.group(1) + m.group(2)
    return None


def sanitise(stem):
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def main():
    labels = {}
    with open(BT / "labels.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels[norm_case(r["wsi_id"])] = (r["Set"], r["subtype"].strip())

    with open(PROBE, newline="", encoding="utf-8") as fh:
        probe = list(csv.DictReader(fh))

    det_dirs = {d.name: d for d in DET_SRC.iterdir() if d.is_dir()}

    rows, skipped = [], []
    for s in probe:
        stem = Path(s["fname"]).stem
        case = case_from_stem(stem)
        hit = labels.get(case) if case else None
        if hit is None:
            skipped.append((stem, "no label row"))
            continue
        if stem not in det_dirs:
            skipped.append((stem, "no CellViT detection folder"))
            continue

        native = s.get("mpp_x") or s.get("aperio_mpp")
        if not native:
            skipped.append((stem, "no mpp in slide properties"))
            continue
        native = float(native)
        nominal = native / 2.0

        rows.append(dict(
            image_id=sanitise(stem),
            label=MAP[hit[1]],
            image_width=int(s["width"]),
            image_height=int(s["height"]),
            mpp=round(nominal, 6),          # NOMINAL -- drives tiling geometry
            native_mpp=round(native, 6),    # TRUE    -- drives physical units
            objective=s.get("objective") or s.get("appmag") or "",
            split=s["split"],
            label_set=hit[0],
            case=case,
            slide_stem=stem,
            slide_path=str(BT / "total_data" / s["split"] / "svs" / s["fname"]),
            det_src=str(det_dirs[stem]),
        ))

    rows.sort(key=lambda r: r["image_id"])

    RUN.mkdir(exist_ok=True)
    for sub in ("detections", "kde_maps", "graphs"):
        (RUN / sub).mkdir(exist_ok=True)
    for sub in ("nuclear_feats", "chromatin_feats", "eosin_feats", "fm_feats"):
        (RUN / "graphs" / sub).mkdir(exist_ok=True)

    with open(RUN / "cellvit_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # link commands: junctions on Windows (run from PowerShell; junctions need
    # no admin), symbolic links elsewhere (run with bash)
    lines = []
    for r in rows:
        link = RUN / "detections" / r["image_id"]
        if not link.exists():
            if os.name == "nt":
                lines.append(f'New-Item -ItemType Junction -Path "{link}" '
                             f'-Target "{r["det_src"]}" | Out-Null')
            else:
                lines.append(f'ln -s "{r["det_src"]}" "{link}"')
    link_script = SCRATCH / ("bt_junctions.ps1" if os.name == "nt" else "bt_symlinks.sh")
    link_script.write_text("\n".join(lines), encoding="utf-8")

    print(f"manifest rows : {len(rows)}")
    print(f"skipped       : {len(skipped)}")
    for stem, why in skipped:
        print(f"    {stem:<40s} {why}")

    print("\nper-class counts")
    tot = Counter(r["label"] for r in rows)
    bysplit = Counter((r["label"], r["split"]) for r in rows)
    ubc = dict(HGSC=201, EC=105, CC=81, LGSC=36, MC=33)
    print(f"  {'class':6s} {'total':>6s} {'Train':>6s} {'Test':>6s} {'UBC-OCEAN':>10s}")
    for c in ("CC", "EC", "HGSC", "LGSC", "MC"):
        print(f"  {c:6s} {tot[c]:6d} {bysplit[(c,'Training')]:6d} "
              f"{bysplit[(c,'Test')]:6d} {ubc[c]:10d}")
    print(f"  {'TOTAL':6s} {sum(tot.values()):6d}")

    print("\nnominal mpp (tiling geometry) -> implied tile_size_wsi_px")
    mm = Counter()
    for r in rows:
        # 256.0 is NODE_SIZE_UM of the graph builder, a nominal um figure at the
        # tiling pixel size r["mpp"]; the tile in pixels is the real quantity.
        tile_px = round(256.0 / (r["mpp"] * 16) * 16)
        mm[(r["mpp"], r["native_mpp"], tile_px, tile_px * r["native_mpp"])] += 1
    for (nom, nat, tpx, um), n in sorted(mm.items()):
        print(f"  native={nat:<8.4f} nominal={nom:<8.4f} tile={tpx:5d} px "
              f"= {um:6.1f} um   n={n}")

    print(f"\nwrote {RUN / 'cellvit_manifest.csv'}")
    print(f"wrote {link_script}  ({len(lines)} junctions to create)")


if __name__ == "__main__":
    main()

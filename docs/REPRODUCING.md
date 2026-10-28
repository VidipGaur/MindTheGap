# Reproducing the pipeline

**None of the stages below has been re-run on the full cohort since the data
loss.** The commands are the ones the recovered scripts document; the costs are
the ones printed in the recovered logs, each sourced below. Anything without a log is
marked UNVERIFIED. The only stages exercised after the loss are the offline
tests (`pytest`) and the graph builder's `--self-test`.

## Environment

Python 3.12 (the only version ever used). On Ubuntu 24.04:

```bash
sudo apt install libvips42t64 libopenslide0          # verified to exist with apt-cache policy
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt -r requirements.txt  # CUDA 12.1 build of torch
# or, CPU only: pip install -r requirements.txt
pip install -r requirements-dev.txt                    # tests, linting
```

`openslide` and `pyvips` install from pip but import only once the system
libraries are present. No module needs them at import time except
`probe_wsi.py`; the extractors load them inside the functions that read slides.

### Environment variables

| Variable | Purpose | bash | PowerShell |
|---|---|---|---|
| `UBC_ROOT` | Dataset root: `train.csv`, `train_images/`, `cellvit_run/` | `export UBC_ROOT=/data/UBC-OCEAN` | `$env:UBC_ROOT = "D:\data\UBC-OCEAN"` |
| `UBC_RUN_DIR` | Run folder below the root, default `cellvit_run`; `bentaeb_run` for the external corpus | `export UBC_RUN_DIR=bentaeb_run` | `$env:UBC_RUN_DIR = "bentaeb_run"` |
| `HF_TOKEN` | Only for the gated encoders (UNI2-h, H-optimus-0, Hibou-L) | `export HF_TOKEN=...` | `$env:HF_TOKEN = "..."` |
| `HF_HOME` | Hugging Face cache location | `export HF_HOME=/data/hf` | `$env:HF_HOME = "D:\data\hf"` |
| `CELLVIT_CACHE` | Where the CellViT checkpoint is downloaded | `export CELLVIT_CACHE=/data/cellvit` | `$env:CELLVIT_CACHE = "D:\data\cellvit"` |

With `UBC_ROOT` unset the root is `./data/UBC-OCEAN` relative to the directory
the first script was started from.

**`UBC_RUN_DIR` reaches the extraction stages only.** `train_abmil.py`,
`fm_subspace_split.py` and `bridge_fm_interpretable.py` always read
`<root>/cellvit_run/`. To run the training side on another corpus, point
`UBC_ROOT` at a separate directory instead.

## Layout under the root

```
$UBC_ROOT/
  train.csv  test.csv  train_images/<id>.png
  cellvit_run/
    cellvit_manifest.csv
    wsi_pyramid_tiff/<id>.tiff
    detections/<id>/{cells.json, cell_detection.json, *.geojson}
    kde_maps/<id>_kde.npz
    graphs/<id>_graph.npz
    graphs/{nuclear_feats, chromatin_feats, eosin_feats, fm_feats}/
    graphs/gatv2_run/          result CSVs, abmil_ckpt/, logs/, bag caches
  paper/numbers.json  paper/figs/
```

## Stages

Run from the repository root. GPU means a CUDA device was used originally. Scripts
live in `scripts/pipeline/`, `scripts/analysis/`, `scripts/verify/` and `scripts/dev/`;
the commands below give the full path.

| # | Stage | Command | Reads | Writes | Hardware | Recorded cost |
|---|---|---|---|---|---|---|
| 1 | Download | `kaggle competitions download -c UBC-OCEAN` (rules must be accepted) | | `train.csv`, `train_images/` | network | UNVERIFIED |
| 2 | PNG to pyramid TIFF | `python smoke/convert_to_pyramid_tiff.py ID ...` (new, written after the loss) | `train_images/` | `wsi_pyramid_tiff/` | CPU | UNVERIFIED |
| 3 | Manifest | `python smoke/build_manifest.py ID ...` (new) | `train.csv`, TIFFs | `cellvit_manifest.csv` | CPU | - |
| 4 | Cell detection | `bash smoke/run_cellvit.sh ID ...` (new; recorded CellViT flags) | TIFFs | `detections/<id>/` | GPU, ~12 GB RAM | largest slide (1.79M cells) about 64 min [1] |
| 5 | Density maps | `python scripts/pipeline/build_kde_density_maps_new.py [--only ID]` | detections, manifest | `kde_maps/` | CPU | UNVERIFIED |
| 6 | Tile graphs | `python scripts/pipeline/build_streamline_graphs_bridge.py [--only ID]` | KDE maps, manifest | `graphs/<id>_graph.npz` | CPU | 456 slides about 3 h 45 min [2] |
| 7 | Nuclear features | `python scripts/pipeline/extract_nuclear_features.py --check`, then without `--check` | detections, graphs | `graphs/nuclear_feats/` | CPU | about 40 s per slide [3] |
| 8 | Chromatin features | `python scripts/pipeline/extract_chromatin_features.py --check`, then `--workers 3` | TIFFs, detections, graphs | `graphs/chromatin_feats/` | CPU | about 30 s per slide at 3 workers [4] |
| 9 | Eosin features | `python scripts/pipeline/extract_eosin_features.py --check`, then without | TIFFs, detections, graphs | `graphs/eosin_feats/` | CPU | 129 s per slide after optimisation [5] |
| 10 | Encoder embeddings | `python scripts/pipeline/embed_tiles_fm.py --check --limit 2`, then `--models phikonv2 uni2h hoptimus0 hibouL` | TIFFs, graphs | `graphs/fm_feats/<id>_<model>.npz` | GPU | tile reading about 56 s per slide; three encoders together about 44 s per slide of GPU, about 5.5 h for the cohort [6] |
| 11 | Clean interpretable model | `python scripts/pipeline/run_cleanfinal.py` | all feature blocks | result CSVs, checkpoints | GPU | UNVERIFIED |
| 12 | Encoder arms and subspace deletion | `python scripts/pipeline/run_multiencoder.py --dry-run`, then without | embeddings, features | result CSVs, residual embeddings | GPU | UNVERIFIED |
| 13 | Phikon-v2 subspace deletion | `python scripts/pipeline/run_subspace.py` | embeddings, features | residual embeddings, result CSVs | GPU | UNVERIFIED |
| 14 | Stain audit | `python scripts/pipeline/run_stainaudit.py [--models uni2h]` | embeddings, features | result CSVs | GPU | UNVERIFIED |
| 15 | Bridge, per repeat | `python scripts/pipeline/queue_bridge_refit.py --wait-pid 0` (`--smoke` first) | checkpoints, features | `bridge_repeats_*` | GPU | UNVERIFIED |
| 16 | Paper numbers | `python scripts/pipeline/paper_numbers.py` | result CSVs | `paper/numbers.json` | CPU | - |
| 17 | Figures | `python scripts/pipeline/make_figures.py [--no-slide]` | `numbers.json`, one slide PNG | `paper/figs/*.pdf` | CPU | - |
| 18 | External check | `python scripts/pipeline/bentaeb_probe.py scripts/pipeline/bentaeb_slides.csv` (writes the slide table `bt_setup.py` reads), then `python scripts/pipeline/bt_setup.py`, then `python scripts/pipeline/run_bentaeb.py`, `python scripts/pipeline/eval_external.py --out "$UBC_ROOT/bentaeb_run/instrument1.json"`, `python scripts/pipeline/per_class_numbers.py` | Ben Taeb slides under `$UBC_ROOT/OCDatasetPaperVidu/` | `bentaeb_run/` | GPU | embeddings for 80 slides took 54.6 min [7] |

Every training run uses 10 repeats x 5 folds with
`StratifiedKFold(5, shuffle=True, random_state=repeat)`.

Pass `--wait-pid 0` to the two `queue_*` scripts: their defaults are process ids
from the original session (see `docs/LIMITATIONS.md` item 7). After any change
to the feature files, pass `--rebuild-cache` to `train_abmil.py`; its bag cache
does not notice new files on disk.

### Sources for the recorded costs

1. An earlier project-state note in the private recovery bundle. No log is
   shipped, so the figure is unverified.
2. The author's timing of the full 456-slide rebuild after the
   backward-integration fix. No log of it is part of this repository, so the
   figure is unverified.
3. A session log in the private recovery bundle (about 45 s per slide on the
   external corpus, from a later session). No log is shipped, so the figure is
   unverified.
4. The author's timing with three worker processes. No log is shipped, so the
   figure is unverified.
5. The author's timing. The eosin extractor originally took 225 s per slide; it
   dropped to 129 s per slide once the tile-level statistics were computed on a
   copy of the tile strided by 2, with only the nucleus bounding boxes
   deconvolved at full resolution. Values changed by about 0.003. No log is
   shipped, so the figures are unverified.
6. A session log in the private recovery bundle. Phikon-v2 alone ran at
   140 tiles/s on the GPU, so tile reading, not the GPU, set the pace; that rate
   is the author's measurement and no log is shipped.
7. A session log in the private recovery bundle. No log is shipped, so the
   figure is unverified.

## Offline checks

```bash
pytest                                   # needs no data, no GPU, no network
python tools/audit_paths.py              # Windows paths left in scripts/
python tools/scan_secrets.py scripts && python tools/scan_secrets.py --strict tools tests smoke docs
```

## Five-slide smoke test

See `smoke/README.md`. It needs a GPU, Kaggle credentials and the system
libraries above. It was run once, on 2026-10-23; see `docs/SMOKE_REPORT.md`.

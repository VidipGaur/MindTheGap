# Mind the Gap: code for a controlled audit of pathology foundation models on ovarian carcinoma subtyping

Code for the manuscript "Mind the Gap: A Controlled Audit of Pathology
Foundation Models on Ovarian Carcinoma Subtyping". It asks how much of what a
pathology foundation model knows can be expressed in named, human-readable
pathology quantities. Four encoders (UNI2-h, H-optimus-0, Hibou-L, Phikon-v2)
and a model built on named per-tile features (nuclear morphometry, chromatin,
eosin and cytoplasm) are trained through the same Additive MIL head on 456
UBC-OCEAN whole-slide images. Three instruments then compare them:
representation substitution, subspace deletion with rank- and variance-matched
nulls, and a bridge regression from named features to the encoders'
per-tile contributions.

> **Status.** Research code accompanying an unpublished manuscript. The scripts
> were recovered after a data-loss event (see [PROVENANCE](docs/PROVENANCE.md)).
> They have not been re-run end to end on the full 456-slide cohort since the
> loss. A 5-slide smoke test ran on the lab GPU machine on 2026-10-23: the
> pipeline ran on 5 slides, 355 of 363 checks passed, 0 failed (see
> [SMOKE_REPORT](docs/SMOKE_REPORT.md)). It reports no accuracy. Numbers in the
> manuscript derive from earlier runs whose raw outputs no longer exist.

## Pipeline

```
UBC-OCEAN PNG slide
  -> pyramidal JPEG TIFF                smoke/convert_to_pyramid_tiff.py   (re-implemented after the loss)
  -> CellViT cell detection             smoke/run_cellvit.sh               (re-implemented after the loss)
  -> KDE density maps                   scripts/pipeline/build_kde_density_maps_new.py
  -> tile graph (512 um tiles)          scripts/pipeline/build_streamline_graphs_bridge.py   (reconstruction, tier B)
  -> named per-tile feature blocks
       nuclear morphometry + immune     scripts/pipeline/extract_nuclear_features.py         (partial, tier C)
       chromatin / haematoxylin         scripts/pipeline/extract_chromatin_features.py
       eosin / cytoplasm                scripts/pipeline/extract_eosin_features.py
  -> encoder tile embeddings            scripts/pipeline/embed_tiles_fm.py
  -> Additive MIL, 10 repeats x 5 folds scripts/pipeline/train_abmil.py
  -> the three instruments              scripts/pipeline/run_multiencoder.py, run_subspace.py, fm_subspace_split.py,
                                        bridge_fm_interpretable.py, queue_bridge_refit.py, run_stainaudit.py
  -> paper numbers and figures          scripts/pipeline/paper_numbers.py -> numbers.json -> scripts/pipeline/make_figures.py
  -> external check (80 slides, frozen) scripts/pipeline/run_bentaeb.py, eval_external.py, per_class_numbers.py
```

The tile graph is used only to define the tiles and their order. Its edges played
no part in the final models.

## Installation

Python 3.12 is required (the only version used). On Ubuntu 24.04:

```bash
sudo apt install libvips42t64 libopenslide0
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt -r requirements.txt   # CUDA 12.1 build of torch
pip install -r requirements-dev.txt                       # tests and linting
```

`requirements.txt` alone gives a CPU-capable install, which is enough for the
tests. Versions are the ones logged on the original machine; see
[PROVENANCE](docs/PROVENANCE.md) for the record and the few that were not logged.
Cell detection and encoder embeddings need an NVIDIA GPU.

## Environment variables

| Variable | Purpose | bash | PowerShell |
|---|---|---|---|
| `UBC_ROOT` | Dataset root (default `./data/UBC-OCEAN`) | `export UBC_ROOT=/data/UBC-OCEAN` | `$env:UBC_ROOT = "D:\data\UBC-OCEAN"` |
| `UBC_RUN_DIR` | Run folder below the root (default `cellvit_run`); extraction stages only | `export UBC_RUN_DIR=bentaeb_run` | `$env:UBC_RUN_DIR = "bentaeb_run"` |
| `HF_TOKEN` | Gated encoders only | `export HF_TOKEN=...` | `$env:HF_TOKEN = "..."` |
| `HF_HOME` | Hugging Face cache | `export HF_HOME=/data/hf` | `$env:HF_HOME = "D:\data\hf"` |
| `CELLVIT_CACHE` | CellViT checkpoint folder | `export CELLVIT_CACHE=/data/cellvit` | `$env:CELLVIT_CACHE = "D:\data\cellvit"` |

Stage-by-stage commands, inputs, outputs and recorded run times are in
[REPRODUCING](docs/REPRODUCING.md).

## Data

The slides are not redistributed. UBC-OCEAN is available from the Kaggle
competition of that name after accepting its rules. The external corpus is from
Ben Taeb et al. No per-tile features, embeddings or checkpoints are released.
Model weights are downloaded from their own repositories under their own terms
([THIRD_PARTY](docs/THIRD_PARTY.md)).

## Tests

```bash
pytest
```

The suite needs no data, no GPU and no network. It covers imports and `--help`
for every script that can be imported without data, path resolution, the
Additive MIL identity (per-tile contributions sum exactly to the bag logit), the
subspace-deletion algebra, stain deconvolution, feature-name contracts, the
graph builder's self-test, and a synthetic 25-slide corpus run through the real
graph builder and training-side loaders.

## Five-slide smoke test

`smoke/run_smoke.sh` downloads one slide per subtype, runs every extraction
stage, and checks shapes, alignment and the counts logged on the original
machine. It needs a GPU, Kaggle credentials and the system libraries. See
[smoke/README](smoke/README.md). It was run once on 2026-10-23; the result is in
[SMOKE_REPORT](docs/SMOKE_REPORT.md).

## Where the paper's results come from

Labels are the LaTeX labels of the manuscript source and may change before
publication.

| Paper item | Computed by | From results written by |
|---|---|---|
| Study design figure (`fig:design`) | `make_figures.py` (`fig1`) | `numbers.json` and one slide drawn under a fixed seed |
| Representation gap: figure and table (`fig:ledger`, `tab:ledger`) | `paper_numbers.py`, `make_figures.py` (`fig2`) | `run_cleanfinal.py` (clean interpretable model), `run_multiencoder.py` (encoders) |
| Named features added to an encoder (`subsec:hybrid`) | `paper_numbers.py` | `run_multiencoder.py` (hybrid arms) |
| Subspace deletion: figure and table (`fig:deletion`, `tab:subspace`) | `paper_numbers.py`, `make_figures.py` (`fig3`) | `run_subspace.py`, `run_multiencoder.py`, both via `fm_subspace_split.py` |
| Bridge: figure and table (`fig:bridge`, `tab:bridge`) | `paper_numbers.py`, `make_figures.py` (`fig4`) | `queue_bridge_refit.py` via `bridge_fm_interpretable.py` |
| Stain specificity figure (`fig:stain`) | `make_figures.py` (`fig5`, `fig5_grouped`) | `run_stainaudit.py` |
| Seed disagreement and abstention (`tab:abstain`) | `stability_vs_ensemble.py`, `stability_table.py` (matched by content; UNVERIFIED) | checkpoints |
| Slide-level control | `paper_numbers.py` | `train_slidelevel_gbm.py` |
| Model-selection rules | `paper_numbers.py` | `run_overnight.py` (stage A) |
| Artefact and batch controls (`subsec:artefacts`) | `resolve_dead_count.py`, `batch_residual.py`, `probe_stain_nonlinear.py` (UNVERIFIED) | feature blocks |
| External check, 80 slides, frozen inference (`subsec:external`) | `eval_external.py`, `per_class_numbers.py` | `run_bentaeb.py` (mapping UNVERIFIED) |

## Hardware and run time

The original runs used one Windows workstation with an NVIDIA RTX 2000 Ada
(16 GB) and 16 GB of system RAM, which was the limit for CellViT. Recorded
examples: building the 456 tile graphs took about 3 h 45 min; nuclear features
about 40 s per slide; encoder embeddings about 5.5 h for three encoders over the
cohort. Sources and the other stages are in [REPRODUCING](docs/REPRODUCING.md).

## Repository layout

```
scripts/
  pipeline/  stages that make data or results, and the runners that chain them (22)
  analysis/  post-hoc analyses, paired comparisons and the overnight and phase drivers (18)
  verify/    the two synthetic self-checks the test suite runs, and a slide probe (3)
  dev/       development-time probes, sanity checks and superseded comparisons (33);
             nothing in the paper's pipeline needs them
tests/     offline test suite
smoke/     five-slide smoke harness
tools/     path audit, secret scan
docs/      provenance, reproduction, inventory, limitations, licences
```

Scripts import each other by bare module name. Each one that does adds the four
folders to `sys.path`, so any script runs directly from the repository root, for
example `python scripts/pipeline/train_abmil.py --help`.

## Citation

See [CITATION.cff](CITATION.cff). The manuscript is in preparation; cite it once
published. Repository: <https://github.com/VidipGaur/MindTheGap>.

## Licence

Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Model weights
and datasets keep their own terms; see [THIRD_PARTY](docs/THIRD_PARTY.md).

## Provenance and status

The original workstation's disk was overwritten in October 2026. The slides,
all intermediate files, checkpoints and result tables were lost. The code was
rebuilt from the logs of two working sessions by replaying every file write and
edit in order, and checked against every later read of each file. Two files are
less certain: the graph builder is a reconstruction that fills unread gaps from
an earlier complete version, and the nuclear-feature extractor holds only what
was read. Some early scripts were never recovered; their stages are
re-implemented in `smoke/` from recorded settings. The portability pass changed
only paths, imports, process handling and packaging. Every change was reviewed
as a diff against the recovered originals, which stay in the authors' private
staging repository because the recovery bundle contains session logs. After
that pass, one function was changed on purpose: `fm_subspace_split.named_features()`
now gives the same 49 columns on a fresh dataset root as the original cached
runs did. That change was written after the loss and checked on synthetic data
only ([NAMED_FEATURES_CACHE](docs/NAMED_FEATURES_CACHE.md)). The full account is
in [PROVENANCE](docs/PROVENANCE.md); known limitations are in
[LIMITATIONS](docs/LIMITATIONS.md).

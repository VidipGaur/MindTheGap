# Five-slide smoke test

Runs the extraction pipeline on one slide per subtype and checks that the outputs
are well formed, aligned with each other, and close to the values logged on the
original machine. It does not train a model and reports no accuracy: five slides
cannot support the 5-fold cross-validation the paper uses.

**Status: run once on real slides (2026-10-23, RTX 2000 Ada 16 GB).** The checker
(`verify_smoke_outputs.py`) and the loader check (`integration_check.py`) are also
tested offline on synthetic files (`tests/test_verify_smoke_outputs.py`,
`tests/test_integration_synthetic.py`). The result and the deviations are in
`docs/SMOKE_REPORT.md`. On a 16 GB machine CellViT on a slide of about 940k cells
needed `RAY_memory_usage_threshold=0.99`.

## Requirements

* An NVIDIA GPU with a working driver (CellViT-SAM-H was run on a 16 GB card).
* `sudo apt install libvips42t64 libopenslide0` (Ubuntu 24.04 names).
* The project venv (`requirements-gpu.txt` + `requirements.txt`), plus
  `pip install kaggle` and the `cellvit` package.
* Kaggle API credentials (`~/.kaggle/kaggle.json`, mode 600, or `KAGGLE_USERNAME`
  and `KAGGLE_KEY`) and the UBC-OCEAN competition rules accepted on that account.
* At least 60 GB free under the smoke root.
* Optional: `HF_TOKEN` with accepted terms for UNI2-h, H-optimus-0 and Hibou-L.
  Phikon-v2 is ungated and is enough for the test.

## Running it

```bash
export UBC_ROOT=/data/UBC-OCEAN-smoke     # fresh directory, outside the repository
export CELLVIT_CACHE=/data/cellvit-cache  # where the CellViT checkpoint is downloaded
bash smoke/run_smoke.sh
```

PowerShell users set the same variables with `$env:UBC_ROOT = "..."`; the shell
scripts themselves need bash.

The run refuses a non-empty `UBC_ROOT` it did not create, and never deletes
anything. Each stage logs to `$UBC_ROOT/cellvit_run/logs/smoke_<stage>.log`, with
stderr in a separate `.err.log`.

## Files

| File | What it does |
|---|---|
| `slides.json` | The panel, the logged values and their sources, widths and tolerances |
| `fetch_slides.sh` | `train.csv`, `test.csv` and the five PNGs from Kaggle, one at a time |
| `convert_to_pyramid_tiff.py` | PNG to tiled pyramidal BigTIFF, JPEG q90 (written after the loss) |
| `build_manifest.py` | `cellvit_manifest.csv` (written after the loss) |
| `run_cellvit.sh` | CellViT-SAM-H with the recorded flags and `--wsi_mpp 0.25` (written after the loss) |
| `run_smoke.sh` | Runs every stage in order |
| `integration_check.py` | Loads the bags through the training-side loaders; one Additive MIL forward pass |
| `verify_smoke_outputs.py` | Checks every output against `slides.json`; exit 1 on any failure |

## Outputs

* `$UBC_ROOT/cellvit_run/smoke_verify.json` and `logs/smoke_verify.log`: the check table.
* `$UBC_ROOT/cellvit_run/integration_plain.json`: the loader check with the
  flag-leak diagnostic, which is expected to refuse one slide per class.
* `$UBC_ROOT/cellvit_run/integration.json`: the loader check without it.
* `$UBC_ROOT/cellvit_run/logs/cellvit_runs.tsv`: CellViT version, GPU, wall time
  per slide and the checkpoint hash.

Record the results in `docs/SMOKE_REPORT.md`.

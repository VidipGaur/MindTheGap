# Five-slide smoke report

The pipeline ran on 5 slides; 355 of 363 checks passed; 0 failed. The other 8 were
2 warnings (constant columns) and 6 with no logged value to compare with (listed below).
The 355 include all four encoders (Phikon-v2, UNI2-h, H-optimus-0, Hibou-L).

This is a run of the extraction pipeline. It reports no accuracy and trains nothing:
five slides cannot support cross-validation.

| Item | Value |
|---|---|
| Date | 2026-10-23 |
| Host | Intel Core i7-14700, 14 GiB RAM |
| GPU, driver, CUDA | NVIDIA RTX 2000 Ada 16 GB, 595.99.02, 13.2 (`nvidia-smi`) |
| Python, torch | 3.12.15, 2.5.1+cu121 |
| `uv pip freeze` SHA-256 | `71091c48be4a87be5d50b5c9ae509e05f741cb4822b9ea2a0b57c7899ac22473` |
| CellViT | 1.0.9 from PyPI (version at time of re-run, not original) |
| CellViT checkpoint | `CellViT-SAM-H-x40-AMP.pth`, SHA-256 `356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4`; MD5 `cdbd74fd...` equals the Zenodo record 15094831 value |
| Code tested | the export as of 2026-10-23, with the named-features fix; `scripts/` was later arranged into folders with no change in logic |

The environment probe and the full check table (the output of
`smoke/verify_smoke_outputs.py`) are kept in the authors' private staging
repository and are not published; rerunning the verifier on a fresh smoke run
regenerates the table.

## Slides

The five PNGs were placed by the author, so Kaggle was not used. `train.csv`
agrees with the panel on label and `is_tma` (False) for all five, and with the PNG
headers on width and height.

## Logged values against observed

Observed values are from the output of `smoke/verify_smoke_outputs.py` (not shipped). "Logged" is the value in `smoke/slides.json`.
Tolerances are first guesses, not derived.

| Slide | Label | n_cells logged / observed | n_tiles | coverage | Spearman | median area um2 |
|---|---|---|---|---|---|---|
| 10548 | CC | 241,934 / 241,934 | 600 / 600 | 0.9796 / 0.9796 | 0.946 / 0.9455 | not logged / 27.5 |
| 10077 | EC | 941,496 / 941,496 | 971 / 971 | 0.9532 / 0.9532 | 0.890 / 0.890 | 38.1 / 38.1 |
| 38535 | HGSC | 327,555 / 327,555 | 760 / 760 | 0.9854 / 0.9854 | 0.938 / 0.938 | 47.8 / 47.8 |
| 5852 | LGSC | 495,915 / 495,915 | 634 / 634 | 0.9771 / 0.9771 | 0.879 / 0.879 | 28.6 / 28.6 |
| 14312 | MC | not logged / 137,858 | not logged / 658 | not logged / 0.9855 | not logged / 0.938 | not logged / 21.5 |

The four slides with logged values reproduced them to the printed precision, well
inside the tolerances. That is a stronger match than the tolerances require, and
it holds for the Tier B graph builder (node counts) and the nuclear extractor
(coverage, Spearman, median area). It says nothing about the parts of those files the
original sessions never read: a gap would have to show up in a value that was logged.

Shuffled-control Spearman: 0.044 (10548), 0.027 (10077), -0.051 (38535), 0.039 (5852),
0.0005 (14312), all within the logged range of -0.02 to +0.05 except 38535 at -0.051,
which is inside the checker's |rho| <= 0.1 limit but 0.001 outside the logged range.

Exact oracle for 10077, all eight values equal: `node_features` (971, 39), `edge_attr`
(6484, 23), `edge_index` (2, 6484), `node_grid_shape` [39, 52], `wsi_shape` [39150, 53143],
`tile_size_wsi_px` 1024, 5 channel names, nuclear block (971, 39).

## Other results

* Conversion: each TIFF opens with OpenSlide and its dimensions equal `train.csv`
  (width and height not swapped), for all five.
* Graph builder `--self-test`: passed, 8 checks printed (numbered 1 and 3 to 9; there is no 2).
* Nuclear `--check` alignment verdict: PASS, coverage min 0.953, rho min 0.88.
* Embeddings, device cuda, run in two passes. Phikon-v2 first (no token needed); then UNI2-h,
  H-optimus-0 and Hibou-L once the author supplied a Hugging Face token. The script prints and
  asserts each model's normalisation:

  | Model | Dim | mean | std |
  |---|---|---|---|
  | phikonv2 | 1024 | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
  | uni2h | 1536 | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
  | hoptimus0 | 1536 | 0.707, 0.579, 0.704 | 0.212, 0.230, 0.177 |
  | hibouL | 1024 | 0.707, 0.576, 0.722 | 0.195, 0.232, 0.182 |

  The checker verifies, for all 20 files (5 slides x 4 encoders): width, row count equal to the
  graph's node count, `grid_rc_checksum` equal to the graph's, finite values, float32 and the
  stored label. All passed.
* Loader integration (`smoke/integration_check.py`): `named_features("morph")` returns 49
  columns and `named_features("stain")` 10 on this fresh root; the same slides are in both
  blocks; the Additive MIL check on 10077 gives max |sum of tile contributions + bias - logit| = 0.
  The plain run (with the flag-leak diagnostic) was refused by the loader with
  `ValueError: n_splits=5 cannot be greater than the number of members in each class.`
  That is the limit of a five-slide corpus, not a pipeline fault.
  Because the branch has the fix, the known bug on the branch without it was not exercised.

## Not passed or not compared

* WARN, 38535 and 5852: `dead_count` and `dead_frac` are constant. The KDE logs show
  `Dead=0` cells for both slides, so this is a property of the data.
* No logged value, so not compared (N/A): 14312 `n_tiles`, `n_cells`, `coverage`,
  `spearman` and `median area`, and the 10548 median area. Observed values are in the table above.
* Not run: Additive MIL cross-validation and any accuracy figure.
* Not done for the gated encoders: the integration check (`integration_check.py`) was run
  before they existed and was not repeated; it reads the named-feature blocks, not embeddings.

## Wall time

CellViT, per slide (`logs/cellvit_runs.tsv`):

| Slide | Seconds | Note |
|---|---|---|
| 10548 | 783 | the first attempt (1346 s) failed, see deviations |
| 38535 | 930 | |
| 5852 | 825 | |
| 14312 | 938 | |
| 10077 | 916 (failed), 1718 (retry, passed) | see deviations |

Other stages, seconds (a slide's stage output that already existed was skipped, so some
stage totals cover fewer than five slides):

| Stage | Seconds |
|---|---|
| PNG to TIFF | pilot 10548 under 90; the other four 86 in total |
| KDE | 10548 7, 38535 8, 5852 10, 10077 18, 14312 5 |
| Graph | 10548 32, 38535 27, 5852 26, 10077 26, 14312 24 |
| Nuclear `--check`, five slides | 74 (10077 33, 10548 8, 14312 4, 38535 13, 5852 16) |
| Nuclear, four slides | 69 |
| Chromatin, four slides | 175 (10548 alone: 31) |
| Eosin, four slides | 126 (10548 alone: 23) |
| Embeddings, Phikon-v2, four slides | 46 (10548 alone: about 18) |
| Embeddings, all four encoders, five slides | 240 (about 70 per slide: image read 16 to 27, then Phikon-v2 3 to 5, UNI2-h 8 to 13, H-optimus-0 14 to 22, Hibou-L 4 to 7) |

Per-slide times for chromatin, eosin and embeddings on the four batch slides were not recorded separately.

## Setup notes

1. Kaggle was not used; the PNGs and `train.csv` were copied in by hand (the
   originals were left untouched). `test.csv` was absent; the nuclear extractor ran without it.
2. Hugging Face token: a token for the three gated encoders was passed only as an environment variable to the
   embedding commands. It is not in any file, log or commit (grepped for in the repo, the run logs and the temp
   directory). The model terms were already accepted on that account.
3. Python 3.12 came from uv (Ubuntu 26.04 has no `python3.12` package), and packages were installed with `uv pip`, not `pip`.
4. `cellvit` pulled `setuptools` down to 65.6.3, which broke `import timm`; it was restored to 78.1.0. `uv pip check` reports the resulting `pathopatch` conflict.
5. The first CellViT download of the checkpoint (via the package) stalled at 1.43 of 2.80 GB, and the package then failed on the truncated file
   (`failed finding central directory`). The file was completed with `curl -C -` from the URL in `cellvit/utils/cache_models.py`
   and its MD5 checked against Zenodo.
6. CellViT on 10077 was killed on the first attempt by Ray's memory monitor (RAM 14.08 of 14.80 GB, `ray.exceptions.OutOfMemoryError`).
   The retry used `RAY_memory_usage_threshold=0.99` and no other change. It passed; RAM used peaked at about 15.0 GB of 15.2 GB
   in the final merge step, with swap taking the overflow. No CellViT flag was changed.
7. A 32 GiB swap file and `vm.swappiness=10` were added. The machine froze on first use: the kernel log shows an out-of-memory kill of a Firefox
   content process (about 12.8 GB resident) with swap full, followed by about 13 minutes of the desktop not responding. Firefox was closed for the runs.

## Harness fixes

* `smoke/fetch_slides.sh`: exits early when `train.csv` and the panel PNGs are already under the root. Before, it stopped on missing Kaggle credentials.
* `smoke/run_cellvit.sh`: reads the CellViT version through `importlib.metadata`; the old `python -m pip show` printed "not installed" in a venv without pip.
  Runs before this fix have "not installed" in `cellvit_runs.tsv`; the version is 1.0.9.

Nothing in `scripts/` was changed. The checker `verify_smoke_outputs.py` was not edited; it already checks the 10077 oracle.

## Open points

1. `run_cellvit.sh` does not set `RAY_memory_usage_threshold`. On a 16 GB machine a slide of 10077's size needs it. Whether the script should set it, or leave it as an environment setting the operator chooses, is open.
2. `pathopatch` declares `setuptools<=65.6.3`, which Python 3.12 cannot import. The run used setuptools 78.1.0; whether the lock file should record that is open.
3. Cell counts and tile counts equal the logged values exactly on four slides. Whether that is expected for a deterministic CellViT, or a sign that the logs came from the same checkpoint and version, is not decided here.

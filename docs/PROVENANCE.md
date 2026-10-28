# Provenance

## What happened to the code

The analysis behind "Mind the Gap" ran from March to early October 2026 on one
Windows workstation with an NVIDIA RTX 2000 Ada (16 GB) and 16 GB of RAM.

* **March-April 2026.** Exploratory work on about 80 whole-slide images:
  orientation fields, "streamline tractography" over cell-density fields, and
  graph attention networks.
* **August 2026.** The pipeline was rebuilt on UBC-OCEAN: CellViT cell detections,
  kernel density maps, streamline graphs and a GATv2 classifier. A bug was found
  and fixed (backward streamline integration had been a no-op). The graph
  approach was later set aside because its edges added nothing.
* **Late August - early September 2026.** The project moved to Additive MIL on
  named per-tile features, with a foundation-model positive control.
* **September 2026.** Three more encoders and the three audit instruments
  (representation substitution, subspace deletion, the bridge regression), a
  stain audit, and a frozen-inference check on an external corpus (Ben Taeb et
  al., 80 slides).
* **Early-to-mid October 2026 (exact date not recorded).** Linux was installed
  over the workstation's Windows disk. Everything on that disk was lost. The
  last recovered session event is dated 2026-10-05.

### Lost

All copies of the UBC-OCEAN slides and the converted pyramid TIFFs (the slides
can be downloaded again from Kaggle); the CellViT outputs; the density maps,
graphs and every per-tile feature block; the foundation-model embeddings; all
model checkpoints; the per-fold result CSVs and `numbers.json`; and the
out-of-fold predictions, which were only ever held in memory for one repeat.

### Survived

* The source code, rebuilt from the logs of two working sessions by replaying every
  file write and edit in time order, plus the shell commands that patched files.
  Session 1 ran from 2026-08-31 to 2026-10-05 (4,948 events: main experiments,
  ablations, stain audit, paper); session 2 from 2026-09-16 to 2026-10-05
  (1,275 events: the external corpus, `corpus_paths.py`, patches to the
  extractors). Both sessions edited some of the same files, so all edits were
  merged into one time-ordered replay (manifest in the private recovery bundle).
* Every printed command output from those sessions (tables, logs, summaries),
  kept in the private recovery bundle.
* Code written inside earlier web chats (the tool calls, not their results),
  kept in the private recovery bundle.
* The manuscript, whose source of truth is the Overleaf project.

### What follows from that

* The numbers in the manuscript come from runs whose raw outputs no longer
  exist. They can be traced only through printed logs. Nothing in this
  repository regenerates them, and no number here should be read as reproduced
  unless a command that reproduced it is named.
* Slide-level paired tests cannot be recomputed, because the out-of-fold
  predictions are gone.
* **The code has not been re-run end to end on the 456-slide cohort since the
  loss.**

## How far to trust each script

The recovery bundle, `recovered_v5.zip` (677 entries: 616 files and 61
directories), is kept unchanged in the authors' private repository, with a
SHA-256 for every file, and is not part of this one because it contains session
logs. Where these docs mention the "recovery bundle" or a `recovered/` path,
they mean that private copy.

| Tier | Meaning | Scripts |
|---|---|---|
| A, verified | The rebuilt text matches every later read of the file in either session, line for line | All root scripts except the two below |
| B, reconstruction | Stitched from what the sessions read plus an older complete version | `build_streamline_graphs_bridge.py` |
| C, partial | Only what was read, plus later edits | `extract_nuclear_features.py` |

**Tier B in detail.** The sessions only ever read about 74% of the graph
builder. The rest was filled from the 21 August v3 builder (complete, from a web
chat), with the endpoint patch and the bidirectional-integration patch applied;
every patch anchor matched exactly once, which supports v3 as the true ancestor.
The last ~290 lines (`verify_graph_file`, `self_test`, `main`) were never seen in
a session and come from v3. One call in `self_test` was adapted to the refactored
API. A version stitched only from reads
is kept in the private recovery bundle for comparison. The seam also cut
self-test check [2] (RK4 drift); see `docs/LIMITATIONS.md` item 10.

**Tier C in detail.** The nuclear extractor is the 668 lines read in session 1
plus five edits from session 2. Anything the original had beyond those lines is
missing. The six spacing and immune-infiltration columns and `INFILTRATE_UM = 30.0`
are present.

**Approximate notes.** Three working notes were rebuilt from shell appends and
encoding repairs. The `main.tex` copies are snapshots.

## Never recovered

`convert_to_pyramid_tiff.py`, `build_cellvit_manifest.py`, `verify_tiffs.py`,
`verify_and_fix_manifest_dimensions.py`, `check_centroid_coordinate_scale.py`,
the Windows CellViT runner and its 90-minute timeout wrapper, and the older
`visualize_attention.py`, `diagnose_endpoints.py`, `apply_bidir_fix.py`,
`check_graph_topology.py` and `inspect_cellvit.py` (the last four exist only as
older web-chat versions). Without the first group the repository cannot
regenerate the earliest stages exactly. `smoke/` re-implements conversion,
manifest and the CellViT call from the recorded settings; those files are new,
written after the loss, and have been run only on the five-slide smoke panel
(`docs/SMOKE_REPORT.md`).

Recorded settings for the missing stages: slides were converted with pyvips to
tiled (256 px), pyramidal BigTIFF with JPEG quality 90; CellViT was run as
`python -m cellvit.detect_cells --model SAM --outdir <dir> --geojson --cpu_count 4
--ray_worker 1 --ray_remote_cpus 2 --memory 12000 --batch_size 2 process_wsi
--wsi_path <tiff> --wsi_mpp 0.25` with the checkpoint `CellViT-SAM-H-x40-AMP.pth`
(an earlier project-state note in the private recovery bundle, sections 3-4). The installed CellViT version was never logged.

## Last known software versions

From the original machine's own logs: Python 3.12.10; torch 2.5.1+cu121;
torch_geometric 2.7.0 (used only by the abandoned GATv2 path; no script in
`scripts/` imports it); timm 1.0.29; transformers 5.16.1; numpy 1.26.4; scipy
1.17.1; pandas 3.0.2; scikit-learn 1.8.0; scikit-image 0.26.0; matplotlib 3.10.8;
OpenCV (cv2) 4.7.0; Pillow 10.4.0; openslide 1.4.3; tiffslide 2.5.1; pyvips 3.1.1;
shapely 2.0.5; numba 0.65.0; zarr 2.18.7; tifffile 2025.5.10; einops 0.8.2. No
versions were logged for huggingface_hub, safetensors, ijson or tqdm. LightGBM
and XGBoost were not installed. `requirements.txt` pins these where they were
logged; `requirements-lock-linux.txt` records the Linux environment the tests
were run in, which is a different machine.

## What the portability pass changed

`scripts/` started as a byte-identical, flat copy of the recovered root scripts. After the pass
they were arranged into four folders (`pipeline/`, `analysis/`, `verify/`, `dev/`; see
`docs/SCRIPT_INVENTORY.md`), which added only `sys.path` lines and script paths, not logic. A
provenance diff against them is kept in the authors' private staging
repository; at the end of the pass, 64
of the 82 scripts differ, nearly all by one to five lines. The changes are:

* **Paths.** Every hard-coded Windows dataset path now derives from
  `corpus_paths.ROOT`, keeping the original type (`Path` or `str`) and the same
  sub-path below the root. `corpus_paths.py` defaults to `./data/UBC-OCEAN`,
  made absolute and exported to `UBC_ROOT` for child processes. Argparse
  `--root` defaults follow it.
* **Script folder.** `sys.path` inserts and `SCRIPTS = ROOT / "scripts"` now use
  the script's own directory, because code and data no longer share a root.
* **Interpreter.** The hard-coded Windows venv interpreter is `sys.executable`.
* **Windows-only process code.** `queue_bridge_refit.py` and
  `queue_explain_final.py` waited on a PID with `tasklist`; `run_phase3.py`
  asked PowerShell whether training was running; `bt_setup.py` wrote a
  PowerShell junction script. Each keeps the Windows code verbatim under
  `if os.name == "nt":` and adds a POSIX equivalent (`ps`, `ln -s`). This is
  the only new control flow, and it is in orchestration, not in any analysis.
* **Backslash path joins** in `dump_names.py` and `inspect_graph.py` became
  `os.path`/`pathlib` joins.
* **Text.** Three docstrings that gave the tile as 256 um (the builder's nominal
  figure; the physical tile is 512 um) now say 512 um; a
  comment in `bt_setup.py` that gave a Windows-only instruction now covers both;
  the `corpus_paths.py` docstring documents `UBC_ROOT` and `UBC_RUN_DIR` with
  bash and PowerShell examples.

The full diff was read line by line after the edits. No hunk changes a
numeric constant, threshold, random seed, feature list, fold logic, CLI default
other than a path, or the order of operations of any analysis. The six DEV-ONLY
scripts (`tools/dev_only.txt`) are unchanged.

## One deliberate change after the portability pass

`fm_subspace_split.named_features()` was changed after the loss, by the author's
decision, so that a fresh dataset root yields the same 49 columns as the cached
path the original runs used. The morph branch now asks `bags_for()` for the
nuclear, chromatin and eosin blocks, and a new `check_bag_cache()` refuses a
bag cache under the clean model's tag that was built from any other blocks.
This is not recovered code. It changes what the function returns only where it
was wrong: on a root with no cache, or with a cache built from other blocks. It
was validated on synthetic data only; the original 456-slide result cannot be
recomputed. Mechanism, evidence for the 49 names, callers and tests are in
`docs/NAMED_FEATURES_CACHE.md`. The private provenance diff shows it as part of
the `fm_subspace_split.py` diff.

## A text-only change after the portability pass

The graph builder's tile is 512 um physical (1024 px at the slides' native
0.5 um/px), but its constants are written at the nominal 0.25 um/px of the
CellViT run, so `NODE_SIZE_UM = 256.0` and `STEP_UM = 20.0` read as half their
physical size. No constant was changed, because changing one alters the lattice.
Only text moved, in `build_streamline_graphs_bridge.py` (the scale paragraph of
the module docstring, the comments on `WSI_MPP`, `NODE_SIZE_UM` and `STEP_UM`,
and the startup log line, which now labels its figures nominal and prints the
pixel and physical tile sizes computed from the constants), in one comment in
`bt_setup.py`, and in the footnote at the top of `train_abmil.py`. See
`docs/LIMITATIONS.md` item 1. The private provenance diff shows these as part
of each script's diff.

## Checks run on the recovered code

* `python scripts/pipeline/build_streamline_graphs_bridge.py --self-test` exits 0 and
  prints "all checks passed" for checks [1] and [3] to [9]: banded-field
  eigenvector orientation, sign-aware interpolation, non-square node geometry,
  an end-to-end build (75 nodes, 112 edges, 5 channels, `edge_attr` 112 x 23,
  node features 75 x 39), and the swapped-manifest, stale-schema and
  out-of-bounds guards. Check [2] is missing (see above) and is re-created in
  `tests/test_builder_rk4_drift.py`, which passes.
* The offline test suite (`pytest`), including the Additive MIL contribution
  identity, the subspace-deletion algebra, stain deconvolution, and a synthetic
  25-slide corpus built by the real graph builder and loaded by the real
  training-side loaders. See the CHANGELOG and the test files for what each
  test asserts.
* Nothing has been run on real slides. See `docs/SMOKE_REPORT.md`.

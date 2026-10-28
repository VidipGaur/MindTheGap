# Limitations

Known limitations of this repository, left as found unless an entry says it was
fixed. Code is referred to by script and by constant, function or flag name, not
by line number, so an edit to a script cannot make a reference here wrong.

## Numeric constants

1. **Tile-size constant.** The tile is 512 um (1024 px at the native 0.5 um/px).
   `NODE_SIZE_UM = 256.0` in `build_streamline_graphs_bridge.py` is that tile
   expressed at the builder's nominal tiling pixel size, `WSI_MPP = 0.25`
   (256 / 0.25 = 1024 px), and takes part in the pixel arithmetic of
   `um_to_px()` and of the node-grid construction. Changing the value alone would
   change the lattice: 512 with the 0.25 nominal still in place would build a
   2048 px lattice. Every micron figure in that script (`NODE_SIZE_UM`,
   `STEP_UM` and the others) is nominal, written at 0.25 um/px where the real
   pixel is 0.5 um, so the physical size is twice the labelled one: the step is
   40 um, not 20, and a grid pixel is 8 um, not 4. The module docstring, the
   comments on `WSI_MPP`, `NODE_SIZE_UM` and `STEP_UM`, and the startup log line
   say so; the log line prints "node tile = 256 nominal um (1024 px = 512 um
   physical)". `bt_setup.main()` uses a literal `256.0` the same way and carries
   a comment saying so. Possible change: rename the constant (for example
   `NODE_SIZE_UM_AT_WSI_MPP`) with no change in value. No number should change.

## Behaviour that is fine for the paper's corpus but fragile

2. **`split("_")[0]` used to recover an id.** On UBC-OCEAN's numeric ids it agrees
   with `corpus_paths.image_id_from_graph()`; on ids such as `10_OOU198_EC_SC` it
   truncates. Left unchanged:
   * on graph or feature file names (an id): `check_dims.py`, `check_nuc.py`
     (in `chk()`), `check_tma.py`, `dump_names.py`, `probe_wsi.py`;
   * on run tags, not ids: `analyse_abmil_nuclear.bags_for()`,
     `bridge_fm_interpretable.main()` (two places), `queue_bridge_refit.one_run()`
     (two places);
   * prose only: the comment in `embed_tiles_fm.main()` and the docstring of
     `corpus_paths.image_id_from_graph()`.
   `tools/audit_paths.py` reports each occurrence as `id_split`, with its
   location, and does not fail on them.
3. **`UBC_RUN_DIR` does not reach the training side.** Only the extraction stages
   and `eval_external.py` go through `corpus_paths`. `train_abmil.py`
   (`RUN_DIR_NAME = "cellvit_run"`), `fm_subspace_split.py` (`GDIR`) and
   `bridge_fm_interpretable.py` build `<root>/cellvit_run/...` themselves. Point
   `UBC_ROOT`, not `UBC_RUN_DIR`, at a different corpus when the training side is
   involved. Preserved as found.
4. **`fm_subspace_split.named_features()` depended on a cache built elsewhere.
   Fixed after the data loss; see `docs/NAMED_FEATURES_CACHE.md`.**
   `analyse_abmil_nuclear.bags_for()` caches by tag only
   (`bag_cache_v5nucchromeos.pt`), and a cache that exists is loaded whatever
   blocks the caller asked for. `named_features("morph")` asked for the nuclear
   block alone. On the original machine the cache had been built with all three
   blocks by `train_abmil.py`, so the call returned the 49 clean columns. On a
   fresh root it returned 38 and saved a nuclear-only cache, after which
   `named_features("stain")` exited with "stain probe columns absent". The morph
   call now asks for all three blocks and refuses a cache under that tag built
   from other blocks. Validated on synthetic data only. Other callers with the
   same dependence are items 19 to 21.
5. **The bag cache does not notice new feature files.** `train_abmil.py` reuses a
   bag cache that predates a partial extraction unless `--rebuild-cache` is
   passed (see the note on `SCHEMA_VERSION` and the `--rebuild-cache` argument
   in `train_abmil.py`). Anyone re-running after an extraction must pass it.
6. **`flag_leak_report()` refuses fewer than five slides per class.**
   `train_abmil.flag_leak_report()` cross-validates with `StratifiedKFold(5)` on every load,
   so `load_bags()` raises `ValueError: n_splits=5 cannot be greater than the
   number of members in each class` on a corpus such as the five-slide smoke
   panel. Harmless at 456 slides. `smoke/integration_check.py` has an opt-in
   `--skip-flag-leak-report` that replaces it with a no-op for that process only.
7. **Stale process ids as defaults.** `queue_explain_final.py` defaults
   `--wait-pid` to 28360 and `queue_bridge_refit.py` to 8180, PIDs from the
   original Windows session. On Linux they now poll `ps -p`; if an unrelated
   process happens to hold that PID the script waits for it. Pass `--wait-pid 0`.
   Defaults were not changed.
8. **POSIX process checks are equivalents, not copies.** `run_phase3.py`
   `gpu_busy()` lists command lines with `ps -axo args=` and keeps those
   containing "python"; the Windows branch filtered processes named
   `python.exe`. The queue scripts use `ps -p PID`. The Windows code is kept
   verbatim under `os.name == "nt"`.
9. **`corpus_paths` now writes `UBC_ROOT` into the environment at import**, so
   child processes started with `cwd=scripts/` resolve the same root. Any
   script that imports `corpus_paths` exports it to everything it launches.

## The reconstruction

10. **Self-test check [2] of the graph builder is missing.** The stitch seam in
    `build_streamline_graphs_bridge.py` falls inside `self_test()`: the reads end after `d_row` is computed and the v3 tail
    resumes at check [3], so the `d_col` computation, the `[2] RK4 drift` print
    and its assertion were lost. `--self-test` prints "all checks passed" over 8
    checks, not 9. `tests/test_builder_rk4_drift.py` re-creates check [2] from the
    earlier web-chat version of the builder (kept in the private recovery bundle) and
    it passes, reproducing the logged forward drift
    |d_col| = [795, 495]. Possible fix: restore the four lines in the builder from
    that source. The v5 wording of check [2] is unknown.
11. **What else the gaps could hide.** About 26% of the builder (the last ~290
    lines, plus scattered gaps) comes from the v3 web-chat version, not from a
    read of the final file. The nuclear extractor (Tier C) holds only what was
    read plus five later edits. Neither file has been run on the full cohort
    since the data loss; the five-slide smoke test (`docs/SMOKE_REPORT.md`) is
    the only run.
12. **`train.csv` dimensions.** An earlier project-state note (private recovery bundle), section 2.3, says `train.csv` width and height are swapped for some WSIs,
    naming 10077, and that a lost script rewrote the manifest from the TIFFs. A
    later check found the opposite: `check_dims.py` printed
    "wsi_shape vs train.csv over 60 graphs: match=60 transposed=0 other=0"
    (session log, private recovery bundle), and the 10077 graph
    has `wsi_shape = [39150, 53143]`. The smoke checker follows the later
    finding; `smoke/build_manifest.py` stops rather than correcting if a TIFF
    disagrees with `train.csv`.

## Missing pieces

13. **Early-stage scripts never recovered:** `convert_to_pyramid_tiff.py`,
    `build_cellvit_manifest.py`, `verify_tiffs.py`,
    `verify_and_fix_manifest_dimensions.py`, `check_centroid_coordinate_scale.py`,
    `cellvit_run.bat`, `run_cellvit_with_timeout.ps1`, and the older
    `visualize_attention.py`, `diagnose_endpoints.py`, `apply_bidir_fix.py`,
    `check_graph_topology.py`, `inspect_cellvit.py` (the last four only as older
    web-chat versions). `smoke/convert_to_pyramid_tiff.py`,
    `smoke/build_manifest.py` and `smoke/run_cellvit.sh` re-implement the first
    stages from recorded settings and are labelled as new; they have been run
    only on the five-slide smoke panel (`docs/SMOKE_REPORT.md`).
14. **CellViT version.** The installed version was never logged. PyPI's `cellvit`
    is 1.0.9 as of 2026-10-16. Whether the August 2026 run used a release with
    the same cell taxonomy and output layout is unknown; `run_cellvit.sh` records
    the version it finds.
15. **Unrecorded dependency versions.** torchvision (0.20.1 chosen as the release
    built for torch 2.5.1), the OpenCV wheel build and headless/GUI variant
    (4.7.0.72 headless chosen), huggingface_hub, safetensors and tqdm. `ijson` was
    not installed originally and is left out; `extract_nuclear_features.py` falls
    back to its own parser without it.
16. **`probe_bbox.py` is not shipped** because it writes into a temporary folder
    from the original session, but it documents the bounding-box convention check
    behind the chromatin extractor. It would be worth porting if that check should
    stay runnable.

## Possible additions

17. **`--save-oof` for `train_abmil.py`.** Out-of-fold predictions are kept in
    memory for repeat 0 only (in `train_abmil.main()`) and were never written to
    disk, which is why slide-level paired tests cannot be recomputed now. An
    opt-in `--save-oof` flag, default off, could write out-of-fold probabilities
    per (repeat, fold); it would add output, not change training.

## Licences not verified

18. Model weights and data: the Hugging Face API reports `license: other` for
    Phikon and Phikon-v2, `cc-by-nc-nd-4.0` for UNI2-h, and `apache-2.0` for
    H-optimus-0 and Hibou-L, and the last three are gated with extra access
    terms. These are metadata tags; the model cards and access terms were not
    read. The CellViT checkpoint's licence and the UBC-OCEAN competition rules
    were not checked. See `docs/THIRD_PARTY.md`.

## Cache handling (found while fixing `named_features()`, left unchanged)

Details and the full caller table are in `docs/NAMED_FEATURES_CACHE.md`.

19. **Two scripts leave a wrong cache behind on a fresh root.**
    `bridge_fm_interpretable.run_arm()` and `stability_vs_ensemble.run()` load the
    clean model's tag with the nuclear block only, and the Phikon-v2 tag with no
    feature block at all (`bags_for()` cannot load embeddings). Both are refused
    by the checkpoint column-count check, but only after `load_bags()` has saved
    a nuclear-only cache under `v5nucchromeos` and a graph-only cache under
    `v5fmphikonv2`. `train_abmil.py` reuses both without an error; on synthetic
    data, `--fm-features phikonv2` over such a cache trains on the 39 graph
    columns. `named_features("morph")` now refuses the first. Options: pass
    `chromatin=True, eosin=True` in the two interpretable arms; refuse a missing
    cache in the encoder arms.
20. **`load_bags()` never compares a cache's blocks with the request.** A guard
    there (the cache records its blocks) would cover every caller at once, but
    it changes `train_abmil.py` and every caller's behaviour on a mismatched
    cache.
21. **`analyse_abmil_nuclear.py` modes ignore `--drop-groups` and
    `--exclude-prefix`.** In the recovered text only `attnensemble` passes
    `--drop-groups` to `bags_for()`, and no mode passes `--exclude-prefix`. The
    original `run_cleanfinal.py` stability step on `v5nucchromeos_clean_final`
    nevertheless printed `bags=456 features=49` and exited 0
    (session log, private recovery bundle), which the recovered
    text could not do. The version that ran probably differed. Not resolved.

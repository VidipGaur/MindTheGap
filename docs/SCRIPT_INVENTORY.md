# Script inventory

All 82 root-level scripts from the recovered bundle. The 76 that ship are arranged in
four folders under `scripts/` (the **Folder** column): `pipeline/`, `analysis/` and
`verify/` hold the scripts the paper's pipeline, results and tests use, by class (one
exception: `bentaeb_probe.py` is class VERIFY but sits in `pipeline/`, because `bt_setup.py`
reads the table it writes), and
`dev/` holds development-time probes, sanity checks and superseded comparisons that
nothing in the pipeline needs. The six DEV-ONLY scripts do not ship ("not shipped");
`scripts/dev/` is a different thing and does ship.

**Class.** PIPELINE produces data or results the paper depends on, or is the
orchestrator that runs it. ANALYSIS reads results or checkpoints to answer one
question; several are historical (they belong to analyses the paper no longer
reports). VERIFY is a one-off check of data, alignment or environment. DEV-ONLY
is session or paper-bundle tooling: kept unchanged from the recovered originals, excluded from
the path audit, the tests and the published repository (`tools/dev_only.txt`).

**Tier** is the recovery trust tier (`docs/PROVENANCE.md`): A verified against
every later read; B reconstruction; C partial.

**Imports without data** is what `tests/test_imports.py` checks with no dataset,
no GPU and `UBC_ROOT` unset. Scripts without an `if __name__ == "__main__":`
guard run their whole body on import, so for them "import" means "run"; they
are compiled but not imported, and listed by what their body needs. Wrapping
them in a `main()` would be a restructure, which this pass does not do.

**Built-in checks** are the script's own `--self-test`, `--smoke`, `--check` or
`--dry-run` modes. Two are used with arguments: `eval_external.py --self-test ARM`
and `embed_tiles_fm.py --check --limit 2`.

Totals by class: PIPELINE 22, ANALYSIS 27, VERIFY 27, DEV-ONLY 6. Shipped, by folder: pipeline 22, analysis 18, verify 3, dev 33.

| Script | Folder | Class | Tier | Lines | Imports without data | Built-in checks | Proposed status | Purpose (first docstring line) |
|---|---|---|---|---|---|---|---|---|
| `analyse_abmil_nuclear.py` | analysis | ANALYSIS | A | 641 | yes (tested) | - | public | Post-hoc analysis of the nuclear-feature ABMIL runs (Task 3 follow-ups) |
| `batch_residual.py` | analysis | ANALYSIS | A | 93 | no: runs on import, needs data | - | public | How much of the signal survives removing everything staining can explain? |
| `bentaeb_manifest.py` | dev | PIPELINE | A | 119 | yes (tested) | - | public | Match Ben Taeb SVS files to labels.csv and report per-class counts |
| `bentaeb_probe.py` | pipeline | VERIFY | A | 96 | yes (tested) | - | public | Probe the Ben Taeb SVS slides: dimensions, native MPP, magnification, |
| `bridge_fm_interpretable.py` | pipeline | PIPELINE | A | 465 | yes (tested) | - | public | How much of what Phikon does per tile is expressible in named pathology terms? |
| `bt_check_edits.py` | dev | VERIFY | A | 68 | no: runs on import, needs data | - | public | Verify the corpus_paths edits: UBC-OCEAN defaults unchanged, Ben Taeb routed |
| `bt_graph_audit.py` | dev | VERIFY | A | 80 | no: runs on import, needs data | - | public | Audit the Ben Taeb tile lattice against the UBC-OCEAN one |
| `bt_reference.py` | dev | ANALYSIS | A | 48 | no: runs on import, needs data | - | public | The UBC-OCEAN reference column, under the SAME statistic the external run |
| `bt_setup.py` | pipeline | PIPELINE | A | 165 | yes (tested) | - | public | Stage 0 -- build the Ben Taeb parallel corpus scaffold |
| `bt_verify_coords.py` | dev | VERIFY | A | 127 | yes (tested) | - | public | Verify the Ben Taeb CellViT coordinate convention before any KDE is built |
| `build_kde_density_maps_new.py` | pipeline | PIPELINE | A | 548 | yes (tested) | - | public | Build multi-channel KDE nuclear density maps from CellViT outputs (UBC-OCEAN) |
| `build_streamline_graphs_bridge.py` | pipeline | PIPELINE | B | 1428 | yes (tested) | --self-test | public | Build streamline-connectivity graphs from KDE nuclear density maps (UBC-OCEAN) |
| `check_bundle.py` | not shipped | DEV-ONLY | A | 154 | not tested | - | blocked: Overleaf bundle checker (paper tooling) | Will this bundle compile on Overleaf? Checked without a TeX installation |
| `check_dims.py` | dev | VERIFY | A | 43 | no: runs on import, needs data | - | public | Confirm graph wsi_shape == train.csv dims (no PNG->TIFF transpose), and |
| `check_drivers.py` | dev | VERIFY | A | 41 | no: runs on import, needs data | - | public | Are nnd_all_um_cv and dead_count real signal or extraction artefacts? |
| `check_eosin_classes.py` | dev | VERIFY | A | 58 | no: runs on import, needs data | - | public | Eosin check on slides chosen ACROSS classes |
| `check_fm_env.py` | dev | VERIFY | A | 76 | no: runs on import, needs network | - | public | Can this machine run a pathology foundation-model positive control? |
| `check_gating.py` | dev | VERIFY | A | 34 | no: runs on import, needs network | - | public | Which pathology encoders can we actually download without a human in the loop? |
| `check_groups.py` | dev | VERIFY | A | 34 | no: runs on import, needs data | - | public | Prints how resolve_groups() splits the clean final feature set |
| `check_nuc.py` | dev | VERIFY | A | 220 | no: runs on import, needs data | - | public | Sanity check extract_nuclear_features.py output against the source graphs |
| `check_nuc2.py` | dev | VERIFY | A | 99 | no: runs on import, needs data | - | public | Aggregate pass + the two issues the per-slide pass surfaced |
| `check_tma.py` | dev | VERIFY | A | 30 | no: runs on import, needs data | - | public | Counts TMA rows in train.csv/test.csv against the graphs on disk |
| `cmp_gbm_vs_mil.py` | dev | ANALYSIS | A | 47 | no: runs on import, needs data | - | public | Paired: does ABMIL beat the best slide-level model on the SAME folds? |
| `cmp_stagec.py` | dev | ANALYSIS | A | 66 | no: runs on import, needs data | - | public | Stage C paired: does any alternative MIL formulation beat ABMIL? |
| `cmp_staged.py` | dev | ANALYSIS | A | 45 | no: runs on import, needs data | - | public | Stage D paired: do the chromatin/texture features add anything? |
| `compare_abmil_configs.py` | analysis | ANALYSIS | A | 118 | yes (tested) | - | public | Paired config comparison for the ABMIL sweeps |
| `corpus_paths.py` | pipeline | PIPELINE | A | 153 | yes (tested) | - | public | Single source of dataset paths and per-slide pixel sizes |
| `corpus_stats.py` | analysis | ANALYSIS | A | 41 | no: runs on import, needs data | - | public | Cohort counts: train.csv rows, rows with a graph, labels |
| `dump_names.py` | dev | VERIFY | A | 20 | no: runs on import, needs data | - | public | Prints graph and nuclear feature names for one slide |
| `embed_tiles_fm.py` | pipeline | PIPELINE | A | 464 | yes (tested) | --check | public | Pathology foundation-model tile embeddings -- the Task 4 positive control |
| `eval_external.py` | pipeline | PIPELINE | A | 567 | yes (tested) | --self-test | public | Instrument 1 (representation substitution) on an EXTERNAL corpus, frozen |
| `excluded.py` | analysis | ANALYSIS | A | 30 | no: runs on import, needs data | - | public | Compares included and excluded WSIs (labels, sizes) |
| `explain_slide.py` | analysis | ANALYSIS | A | 415 | yes (tested) | - | public | Turn one slide's prediction into a sentence in diagnostic vocabulary |
| `extract_chromatin_features.py` | pipeline | PIPELINE | A | 844 | yes (tested) | --smoke, --check | public | Chromatin and texture features from WSI pixels, aligned to the existing graphs |
| `extract_eosin_features.py` | pipeline | PIPELINE | A | 587 | yes (tested) | --check | public | Per-tile CYTOPLASM features from the eosin channel |
| `extract_nuclear_features.py` | pipeline | PIPELINE | C | 677 | yes (tested) | --smoke, --check | public | P4 -- nuclear morphometrics as per-tile features, aligned to the existing graphs |
| `final_ledger.py` | analysis | ANALYSIS | A | 83 | no: runs on import, needs data | - | public | Every number the summary doc needs, computed once, paired where it matters |
| `fix_bom.py` | not shipped | DEV-ONLY | A | 21 | not tested | - | blocked: Windows editor clean-up | Strips UTF-8 BOMs from named scripts (Windows editor artefact) |
| `fm_subspace_split.py` | pipeline | PIPELINE | A | 377 | yes (tested) | --check | public | Split Phikon-v2's embedding into the part our pathology vocabulary explains |
| `gap.py` | dev | ANALYSIS | A | 23 | no: runs on import, needs data | - | public | Train/validation gap under bag subsampling |
| `grep_transcript.py` | not shipped | DEV-ONLY | A | 30 | not tested | - | blocked: parses session transcripts | Pull the dead_count / stability / attention-ensemble outputs back out of the |
| `hf_access_check.py` | dev | VERIFY | A | 21 | no: runs on import, needs network | - | public | Reports which gated encoder repos the HF_TOKEN account can download |
| `hibou_load.py` | dev | VERIFY | A | 55 | no: runs on import, needs network | - | public | Load hibou-L with STOCK transformers classes, no remote code |
| `hibou_probe.py` | dev | VERIFY | A | 27 | no: runs on import, needs network | - | public | Can hibou-L load WITHOUT its custom remote code? |
| `inspect_ckpt.py` | dev | VERIFY | A | 36 | no: runs on import, needs data | - | public | Lists checkpoint contents for the final arms |
| `inspect_graph.py` | dev | VERIFY | A | 31 | no: runs on import, needs data | - | public | Prints the keys and shapes of one graph file |
| `list_stain_cols.py` | dev | VERIFY | A | 23 | no: runs on import, needs data | - | public | Lists the stain-related column groups |
| `make_figures.py` | pipeline | PIPELINE | A | 505 | yes (tested) | - | public | Every figure in the paper, rendered from paper/numbers.json and the corpus |
| `make_zip.py` | not shipped | DEV-ONLY | A | 45 | not tested | - | blocked: builds the Overleaf zip | Build the Overleaf zip with POSIX separators |
| `paired_clean.py` | dev | ANALYSIS | A | 44 | no: runs on import, needs data | - | public | Is the 0.6875 -> 0.6603 drop from cleaning the features real or noise? |
| `paper_numbers.py` | pipeline | PIPELINE | A | 265 | yes (tested) | - | public | Every number the paper quotes, computed from the results CSVs |
| `per_class_numbers.py` | pipeline | PIPELINE | A | 257 | yes (tested) | - | public | Per-class recall and full confusion counts, on BOTH corpora, from frozen |
| `phase3_final.py` | analysis | ANALYSIS | A | 92 | no: runs on import, needs data | - | public | Pull Stage F/G/H numbers and repair the one paired comparison run_phase3 missed |
| `phase3_pairs.py` | analysis | ANALYSIS | A | 87 | no: runs on import, needs data | - | public | The paired comparisons that decide what goes in the paper and what runs next |
| `probe_bbox.py` | not shipped | DEV-ONLY | A | 117 | not tested | - | blocked: writes into a temporary folder from the original session; documents the bbox-convention check | Are cell_detection.json bboxes where I think they are, and is the H channel |
| `probe_stain_nonlinear.py` | analysis | ANALYSIS | A | 144 | yes (tested) | - | public | Is the encoder's stain content LINEARLY accessible, or is there more of it |
| `probe_wsi.py` | verify | VERIFY | A | 70 | no: runs on import, needs data | - | public | Do the pyramid TIFFs share a coordinate space with the graphs and cells.json? |
| `profile_chrom.py` | dev | VERIFY | A | 82 | no: runs on import, needs data | - | public | Where does chromatin extraction actually spend its 250 ms per tile? |
| `profile_eosin.py` | dev | VERIFY | A | 51 | no: runs on import, needs data | - | public | Where does extract_eosin_features spend its 232 ms per tile? |
| `queue_bridge_refit.py` | pipeline | PIPELINE | A | 211 | yes (tested) | --smoke | public | Per-repeat refit of the bridge R^2, queued behind the running chain |
| `queue_explain_final.py` | dev | ANALYSIS | A | 134 | yes (tested) | - | public | Regenerate the per-slide explanations on the FINAL model, queued behind |
| `resolve_dead_count.py` | analysis | ANALYSIS | A | 355 | yes (tested) | - | public | Is `dead_count` biology, or is it a staining/fixation artefact? |
| `run_bentaeb.py` | pipeline | PIPELINE | A | 180 | yes (tested) | - | public | Orchestrator: Instrument 1 on the Ben Taeb et al. external corpus |
| `run_cleanfinal.py` | pipeline | PIPELINE | A | 98 | yes (tested) | - | public | The genuinely clean interpretable headline -- the one number the paper needs |
| `run_multiencoder.py` | pipeline | PIPELINE | A | 499 | yes (tested) | --dry-run | public | Replicate the Phikon-v2 audit programme across three more encoders |
| `run_nodead.py` | analysis | ANALYSIS | A | 114 | yes (tested) | - | public | Re-baseline the model with the convicted `dead_count` features removed |
| `run_overnight.py` | analysis | ANALYSIS | A | 396 | yes (tested) | --dry-run | public | Overnight driver: model selection -> more seeds -> alternative MIL algorithms |
| `run_phase3.py` | analysis | ANALYSIS | A | 408 | yes (tested) | - | public | Phase 3 driver: eosin, the positive control, mixup, and the final ensemble |
| `run_phase4.py` | analysis | ANALYSIS | A | 226 | yes (tested) | - | public | Phase 4 -- clean the books, then move interpretability onto the model that works |
| `run_stainaudit.py` | pipeline | PIPELINE | A | 164 | yes (tested) | - | public | How much of each encoder's advantage is the BATCH SHORTCUT we removed from |
| `run_subspace.py` | pipeline | PIPELINE | A | 138 | yes (tested) | - | public | Does Phikon-v2's accuracy live in the part our vocabulary can describe? |
| `stability_table.py` | analysis | ANALYSIS | A | 66 | no: runs on import, needs data | - | public | Seed stability across the three configurations, at MATCHED seed counts |
| `stability_vs_ensemble.py` | analysis | ANALYSIS | A | 113 | yes (tested) | - | public | What does the 50.4% unanimity number actually cost the ENSEMBLE? |
| `summarise_explanations.py` | dev | ANALYSIS | A | 99 | yes (tested) | - | public | Aggregate the per-slide explanations into a per-subtype picture |
| `test_chrom_load.py` | not shipped | DEV-ONLY | A | 84 | not tested | - | blocked: writes into a temporary folder from the original session | Does train_abmil.py load and group the chromatin block correctly? |
| `test_phikon.py` | dev | VERIFY | A | 52 | no: runs on import, needs network | - | public | Smoke-test the encoders before committing an overnight embedding run |
| `timeline_stats.py` | dev | ANALYSIS | A | 99 | no: runs on import, needs data | - | public | Pull every completed result cell into one place for the timeline doc |
| `train_abmil.py` | pipeline | PIPELINE | A | 1767 | yes (tested) | --smoke | public | ABMIL sweep: what are the tile features worth, with the graph removed entirely? |
| `train_slidelevel_gbm.py` | analysis | ANALYSIS | A | 213 | yes (tested) | - | public | Slide-level control: how much of the 0.67 needs MIL at all? |
| `verify_arch.py` | verify | VERIFY | A | 87 | runs on import, needs nothing (run as a test) | - | public | Sanity checks on the new architectures and selection modes |
| `verify_phase3.py` | verify | VERIFY | A | 108 | runs on import, needs nothing (run as a test) | - | public | Unit-verify the new train_abmil.py machinery before an overnight run |
| `verify_timm_path.py` | dev | VERIFY | A | 58 | no: runs on import, needs network | - | public | Exercise embed_tiles_fm.py's NEW code paths using an UNGATED model |

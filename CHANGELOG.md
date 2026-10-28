# Changelog

All notable changes to this repository are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-10-28

First public release.

### Added

* `LICENSE` (Apache-2.0) and `NOTICE`; `license`, `license-files` and
  `[project.urls]` in `pyproject.toml`; `license` and the repository URL in
  `CITATION.cff`, whose author list is Vidip Gaur and Vidu Gaur.
* `docs/LIMITATIONS.md`, which lists the known limitations (21 entries).
* `tests/test_layout.py`, which checks that every sibling import resolves.

### Changed

* `scripts/` is split into `pipeline/` (22), `analysis/` (18), `verify/` (3) and `dev/` (33).
  No script's logic changed. The 70 scripts that import a sibling add the four
  folders to `sys.path`; the runners that launch a sibling by filename build its
  path from the new layout; tests, `tools/audit_paths.py`, `smoke/`, the pytest
  `pythonpath` and `.gitignore` follow. `pyproject.toml` no longer installs the
  scripts as top-level modules (`py-modules = []`), since one module list cannot
  span four folders.
* `bentaeb_probe.py` is in `pipeline/`: `bt_setup.py` reads the slide table it
  writes. `docs/REPRODUCING.md` stage 18 now lists that step.
* Text only: micron figures in `build_streamline_graphs_bridge.py` are labelled
  nominal (the tile is 512 um physical, 1024 px; `NODE_SIZE_UM = 256.0` is
  unchanged and must stay so), in its docstring, comments and startup log line,
  in a comment in `bt_setup.py` and in a footnote of `train_abmil.py`. Usage
  examples in docstrings give the full script path.
* `CODE_OF_CONDUCT.md` names the reporting contact. Lab-machine identifiers and
  process wording are removed from the docs, and statements that the smoke test
  had not been run are corrected.
* `tools/scan_secrets.py --strict` allows the two published Code of Conduct contact
  addresses (`PUBLISHED_CONTACTS`); any other address still fails.
* Docs refer to code by script and name, not by line number; the cost sources in
  `docs/REPRODUCING.md` no longer cite a file outside the repository.

## [0.2.0] - 2026-10-22

Staging only; not released.

### Fixed

* `fm_subspace_split.named_features("morph")` returned 38 columns instead of
  the clean model's 49 on a dataset root without a bag cache, and saved a
  nuclear-only cache under the clean model's tag. It now loads the nuclear,
  chromatin and eosin blocks, and refuses a cache under that tag built from
  other blocks. Results with a correct cache are unchanged. Written after the
  data loss, not recovered code; validated on synthetic data only
  (`docs/NAMED_FEATURES_CACHE.md`).

### Changed

* `smoke/integration_check.py` asks for the morph block before the stain block.

### Added

* `tests/synthetic_corpus.py` and `tests/test_named_features_cache.py`, which
  pin the 49 names to the list in the recovered logs.

## [0.1.0] - 2026-10-16

First portable version of the recovered code. Staging only; not released.

### Added

* The recovery bundle, unchanged, with SHA-256 for every file (kept in the authors'
  private staging repository; not part of this repository).
* `scripts/`: working copy of the 82 recovered root scripts.
* Pinned `requirements*.txt`, `pyproject.toml`, and a lock file of the Linux
  test environment.
* Offline test suite: imports and `--help` without data, `corpus_paths`, the
  Additive MIL contribution identity, the subspace-deletion algebra, stain
  deconvolution, feature-name contracts, the recovered tree's hashes, a
  re-creation of the graph builder's lost self-test check [2], and a synthetic
  25-slide corpus through the real graph builder and loaders.
* Five-slide smoke harness (`smoke/`) with an output checker; the conversion,
  manifest and CellViT steps are re-implementations written after the loss.
  Not yet run on real slides.
* Tools: path audit, secret scan. (The provenance diff and export tooling live in
  the authors' private staging repository.)
* CI workflow (untested on GitHub) and pre-commit hooks.
* Documentation: provenance, reproduction guide, script inventory, open
  questions, third-party licences, smoke report, availability template.

### Changed

* Hard-coded Windows paths in 64 scripts now derive from `corpus_paths.ROOT`
  (default `./data/UBC-OCEAN`, or `UBC_ROOT`); the interpreter path is
  `sys.executable`.
* Windows-only process checks and the junction script gained POSIX
  equivalents; the Windows code is unchanged.
* Three docstrings now give the tile size as 512 um.

No numeric constant, threshold, seed, feature definition or non-path default
was changed.

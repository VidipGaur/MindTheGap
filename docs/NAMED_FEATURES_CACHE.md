# `named_features()` and the bag cache

This change was written after the data loss, in October 2026. It is not part of
the recovered code. It was validated on synthetic data only; the 456-slide
result it protects cannot be recomputed, because the feature blocks, the bag
cache and the checkpoints were lost.

It is the one deliberate change to science-adjacent code in this repository.
Everything else in `scripts/` differs from the recovered originals only by plumbing (see
`docs/PROVENANCE.md`).

## The problem

`fm_subspace_split.named_features("morph")` defines the 49 named columns that
the subspace-deletion instrument (`fm_subspace_split.py`, run by
`run_subspace.py` and `run_multiencoder.py`) fits its ridge map from. It called

```python
bags_for(ROOT, "v5nucchromeos_clean_final", True,
         drop_groups=["graph", "chrom_texture", "eos_other", "chrom_context"],
         exclude_prefix=["dead_"])
```

with the nuclear block only. `analyse_abmil_nuclear.bags_for()` names its cache
from the tag alone (`tag.split("_")[0]` without `noflag`, here
`bag_cache_v5nucchromeos.pt`) and calls `train_abmil.load_bags()` with
`rebuild=False`. `load_bags()` loads that file whenever it exists and its
`schema` equals `SCHEMA_VERSION`; only otherwise does it build bags, from the
blocks it was given, and save them under that name. The blocks do not enter the
name, and the record of blocks the cache keeps (its `nuclear`, `chromatin`,
`eosin` and `fm` entries) is never compared with the request. The `dead_`
exclusion, the group drops and `drop_degenerate()` all run after the cache, on
whatever columns it holds, and a group drop removes only the groups it names.

On the original machine the cache had been built with all three blocks by
`run_cleanfinal.py` (`train_abmil.py --tag v5nuc --nuclear-features
--chromatin-features --eosin-features`, which composes the tag `v5nucchromeos`).
From it the call took 39 graph + 41 nuclear + 39 chromatin + 35 eosin = 154
columns, excluded 2 `dead_` columns, removed 102 by group (graph 39, chromatin
texture 26, chromatin context 7, other eosin 30) and the constant
`nuc_have_cv2`, leaving 49. On a fresh root the same call built a graph +
nuclear cache: 80 columns, minus the 2 `dead_` columns, the 39 graph columns
and `nuc_have_cv2`, leaving 38. The chromatin and eosin groups are empty there,
so nothing fails. The 11 missing columns are the six chromatin level columns
and the five eosin clearing columns. The call also saved that nuclear-only
cache under the clean model's tag, after which `named_features("stain")` exits
with "stain probe columns absent", and `train_abmil.py` with all three feature
flags reuses it without an error (checked on synthetic data: 79 columns, no
chromatin or eosin block).

## Callers

Callers are named by script and function, not by line number. "Convention"
means the blocks `train_abmil.py` puts in the cache of that name.

| Caller | Tag, cache | Blocks asked for | Convention | Depends on an existing cache |
|---|---|---|---|---|
| `fm_subspace_split.named_features("morph")` | `v5nucchromeos_clean_final`, `v5nucchromeos` | nuclear | nuclear, chromatin, eosin | Yes, silently: 38 columns on a fresh root. Fixed here |
| `fm_subspace_split.named_features("stain")` | same | nuclear, chromatin, eosin | same | Needs the three blocks; a cache without them exits with "stain probe columns absent". Unchanged |
| `fm_subspace_split.main()` | | | | Through `--target`; run as a subprocess by `run_subspace.py` (morph), `run_multiencoder.py` (morph) and `run_stainaudit.py` (stain) |
| `probe_stain_nonlinear.py` | `v5nucchromeos_clean_final`, `v5nucchromeos` | nuclear, chromatin, eosin | same | Needs the three blocks; `names.index` raises otherwise |
| `check_groups.py` | `v5nucchromeos_clean_final`, `v5nucchromeos` | nuclear, chromatin, eosin, clean drops | same | Correct on a fresh root; prints whatever a wrong cache holds |
| `list_stain_cols.py` | `v5nucchromeos_stainprobe`, `v5nucchromeos` | nuclear, chromatin, eosin | same | As above |
| `bridge_fm_interpretable.run_arm()`, interpretable arm | `v5nucchromeos_clean_additive`, `v5nucchromeos` | nuclear; drops graph, chromatin texture, other eosin; excludes `dead_` | nuclear, chromatin, eosin | Yes. On a fresh root `check_feature_match()` refuses (column count against the checkpoint), but only after `load_bags()` has saved a nuclear-only cache under `v5nucchromeos` |
| `bridge_fm_interpretable.run_arm()`, encoder arm | `v5fmphikonv2_fm_addmil`, `v5fmphikonv2` | none (graph only) | embeddings only | Yes. `bags_for()` cannot load an embedding block at all. On a fresh root the count check refuses, after a graph-only cache has been saved under `v5fmphikonv2` |
| `stability_vs_ensemble.run()` | the same two tags as `bridge_fm_interpretable.py` | nuclear with the clean drops; graph only | as above | As for `bridge_fm_interpretable.py` |
| `explain_slide.main()` | `--tag` (default `v5nuc`) | nuclear, plus chromatin and eosin from its flags | by tag | Correct when the flags match the tag; `models_for()` compares the checkpoint's `feat_names` with the loaded names |
| `analyse_abmil_nuclear.py` modes `confusion`, `attention`, `stability`, `ensemble` and `attnensemble` | `--tag` (default `v5nuc`), `--base-tag` (default `v5`) | nuclear only (or none); only `attnensemble` passes `--drop-groups` | nuclear for the defaults | Consistent for the default tags. `collect()` compares column counts with the checkpoint. See the note below |
| `run_overnight.seed_agreement()` | `v5nuc_<arm>`, `v5nuc` | nuclear; drops graph | nuclear | Consistent |
| `stability_table.py` (`ARMS`) | `v5nuc`, `v5nuc_sel_*`, `v5nuc` | nuclear | nuclear | Consistent |
| `train_abmil.load_bags()` | composed from the feature flags | as flagged | defines it | Reuses any cache under the composed name, whatever it holds (see also `docs/LIMITATIONS.md` item 5) |
| `batch_residual.py`, `check_drivers.py` (`load_bags()` directly) | `bag_cache_v5nuc.pt` | nuclear | nuclear | Consistent |
| `resolve_dead_count.main()` (`load_bags()` directly) | `--tag` (default `v5nuc`) | nuclear | nuclear | Consistent for the default |
| `train_slidelevel_gbm.main()` (`load_bags()` directly) | its own tag, plus `nuc` | nuclear if flagged | same | Consistent |
| `test_chrom_load.py` | | | | DEV-ONLY; always rebuilds |
| `eval_external.py` | none | its own loader | | No cache by design (its `load_external_bags()` docstring) |
| `run_cleanfinal.py`, `run_phase3.py`, `run_phase4.py`, `run_nodead.py` | | | | Start `train_abmil.py` and `analyse_abmil_nuclear.py` as subprocesses; covered by those rows |

A note on `analyse_abmil_nuclear.py`: in the recovered text, its `stability`,
`ensemble`, `attention` and `confusion` modes parse `--drop-groups` and
`--exclude-prefix` but do not pass them to `bags_for()`. Yet the original
`run_cleanfinal.py` stability step on `v5nucchromeos_clean_final` printed
`bags=456 features=49` and exited 0
(session log, private recovery bundle). With the recovered
text that call would load 153 columns (all 154 less `nuc_have_cv2`) and be
refused by the checkpoint check.
So the version that ran probably differed from the recovered text. This was
not resolved and is outside this fix.

## The 49 columns

Outcome A: a logged list of the 49 names exists.

* Session log, 2026-09-16 (private recovery bundle):
  `inspect_ckpt.py` printed the `feat_names` of the
  `v5nucchromeos_clean_final` checkpoints. There are 49 names, in order, and
  `in_dim` is 49. These are the columns the clean model was trained on, from
  the same cache, with the same drops and exclusion.
* Session log, 2026-09-04 (private recovery bundle): `fm_subspace_split.py
  --check` logged the `named_features("morph")` call itself. It loaded
  `bag_cache_v5nucchromeos.pt` with nuclear 41, chromatin 39 and eosin 35
  columns and excluded 2 `dead_` columns. It removed 102 columns by group and
  dropped `nuc_have_cv2`, which left 49 columns over 456 slides. The surviving
  groups were nuclear shape 17, spacing 6, density 10, flags 6 (5 after
  `nuc_have_cv2`), chromatin level 6 and eosin clearing 5.

The list comes from the checkpoints, not from `named_features()` itself, which
logged only the count and the group counts. The two agree group by group, and
both go through `load_bags()` with the same tag, drops and exclusion. The list
is in `tests/test_named_features_cache.py` (`LOGGED_49`). A script compared
that transcription with the log text and found it identical, in order.

## The change

In `scripts/pipeline/fm_subspace_split.py` only, 29 lines added and 1 removed, in 3
hunks:

* The morph branch now passes `chromatin=True, eosin=True` to `bags_for()`, the
  blocks the original cache held. On a fresh root it builds the same cache
  `train_abmil.py` builds for that tag.
* A new `check_bag_cache()` runs first in the morph branch. If
  `bag_cache_v5nucchromeos.pt` exists, has the current schema, and was not
  built from exactly the nuclear, chromatin and eosin blocks, it stops with
  `SystemExit`. The message names the file and its blocks and says to delete
  it. A cache with an older schema is left alone, because `load_bags()`
  rebuilds it from the blocks requested.
* Imports: `torch`, and `OUT_SUBFOLDER`, `RUN_DIR_NAME` and `SCHEMA_VERSION`
  from `train_abmil`, which `analyse_abmil_nuclear` already imports.

When a correct cache exists, the guard passes silently. `load_bags()` then
reads the cache as before, and the block arguments it receives are not used.
The cost is one extra read of the cache file per call; `load_bags()` already
reads it twice.

Not changed:

* `bags_for()` and `load_bags()`.
* The drop rules, the `INT_*` constants, feature definitions, defaults and seeds.
* The stain branch, and every other caller.

Keying the cache on its blocks would change the cache name for every
`bags_for()` caller, so an existing correct cache would no longer be found.
That fix was not used.

`smoke/integration_check.py` asked for the stain block first to avoid the old
behaviour. It now asks for the morph block first, so the five-slide run
exercises the fixed call.

## Validation

All on synthetic data (`tests/synthetic_corpus.py`):

* 25 slides, five per class, each with 6 to 14 tiles.
* Every block carries the real column names: the builder's `node_features()`
  for the graph block, and each extractor's `FEATURE_NAMES` for the nuclear,
  chromatin and eosin blocks.
* Values are random. Each block stores the graph's `grid_rc_checksum`.
* `nuc_have_cv2` is held at 1.0, as on the real corpus.

Five slides per class are needed because `flag_leak_report()` refuses fewer.

`tests/test_named_features_cache.py` has 9 tests. Before the fix (commit
eeff07a) 2 passed and 7 were expected failures, each failing for the stated
reason:

* 38 columns on a fresh root;
* the stain call exiting with "stain probe columns absent";
* a nuclear-only cache left behind;
* four caches built from other blocks (nuclear; nuclear and chromatin; nuclear
  and eosin; all three plus embeddings), none of them refused.

After the fix all 9 pass. In the full suite the 7 expected failures became passes.

Old and new code were also run on copies of one synthetic root:

| Code | Cache | Columns | Same list and order as the old code with a correct cache | Same as the logged list |
|---|---|---|---|---|
| the recovered original | correct, all three blocks | 49 | yes | yes |
| `scripts/` after the portability pass | correct, all three blocks | 49 | yes | yes |
| `scripts/` after the portability pass | none (fresh root) | 38 | no | no |
| `scripts/` with the fix | none (fresh root) | 49 | yes | yes |
| `scripts/` with the fix | correct, all three blocks | 49 | yes | yes |

## Limits

* Synthetic data only. Which columns `drop_degenerate()` removes depends on the
  data. On the 456-slide corpus it removed only `nuc_have_cv2`, and the
  synthetic corpus reproduces that by construction, not by measurement. A
  small real corpus, such as the five-slide panel, could have another constant
  column and so fewer than 49.
* The guard trusts the cache's own record of its blocks. It cannot tell a cache
  built before every feature file existed (`docs/LIMITATIONS.md` item 5), or
  one built from other feature files under the same block names.
* Only the clean model's tag is checked. The stain branch and the callers
  listed above as depending on an existing cache are unchanged.

# Copyright 2026 Vidip Gaur, Vidu Gaur
"""fm_subspace_split.named_features("morph") must give the same 49 columns
whatever bag cache already exists under its tag.

The expected list is the feat_names of the v5nucchromeos_clean_final
checkpoints, printed in order by inspect_ckpt.py on the original machine. The original
named_features() run reported the same count and per-group counts.
"""
import pytest
import torch

import fm_subspace_split as S
import synthetic_corpus as SC

LOGGED_49 = [
    "nuc_count", "nuc_per_mm2", "neo_count", "inflam_count",
    "connective_count", "epithelial_count", "neo_frac", "inflam_frac",
    "connective_frac", "epithelial_frac", "neo_area_um2_mean", "neo_area_um2_std",
    "neo_area_um2_cv", "neo_area_um2_p10", "neo_area_um2_p90", "neo_area_p90_p10_ratio",
    "neo_equivdiam_um_mean", "neo_ecc_mean", "neo_ecc_std", "neo_axisratio_mean",
    "neo_circularity_mean", "neo_solidity_mean", "neo_solidity_std", "all_area_um2_mean",
    "all_area_um2_cv", "all_ecc_mean", "all_solidity_mean", "nnd_all_um_mean",
    "nnd_all_um_cv", "nnd_neo_um_mean", "til_inflam_nn_neo_um_mean", "til_frac_neo_infiltrated",
    "til_inflam_per_neo", "morph_measured", "neo_measured", "til_measured",
    "nnd_all_measured", "nnd_neo_measured", "neo_h_mean_mean", "neo_h_ring_diff_mean",
    "neo_h_mean_cv", "all_h_mean_mean", "all_h_ring_diff_mean", "all_h_mean_cv",
    "tile_clear_frac", "tile_pale_frac", "tile_region_frac", "neo_ring_empty_frac_mean",
    "all_ring_empty_frac_mean",
]


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = SC.make_root(tmp_path)
    monkeypatch.setattr(S, "ROOT", r)
    return r


def morph_names():
    feats, names = S.named_features("morph")
    assert {x.shape[1] for x in feats.values()} == {len(names)}
    return names


def test_logged_list_has_49_distinct_names():
    assert len(LOGGED_49) == 49
    assert len(set(LOGGED_49)) == 49


def test_fresh_root_gives_the_logged_columns(root):
    assert morph_names() == LOGGED_49


def test_existing_cache_of_all_three_blocks_gives_the_logged_columns(root):
    SC.build_cache(root, ("nuclear", "chromatin", "eosin"))
    assert morph_names() == LOGGED_49


def test_fresh_root_leaves_a_cache_the_stain_probe_can_use(root):
    morph_names()
    feats, names = S.named_features("stain")
    assert names == S.STAIN_COLS


def test_fresh_root_cache_holds_all_three_blocks(root):
    morph_names()
    cache = root / "cellvit_run" / "graphs" / "gatv2_run" / f"bag_cache_{SC.CACHE_TAG}.pt"
    blob = torch.load(cache, weights_only=False)
    assert [b for b in ("nuclear", "chromatin", "eosin", "fm") if b in blob] == \
        ["nuclear", "chromatin", "eosin"]


@pytest.mark.parametrize("blocks", [
    ("nuclear",),
    ("nuclear", "chromatin"),
    ("nuclear", "eosin"),
    ("nuclear", "chromatin", "eosin", "fm"),
], ids="+".join)
def test_cache_of_other_blocks_is_refused(root, blocks):
    SC.build_cache(root, blocks)
    with pytest.raises(SystemExit, match="nuclear, chromatin and eosin"):
        S.named_features("morph")

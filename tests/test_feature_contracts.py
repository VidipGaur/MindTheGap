# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Feature-name contracts. The expected column counts come from the logs of
the original runs, not from the code under test:

  nuclear 39    nuc_features (971, 39) for slide 10077 and (600, 39) for 10548
  chromatin 39  "features : 39 per tile", printed by extract_chromatin_features.py
                (logged 9 times)
  eosin 35      "35 features per tile", printed by extract_eosin_features.py
                (one earlier run printed 34)
  stain 10      "stain features: 10 columns", printed by fm_subspace_split.py
"""
import pytest

import embed_tiles_fm
import extract_chromatin_features
import extract_eosin_features
import extract_nuclear_features
import fm_subspace_split

EXTRACTORS = [
    (extract_nuclear_features, 39),
    (extract_chromatin_features, 39),
    (extract_eosin_features, 35),
]


@pytest.mark.parametrize("mod, n", EXTRACTORS, ids=lambda v: getattr(v, "__name__", str(v)))
def test_extractor_names(mod, n):
    names = list(mod.FEATURE_NAMES)
    assert len(names) == n
    assert len(set(names)) == len(names)
    assert set(mod.FLAG_COLS) <= set(names)
    assert set(mod.LOG1P_COLS) <= set(names)
    assert not set(mod.FLAG_COLS) & set(mod.LOG1P_COLS)


def test_stain_probe_columns():
    cols = fm_subspace_split.STAIN_COLS
    assert len(cols) == 10
    assert len(set(cols)) == 10
    assert set(cols) <= set(extract_eosin_features.FEATURE_NAMES) | set(extract_chromatin_features.FEATURE_NAMES)


def test_encoder_widths():
    dims = {k: v["dim"] for k, v in embed_tiles_fm.MODELS.items()}
    assert dims == {"phikonv2": 1024, "phikon": 768, "uni2h": 1536,
                    "hoptimus0": 1536, "hibouL": 1024}


@pytest.mark.parametrize("name", ["uni2h", "hoptimus0", "hibouL"])
def test_card_normalisation_is_recorded(name):
    spec = embed_tiles_fm.MODELS[name]
    assert len(spec["card_mean"]) == 3 and len(spec["card_std"]) == 3

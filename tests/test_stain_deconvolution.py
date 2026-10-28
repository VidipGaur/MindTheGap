# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Colour deconvolution in extract_chromatin_features.py on a synthetic image
built from known haematoxylin and eosin densities with the script's own fixed
reference vectors (REF_HE), via Beer-Lambert."""
import numpy as np

import extract_chromatin_features as X


def synthetic(c_h, c_e, shape=(8, 8)):
    """uint8 RGB whose optical density is REF_HE @ [c_h, c_e]."""
    od = X.REF_HE.astype(np.float64) @ np.array([c_h, c_e])
    rgb = 255.0 * 10.0 ** (-od)
    return np.broadcast_to(np.round(rgb).clip(0, 255).astype(np.uint8), (*shape, 3)).copy()


def test_h_and_e_land_in_their_own_channels():
    left = synthetic(0.8, 0.0)
    right = synthetic(0.0, 0.8)
    img = np.concatenate([left, right], axis=1)
    h, e = X.deconvolve(img)
    hl, hr = h[:, :8].mean(), h[:, 8:].mean()
    el, er = e[:, :8].mean(), e[:, 8:].mean()
    assert abs(hl - 0.8) < 0.02 and abs(hr) < 0.02
    assert abs(er - 0.8) < 0.02 and abs(el) < 0.02


def test_mixture_is_recovered():
    h, e = X.deconvolve(synthetic(0.5, 0.3))
    assert abs(h.mean() - 0.5) < 0.02
    assert abs(e.mean() - 0.3) < 0.02


def test_lookup_table_path_matches_least_squares():
    rng = np.random.default_rng(0)
    img = rng.integers(20, 235, size=(16, 16, 3), dtype=np.uint8)
    h_lstsq, _ = X.deconvolve(img)
    h_lut = X.deconvolve_h(img)
    assert np.allclose(h_lut, h_lstsq, atol=1e-4)


def test_tissue_mask_excludes_glass_and_black_padding():
    img = np.zeros((1, 3, 3), np.uint8)
    img[0, 0] = 255                 # glass
    img[0, 1] = 0                   # scanner padding
    img[0, 2] = (150, 90, 160)      # stained tissue
    assert X.tissue_mask(img).tolist() == [[False, False, True]]

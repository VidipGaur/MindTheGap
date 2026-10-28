# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Deletion algebra of fm_subspace_split.py on synthetic data.

The ridge map, the residual, the rank-matched null and the variance-matched
control are computed inline in main(), mixed with file I/O, so they cannot be
called on their own. Instead main() runs unchanged with only its two I/O seams
replaced: named_features() returns synthetic feature blocks, and FMDIR points
at a temporary folder holding synthetic embeddings. The properties are checked
on the residual files main() writes. ALPHAS is patched to a single tiny value so
the orthogonality property is exact up to float32 storage.
"""
import numpy as np
import pytest

import fm_subspace_split as S

N_SLIDES, N_FEAT, E_DIM = 7, 5, 16


@pytest.fixture
def decomposition(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    W_true = rng.normal(size=(N_FEAT + 1, E_DIM))
    feats, emb = {}, {}
    for k in range(N_SLIDES):
        iid = str(1000 + k)
        n = int(rng.integers(40, 90))
        F = rng.normal(size=(n, N_FEAT))
        E = np.hstack([np.ones((n, 1)), F]) @ W_true + rng.normal(size=(n, E_DIM))
        feats[iid], emb[iid] = F, E
        np.savez_compressed(tmp_path / f"{iid}_{S.SRC}.npz",
                            fm_features=E.astype(np.float32),
                            grid_rc_checksum=np.array([7 * k + 3]))
    names = [f"f{j}" for j in range(N_FEAT)]
    monkeypatch.setattr(S, "named_features", lambda target="morph": (feats, names))
    monkeypatch.setattr(S, "FMDIR", tmp_path)
    monkeypatch.setattr(S, "ALPHAS", (1e-9,))
    monkeypatch.setattr("sys.argv", ["fm_subspace_split.py"])
    assert S.main() == 0

    out = {}
    for iid in sorted(feats):
        E = np.load(tmp_path / f"{iid}_{S.SRC}.npz")["fm_features"].astype(np.float64)
        arms = {}
        for tag in ("resid", "residnull", "residctl"):
            with np.load(tmp_path / f"{iid}_{S.SRC}{tag}.npz") as z:
                arms[tag] = z["fm_features"].astype(np.float64)
                arms[tag + "_ck"] = int(z["grid_rc_checksum"][0])
        out[iid] = dict(F=feats[iid], E=E, **arms)
    return out


def stack(d, key):
    return np.vstack([v[key] for v in d.values()])


def test_outputs_keep_shape_and_checksum(decomposition):
    for k, (iid, v) in enumerate(sorted(decomposition.items())):
        for tag in ("resid", "residnull", "residctl"):
            assert v[tag].shape == v["E"].shape
            assert v[tag + "_ck"] == 7 * k + 3


def test_fitted_plus_residual_is_one_linear_map_of_the_features(decomposition):
    Fa = np.hstack([np.ones((len(stack(decomposition, "F")), 1)), stack(decomposition, "F")])
    fitted = stack(decomposition, "E") - stack(decomposition, "resid")
    W, *_ = np.linalg.lstsq(Fa, fitted, rcond=None)
    assert np.abs(Fa @ W - fitted).max() < 1e-4 * np.abs(fitted).max()


def test_residual_is_orthogonal_to_span_of_features(decomposition):
    F = stack(decomposition, "F")
    Fa = np.hstack([np.ones((len(F), 1)), F])
    R = stack(decomposition, "resid")
    E = stack(decomposition, "E")
    rel = np.abs(Fa.T @ R).max() / (np.linalg.norm(Fa) * np.linalg.norm(E))
    assert rel < 1e-6


def test_null_removes_a_component_of_the_same_rank(decomposition):
    E = stack(decomposition, "E")
    real = E - stack(decomposition, "resid")
    null = E - stack(decomposition, "residnull")
    tol = 1e-4 * np.abs(E).max()
    assert np.linalg.matrix_rank(real, tol=tol) == N_FEAT + 1
    assert np.linalg.matrix_rank(null, tol=tol) == N_FEAT + 1


def test_null_is_built_from_other_slides():
    ids = [str(i) for i in range(5)]
    partner = S.shuffled_partner(ids)
    assert all(partner[i] != i for i in ids)
    assert sorted(partner.values()) == ids


def test_resample_rows_matches_row_count():
    rng = np.random.default_rng(0)
    F = np.arange(12.0).reshape(6, 2)
    assert S.resample_rows(F, 6, rng) is F
    G = S.resample_rows(F, 9, rng)
    assert G.shape == (9, 2)
    assert set(map(tuple, G)) <= set(map(tuple, F))


def test_control_deletes_matched_variance(decomposition):
    E = stack(decomposition, "E")
    mu = E.mean(0)
    Ec = E - mu
    total = (Ec ** 2).sum()
    frac = 1.0 - (stack(decomposition, "resid") ** 2).sum() / total
    removed = 1.0 - (stack(decomposition, "residctl") ** 2).sum() / total
    share = np.linalg.eigvalsh(Ec.T @ Ec / len(Ec))
    share = share / share.sum()
    # main() adds random components until the deleted share reaches frac, so it
    # can overshoot by at most one component's share
    assert removed >= frac - 1e-4
    assert removed <= frac + share.max() + 1e-4

# Copyright 2026 Vidip Gaur, Vidu Gaur
"""The paper's interpretability property: in Additive MIL the per-tile, per-class
contributions add up exactly to the bag logit.

There is no separate bias after pooling. The instance head's bias is weighted by
attention that sums to one over the valid tiles, so it is carried inside the
contributions; test_bias_is_inside_the_contributions checks that directly.
"""
import pytest
import torch

import train_abmil as T

N_CLASSES = len(T.CLASSES)


def small_model(in_dim=7, heads=1, seed=0):
    torch.manual_seed(seed)
    m = T.AdditiveMIL(in_dim, N_CLASSES, hidden=16, layers=1, attn_hidden=8,
                      dropout=0.5, heads=heads).double().eval()
    return m


def bag(n=23, in_dim=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, n, in_dim, generator=g, dtype=torch.float64)
    return x, torch.ones(1, n, dtype=torch.bool)


def test_reference_config_builds_additive_mil():
    cfg = next(c for c in T.CONFIGS if c[0] == "addmil_h128_do5")
    assert T.cfg_field(cfg, "arch") == "additive"
    assert isinstance(T.build_model(cfg, 49, N_CLASSES), T.AdditiveMIL)


@pytest.mark.parametrize("heads", [1, 4])
def test_contributions_sum_to_logits(heads):
    m = small_model(heads=heads)
    x, mask = bag()
    with torch.no_grad():
        out, amap, contrib = m(x, mask, return_contrib=True)
    assert contrib.shape == (1, x.shape[1], N_CLASSES)
    assert torch.allclose(contrib.sum(1), out, rtol=0, atol=1e-12)


def test_contributions_match_an_independent_recomputation():
    m = small_model()
    x, mask = bag()
    with torch.no_grad():
        out, _, contrib = m(x, mask, return_contrib=True)
        h = m.encoder(x)
        a = m.attn(h, mask)[..., 0]                 # (1, N)
        psi = m.inst(h)                             # (1, N, C)
        manual = a.unsqueeze(-1) * psi
    assert torch.allclose(a.sum(1), torch.ones(1, dtype=a.dtype), atol=1e-12)
    assert torch.allclose(manual, contrib, atol=1e-12)
    assert torch.allclose(manual.sum(1), out, atol=1e-12)


def test_bias_is_inside_the_contributions():
    m = small_model()
    x, mask = bag()
    bias = m.inst[-1].bias
    with torch.no_grad():
        h = m.encoder(x)
        a = m.attn(h, mask)[..., 0]
        bias_part = (a.unsqueeze(-1) * bias).sum(1)
    assert torch.allclose(bias_part, bias.unsqueeze(0), atol=1e-12)


def test_padding_contributes_nothing():
    m = small_model()
    x, mask = bag(n=11)
    pad = torch.cat([x, torch.randn(1, 9, x.shape[2], dtype=x.dtype) * 50], 1)
    pmask = torch.cat([mask, torch.zeros(1, 9, dtype=torch.bool)], 1)
    with torch.no_grad():
        out, _, _ = m(x, mask, return_contrib=True)
        pout, pamap, pcontrib = m(pad, pmask, return_contrib=True)
    assert torch.allclose(out, pout, atol=1e-12)
    assert torch.count_nonzero(pcontrib[0, 11:]) == 0
    assert torch.count_nonzero(pamap[0, 11:]) == 0


def test_batched_bags_match_single_bags():
    m = small_model()
    (x1, _), (x2, _) = bag(n=5, seed=3), bag(n=12, seed=4)
    X, M, _ = T.pad_collate([(x1[0], 0), (x2[0], 1)])
    X = X.double()
    with torch.no_grad():
        both = m(X, M)
        one = m(X[0:1, :5], M[0:1, :5])
        two = m(X[1:2, :12], M[1:2, :12])
    assert torch.allclose(both[0], one[0], atol=1e-12)
    assert torch.allclose(both[1], two[0], atol=1e-12)

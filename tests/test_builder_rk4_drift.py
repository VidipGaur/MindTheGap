# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Check [2] of the graph builder's self-test ("RK4 drift"), run as a test.

--self-test prints checks [1] and [3]-[9]; this test covers [2] on the same
synthetic field.

The forward drift |d_col| = [795, 495] is the expected post-fix value. The
backward pass must move the opposite way: before the bidirectional fix it was a
bit-identical rerun of forward.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import build_streamline_graphs_bridge as B


@pytest.fixture(scope="module")
def field():
    ds = B.EXPECTED_DOWNSAMPLE
    H, W = 300, 900
    rr = np.arange(H, dtype=np.float32)[:, None]
    dens = (1.0 + np.sin(2 * np.pi * rr / 60.0)).astype(np.float32) * np.ones((1, W), np.float32)
    dens = gaussian_filter(dens, 3.0)
    gr, gc = B.density_gradients(dens, max(1.0, B.um_to_px(B.GRAD_SIGMA_UM, ds)))
    v_r, v_c, coh = B.structure_tensor_from_gradients(
        gr, gc, max(1.0, B.um_to_px(B.TENSOR_SCALES_UM[0], ds)))
    tissue = np.ones((H, W), dtype=bool)
    node_of, _, _, _ = B.build_node_grid(tissue, B.um_to_px(B.NODE_SIZE_UM, ds))
    seeds = np.array([[150.0, 100.0], [80.0, 400.0]], dtype=np.float32)
    step_px = B.um_to_px(B.STEP_UM, ds)
    n_steps = int(B.um_to_px(B.MAX_LENGTH_UM, ds) / step_px)

    def integrate(direction):
        _, end, _, _ = B.integrate_rk4(v_r, v_c, coh, tissue, seeds, node_of, step_px,
                                       n_steps, B.MIN_COHERENCE, B.MAX_ANGLE_DEG,
                                       direction=direction)
        return end
    return seeds, integrate


def test_check2_rk4_follows_the_bands(field):
    seeds, integrate = field
    end = integrate(1.0)
    d_row = np.abs(end[:, 0] - seeds[:, 0])
    d_col = np.abs(end[:, 1] - seeds[:, 1])
    assert d_col.min() > 20 and d_row.max() < 3
    assert d_col.round(1).tolist() == [795.0, 495.0]


def test_backward_pass_moves_the_other_way(field):
    seeds, integrate = field
    fwd, bwd = integrate(1.0), integrate(-1.0)
    assert (fwd[:, 1] > seeds[:, 1]).all()
    assert (bwd[:, 1] < seeds[:, 1]).all()

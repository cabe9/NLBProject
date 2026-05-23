"""Shape and param-sensitivity tests for the direct-ridge baseline.

Guards that the static Gaussian-ridge readout on raw held-in rates returns
the expected tensor shapes and that changing ``ridge_alpha`` actually
changes the output (i.e. the sweep is doing real work).
"""

import numpy as np

from nlb_project.models.ridge_direct import fit_predict_ridge_direct


def test_ridge_direct_shapes():
    """Direct ridge returns train/eval tensors matching the held-out shape."""
    rng = np.random.default_rng(11)
    train_hi = rng.poisson(0.5, (7, 20, 10)).astype(float)
    train_ho = rng.poisson(0.5, (7, 20, 4)).astype(float)
    eval_hi = rng.poisson(0.5, (3, 20, 10)).astype(float)

    out = fit_predict_ridge_direct(
        train_hi,
        train_ho,
        eval_hi,
        ridge_alpha=0.1,
    )

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldin"].shape == eval_hi.shape
    assert out["eval_rates_heldout"].shape == (3, 20, 4)


def test_ridge_direct_params_change_output():
    """Distinct ``ridge_alpha`` values produce distinct predictions."""
    rng = np.random.default_rng(29)
    train_hi = rng.poisson(0.7, (8, 30, 12)).astype(float)
    train_ho = rng.poisson(0.7, (8, 30, 5)).astype(float)
    eval_hi = rng.poisson(0.7, (4, 30, 12)).astype(float)

    out_small_alpha = fit_predict_ridge_direct(
        train_hi,
        train_ho,
        eval_hi,
        ridge_alpha=1e-3,
    )
    out_large_alpha = fit_predict_ridge_direct(
        train_hi,
        train_ho,
        eval_hi,
        ridge_alpha=1e-1,
    )

    assert not np.allclose(
        out_small_alpha["eval_rates_heldout"],
        out_large_alpha["eval_rates_heldout"],
    )

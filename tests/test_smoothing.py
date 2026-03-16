import numpy as np

from nlb_project.smoothing import SmoothingParams, predict_rates


def test_predict_rates_shapes_and_finite():
    rng = np.random.default_rng(0)
    train_hi = rng.poisson(0.4, (5, 30, 4)).astype(float)
    train_ho = rng.poisson(0.4, (5, 30, 2)).astype(float)
    eval_hi = rng.poisson(0.4, (3, 30, 4)).astype(float)

    params = SmoothingParams(kern_sd_ms=30, alpha=0.01, log_offset=1e-4)
    out = predict_rates(train_hi, train_ho, eval_hi, params, bin_size_ms=5)

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldin"].shape == eval_hi.shape
    assert out["eval_rates_heldout"].shape == (3, 30, 2)
    assert np.all(np.isfinite(out["eval_rates_heldout"]))

import numpy as np

from nlb_project.smoothing import SmoothingParams, predict_rates


def test_predictions_change_with_different_params():
    rng = np.random.default_rng(123)
    train_hi = rng.poisson(0.6, (6, 40, 5)).astype(float)
    train_ho = rng.poisson(0.6, (6, 40, 3)).astype(float)
    eval_hi = rng.poisson(0.6, (4, 40, 5)).astype(float)

    p1 = SmoothingParams(kern_sd_ms=20, alpha=0.001, log_offset=1e-4)
    p2 = SmoothingParams(kern_sd_ms=70, alpha=0.1, log_offset=1e-4)

    out1 = predict_rates(train_hi, train_ho, eval_hi, p1, bin_size_ms=5)
    out2 = predict_rates(train_hi, train_ho, eval_hi, p2, bin_size_ms=5)

    assert not np.allclose(out1["eval_rates_heldout"], out2["eval_rates_heldout"])

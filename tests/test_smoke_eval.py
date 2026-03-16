import numpy as np
import pytest


def test_nlb_evaluate_smoke():
    nlb_eval = pytest.importorskip("nlb_tools.evaluation")

    rng = np.random.default_rng(0)
    train_trials, eval_trials, t, n_hi, n_ho = 8, 6, 20, 5, 3

    train_rates_heldin = np.clip(rng.random((train_trials, t, n_hi)), 1e-4, None)
    train_rates_heldout = np.clip(rng.random((train_trials, t, n_ho)), 1e-4, None)
    eval_rates_heldin = np.clip(rng.random((eval_trials, t, n_hi)), 1e-4, None)
    eval_rates_heldout = np.clip(rng.random((eval_trials, t, n_ho)), 1e-4, None)

    eval_spikes_heldout = rng.poisson(0.5, (eval_trials, t, n_ho)).astype(float)
    train_behavior = rng.standard_normal((train_trials, t, 2)).astype(float)
    eval_behavior = rng.standard_normal((eval_trials, t, 2)).astype(float)

    target = {
        "mc_maze": {
            "eval_spikes_heldout": eval_spikes_heldout,
            "train_behavior": train_behavior,
            "eval_behavior": eval_behavior,
        }
    }
    user = {
        "mc_maze": {
            "train_rates_heldin": train_rates_heldin,
            "train_rates_heldout": train_rates_heldout,
            "eval_rates_heldin": eval_rates_heldin,
            "eval_rates_heldout": eval_rates_heldout,
        }
    }

    res = nlb_eval.evaluate(target, user)
    metrics = res[0]["mc_maze_split"]
    assert np.isfinite(metrics["co-bps"])
    assert "vel R2" in metrics

import numpy as np

from nlb_project.models.pca_latent_regression import predict_pca_latent_regression


def test_pca_latent_regression_shapes():
    rng = np.random.default_rng(7)
    train_hi = rng.poisson(0.5, (6, 25, 12)).astype(float)
    train_ho = rng.poisson(0.5, (6, 25, 4)).astype(float)
    eval_hi = rng.poisson(0.5, (3, 25, 12)).astype(float)

    out = predict_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=5,
        ridge_alpha=0.1,
    )

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldin"].shape == eval_hi.shape
    assert out["eval_rates_heldout"].shape == (3, 25, 4)


def test_pca_latent_regression_params_change_output():
    rng = np.random.default_rng(19)
    train_hi = rng.poisson(0.6, (8, 30, 15)).astype(float)
    train_ho = rng.poisson(0.6, (8, 30, 5)).astype(float)
    eval_hi = rng.poisson(0.6, (4, 30, 15)).astype(float)

    out_a = predict_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=5,
        ridge_alpha=0.001,
    )
    out_b = predict_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=12,
        ridge_alpha=1.0,
    )

    assert not np.allclose(out_a["eval_rates_heldout"], out_b["eval_rates_heldout"])

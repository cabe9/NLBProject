from __future__ import annotations

import numpy as np
import pytest

from nlb_project.models.ndt_lite import fit_predict_ndt_lite

pytest.importorskip("torch")


def test_ndt_lite_shapes_and_positive_rates() -> None:
    rng = np.random.default_rng(91)
    train_hi = rng.poisson(0.5, (5, 12, 7)).astype(np.float32)
    train_ho = rng.poisson(0.4, (5, 12, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 12, 7)).astype(np.float32)

    out = fit_predict_ndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        learning_rate=0.003,
        weight_decay=0.0,
        batch_size=2,
        max_epochs=2,
        patience=2,
        mask_prob=0.25,
        heldin_loss_weight=0.1,
        validation_fraction=0.2,
        input_transform="sqrt_zscore",
        seed=0,
        device="cpu",
    )

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldin"].shape == eval_hi.shape
    assert out["eval_rates_heldout"].shape == (2, 12, 3)
    assert np.all(out["eval_rates_heldout"] > 0.0)

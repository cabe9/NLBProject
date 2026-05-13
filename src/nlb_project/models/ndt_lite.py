"""A small Neural Data Transformer-style baseline.

This is intentionally a first neural-sequence baseline, not a full STNDT
reimplementation. It uses temporal self-attention over held-in spike history,
random held-in masking during training, and Poisson rate losses for both
held-out prediction and masked held-in reconstruction.

PyTorch is an optional dependency. Install with ``pip install -e .[neural]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .temporal_features import apply_input_transform


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `ndt_lite` model requires PyTorch. Install it with "
            "`python -m pip install -e '.[neural]'`."
        ) from exc
    return torch, nn, functional


def _resolve_device(torch: Any, device: str) -> Any:
    device = device.lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _reshape_transform(
    train_hi: np.ndarray,
    eval_hi: np.ndarray,
    *,
    input_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_2d = train_hi.reshape(-1, train_hi.shape[-1])
    eval_2d = eval_hi.reshape(-1, eval_hi.shape[-1])
    train_x, eval_x = apply_input_transform(train_2d, eval_2d, transform=input_transform)
    return train_x.reshape(train_hi.shape), eval_x.reshape(eval_hi.shape)


def _poisson_loss(functional: Any, pred: Any, target: Any) -> Any:
    return functional.poisson_nll_loss(pred, target, log_input=False, full=False)


def _temporal_transformer_cls(
    nn: Any,
    functional: Any,
    torch: Any,
    *,
    n_heldin: int,
    n_heldout: int,
    d_model: int,
    max_t_len: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    min_rate: float,
) -> type:
    """Return ``nn.Module`` subclass; defined after Torch import so it stays optional."""

    class TemporalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_heldin, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, max_t_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.heldin_head = nn.Linear(d_model, n_heldin)
            self.heldout_head = nn.Linear(d_model, n_heldout)

        def forward(self, x):
            h = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
            h = self.encoder(h)
            heldin = functional.softplus(self.heldin_head(h)) + float(min_rate)
            heldout = functional.softplus(self.heldout_head(h)) + float(min_rate)
            return heldin, heldout

    return TemporalTransformer


def fit_predict_ndt_lite(
    train_spikes_heldin: np.ndarray,
    train_spikes_heldout: np.ndarray,
    eval_spikes_heldin: np.ndarray,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    mask_prob: float,
    heldin_loss_weight: float,
    validation_fraction: float,
    input_transform: str,
    seed: int,
    lr_schedule: str = "constant",
    ensemble_size: int = 1,
    device: str = "auto",
    min_rate: float = 1e-6,
    grad_clip_norm: float = 1.0,
) -> dict[str, np.ndarray]:
    """Fit a temporal transformer and return NLB-compatible rate predictions."""
    torch, nn, functional = _require_torch()

    if d_model % n_heads != 0:
        raise ValueError("`d_model` must be divisible by `n_heads`")
    if not 0.0 <= mask_prob < 1.0:
        raise ValueError("`mask_prob` must be in [0, 1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("`validation_fraction` must be in [0, 1)")
    ensemble_size = int(ensemble_size)
    if ensemble_size < 1:
        raise ValueError("`ensemble_size` must be at least 1")
    lr_schedule = str(lr_schedule).lower()
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("`lr_schedule` must be one of {'constant', 'cosine'}")

    train_hi = np.asarray(train_spikes_heldin, dtype=np.float32)
    train_ho = np.asarray(train_spikes_heldout, dtype=np.float32)
    eval_hi = np.asarray(eval_spikes_heldin, dtype=np.float32)
    train_x, eval_x = _reshape_transform(train_hi, eval_hi, input_transform=input_transform)

    n_train, _train_t_len, n_heldin = train_hi.shape
    max_t_len = max(train_hi.shape[1], eval_hi.shape[1])
    n_heldout = train_ho.shape[-1]
    batch_size = max(1, int(batch_size))
    device_obj = _resolve_device(torch, device)

    TemporalTransformer = _temporal_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=n_heldin,
        n_heldout=n_heldout,
        d_model=d_model,
        max_t_len=max_t_len,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=float(dropout),
        min_rate=float(min_rate),
    )

    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device_obj)
    y_hi = torch.as_tensor(train_hi, dtype=torch.float32, device=device_obj)
    y_ho = torch.as_tensor(train_ho, dtype=torch.float32, device=device_obj)

    def fit_one(seed_value: int) -> dict[str, np.ndarray]:
        torch.manual_seed(int(seed_value))
        rng = np.random.default_rng(int(seed_value))
        model = TemporalTransformer().to(device_obj)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        scheduler = None
        if lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(max_epochs)),
                eta_min=float(learning_rate) * 0.1,
            )

        indices = np.arange(n_train)
        rng.shuffle(indices)
        n_val = int(round(n_train * float(validation_fraction)))
        val_idx = indices[:n_val]
        fit_idx = indices[n_val:] if n_val > 0 else indices
        if len(fit_idx) == 0:
            fit_idx = indices
            val_idx = np.array([], dtype=int)

        best_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        best_val = float("inf")
        epochs_without_improvement = 0

        def batch_loss(batch_idx: np.ndarray, *, train_mode: bool) -> Any:
            idx_tensor = torch.as_tensor(batch_idx, dtype=torch.long, device=device_obj)
            xb = x_train.index_select(0, idx_tensor)
            target_hi = y_hi.index_select(0, idx_tensor)
            target_ho = y_ho.index_select(0, idx_tensor)
            if train_mode and mask_prob > 0:
                mask = torch.rand_like(xb) < float(mask_prob)
                xb = xb.masked_fill(mask, 0.0)
            else:
                mask = torch.ones_like(xb, dtype=torch.bool)
            pred_hi, pred_ho = model(xb)
            loss = _poisson_loss(functional, pred_ho, target_ho)
            if heldin_loss_weight > 0:
                if bool(mask.any()):
                    loss_hi = _poisson_loss(functional, pred_hi[mask], target_hi[mask])
                else:
                    loss_hi = _poisson_loss(functional, pred_hi, target_hi)
                loss = loss + float(heldin_loss_weight) * loss_hi
            return loss

        for _epoch in range(int(max_epochs)):
            rng.shuffle(fit_idx)
            model.train()
            for start in range(0, len(fit_idx), batch_size):
                batch_idx = fit_idx[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(batch_idx, train_mode=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                if len(val_idx) > 0:
                    val_losses = []
                    for start in range(0, len(val_idx), batch_size):
                        batch_idx = val_idx[start : start + batch_size]
                        val_losses.append(
                            float(batch_loss(batch_idx, train_mode=False).detach().cpu())
                        )
                    val_loss = float(np.mean(val_losses))
                else:
                    val_loss = float(
                        batch_loss(fit_idx[: min(len(fit_idx), batch_size)], train_mode=False)
                        .detach()
                        .cpu()
                    )

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(patience):
                    break

        model.load_state_dict({name: tensor.to(device_obj) for name, tensor in best_state.items()})

        def predict_rates(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=device_obj)
            heldin_batches: list[np.ndarray] = []
            heldout_batches: list[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for start in range(0, x_tensor.shape[0], batch_size):
                    xb = x_tensor[start : start + batch_size]
                    pred_hi, pred_ho = model(xb)
                    heldin_batches.append(pred_hi.detach().cpu().numpy().astype(np.float32))
                    heldout_batches.append(pred_ho.detach().cpu().numpy().astype(np.float32))
            return np.concatenate(heldin_batches, axis=0), np.concatenate(heldout_batches, axis=0)

        train_rates_heldin, train_rates_heldout = predict_rates(train_x)
        eval_rates_heldin, eval_rates_heldout = predict_rates(eval_x)
        return {
            "train_rates_heldin": train_rates_heldin,
            "train_rates_heldout": train_rates_heldout,
            "eval_rates_heldin": eval_rates_heldin,
            "eval_rates_heldout": eval_rates_heldout,
        }

    member_outputs = [fit_one(int(seed) + member_idx) for member_idx in range(ensemble_size)]
    if ensemble_size == 1:
        return member_outputs[0]
    return {
        key: np.mean([member[key] for member in member_outputs], axis=0).astype(np.float32)
        for key in member_outputs[0]
    }

"""Spatiotemporal NDT-style baseline.

This is a bounded STNDT-inspired model for the repo's existing supervised NLB
pipeline. It keeps the successful NDT-lite temporal transformer, then adds a
neuron-token branch that attends across neurons and feeds spatially reweighted
population context back into the temporal state.

It is not a line-for-line reproduction of the original STNDT training stack:
the goal is a clean first implementation that can be evaluated by train/val
co-bps before spending public-test runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ndt_lite import _poisson_loss, _require_torch, _reshape_transform, _resolve_device


def _info_nce_loss(torch: Any, functional: Any, z1: Any, z2: Any, temperature: float) -> Any:
    """Symmetric in-batch InfoNCE over two masked views."""
    if z1.shape[0] < 2:
        return torch.zeros((), dtype=z1.dtype, device=z1.device)
    z1 = functional.normalize(z1, dim=1)
    z2 = functional.normalize(z2, dim=1)
    features = torch.cat([z1, z2], dim=0)
    logits = features @ features.T
    logits = logits / max(float(temperature), 1e-6)
    logits.fill_diagonal_(float("-inf"))
    batch = z1.shape[0]
    labels = torch.arange(2 * batch, device=z1.device)
    labels = (labels + batch) % (2 * batch)
    return functional.cross_entropy(logits, labels)


def _spatiotemporal_transformer_cls(
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
    spatial_n_heads: int,
    dropout: float,
    min_rate: float,
) -> type:
    """Return ``nn.Module`` subclass; defined after Torch import so it stays optional."""

    class SpatioTemporalBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=float(dropout), batch_first=True
            )
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_norm2 = nn.LayerNorm(d_model)
            self.temporal_ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(4 * d_model, d_model),
            )
            self.spatial_attn = nn.MultiheadAttention(
                max_t_len, spatial_n_heads, dropout=float(dropout), batch_first=True
            )
            self.spatial_norm1 = nn.LayerNorm(max_t_len)
            self.spatial_norm2 = nn.LayerNorm(max_t_len)
            self.spatial_update_norm = nn.LayerNorm(max_t_len)
            self.spatial_ff = nn.Sequential(
                nn.Linear(max_t_len, 2 * max_t_len),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(2 * max_t_len, max_t_len),
            )
            self.spatial_to_temporal = nn.Linear(n_heldin, d_model)
            self.temporal_to_spatial = nn.Linear(d_model, n_heldin)
            self.fusion_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(float(dropout))

        def forward(self, h, neuron_state):
            residual = h
            h_norm = self.temporal_norm1(h)
            attn_out, _ = self.temporal_attn(h_norm, h_norm, h_norm, need_weights=False)
            h = residual + self.dropout(attn_out)
            h = h + self.dropout(self.temporal_ff(self.temporal_norm2(h)))

            spatial_residual = neuron_state
            spatial_norm = self.spatial_norm1(neuron_state)
            spatial_out, spatial_weights = self.spatial_attn(
                spatial_norm,
                spatial_norm,
                spatial_norm,
                need_weights=True,
                average_attn_weights=True,
            )
            neuron_state = spatial_residual + self.dropout(spatial_out)
            neuron_state = neuron_state + self.dropout(
                self.spatial_ff(self.spatial_norm2(neuron_state))
            )

            mixed_neurons = torch.bmm(spatial_weights, neuron_state).transpose(1, 2)
            h = self.fusion_norm(h + self.dropout(self.spatial_to_temporal(mixed_neurons)))

            neuron_update = self.temporal_to_spatial(h).transpose(1, 2)
            neuron_state = self.spatial_update_norm(neuron_state + self.dropout(neuron_update))
            return h, neuron_state

    class SpatioTemporalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_heldin, d_model)
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_t_len, d_model))
            self.neuron_pos_embed = nn.Parameter(torch.zeros(1, n_heldin, max_t_len))
            self.input_norm = nn.LayerNorm(d_model)
            self.spatial_input_norm = nn.LayerNorm(max_t_len)
            self.blocks = nn.ModuleList(SpatioTemporalBlock() for _ in range(n_layers))
            self.heldin_head = nn.Linear(d_model, n_heldin)
            self.heldout_head = nn.Linear(d_model, n_heldout)
            self.heldin_rate_bias = nn.Parameter(torch.zeros(n_heldin))
            self.heldout_rate_bias = nn.Parameter(torch.zeros(n_heldout))

        def encode(self, x):
            t_len = x.shape[1]
            h = self.input_proj(x) + self.temporal_pos_embed[:, :t_len, :]
            h = self.input_norm(h)
            neuron_state = x.transpose(1, 2) + self.neuron_pos_embed[:, :, :t_len]
            neuron_state = self.spatial_input_norm(neuron_state)
            for block in self.blocks:
                h, neuron_state = block(h, neuron_state)
            return h

        def forward(self, x):
            h = self.encode(x)
            heldin_logits = self.heldin_head(h) + self.heldin_rate_bias[None, None, :]
            heldout_logits = self.heldout_head(h) + self.heldout_rate_bias[None, None, :]
            heldin = functional.softplus(heldin_logits) + float(min_rate)
            heldout = functional.softplus(heldout_logits) + float(min_rate)
            return heldin, heldout, h

    return SpatioTemporalTransformer


def fit_predict_stndt_lite(
    train_spikes_heldin: np.ndarray,
    train_spikes_heldout: np.ndarray,
    eval_spikes_heldin: np.ndarray,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    spatial_n_heads: int,
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
    contrast_loss_weight: float = 0.0,
    contrast_mask_prob: float = 0.1,
    contrast_temperature: float = 0.07,
    ensemble_size: int = 1,
    device: str = "auto",
    min_rate: float = 1e-6,
    grad_clip_norm: float = 1.0,
) -> dict[str, np.ndarray]:
    """Fit a spatiotemporal transformer and return NLB-compatible rates."""
    torch, nn, functional = _require_torch()

    if d_model % n_heads != 0:
        raise ValueError("`d_model` must be divisible by `n_heads`")
    if not 0.0 <= mask_prob < 1.0:
        raise ValueError("`mask_prob` must be in [0, 1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("`validation_fraction` must be in [0, 1)")
    if not 0.0 <= contrast_mask_prob < 1.0:
        raise ValueError("`contrast_mask_prob` must be in [0, 1)")
    contrast_loss_weight = float(contrast_loss_weight)
    if contrast_loss_weight < 0:
        raise ValueError("`contrast_loss_weight` must be non-negative")
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
    if max_t_len % int(spatial_n_heads) != 0:
        raise ValueError("time length must be divisible by `spatial_n_heads`")
    batch_size = max(1, int(batch_size))
    device_obj = _resolve_device(torch, device)

    SpatioTemporalTransformer = _spatiotemporal_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=n_heldin,
        n_heldout=n_heldout,
        d_model=d_model,
        max_t_len=max_t_len,
        n_layers=n_layers,
        n_heads=n_heads,
        spatial_n_heads=int(spatial_n_heads),
        dropout=float(dropout),
        min_rate=float(min_rate),
    )

    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device_obj)
    y_hi = torch.as_tensor(train_hi, dtype=torch.float32, device=device_obj)
    y_ho = torch.as_tensor(train_ho, dtype=torch.float32, device=device_obj)

    def fit_one(seed_value: int) -> dict[str, np.ndarray]:
        torch.manual_seed(int(seed_value))
        rng = np.random.default_rng(int(seed_value))
        model = SpatioTemporalTransformer().to(device_obj)
        with torch.no_grad():
            min_rate_tensor = torch.tensor(float(min_rate), dtype=torch.float32, device=device_obj)
            heldin_mean = torch.clamp(y_hi.mean(dim=(0, 1)) - min_rate_tensor, min=1e-6)
            heldout_mean = torch.clamp(y_ho.mean(dim=(0, 1)) - min_rate_tensor, min=1e-6)
            model.heldin_rate_bias.copy_(torch.log(torch.expm1(heldin_mean)))
            model.heldout_rate_bias.copy_(torch.log(torch.expm1(heldout_mean)))
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

        def pooled_features(xb) -> Any:
            encoded = model.encode(xb)
            return encoded.mean(dim=1)

        def batch_loss(batch_idx: np.ndarray, *, train_mode: bool) -> Any:
            idx_tensor = torch.as_tensor(batch_idx, dtype=torch.long, device=device_obj)
            xb = x_train.index_select(0, idx_tensor)
            target_hi = y_hi.index_select(0, idx_tensor)
            target_ho = y_ho.index_select(0, idx_tensor)
            if train_mode and mask_prob > 0:
                mask = torch.rand_like(xb) < float(mask_prob)
                xb_masked = xb.masked_fill(mask, 0.0)
            else:
                mask = torch.ones_like(xb, dtype=torch.bool)
                xb_masked = xb

            pred_hi, pred_ho, _features = model(xb_masked)
            loss = _poisson_loss(functional, pred_ho, target_ho)
            if heldin_loss_weight > 0:
                if bool(mask.any()):
                    loss_hi = _poisson_loss(functional, pred_hi[mask], target_hi[mask])
                else:
                    loss_hi = _poisson_loss(functional, pred_hi, target_hi)
                loss = loss + float(heldin_loss_weight) * loss_hi

            if train_mode and contrast_loss_weight > 0:
                mask_a = torch.rand_like(xb) < float(contrast_mask_prob)
                mask_b = torch.rand_like(xb) < float(contrast_mask_prob)
                z1 = pooled_features(xb.masked_fill(mask_a, 0.0))
                z2 = pooled_features(xb.masked_fill(mask_b, 0.0))
                loss = loss + contrast_loss_weight * _info_nce_loss(
                    torch, functional, z1, z2, float(contrast_temperature)
                )
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
                    pred_hi, pred_ho, _features = model(xb)
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

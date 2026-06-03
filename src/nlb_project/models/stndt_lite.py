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


def sample_block_time_mask(
    batch_size: int,
    t_len: int,
    n_neurons: int,
    mask_prob: float,
    span_length: int,
    *,
    device: Any,
) -> Any:
    """Return a ``[B, T, N]`` bool mask with contiguous time spans masked for all neurons."""
    torch, _, _ = _require_torch()
    mask_time = torch.zeros(batch_size, t_len, dtype=torch.bool, device=device)
    target_bins = max(1, int(round(float(mask_prob) * t_len)))
    span = max(1, int(span_length))
    if span >= t_len:
        mask_time[:] = True
        return mask_time.unsqueeze(-1).expand(batch_size, t_len, n_neurons)

    max_attempts = max(t_len * 4, 1)
    for batch_idx in range(batch_size):
        masked_count = 0
        attempts = 0
        while masked_count < target_bins and attempts < max_attempts:
            start = int(torch.randint(0, t_len - span + 1, (1,), device=device).item())
            mask_time[batch_idx, start : start + span] = True
            masked_count = int(mask_time[batch_idx].sum().item())
            attempts += 1
    return mask_time.unsqueeze(-1).expand(batch_size, t_len, n_neurons)


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
    temporal_identity_scale: float,
    spatial_mix_rank: int,
    spatial_mix_scale: float,
    neuron_readout_dim: int,
    neuron_readout_scale: float,
    dropout: float,
    min_rate: float,
    use_mask_token: bool = False,
    unit_calibration: bool = False,
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

            self.neuron_event_embed = None
            if temporal_identity_scale > 0:
                self.neuron_event_embed = nn.Parameter(torch.empty(n_heldin, d_model))
                nn.init.normal_(self.neuron_event_embed, mean=0.0, std=0.02)

            self.spatial_mix_down = None
            self.spatial_mix_up = None
            self.spatial_mix_to_temporal = None
            self.spatial_mix_norm = None
            if spatial_mix_rank > 0:
                self.spatial_mix_down = nn.Linear(max_t_len, spatial_mix_rank)
                self.spatial_mix_up = nn.Linear(spatial_mix_rank, max_t_len)
                self.spatial_mix_to_temporal = nn.Linear(n_heldin, d_model)
                self.spatial_mix_norm = nn.LayerNorm(d_model)

            self.neuron_readout_proj = None
            self.heldin_neuron_embed = None
            self.heldout_neuron_embed = None
            if neuron_readout_dim > 0:
                self.neuron_readout_proj = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, neuron_readout_dim),
                )
                self.heldin_neuron_embed = nn.Parameter(torch.empty(n_heldin, neuron_readout_dim))
                self.heldout_neuron_embed = nn.Parameter(torch.empty(n_heldout, neuron_readout_dim))
                nn.init.normal_(self.heldin_neuron_embed, mean=0.0, std=0.02)
                nn.init.normal_(self.heldout_neuron_embed, mean=0.0, std=0.02)

            self.mask_token = None
            if use_mask_token:
                self.mask_token = nn.Parameter(torch.zeros(n_heldin))

            self.heldout_calib_scale = None
            self.heldout_calib_bias = None
            if unit_calibration:
                self.heldout_calib_scale = nn.Parameter(torch.ones(n_heldout))
                self.heldout_calib_bias = nn.Parameter(torch.zeros(n_heldout))

        def apply_input_mask(self, x, mask):
            """Substitute masked input positions with zero-fill or a learned mask token.

            ``mask`` is a bool tensor with the same shape as ``x``. When
            ``use_mask_token`` is enabled, masked positions are replaced with a
            per-neuron learned scalar (zero-initialised) instead of plain zeros,
            following the STNDT/BERT convention of giving the model a dedicated
            token at masked locations.
            """
            if self.mask_token is None:
                return x.masked_fill(mask, 0.0)
            token = self.mask_token.view(1, 1, -1)
            return torch.where(mask, token, x)

        def encode(self, x, raw_counts=None):
            t_len = x.shape[1]
            h = self.input_proj(x) + self.temporal_pos_embed[:, :t_len, :]
            h = self.input_norm(h)
            if self.neuron_event_embed is not None:
                if raw_counts is None:
                    raw_counts = x
                events = (raw_counts > 0).to(dtype=x.dtype)
                h = h + float(temporal_identity_scale) * (events @ self.neuron_event_embed)
            neuron_state = x.transpose(1, 2) + self.neuron_pos_embed[:, :, :t_len]
            neuron_state = self.spatial_input_norm(neuron_state)
            for block in self.blocks:
                h, neuron_state = block(h, neuron_state)
            if self.spatial_mix_down is not None:
                mixed = self.spatial_mix_up(self.spatial_mix_down(neuron_state))
                h = h + float(spatial_mix_scale) * self.spatial_mix_norm(
                    self.spatial_mix_to_temporal(mixed.transpose(1, 2))
                )
            return h

        def forward(self, x, raw_counts=None):
            h = self.encode(x, raw_counts)
            heldin_logits = self.heldin_head(h) + self.heldin_rate_bias[None, None, :]
            heldout_logits = self.heldout_head(h) + self.heldout_rate_bias[None, None, :]
            if self.neuron_readout_proj is not None:
                readout = self.neuron_readout_proj(h)
                heldin_logits = heldin_logits + float(neuron_readout_scale) * torch.einsum(
                    "btd,nd->btn", readout, self.heldin_neuron_embed
                )
                heldout_logits = heldout_logits + float(neuron_readout_scale) * torch.einsum(
                    "btd,od->bto", readout, self.heldout_neuron_embed
                )
            if self.heldout_calib_scale is not None and self.heldout_calib_bias is not None:
                heldout_logits = (
                    heldout_logits * self.heldout_calib_scale[None, None, :]
                    + self.heldout_calib_bias[None, None, :]
                )
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
    mask_mode: str = "bernoulli",
    span_length: int = 0,
    heldin_loss_weight: float,
    validation_fraction: float,
    input_transform: str,
    seed: int,
    lr_schedule: str = "constant",
    contrast_loss_weight: float = 0.0,
    contrast_mask_prob: float = 0.1,
    contrast_temperature: float = 0.07,
    temporal_identity_scale: float = 0.0,
    spike_loss_weight: float = 0.0,
    spatial_mix_rank: int = 0,
    spatial_mix_scale: float = 0.0,
    neuron_readout_dim: int = 0,
    neuron_readout_scale: float = 0.0,
    use_mask_token: bool = False,
    unit_calibration: bool = False,
    unit_calibration_scale_reg: float = 1.0,
    unit_calibration_bias_reg: float = 1.0,
    warmup_epochs: int = 0,
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
    mask_mode = str(mask_mode).lower()
    if mask_mode not in {"bernoulli", "block_time"}:
        raise ValueError("`mask_mode` must be one of {'bernoulli', 'block_time'}")
    span_length = int(span_length)
    if mask_mode == "block_time" and span_length < 1:
        raise ValueError("`span_length` must be at least 1 when `mask_mode` is 'block_time'")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("`validation_fraction` must be in [0, 1)")
    if not 0.0 <= contrast_mask_prob < 1.0:
        raise ValueError("`contrast_mask_prob` must be in [0, 1)")
    contrast_loss_weight = float(contrast_loss_weight)
    if contrast_loss_weight < 0:
        raise ValueError("`contrast_loss_weight` must be non-negative")
    temporal_identity_scale = float(temporal_identity_scale)
    if temporal_identity_scale < 0:
        raise ValueError("`temporal_identity_scale` must be non-negative")
    spike_loss_weight = float(spike_loss_weight)
    if spike_loss_weight < 0:
        raise ValueError("`spike_loss_weight` must be non-negative")
    spatial_mix_rank = int(spatial_mix_rank)
    if spatial_mix_rank < 0:
        raise ValueError("`spatial_mix_rank` must be non-negative")
    spatial_mix_scale = float(spatial_mix_scale)
    if spatial_mix_scale < 0:
        raise ValueError("`spatial_mix_scale` must be non-negative")
    neuron_readout_dim = int(neuron_readout_dim)
    if neuron_readout_dim < 0:
        raise ValueError("`neuron_readout_dim` must be non-negative")
    neuron_readout_scale = float(neuron_readout_scale)
    if neuron_readout_scale < 0:
        raise ValueError("`neuron_readout_scale` must be non-negative")
    ensemble_size = int(ensemble_size)
    if ensemble_size < 1:
        raise ValueError("`ensemble_size` must be at least 1")
    lr_schedule = str(lr_schedule).lower()
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("`lr_schedule` must be one of {'constant', 'cosine'}")
    use_mask_token = bool(use_mask_token)
    unit_calibration = bool(unit_calibration)
    unit_calibration_scale_reg = float(unit_calibration_scale_reg)
    unit_calibration_bias_reg = float(unit_calibration_bias_reg)
    if unit_calibration_scale_reg < 0:
        raise ValueError("`unit_calibration_scale_reg` must be non-negative")
    if unit_calibration_bias_reg < 0:
        raise ValueError("`unit_calibration_bias_reg` must be non-negative")
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs < 0:
        raise ValueError("`warmup_epochs` must be non-negative")
    if warmup_epochs >= int(max_epochs):
        raise ValueError("`warmup_epochs` must be strictly less than `max_epochs`")

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
        temporal_identity_scale=temporal_identity_scale,
        spatial_mix_rank=spatial_mix_rank,
        spatial_mix_scale=spatial_mix_scale,
        neuron_readout_dim=neuron_readout_dim,
        neuron_readout_scale=neuron_readout_scale,
        dropout=float(dropout),
        min_rate=float(min_rate),
        use_mask_token=use_mask_token,
        unit_calibration=unit_calibration,
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
        sub_schedulers: list[Any] = []
        sub_milestones: list[int] = []
        if warmup_epochs > 0:
            sub_schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=int(warmup_epochs),
                )
            )
            sub_milestones.append(int(warmup_epochs))
        if lr_schedule == "cosine":
            cosine_epochs = max(1, int(max_epochs) - int(warmup_epochs))
            sub_schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_epochs,
                    eta_min=float(learning_rate) * 0.1,
                )
            )
        scheduler: Any
        if not sub_schedulers:
            scheduler = None
        elif len(sub_schedulers) == 1:
            scheduler = sub_schedulers[0]
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=sub_schedulers,
                milestones=sub_milestones,
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

        def pooled_features(xb, raw_counts) -> Any:
            encoded = model.encode(xb, raw_counts)
            return encoded.mean(dim=1)

        def batch_loss(batch_idx: np.ndarray, *, train_mode: bool) -> Any:
            idx_tensor = torch.as_tensor(batch_idx, dtype=torch.long, device=device_obj)
            xb = x_train.index_select(0, idx_tensor)
            target_hi = y_hi.index_select(0, idx_tensor)
            target_ho = y_ho.index_select(0, idx_tensor)
            if train_mode and mask_prob > 0:
                if mask_mode == "bernoulli":
                    mask = torch.rand_like(xb) < float(mask_prob)
                else:
                    batch_dim, t_len, n_neurons = xb.shape
                    mask = sample_block_time_mask(
                        batch_dim,
                        t_len,
                        n_neurons,
                        float(mask_prob),
                        span_length,
                        device=xb.device,
                    )
                xb_masked = model.apply_input_mask(xb, mask)
                raw_masked = target_hi.masked_fill(mask, 0.0)
            else:
                mask = torch.ones_like(xb, dtype=torch.bool)
                xb_masked = xb
                raw_masked = target_hi

            pred_hi, pred_ho, _features = model(xb_masked, raw_masked)
            loss = _poisson_loss(functional, pred_ho, target_ho)
            if heldin_loss_weight > 0:
                if bool(mask.any()):
                    loss_hi = _poisson_loss(functional, pred_hi[mask], target_hi[mask])
                else:
                    loss_hi = _poisson_loss(functional, pred_hi, target_hi)
                loss = loss + float(heldin_loss_weight) * loss_hi
            if train_mode and spike_loss_weight > 0 and bool(mask.any()):
                loss_spike = _poisson_loss(functional, pred_hi[mask], target_hi[mask])
                loss = loss + float(spike_loss_weight) * loss_spike

            if train_mode and contrast_loss_weight > 0:
                mask_a = torch.rand_like(xb) < float(contrast_mask_prob)
                mask_b = torch.rand_like(xb) < float(contrast_mask_prob)
                z1 = pooled_features(
                    model.apply_input_mask(xb, mask_a),
                    target_hi.masked_fill(mask_a, 0.0),
                )
                z2 = pooled_features(
                    model.apply_input_mask(xb, mask_b),
                    target_hi.masked_fill(mask_b, 0.0),
                )
                loss = loss + contrast_loss_weight * _info_nce_loss(
                    torch, functional, z1, z2, float(contrast_temperature)
                )
            if (
                train_mode
                and unit_calibration
                and model.heldout_calib_scale is not None
                and model.heldout_calib_bias is not None
            ):
                scale_err = model.heldout_calib_scale - 1.0
                loss = loss + float(unit_calibration_scale_reg) * torch.sum(scale_err**2)
                loss = loss + float(unit_calibration_bias_reg) * torch.sum(
                    model.heldout_calib_bias**2
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

        def predict_rates(x_np: np.ndarray, raw_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=device_obj)
            raw_tensor = torch.as_tensor(raw_np, dtype=torch.float32, device=device_obj)
            heldin_batches: list[np.ndarray] = []
            heldout_batches: list[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for start in range(0, x_tensor.shape[0], batch_size):
                    xb = x_tensor[start : start + batch_size]
                    raw_b = raw_tensor[start : start + batch_size]
                    pred_hi, pred_ho, _features = model(xb, raw_b)
                    heldin_batches.append(pred_hi.detach().cpu().numpy().astype(np.float32))
                    heldout_batches.append(pred_ho.detach().cpu().numpy().astype(np.float32))
            return np.concatenate(heldin_batches, axis=0), np.concatenate(heldout_batches, axis=0)

        train_rates_heldin, train_rates_heldout = predict_rates(train_x, train_hi)
        eval_rates_heldin, eval_rates_heldout = predict_rates(eval_x, eval_hi)
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

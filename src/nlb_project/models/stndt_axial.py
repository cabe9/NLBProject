"""Axial STNDT-style spatiotemporal transformer.

This model is a bounded next step after ``stndt_lite``. It uses explicit
held-in and held-out neuron identity embeddings, applies latent cross-attention
over neurons at each time bin, applies temporal attention to the resulting
population state, and trains masked held-in reconstruction with a learnable
mask token.

It is still intentionally compact: the pipeline can validate it on train/val
before any public-test scoring is considered.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ndt_lite import _poisson_loss, _require_torch, _reshape_transform, _resolve_device
from .stndt_lite import _info_nce_loss


def _axial_spatiotemporal_transformer_cls(
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
    n_spatial_latents: int,
    dropout: float,
    min_rate: float,
) -> type:
    """Return ``nn.Module`` subclass; defined after Torch import so it stays optional."""

    class AxialSpatioTemporalBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial_latents = nn.Parameter(torch.empty(1, 1, n_spatial_latents, d_model))
            self.neuron_key_norm = nn.LayerNorm(d_model)
            self.latent_query_norm = nn.LayerNorm(d_model)
            self.neuron_to_latent_attn = nn.MultiheadAttention(
                d_model,
                spatial_n_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.latent_norm = nn.LayerNorm(d_model)
            self.latent_ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(4 * d_model, d_model),
            )
            self.token_query_norm = nn.LayerNorm(d_model)
            self.latent_key_norm = nn.LayerNorm(d_model)
            self.latent_to_neuron_attn = nn.MultiheadAttention(
                d_model,
                spatial_n_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.spatial_token_norm = nn.LayerNorm(d_model)
            self.spatial_ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(4 * d_model, d_model),
            )
            self.temporal_fusion = nn.LayerNorm(d_model)
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_attn = nn.MultiheadAttention(
                d_model,
                n_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.temporal_norm2 = nn.LayerNorm(d_model)
            self.temporal_ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(4 * d_model, d_model),
            )
            self.token_temporal_norm = nn.LayerNorm(d_model)
            self.latent_temporal_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(float(dropout))
            nn.init.normal_(self.spatial_latents, mean=0.0, std=0.02)

        def forward(self, tokens, temporal_state):
            batch, t_len, n_neurons, d_model_local = tokens.shape
            n_latents = self.spatial_latents.shape[2]

            latent_queries = self.spatial_latents.expand(batch, t_len, n_latents, d_model_local)
            latent_queries = latent_queries + temporal_state[:, :, None, :]
            latent_flat = self.latent_query_norm(latent_queries).reshape(
                batch * t_len, n_latents, d_model_local
            )
            token_flat = self.neuron_key_norm(tokens).reshape(
                batch * t_len, n_neurons, d_model_local
            )
            latent_out, _ = self.neuron_to_latent_attn(
                latent_flat,
                token_flat,
                token_flat,
                need_weights=False,
            )
            latents = latent_flat + self.dropout(latent_out)
            latents = latents.reshape(batch, t_len, n_latents, d_model_local)
            latents = latents + self.dropout(self.latent_ff(self.latent_norm(latents)))

            token_queries = self.token_query_norm(tokens).reshape(
                batch * t_len, n_neurons, d_model_local
            )
            latent_key_value = self.latent_key_norm(latents).reshape(
                batch * t_len, n_latents, d_model_local
            )
            token_out, _ = self.latent_to_neuron_attn(
                token_queries,
                latent_key_value,
                latent_key_value,
                need_weights=False,
            )
            token_out = token_out.reshape(batch, t_len, n_neurons, d_model_local)
            tokens = tokens + self.dropout(token_out)
            tokens = tokens + self.dropout(self.spatial_ff(self.spatial_token_norm(tokens)))

            population_state = latents.mean(dim=2)
            temporal_state = self.temporal_fusion(temporal_state + population_state)

            temporal_input = self.temporal_norm1(temporal_state)
            temporal_out, _ = self.temporal_attn(
                temporal_input,
                temporal_input,
                temporal_input,
                need_weights=False,
            )
            temporal_state = temporal_state + self.dropout(temporal_out)
            temporal_state = temporal_state + self.dropout(
                self.temporal_ff(self.temporal_norm2(temporal_state))
            )

            tokens = self.token_temporal_norm(tokens + self.dropout(temporal_state[:, :, None, :]))
            latents = self.latent_temporal_norm(
                latents + self.dropout(temporal_state[:, :, None, :])
            )
            return tokens, temporal_state, latents

    class AxialSpatioTemporalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.count_proj = nn.Linear(1, d_model)
            self.time_embed = nn.Parameter(torch.zeros(1, max_t_len, 1, d_model))
            self.heldin_neuron_embed = nn.Parameter(torch.empty(1, 1, n_heldin, d_model))
            self.heldout_neuron_embed = nn.Parameter(torch.empty(1, 1, n_heldout, d_model))
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
            self.input_norm = nn.LayerNorm(d_model)
            self.initial_pool = nn.Linear(d_model, 1)
            self.blocks = nn.ModuleList(AxialSpatioTemporalBlock() for _ in range(n_layers))

            self.heldout_context_norm = nn.LayerNorm(d_model)
            self.heldout_key_norm = nn.LayerNorm(d_model)
            self.heldout_cross_attn = nn.MultiheadAttention(
                d_model,
                spatial_n_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.heldout_norm = nn.LayerNorm(d_model)
            self.heldout_ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(4 * d_model, d_model),
            )

            self.heldin_head = nn.Linear(d_model, 1)
            self.heldout_head = nn.Linear(d_model, 1)
            self.heldin_rate_bias = nn.Parameter(torch.zeros(n_heldin))
            self.heldout_rate_bias = nn.Parameter(torch.zeros(n_heldout))
            self.dropout = nn.Dropout(float(dropout))
            nn.init.normal_(self.heldin_neuron_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.heldout_neuron_embed, mean=0.0, std=0.02)

        def encode(self, x, mask=None):
            batch, t_len, n_neurons = x.shape
            tokens = self.count_proj(x.unsqueeze(-1))
            tokens = tokens + self.time_embed[:, :t_len, :, :] + self.heldin_neuron_embed
            if mask is not None:
                tokens = tokens + mask.to(dtype=tokens.dtype).unsqueeze(-1) * self.mask_token
            tokens = self.input_norm(tokens)

            pool_logits = self.initial_pool(tokens).squeeze(-1)
            pool_weights = functional.softmax(pool_logits, dim=-1)
            temporal_state = torch.sum(pool_weights.unsqueeze(-1) * tokens, dim=2)
            spatial_latents = None
            for block in self.blocks:
                tokens, temporal_state, spatial_latents = block(tokens, temporal_state)
            if spatial_latents is None:
                spatial_latents = tokens
            return tokens, temporal_state, spatial_latents

        def decode_heldout(self, spatial_latents, temporal_state):
            batch, t_len, n_latents, d_model_local = spatial_latents.shape
            query = (
                self.heldout_neuron_embed + self.heldout_context_norm(temporal_state)[:, :, None, :]
            )
            n_heldout_local = query.shape[2]
            query_flat = query.reshape(batch * t_len, n_heldout_local, d_model_local)
            key_value = self.heldout_key_norm(spatial_latents).reshape(
                batch * t_len, n_latents, d_model_local
            )
            decoded, _ = self.heldout_cross_attn(
                query_flat,
                key_value,
                key_value,
                need_weights=False,
            )
            decoded = query_flat + self.dropout(decoded)
            decoded = decoded.reshape(batch, t_len, n_heldout_local, d_model_local)
            decoded = decoded + self.dropout(self.heldout_ff(self.heldout_norm(decoded)))
            return decoded

        def forward(self, x, mask=None):
            tokens, temporal_state, spatial_latents = self.encode(x, mask=mask)
            heldin_logits = self.heldin_head(tokens).squeeze(-1)
            heldin_logits = heldin_logits + self.heldin_rate_bias[None, None, :]

            heldout_tokens = self.decode_heldout(spatial_latents, temporal_state)
            heldout_logits = self.heldout_head(heldout_tokens).squeeze(-1)
            heldout_logits = heldout_logits + self.heldout_rate_bias[None, None, :]

            heldin = functional.softplus(heldin_logits) + float(min_rate)
            heldout = functional.softplus(heldout_logits) + float(min_rate)
            return heldin, heldout, temporal_state

    return AxialSpatioTemporalTransformer


def fit_predict_stndt_axial(
    train_spikes_heldin: np.ndarray,
    train_spikes_heldout: np.ndarray,
    eval_spikes_heldin: np.ndarray,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    spatial_n_heads: int,
    n_spatial_latents: int,
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
    """Fit an axial spatiotemporal transformer and return NLB-compatible rates."""
    torch, nn, functional = _require_torch()

    if d_model % n_heads != 0:
        raise ValueError("`d_model` must be divisible by `n_heads`")
    if d_model % spatial_n_heads != 0:
        raise ValueError("`d_model` must be divisible by `spatial_n_heads`")
    if int(n_layers) < 1:
        raise ValueError("`n_layers` must be at least 1")
    if int(n_spatial_latents) < 1:
        raise ValueError("`n_spatial_latents` must be at least 1")
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
    batch_size = max(1, int(batch_size))
    device_obj = _resolve_device(torch, device)

    AxialSpatioTemporalTransformer = _axial_spatiotemporal_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=n_heldin,
        n_heldout=n_heldout,
        d_model=d_model,
        max_t_len=max_t_len,
        n_layers=n_layers,
        n_heads=n_heads,
        spatial_n_heads=spatial_n_heads,
        n_spatial_latents=int(n_spatial_latents),
        dropout=float(dropout),
        min_rate=float(min_rate),
    )

    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device_obj)
    y_hi = torch.as_tensor(train_hi, dtype=torch.float32, device=device_obj)
    y_ho = torch.as_tensor(train_ho, dtype=torch.float32, device=device_obj)

    def fit_one(seed_value: int) -> dict[str, np.ndarray]:
        torch.manual_seed(int(seed_value))
        rng = np.random.default_rng(int(seed_value))
        model = AxialSpatioTemporalTransformer().to(device_obj)
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

        def pooled_features(xb, mask) -> Any:
            _tokens, temporal_state, _spatial_latents = model.encode(xb, mask=mask)
            return temporal_state.mean(dim=1)

        def batch_loss(batch_idx: np.ndarray, *, train_mode: bool) -> Any:
            idx_tensor = torch.as_tensor(batch_idx, dtype=torch.long, device=device_obj)
            xb = x_train.index_select(0, idx_tensor)
            target_hi = y_hi.index_select(0, idx_tensor)
            target_ho = y_ho.index_select(0, idx_tensor)
            if train_mode and mask_prob > 0:
                mask = torch.rand_like(xb) < float(mask_prob)
                xb_model = xb.masked_fill(mask, 0.0)
            else:
                mask = torch.zeros_like(xb, dtype=torch.bool)
                xb_model = xb

            pred_hi, pred_ho, _features = model(xb_model, mask=mask)
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
                z1 = pooled_features(xb.masked_fill(mask_a, 0.0), mask_a)
                z2 = pooled_features(xb.masked_fill(mask_b, 0.0), mask_b)
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
                    pred_hi, pred_ho, _features = model(xb, mask=None)
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

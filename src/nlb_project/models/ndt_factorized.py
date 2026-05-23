"""Factorized Neural Data Transformer-style baseline.

This model is the first neuron-aware step beyond ``ndt_lite``. It keeps the
same supervised NLB interface, but represents the population as neuron tokens
that are compressed into a small set of learned latent tokens at each time bin.
Those latents are modeled over time and decoded back to held-in and held-out
neuron rates.

PyTorch is an optional dependency. Install with ``pip install -e .[neural]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ndt_lite import _poisson_loss, _require_torch, _reshape_transform, _resolve_device


def _factorized_transformer_cls(
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
    n_latents: int,
    dropout: float,
    min_rate: float,
) -> type:
    """Return ``nn.Module`` subclass; defined after Torch import so it stays optional."""

    class FactorizedLatentBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.time_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.latent_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        def forward(self, latents):
            batch_size, t_len, n_latent_tokens, width = latents.shape
            temporal = (
                latents.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(batch_size * n_latent_tokens, t_len, width)
            )
            temporal = self.time_layer(temporal)
            latents = (
                temporal.reshape(batch_size, n_latent_tokens, t_len, width)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            population = latents.reshape(batch_size * t_len, n_latent_tokens, width)
            population = self.latent_layer(population)
            return population.reshape(batch_size, t_len, n_latent_tokens, width)

    class FactorizedTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.count_proj = nn.Linear(1, d_model)
            self.population_proj = nn.Linear(n_heldin, d_model)
            self.heldin_embed = nn.Parameter(torch.empty(n_heldin, d_model))
            self.heldout_embed = nn.Parameter(torch.empty(n_heldout, d_model))
            self.latent_query = nn.Parameter(torch.empty(n_latents, d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, max_t_len, 1, d_model))
            self.input_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=float(dropout), batch_first=True
            )
            self.input_norm = nn.LayerNorm(d_model)
            self.blocks = nn.ModuleList(FactorizedLatentBlock() for _ in range(n_layers))
            self.heldin_decoder = nn.MultiheadAttention(
                d_model, n_heads, dropout=float(dropout), batch_first=True
            )
            self.heldout_decoder = nn.MultiheadAttention(
                d_model, n_heads, dropout=float(dropout), batch_first=True
            )
            self.heldin_norm = nn.LayerNorm(d_model)
            self.heldout_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(float(dropout))
            self.factorized_rate_head = nn.Linear(d_model, 1)
            self.heldin_global_head = nn.Linear(d_model, n_heldin)
            self.heldout_global_head = nn.Linear(d_model, n_heldout)
            self.heldin_rate_bias = nn.Parameter(torch.zeros(n_heldin))
            self.heldout_rate_bias = nn.Parameter(torch.zeros(n_heldout))

            nn.init.normal_(self.heldin_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.heldout_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.latent_query, mean=0.0, std=0.02)
            nn.init.zeros_(self.heldin_global_head.bias)
            nn.init.zeros_(self.heldout_global_head.bias)

        def encode(self, x):
            batch_size, t_len, n_neurons = x.shape
            neuron_tokens = self.count_proj(x.unsqueeze(-1))
            neuron_tokens = neuron_tokens + self.heldin_embed[None, None, :, :]
            neuron_tokens = neuron_tokens.reshape(batch_size * t_len, n_neurons, d_model)

            queries = self.latent_query[None, :, :].expand(batch_size * t_len, -1, -1)
            latents, _weights = self.input_attn(
                queries, neuron_tokens, neuron_tokens, need_weights=False
            )
            latents = self.input_norm(queries + self.dropout(latents))
            latents = latents.reshape(batch_size, t_len, n_latents, d_model)
            population_token = self.population_proj(x).unsqueeze(2)
            latents = torch.cat([population_token, latents], dim=2)
            latents = latents + self.pos_embed[:, :t_len, :, :]
            for block in self.blocks:
                latents = block(latents)
            return latents

        def decode_factorized_logits(self, latents, neuron_embed, decoder, norm):
            batch_size, t_len, _n_latent_tokens, width = latents.shape
            n_neurons = neuron_embed.shape[0]
            latent_tokens = latents.reshape(batch_size * t_len, _n_latent_tokens, width)
            queries = neuron_embed[None, :, :].expand(batch_size * t_len, -1, -1)
            decoded, _weights = decoder(queries, latent_tokens, latent_tokens, need_weights=False)
            decoded = norm(queries + self.dropout(decoded))
            logits = self.factorized_rate_head(decoded).squeeze(-1)
            return logits.reshape(batch_size, t_len, n_neurons)

        def forward(self, x):
            latents = self.encode(x)
            global_state = latents[:, :, 0, :]
            heldin_logits = self.heldin_global_head(global_state)
            heldin_logits = heldin_logits + self.decode_factorized_logits(
                latents,
                self.heldin_embed,
                self.heldin_decoder,
                self.heldin_norm,
            )
            heldin_logits = heldin_logits + self.heldin_rate_bias[None, None, :]
            heldout_logits = self.heldout_global_head(global_state)
            heldout_logits = heldout_logits + self.decode_factorized_logits(
                latents,
                self.heldout_embed,
                self.heldout_decoder,
                self.heldout_norm,
            )
            heldout_logits = heldout_logits + self.heldout_rate_bias[None, None, :]
            heldin = functional.softplus(heldin_logits) + float(min_rate)
            heldout = functional.softplus(heldout_logits) + float(min_rate)
            return heldin, heldout

    return FactorizedTransformer


def fit_predict_ndt_factorized(
    train_spikes_heldin: np.ndarray,
    train_spikes_heldout: np.ndarray,
    eval_spikes_heldin: np.ndarray,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_latents: int,
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
    """Fit a factorized neuron/time transformer and return rate predictions."""
    torch, nn, functional = _require_torch()

    if d_model % n_heads != 0:
        raise ValueError("`d_model` must be divisible by `n_heads`")
    n_latents = int(n_latents)
    if n_latents < 1:
        raise ValueError("`n_latents` must be at least 1")
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

    FactorizedTransformer = _factorized_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=n_heldin,
        n_heldout=n_heldout,
        d_model=d_model,
        max_t_len=max_t_len,
        n_layers=n_layers,
        n_heads=n_heads,
        n_latents=n_latents,
        dropout=float(dropout),
        min_rate=float(min_rate),
    )

    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device_obj)
    y_hi = torch.as_tensor(train_hi, dtype=torch.float32, device=device_obj)
    y_ho = torch.as_tensor(train_ho, dtype=torch.float32, device=device_obj)

    def fit_one(seed_value: int) -> dict[str, np.ndarray]:
        torch.manual_seed(int(seed_value))
        rng = np.random.default_rng(int(seed_value))
        model = FactorizedTransformer().to(device_obj)
        with torch.no_grad():
            min_rate_tensor = torch.tensor(float(min_rate), dtype=torch.float32, device=device_obj)
            heldin_mean = torch.clamp(y_hi.mean(dim=(0, 1)), min=min_rate_tensor)
            heldout_mean = torch.clamp(y_ho.mean(dim=(0, 1)), min=min_rate_tensor)
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

"""Declarative registry of model families wired into the pipeline.

Every model family is described by a :class:`ModelSpec` that names:

- the ``predict`` function used for inference,
- the scalar parameters each model expects in ``cfg.baseline``,
- the CV sweep axes iterated in ``cfg.improvement``,
- any scalar keys that live in ``cfg.improvement`` with a fallback to
  ``cfg.baseline`` (e.g. ``input_transform``, or ``ridge_alpha`` on the LDS
  path which is a scalar rather than a grid),
- whether the model takes a pluggable rate readout (``output_head`` +
  ``log_offset``), and
- whether the model consumes ``cfg.log_offset`` directly (smoothing).

The registry is the single source of truth for how each model participates
in the pipeline. ``pipeline.py`` contains no per-model branches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import (
    fit_predict_lagged_pca_latent_regression,
    fit_predict_lagged_reduced_rank_regression,
    fit_predict_lagged_ridge_direct,
    fit_predict_lds_pca_latent_regression,
    fit_predict_ndt_factorized,
    fit_predict_ndt_lite,
    fit_predict_pca_latent_regression,
    fit_predict_ridge_direct,
    fit_predict_stndt_axial,
    fit_predict_stndt_lite,
)
from .smoothing import SmoothingParams, predict_rates

Caster = Callable[[Any], Any]


@dataclass(frozen=True)
class SweepAxis:
    """One CV axis.

    Reads ``grid_key`` from ``cfg.improvement``, and binds each value to
    ``param_name`` (after applying ``caster``). Axes are iterated by
    :func:`itertools.product` in the order listed on the spec.
    """

    grid_key: str
    param_name: str
    caster: Caster


@dataclass(frozen=True)
class ModelSpec:
    """Declarative spec for one model family's pipeline hookup."""

    name: str
    predict: Callable[..., dict]
    baseline_params: tuple[tuple[str, Caster], ...]
    sweep_axes: tuple[SweepAxis, ...]
    optional_sweep_axes: tuple[SweepAxis, ...] = ()
    improvement_overrides: tuple[tuple[str, Caster], ...] = ()
    uses_rate_head: bool = True
    passes_log_offset: bool = False
    default_cv_folds: int = 3
    # Extra kwargs pulled from ExperimentConfig and passed to ``predict``,
    # but not serialized in the ``params`` dict written to metrics.csv.
    extra_predict_kwargs_fn: Callable[[Any], dict[str, Any]] = field(default=lambda cfg: {})


def _smoothing_predict(
    train_rates_heldin,
    train_rates_heldout,
    eval_rates_heldin,
    *,
    kern_sd_ms: float,
    alpha: float,
    log_offset: float,
    bin_size_ms: int,
) -> dict:
    """Adapter so smoothing matches the ``(tensors..., **params)`` signature."""
    return predict_rates(
        train_rates_heldin,
        train_rates_heldout,
        eval_rates_heldin,
        SmoothingParams(kern_sd_ms=kern_sd_ms, alpha=alpha, log_offset=log_offset),
        bin_size_ms,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "smoothing": ModelSpec(
        name="smoothing",
        predict=_smoothing_predict,
        baseline_params=(("kern_sd_ms", float), ("alpha", float)),
        sweep_axes=(
            SweepAxis("kern_sd_grid", "kern_sd_ms", float),
            SweepAxis("alpha_grid", "alpha", float),
        ),
        uses_rate_head=False,
        passes_log_offset=True,
        extra_predict_kwargs_fn=lambda cfg: {"bin_size_ms": int(cfg.bin_size_ms)},
    ),
    "pca_latent_regression": ModelSpec(
        name="pca_latent_regression",
        predict=fit_predict_pca_latent_regression,
        baseline_params=(("n_components", int), ("ridge_alpha", float)),
        sweep_axes=(
            SweepAxis("n_components_grid", "n_components", int),
            SweepAxis("ridge_alpha_grid", "ridge_alpha", float),
        ),
    ),
    "ridge_direct": ModelSpec(
        name="ridge_direct",
        predict=fit_predict_ridge_direct,
        baseline_params=(("ridge_alpha", float),),
        sweep_axes=(SweepAxis("ridge_alpha_grid", "ridge_alpha", float),),
    ),
    "lagged_ridge_direct": ModelSpec(
        name="lagged_ridge_direct",
        predict=fit_predict_lagged_ridge_direct,
        baseline_params=(
            ("history_bins", int),
            ("ridge_alpha", float),
            ("input_transform", str),
        ),
        sweep_axes=(
            SweepAxis("history_bins_grid", "history_bins", int),
            SweepAxis("ridge_alpha_grid", "ridge_alpha", float),
        ),
        improvement_overrides=(("input_transform", str),),
    ),
    "lagged_pca_latent_regression": ModelSpec(
        name="lagged_pca_latent_regression",
        predict=fit_predict_lagged_pca_latent_regression,
        baseline_params=(
            ("history_bins", int),
            ("n_components", int),
            ("ridge_alpha", float),
            ("input_transform", str),
        ),
        sweep_axes=(
            SweepAxis("history_bins_grid", "history_bins", int),
            SweepAxis("n_components_grid", "n_components", int),
            SweepAxis("ridge_alpha_grid", "ridge_alpha", float),
        ),
        improvement_overrides=(("input_transform", str),),
    ),
    "lagged_reduced_rank_regression": ModelSpec(
        name="lagged_reduced_rank_regression",
        predict=fit_predict_lagged_reduced_rank_regression,
        baseline_params=(
            ("history_bins", int),
            ("rank", int),
            ("ridge_alpha", float),
            ("input_transform", str),
        ),
        sweep_axes=(
            SweepAxis("history_bins_grid", "history_bins", int),
            SweepAxis("rank_grid", "rank", int),
            SweepAxis("ridge_alpha_grid", "ridge_alpha", float),
        ),
        improvement_overrides=(("input_transform", str),),
    ),
    "lds_pca_latent_regression": ModelSpec(
        name="lds_pca_latent_regression",
        predict=fit_predict_lds_pca_latent_regression,
        baseline_params=(
            ("n_components", int),
            ("ridge_alpha", float),
            ("input_transform", str),
            ("obs_noise_scale", float),
        ),
        sweep_axes=(SweepAxis("n_components_grid", "n_components", int),),
        improvement_overrides=(
            ("ridge_alpha", float),
            ("input_transform", str),
            ("obs_noise_scale", float),
        ),
        default_cv_folds=2,
    ),
    "ndt_lite": ModelSpec(
        name="ndt_lite",
        predict=fit_predict_ndt_lite,
        baseline_params=(
            ("d_model", int),
            ("n_layers", int),
            ("n_heads", int),
            ("dropout", float),
            ("learning_rate", float),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("mask_prob", float),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("lr_schedule", str),
            ("neuron_embedding_scale", float),
            ("ensemble_size", int),
            ("device", str),
        ),
        sweep_axes=(
            SweepAxis("d_model_grid", "d_model", int),
            SweepAxis("n_layers_grid", "n_layers", int),
            SweepAxis("dropout_grid", "dropout", float),
            SweepAxis("learning_rate_grid", "learning_rate", float),
            SweepAxis("mask_prob_grid", "mask_prob", float),
            SweepAxis("lr_schedule_grid", "lr_schedule", str),
            SweepAxis("neuron_embedding_scale_grid", "neuron_embedding_scale", float),
        ),
        optional_sweep_axes=(SweepAxis("ensemble_size_grid", "ensemble_size", int),),
        improvement_overrides=(
            ("n_heads", int),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("ensemble_size", int),
            ("device", str),
        ),
        uses_rate_head=False,
        default_cv_folds=2,
    ),
    "ndt_factorized": ModelSpec(
        name="ndt_factorized",
        predict=fit_predict_ndt_factorized,
        baseline_params=(
            ("d_model", int),
            ("n_layers", int),
            ("n_heads", int),
            ("n_latents", int),
            ("dropout", float),
            ("learning_rate", float),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("mask_prob", float),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("lr_schedule", str),
            ("ensemble_size", int),
            ("device", str),
        ),
        sweep_axes=(
            SweepAxis("d_model_grid", "d_model", int),
            SweepAxis("n_layers_grid", "n_layers", int),
            SweepAxis("n_latents_grid", "n_latents", int),
            SweepAxis("dropout_grid", "dropout", float),
            SweepAxis("learning_rate_grid", "learning_rate", float),
            SweepAxis("mask_prob_grid", "mask_prob", float),
            SweepAxis("lr_schedule_grid", "lr_schedule", str),
        ),
        improvement_overrides=(
            ("n_heads", int),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("ensemble_size", int),
            ("device", str),
        ),
        uses_rate_head=False,
        default_cv_folds=2,
    ),
    "stndt_lite": ModelSpec(
        name="stndt_lite",
        predict=fit_predict_stndt_lite,
        baseline_params=(
            ("d_model", int),
            ("n_layers", int),
            ("n_heads", int),
            ("spatial_n_heads", int),
            ("dropout", float),
            ("learning_rate", float),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("mask_prob", float),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("lr_schedule", str),
            ("contrast_loss_weight", float),
            ("contrast_mask_prob", float),
            ("contrast_temperature", float),
            ("ensemble_size", int),
            ("device", str),
        ),
        sweep_axes=(
            SweepAxis("d_model_grid", "d_model", int),
            SweepAxis("n_layers_grid", "n_layers", int),
            SweepAxis("dropout_grid", "dropout", float),
            SweepAxis("learning_rate_grid", "learning_rate", float),
            SweepAxis("mask_prob_grid", "mask_prob", float),
            SweepAxis("lr_schedule_grid", "lr_schedule", str),
            SweepAxis("contrast_loss_weight_grid", "contrast_loss_weight", float),
        ),
        optional_sweep_axes=(
            SweepAxis("ensemble_size_grid", "ensemble_size", int),
            SweepAxis("temporal_identity_scale_grid", "temporal_identity_scale", float),
            SweepAxis("spike_loss_weight_grid", "spike_loss_weight", float),
            SweepAxis("spatial_mix_rank_grid", "spatial_mix_rank", int),
            SweepAxis("spatial_mix_scale_grid", "spatial_mix_scale", float),
            SweepAxis("neuron_readout_dim_grid", "neuron_readout_dim", int),
            SweepAxis("neuron_readout_scale_grid", "neuron_readout_scale", float),
            SweepAxis("use_mask_token_grid", "use_mask_token", bool),
            SweepAxis("warmup_epochs_grid", "warmup_epochs", int),
        ),
        improvement_overrides=(
            ("n_heads", int),
            ("spatial_n_heads", int),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("contrast_mask_prob", float),
            ("contrast_temperature", float),
            ("ensemble_size", int),
            ("device", str),
        ),
        uses_rate_head=False,
        default_cv_folds=2,
    ),
    "stndt_axial": ModelSpec(
        name="stndt_axial",
        predict=fit_predict_stndt_axial,
        baseline_params=(
            ("d_model", int),
            ("n_layers", int),
            ("n_heads", int),
            ("spatial_n_heads", int),
            ("n_spatial_latents", int),
            ("dropout", float),
            ("learning_rate", float),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("mask_prob", float),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("lr_schedule", str),
            ("contrast_loss_weight", float),
            ("contrast_mask_prob", float),
            ("contrast_temperature", float),
            ("ensemble_size", int),
            ("device", str),
        ),
        sweep_axes=(
            SweepAxis("d_model_grid", "d_model", int),
            SweepAxis("n_layers_grid", "n_layers", int),
            SweepAxis("n_spatial_latents_grid", "n_spatial_latents", int),
            SweepAxis("dropout_grid", "dropout", float),
            SweepAxis("learning_rate_grid", "learning_rate", float),
            SweepAxis("mask_prob_grid", "mask_prob", float),
            SweepAxis("lr_schedule_grid", "lr_schedule", str),
            SweepAxis("contrast_loss_weight_grid", "contrast_loss_weight", float),
        ),
        optional_sweep_axes=(SweepAxis("ensemble_size_grid", "ensemble_size", int),),
        improvement_overrides=(
            ("n_heads", int),
            ("spatial_n_heads", int),
            ("weight_decay", float),
            ("batch_size", int),
            ("max_epochs", int),
            ("patience", int),
            ("heldin_loss_weight", float),
            ("validation_fraction", float),
            ("input_transform", str),
            ("seed", int),
            ("contrast_mask_prob", float),
            ("contrast_temperature", float),
            ("ensemble_size", int),
            ("device", str),
        ),
        uses_rate_head=False,
        default_cv_folds=2,
    ),
}


def get_spec(model_type: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by ``model_type`` with a helpful error."""
    if model_type not in MODEL_REGISTRY:
        known = sorted(MODEL_REGISTRY)
        raise ValueError(f"Unsupported model_type `{model_type}`. Expected one of {known}.")
    return MODEL_REGISTRY[model_type]

"""Typed, validated experiment configuration.

``ExperimentConfig`` is the single entry point for turning a YAML file into
a fully-validated, pipeline-ready object. Validation is registry-driven:
the required keys in the ``baseline`` and ``improvement`` sections come
from the selected model's :class:`~nlb_project.model_registry.ModelSpec`,
so adding or removing a model parameter is a one-place change.

Invalid configs fail fast with a clear message at load time, before any
data is loaded or any model is fit. No silent default fallbacks for
model-specific keys: if the YAML is missing a required field, the loader
raises. The only values with documented repo-wide defaults are the
pluggable rate-readout (``output_head``) and the log-link offset
(``log_offset``), which fall through to the top-level ``ExperimentConfig``
fields as described in :func:`nlb_project.pipeline._rate_head_params`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .model_registry import MODEL_REGISTRY, ModelSpec, get_spec


_VALID_OUTPUT_HEADS = ("linear", "log_link", "poisson_glm")
DEFAULT_OUTPUT_HEAD = "log_link"
_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "dataset_name",
        "data_path",
        "data_prefix",
        "bin_size_ms",
        "train_split",
        "eval_split",
        "include_psth",
        "log_offset",
        "seed",
        "skip_fields",
        "baseline",
        "improvement",
        "output_dir",
    }
)
_ALLOWED_TOP_LEVEL_KEYS = _REQUIRED_TOP_LEVEL_KEYS | {"model_type", "output_head"}
_OPTIONAL_IMPROVEMENT_KEYS = frozenset({"cv_folds", "output_head", "log_offset"})


@dataclass
class ExperimentConfig:
    """Validated experiment configuration.

    ``baseline`` and ``improvement`` remain plain ``dict`` objects so the
    YAML round-trips cleanly and ``json.dumps(sort_keys=True)`` over the
    params dict produces byte-identical output across refactors. All
    validation happens at load time via :meth:`from_mapping`.
    """

    dataset_name: str
    data_path: str | None
    data_prefix: str
    bin_size_ms: int
    train_split: str
    eval_split: str
    include_psth: bool
    log_offset: float
    seed: int
    skip_fields: list[str]
    baseline: dict[str, Any]
    improvement: dict[str, Any]
    output_dir: str
    model_type: str = "smoothing"
    output_head: str = DEFAULT_OUTPUT_HEAD

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        _require_keys(raw, _REQUIRED_TOP_LEVEL_KEYS, context="top-level config")
        _reject_unknown_keys(raw, _ALLOWED_TOP_LEVEL_KEYS, context="top-level config")

        model_type = str(raw.get("model_type", "smoothing"))
        spec = get_spec(model_type)

        output_head = str(raw.get("output_head", DEFAULT_OUTPUT_HEAD))
        _validate_output_head(output_head, context="top-level config")

        baseline = _validate_baseline(raw["baseline"], spec)
        improvement = _validate_improvement(raw["improvement"], spec)

        return cls(
            model_type=model_type,
            dataset_name=str(raw["dataset_name"]),
            data_path=(None if raw["data_path"] is None else str(raw["data_path"])),
            data_prefix=str(raw["data_prefix"]),
            bin_size_ms=int(raw["bin_size_ms"]),
            train_split=str(raw["train_split"]),
            eval_split=str(raw["eval_split"]),
            include_psth=bool(raw["include_psth"]),
            log_offset=float(raw["log_offset"]),
            seed=int(raw["seed"]),
            skip_fields=[str(x) for x in raw["skip_fields"]],
            baseline=baseline,
            improvement=improvement,
            output_dir=str(raw["output_dir"]),
            output_head=output_head,
        )


# ---- helpers ---------------------------------------------------------------


def _require_keys(mapping: dict[str, Any], required: Iterable[str], *, context: str) -> None:
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ValueError(f"{context}: missing required keys {missing}")


def _reject_unknown_keys(
    mapping: dict[str, Any], allowed: Iterable[str], *, context: str
) -> None:
    extras = sorted(set(mapping) - set(allowed))
    if extras:
        raise ValueError(f"{context}: unknown keys {extras}")


def _validate_output_head(head: Any, *, context: str) -> None:
    head_lower = str(head).lower()
    if head_lower not in _VALID_OUTPUT_HEADS:
        raise ValueError(
            f"{context}: output_head must be one of {_VALID_OUTPUT_HEADS}, got {head!r}"
        )


def _validate_baseline(section: Any, spec: ModelSpec) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError(f"`baseline` section for model `{spec.name}` must be a mapping")

    required = {name for name, _ in spec.baseline_params}
    allowed = required | (({"output_head", "log_offset"}) if spec.uses_rate_head else set())

    _require_keys(section, required, context=f"baseline[{spec.name}]")
    _reject_unknown_keys(section, allowed, context=f"baseline[{spec.name}]")

    if "output_head" in section:
        _validate_output_head(section["output_head"], context=f"baseline[{spec.name}]")

    return dict(section)


def _validate_improvement(section: Any, spec: ModelSpec) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError(f"`improvement` section for model `{spec.name}` must be a mapping")

    required_grids = {axis.grid_key for axis in spec.sweep_axes}
    override_keys = {name for name, _ in spec.improvement_overrides}
    allowed = (
        required_grids
        | override_keys
        | _OPTIONAL_IMPROVEMENT_KEYS
    )

    _require_keys(section, required_grids, context=f"improvement[{spec.name}]")
    _reject_unknown_keys(section, allowed, context=f"improvement[{spec.name}]")

    for axis in spec.sweep_axes:
        grid = section[axis.grid_key]
        if not isinstance(grid, list) or len(grid) == 0:
            raise ValueError(
                f"improvement[{spec.name}].{axis.grid_key}: expected non-empty list, "
                f"got {grid!r}"
            )

    if "output_head" in section:
        _validate_output_head(section["output_head"], context=f"improvement[{spec.name}]")

    return dict(section)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} is not a YAML mapping")
    try:
        return ExperimentConfig.from_mapping(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid config {path}: {exc}") from exc


__all__ = ["ExperimentConfig", "load_config", "MODEL_REGISTRY"]

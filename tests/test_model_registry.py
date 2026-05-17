from __future__ import annotations

import inspect

from nlb_project.model_registry import get_spec
from nlb_project.models.ndt_factorized import fit_predict_ndt_factorized
from nlb_project.models.ndt_lite import fit_predict_ndt_lite
from nlb_project.models.stndt_axial import fit_predict_stndt_axial
from nlb_project.models.stndt_lite import fit_predict_stndt_lite


def _assert_registry_param_names_match_fit_predict_signature(model_type, predict) -> None:
    """Guardrail: pipeline passes registry ``params`` as kwargs to fit functions."""
    spec = get_spec(model_type)
    sig = inspect.signature(predict)
    names = set(sig.parameters)

    for param_name, _caster in spec.baseline_params:
        assert param_name in names, f"baseline_params key `{param_name}` missing from signature"

    for param_name, _caster in spec.improvement_overrides:
        assert param_name in names, (
            f"improvement_overrides key `{param_name}` missing from signature"
        )

    for axis in spec.sweep_axes:
        assert axis.param_name in names, f"sweep axis `{axis.param_name}` missing from signature"
    for axis in spec.optional_sweep_axes:
        assert axis.param_name in names, (
            f"optional sweep axis `{axis.param_name}` missing from signature"
        )


def test_ndt_lite_registry_param_names_match_fit_predict_signature() -> None:
    _assert_registry_param_names_match_fit_predict_signature("ndt_lite", fit_predict_ndt_lite)


def test_ndt_factorized_registry_param_names_match_fit_predict_signature() -> None:
    _assert_registry_param_names_match_fit_predict_signature(
        "ndt_factorized", fit_predict_ndt_factorized
    )


def test_stndt_lite_registry_param_names_match_fit_predict_signature() -> None:
    _assert_registry_param_names_match_fit_predict_signature("stndt_lite", fit_predict_stndt_lite)


def test_stndt_axial_registry_param_names_match_fit_predict_signature() -> None:
    _assert_registry_param_names_match_fit_predict_signature("stndt_axial", fit_predict_stndt_axial)

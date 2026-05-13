from __future__ import annotations

import inspect

from nlb_project.model_registry import get_spec
from nlb_project.models.ndt_lite import fit_predict_ndt_lite


def test_ndt_lite_registry_param_names_match_fit_predict_signature() -> None:
    """Guardrail: pipeline passes ``params`` as kwargs to ``fit_predict_ndt_lite``."""
    spec = get_spec("ndt_lite")
    sig = inspect.signature(fit_predict_ndt_lite)
    names = set(sig.parameters)

    for param_name, _caster in spec.baseline_params:
        assert param_name in names, f"baseline_params key `{param_name}` missing from signature"

    for param_name, _caster in spec.improvement_overrides:
        assert param_name in names, (
            f"improvement_overrides key `{param_name}` missing from signature"
        )

    for axis in spec.sweep_axes:
        assert axis.param_name in names, f"sweep axis `{axis.param_name}` missing from signature"

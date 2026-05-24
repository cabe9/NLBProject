from __future__ import annotations

import importlib


def test_installed_cli_modules_import() -> None:
    modules = [
        "nlb_project.cli.run_experiment",
        "nlb_project.cli.get_data",
        "nlb_project.cli.generate_portfolio_artifacts",
        "nlb_project.cli.verify_results",
        "nlb_project.cli.get_public_eval_data",
        "nlb_project.cli.evaluate_public_test",
        "nlb_project.cli.evaluate_ensemble_public_test",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert callable(module.main)

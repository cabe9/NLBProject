.PHONY: setup test run lock get-data portfolio-artifacts

setup:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

test:
	pytest

get-data:
	python -m scripts.get_data --dataset mc_maze --out data/raw

run:
	python -m scripts.run_experiment --config configs/mc_maze_lagged_pca.yaml

portfolio-artifacts:
	python -m scripts.generate_portfolio_artifacts

lock:
	pip-compile requirements/requirements.in -o requirements/requirements.lock

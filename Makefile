.PHONY: setup test run lock get-data

setup:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

test:
	pytest

get-data:
	python -m scripts.get_data --dataset mc_maze --out data/raw

run:
	python -m scripts.run_experiment --config configs/mc_maze_smoothing.yaml

lock:
	pip-compile requirements/requirements.in -o requirements/requirements.lock

.PHONY: setup test run lock get-data portfolio-artifacts verify-results lint format notebook typecheck

setup:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src scripts tests

format:
	ruff format .

notebook:
	jupyter nbconvert --to notebook --execute notebooks/results_walkthrough.ipynb \
		--output results_walkthrough.ipynb

get-data:
	nlb-get-data --dataset mc_maze --out data/raw

run:
	nlb-run-experiment --config configs/mc_maze_lagged_pca.yaml

portfolio-artifacts:
	nlb-generate-portfolio-artifacts

verify-results:
	nlb-verify-results

lock:
	pip-compile requirements/requirements.in -o requirements/requirements.lock

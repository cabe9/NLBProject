.PHONY: setup test run lock get-data public-eval-data public-test portfolio-artifacts verify-results lint format notebook typecheck

NLB_DATA_DIR ?= $(CURDIR)/data/raw
export NLB_DATA_DIR

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

public-eval-data:
	nlb-get-public-eval-data --out data/eval/eval_data_test.h5

run:
	nlb-run-experiment --config configs/mc_maze_lagged_pca.yaml

public-test:
	nlb-evaluate-public-test --config configs/mc_maze_lagged_pca.yaml \
		--eval-data data/eval/eval_data_test.h5

portfolio-artifacts:
	nlb-generate-portfolio-artifacts

verify-results:
	nlb-verify-results

lock:
	pip-compile requirements/requirements.in -o requirements/requirements.lock

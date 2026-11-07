# Commands avoid shell-specific syntax so they work under cmd.exe and sh.
PY = uv run --no-sync python

.PHONY: setup fetch surface train-pricer calibrate smooth forecast viz test lint

setup:
	uv venv --python 3.11
	uv pip install -r requirements.txt
	uv pip install -e .

fetch:
	$(PY) scripts/collect_snapshot.py

surface:
	$(PY) -m nvs.surface.build

train-pricer:
	$(PY) -m nvs.nn.pricer

calibrate:
	$(PY) -m nvs.calibrate

smooth:
	$(PY) -m nvs.nn.smoother

forecast:
	$(PY) -m nvs.nn.forecaster

viz:
	$(PY) -m nvs.viz

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff check .

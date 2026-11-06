PYTHON ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: install prepare train eda test lint api drift docker dvc

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

prepare:
	PYTHONPATH=src $(PYTHON) -m credit_default.data.prepare

train:
	PYTHONPATH=src $(PYTHON) -m credit_default.models.train

eda:
	PYTHONPATH=src $(PYTHON) -m credit_default.eda

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m flake8 src tests

api:
	PYTHONPATH=src $(PYTHON) -m uvicorn credit_default.api.main:app --host 127.0.0.1 --port 8000

drift:
	PYTHONPATH=src $(PYTHON) -m credit_default.monitoring.simulate

docker:
	bash scripts/build_and_run.sh

dvc:
	dvc repro

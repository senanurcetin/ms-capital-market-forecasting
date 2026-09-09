# MSCapital - development and deployment commands.
# If make is unavailable on Windows, the equivalent commands are in the README.

PY ?= python
DATA_ROOT ?= C:/mscapital_data

.PHONY: help install test lint fmt check cov validate schema-check ingest features train drift-test cosine-decomp adversarial period-diff tune recency ship shape feature-audit align export-results site api streamlit mlflow \
        docker-build up down logs clean

help:
	@echo "demo          run the whole project end to end on synthetic data (~30 s)"
	@echo "install       install dependencies (including dev)"
	@echo "test          pytest"
	@echo "cov           pytest with a coverage report"
	@echo "lint          run ruff"
	@echo "check         lint + tests"
	@echo "deploy-check  render every page in a venv built from requirements.txt ALONE"
	@echo "fmt           ruff --fix"
	@echo "train-quick   2 folds on a 25% sample, no MLflow - for smoke testing"
	@echo "validate      check the raw and feature data against their contracts"
	@echo "schema-check  do the BigQuery feature tables still match the SQL in git?"
	@echo "ingest        feather -> parquet -> BigQuery (train)"
	@echo "features      build the BigQuery feature layer and download it"
	@echo "train         walk-forward training (logs to MLflow)"
	@echo "drift-test    does feature drift predict degradation? (no submission needed)"
	@echo "cosine-decomp subgroup decomposition of the pooled cosine metric"
	@echo "adversarial   is the test set later, or different? (calibrated AUC)"
	@echo "period-diff   was the hold-out an unusually easy period?"
	@echo "tune          hyperparameter search + selection-optimism accounting"
	@echo "tune-confirm  re-check the tuned winner on full data, paired"
	@echo "recency       what do the months held back for the hold-out cost?"
	@echo "ship          build the shippable ensemble + write a submission"
	@echo "shape         build sequence-shape features, then test whether they pay"
	@echo "feature-audit how many of the 292 features are actually distinct?"
	@echo "align         weight the loss the way cosine weights rows (it hurts)"
	@echo "api           run FastAPI locally (:8000)"
	@echo "streamlit     run the dashboard locally (:8501)"
	@echo "export-results refresh results/ so the dashboard runs without the pipeline"
	@echo "site          build the static results page (deployed to Vercel)"
	@echo "mlflow        MLflow UI (:5000)"
	@echo "docker-build  build the api, app and full images"
	@echo "up / down     bring the whole stack up/down with docker compose"
	@echo "logs          follow the compose logs"
	@echo "clean         remove local run artefacts (never the data)"

demo:
	$(PY) -m src.demo

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest tests/ -q

cov:
	$(PY) -m pytest tests/ -q --cov --cov-report=term-missing

lint:
	$(PY) -m ruff check src/ api/ streamlit_app/ tests/ scripts/

fmt:
	$(PY) -m ruff check src/ api/ streamlit_app/ tests/ scripts/ --fix

check: lint test

# The check that would have caught the matplotlib defect. `make test` runs in whatever
# virtualenv you happen to be in, which on a development machine is always richer than
# the deployment's - matplotlib, mlflow and shap are all there from the pipeline work.
# This builds an EMPTY interpreter, installs requirements.txt and nothing else, and then
# actually renders each page. Slow (~2 min, it downloads the wheels) so it is not part of
# `make check`; CI runs it on every push.
DEPLOY_VENV := .venv-deploy
# venv puts the interpreter in Scripts/ on Windows and bin/ everywhere else. This project
# is developed on Windows and CI runs on Linux, so the target has to work on both.
ifeq ($(OS),Windows_NT)
DEPLOY_PY := $(DEPLOY_VENV)/Scripts/python
else
DEPLOY_PY := $(DEPLOY_VENV)/bin/python
endif

deploy-check:
	$(PY) -m venv $(DEPLOY_VENV)
	$(DEPLOY_PY) -m pip install -q --upgrade pip
	$(DEPLOY_PY) -m pip install -q -r requirements.txt
	$(DEPLOY_PY) scripts/check_deploy.py

validate:
	$(PY) -m src.data.validation --split train

schema-check:
	$(PY) -m src.data.validation --split train --schema

ingest:
	$(PY) -c "from src.data.ingestion import convert_table; \
	  [convert_table('train', t) for t in ('market','order','transaction')]"
	$(PY) -c "from src.data.bq_loader import ensure_datasets, load_label, load_table; \
	  ensure_datasets(); load_label(); \
	  [load_table('train', t) for t in ('market','order','transaction')]"
	$(PY) -c "from src.data.staging import build_all; build_all('train')"

features:
	$(PY) -c "from src.features.assemble import build_blocks, assemble, download; \
	  build_blocks('train'); assemble('train'); download('train')"

train:
	$(PY) -m src.models.train

drift-test:
	$(PY) -m src.evaluation.drift_robustness --threshold 0.2

cosine-decomp:
	$(PY) -m src.evaluation.cosine_decomposition

adversarial:
	$(PY) -m src.evaluation.adversarial

period-diff:
	$(PY) -m src.evaluation.period_difficulty

tune:
	$(PY) -m src.models.tuning --trials 40

tune-confirm:
	$(PY) -m src.models.tuning --confirm

recency:
	$(PY) -m src.models.recency

ship:
	$(PY) -m src.models.ship

feature-audit:
	$(PY) -m src.evaluation.feature_audit

align:
	$(PY) -m src.models.metric_alignment

shape:
	$(PY) -m src.features.shape_features --split train
	$(PY) -m src.features.shape_features --split train --download
	$(PY) -m src.evaluation.shape_gain

train-quick:
	$(PY) -m src.models.train --quick --folds 2 --sample-frac 0.25 --no-mlflow

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

site:
	$(PY) -m src.data.build_site

export-results:
	$(PY) -m src.data.export_results

streamlit:
	$(PY) -m streamlit run streamlit_app/app.py --server.port 8501

mlflow:
	$(PY) -m mlflow server --host 0.0.0.0 --port 5000 \
	  --backend-store-uri sqlite:///$(DATA_ROOT)/mlruns/mlflow.db \
	  --default-artifact-root $(DATA_ROOT)/mlruns/artifacts

docker-build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

# MSCapital - development and deployment commands.
# If make is unavailable on Windows, the equivalent commands are in the README.

PY ?= python
DATA_ROOT ?= C:/mscapital_data

.PHONY: help install test lint fmt check validate ingest features train drift-test cosine-decomp adversarial period-diff tune api streamlit mlflow \
        docker-build up down logs clean

help:
	@echo "demo          run the whole project end to end on synthetic data (~30 s)"
	@echo "install       install dependencies (including dev)"
	@echo "test          pytest"
	@echo "lint          run ruff"
	@echo "check         lint + tests"
	@echo "validate      check the raw and feature data against their contracts"
	@echo "ingest        feather -> parquet -> BigQuery (train)"
	@echo "features      build the BigQuery feature layer and download it"
	@echo "train         walk-forward training (logs to MLflow)"
	@echo "drift-test    does feature drift predict degradation? (no submission needed)"
	@echo "cosine-decomp subgroup decomposition of the pooled cosine metric"
	@echo "adversarial   is the test set later, or different? (calibrated AUC)"
	@echo "period-diff   was the hold-out an unusually easy period?"
	@echo "tune          hyperparameter search + selection-optimism accounting"
	@echo "tune-confirm  re-check the tuned winner on full data, paired"
	@echo "api           run FastAPI locally (:8000)"
	@echo "streamlit     run the dashboard locally (:8501)"
	@echo "mlflow        MLflow UI (:5000)"
	@echo "up / down     bring the whole stack up/down with docker compose"

demo:
	$(PY) -m src.demo

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src/ api/ streamlit_app/ tests/ --line-length 100

fmt:
	$(PY) -m ruff check src/ api/ streamlit_app/ tests/ --line-length 100 --fix

check: lint test

validate:
	$(PY) -m src.data.validation --split train

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

train-quick:
	$(PY) -m src.models.train --quick --folds 2 --sample-frac 0.25 --no-mlflow

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

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

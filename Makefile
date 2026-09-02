# MSCapital - development and deployment commands.
# If make is unavailable on Windows, the equivalent commands are in the README.

PY ?= python
DATA_ROOT ?= C:/mscapital_data

.PHONY: help install test lint fmt check ingest features train api streamlit mlflow \
        docker-build up down logs clean

help:
	@echo "install       install dependencies (including dev)"
	@echo "test          pytest"
	@echo "lint          run ruff"
	@echo "check         lint + tests"
	@echo "ingest        feather -> parquet -> BigQuery (train)"
	@echo "features      build the BigQuery feature layer and download it"
	@echo "train         walk-forward training (logs to MLflow)"
	@echo "api           run FastAPI locally (:8000)"
	@echo "streamlit     run the dashboard locally (:8501)"
	@echo "mlflow        MLflow UI (:5000)"
	@echo "up / down     bring the whole stack up/down with docker compose"

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src/ api/ streamlit_app/ tests/ --line-length 100

fmt:
	$(PY) -m ruff check src/ api/ streamlit_app/ tests/ --line-length 100 --fix

check: lint test

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

train-quick:
	$(PY) -m src.models.train --quick --folds 2 --sample-frac 0.25 --no-mlflow

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	$(PY) -m streamlit run streamlit_app/app.py --server.port 8501

mlflow:
	$(PY) -m mlflow server --host 0.0.0.0 --port 5000 \
	  --backend-store-uri sqlite:///$(DATA_ROOT)/mlflow/mlflow.db \
	  --default-artifact-root $(DATA_ROOT)/mlflow/artifacts

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

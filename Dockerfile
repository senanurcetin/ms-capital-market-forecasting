# MSCapital - multi-stage, one target per service.
#
# The single-image build was 3.48 GB because the API inherited every dependency in the
# project: MLflow, Streamlit, SHAP (and its numba/llvmlite stack), DuckDB, Polars, the
# Kaggle client and the GCP clients. None of that is needed to load an artefact and
# score a row. Each target now installs only what it actually runs.
#
#   docker build --target api  -t mscapital:api  .
#   docker build --target app  -t mscapital:app  .
#   docker build --target full -t mscapital:full .    # training / MLflow
#
# docker-compose selects the target per service.

# ---------------------------------------------------------------- base
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# libgomp is the runtime OpenMP library LightGBM and XGBoost link against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser && mkdir -p /data

# ---------------------------------------------------------------- api
FROM base AS api

COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# Only what the API imports: the predictor, the config loader, and the app itself.
COPY src/__init__.py src/config.py ./src/
COPY src/inference/ ./src/inference/
COPY configs/ ./configs/
COPY api/ ./api/

RUN chown -R appuser:appuser /app /data
USER appuser

# Starts even without a model; /health then reports "degraded".
ENV MSCAPITAL_MODEL_DIR=/data/models/current
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- app (dashboard)
FROM base AS app

COPY requirements-app.txt .
RUN pip install --upgrade pip && pip install -r requirements-app.txt

COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY configs/ ./configs/

RUN chown -R appuser:appuser /app /data
USER appuser

ENV MSCAPITAL_DATA_ROOT=/data
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# ---------------------------------------------------------------- full
FROM base AS full

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY configs/ ./configs/
COPY sql/ ./sql/

RUN chown -R appuser:appuser /app /data
USER appuser

ENV MSCAPITAL_MODEL_DIR=/data/models/current
EXPOSE 8000 8501 5000
CMD ["python", "-m", "src.demo"]

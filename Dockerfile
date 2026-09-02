# MSCapital - one image, several services (api / streamlit / mlflow). Which one runs is
# chosen by the command in docker-compose, so the dependency tree lives in one place.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# libgomp is required at runtime by LightGBM and XGBoost.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first so a code change does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY configs/ ./configs/
COPY sql/ ./sql/

# Run as a non-root user
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data && chown -R appuser:appuser /app /data
USER appuser

# Starts even without a model; /health then reports "degraded".
ENV MSCAPITAL_MODEL_DIR=/data/models/current

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# MSCapital - tek imaj, iki servis (api / streamlit). Hangi servis calisacagi
# docker-compose'daki command ile secilir; boylece bagimlilik agaci tek yerde durur.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# LightGBM/XGBoost calisma zamani icin libgomp gerekli.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Bagimliliklar once: kod degisince pip katmani yeniden kurulmasin.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY configs/ ./configs/
COPY sql/ ./sql/

# Root olmayan kullanici
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data && chown -R appuser:appuser /app /data
USER appuser

# Model yoksa da ayaga kalkar; /health "degraded" doner.
ENV MSCAPITAL_MODEL_DIR=/data/models/current

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

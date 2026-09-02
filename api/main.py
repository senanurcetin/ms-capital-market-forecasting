"""MSCapital model servisi (FastAPI).

Egitim kodundan BAGIMSIZ: yalnizca src.inference.predictor uzerinden
kaydedilmis artefakti okur. Model yoksa uygulama yine de ayaga kalkar ama
/health "degraded" doner ve tahmin uclari 503 verir - boylece container
saglik kontrolu anlamli kalir ve deploy sirasi modele bagimli olmaz.

ARASTIRMA AMACLIDIR - yatirim tavsiyesi degildir.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predictor import ModelNotLoadedError, Predictor

log = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MSCAPITAL_MODEL_DIR", "C:/mscapital_data/models/current")
DEADBAND = float(os.environ.get("MSCAPITAL_DIRECTION_DEADBAND", "0"))

state: dict[str, Any] = {"predictor": None, "error": None}


def _try_load() -> None:
    try:
        state["predictor"] = Predictor.from_dir(MODEL_DIR)
        state["error"] = None
        log.info("model yuklendi: %s", state["predictor"].info())
    except (ModelNotLoadedError, FileNotFoundError) as exc:
        state["predictor"] = None
        state["error"] = str(exc)
        log.warning("model yuklenemedi (%s) - servis degraded modda", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load()
    yield


app = FastAPI(
    title="MSCapital Market Forecasting API",
    description="Kisa vadeli getiri tahmini. Arastirma amaclidir; yatirim tavsiyesi degildir.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature adi -> deger")


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(..., min_length=1, max_length=10_000)


class PredictResponse(BaseModel):
    predicted_return: float
    direction: str
    model_name: str
    model_version: str


def _predictor() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Servis edilebilir model yok: "
                + str(state["error"])
                + f". MSCAPITAL_MODEL_DIR={MODEL_DIR}"
            ),
        )
    return state["predictor"]


@app.get("/health")
def health() -> dict:
    ready = state["predictor"] is not None
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        "detail": None if ready else state["error"],
    }


@app.get("/model-info")
def model_info() -> dict:
    return _predictor().info()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    p = _predictor()
    try:
        value = float(p.predict([req.features])[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(
        predicted_return=value,
        direction=Predictor.direction(value, DEADBAND),
        model_name=p.bundle.name,
        model_version=p.bundle.version,
    )


@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest) -> dict:
    p = _predictor()
    try:
        values = p.predict(req.rows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n": len(values),
        "predictions": [
            {"predicted_return": float(v), "direction": Predictor.direction(v, DEADBAND)}
            for v in values
        ],
        "model_name": p.bundle.name,
        "model_version": p.bundle.version,
    }


@app.post("/reload")
def reload_model() -> dict:
    """Yeni model kaydedildikten sonra servisi yeniden baslatmadan yukler."""
    _try_load()
    return health()

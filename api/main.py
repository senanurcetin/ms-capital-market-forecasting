"""MSCapital model serving API (FastAPI).

INDEPENDENT OF THE TRAINING CODE: it only reads a saved artefact through
src.inference.predictor. If no model is present the app still starts, /health
reports "degraded" and the prediction endpoints return 503 - so the container
health check stays meaningful and deployment order does not depend on the model.

FOR RESEARCH ONLY - not investment advice.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from api.guards import Metrics, RateLimiter
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from src.inference.predictor import ModelNotLoadedError, Predictor

log = logging.getLogger(__name__)

# One variable decides where everything lives; MSCAPITAL_MODEL_DIR overrides just the model.
DATA_ROOT = os.environ.get("MSCAPITAL_DATA_ROOT", "C:/mscapital_data")
MODEL_DIR = os.environ.get("MSCAPITAL_MODEL_DIR", f"{DATA_ROOT}/models/current")
DEADBAND = float(os.environ.get("MSCAPITAL_DIRECTION_DEADBAND", "0"))
# /reload swaps the served model, so it is not open by default: with no token configured the
# endpoint answers 403, and with one it needs a matching X-Admin-Token header.
ADMIN_TOKEN = os.environ.get("MSCAPITAL_ADMIN_TOKEN", "")
# Limits on what one request may cost. The body cap is checked from Content-Length before
# anything is parsed; the row cap is enforced by the request model. Rate limiting is off
# (0) unless set, because behind a proxy every client shares one address.
MAX_BATCH_ROWS = int(os.environ.get("MSCAPITAL_MAX_BATCH_ROWS", "10000"))
MAX_BODY_BYTES = int(os.environ.get("MSCAPITAL_MAX_BODY_BYTES", str(20 * 1024 * 1024)))
RATE_LIMIT_PER_MIN = int(os.environ.get("MSCAPITAL_RATE_LIMIT_PER_MIN", "0"))
# Only the endpoints that run the model are limited; /health must stay answerable.
_LIMITED_PATHS = {"/predict", "/batch-predict"}

limiter = RateLimiter(RATE_LIMIT_PER_MIN)
metrics = Metrics()

state: dict[str, Any] = {"predictor": None, "error": None}
# Both keys are written together; without the lock a request could see a predictor from one
# load beside the error text from another while /reload is running.
_state_lock = threading.Lock()


def _try_load() -> None:
    try:
        predictor = Predictor.from_dir(MODEL_DIR)
    except (ModelNotLoadedError, FileNotFoundError) as exc:
        with _state_lock:
            state["predictor"] = None
            state["error"] = str(exc)
        log.warning("could not load model (%s) - serving in degraded mode", exc)
        return
    with _state_lock:
        state["predictor"] = predictor
        state["error"] = None
    log.info("model loaded: %s", predictor.info())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load()
    yield


app = FastAPI(
    title="MSCapital Market Forecasting API",
    description="Short-horizon return prediction. For research only; not investment advice.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def guard_and_measure(request: Request, call_next):
    """Reject oversized or over-rate requests, and time everything for /metrics."""
    start = time.perf_counter()
    response = await _guard(request)
    if response is None:
        response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", None) or "unmatched"
    metrics.observe(request.method, path, response.status_code, time.perf_counter() - start)
    return response


async def _guard(request: Request) -> JSONResponse | None:
    if request.method == "POST":
        length = request.headers.get("content-length")
        if length is None and "chunked" in request.headers.get("transfer-encoding", ""):
            return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
        if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse(
                {"detail": f"request body over {MAX_BODY_BYTES} bytes"}, status_code=413
            )
    if request.url.path in _LIMITED_PATHS:
        client = request.client.host if request.client else "unknown"
        retry_after = limiter.check(client)
        if retry_after is not None:
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
    return None


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name -> value")


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(..., min_length=1, max_length=MAX_BATCH_ROWS)


class PredictResponse(BaseModel):
    predicted_return: float
    direction: str
    model_name: str
    model_version: str


def _predictor() -> Predictor:
    with _state_lock:
        predictor, error = state["predictor"], state["error"]
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"No servable model available: {error}. MSCAPITAL_MODEL_DIR={MODEL_DIR}",
        )
    return predictor


@app.get("/health")
def health() -> dict:
    with _state_lock:
        ready, error = state["predictor"] is not None, state["error"]
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        "detail": None if ready else error,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint() -> str:
    """Prometheus text exposition: counts and latencies only, never request contents."""
    return metrics.render()


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
    metrics.add_rows(1)
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
    metrics.add_rows(len(values))
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
def reload_model(x_admin_token: str | None = Header(default=None)) -> dict:
    """Pick up a newly saved model without restarting the service. Needs the admin token."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="/reload is disabled: set MSCAPITAL_ADMIN_TOKEN to enable it",
        )
    if x_admin_token is None or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="missing or invalid X-Admin-Token")
    _try_load()
    return health()

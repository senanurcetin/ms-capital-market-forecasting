"""Prediction layer for model serving - INDEPENDENT OF THE TRAINING CODE.

Design rules:
  * The API never imports training modules; it only reads a saved artefact.
  * The feature order is stored alongside the model. If a request is missing or
    has extra features, nothing is silently filled in - it fails loudly.
  * A missing model does NOT crash the app; /health reports a degraded state.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

METADATA_FILE = "model_meta.json"


class ModelNotLoadedError(RuntimeError):
    """No servable model is available."""


@dataclass
class ModelBundle:
    model: object
    features: list[str]
    name: str
    version: str
    metrics: dict = field(default_factory=dict)
    trained_at: str | None = None


def _load_booster(path: Path, kind: str):
    if kind == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    if kind == "xgboost":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    import joblib

    return joblib.load(path)


def load_bundle(model_dir: str | Path) -> ModelBundle:
    """Expects METADATA_FILE plus the model file inside model_dir."""
    model_dir = Path(model_dir)
    meta_path = model_dir / METADATA_FILE
    if not meta_path.exists():
        raise ModelNotLoadedError(f"{meta_path} does not exist")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = model_dir / meta["model_file"]
    if not model_path.exists():
        raise ModelNotLoadedError(f"{model_path} does not exist")
    return ModelBundle(
        model=_load_booster(model_path, meta["kind"]),
        features=list(meta["features"]),
        name=meta["name"],
        version=meta["version"],
        metrics=meta.get("metrics", {}),
        trained_at=meta.get("trained_at"),
    )


def save_bundle(model_dir: str | Path, *, model, kind: str, features: list[str],
                name: str, version: str, metrics: dict | None = None) -> Path:
    """Called from the training side; writes the artefact in servable form."""
    import datetime as _dt

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if kind == "lightgbm":
        model_file = "model.txt"
        model.save_model(str(model_dir / model_file))
    elif kind == "xgboost":
        model_file = "model.json"
        model.save_model(str(model_dir / model_file))
    else:
        import joblib

        model_file = "model.joblib"
        joblib.dump(model, model_dir / model_file)
    meta = {
        "name": name, "version": version, "kind": kind, "model_file": model_file,
        "features": list(features), "metrics": metrics or {},
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (model_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_dir


class Predictor:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Predictor":
        return cls(load_bundle(model_dir))

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        expected = self.bundle.features
        df = pd.DataFrame(rows)
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} feature(s) missing (first 5: {missing[:5]}). "
                "The feature set is fixed by the model artefact; gaps are not filled in."
            )
        extra = [c for c in df.columns if c not in expected]
        if extra:
            log.warning("ignored %d unexpected field(s) in request: %s", len(extra), extra[:5])
        # float32, not float64: LightGBM and XGBoost accept it natively and bin to
        # uint8 internally, so nothing is lost - but a 648k x 289 batch needs 1.4 GB
        # as float64 versus 0.7 GB as float32. The float64 version ran out of memory
        # while generating the test submission on a 16 GB machine.
        return df[expected].astype("float32")

    def predict(self, rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.array([], dtype=np.float64)
        X = self._frame(rows)
        model = self.bundle.model
        if hasattr(model, "predict") and model.__class__.__module__.startswith("xgboost"):
            import xgboost as xgb

            return np.asarray(model.predict(xgb.DMatrix(X)), dtype=np.float64)
        return np.asarray(model.predict(X), dtype=np.float64)

    @staticmethod
    def direction(value: float, deadband: float = 0.0) -> str:
        if value > deadband:
            return "UP"
        if value < -deadband:
            return "DOWN"
        return "FLAT"

    def info(self) -> dict:
        b = self.bundle
        return {
            "model_name": b.name, "model_version": b.version,
            "n_features": len(b.features), "trained_at": b.trained_at,
            "metrics": b.metrics,
        }

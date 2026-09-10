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
    """A trained model plus everything needed to score a row with it.

    `features` is the part that matters: it pins the exact column ORDER the model was
    fitted on. Gradient boosters index features positionally, so serving rows in a
    different order produces confident nonsense rather than an error.
    """

    model: object
    features: list[str]
    name: str
    version: str
    metrics: dict = field(default_factory=dict)
    trained_at: str | None = None


def _load_booster(path: Path, kind: str):
    if kind == "ensemble":
        # `path` is the directory holding the base models, not a single file - an
        # ensemble is several artefacts plus the weights that combine them.
        from src.inference.ensemble import load_ensemble

        return load_ensemble(path.parent if path.is_file() else path)
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


# Every filename this module can write, plus `ensemble.joblib` from the format that
# preceded it. save_bundle clears the lot before writing, so a directory always holds
# exactly one artefact.
#
# Without this, rebuilding an ensemble over a single-model bundle left a stale `model.txt`
# beside the new `model_lightgbm.txt` - same size, older date, read by nothing. The next
# person to look would have had to diff the metadata to work out which was live.
ARTEFACT_FILES = (
    "model.txt", "model.json", "model.joblib",
    "ensemble_meta.json", "ensemble.joblib",
    "model_lightgbm.txt", "model_xgboost.json", "model_ridge.joblib",
)


def save_bundle(model_dir: str | Path, *, model, kind: str, features: list[str],
                name: str, version: str, metrics: dict | None = None) -> Path:
    """Called from the training side; writes the artefact in servable form."""
    import datetime as _dt

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for stale in ARTEFACT_FILES:
        (model_dir / stale).unlink(missing_ok=True)
    if kind == "ensemble":
        from src.inference.ensemble import META_FILE, save_ensemble

        # `model` is the dict of fitted training wrappers; save_ensemble is the boundary
        # that converts them into files loadable without the training package.
        save_ensemble(model_dir, models=model["models"], weights=model["weights"],
                      features=features)
        model_file = META_FILE
    elif kind == "lightgbm":
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
        "trained_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }
    (model_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_dir


class Predictor:
    """Serving wrapper: dict rows in, predictions out.

    Deliberately independent of the training code - it loads an artefact from disk and
    knows nothing about BigQuery, folds or the feature pipeline. That is what lets the
    API image ship without the training dependencies.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> Predictor:
        """Load an artefact directory written by save_bundle()."""
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
        """Score a batch of rows given as dicts.

        Missing features raise rather than defaulting to zero: a silently imputed column
        is a wrong prediction that looks like a right one.
        """
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
        """Turn a predicted return into UP / DOWN / FLAT.

        Presentation only. The competition metric is scale-invariant, so the sign is
        interpretable while the magnitude is not - a deadband makes that explicit rather
        than implying precision the number does not have.
        """
        if value > deadband:
            return "UP"
        if value < -deadband:
            return "DOWN"
        return "FLAT"

    def info(self) -> dict:
        """Artefact metadata for /model-info: name, version, feature count, metrics."""
        b = self.bundle
        return {
            "model_name": b.name, "model_version": b.version,
            "n_features": len(b.features), "trained_at": b.trained_at,
            "metrics": b.metrics,
        }

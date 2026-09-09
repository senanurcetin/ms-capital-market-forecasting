"""A weighted blend of base models, in a form the serving layer can load on its own.

WHY THIS EXISTS

`ship.py` builds a three-model ensemble and uses it to generate the submission - so the
ensemble is what produced the leaderboard score. But `save_bundle` could only store ONE
model of one kind, so the servable artefact was the bare LightGBM booster, saved under the
name "ensemble". The API's /model-info therefore announced an ensemble while /predict
returned single-model predictions, and the two numbers a reader would compare - the score
in the README and the model behind the endpoint - came from different models.

WHY NOT JUST PICKLE THE WRAPPERS

`ensemble.joblib` already holds the fitted LightGBMModel/XGBoostModel/RidgeModel wrappers,
and loading it would be one line. It is not used here because unpickling imports the
module each class was defined in: the API would pull in `src.models.*`, which is the one
thing the serving layer is designed not to do (see predictor.py) and which is why the API
image can ship without the training dependencies at all.

So each part is stored in a format that loads with the RUNTIME dependencies only:

    lightgbm   model_lightgbm.txt    Booster.save_model  -> lgb.Booster(model_file=...)
    xgboost    model_xgboost.json    Booster.save_model  -> xgb.Booster().load_model(...)
    ridge      model_ridge.joblib    the fitted sklearn Ridge and StandardScaler, plus
                                     the imputation medians as a plain dict

The ridge entry carries the medians deliberately. Unwrapping RidgeModel to its inner
sklearn estimator once lost the median imputation, and predict() then raised
`Input X contains NaN` on the first missing value in the test set - a failure that only
appears on data with gaps, which the training fold does not have.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

META_FILE = "ensemble_meta.json"


def _predict_lightgbm(part: dict, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        part["booster"].predict(X, num_iteration=part["best_iteration"]), dtype=np.float64
    )


def _predict_xgboost(part: dict, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    best = part["best_iteration"]
    # (0, 0) means "every tree"; None is not a valid iteration_range and raises inside
    # Booster.predict. See the same note in src/models/xgboost_model.py.
    rng = (0, best + 1) if best is not None else (0, 0)
    return np.asarray(
        part["booster"].predict(xgb.DMatrix(X, nthread=-1), iteration_range=rng),
        dtype=np.float64,
    )


def _predict_ridge(part: dict, X: pd.DataFrame) -> np.ndarray:
    # Same three steps as RidgeModel.predict, in the same order. Ridge is positional, so
    # the reindex is load-bearing: a frame whose columns arrive in a different order would
    # pair every coefficient with the wrong feature and return confident nonsense.
    medians = pd.Series(part["medians"], dtype="float64")
    filled = X.replace([np.inf, -np.inf], np.nan).fillna(medians)
    return np.asarray(
        part["ridge"].predict(part["scaler"].transform(filled)), dtype=np.float64
    )


_PREDICT = {
    "lightgbm": _predict_lightgbm,
    "xgboost": _predict_xgboost,
    "ridge": _predict_ridge,
}


class EnsembleArtefact:
    """Base models plus blend weights. Satisfies the `predict(X)` contract Predictor uses.

    Predictor treats `bundle.model` as anything with `.predict(frame)`, which is why this
    slots in with no change to the request path.
    """

    def __init__(self, parts: dict[str, dict], weights: dict[str, float],
                 features: list[str]) -> None:
        self.parts = parts
        self.weights = weights
        self.features = list(features)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted blend over the base models, each scored on the fitted column order."""
        frame = X[self.features]
        total = np.zeros(len(frame), dtype=np.float64)
        for name, weight in self.weights.items():
            if weight == 0.0:
                # NNLS can zero a model out entirely; loading it would still cost the
                # memory and the scoring pass.
                continue
            total += weight * _PREDICT[self.parts[name]["kind"]](self.parts[name], frame)
        return total

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-model predictions, unweighted - for showing WHERE a blend disagrees."""
        frame = X[self.features]
        return pd.DataFrame(
            {n: _PREDICT[p["kind"]](p, frame) for n, p in self.parts.items()},
            index=X.index,
        )


def save_ensemble(model_dir: str | Path, *, models: dict, weights: dict[str, float],
                  features: list[str]) -> Path:
    """Write the fitted wrappers out in a form the serving layer can read.

    `models` holds the TRAINING wrappers (LightGBMModel and friends); this is the one
    place allowed to know their internals, because it is the boundary that converts them
    into something loadable without them.
    """
    import joblib

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    entries = {}

    for name, m in models.items():
        if name == "lightgbm":
            m.booster_.save_model(str(model_dir / "model_lightgbm.txt"))
            entries[name] = {"kind": "lightgbm", "file": "model_lightgbm.txt",
                             "best_iteration": m.best_iteration_}
        elif name == "xgboost":
            m.booster_.save_model(str(model_dir / "model_xgboost.json"))
            entries[name] = {"kind": "xgboost", "file": "model_xgboost.json",
                             "best_iteration": m.best_iteration_}
        elif name == "ridge":
            joblib.dump({"ridge": m.model, "scaler": m.scaler,
                         "medians": m.imputer.medians_.to_dict()},
                        model_dir / "model_ridge.joblib")
            entries[name] = {"kind": "ridge", "file": "model_ridge.joblib"}
        else:
            raise ValueError(f"no serving format defined for base model {name!r}")

    meta = {"features": list(features), "weights": dict(weights), "parts": entries}
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_dir


def load_ensemble(model_dir: str | Path) -> EnsembleArtefact:
    """Rebuild the artefact using the runtime dependencies only."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    parts: dict[str, dict] = {}

    for name, entry in meta["parts"].items():
        path = model_dir / entry["file"]
        if entry["kind"] == "lightgbm":
            import lightgbm as lgb

            parts[name] = {"kind": "lightgbm", "booster": lgb.Booster(model_file=str(path)),
                           "best_iteration": entry.get("best_iteration")}
        elif entry["kind"] == "xgboost":
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(path))
            parts[name] = {"kind": "xgboost", "booster": booster,
                           "best_iteration": entry.get("best_iteration")}
        else:
            import joblib

            parts[name] = {"kind": "ridge", **joblib.load(path)}

    return EnsembleArtefact(parts, meta["weights"], meta["features"])

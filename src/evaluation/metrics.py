"""Evaluation metrics.

PRIMARY METRIC: cosine similarity.
    cos(y, yhat) = sum(y*yhat) / (||y|| * ||yhat||)

Its properties drive several modelling decisions:
  * SCALE-INVARIANT: multiplying yhat by a positive constant does NOT change the
    score, so calibrating prediction magnitude is wasted effort.
  * NOT SHIFT-INVARIANT: adding a constant bias to yhat DOES hurt the score.
    (This is where it differs from Pearson correlation, which centres first.)
    Empirically confirmed: a constant-mean predictor scores -0.0036.
    -> predictions should stay centred on zero.
  * Training on MSE/Huber is a legitimate surrogate, but model SELECTION is done
    on cosine.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def cosine_similarity(y_true, y_pred) -> float:
    """The competition's primary metric. Returns 0.0 if either vector is all zeros."""
    y, p = _as_1d(y_true), _as_1d(y_pred)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    denom = np.linalg.norm(y) * np.linalg.norm(p)
    if denom < _EPS:
        return 0.0
    return float(np.dot(y, p) / denom)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_as_1d(y_true) - _as_1d(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_as_1d(y_true) - _as_1d(y_pred)) ** 2)))


def pearson(y_true, y_pred) -> float:
    y, p = _as_1d(y_true), _as_1d(y_pred)
    if np.std(y) < _EPS or np.std(p) < _EPS:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def directional_accuracy(y_true, y_pred, *, ignore_zeros: bool = True) -> float:
    """Sign accuracy. 5.54% of targets are EXACTLY zero (a tick-size artefact), so
    those rows are excluded by default - including them makes the metric misleading."""
    y, p = _as_1d(y_true), _as_1d(y_pred)
    mask = y != 0 if ignore_zeros else np.ones_like(y, dtype=bool)
    if not mask.any():
        return 0.0
    return float(np.mean(np.sign(y[mask]) == np.sign(p[mask])))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """All metrics in one pass."""
    return {
        "cosine": cosine_similarity(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def lgb_cosine_eval(y_pred, dataset):
    """LightGBM custom eval so early stopping tracks cosine, not RMSE."""
    y_true = dataset.get_label()
    return "cosine", cosine_similarity(y_true, y_pred), True  # True = higher is better


def xgb_cosine_eval(y_pred, dmatrix):
    """XGBoost custom eval (feval). XGBoost minimises, so the sign is flipped."""
    return "neg_cosine", -cosine_similarity(dmatrix.get_label(), y_pred)

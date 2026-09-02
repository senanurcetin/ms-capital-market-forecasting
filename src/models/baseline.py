"""Baseline models - the reference points every other score is judged against.

ZeroPredictor is deliberately included: cosine similarity is undefined (norm 0)
for a constant-zero prediction and returns 0.0 here. Seeing that 0.0 is a
necessary control that the other scores really do carry signal.

MeanPredictor is the empirical demonstration that cosine is NOT shift-invariant:
predicting the training mean scores NEGATIVE (-0.0036 on the walk-forward folds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base import MedianImputer, feature_columns


class ZeroPredictor:
    name = "zero"

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> "ZeroPredictor":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)


class MeanPredictor:
    """Predicts the training mean - shows how a constant bias damages the metric."""

    name = "mean"

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> "MeanPredictor":
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value_, dtype=np.float64)


class RidgeModel:
    """Median imputation + standardisation + Ridge.

    The imputer and the scaler are fitted on the TRAINING FOLD ONLY
    (see the NaN policy in base.py).
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.imputer = MedianImputer()
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=0)
        self.features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> "RidgeModel":
        self.features_ = feature_columns(X)
        Xf = self.imputer.fit_transform(X[self.features_])
        Xs = self.scaler.fit_transform(Xf)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = self.imputer.transform(X[self.features_])
        return self.model.predict(self.scaler.transform(Xf))

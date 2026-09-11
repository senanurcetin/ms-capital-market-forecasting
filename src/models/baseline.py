"""Baseline models - the reference points every other score is judged against.

ZeroPredictor is deliberately included: cosine similarity is undefined (norm 0)
for a constant-zero prediction and returns 0.0 here. Seeing that 0.0 is a
necessary control that the other scores really do carry signal.

MeanPredictor is the empirical demonstration that cosine is NOT shift-invariant. A
constant carries no information, so its score is noise around zero - and crucially it can
go NEGATIVE, which a metric bounded below by zero could not do. Across the walk-forward
folds it ranges -0.0071 to +0.0219, negative in three of five, averaging +0.0059. The sign
is decided by whether the fold's target mean happens to agree with the constant, which is
exactly the point: a shift is not free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base import MedianImputer, feature_columns


class ZeroPredictor:
    """Predicts zero everywhere. The floor every other score is read against."""

    name = "zero"

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> ZeroPredictor:
        """Nothing to learn; present only to satisfy the Model contract."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Zeros. Cosine is undefined against a zero-norm vector and scores 0.0."""
        return np.zeros(len(X), dtype=np.float64)


class MeanPredictor:
    """Predicts the training mean - shows how a constant bias damages the metric."""

    name = "mean"

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> MeanPredictor:
        """Store the training mean. Features are ignored by construction."""
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """That one constant, repeated - which scores NEGATIVE under cosine."""
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

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> RidgeModel:
        """Fit imputer, scaler and Ridge in sequence, all on this fold only.

        The column order is recorded so predict() can reindex - Ridge is positional and
        would otherwise pair coefficients with the wrong columns without complaining.
        """
        self.features_ = feature_columns(X)
        Xf = self.imputer.fit_transform(X[self.features_])
        Xs = self.scaler.fit_transform(Xf)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Impute, scale, predict - reindexed to the fitted column order.

        `X[self.features_]` is doing real work: it both selects and reorders. Serving a
        frame whose columns arrive in a different order would otherwise pair each
        coefficient with the wrong feature, silently.
        """
        Xf = self.imputer.transform(X[self.features_])
        return self.model.predict(self.scaler.transform(Xf))

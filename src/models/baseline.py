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

    def __init__(self, alpha: float = 1.0, chunk_rows: int = 100_000) -> None:
        self.alpha = alpha
        self.chunk_rows = chunk_rows
        self.imputer = MedianImputer()
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=0)
        self.features_: list[str] = []

    def _chunks(self, n: int):
        for start in range(0, n, self.chunk_rows):
            yield slice(start, min(start + self.chunk_rows, n))

    def _block(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute one slice of rows, as float64 and still a frame.

        Staying a DataFrame lets the scaler record the feature names, as it always did, so
        serving code that passes it a frame does not trip sklearn's names warning.
        """
        return self.imputer.transform(X[self.features_]).astype(np.float64)

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> RidgeModel:
        """Fit imputer, scaler and Ridge in sequence, all on this fold only.

        The column order is recorded so predict() can reindex - Ridge is positional and
        would otherwise pair coefficients with the wrong columns without complaining.

        The data is streamed in chunks, never materialised whole. Handing the full frame to
        StandardScaler and sklearn's Ridge cost three or four float64 copies of it; on the
        1.2M x 292 training set that is what killed `make ship` on a 16 GB machine. Ridge
        itself is solved from the normal equations accumulated in float64, with the same
        centring sklearn applies for fit_intercept=True, so the coefficients agree with
        Ridge.fit to rounding error (tests/test_baseline_chunked.py pins this).
        """
        self.features_ = feature_columns(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(X)
        self.imputer.fit(X, columns=self.features_)

        self.scaler = StandardScaler()
        for sl in self._chunks(n):
            self.scaler.partial_fit(self._block(X.iloc[sl]))

        p = len(self.features_)
        gram, xty = np.zeros((p, p)), np.zeros(p)
        sum_z, sum_y = np.zeros(p), 0.0
        for sl in self._chunks(n):
            Z = self.scaler.transform(self._block(X.iloc[sl]))
            yc = y[sl]
            gram += Z.T @ Z
            xty += Z.T @ yc
            sum_z += Z.sum(axis=0)
            sum_y += yc.sum()

        z_bar, y_bar = sum_z / n, sum_y / n
        gram -= n * np.outer(z_bar, z_bar)
        xty -= n * z_bar * y_bar
        coef = np.linalg.solve(gram + self.alpha * np.eye(p), xty)

        self.model.coef_ = coef
        self.model.intercept_ = y_bar - z_bar @ coef
        self.model.n_features_in_ = p
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Impute, scale, predict - reindexed to the fitted column order.

        `X[self.features_]` is doing real work: it both selects and reorders. Serving a
        frame whose columns arrive in a different order would otherwise pair each
        coefficient with the wrong feature, silently.
        """
        out = [self.model.predict(self.scaler.transform(self._block(X.iloc[sl])))
               for sl in self._chunks(len(X))]
        return np.concatenate(out) if out else np.empty(0)

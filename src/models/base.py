"""Model interface and shared helpers.

Every model implements the same contract: fit(X, y) / predict(X). That keeps the
walk-forward harness (src/models/train.py) independent of the model type.

NaN POLICY (reflecting what the feature layer actually produces):
  LightGBM and XGBoost handle NaN natively -> left untouched.
  Ridge cannot -> median imputation, but the median is computed on the TRAIN FOLD
  ONLY. Using validation/test statistics would be leakage.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

# Non-feature columns. month is ONLY a split key, never a feature: at test time it
# would fall outside the observed range (71+), so a model using it cannot generalise.
NON_FEATURES = ("sample_id", "month", "target")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


class Model(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class MedianImputer:
    """Median imputation fitted on the training fold, plus infinity cleanup."""

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        clean = X.replace([np.inf, -np.inf], np.nan)
        self.medians_ = clean.median()
        # Columns that are entirely missing fall back to 0.
        self.medians_ = self.medians_.fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("fit() must be called first")
        return X.replace([np.inf, -np.inf], np.nan).fillna(self.medians_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

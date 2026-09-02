"""Baseline modeller: referans noktasi.

ZeroPredictor kasitli olarak dahil: cosine similarity sabit tahminde tanimsiz
(norm 0) veya anlamsizdir; metrigin 0 dondugunu gormek diger skorlarin
gercekten sinyal tasidigini dogrulamak icin gerekli bir kontrol.
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
    """Train ortalamasini tahmin eder. Cosine KAYDIRMA-degismez olmadigi icin
    sabit-bias tahminin metrigi nasil bozdugunu gosteren referans."""

    name = "mean"

    def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> "MeanPredictor":
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value_, dtype=np.float64)


class RidgeModel:
    """Standardize + medyan imputation + Ridge.

    Imputer ve scaler YALNIZ train fold'unda fit edilir (bkz. base.py NaN politikasi).
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

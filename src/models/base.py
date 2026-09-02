"""Model arayuzu ve ortak yardimcilar.

Tum modeller ayni sozlesmeyi uygular: fit(X, y) / predict(X).
Boylece walk-forward harness (src/models/train.py) model tipinden bagimsiz kalir.

NaN POLITIKASI (feature katmanindan gelen gercek durum):
  LightGBM ve XGBoost NaN'i dogal isler -> dokunulmaz.
  Ridge isleyemez -> medyan imputation, ama medyan YALNIZ TRAIN FOLD'unda
  hesaplanir. Validation/test fold'unun istatistigini kullanmak leakage olur.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

# Feature olmayan kolonlar. month SADECE split anahtaridir - feature degil:
# test'te 71+ araliginda olacagi icin modele verilirse genellemez.
NON_FEATURES = ("sample_id", "month", "target")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


class Model(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class MedianImputer:
    """Train fold'unda fit edilen medyan imputer + sonsuz deger temizligi."""

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        clean = X.replace([np.inf, -np.inf], np.nan)
        self.medians_ = clean.median()
        # Tamamen bos kolonlar icin 0
        self.medians_ = self.medians_.fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("once fit() cagrilmali")
        return X.replace([np.inf, -np.inf], np.nan).fillna(self.medians_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

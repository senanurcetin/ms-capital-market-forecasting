"""XGBoost - ikinci agac modeli.

LightGBM'den farkli bolme stratejisi (level-wise + histogram) kullandigi icin
ensemble'da tamamlayici hata yapisi saglamasi beklenir. Ayni cosine erken
durdurmasi; XGBoost minimize ettigi icin metrik NEGATIF cosine olarak verilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation.metrics import cosine_similarity
from src.models.base import feature_columns

DEFAULT_PARAMS: dict = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 8,
    "min_child_weight": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "max_bin": 128,
    "nthread": 0,
    "seed": 42,
}


def _neg_cosine(y_pred: np.ndarray, dmat: xgb.DMatrix) -> tuple[str, float]:
    return "neg_cosine", -cosine_similarity(dmat.get_label(), y_pred)


class XGBoostModel:
    name = "xgboost"

    def __init__(self, params: dict | None = None, num_boost_round: int = 3000,
                 early_stopping_rounds: int = 100) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster_: xgb.Booster | None = None
        self.features_: list[str] = []
        self.best_iteration_: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            eval_set: tuple[pd.DataFrame, np.ndarray] | None = None, **_) -> "XGBoostModel":
        self.features_ = feature_columns(X)
        dtrain = xgb.DMatrix(X[self.features_], label=y, nthread=-1)
        evals, es = [], None
        if eval_set is not None:
            Xv, yv = eval_set
            evals = [(xgb.DMatrix(Xv[self.features_], label=yv, nthread=-1), "valid")]
            es = self.early_stopping_rounds
        self.booster_ = xgb.train(
            self.params, dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals, custom_metric=_neg_cosine if evals else None,
            early_stopping_rounds=es, verbose_eval=200,
        )
        self.best_iteration_ = getattr(self.booster_, "best_iteration", None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("once fit() cagrilmali")
        dm = xgb.DMatrix(X[self.features_], nthread=-1)
        rng = (0, self.best_iteration_ + 1) if self.best_iteration_ is not None else None
        return self.booster_.predict(dm, iteration_range=rng)

    def importance(self, kind: str = "gain") -> pd.Series:
        raw = self.booster_.get_score(importance_type=kind)
        return pd.Series(raw).reindex(self.features_).fillna(0.0).sort_values(ascending=False)

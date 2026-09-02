"""LightGBM - the primary candidate model.

Why primary: there is no symbol column, so the problem reduces to tabular
regression over 1.26M rows with ~294 features on wildly different scales and with
heavy NaN density (e.g. txn_volume_imbalance_1s is NULL in 68% of samples). GBDT
handles all of that natively.

EARLY STOPPING IS DRIVEN BY COSINE, because that is the competition metric and,
being scale-invariant, it does not have to rank models the same way RMSE does.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import cosine_similarity
from src.models.base import feature_columns

DEFAULT_PARAMS: dict = {
    "objective": "regression",       # L2; target ~N(0, 0.0026), not heavy-tailed
    "metric": "None",                # replaced by the custom cosine metric
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_data_in_leaf": 500,         # 1.26M rows -> guard against over-splitting
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "num_threads": 0,
    "verbosity": -1,
    "seed": 42,
}


def _cosine_eval(y_pred: np.ndarray, dataset: lgb.Dataset):
    return "cosine", cosine_similarity(dataset.get_label(), y_pred), True


class LightGBMModel:
    name = "lightgbm"

    def __init__(self, params: dict | None = None, num_boost_round: int = 3000,
                 early_stopping_rounds: int = 100) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster_: lgb.Booster | None = None
        self.features_: list[str] = []
        self.best_iteration_: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            eval_set: tuple[pd.DataFrame, np.ndarray] | None = None, **_) -> "LightGBMModel":
        self.features_ = feature_columns(X)
        dtrain = lgb.Dataset(X[self.features_], label=y, free_raw_data=True)
        valid_sets, callbacks = [], [lgb.log_evaluation(period=200)]
        if eval_set is not None:
            Xv, yv = eval_set
            valid_sets = [lgb.Dataset(Xv[self.features_], label=yv, reference=dtrain)]
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False)
            )
        self.booster_ = lgb.train(
            self.params, dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets or None,
            feval=_cosine_eval if valid_sets else None,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster_.best_iteration or self.num_boost_round
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("fit() must be called first")
        return self.booster_.predict(X[self.features_], num_iteration=self.best_iteration_)

    def importance(self, kind: str = "gain") -> pd.Series:
        return pd.Series(
            self.booster_.feature_importance(importance_type=kind),
            index=self.features_,
        ).sort_values(ascending=False)

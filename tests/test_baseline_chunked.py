"""RidgeModel streams its data in chunks; the answer must not depend on the chunking.

The chunked fit replaced a straight sklearn pipeline (imputer -> StandardScaler -> Ridge)
that needed several float64 copies of the training frame. These tests keep that pipeline
as the reference and require the chunked model to reproduce it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.models.base import MedianImputer
from src.models.baseline import RidgeModel


def _frame(n: int = 1003, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)).astype("float32"),
                     columns=[f"f{i}" for i in range(6)])
    X["f_int"] = rng.integers(0, 5, n)
    X["f_const"] = np.float32(3.0)
    X["f_dup"] = X["f0"]
    X.loc[rng.choice(n, 40, replace=False), "f1"] = np.nan
    X.loc[rng.choice(n, 10, replace=False), "f2"] = np.inf
    X.loc[rng.choice(n, 10, replace=False), "f3"] = -np.inf
    y = (X["f0"].to_numpy() * 0.5 - np.nan_to_num(X["f1"].to_numpy()) * 0.2
         + rng.normal(scale=0.5, size=n) + 0.3)
    return X, y


def _reference(X: pd.DataFrame, y: np.ndarray, alpha: float):
    """The pre-chunking implementation, verbatim."""
    imputer, scaler, model = MedianImputer(), StandardScaler(), Ridge(alpha=alpha, random_state=0)
    model.fit(scaler.fit_transform(imputer.fit_transform(X)), y)
    return lambda Z: model.predict(scaler.transform(imputer.transform(Z)))


@pytest.mark.parametrize("chunk_rows", [1, 7, 250, 1003, 5000])
def test_chunked_fit_matches_the_straight_sklearn_pipeline(chunk_rows):
    X, y = _frame()
    expected = _reference(X, y, alpha=10.0)(X)
    got = RidgeModel(alpha=10.0, chunk_rows=chunk_rows).fit(X, y).predict(X)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-8)


def test_prediction_does_not_depend_on_chunk_size():
    X, y = _frame()
    model = RidgeModel(alpha=1.0, chunk_rows=100).fit(X, y)
    a = model.predict(X)
    model.chunk_rows = 13
    np.testing.assert_allclose(model.predict(X), a, rtol=0, atol=1e-12)


def test_predict_reindexes_to_the_fitted_column_order():
    X, y = _frame()
    model = RidgeModel(alpha=1.0, chunk_rows=64).fit(X, y)
    shuffled = X[list(reversed(X.columns))]
    np.testing.assert_allclose(model.predict(shuffled), model.predict(X), atol=1e-12)


def test_predict_on_an_empty_frame_is_empty():
    X, y = _frame()
    model = RidgeModel(alpha=1.0).fit(X, y)
    assert model.predict(X.iloc[:0]).shape == (0,)


def test_the_inner_estimator_still_predicts_for_the_serving_path():
    """src/inference/ensemble.py persists `model`, `scaler` and the medians separately."""
    X, y = _frame()
    m = RidgeModel(alpha=1.0, chunk_rows=128).fit(X, y)
    Z = m.scaler.transform(m.imputer.transform(X[m.features_]))
    np.testing.assert_allclose(m.model.predict(Z), m.predict(X), atol=1e-12)

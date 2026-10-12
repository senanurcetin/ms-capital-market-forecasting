"""Financial evaluation regressions: numerical boundaries and label alignment.

Only small synthetic arrays are used. These tests do not train models, load
market data, or require any live BigQuery/financial-data credentials.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    cosine_similarity,
    directional_accuracy,
    evaluate,
    lgb_cosine_eval,
    mae,
    pearson,
    rmse,
    xgb_cosine_eval,
)
from src.evaluation.temporal_validation import Fold, split_indices, stability


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        ([1.0, 0.0, -1.0], 1.0),
        ([-1.0, 0.0, 1.0], -1.0),
        ([0.0, 1.0, 0.0], 0.0),
        ([0.0, 0.0, 0.0], 0.0),
    ],
)
def test_cosine_geometry_boundaries(predictions, expected):
    labels = np.array([1.0, 0.0, -1.0])
    assert cosine_similarity(labels, predictions) == pytest.approx(expected, abs=1e-12)


def test_cosine_returns_zero_for_zero_target():
    assert cosine_similarity([0.0, 0.0, 0.0], [1.0, -2.0, 3.0]) == 0.0


def test_cosine_flattening_contract_for_column_vectors():
    labels = np.array([[1.0], [-2.0], [3.0]])
    predictions = [1.0, -2.0, 3.0]
    assert cosine_similarity(labels, predictions) == pytest.approx(1.0)


def test_cosine_negative_scale_reverses_score():
    labels = np.array([1.0, -3.0, 2.0, -2.0])
    predictions = np.array([2.0, -1.0, 3.0, 4.0])
    assert cosine_similarity(labels, predictions * -7) == pytest.approx(
        -cosine_similarity(labels, predictions)
    )


@pytest.mark.parametrize(
    ("labels", "predictions", "ignore_zeros", "expected"),
    [
        ([0, 0], [0, 1], True, 0.0),
        ([0, 0], [0, 1], False, 0.5),
        ([1, -1, 0], [1, 1, 0], True, 0.5),
        ([1, -1, 0], [1, 1, 0], False, 2 / 3),
        ([1, -1], [0, 0], True, 0.0),
    ],
)
def test_directional_accuracy_zero_contract(labels, predictions, ignore_zeros, expected):
    assert directional_accuracy(labels, predictions, ignore_zeros=ignore_zeros) == pytest.approx(
        expected
    )


def test_pearson_ignores_constant_prediction_and_target():
    assert pearson([1, 2, 3], [9, 9, 9]) == 0.0
    assert pearson([2, 2, 2], [1, 3, 5]) == 0.0


def test_pearson_and_cosine_have_distinct_offset_semantics():
    y = np.array([-2.0, -1.0, 1.0, 2.0])
    biased = y + 20.0
    assert pearson(y, biased) == pytest.approx(1.0)
    assert cosine_similarity(y, biased) < 0.2


def test_mae_and_rmse_do_not_confuse_signed_residuals():
    labels = [-2.0, 0.0, 2.0]
    predictions = [-1.0, -2.0, 1.0]
    assert mae(labels, predictions) == pytest.approx(4 / 3)
    assert rmse(labels, predictions) == pytest.approx(np.sqrt(2.0))


def test_all_evaluation_metrics_remain_consistent():
    labels = np.array([1.0, -2.0, 3.0, -4.0])
    predictions = np.array([2.0, -3.0, 4.0, -5.0])
    output = evaluate(labels, predictions)
    assert output["cosine"] == pytest.approx(cosine_similarity(labels, predictions))
    assert output["mae"] == pytest.approx(mae(labels, predictions))
    assert output["rmse"] == pytest.approx(rmse(labels, predictions))
    assert output["pearson"] == pytest.approx(pearson(labels, predictions))
    assert output["directional_accuracy"] == 1.0


class _LabelContainer:
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def get_label(self):
        return self.labels


def test_lightgbm_evaluator_maximises_positive_cosine():
    predictions = np.array([1.0, -2.0, 3.0])
    labels = _LabelContainer([1.0, -2.0, 3.0])
    name, value, higher_is_better = lgb_cosine_eval(predictions, labels)
    assert name == "cosine"
    assert value == pytest.approx(1.0)
    assert higher_is_better is True


def test_xgboost_evaluator_minimises_negative_cosine():
    predictions = np.array([1.0, -2.0, 3.0])
    labels = _LabelContainer([1.0, -2.0, 3.0])
    name, value = xgb_cosine_eval(predictions, labels)
    assert name == "neg_cosine"
    assert value == pytest.approx(-1.0)


def test_stability_uses_sample_standard_deviation_and_one_based_worst_fold():
    scores = [0.30, 0.10, 0.20]
    result = stability(scores)
    assert result["mean"] == pytest.approx(0.20)
    assert result["std"] == pytest.approx(0.10)
    assert result["min"] == pytest.approx(0.10)
    assert result["max"] == pytest.approx(0.30)
    assert result["worst_fold"] == 2


def test_stability_single_fold_has_zero_dispersion():
    assert stability([0.25]) == {
        "mean": 0.25,
        "std": 0.0,
        "min": 0.25,
        "max": 0.25,
        "worst_fold": 1,
    }


def test_temporal_split_is_disjoint_and_drops_embargo_even_if_months_unsorted():
    months = np.array([10, 0, 8, 3, 11, 7, 9, 1, 12, 2])
    fold = Fold(index=1, train_months=(0, 7), embargo_months=(8, 9), val_months=(10, 12))
    train, validation = split_indices(months, fold)
    assert set(months[train]) == {0, 1, 2, 3, 7}
    assert set(months[validation]) == {10, 11, 12}
    assert not set(train) & set(validation)
    assert not ({8, 9} & (set(months[train]) | set(months[validation])))

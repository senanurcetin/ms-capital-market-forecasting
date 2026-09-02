"""Metric unit tests - in particular cosine's scale and shift behaviour."""
import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from src.evaluation.metrics import (
    cosine_similarity, directional_accuracy, evaluate, mae, pearson, rmse,
)

rng = np.random.default_rng(0)


def test_cosine_matches_sklearn():
    y = rng.normal(0, 0.0026, 5000)
    p = y + rng.normal(0, 0.002, 5000)
    expected = sk_cosine(y.reshape(1, -1), p.reshape(1, -1))[0, 0]
    assert cosine_similarity(y, p) == pytest.approx(expected, abs=1e-12)


def test_cosine_is_scale_invariant():
    """Scaling the predictions must NOT change the score."""
    y = rng.normal(0, 0.0026, 1000)
    p = rng.normal(0, 0.0026, 1000)
    base = cosine_similarity(y, p)
    for k in (0.001, 1.0, 7.5, 1000.0):
        assert cosine_similarity(y, p * k) == pytest.approx(base, abs=1e-12)


def test_cosine_is_not_shift_invariant():
    """Adding a bias MUST hurt - this is where cosine differs from Pearson."""
    y = rng.normal(0, 0.0026, 1000)
    p = y.copy()
    perfect = cosine_similarity(y, p)
    shifted = cosine_similarity(y, p + 0.01)
    assert perfect == pytest.approx(1.0, abs=1e-12)
    assert shifted < perfect - 0.05
    # Pearson, by contrast, is unaffected by the shift.
    assert pearson(y, p + 0.01) == pytest.approx(pearson(y, p), abs=1e-10)


def test_cosine_perfect_and_opposite():
    y = rng.normal(0, 1, 100)
    assert cosine_similarity(y, y) == pytest.approx(1.0, abs=1e-12)
    assert cosine_similarity(y, -y) == pytest.approx(-1.0, abs=1e-12)


def test_cosine_zero_prediction_is_zero_not_nan():
    y = rng.normal(0, 1, 100)
    assert cosine_similarity(y, np.zeros(100)) == 0.0


def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(10), np.zeros(11))


def test_directional_accuracy_ignores_exact_zero_targets():
    """5.54% of targets are exactly zero; they must not skew sign accuracy."""
    y = np.array([0.0, 0.0, 1.0, -1.0])
    p = np.array([5.0, 5.0, 1.0, -1.0])
    assert directional_accuracy(y, p) == pytest.approx(1.0)
    assert directional_accuracy(y, p, ignore_zeros=False) == pytest.approx(0.5)


def test_mae_rmse_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert mae(y, p) == pytest.approx(2 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(4 / 3))


def test_evaluate_returns_all_keys():
    y = rng.normal(0, 1, 200)
    out = evaluate(y, y * 3)
    assert set(out) == {"cosine", "mae", "rmse", "pearson", "directional_accuracy"}
    assert out["cosine"] == pytest.approx(1.0, abs=1e-12)
